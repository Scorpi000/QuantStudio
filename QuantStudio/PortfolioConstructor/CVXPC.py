# -*- coding: utf-8 -*-
"""基于 CVXPY 的投资组合构造器"""
import os

import numpy as np
import pandas as pd
import cvxpy as cvx

from .BasePC import PortfolioConstructor
from QuantStudio import __QS_MainPath__, __QS_Error__

class CVXPC(PortfolioConstructor):
    """基于 CVXPY 模块的投资组合构造器"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Model = None# 优化模型
        self._x = None# 决策变量
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def _genModelConstraints(self, x, prepared_constraints, prepared_option):
        CVXConstraints = []
        for iType, iConstraint in prepared_constraints.items():
            if iType=="Box":
                CVXConstraints.extend([x<=iConstraint["ub"].flatten(), x>=iConstraint["lb"].flatten()])
            elif iType=="LinearIn":
                CVXConstraints.append(iConstraint["A"] @ x <= iConstraint["b"].flatten())
            elif iType=="LinearEq":
                CVXConstraints.append(iConstraint["Aeq"] @ x == iConstraint["beq"].flatten())
            elif iType=="Quadratic":
                for jSubConstraint in iConstraint:
                    if "X" in jSubConstraint:
                        jSigma = np.dot(np.dot(jSubConstraint["X"], jSubConstraint["F"]), jSubConstraint["X"].T) + np.diag(jSubConstraint["Delta"].flatten())
                        jSigma = (jSigma + jSigma.T) / 2
                    elif "Sigma" in jSubConstraint:
                        jSigma = jSubConstraint["Sigma"]
                    CVXConstraints.append(cvx.quad_form(x, jSigma) + jSubConstraint["Mu"].T @ x <= jSubConstraint["q"])
            elif iType=="L1":
                for jSubConstraint in iConstraint:
                    CVXConstraints.append(cvx.norm(x - jSubConstraint["c"].flatten(), p=1) <= jSubConstraint["l"])
            elif iType=="Pos":
                for jSubConstraint in iConstraint:
                    CVXConstraints.append(cvx.pos(x - jSubConstraint["c_pos"].flatten()) <= jSubConstraint["l_pos"])
            elif iType=="Neg":
                for jSubConstraint in iConstraint:
                    CVXConstraints.append(cvx.neg(x - jSubConstraint["c_neg"].flatten()) <= jSubConstraint["l_neg"])
            elif iType=="NonZeroNum": raise __QS_Error__("不支持的约束条件: '非零数目约束'!")
        return CVXConstraints
    # 均值方差模型
    def _solveMeanVarianceModel(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        x = cvx.Variable(nvar)
        Obj = 0
        if "f" in prepared_objective: Obj += prepared_objective["f"].T @ x
        if "X" in prepared_objective:
            Sigma = np.dot(np.dot(prepared_objective["X"], prepared_objective["F"]), prepared_objective["X"].T) + np.diag(prepared_objective["Delta"].flatten())
            Sigma = (Sigma + Sigma.T) / 2
            Obj += cvx.quad_form(x, Sigma)
        elif "Sigma" in prepared_objective:
            Obj += cvx.quad_form(x, prepared_objective["Sigma"])
        if "Mu" in prepared_objective: Obj += prepared_objective["Mu"].T @ x
        if "lambda1" in prepared_objective:
            Obj += prepared_objective["lambda1"] * cvx.norm(x - prepared_objective["c"].flatten(), p=1)
        if "lambda2" in prepared_objective:
            Obj += prepared_objective["lambda2"] * cvx.pos(x - prepared_objective["c_pos"].flatten())
        if "lambda3" in prepared_objective:
            Obj += prepared_objective["lambda3"] * cvx.neg(x - prepared_objective["c_neg"].flatten())
        CVXConstraints = self._genModelConstraints(x, prepared_constraints, prepared_option)
        self._Model = cvx.Problem(cvx.Minimize(Obj), CVXConstraints)
        self._x = x
        self._Model.solve(**prepared_option)
        Status = (1 if self._Model.status not in ("infeasible", "unbounded") else 0)
        return (self._x.value, {"Status":Status, "Msg":self._Model.status, "solver_name":self._Model.solver_stats.solver_name,
                                "solve_time":self._Model.solver_stats.solve_time, "setup_time":self._Model.solver_stats.setup_time, 
                                "num_iters":self._Model.solver_stats.num_iters})
    # 风险预算模型
    def _solveRiskBudgetModel(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        x = cvx.Variable(nvar)
        Obj = 0
        if "X" in prepared_objective:
            Sigma = np.dot(np.dot(prepared_objective["X"], prepared_objective["F"]), prepared_objective["X"].T) + np.diag(prepared_objective["Delta"].flatten())
            Sigma = (Sigma + Sigma.T) / 2
            Obj += cvx.quad_form(x, Sigma)
        elif "Sigma" in prepared_objective:
            Obj += cvx.quad_form(x, prepared_objective["Sigma"])
        c = np.dot(prepared_objective["b"].T, np.log(prepared_objective["b"])) - min(1e-4, 1/nvar)
        CVXConstraints = [x >= np.zeros(nvar), prepared_objective["b"].T @ cvx.log(x) >= c]
        self._Model = cvx.Problem(cvx.Minimize(Obj), CVXConstraints)
        self._x = x
        self._Model.solve(**prepared_option)
        Status = (1 if self._Model.status not in ("infeasible", "unbounded") else 0)
        return (x.value / np.sum(x.value), {"Status":Status, "Msg":self._Model.status, "solver_name":self._Model.solver_stats.solver_name,
                                            "solve_time":self._Model.solver_stats.solve_time, "setup_time":self._Model.solver_stats.setup_time, 
                                            "num_iters":self._Model.solver_stats.num_iters})
    # 最大分散化模型
    def _solveMaxDiversificationModel(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        x = cvx.Variable(nvar)
        if "X" in prepared_objective:
            Sigma = np.dot(np.dot(prepared_objective["X"], prepared_objective["F"]), prepared_objective["X"].T) + np.diag(prepared_objective["Delta"].flatten())
            Sigma = (Sigma + Sigma.T) / 2
        elif "Sigma" in prepared_objective:
            Sigma = prepared_objective["Sigma"]
        D = np.diag(1 / np.diag(Sigma)**0.5)
        P = np.dot(np.dot(D, Sigma), D)
        Obj = cvx.quad_form(x, P)
        CVXConstraints = [x >= np.zeros(nvar), cvx.sum(x) == 1]
        self._Model = cvx.Problem(cvx.Minimize(Obj), CVXConstraints)
        self._x = x
        self._Model.solve(**prepared_option)
        Status = (1 if self._Model.status not in ("infeasible", "unbounded") else 0)
        x = np.dot(D, x.value)
        return (x / np.sum(x), {"Status":Status, "Msg":self._Model.status, "solver_name":self._Model.solver_stats.solver_name,
                                "solve_time":self._Model.solver_stats.solve_time, "setup_time":self._Model.solver_stats.setup_time, 
                                "num_iters":self._Model.solver_stats.num_iters})
    def _genOption(self):
        Options = {"verbose":False}
        Options.update(self.OptimOption)
        return Options
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        if self.OptimObjective.Type=="均值方差目标": return self._solveMeanVarianceModel(nvar, prepared_objective, prepared_constraints, prepared_option)
        elif self.OptimObjective.Type=="风险预算目标": return self._solveRiskBudgetModel(nvar, prepared_objective, prepared_constraints, prepared_option)
        elif self.OptimObjective.Type=="最大分散化目标": return self._solveMaxDiversificationModel(nvar, prepared_objective, prepared_constraints, prepared_option)
        else: raise __QS_Error__("不支持的优化目标: '%s'" % self.OptimObjective.Type)
        