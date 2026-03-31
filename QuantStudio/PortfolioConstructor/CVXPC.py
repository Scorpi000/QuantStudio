# -*- coding: utf-8 -*-
"""基于 CVXPY 的投资组合构造器"""
import traceback

import numpy as np
import cvxpy as cvx

from QuantStudio.Core import __QS_Error__
from .BasePC import PortfolioConstructor, MeanVarianceObjective, RiskBudgetObjective, MaxDiversificationObjective


class CVXPC(PortfolioConstructor):
    """基于 CVXPY 模块的投资组合构造器"""
    
    def _genModelConstraints(self, x, prepared_constraints, prepared_option):
        CVXConstraints = []
        for iType, iConstraint in prepared_constraints.items():
            if iType=="Box":
                CVXConstraints.extend([x<=iConstraint["ub"], x>=iConstraint["lb"]])
            elif iType=="LinearIn":
                CVXConstraints.append(iConstraint["A"] @ x <= iConstraint["b"])
            elif iType=="LinearEq":
                CVXConstraints.append(iConstraint["Aeq"] @ x == iConstraint["beq"])
            elif iType=="Quadratic":
                for jSubConstraint in iConstraint:
                    if "X" in jSubConstraint:
                        jSigma = np.dot(np.dot(jSubConstraint["X"], jSubConstraint["F"]), jSubConstraint["X"].T) + np.diag(jSubConstraint["Delta"])
                        jSigma = (jSigma + jSigma.T) / 2
                    elif "Sigma" in jSubConstraint:
                        jSigma = jSubConstraint["Sigma"]
                    CVXConstraints.append(cvx.quad_form(x, jSigma) + jSubConstraint["Mu"] @ x <= jSubConstraint["q"])
            elif iType=="L1":
                for jSubConstraint in iConstraint:
                    CVXConstraints.append(cvx.norm(x - jSubConstraint["c"], p=1) <= jSubConstraint["l"])
            elif iType=="Pos":
                for jSubConstraint in iConstraint:
                    CVXConstraints.append(cvx.sum(cvx.pos(x - jSubConstraint["c_pos"])) <= jSubConstraint["l_pos"])
            elif iType=="Neg":
                for jSubConstraint in iConstraint:
                    CVXConstraints.append(cvx.sum(cvx.neg(x - jSubConstraint["c_neg"])) <= jSubConstraint["l_neg"])
            elif iType=="NonZeroNum":
                for jSubConstraint in iConstraint:
                    jz = cvx.Variable(x.shape[0], boolean=True)
                    CVXConstraints.append(cvx.abs((x - jSubConstraint["b"])) <= jz)
                    CVXConstraints.append(cvx.sum(jz) <= jSubConstraint["N"])
        return CVXConstraints
    
    # 均值方差模型
    def _solveMeanVarianceModel(self, prepared_objective, prepared_constraints, prepared_option):
        x = cvx.Variable(np.sum(self._Mask))
        Obj = 0
        if "f" in prepared_objective: Obj += prepared_objective["f"] @ x
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
            Obj += prepared_objective["lambda2"] * cvx.sum(cvx.pos(x - prepared_objective["c_pos"].flatten()))
        if "lambda3" in prepared_objective:
            Obj += prepared_objective["lambda3"] * cvx.sum(cvx.neg(x - prepared_objective["c_neg"].flatten()))
        CVXConstraints = self._genModelConstraints(x, prepared_constraints, prepared_option)
        if prepared_objective["minmax"]=="min":
            Model = cvx.Problem(cvx.Minimize(Obj), CVXConstraints)
        else:
            Model = cvx.Problem(cvx.Maximize(Obj), CVXConstraints)
        try:
            Model.solve(**prepared_option)
        except:
            return (None, {"status": 0, "msg": traceback.format_exc()})
        else:
            return (x.value, {
                "status": (1 if Model.status not in (cvx.INFEASIBLE, cvx.UNBOUNDED) else 0), 
                "msg": Model.status, 
                "solver_name": Model.solver_stats.solver_name,
                "solve_time": Model.solver_stats.solve_time, 
                "setup_time": Model.solver_stats.setup_time, 
                "num_iters": Model.solver_stats.num_iters
            })
    
    # 风险预算模型
    def _solveRiskBudgetModel(self, prepared_objective, prepared_constraints, prepared_option):
        nVar = np.sum(self._Mask)
        x = cvx.Variable(nVar)
        Obj = 0
        if "X" in prepared_objective:
            Sigma = np.dot(np.dot(prepared_objective["X"], prepared_objective["F"]), prepared_objective["X"].T) + np.diag(prepared_objective["Delta"])
            Sigma = (Sigma + Sigma.T) / 2
            Obj += cvx.quad_form(x, Sigma)
        elif "Sigma" in prepared_objective:
            Obj += cvx.quad_form(x, prepared_objective["Sigma"])
        c = np.dot(prepared_objective["b"], np.log(prepared_objective["b"])) - min(1e-4, 1/nVar)
        CVXConstraints = [x >= np.zeros((nVar,)), prepared_objective["b"] @ cvx.log(x) >= c]
        Model = cvx.Problem(cvx.Minimize(Obj), CVXConstraints)
        try:
            Model.solve(**prepared_option)
        except:
            return (None, {"status": 0, "msg": traceback.format_exc()})
        else:
            return (x.value / np.sum(x.value), {
                "status": (1 if Model.status not in (cvx.INFEASIBLE, cvx.UNBOUNDED) else 0), 
                "msg": Model.status, 
                "solver_name": Model.solver_stats.solver_name,
                "solve_time": Model.solver_stats.solve_time, 
                "setup_time": Model.solver_stats.setup_time, 
                "num_iters": Model.solver_stats.num_iters
            })
    
    # 最大分散化模型
    def _solveMaxDiversificationModel(self, prepared_objective, prepared_constraints, prepared_option):
        nVar = np.sum(self._Mask)
        x = cvx.Variable(nVar)
        if "X" in prepared_objective:
            Sigma = np.dot(np.dot(prepared_objective["X"], prepared_objective["F"]), prepared_objective["X"].T) + np.diag(prepared_objective["Delta"])
            Sigma = (Sigma + Sigma.T) / 2
        elif "Sigma" in prepared_objective:
            Sigma = prepared_objective["Sigma"]
        D = np.diag(1 / np.diag(Sigma)**0.5)
        P = np.dot(np.dot(D, Sigma), D)
        Obj = cvx.quad_form(x, P)
        CVXConstraints = [x >= np.zeros((nVar,)), cvx.sum(x) == 1]
        Model = cvx.Problem(cvx.Minimize(Obj), CVXConstraints)
        try:
            Model.solve(**prepared_option)
        except:
            return (None, {"status": 0, "msg": traceback.format_exc()})
        else:
            x = np.dot(D, x.value)
            return (x / np.sum(x), {
                "status": (1 if Model.status not in (cvx.INFEASIBLE, cvx.UNBOUNDED) else 0), 
                "msg": Model.status, 
                "solver_name": Model.solver_stats.solver_name,
                "solve_time": Model.solver_stats.solve_time, 
                "setup_time": Model.solver_stats.setup_time, 
                "num_iters": Model.solver_stats.num_iters
            })
    
    def _genOption(self):
        return {"verbose": False} | self._QSArgs.OptimOption
    
    def _solve(self, prepared_objective, prepared_constraints, prepared_option):
        if isinstance(self._Objective, MeanVarianceObjective): return self._solveMeanVarianceModel(prepared_objective, prepared_constraints, prepared_option)
        elif isinstance(self._Objective, RiskBudgetObjective): return self._solveRiskBudgetModel(prepared_objective, prepared_constraints, prepared_option)
        elif isinstance(self._Objective, MaxDiversificationObjective): return self._solveMaxDiversificationModel(prepared_objective, prepared_constraints, prepared_option)
        else: raise __QS_Error__("不支持的优化目标: '%s'" % self._Objective)
