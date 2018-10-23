# -*- coding: utf-8 -*-
"""基于 APMonitor 模块的投资组合构造器(TODO)"""
import pandas as pd
import numpy as np
from gekko import GEKKO

from .BasePC import PortfolioConstructor

class APMPC(PortfolioConstructor):
    """基于 APMonitor 模块的投资组合构造器"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Model = None# 优化模型
        self._x = None# 决策变量
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def _genModelConstraints(self, model, x, nvar, prepared_constraints, prepared_option):
        for iType, iConstraint in prepared_constraints.items():
            if iType=="Box":
                for j in range(nvar):
                    if not np.isinf(iConstraint["lb"][j, 0]):
                        x[j].lower = iConstraint["lb"][j, 0]
                    if not np.isinf(iConstraint["ub"][j, 0]):
                        x[j].upper = iConstraint["ub"][j, 0]
            elif iType=="LinearIn":
                for j in range(iConstraint["A"].shape[0]):
                    model.Equation(np.dot(iConstraint["A"][j, :], x)<=iConstraint["b"][j, 0])
            elif iType=="LinearEq":
                for j in range(iConstraint["Aeq"].shape[0]):
                    model.Equation(np.dot(iConstraint["Aeq"][j, :], x)==iConstraint["beq"][j, 0])
            elif iType=="Quadratic":
                for jSubConstraint in iConstraint:
                    if "X" in jSubConstraint:
                        jSigma = np.dot(np.dot(jSubConstraint["X"], jSubConstraint["F"]), jSubConstraint["X"].T) + np.diag(jSubConstraint["Delta"].flatten())
                        jSigma = (jSigma + jSigma.T) / 2
                    elif "Sigma" in jSubConstraint:
                        jSigma = jSubConstraint["Sigma"]
                    model.Equation(np.dot(np.dot(x, jSigma), x) + np.dot(jSubConstraint["Mu"].flatten(), x)<=jSubConstraint["q"])
            elif iType=="L1":
                for jSubConstraint in iConstraint:
                    model.Equation(sum(abs(x - jSubConstraint["c"].flatten()))<=jSubConstraint["l"])
            elif iType=="Pos":
                for jSubConstraint in iConstraint:
                    jc_pos = jSubConstraint["c_pos"].flatten()
                    model.Equation(sum(abs(x - jc_pos) + (x - jc_pos))<=2*jSubConstraint["l_pos"])
            elif iType=="Neg":
                for jSubConstraint in iConstraint:
                    jc_neg = jSubConstraint["c_neg"].flatten()
                    model.Equation(sum(abs(x - jc_neg) - (x - jc_neg))<=2*jSubConstraint["l_neg"])
            elif iType=="NonZeroNum":
                raise __QS_Error__("不支持的约束条件: '非零数目约束'!")
        return (model, x)
    def _genMeanVarianceModel(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        Model = GEKKO(remote=prepared_option.get("remote", False))
        x = Model.Array(Model.Var, (nvar, ))
        Obj = 0
        if "f" in prepared_objective: Obj += np.dot(prepared_objective["f"].flatten(), x)
        if "X" in prepared_objective:
            Sigma = np.dot(np.dot(prepared_objective["X"], prepared_objective["F"]), prepared_objective["X"].T) + np.diag(prepared_objective["Delta"].flatten())
            Sigma = (Sigma + Sigma.T) / 2
            Obj += np.dot(np.dot(x, Sigma), x)
        elif "Sigma" in prepared_objective:
            Obj += np.dot(np.dot(x, prepared_objective["Sigma"]), x)
        if "Mu" in prepared_objective: Obj += np.dot(prepared_objective["Mu"].flatten(), x)
        if "lambda1" in prepared_objective:
            Obj += prepared_objective["lambda1"] * sum(abs(x - prepared_objective["c"].flatten()))
        if "lambda2" in prepared_objective:
            c_pos = prepared_objective["c_pos"].flatten()
            Obj += prepared_objective["lambda2"] / 2 * sum(abs(x - c_pos) + (x - c_pos))
        if "lambda3" in prepared_objective:
            c_neg = prepared_objective["c_neg"].flatten()
            Obj += prepared_objective["lambda3"] / 2 * sum(abs(x - c_neg) + (x - c_neg))
        Model.Obj(Obj)
        self._Model, self._x = self._genModelConstraints(Model, x, nvar, prepared_constraints, prepared_option)
        return 0
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        if self.OptimObjective.Type=="均值方差目标": self._genMeanVarianceModel(nvar, prepared_objective, prepared_constraints, prepared_option)
        else: raise __QS_Error__("不支持的优化目标: '%s'" % self.OptimObjective.Type)
        self._Model.options.IMODE = 3 #steady state optimization
        self._Model.solve()
        return (np.array(self._x.tolist()).flatten(), {"Status":1})