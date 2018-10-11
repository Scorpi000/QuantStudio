# -*- coding: utf-8 -*-
"""基于 pyomo 模块的投资组合构造器(TODO)"""
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from .BasePC import PortfolioConstructor


# 线性目标函数: f'*x
def LinearObjective(model):
    return pyo.summation(model.f, model.x)
# L1 惩罚线性目标: f'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'f':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Linear'}
def L1_LinearObjective(model):
    Expr = pyo.summation(model.f, model.x)
    if hasattr(model, "lambda1"):
        Expr += model.lambda1 * sum(abs(model.x[i]-model.c[i]) for i in model.N)
    if hasattr(model, "lambda2"):
        Expr += model.lambda2 * sum((abs(model.x[i]-model.c_pos[i])+(model.x[i]-model.c_pos[i]))/2 for i in model.N)
    if hasattr(model, "lambda3"):
        Expr += model.lambda3 * sum((abs(model.x[i]-model.c_pos[i])-(model.x[i]-model.c_neg[i]))/2 for i in model.N)
    return Expr
# 二次目标: x'Sigma*x + Mu'*x,{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'type':'Quadratic'},其中，Sigma = X*F*X'+Delta
def QuadraticObjective(model):
    Expr = pyo.summation(model.Mu, model.x)
    for i in model.N:
        Expr += model.Sigma[i, i] * model.x[i]**2
        for j in range(i+1, len(model.N)):
            Expr += 2 * model.Sigma[i, j] * model.x[i] * model.x[j]
    return Expr
# L1 惩罚二次目标: x'Sigma*x + Mu'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Quadratic'}, 其中, Sigma = X*F*X'+Delta
def L1_QuadraticObjective(model):
    Expr = pyo.summation(model.Mu, model.x)
    for i in model.N:
        for j in model.N:
            Expr += model.x[i]*model.Sigma[i, j]*model.x[j]
    if hasattr(model, "lambda1"):
        Expr += model.lambda1 * sum(abs(model.x[i]-model.c[i]) for i in model.N)
    if hasattr(model, "lambda2"):
        Expr += model.lambda2 * sum((abs(model.x[i]-model.c_pos[i])+(model.x[i]-model.c_pos[i]))/2 for i in model.N)
    if hasattr(model, "lambda3"):
        Expr += model.lambda3 * sum((abs(model.x[i]-model.c_pos[i])-(model.x[i]-model.c_neg[i]))/2 for i in model.N)
    return Expr


# 数学形式的约束条件
# Box 约束：lb <= x <= ub,{'lb':array(n,1),'ub':array(n,1),'type':'Box'}
# 线性不等式约束：A * x <= b,{'A':array(m,n),'b':array(m,1),'type':'LinearIn'}
# 线性等式约束：Aeq * x == beq,{'Aeq':array(m,n),'beq':array(m,1),'type':'LinearEq'}
# 二次约束：x'*Sigma*x + Mu'*x <= q,{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'q':double,'type':'Quadratic'},其中，Sigma = X*F*X'+Delta
# L1 范数约束：sum(abs(x-c)) <= l,{'c':array(n,1),'l':double,'type':'L1'}
# 正部总约束：sum((x-c_pos)^+) <= l_pos，{'c_pos':array(n,1),'l_pos':double,'type':'Pos'}
# 负部总约束：sum((x-c_neg)^-) <= l_neg，{'c_neg':array(n,1),'l_neg':double,'type':'Neg'}
# 非零数目约束：sum((x-b)!=0) <= N, {'b':array(n,1),'N':double,'type':'NonZeroNum'}
def BoxConstraint(model, i):
    return (model.lb[i]<=model.x[i]<=model.ub[i])
def LinearInConstraint(model, m1):
    return (sum(model.A[m1, i] * model.x[i] for i in model.N) <= model.b[m1])
def LinearEqConstraint(model, m2):
    return (sum(model.Aeq[m2, i] * model.x[i] for i in model.N) == model.beq[m2])
def QuadraticConstraint(model, suffix):
    Sigma = getattr(model, "Sigma"+suffix)
    Mu = getattr(model, "Mu"+suffix)
    q = getattr(model, "q"+suffix)
    Expr = pyo.summation(Mu, model.x)
    for i in model.N:
        for j in model.N:
            Expr += model.x[i]*Sigma[i, j]*model.x[j]
    return (Expr<=q)
def L1Constraint(model, suffix):
    c = getattr(model, "c"+suffix)
    l = getattr(model, "l"+suffix)
    return (sum(abs(model.x[i]-c[i]) for i in model.N)<=l)
def PosConstraint(model, suffix):
    c_pos = getattr(model, "c_pos"+suffix)
    l_pos = getattr(model, "l_pos"+suffix)
    return (sum((abs(model.x[i]-c_pos[i])+(model.x[i]-c_pos[i]))/2 for i in model.N)<=l_pos)
def NegConstraint(model, suffix):
    c_neg = getattr(model, "c_neg"+suffix)
    l_neg = getattr(model, "l_neg"+suffix)
    return (sum((abs(model.x[i]-c_neg[i])+(model.x[i]-c_neg[i]))/2 for i in model.N)<=l_neg)
def NonZeroNumConstraint(model, suffix):
    b = getattr(model, "b"+suffix)
    N = getattr(model, "N"+suffix)
    return (sum(((model.x[i]-b[i])!=0) for i in model.N)<=N)



class PyomoPC(PortfolioConstructor):
    """基于 pyomo 模块的投资组合构造器"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Model = None# 优化模型
        self._PyomoModelChanged = False
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def _genModelConstraints(self, model, prepared_constraints):
        for iType, iConstraint in prepared_constraints.items():
            if iType=="Box":
                model.lb = pyo.Param(model.N)
                model.ub = pyo.Param(model.N)
                model.BoxConstraint = pyo.Constraint(model.N, rule=BoxConstraint)
            elif iType=="LinearIn":
                model.m1 = pyo.Param(within=pyo.PositiveIntegers)
                model.M1 = pyo.RangeSet(0, model.m1-1)
                model.A = pyo.Param(model.M1, model.N)
                model.b = pyo.Param(model.M1)
                model.LinearInConstraint = pyo.Constraint(model.M1, rule=LinearInConstraint)
            elif iType=="LinearEq":
                model.m2 = pyo.Param(within=pyo.PositiveIntegers)
                model.M2 = pyo.RangeSet(0, model.m2-1)
                model.Aeq = pyo.Param(model.M2, model.N)
                model.beq = pyo.Param(model.M2)
                model.LinearEqConstraint = pyo.Constraint(model.M2, rule=LinearEqConstraint)
            elif iType=="Quadratic":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    setattr(model, "Sigma"+j, pyo.Param(model.N, model.N))
                    setattr(model, "Mu"+j, pyo.Param(model.N))
                    setattr(model, "q"+j, pyo.Param(within=pyo.Reals))
                    setattr(model, "QuadraticConstraint"+j, pyo.Constraint(rule=lambda m: QuadraticConstraint(m, j)))
            elif iType=="L1":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    setattr(model, "l"+j, pyo.Param(within=pyo.Reals))
                    setattr(model, "c"+j, pyo.Param(model.N))
                    setattr(model, "L1Constraint"+j, pyo.Constraint(rule=lambda m: L1Constraint(m, j)))
            elif iType=="Pos":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    setattr(model, "l_pos"+j, pyo.Param(within=pyo.Reals))
                    setattr(model, "c_pos"+j, pyo.Param(model.N))
                    setattr(model, "PosConstraint"+j, pyo.Constraint(rule=lambda m: PosConstraint(m, j)))
            elif iType=="Neg":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    setattr(model, "l_neg"+j, pyo.Param(within=pyo.Reals))
                    setattr(model, "c_neg"+j, pyo.Param(model.N))
                    setattr(model, "NegConstraint"+j, pyo.Constraint(rule=lambda m: NegConstraint(m, j)))
            elif iType=="NonZeroNum":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    setattr(model, "b"+j, pyo.Param(within=model.N))
                    setattr(model, "N"+j, pyo.Param(within=pyo.PositiveIntegers))
                    setattr(model, "NonZeroNumConstraint"+j, pyo.Constraint(rule=lambda m: NonZeroNumConstraint(m, j)))
        return model
    def _loadConstraintData(self, nvar, prepared_constraints):
        Data = {}
        IndexN = np.arange(nvar)
        for iType, iConstraint in prepared_constraints.items():
            if iType=="Box":
                Data["lb"] = dict(list(zip(IndexN, iConstraint["lb"].flatten())))
                Data["ub"] = dict(list(zip(IndexN, iConstraint["ub"].flatten())))
            elif iType=="LinearIn":
                m1 = iConstraint["A"].shape[0]
                Data["m1"] = {None: m1}
                IndexM1 = np.arange(m1)
                Data["b"] = dict(list(zip(IndexM1, iConstraint["b"].flatten())))
                if m1>1: IndexM1M1 = list(zip(IndexM1.repeat(nvar), (np.arange(m1*nvar) % m1)))
                else: IndexM1M1 = list(zip(IndexM1.repeat(nvar), IndexN))
                Data["A"] = dict(list(zip(IndexM1M1, iConstraint["A"].flatten())))
            elif iType=="LinearEq":
                m2 = iConstraint["Aeq"].shape[0]
                Data["m2"] = {None: m2}
                IndexM2 = np.arange(m2)
                Data["beq"] = dict(list(zip(IndexM2, iConstraint["beq"].flatten())))
                if m2>1: IndexM2M2 = list(zip(IndexM2.repeat(nvar), (np.arange(m2*nvar) % m2)))
                else: IndexM2M2 = list(zip(IndexM2.repeat(nvar), IndexN))
                Data["Aeq"] = dict(list(zip(IndexM2M2, iConstraint["Aeq"].flatten())))
            elif iType=="Quadratic":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    Data["q"+j] = {None: jSubConstraint["q"]}
                    Data["Mu"+j] = dict(list(zip(IndexN, jSubConstraint["Mu"].flatten())))
                    if "Sigma" in jSubConstraint:
                        IndexNN = list(zip(IndexN.repeat(nvar), (np.arange(nvar**2) % nvar)))
                        Data["Sigma"+j] = dict(list(zip(IndexNN, jSubConstraint["Sigma"].flatten())))
                    else:
                        Sigma = np.dot(np.dot(jSubConstraint["X"], jSubConstraint["F"]), jSubConstraint["X"].T) + np.diag(jSubConstraint["Delta"].flatten())
                        Sigma = (Sigma + Sigma.T) / 2
                        IndexNN = list(zip(IndexN.repeat(nvar), (np.arange(nvar**2) % nvar)))
                        Data["Sigma"+j] = dict(list(zip(IndexNN, Sigma.flatten())))
            elif iType=="L1":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    Data["l"+j] = {None: jSubConstraint["l"]}
                    Data["c"+j] = dict(list(zip(IndexN, jSubConstraint["c"].flatten())))
            elif iType=="Pos":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    Data["l_pos"+j] = {None: jSubConstraint["l_pos"]}
                    Data["c_pos"+j] = dict(list(zip(IndexN, jSubConstraint["c_pos"].flatten())))
            elif iType=="Neg":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    Data["l_neg"+j] = {None: jSubConstraint["l_neg"]}
                    Data["c_neg"+j] = dict(list(zip(IndexN, jSubConstraint["c_neg"].flatten())))
            elif iType=="NonZeroNum":
                for j, jSubConstraint in enumerate(iConstraint):
                    j = str(j)
                    Data["N"+j] = {None: jSubConstraint["N"]}
                    Data["b"+j] = dict(list(zip(IndexN, jSubConstraint["b"].flatten())))
        return Data
    def _genMeanVarianceModel(self, nvar, prepared_objective, prepared_constraints):
        Model = pyo.AbstractModel()
        Model.n = pyo.Param(within=pyo.PositiveIntegers)
        Model.N = pyo.RangeSet(0, Model.n-1)
        Model.x = pyo.Var(Model.N, domain=pyo.Reals)
        if "f" in prepared_objective: Model.f = pyo.Param(Model.N)
        if "lambda1" in prepared_objective:
            Model.lambda1 = pyo.Param(within=pyo.Reals)
            Model.c = pyo.Param(Model.N)
        if "lambda2" in prepared_objective:
            Model.lambda2 = pyo.Param(within=pyo.Reals)
            Model.c_pos = pyo.Param(Model.N)
        if "lambda3" in prepared_objective:
            Model.lambda3 = pyo.Param(within=pyo.Reals)
            Model.c_neg = pyo.Param(Model.N)
        if ("X" in prepared_objective) or ("Sigma" in prepared_objective): Model.Sigma = pyo.Param(Model.N, Model.N)
        if "Mu" in prepared_objective: Model.Mu = pyo.Param(Model.N)
        if prepared_objective["type"]=="Linear":
            Model.OBJ = pyo.Objective(rule=LinearObjective, sense=pyo.minimize)
        elif prepared_objective["type"]=="L1_Linear":
            Model.OBJ = pyo.Objective(rule=L1_LinearObjective, sense=pyo.minimize)
        elif prepared_objective["type"]=="Quadratic":
            Model.OBJ = pyo.Objective(rule=QuadraticObjective, sense=pyo.minimize)
        elif prepared_objective["type"]=="L1_Quadratic":
            Model.OBJ = pyo.Objective(rule=L1_QuadraticObjective, sense=pyo.minimize)
        return self._genModelConstraints(Model, prepared_constraints)
    def _loadMeanVarianceModelData(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        nvar = int(nvar)
        Data = {"n": {None:nvar}}
        IndexN = np.arange(nvar)
        if "f" in prepared_objective: Data["f"] = dict(list(zip(IndexN, prepared_objective["f"].flatten())))
        if "lambda1" in prepared_objective:
            Data["lambda1"] = {None: prepared_objective["lambda1"]}
            Data["c"] = dict(list(zip(IndexN, prepared_objective["c"].flatten())))
        if "lambda2" in prepared_objective:
            Data["lambda2"] = {None: prepared_objective["lambda2"]}
            Data["c_pos"] = dict(list(zip(IndexN, prepared_objective["c_pos"].flatten())))
        if "lambda3" in prepared_objective:
            Data["lambda3"] = {None: prepared_objective["lambda3"]}
            Data["c_neg"] = dict(list(zip(IndexN, prepared_objective["c_neg"].flatten())))
        if "Sigma" in prepared_objective:
            IndexNN = list(zip(IndexN.repeat(nvar), (np.arange(nvar**2) % nvar)))
            Data["Sigma"] = dict(list(zip(IndexNN, prepared_objective["Sigma"].flatten())))
        elif "X" in prepared_objective:
            Sigma = np.dot(np.dot(prepared_objective["X"], prepared_objective["F"]), prepared_objective["X"].T) + np.diag(prepared_objective["Delta"].flatten())
            IndexNN = list(zip(IndexN.repeat(nvar), (np.arange(nvar**2) % nvar)))
            Sigma = (Sigma + Sigma.T) / 2
            Data["Sigma"] = dict(list(zip(IndexNN, Sigma.flatten())))
        if "Mu" in prepared_objective: Data["Mu"] = dict(list(zip(IndexN, prepared_objective["Mu"].flatten())))
        Data.update(self._loadConstraintData(nvar, prepared_constraints))
        return {None: Data}
    def _init(self):
        self._PyomoModelChanged = self._ModelChanged
        return super()._init()
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        if (self._Model is None) or self._PyomoModelChanged:
            if self.OptimObjective.Type=="均值方差目标": self._Model = self._genMeanVarianceModel(nvar, prepared_objective, prepared_constraints)
            #elif self.OptimObjective.Type=="风险预算目标": self._Model = self._genMeanVarianceModel(nvar, prepared_objective, prepared_constraints)
            #elif self.OptimObjective.Type=="最大夏普率目标": self._Model = self._genMeanVarianceModel(nvar, prepared_objective, prepared_constraints)
            #elif self.OptimObjective.Type=="最大分散化目标": self._Model = self._genMeanVarianceModel(nvar, prepared_objective, prepared_constraints)
            else: raise __QS_Error__("不支持的优化目标: '%s'" % self.OptimObjective.Type)
            self._PyomoModelChanged = False
        Data = self._loadMeanVarianceModelData(nvar, prepared_objective, prepared_constraints, prepared_option)
        Instance = self._Model.create_instance(Data)
        pyo.SolverFactory(prepared_option.get("Solver", "cplex")).solve(Instance)
        x = np.array([Instance.x[i]() for i in range(nvar)])
        return (x, {})