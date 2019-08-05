# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
from traits.api import Float, Bool, Int, Str, Instance, List, Enum, Dict, Either, on_trait_change

from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.Tools.AuxiliaryFun import getFactorList
from QuantStudio.Tools.IDFun import testIDFilterStr, filterID
from QuantStudio import __QS_Object__, __QS_Error__

# 数学形式的约束条件
# Box 约束：lb <= x <= ub,{'lb':array(n,1),'ub':array(n,1),'type':'Box'}
# 线性不等式约束：A * x <= b,{'A':array(m,n),'b':array(m,1),'type':'LinearIn'}
# 线性等式约束：Aeq * x == beq,{'Aeq':array(m,n),'beq':array(m,1),'type':'LinearEq'}
# 二次约束：x'*Sigma*x + Mu'*x <= q,{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'q':double,'type':'Quadratic'},其中，Sigma = X*F*X'+Delta
# L1 范数约束：sum(abs(x-c)) <= l,{'c':array(n,1),'l':double,'type':'L1'}
# 正部总约束：sum((x-c_pos)^+) <= l_pos，{'c_pos':array(n,1),'l_pos':double,'type':'Pos'}
# 负部总约束：sum((x-c_neg)^-) <= l_neg，{'c_neg':array(n,1),'l_neg':double,'type':'Neg'}
# 非零数目约束：sum((x-b)!=0) <= N, {'b':array(n,1),'N':double,'type':'NonZeroNum'}

# 优化目标基类
class OptimizationObjective(__QS_Object__):
    def __init__(self, pc, sys_args={}, config_file=None, **kwargs):
        self._PC = pc
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def Type(self):
        return "优化目标"
    @property
    def SupportedContraintType(self):
        return set()
    @property
    def Dependency(self):
        return {}
    def genObjective(self):
        return {}
# 均值方差优化目标
# 数学形式:
# 线性目标: f'*x,{'f':array(n,1),'type':'Linear'}
# 二次目标: x'Sigma*x + Mu'*x,{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'type':'Quadratic'},其中，Sigma = X*F*X'+Delta
# L1 惩罚线性目标: f'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'f':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Linear'}
# L1 惩罚二次目标: x'Sigma*x + Mu'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Quadratic'}, 其中, Sigma = X*F*X'+Delta
class MeanVarianceObjective(OptimizationObjective):
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=0)
    ExpectedReturnCoef = Float(0.0, arg_type="Double", label="收益项系数", order=1)
    RiskAversionCoef = Float(1.0, arg_type="Double", label="风险厌恶系数", order=2)
    TurnoverPenaltyCoef = Float(0.0, arg_type="Double", label="换手惩罚系数", order=3)
    BuyPenaltyCoef = Float(0.0, arg_type="Double", label="买入惩罚系数", order=4)
    SellPenaltyCoef = Float(0.0, arg_type="Double", label="卖出惩罚系数", order=5)
    @property
    def Type(self):
        return "均值方差目标"
    @property
    def SupportedContraintType(self):
        return {"预算约束", "因子暴露约束", "权重约束", "换手约束", "波动率约束", "预期收益约束", "非零数目约束"}
    @property
    def Dependency(self):
        Dependency = {}
        if self.RiskAversionCoef!=0.0: Dependency["协方差矩阵"] = True
        if (self.TurnoverPenaltyCoef!=0.0) or (self.BuyPenaltyCoef!=0.0) or (self.SellPenaltyCoef!=0.0): Dependency["初始投资组合"] = True
        if self.Benchmark: Dependency["基准投资组合"] = True
        if self.ExpectedReturnCoef!=0.0: Dependency["预期收益"] = True
        return Dependency
    def genObjective(self):
        Sign = -1.0
        self._ObjectiveConstant = 0.0
        if self.ExpectedReturnCoef!=0.0:
            Mu = self.ExpectedReturnCoef * self._PC.ExpectedReturn.values.reshape((self._PC._nID, 1))
            if self.Benchmark:
                self._ObjectiveConstant += -self.ExpectedReturnCoef * np.dot(self._PC.ExpectedReturn.values, self._PC.BenchmarkHolding.values)
                self._ObjectiveConstant += -self.ExpectedReturnCoef * np.dot(self._PC._BenchmarkExtraExpectedReturn.values, self._PC._BenchmarkExtra.values)
        else:
            Mu = np.zeros((self._PC._nID, 1))
        if self.RiskAversionCoef!=0.0:
            Sigma = self._PC.CovMatrix.values
            if self.Benchmark:
                Mu += self.RiskAversionCoef * np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov1.values).reshape((self._PC._nID, 1))
                Mu += self.RiskAversionCoef * np.dot(self._PC.BenchmarkHolding.values, Sigma).reshape((self._PC._nID, 1))
                self._ObjectiveConstant += -self.RiskAversionCoef/2 * np.dot(np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov.values), self._PC._BenchmarkExtra.values)
                self._ObjectiveConstant += -self.RiskAversionCoef * np.dot(np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov1.values), self._PC.BenchmarkHolding.values)
                self._ObjectiveConstant += -self.RiskAversionCoef/2 * np.dot(np.dot(self._PC.BenchmarkHolding.values, Sigma), self._PC.BenchmarkHolding.values)
            if self._PC.FactorCov is None:
                Objective = {"type":"Quadratic",
                             "Sigma":-Sign * self.RiskAversionCoef/2 * Sigma,
                             "Mu":Sign * Mu}
            else:
                Objective = {"type":"Quadratic",
                             "X":self._PC.RiskFactorData.values,
                             "F":-Sign * self.RiskAversionCoef/2 * self._PC.FactorCov.values,
                             "Delta":-Sign * self.RiskAversionCoef/2 * self._PC.SpecificRisk.values.reshape((self._PC._nID, 1))**2,
                             "Mu":Sign * Mu}
        else:
            Objective = {"type":"Linear", "f":Sign * Mu}
        if self.TurnoverPenaltyCoef!=0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda1':Sign * self.TurnoverPenaltyCoef,
                              "c":self._PC.Holding.values.reshape((self._PC._nID, 1))})
            self._ObjectiveConstant += -self.TurnoverPenaltyCoef * self._PC._HoldingExtra.abs().sum()
        if self.BuyPenaltyCoef!=0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda2':Sign * self.BuyPenaltyCoef,
                              "c_pos":self._PC.Holding.values.reshape((self._PC._nID, 1))})
            self._ObjectiveConstant += -self.BuyPenaltyCoef * self._PC._HoldingExtra[self._PC._HoldingExtra>0].sum()
        if self.SellPenaltyCoef!=0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda3':Sign * self.SellPenaltyCoef,
                              "c_neg":self._PC.Holding.values.reshape((self._PC._nID, 1))})
            self._ObjectiveConstant += -self.SellPenaltyCoef * (-self._PC._HoldingExtra[self._PC._HoldingExtra<0].sum())
        return Objective
# 最大夏普率优化目标
# 数学形式: (f'*x + f0) / sqrt(x'Sigma*x + Mu'x + q),{'f':array(n,1),'f0':double,'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Mu':array(n,1),'Delta':array(n,1),'q':double,'type':'Sharpe'},其中，Sigma = X*F*X'+Delta
class MaxSharpeObjective(OptimizationObjective):
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=0)
    @property
    def Type(self):
        return "最大夏普率目标"
    @property
    def SupportedContraintType(self):
        return {"预算约束", "因子暴露约束", "权重约束", "换手约束", "波动率约束", "预期收益约束", "非零数目约束"}
    @property
    def Dependency(self):
        Dependency = {"协方差矩阵":True, "预期收益":True}
        if self.Benchmark: Dependency["基准投资组合"] = True
        return Dependency
    def genObjective(self):
        if not self.Benchmark:
            if self._PC.FactorCov is None:
                return {"type":"Sharpe","f":self._PC.ExpectedReturn.values.reshape((self._PC._nID, 1)),"f0":0.0,
                        "Sigma":self._PC.CovMatrix.values,"Mu":np.zeros((self._PC._nID, 1)),"q":0.0}
            else:
                return {"type":"Sharpe","f":self._PC.ExpectedReturn.values.reshape((self._PC._nID, 1)),"f0":0.0,
                        "X":self._PC.RiskFactorData.values,"F":self._PC.FactorCov.values,"Delta":self._PC.SpecificRisk.values.reshape((self._PC._nID, 1))**2,
                        "Mu":np.zeros((self._PC._nID, 1)),"q":0.0}
        Objective = {"type":"Sharpe","f":self._PC.ExpectedReturn.values.reshape((self._PC._nID, 1)),
                     "f0":-np.dot(self._PC.ExpectedReturn.values, self._PC.BenchmarkHolding.values) - np.dot(self._PC._BenchmarkExtraExpectedReturn.values, self._PC._BenchmarkExtra.values)}
        Sigma = self._PC.CovMatrix.values
        Mu = -np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov1.values).reshape((self._PC._nID, 1))
        Mu -= np.dot(self._PC.BenchmarkHolding.values, Sigma).reshape((self._PC._nID, 1))
        q = np.dot(np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov.values), self._PC._BenchmarkExtra.values)
        q += 2*np.dot(np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov1.values), self._PC.BenchmarkHolding.values)
        q += np.dot(np.dot(self._PC.BenchmarkHolding.values, Sigma), self._PC.BenchmarkHolding.values)
        Objective["Mu"] = Mu
        Objective["q"] = q
        if self._PC.FactorCov is None:
            Objective["Sigma"] = Sigma
        else:
            Objective["X"] = self._PC.RiskFactorData.values
            Objective["F"] = self._PC.FactorCov.values
            Objective["Delta"] = self._PC.SpecificRisk.values.reshape((self._PC._nID, 1))**2
        return Objective

# 风险预算优化目标
# 数学形式: {'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'b':array(n,1),type':'Risk_Budget'}
class RiskBudgetObjective(OptimizationObjective):
    BudgetFactor = Enum("等权", arg_type="SingleOption", label="预算因子", order=0)
    @property
    def Type(self):
        return "风险预算目标"
    @property
    def SupportedContraintType(self):
        return {"预算约束", "权重约束"}
    @property
    def Dependency(self):
        Dependency = {"协方差矩阵":True}
        if self.BudgetFactor!="等权": Dependency["因子"] = [self.BudgetFactor]
        return Dependency
    def __QS_initArgs__(self):
        if self._PC.FactorData is not None:
            FactorNames = ["等权"]+self._PC.FactorData.columns.tolist()
        else: FactorNames = ["等权"]
        self.add_trait("BudgetFactor", Enum(*FactorNames, arg_type="SingleOption", label="预算因子", order=0))
        return super().__QS_initArgs__()
    def genObjective(self):
        if self._PC.FactorCov is None:
            Objective = {"type":"Risk_Budget", "Sigma":self._PC.CovMatrix.values}
        else:
            Objective = {"type":"Risk_Budget", "X":self._PC.RiskFactorData.values, "F":self._PC.FactorCov.values,
                         "Delta":self._PC.SpecificRisk.values.reshape((self._PC._nID, 1))**2}
        if self.BudgetFactor=="等权":
            Objective["b"] = np.zeros((self._PC._nID, 1)) + 1/self._PC._nID
        else:
            Objective["b"] = self._PC.FactorData.loc[:, [self.BudgetFactor]].values
        return Objective









# 最大分散化优化目标
# 数学形式: {'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),type':'Max_Diversification'}
class MaxDiversificationObjective(OptimizationObjective):
    @property
    def Type(self):
        return "最大分散化目标"
    @property
    def SupportedContraintType(self):
        return {"预算约束", "权重约束"}
    @property
    def Dependency(self):
        return {"协方差矩阵":True}
    def genObjective(self):
        if self._PC.FactorCov is None:
            Objective = {"type":"Max_Diversification", "Sigma":self._PC.CovMatrix.values}
        else:
            Objective = {"type":"Max_Diversification", "X":self._PC.RiskFactorData.values, "F":self._PC.FactorCov.values,
                         "Delta":self._PC.SpecificRisk.values.reshape((self._PC._nID, 1))**2}
        return Objective



# 约束条件基类
class Constraint(__QS_Object__):
    def __init__(self, pc, sys_args={}, config_file=None, **kwargs):
        self._PC = pc
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def Type(self):
        return "约束条件"
    @property
    def Dependency(self):
        return {}
    def genConstraint(self):
        return []

# 预算约束: i'*(w-benchmark) <=(==,>=) a, 转换成线性等式约束或线性不等式约束
class BudgetConstraint(Constraint):
    UpLimit = Float(1.0, arg_type="Double", label="限制上限", order=0)
    DownLimit = Float(1.0, arg_type="Double", label="限制下限", order=1)
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=2)
    DropPriority = Float(-1.0, arg_type="Double", label="舍弃优先级", order=3)
    @property
    def Type(self):
        return "预算约束"
    @property
    def Dependency(self):
        Dependency = {}
        if self.Benchmark: Dependency["基准投资组合"] = True
        return Dependency
    def genConstraint(self):
        if self.Benchmark:
            aAdj = self._PC.BenchmarkHolding.sum()+self._PC._BenchmarkExtra.sum()
        else:
            aAdj = 0.0
        Constraints = []
        if self.UpLimit<self.DownLimit: raise __QS_Error__("限制上限必须大于等于限制下限!")
        elif self.UpLimit==self.DownLimit:
            Constraints.append({"type":"LinearEq",
                                "Aeq":np.ones((1, self._PC._nID)),
                                "beq":np.array([[self.UpLimit+aAdj]])})
        else:
            if self.DownLimit>-np.inf:
                Constraints.append({"type":"LinearIn",
                                    "A":-np.ones((1, self._PC._nID)),
                                    "b":-np.array([[self.DownLimit+aAdj]])})
            if self.UpLimit<np.inf:
                Constraints.append({"type":"LinearIn",
                                    "A":np.ones((1, self._PC._nID)),
                                    "b":np.array([[self.UpLimit+aAdj]])})
        return Constraints

# 因子暴露约束: f'*(w-benchmark) <=(==,>=) a, 转换成线性等式约束或线性不等式约束
class FactorExposeConstraint(Constraint):
    FactorType = Enum("数值型", "类别型", arg_type="SingleOption", label="因子类型", order=0)
    FactorNames = List(arg_type="MultiOption", label="因子名称", order=1, option_range=())
    UpLimit = Float(0.0, arg_type="Double", label="限制上限", order=2)
    DownLimit = Float(0.0, arg_type="Double", label="限制下限", order=3)
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=4)
    DropPriority = Float(-1.0, arg_type="Double", label="舍弃优先级", order=5)
    @property
    def Type(self):
        return "因子暴露约束"
    @property
    def Dependency(self):
        Dependency = {"因子":self.FactorNames}
        if self.Benchmark: Dependency["基准投资组合"] = True
        return Dependency
    def __QS_initArgs__(self):
        if self._PC.FactorData is not None:
            FactorNames = self._PC.FactorData.columns.tolist()
        else: FactorNames = []
        self.add_trait("FactorNames", List(arg_type="MultiOption", label="因子名称", order=1, option_range=tuple(FactorNames)))
        self.FactorNames = FactorNames[:1]
        return super().__QS_initArgs__()
    # 生成数值型因子暴露约束条件的优化器条件形式
    def _genNumFactorExposeConstraint(self):
        if self.UpLimit<self.DownLimit: raise __QS_Error__("限制上限必须大于等于限制下限!")
        Constraints = []
        FactorNames = list(self.FactorNames)
        nFactor = len(FactorNames)
        if self.Benchmark:
            aAdj = (np.dot(self._PC.BenchmarkHolding.values, self._PC.FactorData.loc[:, FactorNames].values) + np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraFactorData.loc[:, FactorNames].values)).reshape((nFactor, 1))
        else:
            aAdj = np.zeros((nFactor, 1))
        if self.UpLimit==self.DownLimit:
            Aeq = self._PC.FactorData.loc[:, FactorNames].values.T
            AeqSum = np.abs(Aeq).sum(axis=1)
            Mask = (AeqSum!=0.0)
            Constraints.append({"type":"LinearEq",
                                "Aeq":Aeq[Mask,:],
                                "beq":(self.UpLimit+aAdj)[Mask,:]})
        else:
            A = self._PC.FactorData.loc[:, FactorNames].values.T
            ASum = np.abs(A).sum(axis=1)
            Mask = (ASum!=0.0)
            if self.DownLimit>-np.inf:
                Constraints.append({"type":"LinearIn",
                                    "A":-A[Mask,:],
                                    "b":-(self.DownLimit+aAdj)[Mask,:]})
            if self.UpLimit<np.inf:
                Constraints.append({"type":"LinearIn",
                                    "A":A[Mask,:],
                                    "b":(self.UpLimit+aAdj)[Mask,:]})
        return Constraints
    # 生成类别型因子暴露约束条件的优化器条件形式
    def _genClassFactorExposeConstraint(self):
        if self.UpLimit<self.DownLimit: raise __QS_Error__("限制上限必须大于等于限制下限!")
        Constraints = []
        if self._PC._Dependency.get("基准投资组合", False):
            AllFactorData = self._PC.FactorData.append(self._PC._BenchmarkExtraFactorData)
        else:
            AllFactorData = self._PC.FactorData
        for iFactor in self.FactorNames:
            iFactorData = AllFactorData[iFactor]
            iFactorData = DummyVarTo01Var(iFactorData, ignore_na=True, ignore_nonstring=True)
            nFactor = iFactorData.shape[1]
            if self.Benchmark:
                aAdj = (np.dot(self._PC.BenchmarkHolding.values, iFactorData.loc[self._PC._TargetIDs].values) + np.dot(self._PC._BenchmarkExtra.values, iFactorData.loc[self._PC._BenchmarkExtraIDs].values)).reshape((nFactor, 1))
            else:
                aAdj = np.zeros((nFactor, 1))
            if self.UpLimit==self.DownLimit:
                Aeq = iFactorData.loc[self._PC._TargetIDs].values.T
                AeqSum = np.abs(Aeq).sum(axis=1)
                Mask = (AeqSum!=0.0)
                Constraints.append({"type":"LinearEq",
                                    "Aeq":Aeq[Mask,:],
                                    "beq":(self.UpLimit+aAdj)[Mask, :]})
            else:
                A = iFactorData.loc[self._PC._TargetIDs].values.T
                ASum = np.abs(A).sum(axis=1)
                Mask = (ASum!=0.0)
                if self.DownLimit>-np.inf:
                    Constraints.append({"type":"LinearIn",
                                        "A":-A[Mask,:],
                                        "b":-(self.DownLimit+aAdj)[Mask, :]})
                if self.UpLimit<np.inf:
                    Constraints.append({"type":"LinearIn",
                                        "A":A[Mask,:],
                                        "b":(self.UpLimit+aAdj)[Mask, :]})
        return Constraints
    # 生成因子暴露约束条件的优化器条件形式
    def genConstraint(self):
        if self.FactorType=="数值型":
            return self._genNumFactorExposeConstraint()
        else:
            return self._genClassFactorExposeConstraint()

# 权重约束: (w-benchmark) <=(>=) a, 转换成 Box 约束
class WeightConstraint(Constraint):
    TargetIDs = Str(arg_type="IDFilterStr", label="目标ID", order=0)
    UpLimit = Either(Float(1.0), Str(), arg_type="Double", label="限制上限", order=1)
    DownLimit = Either(Float(0.0), Str(), arg_type="Double", label="限制下限", order=2)
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=3)
    DropPriority = Float(-1.0, arg_type="Double", label="舍弃优先级", order=4)
    @property
    def Type(self):
        return "权重约束"
    @property
    def Dependency(self):
        Dependency = {}
        if self.Benchmark: Dependency["基准投资组合"] = True
        if self.TargetIDs:
            CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(self.TargetIDs)
            if CompiledIDFilterStr is None: raise __QS_Error__("ID 过滤字符串有误!")
            Dependency["因子"] = IDFilterFactors
        if isinstance(self.UpLimit, str):
            Dependency["因子"] = Dependency.get("因子", [])+[self.UpLimit]
        if isinstance(self.DownLimit, str):
            Dependency["因子"] = Dependency.get("因子", [])+[self.DownLimit]
        return Dependency
    def genConstraint(self):
        if not self.TargetIDs:
            if isinstance(self.UpLimit, str): UpConstraint = self._PC.FactorData.loc[:, self.UpLimit]
            else: UpConstraint = pd.Series(self.UpLimit, index=self._PC._TargetIDs)
            if isinstance(self.DownLimit, str): DownConstraint = self._PC.FactorData.loc[:, self.DownLimit]
            else: DownConstraint = pd.Series(self.DownLimit, index=self._PC._TargetIDs)
        else:
            UpConstraint = pd.Series(np.inf, index=self._PC._TargetIDs)
            DownConstraint = pd.Series(-np.inf, index=self._PC._TargetIDs)
            TargetIDs = filterID(self._PC.FactorData, self.TargetIDs)
            TargetIDs = list(set(TargetIDs).intersection(self._PC._TargetIDs))
            if isinstance(self.UpLimit, str): UpConstraint[TargetIDs] = self._PC.FactorData.loc[TargetIDs, self.UpLimit]
            else: UpConstraint[TargetIDs] = self.UpLimit
            if isinstance(self.DownLimit, str): DownConstraint[TargetIDs] = self._PC.FactorData.loc[TargetIDs, self.DownLimit]
            else: DownConstraint[TargetIDs] = self.DownLimit
        if self.Benchmark:
            UpConstraint += self._PC.BenchmarkHolding
            DownConstraint += self._PC.BenchmarkHolding
        return [{"type":"Box", "lb":DownConstraint.values.reshape((self._PC._nID, 1)), "ub":UpConstraint.values.reshape((self._PC._nID, 1))}]
# 换手约束: sum(abs(w-w0)) <=(==) a, 转换成 L1 范数约束, 正部总约束, 负部总约束
class TurnoverConstraint(Constraint):
    ConstraintType = Enum("总换手限制", "总买入限制", "总卖出限制", "买卖限制", "买入限制", "卖出限制", arg_type="SingleOption", label="限制类型", order=0)
    AmtMultiple = Float(1.0, arg_type="Double", label="成交额倍数", order=1)
    UpLimit = Float(0.7, arg_type="Double", label="限制上限", order=2)
    DropPriority = Float(0, arg_type="Double", label="舍弃优先级", order=3)
    @property
    def Type(self):
        return "换手约束"
    @property
    def Dependency(self):
        Dependency = {"初始投资组合":True}
        if (self.ConstraintType in ["买卖限制", "买入限制", "卖出限制"]) and (self.AmtMultiple!=0.0):
            Dependency["成交金额"] = True
            Dependency["总财富"] = True
        return Dependency
    def genConstraint(self):
        HoldingWeight = self._PC.Holding.values.reshape((self._PC._nID, 1))
        if self.ConstraintType=="总换手限制":
            aAdj = self.UpLimit - self._PC._HoldingExtra.abs().sum()
            return [{"type":"L1", "c":HoldingWeight, "l":aAdj}]
        elif self.ConstraintType=="总买入限制":
            aAdj = self.UpLimit + self._PC._HoldingExtra[self._PC._HoldingExtra<0].sum()
            return [{"type":"Pos", "c_pos":HoldingWeight, "l":aAdj}]
        elif self.ConstraintType=="总卖出限制":
            aAdj = self.UpLimit - self._PC._HoldingExtra[self._PC._HoldingExtra>0].sum()
            return [{"type":"Neg", "c_neg":HoldingWeight, "l":aAdj}]
        if self.AmtMultiple==0.0:
            aAdj = np.zeros((self._PC._nID, 1)) + self.UpLimit
        else:
            aAdj = self._PC.AmountFactor.values.reshape((self._PC._nID, 1)) * self.AmtMultiple / self._PC.Wealth
        if self.ConstraintType=="买卖限制":
            return [{"type":"Box", "ub":aAdj+HoldingWeight, "lb":-aAdj+HoldingWeight}]
        elif self.ConstraintType=="买入限制":
            return [{"type":"Box", "ub":aAdj+HoldingWeight, "lb":np.zeros((self._PC._nID, 1))-np.inf}]
        elif self.ConstraintType=="买入限制":
            return [{"type":"Box", "ub":np.zeros((self._PC._nID, 1))+np.inf, "lb":-aAdj+HoldingWeight}]
        return []
# 波动率约束: (w-benchmark)'*Cov*(w-benchmark) <= a, 转换成二次约束
class VolatilityConstraint(Constraint):
    UpLimit = Float(0.06, arg_type="Double", label="限制上限", order=0)
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=1)
    DropPriority = Float(-1.0, arg_type="Double", label="舍弃优先级", order=2)
    @property
    def Type(self):
        return "波动率约束"
    @property
    def Dependency(self):
        Dependency = {"协方差矩阵":True}
        if self.Benchmark: Dependency["基准投资组合"] = True
        return Dependency
    def genConstraint(self):
        if self.Benchmark:
            Sigma = self._PC.CovMatrix.values
            BenchmarkWeight = self._PC.BenchmarkHolding.values
            Mu = -2*np.dot(BenchmarkWeight, Sigma) - 2*np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov1.values)
            Mu = Mu.reshape((self._PC._nID, 1))
            q = self.UpLimit**2 - np.dot(np.dot(BenchmarkWeight, Sigma), BenchmarkWeight)
            q -= 2*np.dot(np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov1.values), BenchmarkWeight)
            q -= np.dot(np.dot(self._PC._BenchmarkExtra.values, self._PC._BenchmarkExtraCov.values), self._PC._BenchmarkExtra.values)
        else:
            Mu = np.zeros((self._PC._nID, 1))
            q = self.UpLimit**2
        Constraint = {"type":"Quadratic", "Mu":Mu, "q":q}
        if self._PC.FactorCov is not None:
            Constraint["X"] = self._PC.RiskFactorData.values
            Constraint["F"] = self._PC.FactorCov.values
            Constraint["Delta"] = self._PC.SpecificRisk.values.reshape((self._PC._nID, 1))**2
        else:
            Constraint['Sigma'] = self._PC.CovMatrix.values
        return [Constraint]

# 预期收益约束: r'*(w-benchmark) >= a, 转换成线性不等式约束
class ExpectedReturnConstraint(Constraint):
    DownLimit = Float(0.0, arg_type="Double", label="限制下限", order=0)
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=1)
    DropPriority = Float(-1.0, arg_type="Double", label="舍弃优先级", order=2)
    @property
    def Type(self):
        return "预期收益约束"
    @property
    def Dependency(self):
        Dependency = {"预期收益":True}
        if self.Benchmark: Dependency["基准投资组合"] = True
        return Dependency
    def genConstraint(self):
        r = self._PC.ExpectedReturn.values.reshape((1, self._PC._nID))
        if self.Benchmark:
            aAdj = -self.DownLimit - np.dot(self._PC._BenchmarkExtraExpectedReturn.values, self._PC._BenchmarkExtra.values) - np.dot(r[0], self._PC.BenchmarkHolding.values)
            return [{"type":"LinearIn", "A":-r, "b":np.array([[aAdj]])}]
        else:
            return [{"type":"LinearIn", "A":-r, "b":np.array([[-self.DownLimit]])}]
# 非零数目约束: sum((w-benchmark!=0)<=N, 转换成非零数目约束
class NonZeroNumConstraint(Constraint):
    UpLimit = Float(150, arg_type="Double", label="限制上限", order=0)
    Benchmark = Bool(False, arg_type="Bool", label="相对基准", order=1)
    DropPriority = Float(-1.0, arg_type="Double", label="舍弃优先级", order=2)
    @property
    def Type(self):
        return "非零数目约束"
    @property
    def Dependency(self):
        Dependency = {}
        if self.Benchmark: Dependency["基准投资组合"] = True
        return Dependency
    def genConstraint(self):
        if self.Benchmark:
            N = (int(self.UpLimit - (self._PC._BenchmarkExtra.values!=0).sum()) if not np.isinf(self.UpLimit) else np.inf)
            return [{"type":"NonZeroNum", "N":N, "b":self._PC.BenchmarkHolding.values.reshape((self._PC._nID, 1))}]
        else:
            return [{"type":"NonZeroNum", "N":(int(self.UpLimit) if not np.isinf(self.UpLimit) else np.inf), "b":np.zeros((self._PC._nID, 1))}]



# 投资组合构造器基类
class PortfolioConstructor(__QS_Object__):
    """投资组合构造器"""
    ExpectedReturn = Instance(pd.Series, arg_type="Series", label="预期收益", order=0)
    CovMatrix = Instance(pd.DataFrame, arg_type="DataFrame", label="协方差矩阵", order=1)
    FactorCov = Instance(pd.DataFrame, arg_type="DataFrame", label="因子协方差阵", order=2)
    RiskFactorData = Instance(pd.DataFrame, arg_type="DataFrame", label="风险因子", order=3)
    SpecificRisk = Instance(pd.Series, arg_type="Series", label="特异性风险", order=4)
    Holding = Instance(pd.Series, arg_type="Series", label="初始投资组合", order=5)
    BenchmarkHolding = Instance(pd.Series, arg_type="Series", label="基准投资组合", order=6)
    AmountFactor = Instance(pd.Series, arg_type="Series", label="成交金额", order=7)
    FactorData = Instance(pd.DataFrame, arg_type="DataFrame", label="因子暴露", order=8)
    Wealth = Float(arg_type="Double", label="总财富", order=9)
    TargetIDs = List(arg_type="IDList", label="目标ID", order=10)
    OptimObjective = Instance(OptimizationObjective, arg_type="object", label="优化目标", order=11)
    Constraints = List(Constraint, arg_type="List", label="约束条件", order=12)
    OptimOption = Dict(arg_type="Dict", label="优化选项", order=13)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        # 优化前必须指定的变量
        #self.ExpectedReturn = None# 当前的预期收益率，Series(index=self.TargetIDs)
        #self.CovMatrix = None# 当前股票收益率的协方差矩阵，DataFrame(index=self.TargetIDs,columns=self.TargetIDs)
        #self.FactorCov = None# 当前因子收益率的协方差矩阵，DataFrame(index=[因子],columns=[因子])
        #self.RiskFactorData = None# 当前的风险因子值，DataFrame(index=self.TargetIDs,columns=[因子])
        #self.SpecificRisk = None# 当前个股的特异风险，Series(index=self.TargetIDs)
        #self.Holding = None# 当前持有的投资组合，Series(index=self.TargetIDs)
        #self.BenchmarkHolding = None# 当前的基准投资组合，Series(index=self.TargetIDs)
        #self.AmountFactor = None# 成交额因子，Series(index=self.TargetIDs)
        #self.Wealth = None# 当前的财富值，double
        #self.FactorData = None# 当前使用的因子值，DataFrame(index=self.TargetIDs,columns=[因子])
        #self.TargetIDs = None# 最终进入优化的股票池,[ID]
        self._nID = None# 最终进入优化的股票数量
        
        # 共享信息
        # 优化前根据参数信息即可生成
        self._isStarted = False# 是否已经进入了运算过程
        self._DataChanged = False# 数据是否被改动
        self._ModelChanged = False# 模型是否被改动, 即优化目标或者约束条件发生了变化
        self._Dependency = {}# 依赖项
        # 优化时根据具体数据生成
        self._BenchmarkExtraIDs = []# 基准相对于TargetIDs多出来的证券ID
        self._BenchmarkExtra = pd.Series()# 基准相对于TargetIDs多出来的证券ID权重，pd.Series(index=self._BenchmarkExtraIDs)
        self._BenchmarkExtraCov = pd.DataFrame()# 基准相对于TargetIDs多出来的证券ID对应的协方差阵，pd.DataFrame(index=self._BenchmarkExtraIDs,columns=self._BenchmarkExtraIDs)
        self._BenchmarkExtraCov1 = pd.DataFrame()# 基准相对于TargetIDs多出来的证券ID关于TargetIDs的协方差阵，pd.DataFrame(index=self._BenchmarkExtraIDs,columns=self.TargetIDs)
        self._BenchmarkExtraExpectedReturn = pd.Series()# 基准相对于TargetIDs多出来的证券ID对应的预期收益，pd.Series(index=self._BenchmarkExtraIDs)
        self._BenchmarkExtraFactorData = pd.DataFrame()# 基准相对于TargetIDs多出来的证券ID对应的因子数据，pd.DataFrame(index=self._BenchmarkExtraIDs,columns=[因子])
        self._HoldingExtraIDs = []# 当前持仓相对于TargetIDs多出来的证券ID
        self._HoldingExtra = pd.Series()# 当前持仓相对于TargetIDs多出来的证券ID权重，pd.Series(index=self._HoldingExtraIDs)
        self._HoldingExtraAmount = pd.Series()# 当前持仓相对于TargetIDs多出来的证券ID对应的因子数据，pd.Series(index=self._HoldingExtraIDs)
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    @on_trait_change("OptimObjective")
    def _on_OptimObjective_changed(self, obj, name, old, new):
        self._ModelChanged = True
    @on_trait_change("Constraints[]")
    def _on_Constraints_changed(self, obj, name, old, new):
        self._ModelChanged = True
    @on_trait_change("TargetIDs")
    def _on_TargetIDs_changed(self, obj, name, old, new):
        if not self._isStarted: self._DataChanged = True
    @on_trait_change("ExpectedReturn")
    def _on_ExpectedReturn_changed(self, obj, name, old, new):
        if not self._isStarted: self._DataChanged = True
    @on_trait_change("CovMatrix")
    def _on_CovMatrix_changed(self, obj, name, old, new):
        if not self._isStarted: self._DataChanged = True
    @on_trait_change("RiskFactorData")
    def _on_RiskFactorData_changed(self, obj, name, old, new):
        if not self._isStarted: self._DataChanged = True
    @on_trait_change("SpecificRisk")
    def _on_SpecificRisk_changed(self, obj, name, old, new):
        if not self._isStarted: self._DataChanged = True
    @on_trait_change("AmountFactor")
    def _on_AmountFactor_changed(self, obj, name, old, new):
        if not self._isStarted: self._DataChanged = True
    @on_trait_change("FactorData")
    def _on_FactorData_changed(self, obj, name, old, new):
        if not self._isStarted: self._DataChanged = True
    # 求解优化问题, 返回: (Series(权重, index=[ID]) 或 None, 其他信息: {})
    def solve(self):
        self._isStarted = True
        self._Dependency = self.init()
        if self._DataChanged: self._preprocessData()
        Objective = self.OptimObjective.genObjective()
        MathConstraints = []
        DropedConstraintInds = {-1:[]}
        DropedConstraints = {-1:[]}
        iStartInd = -1
        for i, iConstraint in enumerate(self.Constraints):
            iMathConstraints = iConstraint.genConstraint()
            MathConstraints.extend(iMathConstraints)
            iEndInd = iStartInd + len(iMathConstraints)
            iPriority = iConstraint.DropPriority
            if (iEndInd-iStartInd!=0) and (iPriority>-1):
                DropedConstraintInds[iPriority] = DropedConstraintInds.get(iPriority,[])+[i for i in range(iStartInd+1,iEndInd+1)]
                DropedConstraints[iPriority] = DropedConstraints.get(iPriority,[])+[str(i)+'-'+iConstraint.Type]
            iStartInd = iEndInd
        ResultInfo = {}
        ReleasedConstraint = []
        Priority = sorted(DropedConstraintInds)
        while (ResultInfo.get("Status",0)!=1) and (Priority!=[]):
            iPriority = Priority.pop(0)
            for j in DropedConstraintInds[iPriority]: MathConstraints[j] = None
            nVar, PreparedObjective, PreparedConstraints, PreparedOption = self._prepareModel(Objective, MathConstraints)
            TargetWeight, ResultInfo = self._solve(nVar, PreparedObjective, PreparedConstraints, PreparedOption)
            ReleasedConstraint += DropedConstraints[iPriority]
        ResultInfo['ReleasedConstraint'] = ReleasedConstraint
        self._isStarted = False
        if TargetWeight is not None: return (pd.Series(TargetWeight, index=self._TargetIDs), ResultInfo)
        else: return (None, ResultInfo)
    # 初始化, 返回依赖信息
    def init(self):
        if self._ModelChanged:
            self._Dependency = {"因子":[]}
            for iConstraint in self.Constraints:
                for jDependency, jValue in iConstraint.Dependency.items():
                    if jDependency=="因子":
                        self._Dependency["因子"].extend(jValue)
                    else:
                        self._Dependency[jDependency] = (self._Dependency.get(jDependency, False) or jValue)
            for jDependency, jValue in self.OptimObjective.Dependency.items():
                if jDependency=="因子":
                    self._Dependency["因子"].extend(jValue)
                else:
                    self._Dependency[jDependency] = (self._Dependency.get(jDependency, False) or jValue)
            self._Dependency["因子"] = list(set(self._Dependency["因子"]))
            self._ModelChanged = False
        return self._Dependency
    # 预处理数据
    def _preprocessData(self):
        if self.TargetIDs: self._TargetIDs = set(self.TargetIDs)
        else: self._TargetIDs = None
        if self._Dependency.get("预期收益", False):
            if self.ExpectedReturn is None: raise __QS_Error__("模型需要预期收益, 但尚未赋值!")
            ExpectedReturn = self.ExpectedReturn.dropna()
            if self._TargetIDs is None: self._TargetIDs = ExpectedReturn.index.tolist()
            else: self._TargetIDs = ExpectedReturn.index.intersection(self._TargetIDs).tolist()
        if self._Dependency.get("协方差矩阵", False):
            if self.FactorCov is not None:
                if self.RiskFactorData is None: raise __QS_Error__("模型需要风险因子暴露, 但尚未赋值!")
                if self.SpecificRisk is None: raise __QS_Error__("模型需要特异性风险, 但尚未赋值!")
                RiskFactorData = self.RiskFactorData.dropna(how="any", axis=0)
                SpecificRisk = self.SpecificRisk.dropna()
                if self._TargetIDs is None: self._TargetIDs = RiskFactorData.index.intersection(SpecificRisk.index).tolist()
                else: self._TargetIDs = RiskFactorData.index.intersection(SpecificRisk.index).intersection(self._TargetIDs).tolist()
            else:
                if self.CovMatrix is None: raise __QS_Error__("模型需要协方差矩阵, 但尚未赋值!")
                CovMatrix = self.CovMatrix.dropna(how="all", axis=0)
                CovMatrix = CovMatrix.loc[:, CovMatrix.index]
                CovMatrix = CovMatrix.dropna(how="any", axis=0)
                if self._TargetIDs is None: self._TargetIDs = CovMatrix.index.tolist()
                else: self._TargetIDs = CovMatrix.index.intersection(self._TargetIDs).tolist()
        if self._Dependency.get("成交金额", False):
            if self.AmountFactor is None: raise __QS_Error__("模型需要成交金额, 但尚未赋值!")
            if self._TargetIDs is None: self._TargetIDs = self.AmountFactor.index.tolist()
            else: self._TargetIDs = self.AmountFactor.index.intersection(self._TargetIDs).tolist()
        if self._Dependency.get("因子", []):
            MissingFactor = set(self._Dependency["因子"]).difference(self.FactorData.columns)
            if MissingFactor: raise __QS_Error__("模型需要因子: %s, 但尚未赋值!" % str(MissingFactor))
            self.FactorData = self.FactorData.loc[:, self._Dependency["因子"]]
            FactorData = self.FactorData.dropna(how="any", axis=0)
            if self._TargetIDs is None: self._TargetIDs = FactorData.index.tolist()
            else: self._TargetIDs = FactorData.index.intersection(self._TargetIDs).tolist()
        if self._TargetIDs is None: raise __QS_Error__("无法确定目标 ID 序列!")
        self._TargetIDs = sorted(self._TargetIDs)
        self._nID = len(self._TargetIDs)
        if self._Dependency.get("基准投资组合", False):
            self._BenchmarkExtraIDs = sorted(self.BenchmarkHolding[self.BenchmarkHolding>0].index.difference(self._TargetIDs))
            self._BenchmarkExtra = self.BenchmarkHolding[self._BenchmarkExtraIDs]
            self.BenchmarkHolding = self.BenchmarkHolding.loc[self._TargetIDs]
            self.BenchmarkHolding[pd.isnull(self.BenchmarkHolding)] = 0.0
        else:
            self._BenchmarkExtraIDs = []
            self.BenchmarkHolding = None
            self._BenchmarkExtra = pd.Series()
        TargetBenchmarkExtraIDs = self._TargetIDs + self._BenchmarkExtraIDs
        if self._Dependency.get("协方差矩阵", False):
            if self.FactorCov is not None:
                RiskFactorDataNAFillVal = self.RiskFactorData.mean()
                SpecialRiskNAFillVal = self.SpecificRisk.mean()
                self.RiskFactorData = self.RiskFactorData.loc[TargetBenchmarkExtraIDs]
                self.SpecificRisk = self.SpecificRisk.loc[TargetBenchmarkExtraIDs]
                self.RiskFactorData = self.RiskFactorData.fillna(RiskFactorDataNAFillVal)
                self.SpecificRisk = self.SpecificRisk.fillna(SpecialRiskNAFillVal)
                CovMatrix = np.dot(np.dot(self.RiskFactorData.values,self.FactorCov.values), self.RiskFactorData.values.T) + np.diag(self.SpecificRisk.values**2)
                self.CovMatrix = pd.DataFrame(CovMatrix, index=TargetBenchmarkExtraIDs, columns=TargetBenchmarkExtraIDs)
                self.RiskFactorData = self.RiskFactorData.loc[self._TargetIDs]
                self.SpecificRisk = self.SpecificRisk.loc[self._TargetIDs]
            self._BenchmarkExtraCov = self.CovMatrix.loc[self._BenchmarkExtraIDs, self._BenchmarkExtraIDs]
            self._BenchmarkExtraCov1 = self.CovMatrix.loc[self._BenchmarkExtraIDs, self._TargetIDs]
            self.CovMatrix = self.CovMatrix.loc[self._TargetIDs, self._TargetIDs]
        if self._Dependency.get("预期收益", False):
            if self.ExpectedReturn is None: raise __QS_Error__("模型需要预期收益, 但尚未赋值!")
            ExpectedReturn = self.ExpectedReturn.loc[TargetBenchmarkExtraIDs]
            self._BenchmarkExtraExpectedReturn = ExpectedReturn.loc[self._BenchmarkExtraIDs]
            self._BenchmarkExtraExpectedReturn = self._BenchmarkExtraExpectedReturn.fillna(0.0)
            self.ExpectedReturn = ExpectedReturn.loc[self._TargetIDs]
            self.ExpectedReturn = self.ExpectedReturn.fillna(0.0)
        if self._Dependency.get("因子", []):
            FactorData = FactorData.loc[TargetBenchmarkExtraIDs, :]
            self._BenchmarkExtraFactorData = FactorData.loc[self._BenchmarkExtraIDs]
            self._BenchmarkExtraFactorData = self._BenchmarkExtraFactorData.fillna(0.0)
            self.FactorData = FactorData.loc[self._TargetIDs]
            self.FactorData = self.FactorData.fillna(0.0)
        if self._Dependency.get("初始投资组合", False):
            if self.Holding is None: raise __QS_Error__("模型需要初始投资组合, 但尚未赋值!")
            self._HoldingExtraIDs = sorted(self.Holding[self.Holding>0].index.difference(self._TargetIDs))
            self._HoldingExtra = self.Holding[self._HoldingExtraIDs]
            Holding = self.Holding.loc[self._TargetIDs+self._HoldingExtraIDs]
            self.Holding = Holding.loc[self._TargetIDs]
            self.Holding[pd.isnull(self.Holding)] = 0.0
        else:
            self._HoldingExtraIDs = []
            self.Holding = None
            self._HoldingExtra = pd.Series()
        if self._Dependency.get("成交金额", False):
            TargetHoldingExtraIDs = self._TargetIDs + self._HoldingExtraIDs
            Amount = self.AmountFactor.loc[TargetHoldingExtraIDs]
            self._HoldingExtraAmount = Amount.loc[self._HoldingExtraIDs]
            self._HoldingExtraAmount = self._HoldingExtraAmount.fillna(0.0)
            self.AmountFactor = Amount.loc[self._TargetIDs]
            self.AmountFactor = self.AmountFactor.fillna(0.0)
        self._DataChanged = False
        return 0
    # 整理条件，形成优化模型
    def _prepareModel(self, objective, contraints):
        nVar = len(self._TargetIDs)
        PreparedConstraints = {}
        for iConstraint in contraints:
            if iConstraint is None: continue
            elif iConstraint['type'] == "Box":
                PreparedConstraints["Box"] = PreparedConstraints.get("Box",{"lb":np.zeros((nVar,1))-np.inf,"ub":np.zeros((nVar,1))+np.inf,"type":"Box"})
                PreparedConstraints["Box"]["lb"] = np.maximum(PreparedConstraints["Box"]["lb"],iConstraint["lb"])
                PreparedConstraints["Box"]["ub"] = np.minimum(PreparedConstraints["Box"]["ub"],iConstraint["ub"])
            elif iConstraint['type'] == "LinearIn":
                PreparedConstraints["LinearIn"] = PreparedConstraints.get("LinearIn",{"A":np.zeros((0,nVar)),"b":np.zeros((0,1)),"type":"LinearIn"})
                PreparedConstraints["LinearIn"]["A"] = np.vstack((PreparedConstraints["LinearIn"]["A"],iConstraint["A"]))
                PreparedConstraints["LinearIn"]["b"] = np.vstack((PreparedConstraints["LinearIn"]["b"],iConstraint["b"]))
            elif iConstraint["type"] == "LinearEq":
                PreparedConstraints["LinearEq"] = PreparedConstraints.get("LinearEq",{"Aeq":np.zeros((0,nVar)),"beq":np.zeros((0,1)),"type":"LinearEq"})
                PreparedConstraints["LinearEq"]["Aeq"] = np.vstack((PreparedConstraints["LinearEq"]["Aeq"],iConstraint["Aeq"]))
                PreparedConstraints["LinearEq"]["beq"] = np.vstack((PreparedConstraints["LinearEq"]["beq"],iConstraint["beq"]))
            else:
                PreparedConstraints[iConstraint["type"]] = PreparedConstraints.get(iConstraint["type"],[])
                PreparedConstraints[iConstraint["type"]].append(iConstraint)
        PreparedOption = self._genOption()
        return (nVar, objective, PreparedConstraints, PreparedOption)
    # 整理选项参数
    def _genOption(self):
        return self.OptimOption
    # 求解一次优化问题, 返回: (array(nvar) 或 None, 其他信息: {})
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        return (None, {})
    # 检查当前的优化问题是否可解
    def _checkSolvability(self):
        for iConstraint in self.Constraints:
            if iConstraint.Type not in self._SupportedContraintType:
                return ("不支持类型为'%s' 的条件" % iConstraint.Type)
        return None
    # 计算波动率约束优化后的实现值
    def _calRealizedVolatilityConstraint(self, optimal_w, constraint):
        if not constraint.Benchmark:
            return np.dot(optimal_w.values, np.dot(optimal_w.values, self.CovMatrix.loc[self._TargetIDs, self._TargetIDs].values))**0.5
        else:
            IDs = list(set(self.BenchmarkHolding.index).intersection(set(self.CovMatrix.index)))
            Mu = -2*((self.BenchmarkHolding.loc[IDs]*self.CovMatrix.loc[IDs,IDs]).T.sum())
            temp = (-Mu/2*self.BenchmarkHolding).sum()
            if pd.isnull(temp): temp = 0.0
            Mu = Mu.loc[self._TargetIDs]
            Mu[pd.isnull(Mu)] = 0.0
            return (np.dot(optimal_w.values, np.dot(optimal_w.values, self.CovMatrix.loc[self._TargetIDs, self._TargetIDs].values)) + np.dot(Mu.values, optimal_w.values) + temp)**0.5