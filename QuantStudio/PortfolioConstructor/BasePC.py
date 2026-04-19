# -*- coding: utf-8 -*-
from typing import List, Optional, Literal, Tuple

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var


class OptimizationObjective(__QS_Object__):
    """优化目标"""
    
    def genObjective(self) -> dict:
        """生成优化目标对应的数学形式
        
        Returns:
            根据不同的优化目标类型，返回不同的形式。以下是两个可能的形式：
            * 线性目标: f' * x + c, 则返回: {'f': array(n, 1), 'constant': c, 'type': 'Linear'}
            * 二次目标: x' * Sigma * x + Mu' * x, 则返回: {'Sigma': array(n, n), 'Mu': array(n,), 'constant': c, 'type': 'Quadratic'}
        """
        raise NotImplementedError


class MeanVarianceObjective(OptimizationObjective):
    """均值方差优化目标
    数学形式: 
    线性目标: f'*x, {'f': array(n, 1), 'type': 'Linear'}
    二次目标: x'*Sigma*x + Mu'*x, {'Sigma': array(n, n), 'X': array(n, k), 'F': array(k, k), 'Delta': array(n,), 'Mu': array(n,), 'type': 'Quadratic'}, 其中 Sigma = X*F*X'+Delta
    L1 惩罚线性目标: f'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'f':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Linear'}
    L1 惩罚二次目标: x'*Sigma*x + Mu'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Quadratic'}, 其中, Sigma = X*F*X'+Delta
    """

    class __QS_ArgClass__(OptimizationObjective.__QS_ArgClass__):
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)
        ExpectedReturnCoef: float = Field(default=0.0, title="收益项系数", frozen=True)
        RiskAversionCoef: float = Field(default=1.0, title="风险厌恶系数", frozen=True)
        TurnoverPenaltyCoef: float = Field(default=0.0, title="换手惩罚系数", frozen=True)
        BuyPenaltyCoef: float = Field(default=0.0, title="买入惩罚系数", frozen=True)
        SellPenaltyCoef: float = Field(default=0.0, title="卖出惩罚系数", frozen=True)
    
    def __init__(self, mask: NDArray[np.bool], expected_return: Optional[NDArray[np.float64]]=None, p0:Optional[NDArray[np.float64]]=None, bmk: Optional[NDArray[np.float64]]=None, factor_cov: Optional[NDArray[np.float64]]=None, factor_data: Optional[NDArray[np.float64]]=None, specific_risk: Optional[NDArray[np.float64]]=None, cov: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化均值方差优化目标对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            expected_return: 预期收益, array(shape=(n,)), 缺失值会被填充为 0
            p0: 初始投资组合, array(shape=(n,)), 缺失值会被填充为 0
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            factor_cov: 因子协方差阵, array(shape=(k, k)), 其中 k 是因子数量
            factor_data: 因子暴露矩阵, array(shape=(n, k))
            specific_risk: 特异性风险, array(shape=(n,))
            cov: 证券协方差阵, array(shape=(n, n)), 如果 factor_cov, factor_data, specific_risk 均非 None, 则使用这三者计算出来的协方差阵；否则使用 cov
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)
        if (self._QSArgs.ExpectedReturnCoef != 0) and (expected_return is None): raise __QS_Error__("优化目标需要预期收益，但入参 expected_return 为 None!")
        if expected_return is None: expected_return = np.zeros(mask.shape)
        else: expected_return = np.where(pd.notnull(expected_return), expected_return, 0)
        if (not (((factor_cov is not None) and (factor_data is not None) and (specific_risk is not None)) or (cov is not None))) and (self._QSArgs.RiskAversionCoef != 0.0):
            raise __QS_Error__("优化目标需要风险矩阵，但入参 factor_cov, factor_data, specific_risk, cov 为 None!")
        if ((self._QSArgs.TurnoverPenaltyCoef != 0.0) or (self._QSArgs.BuyPenaltyCoef != 0.0) or (self._QSArgs.SellPenaltyCoef != 0.0)) and (p0 is None):
            raise __QS_Error__("优化目标需要初始投资组合，但入参 p0 为 None!")
        
        self._Mask = mask
        self._ExpectedReturn = expected_return
        self._P0 = p0
        self._Bmk = bmk
        self._FactorCov = factor_cov
        self._FactorData = factor_data
        self._SpecificRisk = specific_risk
        self._Cov = cov

    def genObjective(self) -> dict:
        ObjectiveConstant = 0.0
        Mu = self._QSArgs.ExpectedReturnCoef * self._ExpectedReturn[self._Mask]
        ObjectiveConstant += - self._QSArgs.ExpectedReturnCoef * np.dot(self._ExpectedReturn, self._Bmk)
        if not np.all(self._Mask):
            ObjectiveConstant += self._QSArgs.ExpectedReturnCoef * np.dot(self._ExpectedReturn[~self._Mask], (0 - self._Bmk[~self._Mask]))

        if self._QSArgs.RiskAversionCoef != 0.0:
            if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
                Sigma = np.dot(np.dot(self._FactorData, self._FactorCov), self._FactorData.T) + np.diag(self._SpecificRisk ** 2)
            elif self._Cov is not None:
                Sigma = self._Cov
            Sigma = np.where(pd.notnull(Sigma), Sigma, 0)
            RiskCoef = - self._QSArgs.RiskAversionCoef / 2
            Mu += -2 * RiskCoef * np.dot(self._Bmk[self._Mask], Sigma[self._Mask][:, self._Mask])
            if not np.all(self._Mask):
                Mu += 2 * RiskCoef * np.dot((0 - self._Bmk[~self._Mask]), Sigma[~self._Mask][:, self._Mask])
            ObjectiveConstant += RiskCoef * np.dot(np.dot(self._Bmk[self._Mask], Sigma[self._Mask][:, self._Mask]), self._Bmk[self._Mask])
            if not np.all(self._Mask):
                ObjectiveConstant += -2 * RiskCoef * np.dot(np.dot((0 - self._Bmk[~self._Mask]), Sigma[~self._Mask][:, self._Mask]), self._Bmk[self._Mask])
                ObjectiveConstant += RiskCoef * np.dot(np.dot((0 - self._Bmk[~self._Mask]), Sigma[~self._Mask][:, ~self._Mask]), (0 - self._Bmk[~self._Mask]))
            if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
                Objective = {
                    "type": "Quadratic",
                    "minmax": "max",
                    "X": self._FactorData[self._Mask],
                    "F": RiskCoef * self._FactorCov,
                    "Delta": RiskCoef * self._SpecificRisk[self._Mask] ** 2,
                    "Mu": Mu
                }
            else:
                Objective = {
                    "type": "Quadratic",
                    "minmax": "max",
                    "Sigma": RiskCoef * Sigma[self._Mask][:, self._Mask],
                    "Mu": Mu
                }
        else:
            Objective = {"type": "Linear", "f": Mu, "minmax": "max"}
        
        if self._P0 is not None: p0 = np.where(pd.notnull(self._P0), self._P0, 0)
        if self._QSArgs.TurnoverPenaltyCoef != 0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda1': self._QSArgs.TurnoverPenaltyCoef, "c": p0[self._Mask]})
            if not np.all(self._Mask):
                ObjectiveConstant += self._QSArgs.TurnoverPenaltyCoef * np.sum(np.abs(0 - p0[~self._Mask]))
        if self._QSArgs.BuyPenaltyCoef != 0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda2': self._QSArgs.BuyPenaltyCoef, "c_pos": p0[self._Mask]})
            if not np.all(self._Mask):
                ObjectiveConstant += self._QSArgs.BuyPenaltyCoef * np.sum(np.clip(0 - p0[~self._Mask], 0, np.inf))
        if self._QSArgs.SellPenaltyCoef != 0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda3': self._QSArgs.SellPenaltyCoef, "c_neg": p0[self._Mask]})
            if not np.all(self._Mask):
                ObjectiveConstant += self._QSArgs.SellPenaltyCoef * np.sum(- np.clip(0 - p0[~self._Mask], -np.inf, 0))
        
        Objective["constant"] = ObjectiveConstant
        return Objective


class MaxSharpeObjective(OptimizationObjective):
    """最大夏普率优化目标
    数学形式: (f'*x + f0) / sqrt(x'Sigma*x + Mu'x + q),{'f':array(n,1),'f0':double,'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Mu':array(n,1),'Delta':array(n,1),'q':double,'type':'Sharpe'},其中，Sigma = X*F*X'+Delta
    """

    class __QS_ArgClass__(OptimizationObjective.__QS_ArgClass__):
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)
    
    def __init__(self, mask: NDArray[np.bool], expected_return: NDArray[np.float64], bmk: Optional[NDArray[np.float64]]=None, factor_cov: Optional[NDArray[np.float64]]=None, factor_data: Optional[NDArray[np.float64]]=None, specific_risk: Optional[NDArray[np.float64]]=None, cov: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化最大夏普率优化目标对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            expected_return: 预期收益, array(shape=(n,)), 缺失值会被填充为 0
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            factor_cov: 因子协方差阵, array(shape=(k, k)), 其中 k 是因子数量
            factor_data: 因子暴露矩阵, array(shape=(n, k))
            specific_risk: 特异性风险, array(shape=(n,))
            cov: 证券协方差阵, array(shape=(n, n)), 如果 factor_cov, factor_data, specific_risk 均非 None, 则使用这三者计算出来的协方差阵；否则使用 cov
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)
        if (self._QSArgs.ExpectedReturnCoef != 0) and (expected_return is None): raise __QS_Error__("优化目标需要预期收益，但入参 expected_return 为 None!")
        if expected_return is None: expected_return = np.zeros(mask.shape)
        else: expected_return = np.where(pd.notnull(expected_return), expected_return, 0)
        if (not (((factor_cov is not None) and (factor_data is not None) and (specific_risk is not None)) or (cov is not None))):
            raise __QS_Error__("优化目标需要风险矩阵，但入参 factor_cov, factor_data, specific_risk, cov 为 None!")
        
        self._Mask = mask
        self._ExpectedReturn = expected_return
        self._Bmk = bmk
        self._FactorCov = factor_cov
        self._FactorData = factor_data
        self._SpecificRisk = specific_risk
        self._Cov = cov

    def genObjective(self) -> dict:
        if not self._QSArgs.Benchmark:
            if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
                return {
                    "type": "Sharpe",
                    "minmax": "max",
                    "f": self._ExpectedReturn[self._Mask],
                    "f0": 0.0,
                    "X": self._FactorData[self._Mask],
                    "F": self._FactorCov,
                    "Delta": self._SpecificRisk[self._Mask] ** 2,
                    "Mu": np.zeros((np.sum(self._Mask), )),
                    "q": 0.0
                }
            else:
                return {
                    "type": "Sharpe",
                    "minmax": "max",
                    "f": self._ExpectedReturn[self._Mask],
                    "f0": 0.0,
                    "Sigma": self._Cov[self._Mask][:, self._Mask],
                    "Mu": np.zeros((np.sum(self._Mask), )),
                    "q": 0.0
                }
        
        if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
            Sigma = np.dot(np.dot(self._FactorData, self._FactorCov), self._FactorData.T) + np.diag(self._SpecificRisk ** 2)
        elif self._Cov is not None:
            Sigma = self._Cov
        Sigma = np.where(pd.notnull(Sigma), Sigma, 0)
        Objective = {
            "type": "Sharpe",
            "f": self._ExpectedReturn[self._Mask],
            "f0": - np.dot(self._ExpectedReturn[self._Mask], self._Bmk[self._Mask]) + (np.dot(self._ExpectedReturn[~self._Mask], (0 - self._Bmk[~self._Mask])) if not np.all(self._Mask) else 0)
        }
        Sigma = np.where(pd.notnull(Sigma), Sigma, 0)
        Mu = -2 * np.dot(self._Bmk[self._Mask], Sigma[self._Mask][:, self._Mask])
        if not np.all(self._Mask):
            Mu += 2 * np.dot((0 - self._Bmk[~self._Mask]), Sigma[~self._Mask][:, self._Mask])
        q = np.dot(np.dot(self._Bmk[self._Mask], Sigma[self._Mask][:, self._Mask]), self._Bmk[self._Mask])
        if not np.all(self._Mask):
            q += -2 * np.dot(np.dot((0 - self._Bmk[~self._Mask]), Sigma[~self._Mask][:, self._Mask]), self._Bmk[self._Mask])
            q += np.dot(np.dot((0 - self._Bmk[~self._Mask]), Sigma[~self._Mask][:, ~self._Mask]), (0 - self._Bmk[~self._Mask]))
        Objective["Mu"] = Mu
        Objective["q"] = q
        if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
            Objective["X"] = self._FactorData[self._Mask]
            Objective["F"] = self._FactorCov
            Objective["Delta"] = self._SpecificRisk[self._Mask] ** 2
        else:
            Objective["Sigma"] = Sigma[self._Mask][:, self._Mask]
        return Objective


class RiskBudgetObjective(OptimizationObjective):
    """风险预算优化目标
    数学形式: {'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'b':array(n,1),type':'Risk_Budget'}
    """

    def __init__(self, mask: NDArray[np.bool], budget: Optional[NDArray[np.float64]]=None, factor_cov: Optional[NDArray[np.float64]]=None, factor_data: Optional[NDArray[np.float64]]=None, specific_risk: Optional[NDArray[np.float64]]=None, cov: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化风险预算优化目标对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            budget: 风险预算, array(shape=(n,)), 缺失值会被填充为 0, None 表示风险平价
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            factor_cov: 因子协方差阵, array(shape=(k, k)), 其中 k 是因子数量
            factor_data: 因子暴露矩阵, array(shape=(n, k))
            specific_risk: 特异性风险, array(shape=(n,))
            cov: 证券协方差阵, array(shape=(n, n)), 如果 factor_cov, factor_data, specific_risk 均非 None, 则使用这三者计算出来的协方差阵；否则使用 cov
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if budget is None: budget = np.ones(mask.shape) / np.sum(mask)
        else: budget = np.where(pd.notnull(budget), budget, 0)

        if (not (((factor_cov is not None) and (factor_data is not None) and (specific_risk is not None)) or (cov is not None))):
            raise __QS_Error__("优化目标需要风险矩阵，但入参 factor_cov, factor_data, specific_risk, cov 为 None!")
        
        self._Mask = mask
        self._Budget = budget
        self._FactorCov = factor_cov
        self._FactorData = factor_data
        self._SpecificRisk = specific_risk
        self._Cov = cov

    def genObjective(self) -> dict:
        if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
            Objective = {
                "type": "Risk_Budget", 
                "minmax": "min",
                "X": self._FactorData[self._Mask], 
                "F": self._FactorCov,
                "Delta": self._SpecificRisk[self._Mask] ** 2
            }
        else:
            Objective = {"type": "Risk_Budget", "Sigma": self._Cov[self._Mask][:, self._Mask]}
        
        Objective["b"] = self._Budget[self._Mask]
        return Objective


class MaxDiversificationObjective(OptimizationObjective):
    """最大分散化优化目标
    数学形式: {'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),type':'Max_Diversification'}
    """

    def __init__(self, mask: NDArray[np.bool], factor_cov: Optional[NDArray[np.float64]]=None, factor_data: Optional[NDArray[np.float64]]=None, specific_risk: Optional[NDArray[np.float64]]=None, cov: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化最大分散化优化目标对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            factor_cov: 因子协方差阵, array(shape=(k, k)), 其中 k 是因子数量
            factor_data: 因子暴露矩阵, array(shape=(n, k))
            specific_risk: 特异性风险, array(shape=(n,))
            cov: 证券协方差阵, array(shape=(n, n)), 如果 factor_cov, factor_data, specific_risk 均非 None, 则使用这三者计算出来的协方差阵；否则使用 cov
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if (not (((factor_cov is not None) and (factor_data is not None) and (specific_risk is not None)) or (cov is not None))):
            raise __QS_Error__("优化目标需要风险矩阵，但入参 factor_cov, factor_data, specific_risk, cov 为 None!")
        
        self._Mask = mask
        self._FactorCov = factor_cov
        self._FactorData = factor_data
        self._SpecificRisk = specific_risk
        self._Cov = cov

    def genObjective(self) -> dict:
        if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
            Objective = {
                "type": "Max_Diversification", 
                "minmax": "max",
                "X": self._FactorData[self._Mask], 
                "F": self._FactorCov,
                "Delta": self._SpecificRisk[self._Mask] ** 2
            }
        else:
            Objective = {"type": "Max_Diversification", "Sigma": self._Cov[self._Mask][:, self._Mask]}
        return Objective


class Constraint(__QS_Object__):
    """约束条件
    数学形式的约束条件
    Box 约束：lb <= x <= ub,{'lb':array(n,1),'ub':array(n,1),'type':'Box'}
    线性不等式约束：A * x <= b,{'A':array(m,n),'b':array(m,1),'type':'LinearIn'}
    线性等式约束：Aeq * x == beq,{'Aeq':array(m,n),'beq':array(m,1),'type':'LinearEq'}
    二次约束：x'*Sigma*x + Mu'*x <= q,{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'q':double,'type':'Quadratic'},其中，Sigma = X*F*X'+Delta
    L1 范数约束：sum(abs(x-c)) <= l,{'c':array(n,1),'l':double,'type':'L1'}
    正部总约束：sum((x-c_pos)^+) <= l_pos，{'c_pos':array(n,1),'l_pos':double,'type':'Pos'}
    负部总约束：sum((x-c_neg)^-) <= l_neg，{'c_neg':array(n,1),'l_neg':double,'type':'Neg'}
    非零数目约束：sum((x-b)!=0) <= N, {'b':array(n,1),'N':double,'type':'NonZeroNum'}
    """

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        DropPriority: float = Field(default=-1.0, title="舍弃优先级", frozen=True, description="当求解失败时，该约束条件可以被舍弃的优先级")

    def genConstraint(self) -> List[dict]:
        raise NotImplementedError


class BudgetConstraint(Constraint):
    """预算约束: i'*(w-benchmark) <=(==,>=) a, 转换成线性等式约束或线性不等式约束"""
    class __QS_ArgClass__(Constraint.__QS_ArgClass__):
        UpLimit: float = Field(default=1.0, title="限制上限", frozen=True)
        DownLimit: float = Field(default=1.0, title="限制下限", frozen=True)
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)

    def __init__(self, mask:NDArray[np.bool], bmk: Optional[NDArray[np.float64]]=None, args:dict = {}, config_file:Optional[str] = None, **kwargs):
        """初始化预算约束条件对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.UpLimit < self._QSArgs.DownLimit: raise __QS_Error__("限制上限必须大于等于限制下限!")
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)

        self._Mask = mask
        self._Bmk = bmk
    
    def genConstraint(self) -> List[dict]:
        Constraints = []
        aAdj = np.sum(self._Bmk)
        if self._QSArgs.UpLimit==self._QSArgs.DownLimit:
            Constraints.append({
                "type":"LinearEq",
                "Aeq": np.ones((1, np.sum(self._Mask))),
                "beq": np.array([[self._QSArgs.UpLimit + aAdj]])}
            )
        else:
            if self._QSArgs.DownLimit > -np.inf:
                Constraints.append({
                    "type": "LinearIn",
                    "A": - np.ones((1, np.sum(self._Mask))),
                    "b": - np.array([[self._QSArgs.DownLimit + aAdj]])
                })
            if self._QSArgs.UpLimit < np.inf:
                Constraints.append({
                    "type": "LinearIn",
                    "A": np.ones((1, np.sum(self._Mask))),
                    "b": np.array([[self._QSArgs.UpLimit + aAdj]])
                })
        return Constraints


class FactorExposeConstraint(Constraint):
    """因子暴露约束: f'*(w-benchmark) <=(==,>=) a, 转换成线性等式约束或线性不等式约束"""

    class __QS_ArgClass__(Constraint.__QS_ArgClass__):
        FactorType: Literal["数值型", "类别型"] = Field(default="数值型", title="因子类型", frozen=True)
        UpLimit: float = Field(default=1.0, title="限制上限", frozen=True)
        DownLimit: float = Field(default=1.0, title="限制下限", frozen=True)
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)
    
    def __init__(self, mask:NDArray[np.bool], factor_data:NDArray, bmk: Optional[NDArray[np.float64]]=None, args:dict = {}, config_file:Optional[str] = None, **kwargs):
        """初始化因子暴露约束条件对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            factor_data: 因子暴露数据, array(shape=(n, k)), 其中 k 是因子数量, 缺失值会被填充为 0
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.UpLimit < self._QSArgs.DownLimit: raise __QS_Error__("限制上限必须大于等于限制下限!")
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)

        self._Mask = mask
        self._FactorData = factor_data
        self._Bmk = bmk
    
    # 生成数值型因子暴露约束条件的优化器条件形式
    def _genNumFactorExposeConstraint(self, mask:NDArray[np.bool], factor_data:NDArray[np.float64], bmk: NDArray[np.float64]):
        Constraints = []
        factor_data = np.where(pd.notnull(factor_data), factor_data, 0)
        aAdj = np.dot(bmk, factor_data)
        A = factor_data[mask].T
        FactorMask = (np.abs(A).sum(axis=1) != 0.0)
        if self._QSArgs.UpLimit == self._QSArgs.DownLimit:
            Constraints.append({
                "type": "LinearEq",
                "Aeq": A[FactorMask, :],
                "beq": (self._QSArgs.UpLimit + aAdj)[FactorMask]
            })
        else:
            if self._QSArgs.DownLimit > -np.inf:
                Constraints.append({
                    "type": "LinearIn",
                    "A": - A[FactorMask, :],
                    "b": - (self._QSArgs.DownLimit + aAdj)[FactorMask]
                })
            if self._QSArgs.UpLimit < np.inf:
                Constraints.append({
                    "type": "LinearIn",
                    "A": A[FactorMask, :],
                    "b": (self._QSArgs.UpLimit + aAdj)[FactorMask]
                })
        return Constraints
    
    # 生成类别型因子暴露约束条件的优化器条件形式
    def _genClassFactorExposeConstraint(self, mask:NDArray[np.bool], factor_data:NDArray[np.float64], bmk: NDArray[np.float64]):
        Constraints = []
        for i in range(factor_data.shape[1]):
            iFactorData = factor_data[:, i]
            iFactorData = DummyVarTo01Var(pd.Series(iFactorData), ignore_na=True, ignore_nonstring=True).values
            aAdj = np.dot(bmk, iFactorData)
            A = iFactorData[mask].T
            FactorMask = (np.abs(A).sum(axis=1) != 0.0)
            if self._QSArgs.UpLimit == self._QSArgs.DownLimit:
                Constraints.append({
                    "type": "LinearEq",
                    "Aeq": A[FactorMask, :],
                    "beq":(self._QSArgs.UpLimit + aAdj)[FactorMask]
                })
            else:
                if self._QSArgs.DownLimit > -np.inf:
                    Constraints.append({
                        "type": "LinearIn",
                        "A": - A[FactorMask, :],
                        "b": - (self._QSArgs.DownLimit + aAdj)[FactorMask]
                    })
                if self._QSArgs.UpLimit < np.inf:
                    Constraints.append({
                        "type": "LinearIn",
                        "A": A[FactorMask,:],
                        "b": (self._QSArgs.UpLimit + aAdj)[FactorMask]
                    })
        return Constraints
    
    def genConstraint(self) -> List[dict]:
        if self._QSArgs.FactorType=="数值型":
            return self._genNumFactorExposeConstraint(self._Mask, self._FactorData, self._Bmk)
        else:
            return self._genClassFactorExposeConstraint(self._Mask, self._FactorData, self._Bmk)


class WeightConstraint(Constraint):
    """权重约束: (w-benchmark) <=(>=) a, 转换成 Box 约束"""

    class __QS_ArgClass__(Constraint.__QS_ArgClass__):
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)
    
    def __init__(self, mask:NDArray[np.bool], bmk: Optional[NDArray[np.float64]]=None, up_limit: Optional[NDArray[np.float64] | float]=None, down_limit: Optional[NDArray[np.float64] | float]=None, args:dict = {}, config_file:Optional[str] = None, **kwargs):
        """初始化权重约束条件对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            up_limit: 约束上限, array(shape=(n,)) 或者 float, 缺失值会被填充为 inf
            down_limit: 约束下限, array(shape=(n,)) 或者 float, 缺失值会被填充为 -inf
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)

        self._Mask = mask
        self._Bmk = bmk
        self._UpLimit = up_limit
        self._DownLimit = down_limit

    def genConstraint(self) -> List[dict]:
        if self._UpLimit is None:
            UpConstraint = np.full(shape=(np.sum(self._Mask),), fill_value=np.inf, dtype=float)
        elif not isinstance(self._UpLimit, np.ndarray):
            UpConstraint = np.full(shape=(np.sum(self._Mask),), fill_value=self._UpLimit, dtype=float)
        else:
            UpConstraint = np.where(pd.isnull(self._UpLimit[self._Mask]), np.inf, self._UpLimit[self._Mask])
        if self._DownLimit is None:
            DownConstraint = np.full(shape=(np.sum(self._Mask),), fill_value=-np.inf, dtype=float)
        elif not isinstance(self._DownLimit, np.ndarray):
            DownConstraint = np.full(shape=(np.sum(self._Mask),), fill_value=self._DownLimit, dtype=float)
        else:
            DownConstraint = np.where(pd.isnull(self._DownLimit[self._Mask]), -np.inf, self._DownLimit[self._Mask])
        if self._QSArgs.Benchmark:
            UpConstraint += self._Bmk[self._Mask]
            DownConstraint += self._Bmk[self._Mask]
        return [{"type": "Box", "lb": DownConstraint, "ub": UpConstraint}]


class TurnoverConstraint(Constraint):
    """换手约束: sum(abs(w-w0)) <=(==) a, 转换成 L1 范数约束, 正部总约束, 负部总约束"""

    class __QS_ArgClass__(Constraint.__QS_ArgClass__):
        ConstraintType: Literal["总换手限制", "总买入限制", "总卖出限制", "买卖限制", "买入限制", "卖出限制"] = Field(default="总换手限制", title="限制类型", frozen=True)
        AmtMultiple: float = Field(default=1.0, title="成交额倍数", frozen=True)
        UpLimit: float = Field(default=0.7, title="限制上限", frozen=True)
    
    def __init__(self, mask: NDArray[np.bool], p0:NDArray[np.float64], wealth:Optional[float]=None, amt: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化换手约束条件对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            p0: 初始投资组合, array(shape=(n,)), 缺失值会被填充为 0
            wealth: 账户总金额
            amt: 证券成交金额, array(shape=(n,)), 缺失值会被填充为 0
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if (self._QSArgs.AmtMultiple != 0):
            if (amt is None): raise __QS_Error__("优化目标需要成交额，但入参 amt 为 None!")
            if (wealth is None): raise __QS_Error__("优化目标需要总持仓金额，但入参 wealth 为 None!")
        self._Mask = mask
        self._P0 = p0
        self._Amt = amt
        self._Wealth = wealth

    def genConstraint(self) -> List[dict]:
        if self._QSArgs.ConstraintType=="总换手限制":
            aAdj = self._QSArgs.UpLimit - np.sum(self._P0[~self._Mask])
            return [{"type": "L1", "c": self._P0[self._Mask], "l": aAdj}]
        elif self._QSArgs.ConstraintType=="总买入限制":
            aAdj = self._QSArgs.UpLimit + np.sum(np.clip(self._P0[~self._Mask], -np.inf, 0))
            return [{"type": "Pos", "c_pos": self._P0[self._Mask], "l": aAdj}]
        elif self._QSArgs.ConstraintType=="总卖出限制":
            aAdj = self._QSArgs.UpLimit - np.sum(np.clip(self._P0[~self._Mask], 0, np.inf))
            return [{"type": "Neg", "c_neg": self._P0[self._Mask], "l": aAdj}]
        if self._QSArgs.AmtMultiple == 0.0:
            aAdj = np.zeros((np.sum(self._Mask), )) + self._QSArgs.UpLimit
        else:
            aAdj = self._Amt[self._Mask] * self._QSArgs.AmtMultiple / self._Wealth
        if self._QSArgs.ConstraintType=="买卖限制":
            return [{"type": "Box", "ub": aAdj + self._P0[self._Mask], "lb": -aAdj + self._P0[self._Mask]}]
        elif self._QSArgs.ConstraintType=="买入限制":
            return [{"type": "Box", "ub": aAdj + self._P0[self._Mask], "lb": np.zeros((np.sum(self._Mask),)) - np.inf}]
        elif self._QSArgs.ConstraintType=="卖出限制":
            return [{"type": "Box", "ub": np.zeros((np.sum(self._Mask),)) + np.inf, "lb": -aAdj + self._P0[self._Mask]}]
        return []


class VolatilityConstraint(Constraint):
    """波动率约束: (w-benchmark)'*Cov*(w-benchmark) <= a, 转换成二次约束"""

    class __QS_ArgClass__(Constraint.__QS_ArgClass__):
        UpLimit: float = Field(default=0.06, title="限制上限", frozen=True)
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)
    
    def __init__(self, mask: NDArray[np.bool], bmk: Optional[NDArray[np.float64]]=None, factor_cov: Optional[NDArray[np.float64]]=None, factor_data: Optional[NDArray[np.float64]]=None, specific_risk: Optional[NDArray[np.float64]]=None, cov: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化波动率约束条件对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            factor_cov: 因子协方差阵, array(shape=(k, k)), 其中 k 是因子数量
            factor_data: 因子暴露矩阵, array(shape=(n, k))
            specific_risk: 特异性风险, array(shape=(n,))
            cov: 证券协方差阵, array(shape=(n, n)), 如果 factor_cov, factor_data, specific_risk 均非 None, 则使用这三者计算出来的协方差阵；否则使用 cov
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)
        if (not (((factor_cov is not None) and (factor_data is not None) and (specific_risk is not None)) or (cov is not None))):
            raise __QS_Error__("优化目标需要风险矩阵，但入参 factor_cov, factor_data, specific_risk, cov 为 None!")
        
        self._Mask = mask
        self._Bmk = bmk
        self._FactorCov = factor_cov
        self._FactorData = factor_data
        self._SpecificRisk = specific_risk
        self._Cov = cov
    
    def genConstraint(self) -> List[dict]:
        if self._QSArgs.Benchmark:
            if (self._FactorCov is not None) and (self._FactorData is not None) and (self._SpecificRisk is not None):
                Sigma = np.dot(np.dot(self._FactorData, self._FactorCov), self._FactorData.T) + np.diag(self._SpecificRisk ** 2)
            elif self._Cov is not None:
                Sigma = self._Cov
            Sigma = np.where(pd.notnull(Sigma), Sigma, 0)
            Mu = -2 * np.dot(self._Bmk[self._Mask], Sigma[self._Mask][:, self._Mask])
            if not np.all(self._Mask):
                Mu += 2 * np.dot((0 - self._Bmk[~self._Mask]), Sigma[~self._Mask][:, self._Mask])
            q = self._QSArgs.UpLimit ** 2 - np.dot(np.dot(self._Bmk[self._Mask], Sigma[self._Mask][:, self._Mask]), self._Bmk[self._Mask])
            if not np.all(self._Mask):
                q -= 2 * np.dot(np.dot(self._Bmk[~self._Mask], Sigma[~self._Mask][:, self._Mask]), self._Bmk[self._Mask])
                q -= np.dot(np.dot(self._Bmk[~self._Mask], Sigma[~self._Mask][:, ~self._Mask]), self._Bmk[~self._Mask])
        else:
            Mu = np.zeros((np.sum(self._Mask), ))
            q = self._QSArgs.UpLimit ** 2
        Constraint = {"type": "Quadratic", "Mu": Mu, "q": q}
        if self.FactorCov is not None:
            Constraint["X"] = self._FactorData[self._Mask]
            Constraint["F"] = self._FactorCov
            Constraint["Delta"] = self._SpecificRisk[self._Mask] ** 2
        else:
            Constraint['Sigma'] = self._Cov[self._Mask][:, self._Mask]
        return [Constraint]


class ExpectedReturnConstraint(Constraint):
    """预期收益约束: r'*(w-benchmark) >= a, 转换成线性不等式约束"""
    class __QS_ArgClass__(Constraint.__QS_ArgClass__):
        DownLimit: float = Field(default=0.0, title="限制下限", frozen=True)
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)
    
    def __init__(self, mask: NDArray[np.bool], expected_return: NDArray[np.float64], bmk: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化预期收益约束条件对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            expected_return: 预期收益, array(shape=(n,)), 缺失值会被填充为 0
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)
        expected_return = np.where(pd.notnull(expected_return), expected_return, 0)
        
        self._Mask = mask
        self._ExpectedReturn = expected_return
        self._Bmk = bmk

    def genConstraint(self) -> List[dict]:
        if self._QSArgs.Benchmark:
            aAdj = - self._QSArgs.DownLimit - np.dot(self._ExpectedReturn, self._Bmk)
            return [{"type": "LinearIn", "A": -self._ExpectedReturn[self._Mask], "b": np.array([[aAdj]])}]
        else:
            return [{"type": "LinearIn", "A": -self._ExpectedReturn[self._Mask], "b": np.array([[-self._QSArgs.DownLimit]])}]


class NonZeroNumConstraint(Constraint):
    """非零数目约束: sum((w-benchmark!=0)<=N, 转换成非零数目约束"""

    class __QS_ArgClass__(Constraint.__QS_ArgClass__):
        UpLimit: int = Field(default=150, title="限制上限", frozen=True)
        Benchmark: bool = Field(default=False, title="相对基准", frozen=True)
    
    def __init__(self, mask: NDArray[np.bool], bmk: Optional[NDArray[np.float64]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化均值方差优化目标对象

        Args:
            mask: array(shape=(n,)), 其中 n 为证券数量, True 表示组合可以选择的目标证券, False 表示组合中不可包含的证券
            bmk: 基准投资组合, array(shape=(n,)), 缺失值会被填充为 0
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        super().__init__(args, config_file, **kwargs)
        if self._QSArgs.Benchmark and (bmk is None): raise __QS_Error__("优化目标需要基准投资组合，但入参 bmk 为 None!")
        if bmk is None: bmk = np.zeros(mask.shape)
        else: bmk = np.where(pd.notnull(bmk), bmk, 0)
        
        self._Mask = mask
        self._Bmk = bmk
    
    def genConstraint(self) -> List[dict]:
        if self._QSArgs.Benchmark:
            N = (int(self._QSArgs.UpLimit - np.sum(self._Bmk[~self._Mask] != 0)) if not np.isinf(self._QSArgs.UpLimit) else np.inf)
            return [{"type": "NonZeroNum", "N": N, "b": self._Bmk[~self._Mask]}]
        else:
            return [{"type": "NonZeroNum", "N": (int(self._QSArgs.UpLimit) if not np.isinf(self._QSArgs.UpLimit) else np.inf), "b": np.zeros((np.sum(self._Mask), ))}]


class PortfolioConstructor(__QS_Object__):
    """投资组合构造器"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        OptimOption: dict = Field(default={}, title="优化选项", frozen=True)

    def __init__(self, mask:NDArray[np.bool], objective: OptimizationObjective, constraints:List[Constraint]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._Mask = mask
        self._Objective = objective
        self._Constrants = constraints
    
    # 求解优化问题, 返回: (Series(权重, index=[ID]) 或 None, 其他信息: {})
    def solve(self) -> Tuple[Optional[NDArray[np.float64]], dict]:
        Objective = self._Objective.genObjective()
        MathConstraints = []
        DropedConstraintInds = {-1: []}
        DropedConstraints = {-1: []}
        iStartInd = -1
        for i, iConstraint in enumerate(self._Constrants):
            iMathConstraints = iConstraint.genConstraint()
            MathConstraints.extend(iMathConstraints)
            iEndInd = iStartInd + len(iMathConstraints)
            iPriority = iConstraint._QSArgs.DropPriority
            if (iEndInd - iStartInd != 0) and (iPriority > -1):
                DropedConstraintInds[iPriority] = DropedConstraintInds.get(iPriority, []) + [i for i in range(iStartInd+1, iEndInd+1)]
                DropedConstraints[iPriority] = DropedConstraints.get(iPriority,[]) + [str(i)+'-'+iConstraint.Type]
            iStartInd = iEndInd
        ResultInfo = {}
        ReleasedConstraint = []
        Priority = sorted(DropedConstraintInds)
        while (ResultInfo.get("Status", 0) != 1) and (Priority != []):
            iPriority = Priority.pop(0)
            for j in DropedConstraintInds[iPriority]: MathConstraints[j] = None
            PreparedConstraints = self._prepareConstrants(MathConstraints)
            PreparedOption = self._genOption()
            TargetWeight, ResultInfo = self._solve(Objective, PreparedConstraints, PreparedOption)
            ReleasedConstraint += DropedConstraints[iPriority]
        ResultInfo['ReleasedConstraint'] = ReleasedConstraint
        if TargetWeight is not None:
            Rslt = np.full(shape=self._Mask.shape, fill_value=np.nan, dtype=float)
            Rslt[self._Mask] = TargetWeight
            return (Rslt, ResultInfo)
        else: return (None, ResultInfo)

    # 整理约束条件
    def _prepareConstrants(self, contraints):
        nVar = np.sum(self._Mask)
        PreparedConstraints = {}
        for iConstraint in contraints:
            if iConstraint is None: continue
            elif iConstraint['type'] == "Box":
                PreparedConstraints["Box"] = PreparedConstraints.get("Box", {"lb": np.zeros((nVar,)) - np.inf, "ub": np.zeros((nVar,)) + np.inf, "type": "Box"})
                PreparedConstraints["Box"]["lb"] = np.maximum(PreparedConstraints["Box"]["lb"], iConstraint["lb"])
                PreparedConstraints["Box"]["ub"] = np.minimum(PreparedConstraints["Box"]["ub"], iConstraint["ub"])
            elif iConstraint['type'] == "LinearIn":
                PreparedConstraints["LinearIn"] = PreparedConstraints.get("LinearIn", {"A": np.zeros((0, nVar)), "b": np.zeros((0, 1)), "type": "LinearIn"})
                PreparedConstraints["LinearIn"]["A"] = np.vstack((PreparedConstraints["LinearIn"]["A"],iConstraint["A"]))
                PreparedConstraints["LinearIn"]["b"] = np.vstack((PreparedConstraints["LinearIn"]["b"],iConstraint["b"]))
            elif iConstraint["type"] == "LinearEq":
                PreparedConstraints["LinearEq"] = PreparedConstraints.get("LinearEq", {"Aeq":np.zeros((0, nVar)), "beq":np.zeros((0, 1)), "type": "LinearEq"})
                PreparedConstraints["LinearEq"]["Aeq"] = np.vstack((PreparedConstraints["LinearEq"]["Aeq"], iConstraint["Aeq"]))
                PreparedConstraints["LinearEq"]["beq"] = np.vstack((PreparedConstraints["LinearEq"]["beq"], iConstraint["beq"]))
            else:
                PreparedConstraints[iConstraint["type"]] = PreparedConstraints.get(iConstraint["type"], [])
                PreparedConstraints[iConstraint["type"]].append(iConstraint)
        return PreparedConstraints
    
    # 整理选项参数
    def _genOption(self):
        return self._QSArgs.OptimOption
    
    # 求解一次优化问题, 返回: (array(nvar) 或 None, 其他信息: {})
    def _solve(self, prepared_objective, prepared_constraints, prepared_option):
        return (None, {})
