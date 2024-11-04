# coding=utf-8
import datetime as dt
from typing import Optional, Dict

import numpy as np
import pandas as pd
from traits.api import Int, Enum, Instance

from QuantStudio import __QS_Error__, QSArgs
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PanelOperator, SectionOperator
import QuantStudio.FactorDataBase.FactorOperators as fo
from QuantStudio.Tools.StrategyTestFun import testPortfolioStrategy_pd
from QuantStudio.Tools import DataPreprocessingFun

# IC
class _ICModelArgs(QSArgs):
    FactorOrder = Enum("降序", "升序", arg_type="SingleOption", label="排序方向", order=0)
    LookBack = Int(1, arg_type="Integer", label="回溯期数", order=1)
    CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=2, option_range=["spearman", "pearson", "kendall"])

class IC(PanelOperator):
    """IC"""
    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        ModelArgs = Instance(_ICModelArgs, arg_type="ArgObject", label="参数", order=2, mutable=False)
    
        def __QS_initArgs__(self, args={}):
            self.ModelArgs = _ICModelArgs(owner=self._Owner)
            return super().__QS_initArgs__(args=args)
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "calcIC", "入参数": 2, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "输出形式": "全截面", "回溯期数": [0, 0], "回溯模式": ["扩张窗口", "扩张窗口"]}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (f._QSArgs.DescriptorSection[0] if f._QSArgs.DescriptorSection[0] else iid)
        Price, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Price = Price.reindex(columns=DTs)
        else:
            DTs = Price.columns
        Return = Price.T.pct_change().T
        if f.UserData["mask"]: 
            Mask, x = pd.DataFrame(x[0].T==1, columns=idt, index=SectionIDs), x[1:]
            Mask = (Mask.reindex(columns=DTs).fillna(False) & Price.notnull())
        else:
            Mask = Price.notnull()
        if f.UserData["cat_data"]: 
            CatData, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
            CatData = CatData.reindex(columns=DTs)
        else:
            CatData = None
        if f.UserData["weight"]: 
            Weight, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
            Weight = Weight.reindex(columns=DTs)
        else:
            Weight = pd.DataFrame(1, columns=DTs, index=SectionIDs)
        if CatData is not None:# 进行收益率的类别调整
            Price = Price.where(CatData.notnull(), np.nan)
            AllCates = np.unique(CatData)
            for iCate in AllCates:
                iMask = ((CatData==iCate) & Mask)
                iWeight = Weight.where(iMask, np.nan)
                iReturn = (Return * iWeight.shift(1, axis=1)).sum(axis=0) / iWeight.shift(1, axis=1).sum(axis=0)
                Return = Return.where(~iMask.shift(1, axis=1).fillna(False), Return - iReturn)
        if args["排序方向"]=="升序": Return = - Return
        IC = pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        Mask = Mask.shift(args["回溯期数"], axis=1).fillna(False)
        for i, iFactorName in enumerate(iid):
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iFactorData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs)
            iFactorData = iFactorData.reindex(columns=DTs).shift(args["回溯期数"], axis=1)
            iMask = (Mask & iFactorData.notnull())
            IC[iFactorName] = Return.where(iMask, np.nan).corrwith(iFactorData, method=args["相关性算法"])
        return IC.reindex(index=idt).values[f._QSArgs.LookBack[0]:]
        
    def __call__(self, price:Factor, *factors, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, weight:Optional[Factor]=None, descriptor_ids=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [price]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        if weight is not None: Factors.append(weight)
        if factors: Factors += factors
        else: raise __QS_Error__(f"算子 {self.__class__}: 必须至少指定一个因子!")
        Args = dict(self._QSArgs["参数"])
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if "回溯期数" not in factor_args:
            factor_args["回溯期数"] = [self._QSArgs["回溯期数"][0]] * len(Factors)
        if "回溯模式" not in factor_args:
            factor_args["回溯模式"] = ["扩张窗口"] * len(factor_args["回溯期数"])
        if "起始时点" not in factor_args:
            factor_args["起始时点"] = [None] * len(factor_args["回溯期数"])
        if "截面ID" not in factor_args:
            factor_args["截面ID"] = [iFactor.Name for iFactor in factors]
        if len(set(factor_args["截面ID"]))<len(factor_args["截面ID"]):
            raise __QS_Error__(f"算子 {self.__class__}: 因子名有重复: {factor_args['截面ID']}")
        if descriptor_ids is not None: factor_args["描述子截面"] = [descriptor_ids] * len(Factors)
        factor_args["参数"] = Args
        f = super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        f.UserData = {
            "mask": (mask is not None),
            "cat_data": (cat_data is not None),
            "weight": (weight is not None)
        }
        return f


# 筛选投资组合
class MaskPortfolio(SectionOperator):
    """筛选投资组合"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "calcMaskPortfolio", "入参数": 1, "最大入参数": 2, "运算时点": "多时点", "输出形式": "全截面", "数据类型":"double"}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)

    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (f._QSArgs.DescriptorSection[0] if f._QSArgs.DescriptorSection[0] else iid)
        Mask = pd.DataFrame(x[0]==1, index=idt, columns=SectionIDs)
        if len(x)==2:
            Weight = pd.DataFrame(x[1], index=idt, columns=SectionIDs)
        else:
            Weight = pd.DataFrame(1, index=idt, columns=SectionIDs)
        if f._QSArgs.CalcDTRuler:
            RebalanceDTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Mask = Mask.reindex(index=RebalanceDTs).fillna(False)
            Weight = Weight.reindex(index=RebalanceDTs)
        Porftolio = Weight.where(Mask, np.nan)
        Porftolio = (Porftolio.T / Porftolio.sum(axis=1)).T
        return Porftolio.reindex(index=idt, columns=iid).values
    
    def __call__(self, mask:Factor, weight:Optional[Factor]=None, descriptor_ids=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [mask]
        if weight is not None: Factors.append(weight)
        Args = dict(self._QSArgs["参数"])
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if descriptor_ids is not None: factor_args["描述子截面"] = [descriptor_ids] * len(Factors)
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


# 分位数组合
def makeQuantilePortfolio(factor:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, weight:Optional[Factor]=None, descriptor_ids=None, rebalance_dts=None, ascending=False, group_num=10, **kwargs):
    rank = fo.SectionRank(ascending=ascending, uniformization=True)
    Rank = rank(factor, mask=mask, cat_data=cat_data, factor_args={"计算时点标尺": rebalance_dts})
    if weight is None: weight = 1
    calcMaskPortfolio = MaskPortfolio()
    Portfolio = []
    for i in range(group_num):
        iLeft, iRight = i / group_num, (i+1) / group_num
        iPortfolio = calcMaskPortfolio((Rank >= iLeft) & (Rank < iRight), weight=weight, descriptor_ids=descriptor_ids)
        Portfolio.append(iPortfolio)
    return Portfolio


# 投资组合净值
class _PortfolioNVModelArgs(QSArgs):
    PriceMiss = Enum("沿用前值", "填充为0", arg_type="SingleOption", label="价格缺失", order=0, option_range=["沿用前值", "填充为0"])
    
class PortfolioNV(PanelOperator):
    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        ModelArgs = Instance(_PortfolioNVModelArgs, arg_type="ArgObject", label="参数", order=2, mutable=False)
        
        def __QS_initArgs__(self, args={}):
            self.ModelArgs = _PortfolioNVModelArgs(owner=self._Owner)
            return super().__QS_initArgs__(args=args)

    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "calcPortfolioNV", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "回溯期数": [0, 0], "输出形式": "全截面", "数据类型":"double"}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)

    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (f._QSArgs.DescriptorSection[0] if f._QSArgs.DescriptorSection[0] else iid)
        Portfolio = pd.DataFrame(x[0], index=idt, columns=SectionIDs)
        Price = pd.DataFrame(x[1], index=idt, columns=SectionIDs)
        if args["价格缺失"]=="沿用前值": Price = Price.fillna(method="pad")
        NV = testPortfolioStrategy_pd(Portfolio, Price)
        return np.reshape(NV.values, (-1, 1)).repeat(len(iid), axis=1)

    def __call__(self, portfolio:Factor, price:Factor, descriptor_ids=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = dict(self._QSArgs["参数"])
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if descriptor_ids is not None: factor_args["描述子截面"] = [descriptor_ids] * 2
        return super().__call__(portfolio, price, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


# 截面相关性
class _SectionCorrelationModelArgs(QSArgs):
    FactorOrder = Enum("降序", "升序", arg_type="SingleOption", label="排序方向", order=0)
    CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=1, option_range=["spearman", "pearson", "kendall"])

class SectionCorrelation(SectionOperator):
    """截面相关性"""
    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs = Instance(_SectionCorrelationModelArgs, arg_type="ArgObject", label="参数", order=2, mutable=False)
    
        def __QS_initArgs__(self, args={}):
            self.ModelArgs = _SectionCorrelationModelArgs(owner=self._Owner)
            return super().__QS_initArgs__(args=args)
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "calcSectionCorrelation", "入参数": 2, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "输出形式": "全截面"}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)

    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (f._QSArgs.DescriptorSection[0] if f._QSArgs.DescriptorSection[0] else iid)
        TargetFactor, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            TargetFactor = TargetFactor.reindex(columns=DTs)
        else:
            DTs = Price.columns
        if f.UserData["mask"]: 
            Mask, x = pd.DataFrame(x[0].T==1, columns=idt, index=SectionIDs), x[1:]
            Mask = (Mask.reindex(columns=DTs).fillna(False) & TargetFactor.notnull())
        else:
            Mask = TargetFactor.notnull()
        if args["排序方向"]=="升序": TargetFactor = - TargetFactor
        Corr = pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        for i, iFactorName in enumerate(iid):
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iFactorData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs)
            iFactorData = iFactorData.reindex(columns=DTs)
            iMask = (Mask & iFactorData.notnull())
            Corr[iFactorName] = TargetFactor.where(iMask, np.nan).corrwith(iFactorData, method=args["相关性算法"])
        return Corr.reindex(index=idt).values
    
    def __call__(self, f:Factor, *factors, mask:Optional[Factor]=None, descriptor_ids=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if factors: Factors += factors
        else: raise __QS_Error__(f"算子 {self.__class__}: 必须至少指定一个因子!")
        Args = dict(self._QSArgs["参数"])
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if "截面ID" not in factor_args: factor_args["截面ID"] = [iFactor.Name for iFactor in factors]
        if len(set(factor_args["截面ID"]))<len(factor_args["截面ID"]):
            raise __QS_Error__(f"算子 {self.__class__}: 因子名有重复: {factor_args['截面ID']}")
        if descriptor_ids is not None: factor_args["描述子截面"] = [descriptor_ids] * len(Factors)
        factor_args["参数"] = Args
        f = super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        f.UserData = {
            "mask": (mask is not None)
        }
        return f



# 截面分位数标准化
class QuantileStandardization(SectionOperator):
    def __init__(self, ascending:bool=True, uniformization:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "calcQuantileStandardization", "入参数": 1, "最大入参数": 3, "数据类型": "double", "运算时点": "多时点", "输出形式": "全截面", "参数": {"uniformization": uniformization, "ascending": ascending, "mask": False, "cat_data": False}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        FactorData, args = x[0], args.copy()
        Mask = (x[1].astype(bool) if args.pop("mask") else [None] * FactorData.shape[0])
        CatData = (x[-1] if args.pop("cat_data") else [None] * FactorData.shape[0])
        Rslt = np.full_like(FactorData, fill_value=np.nan)
        for i in range(FactorData.shape[0]):
            Rslt[i] = DataPreprocessingFun.standardizeQuantile(FactorData[i], mask=Mask[i], cat_data=CatData[i], perturbation=False, **args)
        return Rslt
    
    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, *, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        Args["mask"] = (mask is not None)
        Args["cat_data"] = (cat_data is not None)
        factor_args["参数"] = Args
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


# 正交化
class Orthogonalization(SectionOperator):
    def __init__(self, constant=False, drop_dummy_na=False, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "orthogonalize", "入参数": 1, "最大入参数": 3, "数据类型": "double", "运算时点": "单时点", "输出形式": "全截面", "参数": {"constant": constant, "drop_dummy_na": drop_dummy_na}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Y = x[0]
        if f.UserData["mask"]: 
            Mask, x = (x[0]==1), x[1:]
        else:
            Mask = None
        if f.UserData["dummy_data"]: 
            DummyData, x = x[0], x[1:]
        else:
            DummyData = None
        X = np.array(x, dtype=float).T
        Rslt = DataPreprocessingFun.orthogonalize(Y, X=X, mask=Mask, dummy_data=DummyData, **args)
        return Rslt
        
    def __call__(self, f:Factor, *exog, mask:Optional[Factor]=None, dummy_data:Optional[Factor]=None, descriptor_ids=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if dummy_data is not None: Factors.append(dummy_data)
        if exog: Factors += exog
        else: raise __QS_Error__(f"算子 {self.__class__}: 必须至少指定一个回归因子!")
        Args = dict(self._QSArgs["参数"])
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        factor_args["描述子截面"] = [descriptor_ids] * len(Factors)
        f = super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        f.UserData = {
            "mask": (mask is not None),
            "dummy_data": (dummy_data is not None)
        }
        return f


if __name__=="__main__":
    from QuantStudio.FactorDataBase.FactorDB import DataFactor
    from QuantStudio.Tools.DateTimeFun import getNaturalDay, getMonthLastDateTime
    
    np.random.seed(0)
    IDs = [f"{str(i).zfill(6)}.SZ" for i in range(1, 21)]
    DTRuler = getNaturalDay(dt.datetime(2019, 1, 1), dt.datetime(2020, 12, 31))
    DTs = getNaturalDay(dt.datetime(2020, 1, 1), dt.datetime(2020, 12, 31))
    MonthDTRuler = getMonthLastDateTime(DTRuler)
    MonthDTs = getMonthLastDateTime(DTs)
    Mask = DataFactor(name="Mask", data=pd.DataFrame(np.random.randint(0, 2, size=(len(DTRuler), len(IDs))).astype(bool), index=DTRuler, columns=IDs))
    Industry = DataFactor(name="Industry", data=pd.Series(np.random.choice(["Fin", "TMT", "Ind"], size=(len(IDs),)), index=IDs))    
    #Rtn = DataFactor(name="Return", data=pd.DataFrame(np.random.randn(len(DTRuler), len(IDs)), index=DTRuler, columns=IDs))
    Price = DataFactor(name="Price", data=pd.DataFrame(np.random.rand(len(DTRuler), len(IDs)) * 10, index=DTRuler, columns=IDs))
    Factor1 = DataFactor(name="Factor1", data=pd.DataFrame(np.random.randn(len(DTRuler), len(IDs)), index=DTRuler, columns=IDs))
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTRuler), len(IDs)), index=DTRuler, columns=IDs))
    
    calcIC = IC(sys_args={
        "参数": {
            "回溯期数": 1,
            "相关性算法": "spearman"
        }
    })
    FIC = calcIC(Price, Factor1, Factor2, cat_data=Industry, descriptor_ids=IDs, factor_name="IC", factor_args={"计算时点标尺": MonthDTRuler, "回溯期数": [32-1]*4})
    print(FIC.readData(ids=["Factor1", "Factor2"], dts=MonthDTs, dt_ruler=DTRuler))
    
    #print(Factor1.Name, Factor1.readData(ids=IDs, dts=MonthDTs), sep="\n", end="\n\n")
    #calcMaskPortfolio = MaskPortfolio()
    #FMP = calcMaskPortfolio(Factor1 > 0, descriptor_ids=IDs, factor_name="QP", factor_args={"计算时点标尺": MonthDTRuler})
    #Data = FMP.readData(ids=IDs, dts=MonthDTs)
    #print(Data)
    
    #calcCorr = SectionCorrelation(sys_args={
        #"参数": {
            #"相关性算法": "spearman"
        #}
    #})
    #FCorr = calcCorr(Price, Factor1, Factor2, descriptor_ids=IDs, factor_name="SectionCorrelation", factor_args={"计算时点标尺": MonthDTRuler})
    #print(FCorr.readData(ids=["Factor1", "Factor2"], dts=MonthDTs, dt_ruler=DTRuler))
    
    #orthogonalize = Orthogonalization(constant=True, drop_dummy_na=False)
    #FOth = orthogonalize(Factor1, Factor2, descriptor_ids=IDs, factor_name="Orthogonalization", factor_args={"计算时点标尺": MonthDTRuler})
    #print(FOth.readData(ids=IDs, dts=MonthDTs))
    
    print("===")
