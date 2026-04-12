# -*- coding: utf-8 -*-
"""配置策略"""
import datetime as dt
from typing import Optional, Literal, List, Union

import numpy as np
import pandas as pd

from QuantStudio.Core import __QS_Error__
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.FactorOperation import PanelOperation, PanelOperator, SectionOperator, SectionOperation
from QuantStudio.Tools.StrategyTestFun import backtestPortfolioStrategy, backtestPortfolioStrategy_pd
from QuantStudio.Tools.DataPreprocessingFun import numpy_ffill


class CalcMaskPortfolio(SectionOperator):
    """基于筛选条件构造投资组合的计算算子"""

    def __init__(self, descriptor_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化基于筛选条件构造投资组合的计算算子

        Args:
            descriptor_ids: 依赖因子的截面 ID 序列
            args: 参数集
            config_file: 配置文件地址
        """
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcMaskPortfolio"} | args | {"DTMode": "多时点", "DataType": "double"}
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Mask, x = pd.DataFrame(x[0]==1, index=idt, columns=SectionIDs), x[1:]
        if f._QSArgs.ModelArgs["weight"]:
            Weight, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
        else:
            Weight = pd.DataFrame(1, index=idt, columns=SectionIDs)
        if f._QSArgs.ModelArgs["cat_data"]:
            CatData, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
            CatData = CatData.where(CatData.notnull(), "None")
            if f._QSArgs.ModelArgs["cat_weight"]:
                CatWeight = pd.DataFrame(x[-1], index=idt, columns=SectionIDs)
            else:
                CatWeight = pd.DataFrame(1, index=idt, columns=SectionIDs)
        if f._QSArgs.CalcDTRuler:
            RebalanceDTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Mask = Mask.reindex(index=RebalanceDTs).fillna(False).astype(bool)
            Weight = Weight.reindex(index=RebalanceDTs)
            if f._QSArgs.ModelArgs["cat_data"]:
                CatData = CatData.reindex(index=RebalanceDTs)
                CatWeight = CatWeight.reindex(index=RebalanceDTs)
        if not f._QSArgs.ModelArgs["cat_data"]:
            Porftolio = Weight.where(Mask, np.nan)
            Porftolio = (Porftolio.T / Porftolio.sum(axis=1)).T
            return Porftolio.reindex(index=idt, columns=iid).values
        else:
            #Rslt = pd.DataFrame({"mask": Mask.stack(), "weight": Weight.stack(), "cat_data": CatData.stack(), "cat_weight": CatWeight.stack()}).reset_index()
            Rslt = Panel({"mask": Mask, "weight": Weight, "cat_data": CatData, "cat_weight": CatWeight}).to_frame().reset_index()
            Rslt.columns = ["dt", "id"] + Rslt.columns[2:].tolist()
            if not Rslt["mask"].any(): return np.full(shape=(len(idt), len(iid)), fill_value=np.nan, dtype=float)
            Tmp = Rslt.groupby(["dt", "cat_data"])[["cat_weight"]].sum().reset_index()
            if not f._QSArgs.ModelArgs["cat_weight"]: Tmp["cat_weight"] = 1
            Tmp = pd.merge(Tmp, Tmp.groupby(["dt"])["cat_weight"].sum().to_frame("total_cat_weight"), how="left", left_on=["dt"], right_index=True)
            Tmp["cat_weight"] = Tmp["cat_weight"] / Tmp["total_cat_weight"]
            Rslt = Rslt[Rslt["mask"]]
            Rslt = pd.merge(Rslt, Tmp.loc[:, ["dt", "cat_data", "cat_weight"]], how="left", left_on=["dt", "cat_data"], right_on=["dt", "cat_data"], suffixes=("", "_total"))
            Rslt = pd.merge(Rslt, Rslt.groupby(["dt", "cat_data"])[["weight"]].sum(), how="left", left_on=["dt", "cat_data"], right_index=True, suffixes=("", "_total"))
            Rslt["weight"] = Rslt["weight"] / Rslt["weight_total"] * Rslt["cat_weight_total"]
            return Rslt.set_index(["dt", "id"])["weight"].unstack().reindex(index=idt, columns=iid).values
    
    def __call__(self, mask:Factor, weight:Optional[Factor]=None, cat_data:Optional[Factor]=None, cat_weight:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> SectionOperation:
        factor_args = factor_args.copy()
        Factors = [mask]
        if weight is not None: Factors.append(weight)
        if cat_data is not None:
            Factors.append(cat_data)
            if cat_weight is not None:
                Factors.append(cat_weight)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"weight": (weight is not None), "cat_data": (cat_data is not None), "cat_weight": (cat_weight is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


class CalcPortfolioNV(PanelOperator):
    """投资组合净值计算算子, 基于向量化回测方式"""

    def __init__(self, descriptor_ids:List[str], start_dt:Optional[dt.datetime]=None, calc_type:Literal["numpy", "pandas"]="numpy", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化投资组合净值计算算子

        Args:
            descriptor_ids: 依赖因子的截面 ID 序列
            start_dt: 净值开始日, 如果为 None, 表示从计算的第一个时点开始
            args: 参数集
            config_file: 配置文件地址
        """
        Arity = args.get("Arity", None) or 4
        Args = {"Name": "calcPortfolioNV"} | args | {"DTMode": "多时点", "DataType": "double", "iInitFactor": 0}
        Args["ModelArgs"] = {"calc_type": calc_type} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [None] + [descriptor_ids] * (Arity - 1)
        Args["StartDT"] = [start_dt] * Arity
        Args["LookBack"] = [1] + [0] * (Arity - 1)
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        PortfolioList, Price, FeeRate = x[1:-2], x[-2], x[-1]
        Price = numpy_ffill(Price, axis=0, limit=None)
        NV = np.ones(shape=(Price.shape[0], len(PortfolioList)))
        for i, iPortfolio in enumerate(PortfolioList):
            if args["calc_type"] == "numpy":
                NV[:, i], _ = backtestPortfolioStrategy(portfolio=iPortfolio, price=Price, fee=FeeRate, ffill_price=False)
            else:
                NV[:, i] = backtestPortfolioStrategy_pd(portfolio=pd.DataFrame(iPortfolio, index=idt[1:]).dropna(how="all", axis=0), price=pd.DataFrame(Price, index=idt[1:])).values
        return NV * x[0][0]

    def __call__(self, *portfolio:Factor, price:Factor, init_nv:Union[float, Factor]=1, fee_rate:Union[float, Factor]=0, portfolio_name_list:Optional[List[str]]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        """将算子作用在若干个投资组合因子对象上以产生净值因子

        Args:
            portfolio: 待计算净值的投资组合因子, 因子值是每个时点投资于某个证券的资金权重，如果某个时点的因子值全部为 NaN 表示改时点没有信号，不进行调仓
            price: 证券价格或者净值因子
            init_nv: 初始组合净值因子, 用于指定净值因子的初始值
            fee_rate: 交易费率因子
            portfolio_name_list: 投资组合的名称列表, None 表示由系统自动生成, 非 None 时将作为净值因子的截面 ID 序列，所以不能有重复
            factor_args: 创建净值因子时传递个它的参数集
            kwargs: 创建净值因子时传递给它的其他入参

        Returns:
            投资组合净值因子
        """
        factor_args = factor_args.copy()
        if not portfolio: raise __QS_Error__("投资组合因子不能为空!")
        if portfolio_name_list is not None:
            if factor_args.get("SectionIDs", None) is not None:
                self.Logger.warning(f"CalcPortfolioNV.__call__: 同时指定了投资组合名称列表 portfolio_name_list({portfolio_name_list})以及因子截面ID参数 SectionIDs({factor_args['SectionIDs']})，将使用后者作为因子的截面ID，忽略 portfolio_name_list")
                portfolio_name_list = factor_args["SectionIDs"]
        elif factor_args.get("SectionIDs", None) is not None:
            portfolio_name_list = factor_args["SectionIDs"]
        else:
            portfolio_name_list = [iFactor.Name for iFactor in portfolio]
            if len(set(portfolio_name_list)) != len(portfolio):
                PosNum = int(np.log10(max(1, len(portfolio) - 1))) + 1
                portfolio_name_list = [f"P{str(i).zfill(PosNum)}" for i in range(len(portfolio))]
                self.Logger.info(f"投资组合因子的名称中有重复, 使用系统自动生成的投资组合名称列表: {portfolio_name_list}")
        if len(set(portfolio_name_list)) != len(portfolio):
            raise __QS_Error__(f"投资组合的名称列表 : {portfolio_name_list} 长度不等于投资组合因子列表 portfolio 的长度或者有重复!")
        else:
            SortedIdx = np.argsort(portfolio_name_list)
            if not np.all(SortedIdx == np.arange(len(portfolio_name_list))):
                self.Logger.warning(f"CalcPortfolioNV.__call__: 投资组合的名称列表({portfolio_name_list})不是升序排列，将按照升序重新排列投资组合")
                portfolio, SortedPortfolioNameList = [portfolio[i] for i in SortedIdx], [portfolio_name_list[i] for i in SortedIdx]
            else:
                SortedPortfolioNameList = portfolio_name_list
        factor_args = factor_args.copy()
        factor_args["SectionIDs"] = SortedPortfolioNameList
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"portfolio_name_list": portfolio_name_list}
        kwargs["operator_kwargs"] =  {"descriptor_ids": self._QSArgs.DescriptorSection[1], "start_dt": self._QSArgs.StartDT[0]} | kwargs.get("operator_kwargs", {})
        return super().__call__(init_nv, *portfolio, price, fee_rate, factor_args=factor_args, **kwargs)


class CalcPortfolioReturn(PanelOperator):
    """投资组合收益率计算算子"""
    def __init__(self, descriptor_ids:List[str], lookback:int=31, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化投资组合收益率计算算子

        Args:
            descriptor_ids: 依赖因子的截面 ID 序列
            lookback: 在时间标尺上的回溯期数, 即回溯多久的数据来完成计算
            args: 参数集
            config_file: 配置文件地址
        """
        Arity = args.get("Arity", None) or 2
        Args = {"Name": "calcPortfolioReturn"} | args | {"DTMode": "多时点", "DataType": "double"}
        Args["ModelArgs"] = {} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Price, PortfolioList = pd.DataFrame(x[0], index=idt, columns=SectionIDs), [pd.DataFrame(ix, index=idt, columns=SectionIDs) for ix in x[1:]]
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Price = Price.reindex(index=DTs)
            PortfolioList = [Portfolio.reindex(index=DTs).shift(1) for Portfolio in PortfolioList]
        else:
            DTs = Price.index
            PortfolioList = [Portfolio.shift(1) for Portfolio in PortfolioList]
        Return = Price.pct_change()
        PortfolioReturn = pd.DataFrame(np.nan, index=DTs, columns=iid)
        for i, iPortfolioName in enumerate(iid):
            PortfolioReturn[iPortfolioName] = (PortfolioList[i] * Return).sum(axis=1)
        return PortfolioReturn.reindex(index=idt).values[self._QSArgs.LookBack[0]:]

    def __call__(self, *portfolio:Factor, price:Factor, portfolio_name_list:Optional[List[str]]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        """将算子作用在若干个投资组合因子对象上以产生收益率因子

        Args:
            portfolio: 待计算收益率的投资组合因子, 因子值是每个时点投资于某个证券的资金权重，如果某个时点的因子值全部为 NaN 表示改时点没有信号，不进行调仓
            price: 证券价格或者净值因子
            portfolio_name_list: 投资组合的名称列表, None 表示由系统自动生成, 非 None 时将作为收益率因子的截面 ID 序列，所以不能有重复
            factor_args: 创建收益率因子时传递个它的参数集
            kwargs: 创建收益率因子时传递给它的其他入参

        Returns:
            投资组合收益率因子
        """
        factor_args = factor_args.copy()
        if not portfolio: raise __QS_Error__("投资组合因子不能为空!")
        if portfolio_name_list is not None:
            if factor_args.get("SectionIDs", None) is not None:
                self.Logger.warning(f"CalcPortfolioNV.__call__: 同时指定了投资组合名称列表 portfolio_name_list({portfolio_name_list})以及因子截面ID参数 SectionIDs({factor_args['SectionIDs']}), 将使用后者作为因子的截面ID, 忽略 portfolio_name_list")
                portfolio_name_list = factor_args["SectionIDs"]
        elif factor_args.get("SectionIDs", None) is not None:
            portfolio_name_list = factor_args["SectionIDs"]
        else:
            portfolio_name_list = [iFactor.Name for iFactor in portfolio]
            if len(set(portfolio_name_list)) != len(portfolio):
                PosNum = int(np.log10(max(1, len(portfolio) - 1))) + 1
                portfolio_name_list = [f"P{str(i).zfill(PosNum)}" for i in range(len(portfolio))]
                self.Logger.info(f"投资组合因子的名称中有重复, 使用系统自动生成的投资组合名称列表: {portfolio_name_list}")
        if len(set(portfolio_name_list)) != len(portfolio):
            raise __QS_Error__(f"投资组合的名称列表 : {portfolio_name_list} 长度不等于投资组合因子列表 portfolio 的长度或者有重复!")
        else:
            SortedIdx = np.argsort(portfolio_name_list)
            if not np.all(SortedIdx == np.arange(len(portfolio_name_list))):
                self.Logger.warning(f"CalcPortfolioNV.__call__: 投资组合的名称列表({portfolio_name_list})不是升序排列，将按照升序重新排列投资组合")
                portfolio, SortedPortfolioNameList = [portfolio[i] for i in SortedIdx], [portfolio_name_list[i] for i in SortedIdx]
            else:
                SortedPortfolioNameList = portfolio_name_list
        factor_args["SectionIDs"] = SortedPortfolioNameList
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"portfolio_name_list": portfolio_name_list}
        kwargs["operator_kwargs"] =  {"descriptor_ids": self._QSArgs.DescriptorSection[0], "lookback": self._QSArgs.LookBack[0]} | kwargs.get("operator_kwargs", {})
        return super().__call__(price, *portfolio, factor_args=factor_args, **kwargs)
