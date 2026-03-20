# coding=utf-8
"""Brinson 绩效分析模型"""
import datetime as dt
from typing import List, Literal, Optional

import pandas as pd
import numpy as np
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import DTInitData, DTLocalContext
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData, FactorLocalContext
from QuantStudio.Factor.FactorOperation import PanelOperator, PanelOperation
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatPandasPercentage



class CalcBrinsonModel(PanelOperator):
    """Brinson 绩效分析算子
    前提条件:
    1. 投资组合的权重之和为 1, 与 1 的差值部分归为现金
    2. 两个计算时点之间没有调整策略持仓
    """

    def __init__(self, section_ids:List[str], lookback:int = 31, descriptor_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 Brinson 绩效分析算子

        Args:
            section_ids: 绩效分析因子的截面 ID 序列, 如果使用行业作为资产分类, 该截面应该为所有行业的列表
            lookback: 在时间标尺上的回溯期数, 即回溯多久的数据来完成计算
            descriptor_ids: 依赖因子的截面 ID 序列
        """
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcBrinsonModel"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DataType": "object"}
        Args["ModelArgs"] = {"section_ids": section_ids} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        Args["CompoundType"] = [("BMK", "double"), ("AAP", "double"), ("SSP", "double"), ("TP", "double"), ("AA", "double"), ("SS", "double"), ("IN", "double"), ("AAA", "double")]
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Portfolio, Price, CatData, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), pd.DataFrame(x[0], index=idt, columns=SectionIDs), pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]

        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Price = Price.reindex(columns=DTs)
        else:
            DTs = Price.columns
        Return = Price.T.pct_change().T
        if f._QSArgs.ModelArgs["mask"]: 
            Mask, x = pd.DataFrame(x[0].T==1, columns=idt, index=SectionIDs).reindex(columns=DTs).fillna(False).astype(bool), x[1:]
            Mask = (Mask & Price.notnull())
        else:
            Mask = Price.notnull()
        if f._QSArgs.ModelArgs["cat_data"]: 
            CatData, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs).reindex(columns=DTs), x[1:]
        else:
            CatData = None
        if f._QSArgs.ModelArgs["weight"]: 
            Weight, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs).reindex(columns=DTs), x[1:]
        else:
            Weight = pd.DataFrame(1, columns=DTs, index=SectionIDs)
        if CatData is not None:# 进行收益率的类别调整
            Return = Return.where(CatData.notnull(), np.nan)
            AllCates = CatData.values.flatten()
            AllCates = np.unique(AllCates[pd.notnull(AllCates)])
            for iCate in AllCates:
                iMask = ((CatData==iCate) & Mask)
                iWeight = Weight.where(iMask, np.nan).shift(1, axis=1).copy()
                iReturn = (Return * iWeight).sum(axis=0) / iWeight.sum(axis=0)
                Return = Return.where(~iMask.shift(1, axis=1).fillna(False).astype(bool), Return - iReturn)
        IC, Breadth = pd.DataFrame(index=DTs, columns=iid), pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        Mask = Mask.shift(args["period_lookback"], axis=1).fillna(False).astype(bool)
        for iFactorName in iid:
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iFactorData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs)
            iFactorData = iFactorData.reindex(columns=DTs).shift(args["period_lookback"], axis=1)
            iMask = (Mask & iFactorData.notnull())
            IC[iFactorName] = Return.where(iMask, np.nan).corrwith(iFactorData, method=args["corr_method"])
            Breadth[iFactorName] = iMask.sum(axis=0)
        Rslt = np.array([IC.reindex(index=idt).values[self._QSArgs.LookBack[0]:], Breadth.reindex(index=idt).values[self._QSArgs.LookBack[0]:]])
        return unstructured_to_structured(Rslt.swapaxes(0, -1), dtype=np.dtype([("IC", float), ("Breadth", float)])).T.astype("O")
    
    def __call__(self, p:Factor, price: Factor, cat_data: Factor, bmk:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        """将算子作用在若干个因子对象上以产生 Brinson 绩效分析因子

        Args:
            p: 待分析的投资组合因子
            price: 证券价格或者净值因子
            cat_data: 类别因子, 比如行业等
            bmk: 基准投资组合因子，如果为 None 表示没有基准，考察绝对收益
            factor_args: 创建 IC 因子时传递个它的参数集
            kwargs: 创建 IC 因子时传递给它的其他入参

        Returns:
            Brinson 绩效分析因子，该因子为复合因子，包含的子因子有:
            * BMK: 业绩基准组合收益
            * AAP: 主动资产配置组合收益
            * SSP: 主动个券选择组合收益
            * TP: 实际投资组合收益
            * AA: 资产配置收益(Return of Asset Allocation)
            * SS: 个券选择收益(Return of Stock Selection)
            * IN: 交互作用(Interaction)
            * AAA: 调整的资产配置收益
        """
        Factors = [p, price, cat_data]
        if bmk is not None: Factors.append(bmk)
        if "SectionIDs" not in factor_args:
            factor_args["SectionIDs"] = self._QSArgs.ModelArgs["section_ids"]
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"bmk": (bmk is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)





# 前提条件:
# 1. 投资组合的权重之和为 1, 与 1 的差值部分归为现金
# 2. 两个计算时点之间没有调整策略持仓
class BrinsonModel(BaseModule):
    """Brinson 绩效分析模型"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        #Portfolio = Enum(None, arg_type="SingleOption", label="策略组合", order=0)
        #BenchmarkPortfolio = Enum(None, arg_type="SingleOption", label="基准组合", order=1)
        #GroupFactor = Enum(None, arg_type="SingleOption", label="资产类别", order=2)
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
        CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="计算时点", order=4)
        def __QS_initArgs__(self, args={}):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("Portfolio", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="策略组合", order=0, option_range=DefaultNumFactorList))
            self.add_trait("BenchmarkPortfolio", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="基准组合", order=1, option_range=DefaultNumFactorList))
            self.add_trait("GroupFactor", Enum(*DefaultStrFactorList, arg_type="SingleOption", label="资产类别", order=2, option_range=DefaultStrFactorList))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3, option_range=DefaultNumFactorList))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            
    def __init__(self, factor_table, name="Brinson绩效分析模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        if self._QSArgs.CalcDTs: DTs = self._QSArgs.CalcDTs
        else: DTs = dts
        self._Output["策略组合资产权重"] = pd.DataFrame(0.0, index=DTs[1:], columns=["现金"])
        self._Output["基准组合资产权重"] = pd.DataFrame(0.0, index=DTs[1:], columns=["现金"])
        self._Output["策略组合资产收益"] = pd.DataFrame(0.0, index=DTs[1:], columns=["现金"])
        self._Output["基准组合资产收益"] = pd.DataFrame(0.0, index=DTs[1:], columns=["现金"])
        self._CurCalcInd = 0
        self._IDs = self._FactorTable.getID()
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        PreDT = None
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            if self._CurCalcInd>0: PreDT = self._QSArgs.CalcDTs[self._CurCalcInd - 1]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            if self._CurCalcInd>0: PreDT = self._Model.DateTimeSeries[self._CurCalcInd - 1]
        if PreDT is None: return 0
        Portfolio = self._FactorTable.readData(factor_names=[self._QSArgs.Portfolio, self._QSArgs.BenchmarkPortfolio], dts=[PreDT], ids=self._IDs).iloc[:, 0, :]
        BenchmarkPortfolio, Portfolio = Portfolio.iloc[:, 1], Portfolio.iloc[:, 0]
        Portfolio[pd.isnull(Portfolio)], BenchmarkPortfolio[pd.isnull(BenchmarkPortfolio)] = 0.0, 0.0
        Price = self._FactorTable.readData(factor_names=[self._QSArgs.PriceFactor], dts=[PreDT, idt], ids=self._IDs).iloc[0]
        Return = Price.iloc[1] / Price.iloc[0] - 1
        Return[pd.isnull(Return)] = 0.0
        GroupData = self._FactorTable.readData(factor_names=[self._QSArgs.GroupFactor], ids=self._IDs, dts=[PreDT]).iloc[0, 0, :]
        AllGroups = pd.unique(GroupData[pd.notnull(GroupData)].values).tolist()
        if GroupData.hasnans: AllGroups.append(None)
        for iGroup in AllGroups:
            if iGroup is None: iMask = pd.isnull(GroupData)
            else: iMask = (GroupData==iGroup)
            iGroup = str(iGroup)
            iPortfolio, iBenchmarkPortfolio = Portfolio[iMask], BenchmarkPortfolio[iMask]
            iGroupWeight, iBenchmarkGroupWeight = iPortfolio.sum(), iBenchmarkPortfolio.sum()
            self._Output["策略组合资产权重"].loc[idt, iGroup] = iGroupWeight
            self._Output["基准组合资产权重"].loc[idt, iGroup] = iBenchmarkGroupWeight
            self._Output["策略组合资产收益"].loc[idt, iGroup] = ((iPortfolio * Return[iMask]).sum() / iGroupWeight if iGroupWeight!=0 else 0.0)
            self._Output["基准组合资产收益"].loc[idt, iGroup] = ((iBenchmarkPortfolio * Return[iMask]).sum() / iBenchmarkGroupWeight if iBenchmarkGroupWeight!=0 else 0.0)
        self._Output["策略组合资产权重"].loc[idt, "现金"] = 1 - self._Output["策略组合资产权重"].loc[idt].iloc[1:].sum()
        self._Output["基准组合资产权重"].loc[idt, "现金"] = 1 - self._Output["基准组合资产权重"].loc[idt].iloc[1:].sum()
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        self._Output["策略组合资产权重"].where(pd.notnull(self._Output["策略组合资产权重"]), 0.0, inplace=True)
        self._Output["基准组合资产权重"].where(pd.notnull(self._Output["基准组合资产权重"]), 0.0, inplace=True)
        self._Output["策略组合资产收益"].where(pd.notnull(self._Output["策略组合资产收益"]), 0.0, inplace=True)
        self._Output["基准组合资产收益"].where(pd.notnull(self._Output["基准组合资产收益"]), 0.0, inplace=True)
        self._Output["策略组合收益"] = self._Output["策略组合资产权重"] * self._Output["策略组合资产收益"]
        self._Output["基准组合收益"] = self._Output["基准组合资产权重"] * self._Output["基准组合资产收益"]
        self._Output["主动资产配置组合收益"] = self._Output["策略组合资产权重"] * self._Output["基准组合资产收益"]
        self._Output["主动个券选择组合收益"] = self._Output["基准组合资产权重"] * self._Output["策略组合资产收益"]
        self._Output["主动资产配置超额收益"] = self._Output["主动资产配置组合收益"] - self._Output["基准组合收益"]
        self._Output["主动个券选择超额收益"] = self._Output["主动个券选择组合收益"] - self._Output["基准组合收益"]
        self._Output["交互作用超额收益"] = self._Output["策略组合收益"] - self._Output["主动个券选择组合收益"] - self._Output["主动资产配置组合收益"] + self._Output["基准组合收益"] 
        self._Output["总超额收益"] = self._Output["策略组合收益"] - self._Output["基准组合收益"]
        self._Output["主动资产配置组合收益(修正)"] = (self._Output["策略组合资产权重"] - self._Output["基准组合资产权重"]) * (self._Output["基准组合资产收益"].T - self._Output["基准组合收益"].sum(axis=1)).T
        self._Output["总计"] = pd.DataFrame(self._Output["策略组合资产权重"].sum(axis=1), columns=["策略组合资产权重"])
        self._Output["总计"]["基准组合资产权重"] = self._Output["基准组合资产权重"].sum(axis=1)
        self._Output["总计"]["策略组合收益"] = self._Output["策略组合收益"].sum(axis=1)
        self._Output["总计"]["基准组合收益"] = self._Output["基准组合收益"].sum(axis=1)
        self._Output["总计"]["主动资产配置组合收益"] = self._Output["主动资产配置组合收益"].sum(axis=1)
        self._Output["总计"]["主动资产配置组合收益(修正)"] = self._Output["主动资产配置组合收益(修正)"].sum(axis=1)
        self._Output["总计"]["主动个券选择组合收益"] = self._Output["主动个券选择组合收益"].sum(axis=1)
        self._Output["总计"]["主动资产配置超额收益"] = self._Output["主动资产配置超额收益"].sum(axis=1)
        self._Output["总计"]["主动个券选择超额收益"] = self._Output["主动个券选择超额收益"].sum(axis=1)
        self._Output["总计"]["交互作用超额收益"] = self._Output["交互作用超额收益"].sum(axis=1)
        self._Output["总计"]["总超额收益"] = self._Output["总超额收益"].sum(axis=1)
        self._Output["多期综合"] = pd.DataFrame(dtype=float)
        self._Output["多期综合"]["策略组合收益"] = (self._Output["策略组合收益"] + 1).prod(axis=0) - 1
        self._Output["多期综合"]["基准组合收益"] = (self._Output["基准组合收益"] + 1).prod(axis=0) - 1
        self._Output["多期综合"]["主动资产配置组合收益"] = (self._Output["主动资产配置组合收益"] + 1).prod() - 1
        self._Output["多期综合"]["主动个券选择组合收益"] = (self._Output["主动个券选择组合收益"] + 1).prod() - 1
        self._Output["多期综合"]["主动资产配置超额收益"] = self._Output["多期综合"]["主动资产配置组合收益"] - self._Output["多期综合"]["基准组合收益"]
        self._Output["多期综合"]["主动个券选择超额收益"] = self._Output["多期综合"]["主动个券选择组合收益"] - self._Output["多期综合"]["基准组合收益"]
        self._Output["多期综合"]["交互作用超额收益"] = self._Output["多期综合"]["策略组合收益"] - self._Output["多期综合"]["主动资产配置组合收益"] - self._Output["多期综合"]["主动个券选择组合收益"] + self._Output["多期综合"]["基准组合收益"]
        self._Output["多期综合"]["总超额收益"] = self._Output["多期综合"]["策略组合收益"] - self._Output["多期综合"]["基准组合收益"]
        self._Output["多期综合"].loc["总计"] = (self._Output["总计"] + 1).prod(axis=0) - 1
        k_t = (np.log(1+self._Output["总计"]["策略组合收益"]) - np.log(1+self._Output["总计"]["基准组合收益"])) / (self._Output["总计"]["策略组合收益"] - self._Output["总计"]["基准组合收益"])
        k_t[pd.isnull(k_t)] = 1.0
        if self._Output["多期综合"].loc["总计", "策略组合收益"]!=self._Output["多期综合"].loc["总计", "基准组合收益"]:
            k = (np.log(self._Output["多期综合"].loc["总计", "策略组合收益"]+1) - np.log(self._Output["多期综合"].loc["总计", "基准组合收益"]+1)) / (self._Output["多期综合"].loc["总计", "策略组合收益"] - self._Output["多期综合"].loc["总计", "基准组合收益"])
        else:
            k = 1.0
        self._Output["多期综合"].loc["总计", "主动资产配置超额收益"] = (self._Output["总计"]["主动资产配置超额收益"] * k_t).sum() / k
        self._Output["多期综合"].loc["总计", "主动个券选择超额收益"] = (self._Output["总计"]["主动个券选择超额收益"] * k_t).sum() / k
        self._Output["多期综合"].loc["总计", "交互作用超额收益"] = (self._Output["总计"]["交互作用超额收益"] * k_t).sum() / k
        self._Output["多期综合"].loc["总计", "总超额收益"] = self._Output["多期综合"].loc["总计", "策略组合收益"] - self._Output["多期综合"].loc["总计", "基准组合收益"]
        return 0
    def _repr_html_(self):
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                if iArgName!="计算时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        Formatters = [_QS_formatPandasPercentage]*8
        iHTML = self._Output["多期综合"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        return HTML