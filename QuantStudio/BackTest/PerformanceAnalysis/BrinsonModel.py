# coding=utf-8
"""Brinson 绩效分析模型"""
import datetime as dt
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, Instance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.BackTest.SectionFactor.Portfolio import _QS_plotStatistics

# 前提条件:
# 1. 投资组合的权重之和为 1, 与 1 的差值部分归为现金
# 2. 两个计算时点之间没有调整策略持仓
class BrinsonModel(BaseModule):
    """Brinson 绩效分析模型"""
    #Portfolio = Enum(None, arg_type="SingleOption", label="策略组合", order=0)
    #BenchmarkPortfolio = Enum(None, arg_type="SingleOption", label="基准组合", order=1)
    #GroupFactor = Enum(None, arg_type="SingleOption", label="资产类别", order=2)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=4)
    def __init__(self, factor_table, name="Brinson绩效分析模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("Portfolio", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="策略组合", order=0))
        self.add_trait("BenchmarkPortfolio", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="基准组合", order=1))
        self.add_trait("GroupFactor", Enum(*DefaultStrFactorList, arg_type="SingleOption", label="资产类别", order=2))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        if self.CalcDTs: DTs = self.CalcDTs
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
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            if self._CurCalcInd>0: PreDT = self.CalcDTs[self._CurCalcInd - 1]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            if self._CurCalcInd>0: PreDT = self._Model.DateTimeSeries[self._CurCalcInd - 1]
        if PreDT is None: return 0
        Portfolio = self._FactorTable.readData(factor_names=[self.Portfolio, self.BenchmarkPortfolio], dts=[PreDT], ids=self._IDs).iloc[:, 0, :]
        BenchmarkPortfolio, Portfolio = Portfolio.iloc[:, 1], Portfolio.iloc[:, 0]
        Portfolio[pd.isnull(Portfolio)], BenchmarkPortfolio[pd.isnull(BenchmarkPortfolio)] = 0.0, 0.0
        Price = self._FactorTable.readData(factor_names=[self.PriceFactor], dts=[PreDT, idt], ids=self._IDs).iloc[0]
        Return = Price.iloc[1] / Price.iloc[0] - 1
        Return[pd.isnull(Return)] = 0.0
        GroupData = self._FactorTable.readData(factor_names=[self.GroupFactor], ids=self._IDs, dts=[PreDT]).iloc[0, 0, :]
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
        self._Output["多期综合"] = pd.DataFrame(dtype=np.float)
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
        if len(self.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self.ArgNames:
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