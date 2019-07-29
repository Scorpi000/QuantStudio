# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Float
from traitsui.api import SetEditor, Item
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.BackTest.BackTestModel import BaseModule

def _calcReturn(price, return_type="简单收益率"):
    if return_type=="对数收益率":
        Return = np.log(1 + np.diff(price, axis=0) / np.abs(price[:-1]))
        Return[np.isinf(Return)] = np.nan
        return Return
    elif return_type=="价格变化量": return np.diff(price, axis=0)
    else: return np.diff(price, axis=0) / np.abs(price[:-1])

class TimeSeriesCorrelation(BaseModule):
    """时间序列相关性"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=2)
    ForecastPeriod = Int(1, arg_type="Integer", label="预测期数", order=3)
    Lag = Int(0, arg_type="Integer", label="滞后期数", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    CorrMethod = Enum("pearson", "spearman", "kendall", arg_type="SingleOption", label="相关性算法", order=6)
    SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=7)
    MinSummaryWindow = Int(2, arg_type="Integer", label="最小统计窗口", order=8)
    def __init__(self, factor_table, price_table, name="时间序列相关性", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        self._PriceTable = price_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._PriceTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return (Items, Context)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["滚动相关性"] = {iFactorName:{} for iFactorName in self.TestFactors}# {因子: {时点: DataFrame(index=[因子ID], columns=[证券ID])}}, 
        self._Output["证券ID"] = self._PriceTable.getID()
        self._Output["收益率"] = np.zeros(shape=(0, len(self._Output["证券ID"])))
        self._Output["因子ID"] = self._FactorTable.getID()
        nFactorID = len(self._Output["因子ID"])
        self._Output["因子值"] = {iFactorName:np.zeros(shape=(0, nFactorID)) for iFactorName in self.TestFactors}
        self._CurCalcInd = 0
        return (self._FactorTable, self._PriceTable)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self.ForecastPeriod - self.Lag
            LastInd = self._CurCalcInd - self.ForecastPeriod
            PreDateTime = self.CalcDTs[PreInd]
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self.ForecastPeriod - self.Lag
            LastInd = self._CurCalcInd - self.ForecastPeriod
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0): return 0
        Price = self._PriceTable.readData(dts=[LastDateTime, idt], ids=self._Output["证券ID"], factor_names=[self.PriceFactor]).iloc[0, :, :].values
        self._Output["收益率"] = np.r_[self._Output["收益率"], _calcReturn(Price, return_type=self.ReturnType)]
        FactorData = self._FactorTable.readData(dts=[PreDateTime], ids=self._Output["因子ID"], factor_names=list(self.TestFactors)).iloc[:, 0, :].values.T
        StartInd = int(max(0, self._Output["收益率"].shape[0] - self.SummaryWindow))
        for i, iFactorName in enumerate(self.TestFactors):
            self._Output["因子值"][iFactorName] = np.r_[self._Output["因子值"][iFactorName], FactorData[i:i+1]]
            if self._Output["收益率"].shape[0]>=self.MinSummaryWindow:
                self._Output["滚动相关性"][iFactorName][idt] = pd.DataFrame(np.c_[self._Output["因子值"][iFactorName][StartInd:], self._Output["收益率"][StartInd:]]).corr(method=self.CorrMethod, min_periods=self.MinSummaryWindow).values[:FactorData.shape[1], FactorData.shape[1]:]
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        FactorIDs, PriceIDs = self._Output.pop("因子ID"), self._Output.pop("证券ID")
        LastDT = max(self._Output["滚动相关性"][self.TestFactors[0]])
        self._Output["最后一期相关性"], self._Output["全样本相关性"] = {}, {}
        for iFactorName in self.TestFactors:
            self._Output["最后一期相关性"][iFactorName] = self._Output["滚动相关性"][iFactorName][LastDT].T
            self._Output["全样本相关性"][iFactorName] = pd.DataFrame(np.c_[self._Output["因子值"][iFactorName], self._Output["收益率"]]).corr(method=self.CorrMethod, min_periods=self.MinSummaryWindow).values[:len(FactorIDs), len(FactorIDs):].T
            self._Output["滚动相关性"][iFactorName] = pd.Panel(self._Output["滚动相关性"][iFactorName], major_axis=FactorIDs, minor_axis=PriceIDs).swapaxes(0, 2).to_frame(filter_observations=False).reset_index()
            self._Output["滚动相关性"][iFactorName].columns = ["因子ID", "时点"]+PriceIDs
        self._Output["最后一期相关性"] = pd.Panel(self._Output["最后一期相关性"], major_axis=PriceIDs, minor_axis=FactorIDs).swapaxes(0, 1).to_frame(filter_observations=False).reset_index()
        self._Output["最后一期相关性"].columns = ["因子", "因子ID"]+PriceIDs
        self._Output["全样本相关性"] = pd.Panel(self._Output["全样本相关性"], major_axis=PriceIDs, minor_axis=FactorIDs).swapaxes(0, 1).to_frame(filter_observations=False).reset_index()
        self._Output["全样本相关性"].columns = ["因子", "因子ID"]+PriceIDs
        self._Output.pop("收益率"), self._Output.pop("因子值")
        return 0