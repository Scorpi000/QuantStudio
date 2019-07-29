# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Bool, Float
from traitsui.api import SetEditor, Item
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.MathFun import CartesianProduct
from QuantStudio.BackTest.BackTestModel import BaseModule
from .Correlation import _calcReturn


class OLS(BaseModule):
    """时间序列 OLS"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=2)
    ForecastPeriod = Int(1, arg_type="Integer", label="预测期数", order=3)
    Lag = Int(0, arg_type="Integer", label="滞后期数", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=6)
    MinSummaryWindow = Int(2, arg_type="Integer", label="最小统计窗口", order=7)
    Constant = Bool(True, arg_type="Bool", label="常数项", order=8)
    def __init__(self, factor_table, price_table, name="时间序列OLS", sys_args={}, **kwargs):
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
        self._Output["证券ID"] = self._PriceTable.getID()
        nID = len(self._Output["证券ID"])
        self._Output["收益率"] = np.zeros(shape=(0, nID))
        self._Output["滚动统计量"] = {"R平方":np.zeros((0, nID)), "调整R平方":np.zeros((0, nID)), "t统计量":{}, "F统计量":np.zeros((0, nID))}
        self._Output["因子ID"] = self._FactorTable.getID()
        nFactorID = len(self._Output["因子ID"])
        self._Output["因子值"] = np.zeros((0, nFactorID*len(self.TestFactors)))
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
        FactorData = self._FactorTable.readData(dts=[PreDateTime], ids=self._Output["因子ID"], factor_names=list(self.TestFactors)).iloc[:, 0, :].values.flatten(order="F")
        self._Output["因子值"] = np.r_[self._Output["因子值"], FactorData.reshape((1, FactorData.shape[0]))]
        if self._Output["收益率"].shape[0]<self.MinSummaryWindow: return 0
        StartInd = int(max(0, self._Output["收益率"].shape[0] - self.SummaryWindow))
        X = self._Output["因子值"][StartInd:]
        if self.Constant: X = sm.add_constant(X, prepend=True)
        nID = len(self._Output["证券ID"])
        Statistics = {"R平方":np.full((1, nID), np.nan), "调整R平方":np.full((1, nID), np.nan), "t统计量":np.full((X.shape[1], nID), np.nan), "F统计量":np.full((1, nID), np.nan)}
        for i, iID in enumerate(self._Output["证券ID"]):
            Y = self._Output["收益率"][StartInd:, i]
            try:
                Result = sm.OLS(Y, X, missing="drop").fit()
                Statistics["R平方"][0, i] = Result.rsquared
                Statistics["调整R平方"][0, i] = Result.rsquared_adj
                Statistics["F统计量"][0, i] = Result.fvalue
                Statistics["t统计量"][:, i] = Result.tvalues
            except:
                pass
        self._Output["滚动统计量"]["R平方"] = np.r_[self._Output["滚动统计量"]["R平方"], Statistics["R平方"]]
        self._Output["滚动统计量"]["调整R平方"] = np.r_[self._Output["滚动统计量"]["调整R平方"], Statistics["调整R平方"]]
        self._Output["滚动统计量"]["F统计量"] = np.r_[self._Output["滚动统计量"]["F统计量"], Statistics["F统计量"]]
        self._Output["滚动统计量"]["t统计量"][idt] = Statistics["t统计量"]
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        FactorIDs, PriceIDs = self._Output.pop("因子ID"), self._Output.pop("证券ID")
        DTs = sorted(self._Output["滚动统计量"]["t统计量"])
        self._Output["最后一期统计量"] = pd.DataFrame({"R平方": self._Output["滚动统计量"]["R平方"][-1], "调整R平方": self._Output["滚动统计量"]["调整R平方"][-1],
                                                      "F统计量": self._Output["滚动统计量"]["F统计量"][-1]}, index=PriceIDs).loc[:, ["R平方", "调整R平方", "F统计量"]]
        Index = pd.MultiIndex.from_product([self.TestFactors, FactorIDs], names=["因子", "因子ID"])
        if self.Constant: Index = Index.insert(0, ("Constant", "Constant"))
        self._Output["最后一期t统计量"] = pd.DataFrame(self._Output["滚动统计量"]["t统计量"][DTs[-1]], index=Index, columns=PriceIDs).reset_index()
        self._Output["全样本统计量"], self._Output["全样本t统计量"] = pd.DataFrame(index=PriceIDs, columns=["R平方", "调整R平方", "F统计量"]), pd.DataFrame(index=Index, columns=PriceIDs)
        X = self._Output["因子值"]
        if self.Constant: X = sm.add_constant(X, prepend=True)
        for i, iID in enumerate(PriceIDs):
            Y = self._Output["收益率"][:, i]
            try:
                Result = sm.OLS(Y, X, missing="drop").fit()
                self._Output["全样本统计量"].iloc[i, 0] = Result.rsquared
                self._Output["全样本统计量"].iloc[i, 1] = Result.rsquared_adj
                self._Output["全样本统计量"].iloc[i, 2] = Result.fvalue
                self._Output["全样本t统计量"].iloc[:, i] = Result.tvalues
            except:
                pass
        self._Output["滚动统计量"]["R平方"] = pd.DataFrame(self._Output["滚动统计量"]["R平方"], index=DTs, columns=PriceIDs)
        self._Output["滚动统计量"]["调整R平方"] = pd.DataFrame(self._Output["滚动统计量"]["调整R平方"], index=DTs, columns=PriceIDs)
        self._Output["滚动统计量"]["F统计量"] = pd.DataFrame(self._Output["滚动统计量"]["F统计量"], index=DTs, columns=PriceIDs)
        self._Output["滚动t统计量"] = pd.Panel(self._Output["滚动统计量"].pop("t统计量"), major_axis=Index, minor_axis=PriceIDs)
        self._Output["滚动t统计量"] = self._Output["滚动t统计量"].swapaxes(0, 2).to_frame(filter_observations=False).reset_index()
        self._Output["滚动t统计量"].columns = ["因子", "因子ID", "时点"]+PriceIDs
        self._Output.pop("收益率"), self._Output.pop("因子值")
        return 0