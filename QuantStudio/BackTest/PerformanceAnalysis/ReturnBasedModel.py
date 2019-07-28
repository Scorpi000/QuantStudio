# coding=utf-8
"""基于收益率的绩效分析模型"""
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
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.BackTest.SectionFactor.Portfolio import _QS_plotStatistics

class ReturnBasedStyleAnalysisModel(BaseModule):
    """基于收益率回归的风格分析模型"""
    #TargetNAV = Enum(None, arg_type="SingleOption", label="目标净值", order=0)
    TargetIDs = ListStr(arg_type="StrList", label="目标ID", order=1)
    #StyleNAV = Enum(None, arg_type="SingleOption", label="风格净值", order=2)
    StyleIDs = ListStr(arg_type="StrList", label="风格ID", order=3)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=4)
    SummaryWindow = Float(240, arg_type="Integer", label="统计窗口", order=5)
    MinSummaryWindow = Int(20, arg_type="Integer", label="最小统计窗口", order=6)
    def __init__(self, target_table, style_table, name="基于收益率回归的风格分析模型", sys_args={}, **kwargs):
        self._TargetTable = target_table
        self._StyleTable = style_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._TargetTable.getFactorMetaData(key="DataType")))
        self.add_trait("TargetNAV", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="目标净值", order=0))
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._StyleTable.getFactorMetaData(key="DataType")))
        self.add_trait("StyleNAV", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="风格净值", order=2))
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        if not self.TargetIDs: self._Output["目标ID"] = self._TargetTable.getID()
        else: self._Output["目标ID"] = list(self.TargetIDs)
        nTargetID = len(self._Output["目标ID"])
        self._Output["目标收益率"] = np.zeros(shape=(0, nTargetID))
        self._Output["滚动统计量"] = {"R平方":np.zeros((0, nTargetID)), "调整R平方":np.zeros((0, nTargetID)), "t统计量":{}, "F统计量":np.zeros((0, nTargetID))}
        if not self.StyleIDs: self._Output["风格ID"] = self._StyleTable.getID()
        else: self._Output["风格ID"] = list(self.StyleIDs)
        nStyleID = len(self._Output["风格ID"])
        self._Output["风格指数收益率"] = np.zeros((0, nStyleID))
        self._CurCalcInd = 0
        # TODO
        self._nMinSample = (max(2, self.MinSummaryWindow) if np.isinf(self.MinSummaryWindow) else max(2, self.MinSummaryWindow))
        return (self._FactorTable, self._TargetTable)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self.LookBack
            PreDateTime = self.CalcDTs[PreInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self.LookBack
            PreDateTime = self._Model.DateTimeSeries[PreInd]
        if PreInd<0: return 0
        TargetNAV = self._PriceTable.readData(dts=[PreDateTime, idt], ids=self._Output["目标ID"], factor_names=[self.PriceFactor]).iloc[0, :, :].values
        self._Output["收益率"] = np.r_[self._Output["收益率"], _calcReturn(Price, return_type=self.ReturnType)]
        FactorData = self._FactorTable.readData(dts=[PreDateTime], ids=self._Output["风格ID"], factor_names=list(self.TestFactors)).iloc[:, 0, :].values.flatten(order="F")
        self._Output["因子值"] = np.r_[self._Output["因子值"], FactorData.reshape((1, FactorData.shape[0]))]
        if self._Output["收益率"].shape[0]<self._nMinSample: return 0
        StartInd = int(max(0, self._Output["收益率"].shape[0] - self.SummaryWindow))
        X = self._Output["因子值"][StartInd:]
        if self.Constant: X = sm.add_constant(X, prepend=True)
        nID = len(self._Output["目标ID"])
        Statistics = {"R平方":np.full((1, nID), np.nan), "调整R平方":np.full((1, nID), np.nan), "t统计量":np.full((X.shape[1], nID), np.nan), "F统计量":np.full((1, nID), np.nan)}
        for i, iID in enumerate(self._Output["目标ID"]):
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
        FactorIDs, PriceIDs = self._Output.pop("风格ID"), self._Output.pop("目标ID")
        DTs = sorted(self._Output["滚动统计量"]["t统计量"])
        self._Output["最后一期统计量"] = pd.DataFrame({"R平方": self._Output["滚动统计量"]["R平方"][-1], "调整R平方": self._Output["滚动统计量"]["调整R平方"][-1],
                                                      "F统计量": self._Output["滚动统计量"]["F统计量"][-1]}, index=PriceIDs).loc[:, ["R平方", "调整R平方", "F统计量"]]
        Index = pd.MultiIndex.from_product([self.TestFactors, FactorIDs], names=["因子", "风格ID"])
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
        self._Output["滚动t统计量"].columns = ["因子", "风格ID", "时点"]+PriceIDs
        self._Output.pop("收益率"), self._Output.pop("因子值")
        return 0