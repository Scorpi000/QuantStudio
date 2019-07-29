# coding=utf-8
"""基于收益率的绩效分析模型"""
import datetime as dt
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, Float, Int
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.TimeSeriesFactor.Correlation import _calcReturn
from QuantStudio.Tools.AuxiliaryFun import getFactorList
from QuantStudio.Tools.MathFun import regressByCVX


class ReturnBasedStyleModel(BaseModule):
    """基于收益率回归的风格分析模型"""
    #TargetNAV = Enum(None, arg_type="SingleOption", label="目标净值", order=0)
    TargetIDs = ListStr(arg_type="StrList", label="目标ID", order=1)
    #StyleNAV = Enum(None, arg_type="SingleOption", label="风格净值", order=2)
    StyleIDs = ListStr(arg_type="StrList", label="风格ID", order=3)
    ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    SummaryWindow = Float(240, arg_type="Integer", label="统计窗口", order=6)
    MinSummaryWindow = Int(20, arg_type="Integer", label="最小统计窗口", order=7)
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
        self._Output["目标净值"] = np.zeros(shape=(0, nTargetID))
        if not self.StyleIDs: self._Output["风格ID"] = self._StyleTable.getID()
        else: self._Output["风格ID"] = list(self.StyleIDs)
        nStyleID = len(self._Output["风格ID"])
        self._Output["风格指数净值"] = np.zeros((0, nStyleID))
        self._Output["滚动回归系数"] = {iID:[] for iID in self._Output["目标ID"]}
        self._Output["滚动回归R平方"] = []
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._StyleTable, self._TargetTable)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        TargetNAV = self._TargetTable.readData(dts=[idt], ids=self._Output["目标ID"], factor_names=[self.TargetNAV]).iloc[0, :, :].values
        self._Output["目标净值"] = np.r_[self._Output["目标净值"], TargetNAV]
        StyleNAV = self._StyleTable.readData(dts=[idt], ids=self._Output["风格ID"], factor_names=[self.StyleNAV]).iloc[0, :, :].values
        self._Output["风格指数净值"] = np.r_[self._Output["风格指数净值"], StyleNAV]
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        if self._Output["目标净值"].shape[0]-1<self.MinSummaryWindow: return 0
        StartInd = int(max(0, self._Output["目标净值"].shape[0] - 1 - self.SummaryWindow))
        X = _calcReturn(self._Output["风格指数净值"][StartInd:, :], return_type=self.ReturnType)
        Y = _calcReturn(self._Output["目标净值"][StartInd:, :], return_type=self.ReturnType)
        nTargetID, nStyleID = len(self._Output["目标ID"]), len(self._Output["风格ID"])
        Rsquared = np.full((nTargetID, ), np.nan)
        for i, iID in enumerate(self._Output["目标ID"]):
            iMask = ((np.sum(pd.isnull(X), axis=1)==0) & (pd.notnull(Y[:, i])))
            try:
                iBeta = regressByCVX(Y[:, i], X, weight=None, constraints={"Box": {"ub": np.ones((nStyleID, )), "lb": np.zeros((nStyleID, ))},
                                                                                                                      "LinearEq": {"Aeq": np.ones((1, nStyleID)), "beq": 1}})
            except:
                iBeta = None
            if iBeta is None:
                self._Output["滚动回归系数"][iID].append(np.full((nStyleID, ), np.nan))
            else:
                self._Output["滚动回归系数"][iID].append(iBeta)
                Rsquared[i] = 1 - np.nansum((Y[:, i][iMask] - np.dot(X[iMask], iBeta))**2) / np.nansum((Y[:, i][iMask] - np.nanmean(Y[:, i][iMask]))**2)
        self._Output["滚动回归R平方"].append(Rsquared)
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        DTs, StyleIDs, TargetIDs = self._Output.pop("时点"), self._Output.pop("风格ID"), self._Output.pop("目标ID")
        nTargetID, nStyleID = len(TargetIDs), len(StyleIDs)
        X = _calcReturn(self._Output["风格指数净值"], return_type=self.ReturnType)
        Y = _calcReturn(self._Output["目标净值"], return_type=self.ReturnType)
        self._Output["全样本回归系数"] = np.full(shape=(nStyleID, nTargetID), fill_value=np.nan)
        self._Output["全样本回归R平方"] = np.full(shape=(nTargetID, ), fill_value=np.nan)
        for i, iID in enumerate(TargetIDs):
            iMask = ((np.sum(pd.isnull(X), axis=1)==0) & (pd.notnull(Y[:, i])))
            try:
                iBeta = regressByCVX(Y[:, i], X, weight=None, constraints={"Box": {"ub": np.ones((nStyleID, )), "lb": np.zeros((nStyleID, ))},
                                                                                                                      "LinearEq": {"Aeq": np.ones((1, nStyleID)), "beq": 1}})
            except:
                iBeta = None
            if iBeta is not None:
                self._Output["全样本回归系数"][:, i] = iBeta
                self._Output["全样本回归R平方"][i] = 1 - np.nansum((Y[:, i][iMask] - np.dot(X[iMask], iBeta))**2) / np.nansum((Y[:, i][iMask] - np.nanmean(Y[:, i][iMask]))**2)
            self._Output["滚动回归系数"][iID] = pd.DataFrame(self._Output["滚动回归系数"][iID], index=DTs, columns=self.StyleIDs)
        self._Output["全样本回归系数"] = pd.DataFrame(self._Output["全样本回归系数"], index=StyleIDs, columns=TargetIDs)
        self._Output["全样本回归R平方"] = pd.DataFrame(self._Output["全样本回归R平方"], index=TargetIDs, columns=["全样本回归R平方"])
        self._Output["滚动回归R平方"] = pd.DataFrame(self._Output["滚动回归R平方"], index=DTs, columns=TargetIDs)
        self._Output["目标净值"] = pd.DataFrame(self._Output["目标净值"], index=self._Model.DateTimeSeries, columns=self.TargetIDs)
        self._Output["风格指数净值"] = pd.DataFrame(self._Output["风格指数净值"], index=self._Model.DateTimeSeries, columns=self.StyleIDs)
        return 0