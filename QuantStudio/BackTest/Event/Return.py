# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, on_trait_change
from traitsui.api import SetEditor, Item
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.BackTest.BackTestModel import BaseModule

class AbnormalReturn(BaseModule):
    """异常收益率"""
    EventFilter = Str(arg_type="String", label="事件定义", order=0)
    #NormalReturn = Enum(None, arg_type="SingleOption", label="正常收益率", order=1)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
    def __init__(self, factor_table, name="异常收益率", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("NormalReturn", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="正常收益率", order=1))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["IC"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["股票数"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self.CalcDTs[PreInd]
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0):
            for iFactorName in self.TestFactors:
                self._Output["IC"][iFactorName].append(np.nan)
                self._Output["股票数"][iFactorName].append(np.nan)
            self._Output["时点"].append(idt)
            return 0
        PreIDs = self._FactorTable.getFilteredID(idt=PreDateTime, id_filter_str=self.IDFilter)
        FactorExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self.TestFactors)).iloc[:, 0, :]
        CurPrice = self._FactorTable.readData(dts=[idt], ids=PreIDs, factor_names=[self.PriceFactor]).iloc[0, 0, :]
        LastPrice = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.PriceFactor]).iloc[0, 0, :]
        Ret = CurPrice/LastPrice-1
        if self.IndustryFactor!="无":# 进行收益率的行业调整
            IndustryData = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.IndustryFactor]).iloc[0, 0, :]
            AllIndustry = IndustryData.unique()
            if self.WeightFactor=="等权":
                for iIndustry in AllIndustry:
                    iMask = (IndustryData==iIndustry)
                    Ret[iMask] -= Ret[iMask].mean()
            else:
                WeightData = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.WeightFactor]).iloc[0, 0, :]
                for iIndustry in AllIndustry:
                    iMask = (IndustryData==iIndustry)
                    iWeight = WeightData[iMask]
                    iRet = Ret[iMask]
                    Ret[iMask] -= (iRet*iWeight).sum() / iWeight[pd.notnull(iWeight) & pd.notnull(iRet)].sum(skipna=False)
        for iFactorName in self.TestFactors:
            self._Output["IC"][iFactorName].append(FactorExpose[iFactorName].corr(Ret, method=self.CorrMethod))
            self._Output["股票数"][iFactorName].append(pd.notnull(FactorExpose[iFactorName]).sum())
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        CalcDateTimes = self._Output.pop("时点")
        self._Output["股票数"] = pd.DataFrame(self._Output["股票数"], index=CalcDateTimes)
        self._Output["IC"] = pd.DataFrame(self._Output["IC"], index=CalcDateTimes)
        for i, iFactorName in enumerate(self.TestFactors):
            if self.FactorOrder[iFactorName]=="升序": self._Output["IC"][iFactorName] = -self._Output["IC"][iFactorName]
        self._Output["IC的移动平均"] = self._Output["IC"].copy()
        for i in range(len(CalcDateTimes)):
            if i<self.RollAvgPeriod-1: self._Output["IC的移动平均"].iloc[i,:] = np.nan
            else: self._Output["IC的移动平均"].iloc[i,:] = self._Output["IC"].iloc[i-self.RollAvgPeriod+1:i+1, :].mean()
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["IC"].columns)
        self._Output["统计数据"]["平均值"] = self._Output["IC"].mean()
        self._Output["统计数据"]["标准差"] = self._Output["IC"].std()
        self._Output["统计数据"]["最小值"] = self._Output["IC"].min()
        self._Output["统计数据"]["最大值"] = self._Output["IC"].max()
        self._Output["统计数据"]["IC_IR"] = self._Output["统计数据"]["平均值"] / self._Output["统计数据"]["标准差"]
        self._Output["统计数据"]["t统计量"] = np.nan
        self._Output["统计数据"]["平均股票数"] = self._Output["股票数"].mean()
        self._Output["统计数据"]["IC×Sqrt(N)"] = self._Output["统计数据"]["平均值"]*np.sqrt(self._Output["统计数据"]["平均股票数"])
        self._Output["统计数据"]["有效期数"] = 0.0
        for iFactor in self._Output["IC"]: self._Output["统计数据"].loc[iFactor,"有效期数"] = pd.notnull(self._Output["IC"][iFactor]).sum()
        self._Output["统计数据"]["t统计量"] = (self._Output["统计数据"]["有效期数"]**0.5)*self._Output["统计数据"]["IC_IR"]
        return 0

