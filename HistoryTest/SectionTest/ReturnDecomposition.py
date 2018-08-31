# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, on_trait_change
from traitsui.api import SetEditor, Item

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.Tools.ExcelFun import copyChart
from QuantStudio.HistoryTest.HistoryTestModel import BaseModule

class FamaMacBethRegression(BaseModule):
    """Fama-MacBeth 回归"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    IndustryFactor = Enum("无", arg_type="SingleOption", label="行业因子", order=2)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=3)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=4)
    RollAvgPeriod = Int(12, arg_type="Integer", label="滚动平均期数", order=5)
    def __init__(self, factor_table, sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name="Fama-MacBeth 回归", sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=3))
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return (Items, Context)
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        self._Output = {"Pure Return":[], "Raw Return":[], "时点":[]}
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt):
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            LastInd = self._CurCalcInd - 1
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            LastInd = self._CurCalcInd - 1
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        nFactor = len(self.TestFactors)
        if LastInd<0:
            self._Output["Pure Return"].append((np.nan, )*nFactor)
            self._Output["Raw Return"].append((np.nan, )*nFactor)
            self._Output["时点"].append(idt)
            return self._Output
        LastIDs = self._FactorTable.getFilteredID(idt=LastDateTime, id_filter_str=self.IDFilter)
        FactorData = self._FactorTable.readData(dts=[LastDateTime], ids=LastIDs, factor_names=list(self.TestFactors)).iloc[:,0,:]
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=LastIDs, factor_names=[self.PriceFactor]).iloc[0]
        Ret = Price.iloc[1] / Price.iloc[0] - 1
        # 展开Dummy因子
        if self.IndustryFactor!="无":
            DummyFactorData = self._FactorTable.readData(dts=[LastDateTime], ids=LastIDs, factor_names=[self.IndustryFactor]).iloc[0,0,:]
            Mask = pd.notnull(DummyFactorData)
            DummyFactorData = DummyVarTo01Var(DummyFactorData[Mask], ignore_na=True)
            FactorData = pd.merge(FactorData.loc[Mask], DummyFactorData, left_index=True, right_index=True)
        # 回归
        yData = Ret[FactorData.index].values
        xData = FactorData.values
        if self.IndustryFactor=="无":
            xData = sm.add_constant(xData, prepend=False)
            LastInds = [nFactor]
        else:
            LastInds = [nFactor+i for i in range(xData.shape[1]-nFactor)]
        try:
            Result = sm.OLS(yData, xData, missing="drop").fit()
            self._Output["Pure Return"].append(Result.params[0:nFactor])
        except:
            self._Output["Pure Return"].append(np.zeros(nFactor)+np.nan)
        self._Output["Raw Return"].append(np.zeros(nFactor)+np.nan)
        for i, iFactorName in enumerate(self.TestFactors):
            iXData = xData[:,[i]+LastInds]
            try:
                Result = sm.OLS(yData, iXData, missing="drop").fit()
                self._Output["Raw Return"][-1][i] = Result.params[0]
            except:
                pass
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        self._Output["Pure Return"] = pd.DataFrame(self._Output["Pure Return"], index=self._Output["时点"], columns=list(self.TestFactors))
        self._Output["Raw Return"] = pd.DataFrame(self._Output["Raw Return"], index=self._Output["时点"], columns=list(self.TestFactors))
        self._Output["滚动t统计量_Pure"] = pd.DataFrame(np.nan, index=self._Output["时点"], columns=list(self.TestFactors))
        self._Output["滚动t统计量_Raw"] = pd.DataFrame(np.nan, index=self._Output["时点"], columns=list(self.TestFactors))
        nDT = self._Output["Raw Return"].shape[0]
        # 计算滚动t统计量
        for i in range(nDT):
            if i<self.RollAvgPeriod-1: continue
            iReturn = self._Output["Pure Return"].iloc[i-self.RollAvgPeriod+1:i+1, :]
            self._Output["滚动t统计量_Pure"].iloc[i] = iReturn.mean(axis=0) / iReturn.std(axis=0) * pd.notnull(iReturn).sum(axis=0)**0.5
            iReturn = self._Output["Raw Return"].iloc[i-self.RollAvgPeriod+1:i+1, :]
            self._Output["滚动t统计量_Raw"].iloc[i] = iReturn.mean(axis=0) / iReturn.std(axis=0) * pd.notnull(iReturn).sum(axis=0)**0.5
        nYear = (self._Output["时点"][-1] - self._Output["时点"][0]).days / 365
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["Pure Return"].columns)
        self._Output["统计数据"]["年化收益率(Pure)"] = ((1 + self._Output["Pure Return"]).prod())**(1/nYear) - 1
        self._Output["统计数据"]["跟踪误差(Pure)"] = self._Output["Pure Return"].std() * np.sqrt(nDT/nYear)
        self._Output["统计数据"]["信息比率(Pure)"] = self._Output["统计数据"]["年化收益率(Pure)"] / self._Output["统计数据"]["跟踪误差(Pure)"]
        self._Output["统计数据"]["胜率(Pure)"] = (self._Output["Pure Return"]>0).sum() / nDT
        self._Output["统计数据"]["t统计量(Pure)"] = self._Output["Pure Return"].mean() / self._Output["Pure Return"].std() * np.sqrt(nDT)
        self._Output["统计数据"]["年化收益率(Raw)"] = (1 + self._Output["Raw Return"]).prod()**(1/nYear) - 1
        self._Output["统计数据"]["跟踪误差(Raw)"] = self._Output["Raw Return"].std() * np.sqrt(nDT/nYear)
        self._Output["统计数据"]["信息比率(Raw)"] = self._Output["统计数据"]["年化收益率(Raw)"] / self._Output["统计数据"]["跟踪误差(Raw)"]
        self._Output["统计数据"]["胜率(Raw)"] = (self._Output["Raw Return"]>0).sum() / nDT
        self._Output["统计数据"]["t统计量(Raw)"] = self._Output["Raw Return"].mean() / self._Output["Raw Return"].std() * np.sqrt(nDT)
        self._Output["统计数据"]["年化收益率(Pure-Naive)"] = (1 + self._Output["Pure Return"] - self._Output["Raw Return"]).prod()**(1/nYear) - 1
        self._Output["统计数据"]["跟踪误差(Pure-Naive)"] = (self._Output["Pure Return"] - self._Output["Raw Return"]).std() * np.sqrt(nDT/nYear)
        self._Output["统计数据"]["信息比率(Pure-Naive)"] = self._Output["统计数据"]["年化收益率(Pure-Naive)"] / self._Output["统计数据"]["跟踪误差(Pure-Naive)"]
        self._Output["统计数据"]["胜率(Pure-Naive)"] = (self._Output["Pure Return"] - self._Output["Raw Return"]>0).sum() / nDT
        self._Output["统计数据"]["t统计量(Pure-Naive)"] = (self._Output["Pure Return"] - self._Output["Raw Return"]).mean() / (self._Output["Pure Return"] - self._Output["Raw Return"]).std() * np.sqrt(nDT)
        self._Output.pop("时点")
        return 0