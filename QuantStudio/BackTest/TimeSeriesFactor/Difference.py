# coding=utf-8
import datetime as dt
from copy import deepcopy

import numpy as np
import pandas as pd
from traits.api import Enum, List, Int, Float
from scipy import stats

from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.TimeSeriesFactor.Correlation import _calcReturn
from QuantStudio.Tools.api import Panel

class QuantileDifference(BaseModule):
    """分位数法"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        #TestFactor = Enum(None, arg_type="SingleOption", label="测试因子", order=0)
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
        ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=2)
        ForecastPeriod = Int(1, arg_type="Integer", label="预测期数", order=3)
        Lag = Int(0, arg_type="Integer", label="滞后期数", order=4)
        CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
        SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=6)
        MinSummaryWindow = Int(2, arg_type="Integer", label="最小统计窗口", order=7)
        GroupNum = Int(3, arg_type="Integer", label="分组数", order=8)
        def __QS_initArgs__(self):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("TestFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="测试因子", order=0))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            
    def __init__(self, factor_table, name="分位数法", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["证券ID"] = self._FactorTable.getID()
        nID = len(self._Output["证券ID"])
        self._Output["收益率"] = np.zeros(shape=(0, nID))
        self._Output["因子值"] = np.zeros(shape=(0, nID))
        self._Output["滚动t统计量"] = {iID: {} for iID in self._Output["证券ID"]}# {ID: {时点: DataFrame(index=[分位组], columns=[分位组])}}
        self._Output["滚动p值"] = deepcopy(self._Output["滚动t统计量"])# {ID: {时点: DataFrame(index=[分位组], columns=[分位组])}}
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self._QSArgs.orecastPeriod - self._QSArgs.Lag
            LastInd = self._CurCalcInd - self._QSArgs.ForecastPeriod
            PreDateTime = self._QSArgs.CalcDTs[PreInd]
            LastDateTime = self._QSArgs.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self._QSArgs.ForecastPeriod - self._QSArgs.Lag
            LastInd = self._CurCalcInd - self._QSArgs.ForecastPeriod
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0): return 0
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=self._Output["证券ID"], factor_names=[self._QSArgs.PriceFactor]).iloc[0, :, :].values
        self._Output["收益率"] = np.r_[self._Output["收益率"], _calcReturn(Price, return_type=self._QSArgs.ReturnType)]
        FactorData = self._FactorTable.readData(dts=[PreDateTime], ids=self._Output["证券ID"], factor_names=[self._QSArgs.TestFactor]).iloc[0, 0, :].values
        self._Output["因子值"] = np.r_[self._Output["因子值"], FactorData.reshape((1,-1))]
        if self._Output["收益率"].shape[0]<self._QSArgs.MinSummaryWindow: return 0
        StartInd = int(max(0, self._Output["收益率"].shape[0] - self._QSArgs.SummaryWindow))
        FactorData, Return = self._Output["因子值"][StartInd:], self._Output["收益率"][StartInd:, :]
        Mask = {}
        for j in range(self._QSArgs.GroupNum):
            if j==0: Mask[j] = (FactorData<=np.percentile(FactorData, (j+1)/self._QSArgs.GroupNum*100, axis=0))
            else: Mask[j] = ((FactorData>np.percentile(FactorData, j/self._QSArgs.GroupNum*100, axis=0)) & (FactorData<=np.percentile(FactorData, (j+1)/self._QSArgs.GroupNum*100, axis=0)))
        for i, iID in enumerate(self._Output["证券ID"]):
            itStat, ipValue = np.full(shape=(self._QSArgs.GroupNum, self._QSArgs.GroupNum), fill_value=np.nan), np.full(shape=(self._QSArgs.GroupNum, self._QSArgs.GroupNum), fill_value=np.nan)
            for j in range(self._QSArgs.GroupNum):
                for k in range(j+1, self._QSArgs.GroupNum):
                    jkResult = stats.ttest_ind(Return[Mask[j][:, i], i], Return[Mask[k][:, i], i], equal_var=True, nan_policy="omit")
                    itStat[j, k], ipValue[j, k] = jkResult.statistic, jkResult.pvalue
                    itStat[k, j], ipValue[k, j] = -itStat[j, k], ipValue[j, k]
            self._Output["滚动t统计量"][iID][idt], self._Output["滚动p值"][iID][idt] = itStat, ipValue
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        FactorData, Return, IDs = self._Output.pop("因子值"), self._Output.pop("收益率"), self._Output.pop("证券ID")
        self._Output["全样本t统计量"], self._Output["全样本p值"] = {}, {}
        Mask = {}
        for j in range(self._QSArgs.GroupNum):
            if j==0: Mask[j] = (FactorData<=np.percentile(FactorData, (j+1)/self._QSArgs.GroupNum*100, axis=0))
            else: Mask[j] = ((FactorData>np.percentile(FactorData, j/self._QSArgs.GroupNum*100, axis=0)) & (FactorData<=np.percentile(FactorData, (j+1)/self._QSArgs.GroupNum*100, axis=0)))
        for i, iID in enumerate(IDs):
            itStat, ipValue = np.full(shape=(self._QSArgs.GroupNum, self._QSArgs.GroupNum), fill_value=np.nan), np.full(shape=(self._QSArgs.GroupNum, self._QSArgs.GroupNum), fill_value=np.nan)
            for j in range(self._QSArgs.GroupNum):
                for k in range(j+1, self._QSArgs.GroupNum):
                    jkResult = stats.ttest_ind(Return[Mask[j][:, i], i], Return[Mask[k][:, i], i], equal_var=True, nan_policy="omit")
                    itStat[j, k], ipValue[j, k] = jkResult.statistic, jkResult.pvalue
                    itStat[k, j], ipValue[k, j] = -itStat[j, k], ipValue[j, k]
            self._Output["全样本t统计量"][iID], self._Output["全样本p值"][iID] = itStat, ipValue
        DTs = sorted(self._Output["滚动t统计量"][IDs[0]])
        for iID in IDs:
            self._Output["滚动t统计量"][iID] = Panel(self._Output["滚动t统计量"][iID]).to_frame(filter_observations=False)
            self._Output["滚动p值"][iID] = Panel(self._Output["滚动p值"][iID]).to_frame(filter_observations=False)
            self._Output["全样本t统计量"][iID] = pd.DataFrame(self._Output["全样本t统计量"][iID]).stack(dropna=False)
            self._Output["全样本p值"][iID] = pd.DataFrame(self._Output["全样本p值"][iID]).stack(dropna=False)
        self._Output["滚动t统计量"] = Panel(self._Output["滚动t统计量"]).to_frame(filter_observations=False)
        self._Output["滚动t统计量"].index.names = ["分位数组1", "分位数组2", "时点"]
        self._Output["滚动t统计量"] = self._Output["滚动t统计量"].reset_index()
        self._Output["滚动t统计量"] = self._Output["滚动t统计量"][self._Output["滚动t统计量"]["分位数组1"]!=self._Output["滚动t统计量"]["分位数组2"]]
        self._Output["滚动p值"] = Panel(self._Output["滚动p值"]).to_frame(filter_observations=False)
        self._Output["滚动p值"].index.names = ["分位数组1", "分位数组2", "时点"]
        self._Output["滚动p值"] = self._Output["滚动p值"].reset_index()
        self._Output["滚动p值"] = self._Output["滚动p值"][self._Output["滚动p值"]["分位数组1"]!=self._Output["滚动p值"]["分位数组2"]]
        self._Output["全样本t统计量"] = pd.DataFrame(self._Output["全样本t统计量"]).reset_index()
        self._Output["全样本p值"] = pd.DataFrame(self._Output["全样本p值"]).reset_index()
        self._Output["全样本t统计量"].columns = self._Output["全样本p值"].columns = ["分位数组1", "分位数组2"]+IDs
        self._Output["全样本t统计量"] = self._Output["全样本t统计量"][self._Output["全样本t统计量"]["分位数组1"]!=self._Output["全样本t统计量"]["分位数组2"]]
        self._Output["全样本p值"] = self._Output["全样本p值"][self._Output["全样本p值"]["分位数组1"]!=self._Output["全样本p值"]["分位数组2"]]
        self._Output["最后一期t统计量"] = self._Output["滚动t统计量"][self._Output["滚动t统计量"]["时点"]==DTs[-1]]
        self._Output["最后一期p值"] = self._Output["滚动p值"][self._Output["滚动p值"]["时点"]==DTs[-1]]
        self._Output["最后一期t统计量"].pop("时点")
        self._Output["最后一期p值"].pop("时点")
        return 0