# coding=utf-8
import datetime as dt
from copy import deepcopy
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import Enum, List, Int, Str, Float, Dict, ListStr
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import calcMaxDrawdownRate
from QuantStudio.BackTest.BackTestModel import BaseModule
from .Correlation import _calcReturn

class QuantileTiming(BaseModule):
    """分位数择时"""
    #TestFactor = Enum(None, arg_type="SingleOption", label="测试因子", order=0)
    FactorIDs = ListStr(arg_type="List", label="因子ID", order=1)
    FactorOrder = Dict(key_trait=Str(), value_trait=Enum("降序", "升序"), arg_type="ArgDict", label="排序方向", order=2)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=4)
    SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=5)
    MinSummaryWindow = Int(2, arg_type="Integer", label="最小统计窗口", order=6)
    GroupNum = Int(3, arg_type="Integer", label="分组数", order=7)
    def __init__(self, factor_table, price_table, name="分位数择时", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        self._PriceTable = price_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="测试因子", order=0))
        self.FactorIDs = self._FactorTable.getID()
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._PriceTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ["价","Price","price"])
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        SecurityIDs = self._PriceTable.getID()
        nDT, nSecurityID, nFactorID = len(dts), len(SecurityIDs), len(self.FactorIDs)
        self._Output["证券ID"] = SecurityIDs
        self._Output["证券收益率"] = np.zeros(shape=(nDT, nSecurityID))
        self._Output["因子值"] = np.full(shape=(nDT, nFactorID), fill_value=np.nan)
        self._Output["因子符号"] = 2 * np.array([(self.FactorOrder.get(iID, "降序")=="升序") for i, iID in enumerate(self.FactorIDs)]).astype(float) - 1
        self._Output["最新信号"] = np.full(shape=(nFactorID, self.GroupNum), fill_value=np.nan)
        self._Output["信号"] = np.full(shape=(nDT, nFactorID, self.GroupNum), fill_value=np.nan)
        self._Output["信号收益率"] = np.full(shape=(nFactorID, nSecurityID, nDT, self.GroupNum), fill_value=np.nan)
        self._CurCalcInd = 0
        return (self._FactorTable, self._PriceTable)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        DTIdx = self._Model.DateTimeIndex
        if DTIdx>0:
            Price = self._PriceTable.readData(dts=self._Model.DateTimeSeries[DTIdx-1:DTIdx+1], ids=self._Output["证券ID"], factor_names=[self.PriceFactor]).iloc[0, :, :].values
            Return = _calcReturn(Price, return_type="简单收益率")
            self._Output["证券收益率"][DTIdx:DTIdx+1, :] = Return
            Mask = pd.notnull(self._Output["最新信号"])
            if np.any(Mask):
                self._Output["信号收益率"][np.arange(len(self.FactorIDs))[Mask], :, DTIdx, self._Output["最新信号"][Mask].astype(int)] = np.repeat(Return, np.sum(Mask), axis=0)
        self._Output["因子值"][DTIdx, :] = self._FactorTable.readData(dts=self._Model.DateTimeSeries[DTIdx:DTIdx+1], ids=list(self.FactorIDs), factor_names=[self.TestFactor]).iloc[0, 0, :].values
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        if DTIdx+1<self.MinSummaryWindow: return 0
        FactorData = self._Output["因子值"][int(max(0, DTIdx + 1 - self.SummaryWindow)):DTIdx+1, :] * self._Output["因子符号"]
        Quantiles = np.nanpercentile(FactorData, np.arange(1, self.GroupNum+1) / self.GroupNum * 100, axis=0)
        Signal = np.sum(Quantiles<FactorData[-1, :], axis=0).astype(float)
        Signal[pd.isnull(FactorData[-1, :])] = np.nan
        self._Output["最新信号"] = Signal
        Mask = pd.notnull(Signal)
        if np.any(Mask):
            self._Output["信号"][DTIdx, np.arange(len(self.FactorIDs))[Mask], Signal[Mask].astype(int)] = 1
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        DTs = self._Model.DateTimeSeries
        self._Output["证券收益率"] = pd.DataFrame(self._Output["证券收益率"], index=DTs, columns=self._Output["证券ID"])
        self._Output["证券净值"] = (self._Output["证券收益率"] + 1).cumprod()
        self._Output["因子值"] = pd.DataFrame(self._Output["因子值"], index=DTs, columns=self.FactorIDs)
        Groups = np.arange(1, self.GroupNum+1).astype(str)
        Signal = self._Output["信号"]
        self._Output["信号"] = {}
        for i, iID in enumerate(self.FactorIDs):
            iSignal = pd.DataFrame(Signal[:, i, :], index=DTs, columns=Groups)
            iDTs = pd.notnull(iSignal).any(axis=1)
            iDTs = iDTs[iDTs].index
            if iDTs.shape[0]>0:
                self._Output["信号"][iID] = iSignal.loc[iDTs[0]:]
            else:
                self._Output["信号"][iID] = pd.DataFrame(columns=Groups)
        SignalReturn = self._Output["信号收益率"]
        self._Output["信号收益率"] = {}
        self._Output["信号净值"] = {}
        self._Output["统计数据"] = {}
        nDate = len(DTs)
        nYear = (DTs[-1] - DTs[0]).days / 365
        for j, jSecurityID in enumerate(self._Output["证券ID"]):
            self._Output["信号收益率"][jSecurityID] = {}
            self._Output["信号净值"][jSecurityID] = {}
            self._Output["统计数据"][jSecurityID] = {}
            for i, iID in enumerate(self.FactorIDs):
                ijSignalReturn = pd.DataFrame(SignalReturn[i, j, :, :], index=DTs, columns=Groups).loc[self._Output["信号"][iID].index]
                if ijSignalReturn.shape[0]==0:
                    self._Output["信号收益率"][jSecurityID][iID] = ijSignalReturn
                    self._Output["信号净值"][jSecurityID][iID] = ijSignalReturn
                    self._Output["统计数据"][jSecurityID][iID] = pd.DataFrame(index=ijSignalReturn.columns, columns=["总收益率", "年化收益率", "波动率", "Sharpe比率", "胜率", "最大回撤率", "最大回撤开始时间", "最大回撤结束时间"])
                else:
                    ijNV = (1+ijSignalReturn.fillna(0)).cumprod()
                    self._Output["信号收益率"][jSecurityID][iID] = ijSignalReturn
                    self._Output["信号净值"][jSecurityID][iID] = ijNV
                    ijStat = pd.DataFrame(index=ijNV.columns)
                    ijStat["总收益率"] = ijNV.iloc[-1, :] - 1
                    ijStat["年化收益率"] = ijNV.iloc[-1, :] ** (1 / nYear) - 1
                    ijStat["波动率"] = ijSignalReturn.std() * np.sqrt(nDate / nYear)
                    ijStat["Sharpe比率"] = ijStat["年化收益率"] / ijStat["波动率"]
                    ijStat["胜率"] = (ijSignalReturn>0).sum() / pd.notnull(ijSignalReturn).sum()
                    ijStat["最大回撤率"] = pd.Series(np.nan, index=ijStat.index)
                    ijStat["最大回撤开始时间"] = pd.Series(index=ijStat.index, dtype="O")
                    ijStat["最大回撤结束时间"] = pd.Series(index=ijStat.index, dtype="O")
                    for iCol in ijNV.columns:
                        iMaxDD, iStartPos, iEndPos = calcMaxDrawdownRate(ijNV.loc[:, iCol].values)
                        ijStat.loc[iCol, "最大回撤率"] = abs(iMaxDD)
                        ijStat.loc[iCol, "最大回撤开始时间"] = (ijNV.index[iStartPos] if iStartPos is not None else None)
                        ijStat.loc[iCol, "最大回撤结束时间"] = (ijNV.index[iEndPos] if iEndPos is not None else None)
                    self._Output["统计数据"][jSecurityID][iID] = ijStat
        self._Output.pop("最新信号")
        self._Output.pop("因子符号")
        self._Output.pop("证券ID")
        return 0