# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, Float
from traitsui.api import SetEditor, Item
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.Tools.IDFun import testIDFilterStr

def _calcReturn(price, return_type="简单收益率"):
    if return_type=="对数收益率":
        Return = np.log(1 + np.diff(price, axis=0) / np.abs(price[:-1]))
        Return[np.isinf(Return)] = np.nan
        return Return
    elif return_type=="绝对变化量": return np.diff(price, axis=0)
    else: return np.diff(price, axis=0) / np.abs(price[:-1])

class CMRM(BaseModule):
    """均值常数模型"""
    EventFilter = Str(arg_type="String", label="事件定义", order=0)
    EventPreWindow = Int(20, arg_type="Integer", label="事件前窗口", order=1)
    EventPostWindow = Int(20, arg_type="Integer", label="事件后窗口", order=2)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    ReturnType = Enum("简单收益率", "对数收益率", "绝对变化量", arg_type="SingleOption", label="收益率类型", order=4)
    EstWindow = Int(240, arg_type="Integer", label="估计窗口", order=5)
    EstSampleFilter = Str(arg_type="String", label="样本筛选", order=6)
    EstSampleLen = Int(20, arg_type="Integer", label="估计样本量", order=7)
    def __init__(self, factor_table, name="均值常数模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["正常收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["异常收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["异常协方差"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["事件记录"] = np.full(shape=(0, 3), fill_value=None, dtype="O")# [ID, 时点, 事件发生距离当前的时点数]
        self._AllDTs = self._FactorTable.getDateTime()
        if self.EstSampleFilter: self._CompiledIDFilterStr, self._FilterFactors = testIDFilterStr(self.EstSampleFilter, self._FactorTable.FactorNames)
        if not self._AllDTs: self._AllDTs = dts
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow+self.EstWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID, EventWindow = len(IDs), self.EventPreWindow+1+self.EventPostWindow
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            self._Output["正常收益率"] = np.r_[self._Output["正常收益率"], np.full(shape=(nID, EventWindow), fill_value=np.nan)]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], np.full(shape=(nID, EventWindow), fill_value=np.nan)]
            self._Output["异常协方差"] = np.r_[self._Output["异常协方差"], np.full(shape=(nID, EventWindow, EventWindow), fill_value=np.nan)]
            EstStartInd = CurInd - self.EventPreWindow - self.EstWindow - 1
            Price = self._FactorTable.readData(dts=self._AllDTs[EstStartInd:CurInd+1], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
            Return = _calcReturn(Price.values, return_type=self.ReturnType)
            EstReturn = Return[:self.EstWindow]
            if self.EstSampleFilter:
                temp = self._FactorTable.readData(dts=self._AllDTs[EstStartInd+1:EstStartInd+self.EstWindow+1], ids=IDs, factor_names=self._FilterFactors)
                FilterMask = eval(self._CompiledIDFilterStr).values
            else:
                FilterMask = np.full(EstReturn.shape, fill_value=True)
            FilterMask = (FilterMask & pd.notnull(EstReturn))
            FilterMask = (FilterMask & (np.flipud(np.cumsum(np.flipud(FilterMask), axis=0))<=self.EstSampleLen))
            EstReturn[~FilterMask] = np.nan
            ExpectedReturn, Var = np.nanmean(EstReturn, axis=0), np.nanvar(EstReturn, axis=0, ddof=1)
            FilterMask = ((np.sum(FilterMask, axis=0)<self.EstSampleLen) | (Var<1e-6))
            ExpectedReturn[FilterMask] = np.nan
            Var[FilterMask] = np.nan
            self._Output["正常收益率"][-nID:, :] = ExpectedReturn.reshape((nID, 1)).repeat(EventWindow, axis=1)
            self._Output["异常收益率"][-nID:, :self.EventPreWindow+1] = (Return[self.EstWindow:] - ExpectedReturn).T
            CovMatrix = (np.eye(EventWindow)+np.ones((EventWindow, EventWindow))/self.EstSampleLen).reshape((1, EventWindow, EventWindow)).repeat(nID, axis=0)
            self._Output["异常协方差"][-nID:, :, :] = (CovMatrix.T*Var).T
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        if np.sum(Mask)==0: return 0
        IDs = self._Output["事件记录"][:, 0][Mask]
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        self._Output["异常收益率"][RowPos, ColPos] = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - self._Output["正常收益率"][RowPos, ColPos])
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        AR = self._Output["异常收益率"]# 单时点累积异常收益率
        CAR = np.nancumsum(AR, axis=1)# 向前累积异常收益率
        FBCAR = np.c_[np.fliplr(np.nancumsum(np.fliplr(AR[:, :self.EventPreWindow+1]), axis=1))[:, :self.EventPreWindow], np.nancumsum(AR[:, self.EventPreWindow:], axis=1)]# 向前向后累积异常收益率
        AR_Var = np.full(AR.shape, fill_value=np.nan)
        CAR_Var = np.full(AR.shape, fill_value=np.nan)
        FBCAR_Var = np.full(AR.shape, fill_value=np.nan)
        for i in range(AR.shape[1]):
            ei = np.zeros(AR.shape[1])
            ei[i] = 1
            AR_Var[:, i] = np.dot(np.dot(self._Output["异常协方差"], ei), ei)
            ei[:i] = 1
            CAR_Var[:, i] = np.dot(np.dot(self._Output["异常协方差"], ei), ei)
            ei[:] = 0
            ei[i:self.EventPreWindow+1] = 1
            ei[self.EventPreWindow:i+1] = 1
            FBCAR_Var[:, i] = np.dot(np.dot(self._Output["异常协方差"], ei), ei)
        AR_Avg, AR_Avg_Var = np.nanmean(AR, axis=0), np.nansum(AR_Var, axis=0) / np.sum(~np.isnan(AR_Var), axis=0)**2
        CAR_Avg, CAR_Avg_Var = np.nanmean(CAR, axis=0), np.nansum(CAR_Var, axis=0) / np.sum(~np.isnan(CAR_Var), axis=0)**2
        FBCAR_Avg, FBCAR_Avg_Var = np.nanmean(FBCAR, axis=0), np.nansum(FBCAR_Var, axis=0) / np.sum(~np.isnan(FBCAR_Var), axis=0)**2
        self._Output["J1统计量"] = {"异常收益率": pd.DataFrame(AR_Avg, index=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), columns=["单时点"])}
        self._Output["J1统计量"]["异常收益率"]["向前累积"] = CAR_Avg
        self._Output["J1统计量"]["异常收益率"]["向前向后累积"] = FBCAR_Avg
        self._Output["J1统计量"]["J1"] = pd.DataFrame(AR_Avg / AR_Avg_Var**0.5, index=self._Output["J1统计量"]["异常收益率"].index, columns=["单时点"])
        self._Output["J1统计量"]["J1"]["向前累积"] = CAR_Avg / CAR_Avg_Var**0.5
        self._Output["J1统计量"]["J1"]["向前向后累积"] = FBCAR_Avg / FBCAR_Avg_Var**0.5
        self._Output["J1统计量"]["p值"] = pd.DataFrame(norm.sf(np.abs(self._Output["J1统计量"]["J1"].values)), index=self._Output["J1统计量"]["J1"].index, columns=self._Output["J1统计量"]["J1"].columns)
        SAR, SCAR, SFBCAR = AR / AR_Var**0.5, CAR / CAR_Var**0.5, FBCAR / FBCAR_Var**0.5
        SAR[np.isinf(SAR)] = SCAR[np.isinf(SCAR)] = SFBCAR[np.isinf(SFBCAR)] = np.nan
        SAR_Avg, SCAR_Avg, SFBCAR_Avg = np.nanmean(SAR, axis=0), np.nanmean(SCAR, axis=0), np.nanmean(SFBCAR, axis=0)
        self._Output["J2统计量"] = {"标准化异常收益率": pd.DataFrame(SAR_Avg, index=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), columns=["单时点"])}
        self._Output["J2统计量"]["标准化异常收益率"]["向前累积"] = SCAR_Avg
        self._Output["J2统计量"]["标准化异常收益率"]["向前向后累积"] = SFBCAR_Avg
        self._Output["J2统计量"]["J2"] = pd.DataFrame(SAR_Avg * (np.sum(~np.isnan(SAR), axis=0) * (self.EstSampleLen - 4) / (self.EstSampleLen - 2))**0.5, index=self._Output["J2统计量"]["标准化异常收益率"].index, columns=["单时点"])
        self._Output["J2统计量"]["J2"]["向前累积"] = SCAR_Avg * (np.sum(~np.isnan(SCAR), axis=0) * (self.EstSampleLen - 4) / (self.EstSampleLen - 2))**0.5
        self._Output["J2统计量"]["J2"]["向前向后累积"] = SFBCAR_Avg * (np.sum(~np.isnan(SFBCAR), axis=0) * (self.EstSampleLen - 4) / (self.EstSampleLen - 2))**0.5
        self._Output["J2统计量"]["p值"] = pd.DataFrame(norm.sf(np.abs(self._Output["J2统计量"]["J2"].values)), index=self._Output["J2统计量"]["J2"].index, columns=self._Output["J2统计量"]["J2"].columns)
        N = np.sum(~np.isnan(AR), axis=0)
        self._Output["J3统计量"] = {"J3": pd.DataFrame((np.sum(AR>0, axis=0)/N - 0.5)*N**0.5/0.5, index=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), columns=["单时点"])}
        N = np.sum(~np.isnan(CAR), axis=0)
        self._Output["J3统计量"] ["J3"]["向前累积"] = (np.sum(CAR>0, axis=0)/N - 0.5)*N**0.5/0.5
        N = np.sum(~np.isnan(FBCAR), axis=0)
        self._Output["J3统计量"] ["J3"]["向前向后累积"] = (np.sum(FBCAR>0, axis=0)/N - 0.5)*N**0.5/0.5
        self._Output["J3统计量"]["p值"] = pd.DataFrame(norm.sf(np.abs(self._Output["J3统计量"]["J3"].values)), index=self._Output["J3统计量"]["J3"].index, columns=self._Output["J3统计量"]["J3"].columns)
        AR = AR[np.sum(np.isnan(AR), axis=1)==0, :]
        N = AR.shape[0]
        L2 = self.EventPreWindow+1+self.EventPostWindow
        K = np.full_like(AR, np.nan)
        K[np.arange(N).repeat(L2), np.argsort(AR, axis=1).flatten()] = (np.arange(L2*N) % L2) + 1
        J4 = np.nansum(K-(L2+1)/2, axis=0) / N
        J4 = J4 / (np.nansum(J4**2) / L2)**0.5
        self._Output["J4统计量"] = {"J4": pd.DataFrame(J4, index=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), columns=["单时点"])}
        self._Output["J4统计量"]["p值"] = pd.DataFrame(norm.sf(np.abs(self._Output["J4统计量"]["J4"].values)), index=self._Output["J4统计量"]["J4"].index, columns=self._Output["J4统计量"]["J4"].columns)
        Index = pd.MultiIndex.from_arrays(self._Output.pop("事件记录")[:,:2].T, names=["ID", "时点"])
        self._Output["正常收益率"] = pd.DataFrame(self._Output["正常收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index).reset_index()
        self._Output["异常收益率"] = pd.DataFrame(self._Output["异常收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index).reset_index()
        self._Output.pop("异常协方差")
        return 0

class MAM(CMRM):
    """市场调整模型"""
    #BenchmarkPrice = Enum(None, arg_type="SingleOption", label="基准价格", order=8)
    BenchmarkID = Str(arg_type="String", label="基准ID", order=9)
    def __init__(self, factor_table, benchmark_ft, name="市场调整模型", sys_args={}, **kwargs):
        self._BenchmarkFT = benchmark_ft
        return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._BenchmarkFT.getFactorMetaData(key="DataType")))
        self.add_trait("BenchmarkPrice", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="基准价格", order=6))
        self.BenchmarkPrice = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.BenchmarkID = self._BenchmarkFT.getID(ifactor_name=self.BenchmarkPrice)[0]
        return super().__QS_initArgs__()
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        return super().__QS_start__(mdl=mdl, dts=dts, **kwargs) + (self._BenchmarkFT, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow+self.EstWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID, EventWindow = len(IDs), self.EventPreWindow+1+self.EventPostWindow
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            self._Output["正常收益率"] = np.r_[self._Output["正常收益率"], np.full(shape=(nID, EventWindow), fill_value=np.nan)]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], np.full(shape=(nID, EventWindow), fill_value=np.nan)]
            self._Output["异常协方差"] = np.r_[self._Output["异常协方差"], np.full(shape=(nID, EventWindow, EventWindow), fill_value=np.nan)]
            EstStartInd = CurInd - self.EventPreWindow - self.EstWindow - 1
            Price = self._FactorTable.readData(dts=self._AllDTs[EstStartInd:CurInd+1], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
            Return = _calcReturn(Price.values, return_type=self.ReturnType)
            BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=self._AllDTs[EstStartInd:CurInd+1]).iloc[0, :, :]
            BReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(nID, axis=1)
            EstReturn = Return[:self.EstWindow]
            if self.EstSampleFilter:
                temp = self._FactorTable.readData(dts=self._AllDTs[EstStartInd+1:EstStartInd+self.EstWindow+1], ids=IDs, factor_names=self._FilterFactors)
                FilterMask = eval(self._CompiledIDFilterStr).values
            else:
                FilterMask = np.full(EstReturn.shape, fill_value=True)
            FilterMask = (FilterMask & pd.notnull(EstReturn) & pd.notnull(BReturn[:self.EstWindow]))
            FilterMask = (FilterMask & (np.flipud(np.cumsum(np.flipud(FilterMask), axis=0))<=self.EstSampleLen))
            EstReturn[~FilterMask] = np.nan
            Var = np.nanvar(EstReturn - BReturn[:self.EstWindow], axis=0, ddof=1)
            FilterMask = ((np.sum(FilterMask, axis=0)<self.EstSampleLen) | (Var<1e-6))
            Var[FilterMask] = np.nan
            self._Output["正常收益率"][-nID:, :self.EventPreWindow+1] = BReturn[self.EstWindow:].T
            self._Output["异常收益率"][-nID:, :self.EventPreWindow+1] = (Return[self.EstWindow:] - BReturn[self.EstWindow:]).T
            CovMatrix = np.eye(EventWindow).reshape((1, EventWindow, EventWindow)).repeat(nID, axis=0)
            self._Output["异常协方差"][-nID:, :, :] = (CovMatrix.T*Var).T
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        if np.sum(Mask)==0: return 0
        IDs = self._Output["事件记录"][:, 0][Mask]
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=[self._AllDTs[CurInd-1], idt]).iloc[0, :, 0]
        BReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(len(IDs), axis=0)
        self._Output["正常收益率"][RowPos, ColPos] = BReturn
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        self._Output["异常收益率"][RowPos, ColPos] = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - BReturn)
        return 0

class MM(CMRM):
    """市场模型"""
    #BenchmarkPrice = Enum(None, arg_type="SingleOption", label="基准价格", order=8)
    BenchmarkID = Str(arg_type="String", label="基准ID", order=9)
    RiskFreeRate = Enum(None, arg_type="SingleOption", label="无风险利率", order=10)
    RiskFreeRateID = Str(arg_type="String", label="无风险利率ID", order=11)
    def __init__(self, factor_table, benchmark_ft, rate_table=None, name="市场模型", sys_args={}, **kwargs):
        self._BenchmarkFT = benchmark_ft# 提供基准数据的因子表
        self._RateFT = rate_table# 提供无风险利率的因子表
        return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._BenchmarkFT.getFactorMetaData(key="DataType")))
        self.add_trait("BenchmarkPrice", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="基准价格", order=6))
        self.BenchmarkPrice = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.BenchmarkID = self._BenchmarkFT.getID(ifactor_name=self.BenchmarkPrice)[0]
        if self._RateFT is not None:
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._RateFT.getFactorMetaData(key="DataType")))
            self.add_trait("RiskFreeRate", Enum(None, *DefaultNumFactorList, arg_type="SingleOption", label="无风险利率", order=8))
        return super().__QS_initArgs__()
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output["市场超额收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["Beta"] = np.zeros(shape=(0,))
        self._Output["Alpha"] = np.zeros(shape=(0,))
        self._Output["Var"] = np.zeros(shape=(0,))
        if self._RateFT is None: return Rslt + (self._BenchmarkFT, )
        else: return Rslt + (self._BenchmarkFT, self._RateFT)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow+self.EstWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID, EventWindow = len(IDs), self.EventPreWindow+1+self.EventPostWindow
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            self._Output["正常收益率"] = np.r_[self._Output["正常收益率"], np.full(shape=(nID, EventWindow), fill_value=np.nan)]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], np.full(shape=(nID, EventWindow), fill_value=np.nan)]
            self._Output["异常协方差"] = np.r_[self._Output["异常协方差"], np.full(shape=(nID, EventWindow, EventWindow), fill_value=np.nan)]
            self._Output["市场超额收益率"] = np.r_[self._Output["异常收益率"], np.full(shape=(nID, EventWindow), fill_value=np.nan)]
            EstStartInd = CurInd - self.EventPreWindow - self.EstWindow - 1
            Price = self._FactorTable.readData(dts=self._AllDTs[EstStartInd:CurInd+1], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
            Return = _calcReturn(Price.values, return_type=self.ReturnType)
            BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=self._AllDTs[EstStartInd:CurInd+1]).iloc[0, :, :]
            BReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(nID, axis=1)
            if (self._RateFT is not None) and bool(self.RiskFreeRate) and bool(self.RiskFreeRateID):
                RFRate = self._RateFT.readData(factor_names=[self.RiskFreeRate], ids=[self.RiskFreeRateID], dts=self._AllDTs[EstStartInd+1:CurInd+1]).iloc[0, :, :].values.repeat(nID, axis=1)
                Return -= RFRate
                BReturn -= RFRate
                RFRate = RFRate[self.EstWindow:]
            else:
                RFRate = 0.0
            EstReturn = Return[:self.EstWindow]
            if self.EstSampleFilter:
                temp = self._FactorTable.readData(dts=self._AllDTs[EstStartInd+1:EstStartInd+self.EstWindow+1], ids=IDs, factor_names=self._FilterFactors)
                FilterMask = eval(self._CompiledIDFilterStr).values
            else:
                FilterMask = np.full(EstReturn.shape, fill_value=True)
            FilterMask = (FilterMask & pd.notnull(EstReturn) & pd.notnull(BReturn[:self.EstWindow]))
            FilterMask = (FilterMask & (np.flipud(np.cumsum(np.flipud(FilterMask), axis=0))<=self.EstSampleLen))
            EstReturn[~FilterMask] = np.nan
            Alpha, Beta, Var = np.full(shape=(nID,), fill_value=np.nan), np.full(shape=(nID,), fill_value=np.nan), np.full(shape=(nID,), fill_value=np.nan)
            for i, iID in enumerate(IDs):
                iX = sm.add_constant(BReturn[:self.EstWindow, i], prepend=True)
                try:
                    iRslt = sm.OLS(EstReturn[:, i], iX, missing="drop").fit()
                    Alpha[i], Beta[i] = iRslt.params
                    Var[i] = iRslt.mse_resid
                except:
                    pass
            self._Output["Alpha"] = np.r_[self._Output["Alpha"], Alpha]
            self._Output["Beta"] = np.r_[self._Output["Beta"], Beta]
            self._Output["Var"] = np.r_[self._Output["Var"], Var]
            ExpectedReturn = Alpha + np.dot(BReturn[self.EstWindow:, :1], Beta.reshape((1, nID)))
            self._Output["正常收益率"][-nID:, :self.EventPreWindow+1] = (ExpectedReturn + RFRate).T
            self._Output["异常收益率"][-nID:, :self.EventPreWindow+1] = (Return[self.EstWindow:] - ExpectedReturn).T
            self._Output["市场超额收益率"][-nID:, :self.EventPreWindow+1] = BReturn[self.EstWindow:].T
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        if np.sum(Mask)==0: return 0
        IDs = self._Output["事件记录"][:, 0][Mask]
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=[self._AllDTs[CurInd-1], idt]).iloc[0, :, 0]
        BReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(len(IDs), axis=0)
        if (self._RateFT is not None) and bool(self.RiskFreeRate) and bool(self.RiskFreeRateID):
            RFRate = self._RateFT.readData(factor_names=[self.RiskFreeRate], ids=[self.RiskFreeRateID], dts=[idt]).iloc[0, 0, 0]
            BReturn -= RFRate
        else:
            RFRate = 0.0
        Alpha, Beta = self._Output["Alpha"][RowPos], self._Output["Beta"][RowPos]
        ExpectedReturn = Alpha + Beta * BReturn + RFRate
        self._Output["正常收益率"][RowPos, ColPos] = ExpectedReturn
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        self._Output["异常收益率"][RowPos, ColPos] = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - ExpectedReturn)
        self._Output["市场超额收益率"][RowPos, ColPos] = BReturn
        for i in range(len(IDs)):
            if ColPos[i]<EventWindow-1: continue
            X = sm.add_constant(self._Output["市场超额收益率"][RowPos[i], :], prepend=True)
            self._Output["异常协方差"][RowPos[i], :, :] = (np.eye(EventWindow)+np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)) * self._Output["Var"][RowPos[i]]
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        if np.sum(Mask)>0:
            RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
            for i in range(RowPos.shape[0]):
                X = self._Output["市场超额收益率"][RowPos[i], :]
                iMask = pd.notnull(X)
                X = sm.add_constant(X[iMask], prepend=True)
                self._Output["异常协方差"][RowPos[i], iMask, iMask] = (np.eye(X.shape[0])+np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)) * self._Output["Var"][RowPos[i]]
        Index = pd.MultiIndex.from_arrays(self._Output["事件记录"][:,:2].T, names=["ID", "时点"])
        self._Output["回归估计量"] = pd.DataFrame(self._Output.pop("Alpha"), index=Index, columns=["Apha"])
        self._Output["回归估计量"]["Beta"] = self._Output.pop("Beta")
        self._Output["回归估计量"]["Sigma2"] = self._Output.pop("Var")
        self._Output.pop("市场超额收益率")
        return 0

class CBBM(CMRM):# TODO
    """特征基准模型"""
    #CharacteristicFactor = Enum(None, arg_type="SingleOption", label="特征因子", order=5)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=6)
    FilterRatio = Float(0.1, arg_type="Double", label="筛选比例", order=7)
    def __init__(self, factor_table, name="特征基准模型", sys_args={}, **kwargs):
        return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("CharacteristicFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="特征因子", order=5))
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        CurInd = self._AllDTs.index(idt)
        if CurInd<self.EventPreWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID = len(IDs)
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            PreInd = CurInd - self.EventPreWindow
            CIDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
            CFactor = self._FactorTable.readData(dts=[self._AllDTs[PreInd:CurInd+1]], ids=IDs, factor_names=[self.CharacteristicFactor]).iloc[0, :, :].values
            ExpectedReturn = self._FactorTable.readData(dts=[self._AllDTs[PreInd]], ids=IDs, factor_names=[self.ExpectedReturn]).iloc[0, :, :].values
            self._Output["正常收益率"] = np.r_[self._Output["正常收益率"], ExpectedReturn.T.repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)]
            if CurInd-PreInd>1:
                Price = self._FactorTable.readData(dts=self._AllDTs[PreInd:CurInd], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
                Return = _calcReturn(Price.values, return_type=self.ReturnType)
                AbnormalReturn = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
                AbnormalReturn[:, :self.EventPreWindow] = (Return - ExpectedReturn).T
                self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], AbnormalReturn]
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        IDs = self._Output["事件记录"][:, 0][Mask]
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        AbnormalReturn = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - self._Output["正常收益率"][Mask, 0])
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        self._Output["异常收益率"][RowPos, ColPos] = AbnormalReturn
        return 0