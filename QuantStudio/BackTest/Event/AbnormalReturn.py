# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, on_trait_change, Float
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

def _calcReturn(price, return_type="简单收益率"):
    if return_type=="对数收益率":
        Return = np.log(1 + np.diff(price, axis=0) / np.abs(price[:-1]))
        Return[np.isinf(Return)] = np.nan
        return Return
    elif return_type=="价格变化量": return np.diff(price, axis=0)
    else: return np.diff(price, axis=0) / np.abs(price[:-1])

class CMRM(BaseModule):
    """均值常数模型"""
    EventFilter = Str(arg_type="String", label="事件定义", order=0)
    EventPreWindow = Int(20, arg_type="Integer", label="事件前窗口", order=1)
    EventPostWindow = Int(20, arg_type="Integer", label="事件后窗口", order=2)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    ReturnType = Enum("对数收益率", "简单收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=4)
    EstWindow = Int(60, arg_type="Integer", label="估计窗口", order=5)
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
        self._Output["预期收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["异常收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["异常方差"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["事件记录"] = np.full(shape=(0, 3), fill_value=None, dtype="O")# [ID, 时点, 事件后期数]
        self._AllDTs = self._FactorTable.getDateTime()
        if not self._AllDTs: self._AllDTs = dts
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow+self.EstWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID = len(IDs)
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            Temp = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
            self._Output["预期收益率"] = np.r_[self._Output["预期收益率"], Temp]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], Temp]
            self._Output["异常方差"] = np.r_[self._Output["异常方差"], Temp]
            EstStartInd = CurInd - self.EventPreWindow - self.EstWindow - 1
            Price = self._FactorTable.readData(dts=self._AllDTs[EstStartInd:CurInd+1], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
            Return = _calcReturn(Price.values, return_type=self.ReturnType)
            ExpectedReturn, Var = np.nanmean(Return[:self.EstWindow], axis=0), np.nanvar(Return[:self.EstWindow], axis=0, ddof=1)
            self._Output["预期收益率"][-nID:, :] = ExpectedReturn.reshape((nID, 1)).repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)
            self._Output["异常收益率"][-nID:, :self.EventPreWindow+1] = (Return[self.EstWindow:] - ExpectedReturn).T
            self._Output["异常方差"][-nID:, :] = Var.reshape((nID, 1)).repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        IDs = self._Output["事件记录"][:, 0][Mask]
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        self._Output["异常收益率"][RowPos, ColPos] = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - self._Output["预期收益率"][RowPos, ColPos])
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        Index = pd.MultiIndex.from_arrays(self._Output.pop("事件记录")[:,:2].T, names=["ID", "时点"])
        self._Output["预期收益率"] = pd.DataFrame(self._Output["预期收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index).reset_index()
        self._Output["异常收益率"] = pd.DataFrame(self._Output["异常收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index)
        self._Output["异常方差"] = pd.DataFrame(self._Output["异常方差"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index)
        NaMask = (self._Output["异常方差"]<1e-6)
        self._Output["异常收益率"][NaMask] = np.nan
        self._Output["异常方差"][NaMask] = np.nan
        AR, AR_Var = self._Output["异常收益率"], self._Output["异常方差"]
        AR_Avg, AR_Avg_Var = AR.mean(axis=0), AR_Var.mean(axis=0)
        CAR, CAR_Var = AR.cumsum(axis=1, skipna=True), AR_Var.cumsum(axis=1, skipna=True)
        CAR_Avg, CAR_Avg_Var = CAR.mean(axis=0), CAR_Var.mean(axis=0)
        StageCAR = pd.merge(AR.loc[:, reversed(AR.columns[:self.EventPreWindow+1])].cumsum(axis=1).loc[:, AR.columns[:self.EventPreWindow]], AR.loc[:, AR.columns[self.EventPreWindow:]].cumsum(axis=1), left_index=True, right_index=True)
        StageCAR_Var = pd.merge(AR_Var.loc[:, reversed(AR_Var.columns[:self.EventPreWindow+1])].cumsum(axis=1).loc[:, AR_Var.columns[:self.EventPreWindow]], AR_Var.loc[:, AR_Var.columns[self.EventPreWindow:]].cumsum(axis=1), left_index=True, right_index=True)
        StageCAR_Avg, StageCAR_Avg_Var = StageCAR.mean(axis=0), StageCAR_Var.mean(axis=0)
        self._Output["J1统计量"] = pd.DataFrame(AR_Avg, columns=["异常收益率"])
        self._Output["J1统计量"]["累积异常收益率"] = CAR_Avg
        self._Output["J1统计量"]["前后累积异常收益率"] = StageCAR_Avg
        self._Output["J1统计量"]["异常收益统计量"] = AR_Avg / AR_Avg_Var**0.5
        self._Output["J1统计量"]["累积异常收益统计量"] = CAR_Avg / CAR_Avg_Var**0.5
        self._Output["J1统计量"]["前后累积异常收益统计量"] = StageCAR_Avg / StageCAR_Avg_Var**0.5
        self._Output["J1统计量"]["异常收益p值"] = 2 * norm.sf(np.abs(self._Output["J1统计量"]["异常收益统计量"].values))
        self._Output["J1统计量"]["累积异常收益p值"] = 2 * norm.sf(np.abs(self._Output["J1统计量"]["累积异常收益统计量"].values))
        self._Output["J1统计量"]["前后累积异常收益p值"] = 2 * norm.sf(np.abs(self._Output["J1统计量"]["前后累积异常收益统计量"].values))
        SAR, SCAR, SStageCAR = AR / AR_Var**0.5, CAR / CAR_Var**0.5, StageCAR / StageCAR_Var**0.5
        SAR, SCAR, SStageCAR = SAR.where((~np.isinf(SAR)), np.nan), SCAR.where((~np.isinf(SCAR)), np.nan), SStageCAR.where((~np.isinf(SStageCAR)), np.nan)
        SAR_Avg, SCAR_Avg, SStageCAR_Avg = SAR.mean(axis=0), SCAR.mean(axis=0), SStageCAR.mean(axis=0)
        self._Output["J2统计量"] = pd.DataFrame(SAR_Avg, columns=["标准异常收益率"])
        self._Output["J2统计量"]["标准累积异常收益率"] = SCAR_Avg
        self._Output["J2统计量"]["标准前后累积异常收益率"] = SStageCAR_Avg
        self._Output["J2统计量"]["异常收益统计量"] = SAR_Avg * (SAR.count(axis=0) * (self.EstWindow - 4) / (self.EstWindow - 2))**0.5
        self._Output["J2统计量"]["累积异常收益统计量"] = SCAR_Avg * (SCAR.count(axis=0) * (self.EstWindow - 4) / (self.EstWindow - 2))**0.5
        self._Output["J2统计量"]["前后累积异常收益统计量"] = SStageCAR_Avg * (SStageCAR.count(axis=0) * (self.EstWindow - 4) / (self.EstWindow - 2))**0.5
        self._Output["J2统计量"]["异常收益p值"] = 2 * norm.sf(np.abs(self._Output["J2统计量"]["异常收益统计量"].values))
        self._Output["J2统计量"]["累积异常收益p值"] = 2 * norm.sf(np.abs(self._Output["J2统计量"]["累积异常收益统计量"].values))
        self._Output["J2统计量"]["前后累积异常收益p值"] = 2 * norm.sf(np.abs(self._Output["J2统计量"]["前后累积异常收益统计量"].values))
        self._Output["异常收益率"], self._Output["异常方差"] = self._Output["异常收益率"].reset_index(), self._Output["异常方差"].reset_index()
        return 0

class MAM(CMRM):
    """市场调整模型"""
    #BenchmarkPrice = Enum(None, arg_type="SingleOption", label="基准价格", order=6)
    BenchmarkID = Str(arg_type="String", label="基准ID", order=7)
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
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        return Rslt + (self._BenchmarkFT, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow+self.EstWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID = len(IDs)
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            Temp = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
            self._Output["预期收益率"] = np.r_[self._Output["预期收益率"], Temp]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], Temp]
            self._Output["异常方差"] = np.r_[self._Output["异常方差"], Temp]
            EstStartInd = CurInd - self.EventPreWindow - self.EstWindow - 1
            Price = self._FactorTable.readData(dts=self._AllDTs[EstStartInd:CurInd+1], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
            Return = _calcReturn(Price.values, return_type=self.ReturnType)
            BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=self._AllDTs[EstStartInd:CurInd+1]).iloc[0, :, :]
            ExpectedReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(nID, axis=1)
            self._Output["预期收益率"][-nID:, :self.EventPreWindow+1] = ExpectedReturn[self.EstWindow:].T
            self._Output["异常收益率"][-nID:, :self.EventPreWindow+1] = (Return[self.EstWindow:] - ExpectedReturn[self.EstWindow:]).T
            self._Output["异常方差"][-nID:, :] = np.nanvar(Return[:self.EstWindow] - ExpectedReturn[:self.EstWindow], axis=0, ddof=1).reshape((nID, 1)).repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        IDs = self._Output["事件记录"][:, 0][Mask]
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=[self._AllDTs[CurInd-1], idt]).iloc[0, :, 0]
        ExpectedReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(len(IDs), axis=0)
        self._Output["预期收益率"][RowPos, ColPos] = ExpectedReturn
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        self._Output["异常收益率"][RowPos, ColPos] = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - ExpectedReturn)
        return 0

class MM(CMRM):
    """市场模型"""
    #BenchmarkPrice = Enum(None, arg_type="SingleOption", label="基准价格", order=6)
    BenchmarkID = Str(arg_type="String", label="基准ID", order=7)
    RiskFreeRate = Enum(None, arg_type="SingleOption", label="无风险利率", order=8)
    RiskFreeRateID = Str(arg_type="String", label="无风险利率ID", order=9)
    HalfLife = Float(np.inf, arg_type="Integer", label="半衰期", order=10)
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
        self._Output["Beta"] = np.zeros(shape=(0))
        self._Output["Alpha"] = np.zeros(shape=(0))
        if self._RateFT is None: return Rslt + (self._BenchmarkFT, )
        else: return Rslt + (self._BenchmarkFT, self._RateFT)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow+self.EstWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID = len(IDs)
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            Temp = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
            self._Output["预期收益率"] = np.r_[self._Output["预期收益率"], Temp]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], Temp]
            self._Output["异常方差"] = np.r_[self._Output["异常方差"], Temp]
            EstStartInd = CurInd - self.EventPreWindow - self.EstWindow - 1
            Price = self._FactorTable.readData(dts=self._AllDTs[EstStartInd:CurInd+1], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
            Return = _calcReturn(Price.values, return_type=self.ReturnType)
            BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=self._AllDTs[EstStartInd:CurInd+1]).iloc[0, :, 0]
            BReturn = _calcReturn(BPrice.values, return_type=self.ReturnType)
            if (self._RateFT is not None) and bool(self.RiskFreeRate) and bool(self.RiskFreeRateID):
                RFRate = self._RateFT.readData(factor_names=[self.RiskFreeRate], ids=[self.RiskFreeRateID], dts=self._AllDTs[EstStartInd+1:CurInd+1]).iloc[0, :, 0]
                Return = Return.sub(RFRate, axis="index")
                BReturn -= RFRate
            else:
                RFRate = 0.0
            Weight = (0.5**(1/self.HalfLife))**np.arange(self.EstWindow)
            Weight = Weight[::-1] / np.sum(Weight)
            X = sm.add_constant(BReturn[:self.EstWindow], prepend=True)
            Beta = Alpha = Var = np.full(shape=(nID,), fill_value=np.nan)
            for i, iID in enumerate(IDs):
                try:
                    iRslt = sm.WLS(Return[:self.EstWindow, i], X, weights=Weight, missing="drop").fit()
                    Alpha[i], Beta[i] = iRslt.params
                    Var[i] = iRslt.mse_resid
                except:
                    pass
            self._Output["Beta"] = np.r_[self._Output["Beta"], Beta]
            self._Output["Alpha"] = np.r_[self._Output["Alpha"], Alpha]
            ExpectedReturn = Alpha + np.dot(BReturn[self.EstWindow:].reshape((self.EventPreWindow+1, 1)), Beta.reshape((1, nID)))
            self._Output["预期收益率"][-nID:, :self.EventPreWindow+1] = ExpectedReturn.T + RFRate
            self._Output["异常收益率"][-nID:, :self.EventPreWindow+1] = (Return[self.EstWindow:] - ExpectedReturn).T
            self._Output["异常方差"][-nID:, :] = Var.reshape((nID, 1)).repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
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
        self._Output["预期收益率"][RowPos, ColPos] = ExpectedReturn
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        self._Output["异常收益率"][RowPos, ColPos] = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - ExpectedReturn)
        return 0
    def __QS_end__(self):
        self._Output.pop("Beta")
        self._Output.pop("Alpha")
        return super().__QS_end__()

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
            self._Output["预期收益率"] = np.r_[self._Output["预期收益率"], ExpectedReturn.T.repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)]
            if CurInd-PreInd>1:
                Price = self._FactorTable.readData(dts=self._AllDTs[PreInd:CurInd], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
                Return = _calcReturn(Price.values, return_type=self.ReturnType)
                AbnormalReturn = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
                AbnormalReturn[:, :self.EventPreWindow] = (Return - ExpectedReturn).T
                self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], AbnormalReturn]
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        IDs = self._Output["事件记录"][:, 0][Mask]
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        AbnormalReturn = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - self._Output["预期收益率"][Mask, 0])
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        self._Output["异常收益率"][RowPos, ColPos] = AbnormalReturn
        return 0