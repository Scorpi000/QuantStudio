# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, on_trait_change, Float
from traitsui.api import SetEditor, Item
from scipy.stats import t
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.BackTest.BackTestModel import BaseModule

def _calcReturn(price, return_type="简单收益率"):
    if return_type=="对数收益率":
        Return = np.log(1 + np.diff(price, axis=0) / np.abs(price0[:-1]))
        Return[np.isinf(Return)] = np.nan
        return Return
    elif return_type=="价格变化量": return np.diff(price, axis=0)
    else: return np.diff(price, axis=0) / np.abs(price[:-1])

class CMRM(BaseModule):
    """均值常数模型"""
    EventFilter = Str(arg_type="String", label="事件定义", order=0)
    EventPreWindow = Int(20, arg_type="Integer", label="事件前窗长", order=1)
    EventPostWindow = Int(20, arg_type="Integer", label="事件后窗长", order=2)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=4)
    #ExpectedReturn = Enum(None, arg_type="SingleOption", label="预期收益率", order=5)
    def __init__(self, factor_table, name="均值常数模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("ExpectedReturn", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="预期收益率", order=5))
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["预期收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["异常收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["事件记录"] = np.full(shape=(0, 3), fill_value=None, dtype="O")# [ID, 时点, 事件后期数]
        self._AllDTs = self._FactorTable.getDateTime()
        if not self._AllDTs: self._AllDTs = dts
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID = len(IDs)
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            ExpectedReturn = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
            self._Output["预期收益率"] = np.r_[self._Output["预期收益率"], ExpectedReturn]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], ExpectedReturn]
            PreInd = CurInd - self.EventPreWindow - 1
            ExpectedReturn = self._FactorTable.readData(dts=[self._AllDTs[PreInd]], ids=IDs, factor_names=[self.ExpectedReturn]).iloc[0, :, :].values
            self._Output["预期收益率"][-nID:, :] = ExpectedReturn.T.repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)
            if CurInd-PreInd>1:
                Price = self._FactorTable.readData(dts=self._AllDTs[PreInd:CurInd], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
                Return = _calcReturn(Price.values, return_type=self.ReturnType)
                self._Output["异常收益率"][-nID:, :CurInd-PreInd-1] = (Return - ExpectedReturn).T
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        IDs = self._Output["事件记录"][:, 0][Mask]
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        AbnormalReturn = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - self._Output["预期收益率"][Mask, 0])
        self._Output["异常收益率"][RowPos, ColPos] = AbnormalReturn
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        Index = pd.MultiIndex.from_arrays(self._Output.pop("事件记录")[:,:2].T, names=["ID", "时点"])
        self._Output["预期收益率"] = pd.DataFrame(self._Output["预期收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index)
        self._Output["异常收益率"] = pd.DataFrame(self._Output["异常收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index)
        AR_Avg, AR_Std, AR_N = self._Output["异常收益率"].mean(axis=0), self._Output["异常收益率"].std(axis=0, ddof=1), self._Output["异常收益率"].count(axis=0)
        CAR = self._Output["异常收益率"].cumsum(axis=1, skipna=True)
        CAR_Avg, CAR_Std, CAR_N = CAR.mean(axis=0), CAR.std(axis=0), CAR.count(axis=0)
        StageCAR = CAR.copy()
        StageCAR.iloc[:, self.EventPreWindow+1:] = (StageCAR.iloc[:, self.EventPreWindow+1:].T - StageCAR.iloc[:, self.EventPreWindow]).T
        StageCAR_Avg, StageCAR_Std, StageCAR_N = StageCAR.mean(axis=0), StageCAR.std(axis=0), StageCAR.count(axis=0)
        self._Output["t检验"] = pd.DataFrame(AR_Avg, columns=["平均异常收益率"])
        self._Output["t检验"]["平均累积异常收益率"] = CAR_Avg
        self._Output["t检验"]["平均分段累积异常收益率"] = StageCAR_Avg
        self._Output["t检验"]["异常收益t统计量"] = AR_Avg / AR_Std * AR_N**0.5
        self._Output["t检验"]["累积异常收益t统计量"] = CAR_Avg / CAR_Std * CAR_N**0.5
        self._Output["t检验"]["分段累积异常收益t统计量"] = StageCAR_Avg / StageCAR_Std * StageCAR_N**0.5
        self._Output["t检验"]["异常收益p值"] = 2 * t.sf(np.abs(self._Output["t检验"]["异常收益t统计量"].values), AR_N.values - 1)
        self._Output["t检验"]["累积异常收益p值"] = 2 * t.sf(np.abs(self._Output["t检验"]["累积异常收益t统计量"].values), CAR_N.values - 1)
        self._Output["t检验"]["分段累积异常收益p值"] = 2 * t.sf(np.abs(self._Output["t检验"]["分段累积异常收益t统计量"].values), StageCAR_N.values - 1)
        self._Output["预期收益率"], self._Output["异常收益率"] = self._Output["预期收益率"].reset_index(), self._Output["异常收益率"].reset_index()
        return 0

class MAM(CMRM):
    """市场调整模型"""
    #BenchmarkPrice = Enum(None, arg_type="SingleOption", label="基准价格", order=5)
    BenchmarkID = Str(arg_type="String", label="基准ID", order=6)
    def __init__(self, factor_table, benchmark_ft, name="市场调整模型", sys_args={}, **kwargs):
        self._BenchmarkFT = benchmark_ft
        return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._BenchmarkFT.getFactorMetaData(key="DataType")))
        self.add_trait("BenchmarkPrice", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="基准价格", order=5))
        self.BenchmarkPrice = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.BenchmarkID = self._BenchmarkFT.getID(ifactor_name=self.BenchmarkPrice)[0]
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        return Rslt + (self._BenchmarkFT, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        CurInd = self._AllDTs.index(idt)
        if CurInd<=self.EventPreWindow: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID = len(IDs)
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            ExpectedReturn = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
            self._Output["预期收益率"] = np.r_[self._Output["预期收益率"], ExpectedReturn]
            self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], ExpectedReturn]
            PreInd = CurInd - self.EventPreWindow - 1
            if CurInd-PreInd>1:
                BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=self._AllDTs[PreInd:CurInd]).iloc[0, :, :]
                ExpectedReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(nID, axis=1).T
                self._Output["预期收益率"][-nID:, :CurInd-PreInd-1] = ExpectedReturn
                Price = self._FactorTable.readData(dts=self._AllDTs[PreInd:CurInd], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
                Return = _calcReturn(Price.values, return_type=self.ReturnType).T
                self._Output["异常收益率"][-nID:, :CurInd-PreInd-1] = Return - ExpectedReturn
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        IDs = self._Output["事件记录"][:, 0][Mask]
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        BPrice = self._BenchmarkFT.readData(factor_names=[self.BenchmarkPrice], ids=[self.BenchmarkID], dts=[self._AllDTs[CurInd-1], idt]).iloc[0, :, 0]
        ExpectedReturn = _calcReturn(BPrice.values, return_type=self.ReturnType).repeat(len(IDs), axis=0)
        self._Output["预期收益率"][RowPos, ColPos] = ExpectedReturn
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        AbnormalReturn = (_calcReturn(Price.values, return_type=self.ReturnType)[0] - ExpectedReturn)
        self._Output["异常收益率"][RowPos, ColPos] = AbnormalReturn
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