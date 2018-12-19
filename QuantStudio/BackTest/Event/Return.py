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
    EventPreWindow = Int(20, arg_type="Integer", label="事件前窗长", order=3)
    EventPostWindow = Int(20, arg_type="Integer", label="事件后窗长", order=4)
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
        self._Output["正常收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["异常收益率"] = np.zeros(shape=(0, self.EventPreWindow+1+self.EventPostWindow))
        self._Output["事件记录"] = np.full(shape=(0, 3), fill_value=None, dtype="O")# [ID, 时点, 事件后期数]
        self._AllDTs = self._FactorTable.getDateTime()
        if not self._AllDTs: self._AllDTs = dts
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        CurInd = self._AllDTs.index(idt)
        if CurInd==0: return 0
        self._Output["事件记录"][:, 2] += 1
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.EventFilter)
        nID = len(IDs)
        if nID>0:
            self._Output["事件记录"] = np.r_[self._Output["事件记录"], np.c_[IDs, [idt]*nID, np.zeros(shape=(nID, 1))]]
            PreInd = max(0, CurInd - self.EventPreWindow - 1)
            NormalReturn = self._FactorTable.readData(dts=[self._AllDTs[PreInd]], ids=IDs, factor_names=[self.NormalReturn]).iloc[0, :, :]
            self._Output["正常收益率"] = np.r_[self._Output["正常收益率"], NormalReturn.values.T.repeat(self.EventPreWindow+1+self.EventPostWindow, axis=1)]
            if CurInd-PreInd>1:
                Price = self._FactorTable.readData(dts=self._AllDTs[PreInd:CurInd], ids=IDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
                AbnormalReturn = np.full(shape=(nID, self.EventPreWindow+1+self.EventPostWindow), fill_value=np.nan)
                AbnormalReturn[:, self.EventPreWindow-Price.shape[0]+1:self.EventPreWindow] = (Price.iloc[1:].values / Price.iloc[:-1].values - 1 - NormalReturn.values).T
                self._Output["异常收益率"] = np.r_[self._Output["异常收益率"], AbnormalReturn]
        Mask = (self._Output["事件记录"][:, 2]<=self.EventPostWindow)
        IDs = self._Output["事件记录"][:, 0][Mask]
        Price = self._FactorTable.readData(dts=[self._AllDTs[CurInd-1], idt], ids=sorted(set(IDs)), factor_names=[self.PriceFactor]).iloc[0, :, :].loc[:, IDs]
        AbnormalReturn = (Price.iloc[1].values / Price.iloc[0].values - 1 - self._Output["正常收益率"][Mask, 0])
        RowPos, ColPos = np.arange(self._Output["异常收益率"].shape[0])[Mask].tolist(), (self._Output["事件记录"][Mask, 2]+self.EventPreWindow).astype(np.int)
        self._Output["异常收益率"][RowPos, ColPos] = AbnormalReturn
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        Index = pd.MultiIndex.from_arrays(self._Output.pop("事件记录")[:,:2].T, names=["ID", "时点"])
        self._Output["正常收益率"] = pd.DataFrame(self._Output["正常收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index)
        self._Output["异常收益率"] = pd.DataFrame(self._Output["异常收益率"], columns=np.arange(-self.EventPreWindow, 1+self.EventPostWindow), index=Index)
        self._Output["统计数据"] = pd.DataFrame(self._Output["异常收益率"].mean(), columns=["异常收益率"])
        self._Output["统计数据"]["累积异常收益率"] = self._Output["异常收益率"].cumsum(axis=1).mean()
        self._Output["统计数据"]["分段累积异常收益率"] = self._Output["统计数据"]["累积异常收益率"]
        self._Output["统计数据"]["分段累积异常收益率"].iloc[self.EventPreWindow+1:] -= self._Output["统计数据"]["分段累积异常收益率"].iloc[self.EventPreWindow]
        self._Output["正常收益率"], self._Output["异常收益率"] = self._Output["正常收益率"].reset_index(), self._Output["异常收益率"].reset_index()
        return 0