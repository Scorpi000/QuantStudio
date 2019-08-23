# coding=utf-8
import datetime as dt
from copy import deepcopy
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from traits.api import Enum, List, Int, Str, Dict, Float
from traitsui.api import SetEditor, Item
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.BackTest.BackTestModel import BaseModule

class Cointegration(BaseModule):
    """协整检验"""
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=0)
    PriceType = Enum("原始价格", "对数价格", arg_type="SingleOption", label="价格类型", order=1)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=2)
    SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=3)
    MinSummaryWindow = Int(120, arg_type="Integer", label="最小统计窗口", order=4)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=5)
    CointArgs = Dict(arg_type="Dict", label="检验参数", order=6)
    def __init__(self, factor_table, name="协整检验", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=0))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        self._IDs = self._FactorTable.getID()
        self._Output = {}
        self._Output["价格"] = np.zeros(shape=(0, len(self._IDs)))
        self._Output["统计量"] = {}# {时点: DataFrame(index=[ID], columns=[ID])}
        self._Output["p值"] = {}# {时点: DataFrame(index=[ID], columns=[ID])}
        self._CurCalcInd = 0
        return super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        Price = self._FactorTable.readData(dts=[idt], ids=self._IDs, factor_names=[self.PriceFactor]).iloc[0, :, :].values
        if self.PriceType=="对数价格":
            Price = np.log(Price)
            Price[np.isinf(Price)] = np.nan
        self._Output["价格"] = np.r_[self._Output["价格"], Price]
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        StartInd = int(max(0, self._Output["价格"].shape[0] - self.SummaryWindow))
        if self._Output["价格"].shape[0] - StartInd < self.MinSummaryWindow: return 0
        Price, nID = self._Output["价格"][StartInd:], self._Output["价格"].shape[1]
        IDMask = self._FactorTable.getIDMask(idt=idt, ids=self._IDs, id_filter_str=self.IDFilter).values
        Price = Price[:, IDMask]
        Mask = pd.notnull(Price)
        Statistics, pValue = np.full(shape=(Price.shape[1], Price.shape[1]), fill_value=np.nan), np.full(shape=(Price.shape[1], Price.shape[1]), fill_value=np.nan)
        for i in range(Price.shape[1]):
            for j in range(i+1, Price.shape[1]):
                ijMask = (Mask[:, i] & Mask[:, j])
                try:
                    iRslt = sm.tsa.stattools.coint(Price[:,i][ijMask], Price[:,j][ijMask], **self.CointArgs)
                    Statistics[i, j] = Statistics[j, i] = iRslt[0]
                    pValue[i, j] = pValue[j, i] = iRslt[1]
                except:
                    pass
        self._Output["统计量"][idt], self._Output["p值"][idt] = pd.DataFrame(index=self._IDs, columns=self._IDs), pd.DataFrame(index=self._IDs, columns=self._IDs)
        self._Output["统计量"][idt].iloc[IDMask, IDMask] = Statistics
        self._Output["p值"][idt].iloc[IDMask, IDMask] = pValue
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        DTs = sorted(self._Output["统计量"])
        self._Output["最后一期检验"] = {"统计量": self._Output["统计量"][DTs[-1]], "p值": self._Output["p值"][DTs[-1]]}
        Price = self._Output.pop("价格")
        if np.isinf(self.SummaryWindow) and (DTs[-1]==self._iDT) and (not self.IDFilter):
            self._Output["全样本检验"] = deepcopy(self._Output["最后一期检验"])
        else:
            Mask = pd.notnull(Price)
            Statistics, pValue = np.full(shape=(Price.shape[1], Price.shape[1]), fill_value=np.nan), np.full(shape=(Price.shape[1], Price.shape[1]), fill_value=np.nan)
            for i in range(Price.shape[1]):
                for j in range(i+1, Price.shape[1]):
                    ijMask = (Mask[:, i] & Mask[:, j])
                    try:
                        iRslt = sm.tsa.stattools.coint(Price[:,i][ijMask], Price[:,j][ijMask], **self.CointArgs)
                        Statistics[i, j] = Statistics[j, i] = iRslt[0]
                        pValue[i, j] = pValue[j, i] = iRslt[1]
                    except:
                        pass
            self._Output["全样本检验"] = {"统计量": pd.DataFrame(Statistics, index=self._IDs, columns=self._IDs), "p值": pd.DataFrame(pValue, index=self._IDs, columns=self._IDs)}
        self._Output["滚动检验"] = {"统计量": pd.Panel(self._Output.pop("统计量")).loc[DTs].swapaxes(0, 1).to_frame(filter_observations=False).reset_index(),
                                    "p值": pd.Panel(self._Output.pop("p值")).loc[DTs].swapaxes(0, 1).to_frame(filter_observations=False).reset_index()}
        Cols = self._Output["滚动检验"]["统计量"].columns.tolist()
        Cols[0], Cols[1] = "时点", "ID"
        self._Output["滚动检验"]["统计量"].columns = self._Output["滚动检验"]["p值"].columns = Cols
        return 0