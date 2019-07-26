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

def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x*100, )
def _QS_formatPandasPercentage(x):
    return '{0:.2f}%'.format(x*100)
class BiasTest(BaseModule):
    """BiasTest"""
    def __init__(self, factor_table, name="BiasTest", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=3))
        self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=4))
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        return 0