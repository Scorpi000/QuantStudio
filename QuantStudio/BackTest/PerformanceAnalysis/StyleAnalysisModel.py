# coding=utf-8
"""风格分析模型"""
import datetime as dt
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, Instance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.BackTest.SectionFactor.Portfolio import _QS_plotStatistics

# TODO
class ReturnBasedStyleAnalysisModel(BaseModule):
    """基于收益率回归的风格分析模型"""
    #TargetNAV = Enum(None, arg_type="SingleOption", label="目标净值", order=0)
    #StyleNAV = Enum(None, arg_type="SingleOption", label="风格净值", order=1)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=2)
    LookBack = Int(240, arg_type="Integer", label="回溯期数", order=3)
    def __init__(self, target_table, style_table, name="基于收益率回归的风格分析模型", sys_args={}, **kwargs):
        self._TargetTable = target_table
        self._StyleTable = style_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._TargetTable.getFactorMetaData(key="DataType")))
        self.add_trait("TargetNAV", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="目标净值", order=0))
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._StyleTable.getFactorMetaData(key="DataType")))
        self.add_trait("StyleNAV", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="风格净值", order=1))
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        return (self._TargetTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        return 0