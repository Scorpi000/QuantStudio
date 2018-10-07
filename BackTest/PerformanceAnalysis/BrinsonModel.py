# coding=utf-8
"""Brinson 绩效分析模型"""
import datetime as dt
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, Instance, on_trait_change
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.BackTest.SectionFactor.Portfolio import _QS_plotStatistics


class BrinsonModel(BaseModule):
    """Brinson 绩效分析模型"""
    Portfolio = Enum(None, arg_type="SingleOption", label="策略组合", order=0)
    BenchmarkPortfolio = Enum(None, arg_type="SingleOption", label="基准组合", order=1)
    GroupFactor = Enum(None, arg_type="SingleOption", label="资产类别", order=2)
    PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=4)
    def __init__(self, factor_table, name="Brinson绩效分析模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("Portfolio", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="策略组合", order=0))
        self.add_trait("BenchmarkPortfolio", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="基准组合", order=1))
        self.add_trait("GroupFactor", Enum(*DefaultStrFactorList, arg_type="SingleOption", label="资产类别", order=2))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def __QS_start__(self, mdl, dts, **kwargs):
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["业绩基准"] = pd.DataFrame(0.0, columns=self.AttributeFactors)
        self._Output["主动资产配置"] = pd.DataFrame(0.0, columns=self.AttributeFactors)
        self._Output["主动个券选择"] = pd.DataFrame(0.0, columns=self.AttributeFactors)
        self._Output["策略组合"] = pd.DataFrame(0.0, columns=self.AttributeFactors)
        self._CurCalcInd = 0
        self._IDs = self._FactorTable.getID()
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        return 0
    def __QS_end__(self):
        return 0