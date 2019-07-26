# coding=utf-8
"""基于持仓数据的绩效分析模型"""
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
from QuantStudio.RiskDataBase.RiskDB import RiskTable
from QuantStudio.RiskModel.RiskModelFun import dropRiskMatrixNA
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.BackTest.SectionFactor.Portfolio import _QS_plotStatistics

class FMPModel(BaseModule):
    """基于特征因子模拟组合的绩效分析模型"""
    #Portfolio = Enum(None, arg_type="SingleOption", label="策略组合", order=0)
    #BenchmarkPortfolio = Enum("无", arg_type="SingleOption", label="基准组合", order=1)
    AttributeFactors = ListStr(arg_type="MultiOption", label="特征因子", order=2, option_range=())
    #IndustryFactor = Enum("无", arg_type="SingleOption", label="行业因子", order=3)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=4)
    RiskTable = Instance(RiskTable, arg_type="RiskTable", label="风险表", order=5)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=6)
    def __init__(self, factor_table, name="因子模拟组合绩效分析模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("Portfolio", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="策略组合", order=0))
        self.add_trait("BenchmarkPortfolio", Enum(*(["无"]+DefaultNumFactorList), arg_type="SingleOption", label="基准组合", order=1))
        self.add_trait("AttributeFactors", ListStr(arg_type="MultiOption", label="特征因子", order=2, option_range=tuple(DefaultNumFactorList)))
        self.AttributeFactors.append(DefaultNumFactorList[-1])
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=3))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=4))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def _normalizePortfolio(self, portfolio):
        NegMask = (portfolio<0)
        TotalNegWeight = portfolio[NegMask].sum()
        if TotalNegWeight!=0: portfolio[NegMask] = portfolio[NegMask] / TotalNegWeight
        PosMask = (portfolio>0)
        TotalPosWeight = portfolio[PosMask].sum()
        if TotalPosWeight!=0:
            portfolio[PosMask] = portfolio[PosMask] / TotalPosWeight
            portfolio[NegMask] = portfolio[NegMask] * TotalNegWeight / TotalPosWeight
        return portfolio
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self.RiskTable.start(dts=dts)
        self._Output = {}
        self._Output["因子暴露"] = pd.DataFrame(columns=self.AttributeFactors)
        self._Output["风险调整的因子暴露"] = pd.DataFrame(columns=self.AttributeFactors)
        self._Output["风险贡献"] = pd.DataFrame(columns=self.AttributeFactors+["Alpha"])
        self._Output["收益贡献"] = pd.DataFrame(columns=self.AttributeFactors+["Alpha"])
        self._Output["因子收益"] = pd.DataFrame(columns=self.AttributeFactors)
        self._CurCalcInd = 0
        self._IDs = self._FactorTable.getID()
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        PreDT = None
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            if self._CurCalcInd>0: PreDT = self.CalcDTs[self._CurCalcInd - 1]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            if self._CurCalcInd>0: PreDT = self._Model.DateTimeSeries[self._CurCalcInd - 1]
        if PreDT is None: return 0
        Portfolio = self._FactorTable.readData(factor_names=[self.Portfolio], dts=[PreDT], ids=self._IDs).iloc[0, 0]
        Portfolio = self._normalizePortfolio(Portfolio[pd.notnull(Portfolio) & (Portfolio!=0)])
        if self.BenchmarkPortfolio!="无":
            BenchmarkPortfolio = self._FactorTable.readData(factor_names=[self.BenchmarkPortfolio], dts=[PreDT], ids=self._IDs).iloc[0, 0]
            BenchmarkPortfolio = self._normalizePortfolio(BenchmarkPortfolio[pd.notnull(BenchmarkPortfolio) & (BenchmarkPortfolio!=0)])
            IDs = Portfolio.index.union(BenchmarkPortfolio.index)
            if Portfolio.shape[0]>0:
                Portfolio = Portfolio.loc[IDs]
                Portfolio.fillna(0.0, inplace=True)
            else: Portfolio = pd.Series(0.0, index=IDs)
            if BenchmarkPortfolio.shape[0]>0:
                BenchmarkPortfolio = BenchmarkPortfolio.loc[IDs]
                BenchmarkPortfolio.fillna(0.0, inplace=True)
            else:
                BenchmarkPortfolio = pd.Series(0.0, index=IDs)
            Portfolio = Portfolio - BenchmarkPortfolio
        # 计算因子模拟组合
        self.RiskTable.move(PreDT, **kwargs)
        CovMatrix = dropRiskMatrixNA(self.RiskTable.readCov(dts=[PreDT], ids=Portfolio.index.tolist()).iloc[0])
        FactorExpose = self._FactorTable.readData(factor_names=list(self.AttributeFactors), ids=IDs, dts=[PreDT]).iloc[:,0].dropna(axis=0)
        IDs = FactorExpose.index.intersection(CovMatrix.index).tolist()
        CovMatrix, FactorExpose = CovMatrix.loc[IDs, IDs], FactorExpose.loc[IDs, :]
        if self.IndustryFactor!="无":
            IndustryData = self._FactorTable.readData(factor_names=[self.IndustryFactor], ids=IDs, dts=[PreDT]).iloc[0, 0, :]
            DummyData = DummyVarTo01Var(IndustryData, ignore_nonstring=True)
            DummyData.columns.values[pd.isnull(DummyData.columns)] = "None"
            FactorExpose = pd.merge(FactorExpose, DummyData, left_index=True, right_index=True)
        CovMatrixInv = np.linalg.inv(CovMatrix.values)
        FMPHolding = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(FactorExpose.values.T, CovMatrixInv), FactorExpose.values)), FactorExpose.values.T), CovMatrixInv)
        # 计算持仓对因子模拟组合的投资组合
        Portfolio = self._normalizePortfolio(Portfolio.loc[IDs])
        Beta = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(FMPHolding, CovMatrix.values), FMPHolding.T)), FMPHolding), CovMatrix.values), Portfolio.values)
        Price = self._FactorTable.readData(factor_names=[self.PriceFactor], dts=[PreDT, idt], ids=IDs).iloc[0]
        Return = Price.iloc[1] / Price.iloc[0] - 1
        # 计算各统计指标
        if FactorExpose.shape[1]>self._Output["因子暴露"].shape[1]:
            FactorNames = FactorExpose.columns.tolist()
            self._Output["因子暴露"] = self._Output["因子暴露"].loc[:, FactorNames]
            self._Output["风险调整的因子暴露"] = self._Output["风险调整的因子暴露"].loc[:, FactorNames]
            self._Output["风险贡献"] = self._Output["风险贡献"].loc[:, FactorNames+["Alpha"]]
            self._Output["收益贡献"] = self._Output["收益贡献"].loc[:, FactorNames+["Alpha"]]
            self._Output["因子收益"] = self._Output["因子收益"].loc[:, FactorNames]
        self._Output["因子暴露"].loc[PreDT, FactorExpose.columns] = Beta
        self._Output["风险调整的因子暴露"].loc[PreDT, FactorExpose.columns] = np.sqrt(np.diag(np.dot(np.dot(FMPHolding, CovMatrix.values), FMPHolding.T))) * Beta
        RiskContribution = np.dot(np.dot(FMPHolding, CovMatrix.values), Portfolio.values) / np.sqrt(np.dot(np.dot(Portfolio.values, CovMatrix.values), Portfolio.values)) * Beta
        self._Output["风险贡献"].loc[idt, FactorExpose.columns] = RiskContribution
        self._Output["风险贡献"].loc[idt, "Alpha"] = np.sqrt(np.dot(np.dot(Portfolio.values, CovMatrix), Portfolio.values)) - np.nansum(RiskContribution)
        self._Output["因子收益"].loc[idt, FactorExpose.columns] = np.nansum(Return.values * FMPHolding, axis=1)
        self._Output["收益贡献"].loc[idt, FactorExpose.columns] = self._Output["因子收益"].loc[idt, FactorExpose.columns] * self._Output["因子暴露"].loc[PreDT, FactorExpose.columns]
        self._Output["收益贡献"].loc[idt, "Alpha"] = (Portfolio * Return).sum() - self._Output["收益贡献"].loc[idt, FactorExpose.columns].sum()
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        self.RiskTable.end()
        self._Output["风险贡献占比"] = self._Output["风险贡献"].divide(self._Output["风险贡献"].sum(axis=1), axis=0)
        self._Output["历史均值"] = pd.DataFrame(columns=["因子暴露", "风险调整的因子暴露", "风险贡献", "风险贡献占比", "收益贡献"], index=self._Output["收益贡献"].columns)
        self._Output["历史均值"]["因子暴露"] = self._Output["因子暴露"].mean(axis=0)
        self._Output["历史均值"]["风险调整的因子暴露"] = self._Output["风险调整的因子暴露"].mean(axis=0)
        self._Output["历史均值"]["风险贡献"] = self._Output["风险贡献"].mean(axis=0)
        self._Output["历史均值"]["风险贡献占比"] = self._Output["风险贡献占比"].mean(axis=0)
        self._Output["历史均值"]["收益贡献"] = self._Output["收益贡献"].mean(axis=0)
        self._IDs = None
        return 0
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = 2, 3
        Fig = plt.figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        AxesGrid = gridspec.GridSpec(nRow, nCol)
        xData = np.arange(1, self._Output["历史均值"].shape[0])
        xTickLabels = [str(iInd) for iInd in self._Output["历史均值"].index]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(plt.subplot(AxesGrid[0, 0]), xData[:-1], xTickLabels[:-1], self._Output["历史均值"]["因子暴露"].iloc[:-1], FloatFormatter)
        _QS_plotStatistics(plt.subplot(AxesGrid[0, 1]), xData[:-1], xTickLabels[:-1], self._Output["历史均值"]["风险调整的因子暴露"].iloc[:-1], FloatFormatter)
        _QS_plotStatistics(plt.subplot(AxesGrid[0, 2]), xData[:-1], xTickLabels[:-1], self._Output["历史均值"]["因子收益"].iloc[:-1], PercentageFormatter)
        _QS_plotStatistics(plt.subplot(AxesGrid[1, 0]), xData, xTickLabels, self._Output["历史均值"]["收益贡献"], PercentageFormatter)
        _QS_plotStatistics(plt.subplot(AxesGrid[1, 1]), xData, xTickLabels, self._Output["历史均值"]["风险贡献"], FloatFormatter)
        _QS_plotStatistics(plt.subplot(AxesGrid[1, 2]), xData, xTickLabels, self._Output["历史均值"]["风险贡献占比"], PercentageFormatter)
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self.ArgNames:
                if iArgName=="风险表":
                    HTML += "<li>"+iArgName+": "+self.RiskTable.Name+"</li>"
                elif iArgName!="计算时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        Formatters = [lambda x:'{0:.4f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x:'{0:.4f}'.format(x), _QS_formatPandasPercentage]
        iHTML = self._Output["历史均值"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = self.genMatplotlibFig()
        # figure 保存为二进制文件
        Buffer = BytesIO()
        plt.savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML