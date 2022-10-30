# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Float
from matplotlib.figure import Figure

from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.BackTest.BackTestModel import BaseModule

def _calcReturn(price, return_type="简单收益率"):
    if return_type=="对数收益率":
        Return = np.log(1 + np.diff(price, axis=0) / np.abs(price[:-1]))
        Return[np.isinf(Return)] = np.nan
        return Return
    elif return_type=="价格变化量": return np.diff(price, axis=0)
    else: return np.diff(price, axis=0) / np.abs(price[:-1])

def _formatSummary(summary):
    FormattedStats = pd.DataFrame(index=summary.index, columns=summary.columns, dtype="O")
    PercentageFormatFun = np.vectorize(lambda x: ("%.2f%%" % (x*100, )))
    FormattedStats.iloc[:, :] = PercentageFormatFun(summary.values)
    return FormattedStats

class TimeSeriesCorrelation(BaseModule):
    """时间序列相关性"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
        ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=2)
        ForecastPeriod = Int(1, arg_type="Integer", label="预测期数", order=3)
        Lag = Int(0, arg_type="Integer", label="滞后期数", order=4)
        CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
        CorrMethod = Enum("pearson", "spearman", "kendall", arg_type="SingleOption", label="相关性算法", order=6)
        SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=7)
        MinSummaryWindow = Int(2, arg_type="Integer", label="最小统计窗口", order=8)
        def __QS_initArgs__(self):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
            self.TestFactors.append(DefaultNumFactorList[0])
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    
    def __init__(self, factor_table, name="时间序列相关性", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["证券ID"] = self._FactorTable.getID()
        nID = len(self._Output["证券ID"])
        self._Output["滚动相关性"] = {iFactorName:{} for iFactorName in self._QSArgs.TestFactors}# {因子: DataFrame(index=[时点], columns=[ID])} 
        self._Output["收益率"] = np.zeros(shape=(0, nID))
        self._Output["因子值"] = {iFactorName:np.zeros(shape=(0, nID)) for iFactorName in self._QSArgs.TestFactors}
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self._QSArgs.ForecastPeriod - self._QSArgs.Lag
            LastInd = self._CurCalcInd - self._QSArgs.ForecastPeriod
            PreDateTime = self._QSArgs.CalcDTs[PreInd]
            LastDateTime = self._QSArgs.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self._QSArgs.ForecastPeriod - self._QSArgs.Lag
            LastInd = self._CurCalcInd - self._QSArgs.ForecastPeriod
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0): return 0
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=self._Output["证券ID"], factor_names=[self._QSArgs.PriceFactor]).iloc[0, :, :].values
        self._Output["收益率"] = np.r_[self._Output["收益率"], _calcReturn(Price, return_type=self._QSArgs.ReturnType)]
        FactorData = self._FactorTable.readData(dts=[PreDateTime], ids=self._Output["证券ID"], factor_names=list(self._QSArgs.TestFactors)).iloc[:, 0, :].values.T
        StartInd = int(max(0, self._Output["收益率"].shape[0] - self._QSArgs.SummaryWindow))
        for i, iFactorName in enumerate(self._QSArgs.TestFactors):
            self._Output["因子值"][iFactorName] = np.r_[self._Output["因子值"][iFactorName], FactorData[i:i+1]]
            if self._Output["收益率"].shape[0]>=self._QSArgs.MinSummaryWindow:
                self._Output["滚动相关性"][iFactorName][idt] = np.diag(pd.DataFrame(np.c_[self._Output["因子值"][iFactorName][StartInd:], self._Output["收益率"][StartInd:]]).corr(method=self._QSArgs.CorrMethod, min_periods=self._QSArgs.MinSummaryWindow).values, k=FactorData.shape[1])
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        IDs = self._Output.pop("证券ID")
        self._Output["统计数据"] = {}
        for iFactorName in self._QSArgs.TestFactors:
            self._Output["滚动相关性"][iFactorName] = pd.DataFrame(self._Output["滚动相关性"][iFactorName], index=IDs).T.sort_index(axis=0)
            self._Output["统计数据"][iFactorName] = pd.DataFrame({"平均值": self._Output["滚动相关性"][iFactorName].mean(), "中位数": self._Output["滚动相关性"][iFactorName].median(), 
                                                                                                           "最小值": self._Output["滚动相关性"][iFactorName].min(), "最大值": self._Output["滚动相关性"][iFactorName].max()})
        self._Output.pop("收益率"), self._Output.pop("因子值")
        return 0
    def genMatplotlibFig(self, file_path=None, target_factor=None):
        if target_factor is None: target_factor = self._QSArgs.TestFactors[0]
        iData = self._Output["滚动相关性"][target_factor]
        nID = iData.shape[1]
        xData = np.arange(0, iData.shape[0])
        xTicks = np.arange(0, iData.shape[0], int(iData.shape[0]/10))
        xTickLabels = [iData.index[i].strftime("%Y-%m-%d") for i in xTicks]
        nRow, nCol = nID//3+(nID%3!=0), min(3, nID)
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        for j, jID in enumerate(iData.columns):
            iAxes = Fig.add_subplot(nRow, nCol, j+1)
            iAxes.bar(xData, iData.values[:, j], color="steelblue")
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.set_title(target_factor+" - "+str(jID)+" : 滚动相关性")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                if iArgName!="计算时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        for iFactor in self._QSArgs.TestFactors:
            iOutput = self._Output["统计数据"][iFactor]
            iHTML = iFactor+" - 统计数据 : "
            iHTML += _formatSummary(iOutput).to_html()
            Pos = iHTML.find(">")
            HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            Fig = self.genMatplotlibFig(target_factor=iFactor)
            # figure 保存为二进制文件
            Buffer = BytesIO()
            Fig.savefig(Buffer)
            PlotData = Buffer.getvalue()
            # 图像数据转化为 HTML 格式
            ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
            HTML += ('<img src="%s">' % ImgStr)
        return HTML