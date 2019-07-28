# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict
from traitsui.api import SetEditor, Item
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage

class FamaMacBethRegression(BaseModule):
    """Fama-MacBeth 回归"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    #IndustryFactor = Enum("无", arg_type="SingleOption", label="行业因子", order=2)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=3)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=4)
    RollAvgPeriod = Int(12, arg_type="Integer", label="滚动平均期数", order=5)
    def __init__(self, factor_table, name="Fama-MacBeth 回归", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=2))
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return (Items, Context)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {"Pure Return":[], "Raw Return":[], "时点":[], "回归R平方":[], "回归调整R平方":[], "回归F统计量":[], "回归t统计量(Raw Return)":[], "回归t统计量(Pure Return)":[]}
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            LastInd = self._CurCalcInd - 1
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            LastInd = self._CurCalcInd - 1
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        nFactor = len(self.TestFactors)
        self._Output["Pure Return"].append(np.full(shape=(nFactor,), fill_value=np.nan))
        self._Output["Raw Return"].append(np.full(shape=(nFactor,), fill_value=np.nan))
        self._Output["回归t统计量(Pure Return)"].append(np.full(shape=(nFactor,), fill_value=np.nan))
        self._Output["回归t统计量(Raw Return)"].append(np.full(shape=(nFactor,), fill_value=np.nan))
        self._Output["回归F统计量"].append(np.full(shape=(nFactor+1,), fill_value=np.nan))
        self._Output["回归R平方"].append(np.full(shape=(nFactor+1,), fill_value=np.nan))
        self._Output["回归调整R平方"].append(np.full(shape=(nFactor+1,), fill_value=np.nan))
        self._Output["时点"].append(idt)
        if LastInd<0: return 0
        LastIDs = self._FactorTable.getFilteredID(idt=LastDateTime, id_filter_str=self.IDFilter)
        FactorData = self._FactorTable.readData(dts=[LastDateTime], ids=LastIDs, factor_names=list(self.TestFactors)).iloc[:,0,:]
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=LastIDs, factor_names=[self.PriceFactor]).iloc[0]
        Ret = Price.iloc[1] / Price.iloc[0] - 1
        # 展开Dummy因子
        if self.IndustryFactor!="无":
            DummyFactorData = self._FactorTable.readData(dts=[LastDateTime], ids=LastIDs, factor_names=[self.IndustryFactor]).iloc[0,0,:]
            Mask = pd.notnull(DummyFactorData)
            DummyFactorData = DummyVarTo01Var(DummyFactorData[Mask], ignore_na=True)
            FactorData = pd.merge(FactorData.loc[Mask], DummyFactorData, left_index=True, right_index=True)
        # 回归
        yData = Ret[FactorData.index].values
        xData = FactorData.values
        if self.IndustryFactor=="无":
            xData = sm.add_constant(xData, prepend=False)
            LastInds = [nFactor]
        else:
            LastInds = [nFactor+i for i in range(xData.shape[1]-nFactor)]
        try:
            Result = sm.OLS(yData, xData, missing="drop").fit()
            self._Output["Pure Return"][-1] = Result.params[0:nFactor]
            self._Output["回归t统计量(Pure Return)"][-1] = Result.tvalues[0:nFactor]
            self._Output["回归F统计量"][-1][-1] = Result.fvalue
            self._Output["回归R平方"][-1][-1] = Result.rsquared
            self._Output["回归调整R平方"][-1][-1] = Result.rsquared_adj
        except:
            pass
        for i, iFactorName in enumerate(self.TestFactors):
            iXData = xData[:,[i]+LastInds]
            try:
                Result = sm.OLS(yData, iXData, missing="drop").fit()
                self._Output["Raw Return"][-1][i] = Result.params[0]
                self._Output["回归t统计量(Raw Return)"][-1][i] = Result.tvalues[0]
                self._Output["回归F统计量"][-1][i] = Result.fvalue
                self._Output["回归R平方"][-1][i] = Result.rsquared
                self._Output["回归调整R平方"][-1][i] = Result.rsquared_adj
            except:
                pass
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        FactorNames = list(self.TestFactors)
        self._Output["Pure Return"] = pd.DataFrame(self._Output["Pure Return"], index=self._Output["时点"], columns=FactorNames)
        self._Output["Raw Return"] = pd.DataFrame(self._Output["Raw Return"], index=self._Output["时点"], columns=FactorNames)
        self._Output["滚动t统计量_Pure"] = pd.DataFrame(np.nan, index=self._Output["时点"], columns=FactorNames)
        self._Output["滚动t统计量_Raw"] = pd.DataFrame(np.nan, index=self._Output["时点"], columns=FactorNames)
        self._Output["回归t统计量(Raw Return)"] = pd.DataFrame(self._Output["回归t统计量(Raw Return)"], index=self._Output["时点"], columns=FactorNames)
        self._Output["回归t统计量(Pure Return)"] = pd.DataFrame(self._Output["回归t统计量(Pure Return)"], index=self._Output["时点"], columns=FactorNames)
        self._Output["回归F统计量"] = pd.DataFrame(self._Output["回归F统计量"], index=self._Output["时点"], columns=FactorNames+["所有因子"])
        self._Output["回归R平方"] = pd.DataFrame(self._Output["回归R平方"], index=self._Output["时点"], columns=FactorNames+["所有因子"])
        self._Output["回归调整R平方"] = pd.DataFrame(self._Output["回归调整R平方"], index=self._Output["时点"], columns=FactorNames+["所有因子"])
        nDT = self._Output["Raw Return"].shape[0]
        # 计算滚动t统计量
        for i in range(nDT):
            if i<self.RollAvgPeriod-1: continue
            iReturn = self._Output["Pure Return"].iloc[i-self.RollAvgPeriod+1:i+1, :]
            self._Output["滚动t统计量_Pure"].iloc[i] = iReturn.mean(axis=0) / iReturn.std(axis=0) * pd.notnull(iReturn).sum(axis=0)**0.5
            iReturn = self._Output["Raw Return"].iloc[i-self.RollAvgPeriod+1:i+1, :]
            self._Output["滚动t统计量_Raw"].iloc[i] = iReturn.mean(axis=0) / iReturn.std(axis=0) * pd.notnull(iReturn).sum(axis=0)**0.5
        nYear = (self._Output["时点"][-1] - self._Output["时点"][0]).days / 365
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["Pure Return"].columns)
        self._Output["统计数据"]["年化收益率(Pure)"] = ((1 + self._Output["Pure Return"]).prod())**(1/nYear) - 1
        self._Output["统计数据"]["跟踪误差(Pure)"] = self._Output["Pure Return"].std() * np.sqrt(nDT/nYear)
        self._Output["统计数据"]["信息比率(Pure)"] = self._Output["统计数据"]["年化收益率(Pure)"] / self._Output["统计数据"]["跟踪误差(Pure)"]
        self._Output["统计数据"]["胜率(Pure)"] = (self._Output["Pure Return"]>0).sum() / nDT
        self._Output["统计数据"]["t统计量(Pure)"] = self._Output["Pure Return"].mean() / self._Output["Pure Return"].std() * np.sqrt(nDT)
        self._Output["统计数据"]["年化收益率(Raw)"] = (1 + self._Output["Raw Return"]).prod()**(1/nYear) - 1
        self._Output["统计数据"]["跟踪误差(Raw)"] = self._Output["Raw Return"].std() * np.sqrt(nDT/nYear)
        self._Output["统计数据"]["信息比率(Raw)"] = self._Output["统计数据"]["年化收益率(Raw)"] / self._Output["统计数据"]["跟踪误差(Raw)"]
        self._Output["统计数据"]["胜率(Raw)"] = (self._Output["Raw Return"]>0).sum() / nDT
        self._Output["统计数据"]["t统计量(Raw)"] = self._Output["Raw Return"].mean() / self._Output["Raw Return"].std() * np.sqrt(nDT)
        self._Output["统计数据"]["年化收益率(Pure-Naive)"] = (1 + self._Output["Pure Return"] - self._Output["Raw Return"]).prod()**(1/nYear) - 1
        self._Output["统计数据"]["跟踪误差(Pure-Naive)"] = (self._Output["Pure Return"] - self._Output["Raw Return"]).std() * np.sqrt(nDT/nYear)
        self._Output["统计数据"]["信息比率(Pure-Naive)"] = self._Output["统计数据"]["年化收益率(Pure-Naive)"] / self._Output["统计数据"]["跟踪误差(Pure-Naive)"]
        self._Output["统计数据"]["胜率(Pure-Naive)"] = (self._Output["Pure Return"] - self._Output["Raw Return"]>0).sum() / nDT
        self._Output["统计数据"]["t统计量(Pure-Naive)"] = (self._Output["Pure Return"] - self._Output["Raw Return"]).mean() / (self._Output["Pure Return"] - self._Output["Raw Return"]).std() * np.sqrt(nDT)
        self._Output["回归统计量均值"] = pd.DataFrame(index=FactorNames+["所有因子"])
        self._Output["回归统计量均值"]["t统计量(Raw Return)"] = self._Output["回归t统计量(Raw Return)"].mean()
        self._Output["回归统计量均值"]["t统计量(Pure Return)"] = self._Output["回归t统计量(Pure Return)"].mean()
        self._Output["回归统计量均值"]["F统计量"] = self._Output["回归F统计量"].mean()
        self._Output["回归统计量均值"]["R平方"] = self._Output["回归R平方"].mean()
        self._Output["回归统计量均值"]["调整R平方"] = self._Output["回归调整R平方"].mean()
        self._Output.pop("时点")
        return 0
    def _plotStatistics(self, axes, x_data, x_ticklabels, left_data, left_formatter, right_data=None, right_formatter=None, right_axes=True):
        axes.yaxis.set_major_formatter(left_formatter)
        axes.bar(x_data, left_data.values, label=left_data.name, color="b")
        if right_data is not None:
            if right_axes:
                axes.legend(loc='upper left')
                right_axes = axes.twinx()
                right_axes.yaxis.set_major_formatter(right_formatter)
                right_axes.plot(x_data, right_data.values, label=right_data.name, color="r", alpha=0.6, lw=3)
                right_axes.legend(loc="upper right")
            else:
                axes.plot(x_data, right_data.values, label=right_data.name, color="r", alpha=0.6, lw=3)
                axes.legend(loc='best')
        else:
            axes.legend(loc='best')
        axes.set_xticks(x_data)
        axes.set_xticklabels(x_ticklabels)
        return axes
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = 1, 3
        Fig = plt.figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        AxesGrid = gridspec.GridSpec(nRow, nCol)
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        xData = np.arange(0, self._Output["统计数据"].shape[0])
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"].index]
        iAxes = plt.subplot(AxesGrid[0, 0])
        iAxes.yaxis.set_major_formatter(PercentageFormatter)
        iAxes.bar(xData, self._Output["统计数据"]["年化收益率(Raw)"].values, width=-0.25, align="edge", color="r", label="年化收益率(Raw)")
        iAxes.bar(xData, self._Output["统计数据"]["年化收益率(Pure)"].values, width=0.25, align="edge", color="b", label="年化收益率(Pure)")
        iAxes.set_xticks(xData)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc='best')
        iAxes.set_title("年化收益率")
        iAxes = plt.subplot(AxesGrid[0, 1])
        iAxes.yaxis.set_major_formatter(FloatFormatter)
        iAxes.bar(xData, self._Output["统计数据"]["t统计量(Raw)"].values, width=-0.25, align="edge", color="r", label="t统计量(Raw)")
        iAxes.bar(xData, self._Output["统计数据"]["t统计量(Pure)"].values, width=0.25, align="edge", color="b", label="t统计量(Pure)")
        iAxes.set_xticks(xData)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc='best')
        iAxes.set_title("t统计量")
        iAxes = plt.subplot(AxesGrid[0, 2])
        iAxes.yaxis.set_major_formatter(PercentageFormatter)
        iAxes.bar(xData, self._Output["统计数据"]["年化收益率(Pure-Naive)"].values, color="r", label="年化收益率(Pure-Naive)")
        iAxes.set_xticks(xData)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc='upper left')
        iAxes.set_title("Pure-Naive")
        RAxes = iAxes.twinx()
        RAxes.yaxis.set_major_formatter(FloatFormatter)
        RAxes.plot(xData, self._Output["统计数据"]["t统计量(Pure-Naive)"].values, color="b", alpha=0.6, lw=3, label="t统计量(Pure-Naive)")
        RAxes.legend(loc='upper right')
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self.ArgNames:
                if iArgName!="计算时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        FloatFormatFun = lambda x:'{0:.2f}'.format(x)
        Formatters = [_QS_formatPandasPercentage]*2+[FloatFormatFun, _QS_formatPandasPercentage, FloatFormatFun]
        Formatters += [_QS_formatPandasPercentage]*2+[FloatFormatFun, _QS_formatPandasPercentage, FloatFormatFun]
        Formatters += [_QS_formatPandasPercentage]*2+[FloatFormatFun, _QS_formatPandasPercentage, FloatFormatFun]
        iHTML = self._Output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += '<div align="left" style="font-size:1em"><strong>回归统计量</strong></div>'
        iHTML = self._Output["回归统计量均值"].to_html(formatters=[FloatFormatFun]*5)
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