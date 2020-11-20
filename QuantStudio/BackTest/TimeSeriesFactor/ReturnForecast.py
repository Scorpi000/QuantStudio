# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Bool, Float, Dict
from traitsui.api import SetEditor, Item
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import testTimingStrategy, summaryStrategy, formatStrategySummary, summaryTimingStrategy, formatTimingStrategySummary
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.TimeSeriesFactor.Correlation import _calcReturn

class ReturnForecast(BaseModule):
    """收益率预测"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=2)
    ForecastPeriod = Int(1, arg_type="Integer", label="预测期数", order=3)
    Lag = Int(0, arg_type="Integer", label="滞后期数", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    SampleDTs = List(dt.datetime, arg_type="DateList", label="样本时点", order=6)
    SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=7)
    MinSummaryWindow = Int(2, arg_type="Integer", label="最小统计窗口", order=8)
    ModelArgs = Dict(value={}, arg_type="Dict", label="模型参数", order=9)
    HTMLPrintStatistics = List([], arg_type="List", label="打印统计量", order=10)
    HTMLPlotStatistics = List([], arg_type="List", label="绘图统计量", order=11)
    def __init__(self, factor_table, name="收益率预测", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return (Items, Context)
    # 用户实现的模型算子
    # 输入参数 : 
    # idt : 运行时点, datetime
    # iid : 标的 ID, str
    # sample_factor_data : 用于训练模型的因子样本数据, DataFrame(index=[时点], columns=[因子])
    # sample_return : 用于训练模型的对应的收益率样本数据, Series(index=[时点])
    # new_factor_data : 用于产生预测值的最新因子数据, Series(index=[因子])
    # args : 模型参数 : dict
    # 返回值 : 
    # ForecastReturn : 模型产生的预测收益率, float
    # ModelStatistics : 模型统计量, Series(index=[统计量]) 或者 None
    # FactorStatistics : 因子统计量, DataFrame(index=[因子], columns=[统计量]) 或者 None
    def ModelOperator(self, idt, iid, sample_factor_data, sample_return, new_factor_data):
        return np.nan, pd.Series(), pd.DataFrame()
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["证券ID"] = self._FactorTable.getID()
        self._Output["证券价格"] = pd.DataFrame(columns=self._Output["证券ID"])
        self._Output["样本收益率"] = pd.DataFrame(columns=self._Output["证券ID"])
        self._Output["预测收益率"] = pd.DataFrame(columns=self._Output["证券ID"])
        self._Output["因子值"] = {iID: pd.DataFrame(columns=self.TestFactors) for iID in self._Output["证券ID"]}
        self._Output["模型统计量"] = {}
        self._Output["因子统计量"] = {}
        self._CurCalcInd = 0
        self._CurSampleInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        self._Output["证券价格"].loc[idt] = self._FactorTable.readData(dts=[idt], ids=self._Output["证券ID"], factor_names=[self.PriceFactor]).iloc[0, 0, :].values
        isSampleDT = True
        if self.SampleDTs:
            if idt not in self.SampleDTs[self._CurSampleInd:]:
                isSampleDT = False
            else:
                self._CurSampleInd = self.SampleDTs[self._CurSampleInd:].index(idt) + self._CurSampleInd
                if self._CurSampleInd - self.ForecastPeriod - self.Lag>=0:
                    FactorDT = self.SampleDTs[self._CurSampleInd - self.ForecastPeriod - self.Lag]
                    RtnStartDT = self.SampleDTs[self._CurSampleInd - self.ForecastPeriod]
                    NewFactorDT = self.SampleDTs[self._CurSampleInd - self.Lag]
                else:
                    isSampleDT = False
        else:
            self._CurSampleInd = self._Model.DateTimeIndex
            if self._CurSampleInd - self.ForecastPeriod - self.Lag<0:
                isSampleDT = False
            else:            
                FactorDT = self._Model.DateTimeSeries[self._CurSampleInd - self.ForecastPeriod - self.Lag]
                RtnStartDT = self._Model.DateTimeSeries[self._CurSampleInd - self.ForecastPeriod]
                NewFactorDT = self._Model.DateTimeSeries[self._CurSampleInd - self.Lag]
        if isSampleDT:
            Price = self._Output["证券价格"].loc[[RtnStartDT, idt]]
            self._Output["样本收益率"].loc[idt] = _calcReturn(Price, return_type=self.ReturnType).iloc[-1].values
            FactorData = self._FactorTable.readData(dts=[FactorDT], ids=self._Output["证券ID"], factor_names=list(self.TestFactors)).iloc[:, 0]
            for i, iID in enumerate(self._Output["证券ID"]): self._Output["因子值"][iID].loc[FactorDT] = FactorData.iloc[i].values
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        if self._Output["样本收益率"].shape[0]<max(1, self.MinSummaryWindow): return 0
        FactorData = self._FactorTable.readData(dts=[NewFactorDT], ids=self._Output["证券ID"], factor_names=list(self.TestFactors)).iloc[:, 0]
        StartInd = int(max(0, self._Output["样本收益率"].shape[0] - self.SummaryWindow))
        ForecastReturn = np.full((len(self._Output["证券ID"]),), np.nan)
        for i, iID in enumerate(self._Output["证券ID"]):
            Y = self._Output["样本收益率"].iloc[StartInd:, i]
            X = self._Output["因子值"][iID].iloc[StartInd:]
            NewX = FactorData.iloc[i]
            ForecastReturn[i], iModelStatistics, iFactorStatistics = self.ModelOperator(idt, iID, X, Y, NewX)
            if (iModelStatistics is not None) and (iModelStatistics.shape[0]>0):
                self._Output["模型统计量"].setdefault(iID, pd.DataFrame(columns=iModelStatistics.index)).loc[idt] = iModelStatistics
            if iFactorStatistics is not None:
                for jKey in iFactorStatistics:
                    self._Output["因子统计量"].setdefault(iID, {}).setdefault(jKey, pd.DataFrame(columns=self.TestFactors)).loc[idt] = iFactorStatistics[jKey]
        self._Output["预测收益率"].loc[idt] = ForecastReturn
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        IDs = self._Output.pop("证券ID")
        Price = self._Output.pop("证券价格")
        if self._Output["样本收益率"].shape[0]==0:
            self._Output.pop("模型统计量")
            self._Output.pop("因子统计量")
            return 0
        # 全样本训练
        self._Output["全样本模型统计量"] = pd.DataFrame(columns=IDs)
        self._Output["全样本因子统计量"] = {}
        for i, iID in enumerate(IDs):
            Y = self._Output["样本收益率"].iloc[:, i]
            X = self._Output["因子值"][iID]
            NewX = self._Output["因子值"][iID].iloc[-1]
            _, iModelStatistics, iFactorStatistics = self.ModelOperator(self._Model.DateTimeSeries[-1], iID, X, Y, NewX)
            if (iModelStatistics is not None) and (iModelStatistics.shape[0]>0):
                self._Output["全样本模型统计量"][iID] = iModelStatistics
            if (iFactorStatistics is not None) and (iFactorStatistics.shape[1]>0):
                self._Output["全样本因子统计量"][iID] = iFactorStatistics
        if self._Output["全样本模型统计量"].shape[0]>0:
            self._Output["全样本模型统计量"] = self._Output["全样本模型统计量"].T
        else:
            self._Output.pop("全样本模型统计量")
        if self._Output["全样本因子统计量"]:
            self._Output["全样本因子统计量"] = dict(pd.Panel(self._Output["全样本因子统计量"]).swapaxes(0, 2).loc[:, :, IDs])
        else:
            self._Output.pop("全样本因子统计量")
        # 滚动预测
        self._Output["滚动预测"] = {}
        self._Output["滚动预测"]["预测收益率"] = self._Output.pop("预测收益率").astype(np.float)
        if self.CalcDTs:
            self._Output["滚动预测"]["实际收益率"] = pd.DataFrame(_calcReturn(Price.loc[self.CalcDTs].values, return_type=self.ReturnType), index=self.CalcDTs[1:], columns=IDs)
        else:
            self._Output["滚动预测"]["实际收益率"] = pd.DataFrame(_calcReturn(Price.loc[self._Model.DateTimeSeries].values, return_type=self.ReturnType), index=self._Model.DateTimeSeries[1:], columns=IDs)
        self._Output["滚动预测"]["实际收益率"] = self._Output["滚动预测"]["实际收益率"].shift(-1).loc[self._Output["滚动预测"]["预测收益率"].index]
        self._Output["滚动预测"]["绝对误差"] = (self._Output["滚动预测"]["预测收益率"] - self._Output["滚动预测"]["实际收益率"]).abs()
        self._Output["滚动预测"]["统计数据"] = pd.DataFrame({"均方误差": ((self._Output["滚动预测"]["预测收益率"] - self._Output["滚动预测"]["实际收益率"])**2).mean(), 
                                                                          "平均绝对误差": self._Output["滚动预测"]["绝对误差"].mean(), 
                                                                          "最大绝对误差": self._Output["滚动预测"]["绝对误差"].max(), 
                                                                          "最小绝对误差": self._Output["滚动预测"]["绝对误差"].min()}, columns=["均方误差", "平均绝对误差", "最大绝对误差", "最小绝对误差"])
        self._Output["滚动预测"]["统计数据"]["真正例(TP)"] = ((self._Output["滚动预测"]["实际收益率"]>=0) & (self._Output["滚动预测"]["预测收益率"]>=0)).sum()
        self._Output["滚动预测"]["统计数据"]["假正例(FP)"] = ((self._Output["滚动预测"]["实际收益率"]<0) & (self._Output["滚动预测"]["预测收益率"]>=0)).sum()
        self._Output["滚动预测"]["统计数据"]["真负例(TN)"] = ((self._Output["滚动预测"]["实际收益率"]<0) & (self._Output["滚动预测"]["预测收益率"]<0)).sum()
        self._Output["滚动预测"]["统计数据"]["假负例(FN)"] = ((self._Output["滚动预测"]["实际收益率"]>=0) & (self._Output["滚动预测"]["预测收益率"]<0)).sum()
        self._Output["滚动预测"]["统计数据"]["胜率"] = (self._Output["滚动预测"]["统计数据"]["真正例(TP)"] + self._Output["滚动预测"]["统计数据"]["真负例(TN)"]) / (pd.notnull(self._Output["滚动预测"]["实际收益率"]) & pd.notnull(self._Output["滚动预测"]["预测收益率"])).sum()
        self._Output["滚动预测"]["统计数据"]["看多精确率(Precision)"] = self._Output["滚动预测"]["统计数据"]["真正例(TP)"] / (self._Output["滚动预测"]["统计数据"]["真正例(TP)"] + self._Output["滚动预测"]["统计数据"]["假正例(FP)"])
        self._Output["滚动预测"]["统计数据"]["看多召回率(Recall)"] = self._Output["滚动预测"]["统计数据"]["真正例(TP)"] / (self._Output["滚动预测"]["统计数据"]["真正例(TP)"] + self._Output["滚动预测"]["统计数据"]["假负例(FN)"])
        self._Output["滚动预测"]["统计数据"]["看空精确率(Precision)"] = self._Output["滚动预测"]["统计数据"]["真负例(TN)"] / (self._Output["滚动预测"]["统计数据"]["真负例(TN)"] + self._Output["滚动预测"]["统计数据"]["假负例(FN)"])
        self._Output["滚动预测"]["统计数据"]["看空召回率(Recall)"] = self._Output["滚动预测"]["统计数据"]["真负例(TN)"] / (self._Output["滚动预测"]["统计数据"]["真负例(TN)"] + self._Output["滚动预测"]["统计数据"]["假正例(FP)"])
        # 择时策略
        StartIdx = self._Model.DateTimeSeries.index(self._Output["滚动预测"]["预测收益率"].index[0])
        DTs = self._Model.DateTimeSeries[StartIdx:]
        Signal = np.sign(self._Output["滚动预测"]["预测收益率"]).loc[DTs].values
        Price = Price.fillna(method="ffill").fillna(method="bfill").values[StartIdx:]
        nYear = (DTs[-1] - DTs[0]).days / 365
        NV, _, _ = testTimingStrategy(Signal, Price)
        self._Output["择时策略"] = {"多空净值": pd.DataFrame(NV, index=DTs, columns=IDs)}
        self._Output["择时策略"]["多空统计"], self._Output["择时策略"]["多头统计"], self._Output["择时策略"]["空头统计"], _, _ = summaryTimingStrategy(Signal, Price, n_per_year=len(DTs)/nYear)
        self._Output["择时策略"]["多空统计"].columns = self._Output["择时策略"]["多头统计"].columns = self._Output["择时策略"]["空头统计"].columns = IDs
        self._Output["择时策略"]["统计数据"] = summaryStrategy(NV, DTs, risk_free_rate=0.0)
        self._Output["择时策略"]["统计数据"].columns = IDs
        Signal[Signal<0] = 0
        NV, _, _ = testTimingStrategy(Signal, Price)
        self._Output["择时策略"]["纯多头策略净值"] = pd.DataFrame(NV, index=DTs, columns=IDs)
        self._Output["择时策略"]["纯多头策略统计数据"] = summaryStrategy(NV, DTs, risk_free_rate=0.0)
        self._Output["择时策略"]["纯多头策略统计数据"].columns = IDs
        self._Output["择时策略"]["标的净值"] = pd.DataFrame(Price / Price[0], index=DTs, columns=IDs)
        self._Output.pop("因子值")
        if not self._Output["模型统计量"]:
            self._Output.pop("模型统计量")
        else:
            self._Output["模型统计量"] = dict(pd.Panel(self._Output["模型统计量"]).swapaxes(0, 2).loc[:, :, IDs])
        if not self._Output["因子统计量"]: self._Output.pop("因子统计量")
        return 0
    def genMatplotlibFig(self, file_path=None):
        iTargetNV = self._Output["择时策略"]["标的净值"]
        if len(self.HTMLPlotStatistics)>0:
            iStatistics = self._Output.get("模型统计量", None)
        else:
            iStatistics = None
        iForecastReturn = self._Output["滚动预测"]["预测收益率"]
        iRealReturn = self._Output["滚动预测"]["实际收益率"]
        iLSNV = self._Output["择时策略"]["多空净值"]
        iLNV = self._Output["择时策略"]["纯多头策略净值"]
        nID = iForecastReturn.shape[1]
        xData1 = np.arange(0, iForecastReturn.shape[0])
        xTicks1 = np.arange(0, iForecastReturn.shape[0], int(iForecastReturn.shape[0]/10))
        xTickLabels1 = [iForecastReturn.index[i].strftime("%Y-%m-%d") for i in xTicks1]
        xData2 = np.arange(0, iLSNV.shape[0])
        xTicks2 = np.arange(0, iLSNV.shape[0], int(iLSNV.shape[0]/10))
        xTickLabels2 = [iLSNV.index[i].strftime("%Y-%m-%d") for i in xTicks2]
        nCol = 2 + int(iStatistics is not None)
        Fig = Figure(figsize=(8+8*nCol, 8*nID))
        for j, jID in enumerate(iTargetNV.columns):
            iAxes = Fig.add_subplot(nID, nCol, j*nCol+1)
            iAxes.bar(xData1-0.3, iRealReturn.iloc[:, j].values, 0.3, color="indianred", label="实际收益率")
            iAxes.bar(xData1, iForecastReturn.iloc[:, j].values, 0.3, color="steelblue", label="预测收益率")
            iAxes.set_xticks(xTicks1)
            iAxes.set_xticklabels(xTickLabels1)
            iAxes.legend()
            iAxes.set_title(str(jID)+" : 滚动预测")
            iAxes = Fig.add_subplot(nID, nCol, j*nCol+2)
            iAxes.plot(xData2, iLSNV.iloc[:, j].values, label="多空净值", color="steelblue", lw=2.5)
            iAxes.plot(xData2, iLNV.iloc[:, j].values, label="纯多头净值", color="indianred", lw=2.5)
            iAxes.plot(xData2, iTargetNV.iloc[:, j].values, label="标的净值", color="forestgreen", lw=2.5)
            iAxes.set_xticks(xTicks2)
            iAxes.set_xticklabels(xTickLabels2)
            iAxes.legend()
            iAxes.set_title(str(jID)+" : 择时策略")
            if iStatistics is not None:
                iAxes = Fig.add_subplot(nID, nCol, j*nCol+3)
                for iKey in self.HTMLPlotStatistics:
                    if iKey in iStatistics:
                        iAxes.plot(xData1, iStatistics[iKey].iloc[:, j].values, label=iKey, lw=2.5)
                    else:
                        self._QS_Logger.warning("未生成模型统计量 : %s" % iKey)
                iAxes.set_xticks(xTicks1)
                iAxes.set_xticklabels(xTickLabels1)
                iAxes.legend()
                iAxes.set_title(str(jID)+" : 模型统计量")
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
        if "全样本模型统计量" in self._Output:
            HTML += "全样本模型统计量 : "
            iHTML = self._Output["全样本模型统计量"].to_html(formatters=[lambda x:'{0:.4f}'.format(x)]*self._Output["全样本模型统计量"].shape[1])
            Pos = iHTML.find(">")
            HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        for iKey in self.HTMLPrintStatistics:
            if ("全样本因子统计量" in self._Output) and (iKey in self._Output["全样本因子统计量"]):
                HTML += f"全样本因子统计量 - {iKey} : "
                iHTML = self._Output["全样本因子统计量"][iKey].to_html(formatters=[lambda x:'{0:.4f}'.format(x)]*self._Output["全样本因子统计量"][iKey].shape[1])
                Pos = iHTML.find(">")
                HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            else:
                self._QS_Logger.warning("未生成全样本因子统计量 : %s" % iKey)
        HTML += "滚动预测 - 统计数据: "
        Formatters = [lambda x:'{0:.4f}'.format(x)]*4+[lambda x:'{0:d}'.format(int(x))]*4+[lambda x:'{0:.2f}%'.format(x*100)]*5
        iHTML = self._Output["滚动预测"]["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += "滚动预测 - 统计数据: "
        Formatters = [lambda x:'{0:.4f}'.format(x)]*4+[lambda x:'{0:d}'.format(int(x))]*4+[lambda x:'{0:.2f}%'.format(x*100)]*5
        iHTML = self._Output["滚动预测"]["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += "择时策略 - 统计数据 : "
        iHTML = formatStrategySummary(self._Output["择时策略"]["统计数据"]).to_html()
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += "择时策略 - 纯多头统计数据 : "
        iHTML = formatStrategySummary(self._Output["择时策略"]["纯多头策略统计数据"]).to_html()
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += "择时策略 - 多空统计 : "
        iHTML = formatTimingStrategySummary(self._Output["择时策略"]["多空统计"]).to_html()
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += "择时策略 - 多头统计 : "
        iHTML = formatTimingStrategySummary(self._Output["择时策略"]["多头统计"]).to_html()
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += "择时策略 - 空头统计 : "
        iHTML = formatTimingStrategySummary(self._Output["择时策略"]["空头统计"]).to_html()
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = self.genMatplotlibFig()
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML

class OLS(ReturnForecast):
    """时间序列 OLS"""
    def __init__(self, factor_table, name="时间序列OLS", sys_args={}, **kwargs):
        return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.ModelArgs = {"常数项": True}
        self.HTMLPrintStatistics = ["回归系数", "t统计量"]
        self.HTMLPlotStatistics = ["R平方", "调整R平方"]
    def ModelOperator(self, idt, iid, sample_factor_data, sample_return, new_factor_data):
        ForecastReturn = np.nan
        FactorStatistics = pd.DataFrame(index=sample_factor_data.columns, columns=["回归系数", "t统计量"])
        Y= sample_return.values
        hasConstant = self.ModelArgs.get("常数项", True)
        if hasConstant:
            X = sm.add_constant(sample_factor_data.values, prepend=True)
            NewX = np.r_[1, new_factor_data.values]
            ModelStatistics = pd.Series(np.nan, index=["R平方", "调整R平方", "F统计量", "常数项", "常数项t统计量"])
        else:
            X = sample_factor_data.values
            NewX = new_factor_data.values
            ModelStatistics = pd.Series(np.nan, index=["R平方", "调整R平方", "F统计量"])
        try:
            Result = sm.OLS(Y, X, missing="drop").fit()
        except Exception as e:
            self._QS_Logger.warning("%s : '%s' 在 %s 时的回归失败 : %s" % (self.Name, iid, idt.strftime("%Y-%m-%d"), str(e)))
        else:
            ForecastReturn = np.nansum(Result.params * NewX)
            FactorStatistics["回归系数"] = Result.params[int(hasConstant):]
            FactorStatistics["t统计量"] = Result.tvalues[int(hasConstant):]
            ModelStatistics["R平方"] = Result.rsquared
            ModelStatistics["调整R平方"] = Result.rsquared_adj
            ModelStatistics["F统计量"] = Result.fvalue
            if hasConstant:
                ModelStatistics["常数项"] = Result.params[0]
                ModelStatistics["常数项t统计量"] = Result.tvalues[0]
        return ForecastReturn, ModelStatistics, FactorStatistics