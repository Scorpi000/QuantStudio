# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Bool, Float
from traitsui.api import SetEditor, Item
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import testTimingStrategy, summaryStrategy, formatStrategySummary, summaryTimingStrategy, formatTimingStrategySummary
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.TimeSeriesFactor.Correlation import _calcReturn


class OLS(BaseModule):
    """时间序列 OLS"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    ReturnType = Enum("简单收益率", "对数收益率", "价格变化量", arg_type="SingleOption", label="收益率类型", order=2)
    ForecastPeriod = Int(1, arg_type="Integer", label="预测期数", order=3)
    Lag = Int(0, arg_type="Integer", label="滞后期数", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=6)
    MinSummaryWindow = Int(2, arg_type="Integer", label="最小统计窗口", order=7)
    Constant = Bool(True, arg_type="Bool", label="常数项", order=8)
    def __init__(self, factor_table, name="时间序列OLS", sys_args={}, **kwargs):
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
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["证券ID"] = self._FactorTable.getID()
        nID, nFactor = len(self._Output["证券ID"]), len(self.TestFactors)
        self._Output["证券价格"] = np.full((len(dts), nID), np.nan)
        self._Output["因子值"] = np.zeros((0, nID, nFactor))
        self._Output["实际收益率"] = np.zeros(shape=(0, nID))
        self._Output["预测收益率"] = np.zeros(shape=(0, nID))
        self._Output["滚动回归"] = {"回归系数": {}, "R平方":np.zeros((0, nID)), "调整R平方":np.zeros((0, nID)), "t统计量":{}, "F统计量":np.zeros((0, nID))}
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]:
                self._Output["证券价格"][self._Model.DateTimeIndex] = self._FactorTable.readData(dts=[idt], ids=self._Output["证券ID"], factor_names=[self.PriceFactor]).iloc[0, 0, :].values
                return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self.ForecastPeriod - self.Lag
            LastInd = self._CurCalcInd - self.ForecastPeriod
            PreDateTime = self.CalcDTs[PreInd]
            LastDateTime = self.CalcDTs[LastInd]
            LagDateTime = self.CalcDTs[self._CurCalcInd - self.Lag]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self.ForecastPeriod - self.Lag
            LastInd = self._CurCalcInd - self.ForecastPeriod
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
            LagDateTime = self._Model.DateTimeSeries[self._CurCalcInd - self.Lag]
        if (PreInd<0) or (LastInd<0):
            self._Output["证券价格"][self._Model.DateTimeIndex] = self._FactorTable.readData(dts=[idt], ids=self._Output["证券ID"], factor_names=[self.PriceFactor]).iloc[0, 0, :].values
            return 0
        else:
            nID, nFactor = len(self._Output["证券ID"]), len(self.TestFactors)
            Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=self._Output["证券ID"], factor_names=[self.PriceFactor]).iloc[0, :, :].values
            self._Output["证券价格"][self._Model.DateTimeIndex] = Price[-1]
        self._Output["实际收益率"] = np.r_[self._Output["实际收益率"], _calcReturn(Price, return_type=self.ReturnType)]
        FactorData = self._FactorTable.readData(dts=[PreDateTime, LagDateTime], ids=self._Output["证券ID"], factor_names=list(self.TestFactors)).values
        self._Output["因子值"] = np.r_[self._Output["因子值"], FactorData[:, 0, :].T.reshape((1, nID, -1))]
        if self._Output["实际收益率"].shape[0]<self.MinSummaryWindow: return 0
        StartInd = int(max(0, self._Output["实际收益率"].shape[0] - self.SummaryWindow))
        Statistics = {"R平方":np.full((1, nID), np.nan), "调整R平方":np.full((1, nID), np.nan), "t统计量":np.full((nFactor+int(self.Constant), nID), np.nan), "F统计量":np.full((1, nID), np.nan)}
        ForecastReturn = np.full((1, nID), np.nan)
        Params = np.full((nFactor+int(self.Constant), nID), np.nan)
        for i, iID in enumerate(self._Output["证券ID"]):
            Y = self._Output["实际收益率"][StartInd:, i]
            X = self._Output["因子值"][StartInd:, i]
            if self.Constant: X = sm.add_constant(X, prepend=True)
            try:
                Result = sm.OLS(Y, X, missing="drop").fit()
            except Exception as e:
                self._QS_Logger.warning("%s : '%s' 在 %s 时的回归失败 : %s" % (self.Name, iID, idt.strftime("%Y-%m-%d"), str(e)))
            else:
                Params[:, i] = Result.params
                ForecastReturn[0, i] = Result.params[0] + np.nansum(Result.params[1:] * FactorData[:, 1, i])
                Statistics["R平方"][0, i] = Result.rsquared
                Statistics["调整R平方"][0, i] = Result.rsquared_adj
                Statistics["F统计量"][0, i] = Result.fvalue
                Statistics["t统计量"][:, i] = Result.tvalues
        self._Output["预测收益率"] = np.r_[self._Output["预测收益率"], ForecastReturn]
        self._Output["滚动回归"]["回归系数"][idt] = Params
        self._Output["滚动回归"]["R平方"] = np.r_[self._Output["滚动回归"]["R平方"], Statistics["R平方"]]
        self._Output["滚动回归"]["调整R平方"] = np.r_[self._Output["滚动回归"]["调整R平方"], Statistics["调整R平方"]]
        self._Output["滚动回归"]["F统计量"] = np.r_[self._Output["滚动回归"]["F统计量"], Statistics["F统计量"]]
        self._Output["滚动回归"]["t统计量"][idt] = Statistics["t统计量"]
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        IDs = self._Output.pop("证券ID")
        DTs = sorted(self._Output["滚动回归"]["t统计量"])
        Index = pd.Index(self.TestFactors, name="因子")
        if self.Constant: Index = Index.insert(0, "Constant")
        #self._Output["最后一期统计量"] = pd.DataFrame({"R平方": self._Output["滚动回归"]["R平方"][-1], "调整R平方": self._Output["滚动回归"]["调整R平方"][-1],
                                                      #"F统计量": self._Output["滚动回归"]["F统计量"][-1]}, index=IDs).loc[:, ["R平方", "调整R平方", "F统计量"]]
        #self._Output["最后一期t统计量"] = pd.DataFrame(self._Output["滚动回归"]["t统计量"][DTs[-1]], index=Index, columns=IDs)
        # 全样本回归
        self._Output["全样本回归"] = {"统计量": pd.DataFrame(index=IDs, columns=["R平方", "调整R平方", "F统计量"]), 
                                                          "t统计量": pd.DataFrame(index=IDs, columns=Index), 
                                                          "回归系数": pd.DataFrame(index=IDs, columns=Index)} 
        for i, iID in enumerate(IDs):
            Y = self._Output["实际收益率"][:, i]
            X = self._Output["因子值"][:, i]
            if self.Constant: X = sm.add_constant(X, prepend=True)
            try:
                Result = sm.OLS(Y, X, missing="drop").fit()
            except Exception as e:
                self._QS_Logger.warning("%s : '%s' 全样本回归失败 : %s" % (self.Name, iID, str(e)))
            else:
                self._Output["全样本回归"]["回归系数"].iloc[i] = Result.params
                self._Output["全样本回归"]["统计量"].iloc[i, 0] = Result.rsquared
                self._Output["全样本回归"]["统计量"].iloc[i, 1] = Result.rsquared_adj
                self._Output["全样本回归"]["统计量"].iloc[i, 2] = Result.fvalue
                self._Output["全样本回归"]["t统计量"].iloc[i] = Result.tvalues
        # 滚动回归
        self._Output["滚动回归"]["R平方"] = pd.DataFrame(self._Output["滚动回归"]["R平方"], index=DTs, columns=IDs)
        self._Output["滚动回归"]["调整R平方"] = pd.DataFrame(self._Output["滚动回归"]["调整R平方"], index=DTs, columns=IDs)
        self._Output["滚动回归"]["F统计量"] = pd.DataFrame(self._Output["滚动回归"]["F统计量"], index=DTs, columns=IDs)
        self._Output["滚动回归"]["t统计量"] = pd.Panel(self._Output["滚动回归"]["t统计量"], major_axis=Index, minor_axis=IDs)
        self._Output["滚动回归"]["t统计量"] = dict(self._Output["滚动回归"]["t统计量"].swapaxes(0, 1))
        self._Output["滚动回归"]["回归系数"] = pd.Panel(self._Output["滚动回归"]["回归系数"], major_axis=Index, minor_axis=IDs)
        self._Output["滚动回归"]["回归系数"] = dict(self._Output["滚动回归"]["回归系数"].swapaxes(0, 1))
        # 滚动预测
        self._Output["滚动预测"] = {}
        self._Output["滚动预测"]["预测收益率"] = pd.DataFrame(self._Output.pop("预测收益率"), index=DTs, columns=IDs)
        self._Output["滚动预测"]["实际收益率"] = pd.DataFrame(np.r_[self._Output.pop("实际收益率")[-len(DTs)+1:], np.full((1, len(IDs)), np.nan)], index=DTs, columns=IDs)
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
        Price = pd.DataFrame(self._Output.pop("证券价格")).fillna(method="ffill").fillna(method="bfill").values[StartIdx:]
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
        return 0
    def genMatplotlibFig(self, file_path=None):
        iTargetNV = self._Output["择时策略"]["标的净值"]
        iR2 = self._Output["滚动回归"]["R平方"]
        iAdjR2 = self._Output["滚动回归"]["调整R平方"]
        iForecastReturn = self._Output["滚动预测"]["预测收益率"]
        iRealReturn = self._Output["滚动预测"]["实际收益率"]
        iLSNV = self._Output["择时策略"]["多空净值"]
        iLNV = self._Output["择时策略"]["纯多头策略净值"]
        nID = iR2.shape[1]
        xData1 = np.arange(0, iR2.shape[0])
        xTicks1 = np.arange(0, iR2.shape[0], int(iR2.shape[0]/10))
        xTickLabels1 = [iR2.index[i].strftime("%Y-%m-%d") for i in xTicks1]
        xData2 = np.arange(0, iLSNV.shape[0])
        xTicks2 = np.arange(0, iLSNV.shape[0], int(iLSNV.shape[0]/10))
        xTickLabels2 = [iLSNV.index[i].strftime("%Y-%m-%d") for i in xTicks2]
        Fig = Figure(figsize=(32, 8*nID))
        for j, jID in enumerate(iTargetNV.columns):
            iAxes = Fig.add_subplot(nID, 3, j*3+1)
            iAxes.plot(xData1, iR2.iloc[:, j].values, label="R2", color="indianred", lw=2.5)
            iAxes.plot(xData1, iAdjR2.iloc[:, j].values, label="调整R2", color="steelblue", lw=2.5)
            iAxes.set_xticks(xTicks1)
            iAxes.set_xticklabels(xTickLabels1)
            iAxes.legend()
            iAxes.set_title(str(jID)+" : 滚动回归 R2")
            iAxes = Fig.add_subplot(nID, 3, j*3+2)
            iAxes.bar(xData1-0.3, iRealReturn.iloc[:, j].values, 0.3, color="indianred", label="实际收益率")
            iAxes.bar(xData1, iForecastReturn.iloc[:, j].values, 0.3, color="steelblue", label="预测收益率")
            iAxes.set_xticks(xTicks1)
            iAxes.set_xticklabels(xTickLabels1)
            iAxes.legend()
            iAxes.set_title(str(jID)+" : 滚动预测")
            iAxes = Fig.add_subplot(nID, 3, j*3+3)
            iAxes.plot(xData2, iLSNV.iloc[:, j].values, label="多空净值", color="steelblue", lw=2.5)
            iAxes.plot(xData2, iLNV.iloc[:, j].values, label="纯多头净值", color="indianred", lw=2.5)
            iAxes.plot(xData2, iTargetNV.iloc[:, j].values, label="标的净值", color="forestgreen", lw=2.5)
            iAxes.set_xticks(xTicks2)
            iAxes.set_xticklabels(xTickLabels2)
            iAxes.legend()
            iAxes.set_title(str(jID)+" : 择时策略")
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
        HTML += "全样本回归 - 回归系数 : "
        iHTML = self._Output["全样本回归"]["回归系数"].to_html(formatters=[lambda x:'{0:.4f}'.format(x)]*self._Output["全样本回归"]["回归系数"].shape[1])
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += "全样本回归 - t 统计量 : "
        iHTML = self._Output["全样本回归"]["t统计量"].to_html(formatters=[lambda x:'{0:.2f}'.format(x)]*self._Output["全样本回归"]["t统计量"].shape[1])
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