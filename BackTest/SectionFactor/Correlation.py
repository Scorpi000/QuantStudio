# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, Int, Str, Instance, Dict, Bool, on_trait_change
from traitsui.api import SetEditor, Item
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate
import matplotlib

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList
from QuantStudio.Tools.ExcelFun import copyChart
from QuantStudio.RiskModel.RiskDataSource import RiskDataSource
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage

class SectionCorrelation(BaseModule):
    """因子截面相关性"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    FactorOrder = Dict(key_trait=Str(), value_trait=Enum("降序", "升序"), arg_type="ArgDict", label="排序方向", order=1)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=2)
    CorrMethod = ListStr(["spearman"], arg_type="MultiOption", label="相关性算法", order=3, option_range=("spearman", "pearson", "kendall"))
    RiskDS = Instance(RiskDataSource, arg_type="RiskDS", label="风险数据源", order=4)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=5)
    def __init__(self, factor_table, name="因子截面相关性", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
    @on_trait_change("TestFactors[]")
    def _on_TestFactors_changed(self, obj, name, old, new):
        self.FactorOrder = {iFactorName:self.FactorOrder.get(iFactorName, "降序") for iFactorName in self.TestFactors}
    @on_trait_change("RiskDS")
    def _on_RiskDS_changed(self, obj, name, old, new):
        if new is None:
            self.add_trait("CorrMethod", ListStr(arg_type="MultiOption", label="相关性算法", order=3, option_range=("spearman", "pearson", "kendall")))
        else:
            self.add_trait("CorrMethod", ListStr(arg_type="MultiOption", label="相关性算法", order=3, option_range=("spearman", "pearson", "kendall", "factor-score correlation", "factor-portfolio correlation")))
        self.CorrMethod = list(set(self.CorrMethod).intersection(set(self.CorrMethod.option_range)))
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        Items[3].editor = SetEditor(values=self.trait("CorrMethod").option_range)
        return (Items, Context)
    def __QS_start__(self, mdl, dts, **kwargs):
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {"FactorPair":[]}
        for i, iFactor in enumerate(self.TestFactors):
            for j, jFactor in enumerate(self.TestFactors):
                if j>i: self._Output["FactorPair"].append(iFactor+"-"+jFactor)
        nPair = len(self._Output["FactorPair"])
        self._Output.update({iMethod:[[] for i in range(nPair)] for iMethod in self.CorrMethod})
        self._CorrMatrixNeeded = (("factor-score correlation" in self.CorrMethod) or ("factor-portfolio correlation" in self.CorrMethod))
        if self._CorrMatrixNeeded and (self.RiskDS is not None): self.RiskDS.start()
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.IDFilter)
        FactorExpose = self._FactorTable.readData(dts=[idt], ids=IDs, factor_names=list(self.TestFactors)).iloc[:, 0, :].astype("float")
        if self._CorrMatrixNeeded and (self.RiskDS is not None):
            self.RiskDS.move(idt)
            CovMatrix = self.RiskDS.getDateCov(idt, ids=IDs, drop_na=True)
            FactorIDs = {}
        else:
            CovMatrix = None
        PairInd = 0
        for i,iFactor in enumerate(self.TestFactors):
            iFactorExpose = FactorExpose[iFactor]
            if self._CorrMatrixNeeded:
                iIDs = FactorIDs.get(iFactor)
                if iIDs is None:
                    if CovMatrix is not None:
                        FactorIDs[iFactor] = list(set(CovMatrix.index).intersection(set(iFactorExpose[pd.notnull(iFactorExpose)].index)))
                    else:
                        FactorIDs[iFactor] = list(iFactorExpose[pd.notnull(iFactorExpose)].index)
                    iIDs = FactorIDs[iFactor]
            for j,jFactor in enumerate(self.TestFactors):
                if j>i:
                    jFactorExpose = FactorExpose[jFactor]
                    if self._CorrMatrixNeeded:
                        jIDs = FactorIDs.get(jFactor)
                        if jIDs is None:
                            if CovMatrix is not None:
                                FactorIDs[jFactor] = list(set(CovMatrix.index).intersection(set(jFactorExpose[pd.notnull(jFactorExpose)].index)))
                            else:
                                FactorIDs[jFactor] = list(jFactorExpose[pd.notnull(jFactorExpose)].index)
                            jIDs = FactorIDs[jFactor]
                        IDs = list(set(iIDs).intersection(set(jIDs)))
                        iTempExpose = iFactorExpose.loc[IDs].values
                        jTempExpose = jFactorExpose.loc[IDs].values
                        if CovMatrix is not None:
                            TempCovMatrix = CovMatrix.loc[IDs,IDs].values
                        else:
                            nID = len(IDs)
                            TempCovMatrix = np.eye(nID,nID)
                    for kMethod in self.CorrMethod:
                        if kMethod=="factor-score correlation":
                            ijCov = np.dot(iTempExpose.T,np.dot(TempCovMatrix,jTempExpose))
                            iStd = np.sqrt(np.dot(iTempExpose.T,np.dot(TempCovMatrix,iTempExpose)))
                            jStd = np.sqrt(np.dot(jTempExpose.T,np.dot(TempCovMatrix,jTempExpose)))
                            self._Output[kMethod][PairInd].append(ijCov/iStd/jStd)
                        elif kMethod=="factor-portfolio correlation":
                            TempCovMatrixInv = np.linalg.inv(TempCovMatrix)
                            ijCov = np.dot(iTempExpose.T,np.dot(TempCovMatrixInv,jTempExpose))
                            iStd = np.sqrt(np.dot(iTempExpose.T,np.dot(TempCovMatrixInv,iTempExpose)))
                            jStd = np.sqrt(np.dot(jTempExpose.T,np.dot(TempCovMatrixInv,jTempExpose)))
                            self._Output[kMethod][PairInd].append(ijCov/iStd/jStd)
                        else:
                            self._Output[kMethod][PairInd].append(FactorExpose[iFactor].corr(FactorExpose[jFactor], method=kMethod))
                    PairInd += 1
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        for iMethod in self.CorrMethod:
            self._Output[iMethod] = pd.DataFrame(np.array(self._Output[iMethod]).T, columns=self._Output["FactorPair"], index=self._Output["时点"])
            iAvgName = iMethod+"均值"
            self._Output[iAvgName] = pd.DataFrame(index=list(self.TestFactors), columns=list(self.TestFactors), dtype="float")
            for i, iFactor in enumerate(self.TestFactors):
                for j, jFactor in enumerate(self.TestFactors):
                    if j>i:
                        if self.FactorOrder[iFactor]!=self.FactorOrder[jFactor]:
                            self._Output[iMethod][iFactor+"-"+jFactor] = -self._Output[iMethod][iFactor+"-"+jFactor]
                        self._Output[iAvgName].loc[iFactor, jFactor] = self._Output[iMethod][iFactor+"-"+jFactor].mean()
                    elif j<i:
                        self._Output[iAvgName].loc[iFactor, jFactor] = self._Output[iAvgName].loc[jFactor,iFactor]
                    else:
                        self._Output[iAvgName].loc[iFactor, jFactor] = 1
        self._Output.pop("FactorPair")
        self._Output.pop("时点")
        if (self.RiskDS is not None) and self._CorrMatrixNeeded: self.RiskDS.end()
        return 0
    def genExcelReport(self, xl_book, sheet_name):
        xl_book.sheets.add(name=sheet_name)
        CurSheet = xl_book.sheets[sheet_name]
        nFactor = len(self.TestFactors)
        for j, jMethod in enumerate(self.CorrMethod):
            jData = self._Output[jMethod+"均值"]
            # 写入统计数据
            CurSheet[(nFactor+1)*j, 0].value = jData
            CurSheet[(nFactor+1)*j, 0].value = jMethod+"相关性"
            # 形成热图
            CurSheet[(nFactor+1)*j+1:(nFactor+1)*j+1+nFactor, 1:1+nFactor].select()
            iFormatConditions = xl_book.app.selection.api.FormatConditions
            iFormatConditions.AddColorScale(2)
            iFormatConditions(iFormatConditions.Count).SetFirstPriority()
            iFormatConditions(1).ColorScaleCriteria(1).Type = 1
            iFormatConditions(1).ColorScaleCriteria(1).FormatColor.Color = 16776444
            iFormatConditions(1).ColorScaleCriteria(1).FormatColor.TintAndShade = 0
            iFormatConditions(1).ColorScaleCriteria(2).Type = 2
            iFormatConditions(1).ColorScaleCriteria(2).FormatColor.Color = 7039480
            iFormatConditions(1).ColorScaleCriteria(2).FormatColor.TintAndShade = 0
        return 0
    def _plotHeatMap(self, axes, df, title):
        axes.pcolor(df.values, cmap=matplotlib.cm.Reds)
        axes.set_xticks(np.arange(df.shape[0])+0.5, minor=False)
        axes.set_yticks(np.arange(df.shape[1])+0.5, minor=False)
        axes.invert_yaxis()
        axes.xaxis.tick_top()
        axes.set_xticklabels(df.index.tolist(), minor=False)
        axes.set_yticklabels(df.columns.tolist(), minor=False)
        axes.set_title(title)
        return axes
    def genMatplotlibFig(self, file_path=None):
        nMethod = len(self.CorrMethod)
        nRow, nCol = nMethod//3+1, min(3, nMethod)
        Fig = plt.figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        AxesGrid = gridspec.GridSpec(nRow, nCol)
        for i, iMethod in enumerate(self.CorrMethod):
            iAvgName = iMethod+"均值"
            iAxes = plt.subplot(AxesGrid[i//nCol, i%nCol])
            self._plotHeatMap(iAxes, self._Output[iAvgName], iAvgName)
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self.ArgNames:
                if iArgName=="风险数据源":
                    if self.RiskDS is None: HTML += "<li>"+iArgName+": None</li>"
                    else: HTML += "<li>"+iArgName+": "+self.RiskDS.Name+"</li>"
                elif iArgName!="计算时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        for i, iMethod in enumerate(self.CorrMethod):
            iAvgName = iMethod+"均值"
            iHTML = self._Output[iAvgName].style.background_gradient(cmap="Reds").set_precision(2).render()
            HTML += '<div align="left" style="font-size:1em"><strong>'+iAvgName+'</strong></div>'+iHTML
        return HTML

class FactorTurnover(BaseModule):
    """因子换手率"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=1)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=2)
    def __init__(self, factor_table, name="因子换手率", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return (Items, Context)
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        self._Output = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt):
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            LastInd = self._CurCalcInd - 1
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            LastInd = self._CurCalcInd - 1
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if LastInd<0:
            for iFactorName in self.TestFactors:
                self._Output[iFactorName].append(0.0)
            self._Output["时点"].append(idt)
            return 0
        LastIDs = self._FactorTable.getFilteredID(idt=LastDateTime, id_filter_str=self.IDFilter)
        PreFactorExpose = self._FactorTable.readData(dts=[LastDateTime], ids=LastIDs, factor_names=list(self.TestFactors)).iloc[:,0,:].astype("float")
        CurFactorExpose = self._FactorTable.readData(dts=[idt], ids=LastIDs, factor_names=list(self.TestFactors)).iloc[:,0,:].astype("float")
        for iFactorName in self.TestFactors: self._Output[iFactorName].append(CurFactorExpose[iFactorName].corr(PreFactorExpose[iFactorName]))
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        self._Output = {"因子换手率":pd.DataFrame(self._Output, index=self._Output.pop("时点"))}
        self._Output["统计数据"] = pd.DataFrame(self._Output["因子换手率"].mean(), columns=["平均值"])
        self._Output["统计数据"]["标准差"] = self._Output["因子换手率"].std()
        self._Output["统计数据"]["最小值"] = self._Output["因子换手率"].min()
        self._Output["统计数据"]["最大值"] = self._Output["因子换手率"].max()
        self._Output["统计数据"]["中位数"] = self._Output["因子换手率"].median()
        return 0
    def genExcelReport(self, xl_book, sheet_name):
        xl_book.sheets["因子换手率"].api.Copy(Before=xl_book.sheets[0].api)
        xl_book.sheets[0].name = sheet_name
        CurSheet = xl_book.sheets[sheet_name]
        nDT = self._Output["因子换手率"].shape[0]
        # 写入统计数据
        CurSheet[1, 0].expand().clear_contents()
        CurSheet[1, 0].options(transpose=True).value = list(self._Output["统计数据"].index)
        FormatFun = np.vectorize(lambda x:("%.2f%%" % x) if pd.notnull(x) else None)
        CurSheet[1, 1].value = FormatFun(self._Output["统计数据"].values*100)
        CurSheet.api.ListObjects(1).Resize(CurSheet[0:self._Output["统计数据"].shape[0]+1, 0:self._Output["统计数据"].shape[1]+1].api)
        # 写入日期序列
        CurSheet[0, 7].expand().clear_contents()
        CurSheet[0, 7].value = "时点"
        CurSheet[1, 7].options(transpose=True).value = [iDT.strftime("%Y-%m-%d") for iDT in self._Output["因子换手率"].index]
        # 写入时间序列数据
        for i, iFactor in enumerate(self._Output["统计数据"].index):
            iCol = 9 + i
            CurSheet[0, iCol-1].value = iFactor+"-截面相关性"
            CurSheet[1, iCol-1].value = FormatFun(self._Output["因子换手率"][[iFactor]].values*100)
            # 绘制图线
            Chrt = copyChart(xl_book, sheet_name, "因子换手率", 6, iCol-1, sheet_name, iFactor+"-因子换手率").api[1]
            Chrt.SeriesCollection(1).Values = CurSheet[1:nDT+1, iCol-1].api
            Chrt.SeriesCollection(1).Name = iFactor+"-截面相关性"
            Chrt.SeriesCollection(1).XValues = CurSheet[1:nDT+1, 7].api
        CurSheet.charts["因子换手率"].delete()
        return 0
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = self._Output["因子换手率"].shape[1]//3+1, min(3, self._Output["因子换手率"].shape[1])
        Fig = plt.figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        AxesGrid = gridspec.GridSpec(nRow, nCol)
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        for i in range(self._Output["因子换手率"].shape[1]):
            iAxes = plt.subplot(AxesGrid[i//nCol, i%nCol])
            iAxes.yaxis.set_major_formatter(yMajorFormatter)
            iAxes.xaxis_date()
            iAxes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
            iAxes.stackplot(self._Output["因子换手率"].index, self._Output["因子换手率"].iloc[:, i].values, color="b")
            iAxes.set_title(self._Output["因子换手率"].columns[i])
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
        iHTML += self._Output["统计数据"].to_html(formatters=[_QS_formatPandasPercentage]*5)
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