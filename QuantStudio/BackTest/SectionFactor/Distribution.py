# coding=utf-8
import base64
from io import BytesIO
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict
from traitsui.api import SetEditor, Item
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate
import matplotlib

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage

class IndustryDistribution(BaseModule):
    """因子值行业分布"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    #IndustryFactor = Enum(None, arg_type="SingleOption", label="行业因子", order=1)
    Threshold = Enum("中位数","平均数","25%分位数","75%分位数",arg_type="SingleOption", label="阈值", order=2)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=3)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=4)
    def __init__(self, factor_table, name="因子值行业分布", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("IndustryFactor", Enum(*DefaultStrFactorList, arg_type="SingleOption", label="行业因子", order=1))
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return (Items, Context)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        AllIndustries = pd.unique(self._FactorTable.readData(factor_names=[self.IndustryFactor], dts=self._FactorTable.getDateTime(ifactor_name=self.IndustryFactor), ids=self._FactorTable.getID(ifactor_name=self.IndustryFactor)).iloc[0].values.flatten())
        Mask = pd.isnull(AllIndustries)
        if np.sum(Mask)>0: AllIndustries = AllIndustries[~Mask].tolist()+[None]
        self._Output = {iFactorName:{iIndustry:[] for iIndustry in AllIndustries} for iFactorName in self.TestFactors}
        self._Output["历史平均值"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["历史标准差"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["行业分类"] = AllIndustries
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.IDFilter)
        FactorExpose = self._FactorTable.readData(dts=[idt], ids=IDs, factor_names=list(self.TestFactors)+[self.IndustryFactor]).iloc[:,0,:]
        IndustryData, FactorExpose = FactorExpose.iloc[:, -1], FactorExpose.iloc[:, :-1].astype("float")
        Threshold = {}
        Mask = {}
        for iFactorName in self.TestFactors:
            Mask[iFactorName] = pd.notnull(FactorExpose[iFactorName])
            if self.Threshold=="中位数":
                Threshold[iFactorName] = FactorExpose[iFactorName].median()
            elif self.Threshold=="平均值":
                Threshold[iFactorName] = FactorExpose[iFactorName].mean()
            elif self.Threshold=="25%分位数":
                Threshold[iFactorName] = FactorExpose[iFactorName].quantile(0.25)
            elif self.Threshold=="75%分位数":
                Threshold[iFactorName] = FactorExpose[iFactorName].quantile(0.75)
        for jIndustry in self._Output["行业分类"]:
            if pd.isnull(jIndustry): jMask = pd.isnull(IndustryData)
            else: jMask = (IndustryData==jIndustry)
            for iFactorName in self.TestFactors:
                ijMask = (jMask & Mask[iFactorName])
                ijNum = ijMask.sum()
                if ijNum!=0: self._Output[iFactorName][jIndustry].append((FactorExpose[iFactorName][ijMask]>=Threshold[iFactorName]).sum()/ijNum)
                else: self._Output[iFactorName][jIndustry].append(np.nan)
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        for iFactorName in self.TestFactors:
            self._Output[iFactorName] = pd.DataFrame(self._Output[iFactorName], index=self._Output["时点"], columns=self._Output["行业分类"])
            self._Output["历史平均值"][iFactorName] = self._Output[iFactorName].mean()
            self._Output["历史标准差"][iFactorName] = self._Output[iFactorName].std()
        self._Output["历史平均值"] = pd.DataFrame(self._Output["历史平均值"], columns=list(self.TestFactors))
        self._Output["历史标准差"] = pd.DataFrame(self._Output["历史标准差"], columns=list(self.TestFactors))
        self._Output.pop("行业分类")
        self._Output.pop("时点")
        return 0
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = self._Output["历史平均值"].shape[1]//3+(self._Output["历史平均值"].shape[1]%3!=0), min(3, self._Output["历史平均值"].shape[1])
        Fig = plt.figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        AxesGrid = gridspec.GridSpec(nRow, nCol)
        xData = np.arange(0, self._Output["历史平均值"].shape[0])
        xTickLabels = [str(iIndustry) for iIndustry in self._Output["历史平均值"].index]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        for i, iFactorName in enumerate(self._Output["历史平均值"].columns):
            iAxes = plt.subplot(AxesGrid[i//nCol, i%nCol])
            iAxes.yaxis.set_major_formatter(yMajorFormatter)
            iAxes.bar(xData, self._Output["历史平均值"].iloc[:, i].values, color="b", label="历史平均值")
            iAxes.set_title(iFactorName+"-历史平均值")
            iAxes.set_xticks(xData)
            iAxes.set_xticklabels(xTickLabels)
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
        Formatters = [_QS_formatPandasPercentage]*self._Output["历史平均值"].shape[1]
        iHTML = self._Output["历史平均值"].to_html(formatters=Formatters)
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