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
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.BackTest.BackTestModel import BaseModule

def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x*100, )
def _QS_formatPandasPercentage(x):
    return '{0:.2f}%'.format(x*100)
class IC(BaseModule):
    """IC"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    FactorOrder = Dict(key_trait=Str(), value_trait=Enum("降序", "升序"), arg_type="ArgDict", label="排序方向", order=1)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
    #ClassFactor = Enum("无", arg_type="SingleOption", label="类别因子", order=3)
    #WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    LookBack = Int(1, arg_type="Integer", label="回溯期数", order=6)
    CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=7)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=8)
    RollAvgPeriod = Int(12, arg_type="Integer", label="滚动平均期数", order=9)
    def __init__(self, factor_table, name="IC", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("ClassFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="类别因子", order=3))
        self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=4))
    @on_trait_change("TestFactors[]")
    def _on_TestFactors_changed(self, obj, name, old, new):
        self.FactorOrder = {iFactorName:self.FactorOrder.get(iFactorName, "降序") for iFactorName in self.TestFactors}
    def getViewItems(self, context_name=""):
        Items, Context = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return (Items, Context)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._Output["IC"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["截面宽度"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self.CalcDTs[PreInd]
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0):
            for iFactorName in self.TestFactors:
                self._Output["IC"][iFactorName].append(np.nan)
                self._Output["截面宽度"][iFactorName].append(np.nan)
            self._Output["时点"].append(idt)
            return 0
        PreIDs = self._FactorTable.getFilteredID(idt=PreDateTime, id_filter_str=self.IDFilter)
        FactorExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self.TestFactors)).iloc[:, 0, :]
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=PreIDs, factor_names=[self.PriceFactor]).iloc[0, :, :]
        Ret = Price.iloc[-1] / Price.iloc[0] - 1
        if self.ClassFactor!="无":# 进行收益率的类别调整
            IndustryData = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.ClassFactor]).iloc[0, 0, :]
            AllIndustry = IndustryData.unique()
            if self.WeightFactor=="等权":
                for iIndustry in AllIndustry:
                    iMask = (IndustryData==iIndustry)
                    Ret[iMask] -= Ret[iMask].mean()
            else:
                WeightData = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.WeightFactor]).iloc[0, 0, :]
                for iIndustry in AllIndustry:
                    iMask = (IndustryData==iIndustry)
                    iWeight = WeightData[iMask]
                    iRet = Ret[iMask]
                    Ret[iMask] -= (iRet*iWeight).sum() / iWeight[pd.notnull(iWeight) & pd.notnull(iRet)].sum(skipna=False)
        for iFactorName in self.TestFactors:
            self._Output["IC"][iFactorName].append(FactorExpose[iFactorName].corr(Ret, method=self.CorrMethod))
            self._Output["截面宽度"][iFactorName].append(pd.notnull(FactorExpose[iFactorName]).sum())
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        CalcDateTimes = self._Output.pop("时点")
        self._Output["截面宽度"] = pd.DataFrame(self._Output["截面宽度"], index=CalcDateTimes)
        self._Output["IC"] = pd.DataFrame(self._Output["IC"], index=CalcDateTimes)
        for i, iFactorName in enumerate(self.TestFactors):
            if self.FactorOrder[iFactorName]=="升序": self._Output["IC"][iFactorName] = -self._Output["IC"][iFactorName]
        self._Output["IC的移动平均"] = self._Output["IC"].copy()
        for i in range(len(CalcDateTimes)):
            if i<self.RollAvgPeriod-1: self._Output["IC的移动平均"].iloc[i,:] = np.nan
            else: self._Output["IC的移动平均"].iloc[i,:] = self._Output["IC"].iloc[i-self.RollAvgPeriod+1:i+1, :].mean()
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["IC"].columns)
        self._Output["统计数据"]["平均值"] = self._Output["IC"].mean()
        self._Output["统计数据"]["标准差"] = self._Output["IC"].std()
        self._Output["统计数据"]["最小值"] = self._Output["IC"].min()
        self._Output["统计数据"]["最大值"] = self._Output["IC"].max()
        self._Output["统计数据"]["IC_IR"] = self._Output["统计数据"]["平均值"] / self._Output["统计数据"]["标准差"]
        self._Output["统计数据"]["t统计量"] = np.nan
        self._Output["统计数据"]["平均截面宽度"] = self._Output["截面宽度"].mean()
        self._Output["统计数据"]["IC×Sqrt(N)"] = self._Output["统计数据"]["平均值"]*np.sqrt(self._Output["统计数据"]["平均截面宽度"])
        self._Output["统计数据"]["有效期数"] = 0.0
        for iFactor in self._Output["IC"]: self._Output["统计数据"].loc[iFactor,"有效期数"] = pd.notnull(self._Output["IC"][iFactor]).sum()
        self._Output["统计数据"]["t统计量"] = (self._Output["统计数据"]["有效期数"]**0.5)*self._Output["统计数据"]["IC_IR"]
        return 0
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = self._Output["IC"].shape[1]//3+(self._Output["IC"].shape[1]%3!=0), min(3, self._Output["IC"].shape[1])
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(0, self._Output["IC"].shape[0])
        xTicks = np.arange(0, self._Output["IC"].shape[0], max(1, int(self._Output["IC"].shape[0]/10)))
        xTickLabels = [self._Output["IC"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        for i in range(self._Output["IC"].shape[1]):
            iAxes = Fig.add_subplot(nRow, nCol, i+1)
            iAxes.yaxis.set_major_formatter(yMajorFormatter)
            iAxes.plot(xData, self._Output["IC的移动平均"].iloc[:, i].values, label="IC的移动平均", color="indianred", lw=2.5)
            iAxes.bar(xData, self._Output["IC"].iloc[:, i].values, label="IC", color="steelblue")
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc='best')
            iAxes.set_title(self._Output["IC"].columns[i])
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
        Formatters = [_QS_formatPandasPercentage]*4+[lambda x:'{0:.4f}'.format(x)]+[lambda x:'{0:.2f}'.format(x)]*3+[lambda x:'{0:.0f}'.format(x)]
        iHTML = self._Output["统计数据"].to_html(formatters=Formatters)
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


class RiskAdjustedIC(IC):    
    """风险调整的 IC"""
    RiskFactors = ListStr(arg_type="MultiOption", label="风险因子", order=2.5, option_range=())
    def __init__(self, factor_table, name="风险调整的 IC", sys_args={}, **kwargs):
        return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.remove_trait("WeightFactor")
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self.CalcDTs[PreInd]
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0):
            for iFactorName in self.TestFactors:
                self._Output["IC"][iFactorName].append(np.nan)
                self._Output["截面宽度"][iFactorName].append(np.nan)
            self._Output["时点"].append(idt)
            return 0
        PreIDs = self._FactorTable.getFilteredID(idt=PreDateTime, id_filter_str=self.IDFilter)
        FactorExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self.TestFactors)).iloc[:,0,:]
        if self.RiskFactors:
            RiskExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self.RiskFactors)).iloc[:,0,:]
            RiskExpose["constant"] = 1.0
        else:
            RiskExpose = pd.DataFrame(1.0, index=PreIDs, columns=["constant"])
        CurPrice = self._FactorTable.readData(dts=[idt], ids=PreIDs, factor_names=[self.PriceFactor]).iloc[0,0,:]
        LastPrice = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.PriceFactor]).iloc[0,0,:]
        Ret = CurPrice/LastPrice-1
        Mask = (pd.isnull(RiskExpose).sum(axis=1)==0)
        # 展开Dummy因子
        if self.ClassFactor!="无":
            DummyFactorData = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=[self.ClassFactor]).iloc[0,0,:]
            _,_,_,DummyFactorData = prepareRegressData(np.ones(DummyFactorData.shape[0]), dummy_data=DummyFactorData.values)
        iMask = (pd.notnull(Ret) & Mask)
        Ret = Ret[iMask]
        iX = RiskExpose.loc[iMask].values
        if self.ClassFactor!="无":
            iDummy = DummyFactorData[iMask.values]
            iDummy = iDummy[:,(np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
            iX = np.hstack((iX,iDummy[:,:-1]))
        try:
            Result = sm.OLS(Ret.values, iX, missing="drop").fit()
        except:
            return self._moveNone(idt)
        RiskAdjustedRet = pd.Series(Result.resid, index=Ret.index)
        for iFactorName in self.TestFactors:
            iFactorExpose = FactorExpose[iFactorName]
            iMask = (Mask & pd.notnull(iFactorExpose))
            iFactorExpose = iFactorExpose[iMask]
            iX = RiskExpose.loc[iMask].values
            if self.ClassFactor!="无":
                iDummy = DummyFactorData[iMask.values]
                iDummy = iDummy[:,(np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
                iX = np.hstack((iX,iDummy[:,:-1]))
            try:
                Result = sm.OLS(iFactorExpose.values,iX,missing="drop").fit()
            except:
                self._Output["IC"][iFactorName].append(np.nan)
                self._Output["截面宽度"][iFactorName].append(0)
                continue
            iFactorExpose = pd.Series(Result.resid,index=iFactorExpose.index)
            self._Output["IC"][iFactorName].append(iFactorExpose.corr(RiskAdjustedRet, method=self.CorrMethod))
            self._Output["截面宽度"][iFactorName].append(pd.notnull(iFactorExpose).sum())
        self._Output["时点"].append(idt)
        return 0

class ICDecay(BaseModule):
    """IC 衰减"""
    #TestFactor = Enum(None, arg_type="SingleOption", label="测试因子", order=0)
    FactorOrder = Enum("降序","升序", arg_type="SingleOption", label="排序方向", order=1)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
    #ClassFactor = Enum("无", arg_type="SingleOption", label="类别因子", order=3)
    #WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    LookBack = ListInt(np.arange(1,13).tolist(), arg_type="NultiOpotion", label="回溯期数", order=6)
    CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=7)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=8)
    def __init__(self, factor_table, name="IC 衰减", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="测试因子", order=0))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("ClassFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="类别因子", order=3))
        self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=4))
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {"IC":[[] for i in self.LookBack]}
        self._Output["时点"] = []
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
        if (LastInd<0):
            for i, iRollBack in enumerate(self.LookBack):
                self._Output["IC"][i].append(np.nan)
            self._Output["时点"].append(idt)
            return 0
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=self._FactorTable.getID(ifactor_name=self.PriceFactor), factor_names=[self.PriceFactor]).iloc[0]
        Ret = Price.iloc[1] / Price.iloc[0] - 1
        for i, iRollBack in enumerate(self.LookBack):
            iPreInd = self._CurCalcInd - iRollBack
            if iPreInd<0:
                self._Output["IC"][i].append(np.nan)
                continue
            iPreDT = self.CalcDTs[iPreInd]
            iPreIDs = self._FactorTable.getFilteredID(idt=iPreDT, id_filter_str=self.IDFilter)
            iRet = Ret.loc[iPreIDs].copy()
            if self.ClassFactor!="无":
                IndustryData = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self.ClassFactor]).iloc[0,0,:]
                AllIndustry = IndustryData.unique()
                # 进行收益率的类别调整
                if self.WeightFactor=="等权":
                    for iIndustry in AllIndustry:
                        iRet[IndustryData==iIndustry] -= iRet[IndustryData==iIndustry].mean()
                else:
                    WeightData = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self.WeightFactor]).iloc[0,0,:]
                    for iIndustry in AllIndustry:
                        iWeight = WeightData[IndustryData==iIndustry]
                        iiRet = iRet[IndustryData==iIndustry]
                        iRet[IndustryData==iIndustry] -= (iiRet * iWeight).sum() / iWeight[pd.notnull(iWeight) & pd.notnull(iiRet)].sum(skipna=False)
            iFactorExpose = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self.TestFactor]).iloc[0,0,:]
            self._Output["IC"][i].append(iFactorExpose.corr(iRet, method=self.CorrMethod))
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        self._Output["IC"] = pd.DataFrame(np.array(self._Output["IC"]).T, index=self._Output.pop("时点"), columns=list(self.LookBack))
        if self.FactorOrder=="升序": self._Output["IC"] = -self._Output["IC"]
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["IC"].columns)
        self._Output["统计数据"]["IC平均值"] = self._Output["IC"].mean()
        nDT = pd.notnull(self._Output["IC"]).sum()
        self._Output["统计数据"]["标准差"] = self._Output["IC"].std()
        self._Output["统计数据"]["IC_IR"] = self._Output["统计数据"]["IC平均值"] / self._Output["统计数据"]["标准差"]
        self._Output["统计数据"]["t统计量"] = self._Output["统计数据"]["IC_IR"] * nDT**0.5
        self._Output["统计数据"]["胜率"] = (self._Output["IC"]>0).sum() / nDT
        return 0
    def genMatplotlibFig(self, file_path=None):
        Fig = Figure(figsize=(16, 8))
        xData = np.arange(0, self._Output["统计数据"].shape[0])
        xTickLabels = [str(i) for i in self._Output["统计数据"].index]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        Axes = Fig.add_subplot(1, 1, 1)
        Axes.yaxis.set_major_formatter(yMajorFormatter)
        Axes.bar(xData, self._Output["统计数据"]["IC平均值"].values, label="IC", color="steelblue")
        Axes.set_xticks(xData)
        Axes.set_xticklabels(xTickLabels)
        Axes.legend(loc='upper left')
        RAxes = Axes.twinx()
        RAxes.yaxis.set_major_formatter(yMajorFormatter)
        RAxes.plot(xData, self._Output["统计数据"]["胜率"].values, label="胜率", color="indianred", lw=2.5)
        RAxes.legend(loc="upper right")
        plt.setp(Axes.get_xticklabels(), visible=True, rotation=0, ha='center')
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
        Formatters = [_QS_formatPandasPercentage]*2+[lambda x:'{0:.4f}'.format(x), lambda x:'{0:.2f}'.format(x), _QS_formatPandasPercentage]
        iHTML = self._Output["统计数据"].to_html(formatters=Formatters)
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