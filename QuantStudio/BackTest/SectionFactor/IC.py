# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, ListInt, Int, Str
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.SectionFactor.BTSectionOperator import IC as ICOperator
import QuantStudio.FactorDataBase.FactorOperators as fo


def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x*100, )

def _QS_formatPandasPercentage(x):
    return '{0:.2f}%'.format(x*100)

class IC(BaseModule):
    """IC"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        FactorOrder = Enum("降序", "升序", arg_type="SingleOption", label="排序方向", order=0)
        CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="计算时点", order=1)
        LookBack = Int(1, arg_type="Integer", label="回溯期数", order=2)
        CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=3, option_range=["spearman", "pearson", "kendall"])
        RollAvgPeriod = Int(12, arg_type="Integer", label="滚动平均期数", order=4)
        
    def __init__(self, price:Factor, *factors, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, weight:Optional[Factor]=None, section_ids=None, name="IC", sys_args={}, **kwargs):
        self._Price = price
        self._Factors = factors
        self._Mask = mask
        self._CatData = cat_data
        self._Weight = weight
        self._SectionIDs = section_ids
        self._FactorQSIDs = [iFactor._QSID for iFactor in self._Factors]
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    
    def __QS_start__(self, mdl, dts, **kwargs):
        Tasks = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        self._IC = ICOperator(sys_args={
            "参数": {
                "排序方向": self._QSArgs.FactorOrder,
                "回溯期数": self._QSArgs.LookBack,
                "相关性算法": self._QSArgs.CorrMethod
            }
        })(self._Price, *self._Factors, mask=self._Mask, cat_data=self._CatData, weight=self._Weight, descriptor_ids=self._SectionIDs, factor_name=self.Name, factor_args={"截面ID": self._FactorQSIDs, "计算时点标尺": self._QSArgs.CalcDTs})
        notnull = fo.NotNull()
        calcBreadth = fo.Aggregate(aggr_func=np.nansum, descriptor_ids=self._SectionIDs)
        Breadths = [calcBreadth((self._Mask & notnull(iFactor)) if self._Mask else notnull(iFactor), factor_args={"计算时点标尺": self._QSArgs.CalcDTs}) for i, iFactor in enumerate(self._Factors)]
        self._Breadth = fo.ConcatSection(descriptor_sections=[[iID] for iID in self._FactorQSIDs])(*Breadths, factor_name=self.Name, factor_args={"截面ID": self._FactorQSIDs, "计算时点标尺": self._QSArgs.CalcDTs})
        Tasks += [(self._IC, self._FactorQSIDs), (self._Breadth, self._FactorQSIDs)]
        return Tasks
    
    def __QS_end__(self, factor_data):
        super().__QS_end__(factor_data)
        NameMappings = {iFactor._QSID: iFactor.Name for iFactor in self._Factors}
        self._Output["IC"] = factor_data[self._IC._QSID].rename(columns=NameMappings)
        self._Output["截面宽度"] = factor_data[self._Breadth._QSID].rename(columns=NameMappings)
        if self._QSArgs.CalcDTs:
            CalcDTs = sorted(self._Output["IC"].index.intersection(self._QSArgs.CalcDTs))
            self._Output["IC"] = self._Output["IC"].reindex(index=CalcDTs)
            self._Output["截面宽度"] = self._Output["截面宽度"].reindex(index=CalcDTs)
        self._Output["IC的移动平均"] = self._Output["IC"].copy()
        for i in range(self._Output["IC"].shape[0]):
            if i < self._QSArgs.RollAvgPeriod-1: self._Output["IC的移动平均"].iloc[i, :] = np.nan
            else: self._Output["IC的移动平均"].iloc[i, :] = self._Output["IC"].iloc[i-self._QSArgs.RollAvgPeriod+1:i+1, :].mean()
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["IC"].columns)
        self._Output["统计数据"]["平均值"] = self._Output["IC"].mean()
        self._Output["统计数据"]["标准差"] = self._Output["IC"].std()
        self._Output["统计数据"]["最小值"] = self._Output["IC"].min()
        self._Output["统计数据"]["最大值"] = self._Output["IC"].max()
        self._Output["统计数据"]["IC_IR"] = self._Output["统计数据"]["平均值"] / self._Output["统计数据"]["标准差"]
        self._Output["统计数据"]["有效期数"] = self._Output["IC"].notnull().sum()
        self._Output["统计数据"]["t统计量"] = self._Output["统计数据"]["IC_IR"] * self._Output["统计数据"]["有效期数"]**0.5
        self._Output["统计数据"]["平均截面宽度"] = self._Output["截面宽度"].mean()
        self._Output["统计数据"]["IC×Sqrt(N)"] = self._Output["统计数据"]["平均值"] * np.sqrt(self._Output["统计数据"]["平均截面宽度"])
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
            iRightAxes = iAxes.twinx()
            iRightAxes.plot(xData, self._Output["截面宽度"].iloc[:, i].values, label="截面宽度", color="k", lw=1.5)
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iRightAxes.legend(loc="upper right")
            iAxes.set_title(self._Output["IC"].columns[i])
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
        Formatters = [_QS_formatPandasPercentage]*4+[lambda x:'{0:.4f}'.format(x), lambda x:'{0:.0f}'.format(x)]+[lambda x:'{0:.2f}'.format(x)]*3
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
    class __QS_ArgClass__(IC.__QS_ArgClass__):
        # RiskFactors = ListStr(arg_type="MultiOption", label="风险因子", order=2.5, option_range=())
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.remove_trait("WeightFactor")
            DefaultNumFactorList, _ = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("RiskFactors", ListStr(arg_type="MultiOption", label="风险因子", order=2.5, option_range=tuple(DefaultNumFactorList)))
    
    def __init__(self, factor_table, name="风险调整的 IC", sys_args={}, **kwargs):
        return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self._QSArgs.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self._QSArgs.CalcDTs[PreInd]
            LastDateTime = self._QSArgs.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self._QSArgs.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0):
            for iFactorName in self._QSArgs.TestFactors:
                self._Output["IC"][iFactorName].append(np.nan)
                self._Output["截面宽度"][iFactorName].append(np.nan)
            self._Output["时点"].append(idt)
            return 0
        PreIDs = self._FactorTable.getFilteredID(idt=PreDateTime, id_filter_str=self._QSArgs.IDFilter)
        FactorExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self._QSArgs.TestFactors)).iloc[:,0,:]
        if self._QSArgs.RiskFactors:
            RiskExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self._QSArgs.RiskFactors)).iloc[:,0,:]
            RiskExpose["constant"] = 1.0
        else:
            RiskExpose = pd.DataFrame(1.0, index=PreIDs, columns=["constant"])
        CurPrice = self._FactorTable.readData(dts=[idt], ids=PreIDs, factor_names=[self._QSArgs.PriceFactor]).iloc[0,0,:]
        LastPrice = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self._QSArgs.PriceFactor]).iloc[0,0,:]
        Ret = CurPrice/LastPrice-1
        Mask = (pd.isnull(RiskExpose).sum(axis=1)==0)
        # 展开Dummy因子
        if self._QSArgs.ClassFactor!="无":
            DummyFactorData = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=[self._QSArgs.ClassFactor]).iloc[0,0,:]
            _,_,_,DummyFactorData = prepareRegressData(np.ones(DummyFactorData.shape[0]), dummy_data=DummyFactorData.values)
        iMask = (pd.notnull(Ret) & Mask)
        Ret = Ret[iMask]
        iX = RiskExpose.loc[iMask].values
        if self._QSArgs.ClassFactor!="无":
            iDummy = DummyFactorData[iMask.values]
            iDummy = iDummy[:,(np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
            iX = np.hstack((iX,iDummy[:,:-1]))
        try:
            Result = sm.OLS(Ret.values, iX, missing="drop").fit()
        except:
            return self._moveNone(idt)
        RiskAdjustedRet = pd.Series(Result.resid, index=Ret.index)
        for iFactorName in self._QSArgs.TestFactors:
            iFactorExpose = FactorExpose[iFactorName]
            iMask = (Mask & pd.notnull(iFactorExpose))
            iFactorExpose = iFactorExpose[iMask]
            iX = RiskExpose.loc[iMask].values
            if self._QSArgs.ClassFactor!="无":
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
            self._Output["IC"][iFactorName].append(iFactorExpose.corr(RiskAdjustedRet, method=self._QSArgs.CorrMethod))
            self._Output["截面宽度"][iFactorName].append(pd.notnull(iFactorExpose).sum())
        self._Output["时点"].append(idt)
        return 0

class ICDecay(BaseModule):
    """IC 衰减"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        #TestFactor = Enum(None, arg_type="SingleOption", label="测试因子", order=0)
        FactorOrder = Enum("降序","升序", arg_type="SingleOption", label="排序方向", order=1, option_range=["降序","升序"])
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
        #ClassFactor = Enum("无", arg_type="SingleOption", label="类别因子", order=3)
        #WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=4)
        CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="计算时点", order=5)
        LookBack = ListInt(np.arange(1,13).tolist(), arg_type="NultiOpotion", label="回溯期数", order=6)
        CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=7, option_range=["spearman", "pearson", "kendall"])
        IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=8)
        def __QS_initArgs__(self, args={}):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("TestFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="测试因子", order=0, option_range=DefaultNumFactorList))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2, option_range=DefaultNumFactorList))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            self.add_trait("ClassFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="类别因子", order=3, option_range=["无"]+DefaultStrFactorList))
            self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=4, option_range=["等权"]+DefaultNumFactorList))
    
    def __init__(self, factor_table, name="IC 衰减", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {"IC":[[] for i in self._QSArgs.LookBack]}
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            LastInd = self._CurCalcInd - 1
            LastDateTime = self._QSArgs.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            LastInd = self._CurCalcInd - 1
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (LastInd<0):
            for i, iRollBack in enumerate(self._QSArgs.LookBack):
                self._Output["IC"][i].append(np.nan)
            self._Output["时点"].append(idt)
            return 0
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=self._FactorTable.getID(ifactor_name=self._QSArgs.PriceFactor), factor_names=[self._QSArgs.PriceFactor]).iloc[0]
        Ret = Price.iloc[1] / Price.iloc[0] - 1
        for i, iRollBack in enumerate(self._QSArgs.LookBack):
            iPreInd = self._CurCalcInd - iRollBack
            if iPreInd<0:
                self._Output["IC"][i].append(np.nan)
                continue
            iPreDT = self._QSArgs.CalcDTs[iPreInd]
            iPreIDs = self._FactorTable.getFilteredID(idt=iPreDT, id_filter_str=self._QSArgs.IDFilter)
            iRet = Ret.reindex(index=iPreIDs, copy=True)
            if self._QSArgs.ClassFactor!="无":
                IndustryData = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self._QSArgs.ClassFactor]).iloc[0,0,:]
                AllIndustry = IndustryData.unique()
                # 进行收益率的类别调整
                if self._QSArgs.WeightFactor=="等权":
                    for iIndustry in AllIndustry:
                        iRet[IndustryData==iIndustry] -= iRet[IndustryData==iIndustry].mean()
                else:
                    WeightData = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self._QSArgs.WeightFactor]).iloc[0,0,:]
                    for iIndustry in AllIndustry:
                        iWeight = WeightData[IndustryData==iIndustry]
                        iiRet = iRet[IndustryData==iIndustry]
                        iRet[IndustryData==iIndustry] -= (iiRet * iWeight).sum() / iWeight[pd.notnull(iWeight) & pd.notnull(iiRet)].sum(skipna=False)
            iFactorExpose = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self._QSArgs.TestFactor]).iloc[0,0,:]
            self._Output["IC"][i].append(iFactorExpose.corr(iRet, method=self._QSArgs.CorrMethod))
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        self._Output["IC"] = pd.DataFrame(np.array(self._Output["IC"]).T, index=self._Output.pop("时点"), columns=list(self._QSArgs.LookBack))
        if self._QSArgs.FactorOrder=="升序": self._Output["IC"] = -self._Output["IC"]
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