# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, Literal, List, Dict, Any
from multiprocessing import Queue, Event

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Factor import Factor, FactorContext, FactorInitData
from QuantStudio.Core.FactorOperation import PanelOperator, SectionOperator
from QuantStudio.BackTest.BackTestModel import BTLocalContext, BTOutputNode, BTReportNode, BTInitData
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData

def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x*100, )

def _QS_formatPandasPercentage(x):
    return '{0:.2f}%'.format(x*100)

class Breadth(SectionOperator):
    def __init__(self, descriptor_ids=None, args={}, config_file=None, **kwargs):
        Args = {"Name": "Breadth"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DescriptorSection": [descriptor_ids], "DataType": "double"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
        else:
            DTs = idt
        if f.UserData["mask"]: 
            Mask, x = pd.DataFrame(x[0]==1, index=idt, columns=SectionIDs), x[1:]
            Mask = Mask.reindex(columns=DTs).fillna(False)
        else:
            Mask = pd.DataFrame(True, index=DTs, columns=SectionIDs)
        Breadth = pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        Mask = Mask.shift(args["period_lookback"], axis=0).fillna(False)
        for i, iFactorName in enumerate(iid):
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iFactorData = pd.DataFrame(x[iIdx], index=idt, columns=SectionIDs)
            iFactorData = iFactorData.reindex(index=DTs).shift(args["period_lookback"], axis=0)
            iMask = (Mask & iFactorData.notnull())
            Breadth[iFactorName] = iMask.sum(axis=1)
        return Breadth.reindex(index=idt).values[self._QSArgs.LookBack[0]:]    
    
    def __call__(self, *x, mask: Optional[Factor]=None, factor_args:Dict={}, **kwargs):
        Factors = []
        if mask is not None: Factors.append(mask)
        if not x: raise __QS_Error__("测试因子列表 x 不可为空!")
        else: Factors += x
        Args = {
            "Arity": len(Factors),
            "DescriptorSection": [self._QSArgs.DescriptorSection[0]] * len(Factors)
        }
        factor_args = {"SectionIDs": [f"x-{i}" for i in range(len(x))]} | factor_args
        f = super(Breadth, self.new(args=Args)).__call__(*Factors, factor_args=factor_args, **kwargs)
        f.UserData = {"mask": (mask is not None)}
        return f    

class CalcIC(PanelOperator):
    def __init__(self, lookback:int = 31, period_lookback:int=1, corr_method:Literal["spearman", "pearson", "kendall"]="spearman", descriptor_ids=None, args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcIC"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DataType": "object"}
        Args["ModelArgs"] = {"corr_method": corr_method, "period_lookback": period_lookback} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Price, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Price = Price.reindex(columns=DTs)
        else:
            DTs = Price.columns
        Return = Price.T.pct_change().T
        if f.UserData["mask"]: 
            Mask, x = pd.DataFrame(x[0].T==1, columns=idt, index=SectionIDs), x[1:]
            Mask = (Mask.reindex(columns=DTs).fillna(False) & Price.notnull())
        else:
            Mask = Price.notnull()
        if f.UserData["cat_data"]: 
            CatData, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
            CatData = CatData.reindex(columns=DTs)
        else:
            CatData = None
        if f.UserData["weight"]: 
            Weight, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
            Weight = Weight.reindex(columns=DTs)
        else:
            Weight = pd.DataFrame(1, columns=DTs, index=SectionIDs)
        if CatData is not None:# 进行收益率的类别调整
            Price = Price.where(CatData.notnull(), np.nan)
            AllCates = np.unique(CatData)
            for iCate in AllCates:
                iMask = ((CatData==iCate) & Mask)
                iWeight = Weight.where(iMask, np.nan)
                iReturn = (Return * iWeight.shift(1, axis=1)).sum(axis=0) / iWeight.shift(1, axis=1).sum(axis=0)
                Return = Return.where(~iMask.shift(1, axis=1).fillna(False), Return - iReturn)
        IC, Breadth = pd.DataFrame(index=DTs, columns=iid), pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        Mask = Mask.shift(args["period_lookback"], axis=1).fillna(False)
        for i, iFactorName in enumerate(iid):
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iFactorData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs)
            iFactorData = iFactorData.reindex(columns=DTs).shift(args["period_lookback"], axis=1)
            iMask = (Mask & iFactorData.notnull())
            IC[iFactorName] = Return.where(iMask, np.nan).corrwith(iFactorData, method=args["corr_method"])
            Breadth[iFactorName] = iMask.sum(axis=0)
        Rslt = np.array([IC.reindex(index=idt).values[self._QSArgs.LookBack[0]:], Breadth.reindex(index=idt).values[self._QSArgs.LookBack[0]:]])
        return unstructured_to_structured(Rslt.swapaxes(0, -1), dtype=np.dtype([("IC", float), ("Breadth", float)])).T.astype("O")
        
    def __call__(self, *x, price:Factor, mask: Optional[Factor]=None, cat_data: Optional[Factor]=None, weight: Optional[Factor]=None, factor_args:Dict={}, **kwargs):
        Factors = [price]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        if weight is not None: Factors.append(weight)
        if not x: raise __QS_Error__("测试因子列表 x 不可为空!")
        else: Factors += x
        if "SectionIDs" not in factor_args:
            PosNum = int(np.log10(len(x))) + 1
            SectionIDs = [f"x-{str(i).zfill(PosNum)}" for i in range(len(x))]
            factor_args["SectionIDs"] = SectionIDs
        elif len(set(factor_args["SectionIDs"]))!=len(x) or (sorted(factor_args["SectionIDs"])!=factor_args["SectionIDs"]):
            raise __QS_Error__(f"截面ID : {factor_args['SectionIDs']} 长度不等于测试因子列表 x 的长度, 或者有重复, 或者非升序排列!")
        f = super().__call__(*Factors, factor_args=factor_args, **kwargs)
        f.UserData = {"mask": (mask is not None), "cat_data": (cat_data is not None), "weight": (weight is not None), "factor_name_list": [f.Name for f in x]}
        return f

class ICOutput(BTOutputNode):
    """ICOutput"""
    class __QS_ArgClass__(BTOutputNode.__QS_ArgClass__):
        Name: str = Field(default="ICOutput", frozen=True, title="名称")
        RollingAvgPeriod: int = Field(default=12, frozen=True, title="移动平均期数")
        
    def __init__(self, ic: Factor, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=[ic], args=args, config_file=config_file, **kwargs)
    
    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=self.Deps[i].getID()) for i, iInitData in enumerate(InitData)]
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> dict:
        IC, Breadth = bwd_data_list[0].map(lambda x: x[0]), bwd_data_list[0].map(lambda x: x[1])
        SectionIDs = self.Deps[0].getID()
        if not SectionIDs:
            FactorNameList = self.Deps[0].UserData.get("factor_name_list", SectionIDs)
            FactorNameList = [FactorNameList[SectionIDs.index(iID)] for iID in IC.columns]
            IC.columns = Breadth.columns = FactorNameList
            IC = IC.dropna(how="all", axis=0)
            Breadth = Breadth.reindex(index=IC.index)
        Output = {"截面宽度": Breadth, "IC": IC}
        Output["IC的移动平均"] = Output["IC"].copy()
        for i in range(Output["IC"].shape[0]):
            if i<self._QSArgs.RollingAvgPeriod-1: Output["IC的移动平均"].iloc[i, :] = np.nan
            else: Output["IC的移动平均"].iloc[i, :] = Output["IC"].iloc[i-self._QSArgs.RollingAvgPeriod+1:i+1, :].mean()
        Output["统计数据"] = pd.DataFrame(index=Output["IC"].columns)
        Output["统计数据"]["平均值"] = Output["IC"].mean()
        Output["统计数据"]["标准差"] = Output["IC"].std()
        Output["统计数据"]["最小值"] = Output["IC"].min()
        Output["统计数据"]["最大值"] = Output["IC"].max()
        Output["统计数据"]["IC_IR"] = Output["统计数据"]["平均值"] / Output["统计数据"]["标准差"]
        Output["统计数据"]["t统计量"] = np.nan
        Output["统计数据"]["平均截面宽度"] = Output["截面宽度"].mean()
        Output["统计数据"]["IC×Sqrt(N)"] = Output["统计数据"]["平均值"]*np.sqrt(Output["统计数据"]["平均截面宽度"])
        Output["统计数据"]["有效期数"] = 0.0
        for iFactor in Output["IC"]: Output["统计数据"].loc[iFactor, "有效期数"] = pd.notnull(Output["IC"][iFactor]).sum()
        Output["统计数据"]["t统计量"] = Output["统计数据"]["有效期数"]**0.5 * Output["统计数据"]["IC_IR"]
        return Output
    
class ICReport(BTReportNode):
    """ICReport"""
    class __QS_ArgClass__(BTReportNode.__QS_ArgClass__):
        Name: str = Field(default="ICReport", frozen=True, title="名称")    
    
    def genMatplotlibFig(self, output, file_path=None):
        nRow, nCol = output["IC"].shape[1]//3+(output["IC"].shape[1]%3!=0), min(3, output["IC"].shape[1])
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(0, output["IC"].shape[0])
        xTicks = np.arange(0, output["IC"].shape[0], max(1, int(output["IC"].shape[0]/10)))
        xTickLabels = [output["IC"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        for i in range(output["IC"].shape[1]):
            iAxes = Fig.add_subplot(nRow, nCol, i+1)
            iAxes.yaxis.set_major_formatter(yMajorFormatter)
            iAxes.plot(xData, output["IC的移动平均"].iloc[:, i].values, label="IC的移动平均", color="indianred", lw=2.5)
            iAxes.bar(xData, output["IC"].iloc[:, i].values, label="IC", color="steelblue")
            iRightAxes = iAxes.twinx()
            iRightAxes.plot(xData, output["截面宽度"].iloc[:, i].values, label="截面宽度", color="k", lw=1.5)
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iRightAxes.legend(loc="upper right")
            iAxes.set_title(output["IC"].columns[i])
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    
    def backward_compute(self, path: List[str], bwd_data_list: List[dict], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> str:
        HTML = ""
        Output = bwd_data_list[0]
        Formatters = [_QS_formatPandasPercentage]*4+[lambda x:'{0:.4f}'.format(x)]+[lambda x:'{0:.2f}'.format(x)]*3+[lambda x:'{0:.0f}'.format(x)]
        iHTML = Output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = self.genMatplotlibFig(Output)
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML


#class RiskAdjustedIC(IC):    
    #"""风险调整的 IC"""
    #class __QS_ArgClass__(IC.__QS_ArgClass__):
        ## RiskFactors = ListStr(arg_type="MultiOption", label="风险因子", order=2.5, option_range=())
        #def __QS_initArgs__(self, args={}):
            #super().__QS_initArgs__(args=args)
            #self.remove_trait("WeightFactor")
            #DefaultNumFactorList, _ = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            #self.add_trait("RiskFactors", ListStr(arg_type="MultiOption", label="风险因子", order=2.5, option_range=tuple(DefaultNumFactorList)))
    
    #def __init__(self, factor_table, name="风险调整的 IC", sys_args={}, **kwargs):
        #return super().__init__(factor_table=factor_table, name=name, sys_args=sys_args, **kwargs)
    
    #def __QS_move__(self, idt, **kwargs):
        #if self._iDT==idt: return 0
        #self._iDT = idt
        #if self._QSArgs.CalcDTs:
            #if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            #self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            #PreInd = self._CurCalcInd - self._QSArgs.LookBack
            #LastInd = self._CurCalcInd - 1
            #PreDateTime = self._QSArgs.CalcDTs[PreInd]
            #LastDateTime = self._QSArgs.CalcDTs[LastInd]
        #else:
            #self._CurCalcInd = self._Model.DateTimeIndex
            #PreInd = self._CurCalcInd - self._QSArgs.LookBack
            #LastInd = self._CurCalcInd - 1
            #PreDateTime = self._Model.DateTimeSeries[PreInd]
            #LastDateTime = self._Model.DateTimeSeries[LastInd]
        #if (PreInd<0) or (LastInd<0):
            #for iFactorName in self._QSArgs.TestFactors:
                #self._Output["IC"][iFactorName].append(np.nan)
                #self._Output["截面宽度"][iFactorName].append(np.nan)
            #self._Output["时点"].append(idt)
            #return 0
        #PreIDs = self._FactorTable.getFilteredID(idt=PreDateTime, id_filter_str=self._QSArgs.IDFilter)
        #FactorExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self._QSArgs.TestFactors)).iloc[:,0,:]
        #if self._QSArgs.RiskFactors:
            #RiskExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self._QSArgs.RiskFactors)).iloc[:,0,:]
            #RiskExpose["constant"] = 1.0
        #else:
            #RiskExpose = pd.DataFrame(1.0, index=PreIDs, columns=["constant"])
        #CurPrice = self._FactorTable.readData(dts=[idt], ids=PreIDs, factor_names=[self._QSArgs.PriceFactor]).iloc[0,0,:]
        #LastPrice = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self._QSArgs.PriceFactor]).iloc[0,0,:]
        #Ret = CurPrice/LastPrice-1
        #Mask = (pd.isnull(RiskExpose).sum(axis=1)==0)
        ## 展开Dummy因子
        #if self._QSArgs.ClassFactor!="无":
            #DummyFactorData = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=[self._QSArgs.ClassFactor]).iloc[0,0,:]
            #_,_,_,DummyFactorData = prepareRegressData(np.ones(DummyFactorData.shape[0]), dummy_data=DummyFactorData.values)
        #iMask = (pd.notnull(Ret) & Mask)
        #Ret = Ret[iMask]
        #iX = RiskExpose.loc[iMask].values
        #if self._QSArgs.ClassFactor!="无":
            #iDummy = DummyFactorData[iMask.values]
            #iDummy = iDummy[:,(np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
            #iX = np.hstack((iX,iDummy[:,:-1]))
        #try:
            #Result = sm.OLS(Ret.values, iX, missing="drop").fit()
        #except:
            #return self._moveNone(idt)
        #RiskAdjustedRet = pd.Series(Result.resid, index=Ret.index)
        #for iFactorName in self._QSArgs.TestFactors:
            #iFactorExpose = FactorExpose[iFactorName]
            #iMask = (Mask & pd.notnull(iFactorExpose))
            #iFactorExpose = iFactorExpose[iMask]
            #iX = RiskExpose.loc[iMask].values
            #if self._QSArgs.ClassFactor!="无":
                #iDummy = DummyFactorData[iMask.values]
                #iDummy = iDummy[:,(np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
                #iX = np.hstack((iX,iDummy[:,:-1]))
            #try:
                #Result = sm.OLS(iFactorExpose.values,iX,missing="drop").fit()
            #except:
                #self._Output["IC"][iFactorName].append(np.nan)
                #self._Output["截面宽度"][iFactorName].append(0)
                #continue
            #iFactorExpose = pd.Series(Result.resid,index=iFactorExpose.index)
            #self._Output["IC"][iFactorName].append(iFactorExpose.corr(RiskAdjustedRet, method=self._QSArgs.CorrMethod))
            #self._Output["截面宽度"][iFactorName].append(pd.notnull(iFactorExpose).sum())
        #self._Output["时点"].append(idt)
        #return 0

#class ICDecay(PanelOperator):
    #"""IC 衰减"""
    #class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        ##TestFactor = Enum(None, arg_type="SingleOption", label="测试因子", order=0)
        #FactorOrder = Enum("降序","升序", arg_type="SingleOption", label="排序方向", order=1, option_range=["降序","升序"])
        ##PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
        ##ClassFactor = Enum("无", arg_type="SingleOption", label="类别因子", order=3)
        ##WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=4)
        #CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="计算时点", order=5)
        #LookBack = ListInt(np.arange(1,13).tolist(), arg_type="NultiOpotion", label="回溯期数", order=6)
        #CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=7, option_range=["spearman", "pearson", "kendall"])
        #IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=8)
        #def __QS_initArgs__(self, args={}):
            #DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            #self.add_trait("TestFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="测试因子", order=0, option_range=DefaultNumFactorList))
            #self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2, option_range=DefaultNumFactorList))
            #self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            #self.add_trait("ClassFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="类别因子", order=3, option_range=["无"]+DefaultStrFactorList))
            #self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=4, option_range=["等权"]+DefaultNumFactorList))
    
    #def __init__(self, factor_table, name="IC 衰减", sys_args={}, **kwargs):
        #self._FactorTable = factor_table
        #super().__init__(name=name, sys_args=sys_args, **kwargs)
    #def __QS_start__(self, mdl, dts, **kwargs):
        #if self._isStarted: return ()
        #super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        #self._Output = {"IC":[[] for i in self._QSArgs.LookBack]}
        #self._Output["时点"] = []
        #self._CurCalcInd = 0
        #return (self._FactorTable, )
    #def __QS_move__(self, idt, **kwargs):
        #if self._iDT==idt: return 0
        #self._iDT = idt
        #if self._QSArgs.CalcDTs:
            #if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            #self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            #LastInd = self._CurCalcInd - 1
            #LastDateTime = self._QSArgs.CalcDTs[LastInd]
        #else:
            #self._CurCalcInd = self._Model.DateTimeIndex
            #LastInd = self._CurCalcInd - 1
            #LastDateTime = self._Model.DateTimeSeries[LastInd]
        #if (LastInd<0):
            #for i, iRollBack in enumerate(self._QSArgs.LookBack):
                #self._Output["IC"][i].append(np.nan)
            #self._Output["时点"].append(idt)
            #return 0
        #Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=self._FactorTable.getID(ifactor_name=self._QSArgs.PriceFactor), factor_names=[self._QSArgs.PriceFactor]).iloc[0]
        #Ret = Price.iloc[1] / Price.iloc[0] - 1
        #for i, iRollBack in enumerate(self._QSArgs.LookBack):
            #iPreInd = self._CurCalcInd - iRollBack
            #if iPreInd<0:
                #self._Output["IC"][i].append(np.nan)
                #continue
            #iPreDT = self._QSArgs.CalcDTs[iPreInd]
            #iPreIDs = self._FactorTable.getFilteredID(idt=iPreDT, id_filter_str=self._QSArgs.IDFilter)
            #iRet = Ret.reindex(index=iPreIDs, copy=True)
            #if self._QSArgs.ClassFactor!="无":
                #IndustryData = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self._QSArgs.ClassFactor]).iloc[0,0,:]
                #AllIndustry = IndustryData.unique()
                ## 进行收益率的类别调整
                #if self._QSArgs.WeightFactor=="等权":
                    #for iIndustry in AllIndustry:
                        #iRet[IndustryData==iIndustry] -= iRet[IndustryData==iIndustry].mean()
                #else:
                    #WeightData = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self._QSArgs.WeightFactor]).iloc[0,0,:]
                    #for iIndustry in AllIndustry:
                        #iWeight = WeightData[IndustryData==iIndustry]
                        #iiRet = iRet[IndustryData==iIndustry]
                        #iRet[IndustryData==iIndustry] -= (iiRet * iWeight).sum() / iWeight[pd.notnull(iWeight) & pd.notnull(iiRet)].sum(skipna=False)
            #iFactorExpose = self._FactorTable.readData(dts=[iPreDT], ids=iPreIDs, factor_names=[self._QSArgs.TestFactor]).iloc[0,0,:]
            #self._Output["IC"][i].append(iFactorExpose.corr(iRet, method=self._QSArgs.CorrMethod))
        #self._Output["时点"].append(idt)
        #return 0
    #def __QS_end__(self):
        #if not self._isStarted: return 0
        #super().__QS_end__()
        #self._Output["IC"] = pd.DataFrame(np.array(self._Output["IC"]).T, index=self._Output.pop("时点"), columns=list(self._QSArgs.LookBack))
        #if self._QSArgs.FactorOrder=="升序": self._Output["IC"] = -self._Output["IC"]
        #self._Output["统计数据"] = pd.DataFrame(index=self._Output["IC"].columns)
        #self._Output["统计数据"]["IC平均值"] = self._Output["IC"].mean()
        #nDT = pd.notnull(self._Output["IC"]).sum()
        #self._Output["统计数据"]["标准差"] = self._Output["IC"].std()
        #self._Output["统计数据"]["IC_IR"] = self._Output["统计数据"]["IC平均值"] / self._Output["统计数据"]["标准差"]
        #self._Output["统计数据"]["t统计量"] = self._Output["统计数据"]["IC_IR"] * nDT**0.5
        #self._Output["统计数据"]["胜率"] = (self._Output["IC"]>0).sum() / nDT
        #return 0
    #def genMatplotlibFig(self, file_path=None):
        #Fig = Figure(figsize=(16, 8))
        #xData = np.arange(0, self._Output["统计数据"].shape[0])
        #xTickLabels = [str(i) for i in self._Output["统计数据"].index]
        #yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        #Axes = Fig.add_subplot(1, 1, 1)
        #Axes.yaxis.set_major_formatter(yMajorFormatter)
        #Axes.bar(xData, self._Output["统计数据"]["IC平均值"].values, label="IC", color="steelblue")
        #Axes.set_xticks(xData)
        #Axes.set_xticklabels(xTickLabels)
        #Axes.legend(loc='upper left')
        #RAxes = Axes.twinx()
        #RAxes.yaxis.set_major_formatter(yMajorFormatter)
        #RAxes.plot(xData, self._Output["统计数据"]["胜率"].values, label="胜率", color="indianred", lw=2.5)
        #RAxes.legend(loc="upper right")
        #plt.setp(Axes.get_xticklabels(), visible=True, rotation=0, ha='center')
        #if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        #return Fig
    #def _repr_html_(self):
        #if len(self._QSArgs.ArgNames)>0:
            #HTML = "参数设置: "
            #HTML += '<ul align="left">'
            #for iArgName in self._QSArgs.ArgNames:
                #if iArgName!="计算时点":
                    #HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                #elif self.Args[iArgName]:
                    #HTML += "<li>"+iArgName+": 自定义时点</li>"
                #else:
                    #HTML += "<li>"+iArgName+": 所有时点</li>"
            #HTML += "</ul>"
        #else:
            #HTML = ""
        #Formatters = [_QS_formatPandasPercentage]*2+[lambda x:'{0:.4f}'.format(x), lambda x:'{0:.2f}'.format(x), _QS_formatPandasPercentage]
        #iHTML = self._Output["统计数据"].to_html(formatters=Formatters)
        #Pos = iHTML.find(">")
        #HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        #Fig = self.genMatplotlibFig()
        ## figure 保存为二进制文件
        #Buffer = BytesIO()
        #Fig.savefig(Buffer, bbox_inches='tight')
        #PlotData = Buffer.getvalue()
        ## 图像数据转化为 HTML 格式
        #ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        #HTML += ('<img src="%s">' % ImgStr)
        #return HTML