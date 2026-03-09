# coding=utf-8
import base64
from io import BytesIO
import datetime as dt
from itertools import combinations
from typing import Literal, Optional, List, Any

import numpy as np
import pandas as pd
from pydantic import Field
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio.Core import __QS_Error__
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData
from QuantStudio.Factor.FactorOperation import PanelOperator, SectionOperator, PanelOperation, SectionOperation
from QuantStudio.BackTest.BackTestModel import BTLocalContext, BTNode, BTInitData
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage


class CalcSectionCorrelation(SectionOperator):
    """因子截面相关性算子"""
    def __init__(self, corr_method:Literal["spearman", "pearson", "kendall"]="spearman", descriptor_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcSectionCorrelation"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DataType": "double"}
        Args["ModelArgs"] = {"corr_method": corr_method} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        if f._QSArgs.ModelArgs["mask"]: 
            Mask, x = pd.DataFrame(x[0].T==1, columns=idt, index=SectionIDs), x[1:]
        else:
            Mask = pd.DataFrame(True, columns=idt, index=SectionIDs)
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Mask = Mask.reindex(columns=DTs).fillna(False)
        else:
            DTs = Mask.columns
        Corr = pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        for ijFactorName in iid:
            if ijFactorName not in FactorNames: continue
            iIdx, jIdx = f._QSArgs.ModelArgs["section_id_mapping"][ijFactorName].split("-")
            iIdx, jIdx = int(iIdx), int(jIdx)
            if isinstance(x[iIdx], pd.DataFrame): iFactorData = x[iIdx]
            else:
                iFactorData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs).reindex(columns=DTs)
                x[iIdx] = iFactorData
            if isinstance(x[jIdx], pd.DataFrame): jFactorData = x[jIdx]
            else:
                jFactorData = pd.DataFrame(x[jIdx].T, columns=idt, index=SectionIDs).reindex(columns=DTs)
                x[jIdx] = jFactorData
            ijMask = (Mask & iFactorData.notnull() & jFactorData.notnull())
            Corr[ijFactorName] = iFactorData.where(ijMask, np.nan).corrwith(jFactorData.where(ijMask, np.nan), method=args["corr_method"])
        return Corr.reindex(index=idt).values
        
    def __call__(self, *x:Factor, mask:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> SectionOperation:
        if len(x) < 2: raise __QS_Error__(f"算子 {self.__class__}: 必须至少指定两个因子!")
        Factors = []
        if mask is not None: Factors.append(mask)
        Factors += x
        PosNum = int(np.log10(len(x)))
        DefaultSectionIDs = [f"{str(i).zfill(PosNum)}-{str(j).zfill(PosNum)}" for i, j in combinations(range(len(x)), r=2)]
        if "SectionIDs" not in factor_args:
            factor_args["SectionIDs"] = DefaultSectionIDs
        elif len(set(factor_args["SectionIDs"]))!=len(DefaultSectionIDs) or (sorted(factor_args["SectionIDs"])!=factor_args["SectionIDs"]):
            raise __QS_Error__(f"截面ID : {factor_args['SectionIDs']} 长度不等于因子列表 x 两两组合的长度, 或者有重复, 或者非升序排列!")
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "section_id_mapping": dict(zip(factor_args["SectionIDs"], DefaultSectionIDs)), "factor_name_list": [f.Name for f in x]}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class SectionCorrelation(BTNode):
    """因子截面相关性"""
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="SectionCorrelation", frozen=True, title="名称")
        FactorNameList: Optional[List[str]] = Field(default=None, frozen=True, title="因子列表")
        
    def __init__(self, section_corr: Factor, args:dict={}, config_file:Optional[str]=None, **kwargs):
        return super().__init__(deps=[section_corr], args=args, config_file=config_file, **kwargs)
    
    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if isinstance(getattr(self.Deps[0], "Operator", None), CalcSectionCorrelation):
            ModelArgs = self.Deps[0].Operator._QSArgs.ModelArgs
            HTML += f"<li>相关性方法: {ModelArgs['corr_method']}</li>"
        if self.Deps[0]._QSArgs.CalcDTRuler:
            HTML += "<li>计算时点: 自定义时点</li>"
        else:
            HTML += "<li>计算时点: 所有时点</li>"
        HTML += "</ul>"
        iHTML = output["平均值"].style.background_gradient(cmap="Reds").set_properties(precision=2).to_html()
        HTML += '<div align="left" style="font-size:1em"><strong>平均相关性</strong></div>' + iHTML
        return HTML

    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=self.Deps[i].getID()) for i, iInitData in enumerate(InitData)]
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> dict:
        Corr = bwd_data_list[0]
        SectionIDs = self.Deps[0].getID()
        if not SectionIDs: SectionIDs = Corr.columns.tolist()
        nFactor = int(round(((1 + 8*len(SectionIDs)) ** 0.5 + 1) / 2, 0))
        if nFactor * (nFactor - 1) / 2 != len(SectionIDs):
            raise __QS_Error__("截面长度不是 n * (n - 1) / 2!")
        if self._QSArgs.FactorNameList:
            FactorNameList = self._QSArgs.FactorNameList
        else:
            FactorNameList = self.Deps[0].Args.ModelArgs.get("factor_name_list", [str(i) for i in range(nFactor)])
        if len(FactorNameList) != nFactor:
            raise __QS_Error__("因子数量和数据推断出的因子数量不相等!")
        Corr.columns = [f"{FactorNameList[i]}-{FactorNameList[j]}" for i, j in combinations(range(nFactor), r=2)]
        Corr = Corr.dropna(how="all", axis=0)
        Output = {"截面相关性": Corr}
        Avg = Corr.mean()
        Avg.index = [f"{i}-{j}" for i, j in combinations(range(nFactor), r=2)]
        Output["平均值"] = pd.DataFrame(index=FactorNameList, columns=FactorNameList, dtype=float)
        for i, iFactor in enumerate(FactorNameList):
            for j, jFactor in enumerate(FactorNameList):
                if j > i:
                    Output["平均值"].loc[iFactor, jFactor] = Avg.loc[f"{i}-{j}"]
                elif j < i:
                    Output["平均值"].loc[iFactor, jFactor] = Output["平均值"].loc[jFactor, iFactor]
                else:
                    Output["平均值"].loc[iFactor, jFactor] = 1
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output

class CalcFactorTurnover(PanelOperator):
    """因子换手率算子"""
    def __init__(self, lookback:int=31, period_lookback:int=1, corr_method:Literal["spearman", "pearson", "kendall"]="spearman", descriptor_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcIC"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DataType": "double"}
        Args["ModelArgs"] = {"corr_method": corr_method, "period_lookback": period_lookback, "corr_method": corr_method} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        if f._QSArgs.ModelArgs["mask"]: 
            Mask, x = pd.DataFrame(x[0].T==1, columns=idt, index=SectionIDs), x[1:]
        else:
            Mask = pd.DataFrame(True, columns=idt, index=SectionIDs)
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Mask = Mask.reindex(columns=DTs).fillna(False)
        else:
            DTs = Mask.columns
        FactorTurnover = pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        Mask = Mask.shift(args["period_lookback"], axis=1).fillna(False)
        for iFactorName in iid:
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs).reindex(columns=DTs)
            iPreData = iData.shift(args["period_lookback"], axis=1)
            iMask = (Mask & iPreData.notnull())
            FactorTurnover[iFactorName] = iData.where(iMask, np.nan).corrwith(iPreData, method=args["corr_method"])
        return FactorTurnover.reindex(index=idt).values[self._QSArgs.LookBack[0]:]
    
    def __call__(self, *x:Factor, mask: Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        Factors = []
        if mask is not None: Factors.append(mask)
        if not x: raise __QS_Error__("因子列表 x 不可为空!")
        else: Factors += x
        if "SectionIDs" not in factor_args:
            PosNum = int(np.log10(len(x))) + 1
            SectionIDs = [f"x-{str(i).zfill(PosNum)}" for i in range(len(x))]
            factor_args["SectionIDs"] = SectionIDs
        elif len(set(factor_args["SectionIDs"]))!=len(x) or (sorted(factor_args["SectionIDs"])!=factor_args["SectionIDs"]):
            raise __QS_Error__(f"截面ID : {factor_args['SectionIDs']} 长度不等于测试因子列表 x 的长度, 或者有重复, 或者非升序排列!")
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "factor_name_list": [f.Name for f in x]}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class FactorTurnover(BTNode):
    """因子换手率: 当期因子值和往期因子值横截面上的线性相关系数"""
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="SectionCorrelation", frozen=True, title="名称")
        FactorNameList: Optional[List[str]] = Field(default=None, frozen=True, title="因子列表")
        
    def __init__(self, factor_turnover: Factor, args:dict={}, config_file:Optional[str]=None, **kwargs):
        return super().__init__(deps=[factor_turnover], args=args, config_file=config_file, **kwargs)
    
    def genMatplotlibFig(self, output, file_path=None):
        nRow, nCol = output["因子换手率"].shape[1]//3+(output["因子换手率"].shape[1]%3!=0), min(3, output["因子换手率"].shape[1])
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        for i in range(output["因子换手率"].shape[1]):
            iAxes = Fig.add_subplot(nRow, nCol, i+1)
            iAxes.yaxis.set_major_formatter(yMajorFormatter)
            iAxes.xaxis_date()
            iAxes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
            iAxes.stackplot(output["因子换手率"].index, output["因子换手率"].iloc[:, i].values, color="steelblue")
            iAxes.set_title(output["因子换手率"].columns[i])
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig

    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if isinstance(getattr(self.Deps[0], "Operator", None), CalcFactorTurnover):
            ModelArgs = self.Deps[0].Operator._QSArgs.ModelArgs
            HTML += f"<li>相关性方法: {ModelArgs['corr_method']}</li>"
            HTML += f"<li>回溯期数: {ModelArgs['period_lookback']}</li>"
        if self.Deps[0]._QSArgs.CalcDTRuler:
            HTML += "<li>计算时点: 自定义时点</li>"
        else:
            HTML += "<li>计算时点: 所有时点</li>"
        HTML += "</ul>"
        iHTML = output["统计数据"].to_html(formatters=[_QS_formatPandasPercentage]*5)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = self.genMatplotlibFig(output=output)
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML

    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=self.Deps[i].getID()) for i, iInitData in enumerate(InitData)]
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> dict:
        FactorTurnover = bwd_data_list[0]
        if self._QSArgs.FactorNameList:
            FactorNameList = self._QSArgs.FactorNameList
        else:
            FactorNameList = self.Deps[0].Args.ModelArgs.get("factor_name_list", FactorTurnover.columns)
        FactorTurnover.columns = FactorNameList
        FactorTurnover = FactorTurnover.dropna(how="all", axis=0)
        Output = {"因子换手率": FactorTurnover}
        Output["统计数据"] = pd.DataFrame(FactorTurnover.mean(), columns=["平均值"])
        Output["统计数据"]["标准差"] = FactorTurnover.std()
        Output["统计数据"]["最小值"] = FactorTurnover.min()
        Output["统计数据"]["最大值"] = FactorTurnover.max()
        Output["统计数据"]["中位数"] = FactorTurnover.median()
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
