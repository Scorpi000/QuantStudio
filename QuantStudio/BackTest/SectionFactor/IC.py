# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, Literal, List, Dict, Any

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData
from QuantStudio.Factor.FactorOperation import PanelOperator, SectionOperator, PanelOperation, SectionOperation
from QuantStudio.BackTest.BackTestModel import BTLocalContext, BTNode, BTInitData


def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x*100, )

def _QS_formatPandasPercentage(x):
    return '{0:.2f}%'.format(x*100)

class CalcIC(PanelOperator):
    """IC 算子"""
    def __init__(self, lookback:int = 31, period_lookback:int=1, corr_method:Literal["spearman", "pearson", "kendall"]="spearman", descriptor_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
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
        for iFactorName in iid:
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iFactorData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs)
            iFactorData = iFactorData.reindex(columns=DTs).shift(args["period_lookback"], axis=1)
            iMask = (Mask & iFactorData.notnull())
            IC[iFactorName] = Return.where(iMask, np.nan).corrwith(iFactorData, method=args["corr_method"])
            Breadth[iFactorName] = iMask.sum(axis=0)
        Rslt = np.array([IC.reindex(index=idt).values[self._QSArgs.LookBack[0]:], Breadth.reindex(index=idt).values[self._QSArgs.LookBack[0]:]])
        return unstructured_to_structured(Rslt.swapaxes(0, -1), dtype=np.dtype([("IC", float), ("Breadth", float)])).T.astype("O")
        
    def __call__(self, *x:Factor, price:Factor, mask: Optional[Factor]=None, cat_data: Optional[Factor]=None, weight: Optional[Factor]=None, factor_args:Dict={}, **kwargs) -> PanelOperation:
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

class IC(BTNode):
    """IC"""
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="IC", frozen=True, title="名称")
        FactorNameList: Optional[List[str]] = Field(default=None, frozen=True, title="因子列表")
        RollingAvgPeriod: int = Field(default=12, frozen=True, title="移动平均期数")
        
    def __init__(self, ic: Factor, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=[ic], args=args, config_file=config_file, **kwargs)
    
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
    
    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if isinstance(getattr(self.Deps[0], "Operator", None), CalcIC):
            ModelArgs = self.Deps[0].Operator._QSArgs.ModelArgs
            HTML += f"<li>相关性方法: {ModelArgs['corr_method']}</li>"
            HTML += f"<li>回溯期数: {ModelArgs['period_lookback']}</li>"
        if self.Deps[0]._QSArgs.CalcDTRuler:
            HTML += "<li>计算时点: 自定义时点</li>"
        else:
            HTML += "<li>计算时点: 所有时点</li>"
        HTML += f"<li>移动平均期数: {self._QSArgs.RollingAvgPeriod}</li>"
        HTML += "</ul>"
        Formatters = [_QS_formatPandasPercentage]*4+[lambda x:'{0:.4f}'.format(x)]+[lambda x:'{0:.2f}'.format(x)]*3+[lambda x:'{0:.0f}'.format(x)]
        iHTML = output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = self.genMatplotlibFig(output)
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
        IC, Breadth = bwd_data_list[0].map(lambda x: x[0]), bwd_data_list[0].map(lambda x: x[1])
        if self._QSArgs.FactorNameList:
            FactorNameList = self._QSArgs.FactorNameList
        else:
            FactorNameList = self.Deps[0].UserData.get("factor_name_list", IC.columns)
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
        Output["统计数据"]["IC×Sqrt(N)"] = Output["统计数据"]["平均值"] * np.sqrt(Output["统计数据"]["平均截面宽度"])
        Output["统计数据"]["有效期数"] = 0.0
        for iFactor in Output["IC"]: Output["统计数据"].loc[iFactor, "有效期数"] = pd.notnull(Output["IC"][iFactor]).sum()
        Output["统计数据"]["t统计量"] = Output["统计数据"]["有效期数"]**0.5 * Output["统计数据"]["IC_IR"]
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
