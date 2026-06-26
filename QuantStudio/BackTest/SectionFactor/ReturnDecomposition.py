# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, List, Any, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import Field
from numpy.lib.recfunctions import unstructured_to_structured

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import DTInitData, DTLocalContext
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData, FactorLocalContext
from QuantStudio.Factor.FactorOperation import PanelOperator, PanelOperation
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var


class CalcFamaMacBethRegression(PanelOperator):
    """Fama-MacBeth 回归算子"""

    def __init__(self, descriptor_ids:List[str], lookback:int = 31, period_lookback:int=1, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 Fama-MacBeth 回归算子

        Args:
            descriptor_ids: 依赖因子的截面 ID 序列
            lookback: 在时间标尺上的回溯期数, 即回溯多久的数据来完成计算
            period_lookback: 在计算标尺上的回溯期数, 数据的时间序列是日度的，但回归的计算时间序列是月度的，该参数表示用回溯多少个月的因子值来和当前收益率来回归
        """
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcFamaMacBethRegression"} | args | {"DTMode": "单时点", "DataType": "object"}
        Args["ModelArgs"] = {"period_lookback": period_lookback} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        Args["CompoundType"] = [
            ("PureReturn", "double"), ("PureT", "double"), ("PureF", "double"), ("PureR", "double"), ("PureRAdj", "double"),
            ("RawReturn", "double"), ("RawT", "double"), ("RawF", "double"), ("RawR", "double"), ("RawRAdj", "double")
        ]
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Price, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Price = Price.reindex(index=DTs)
        else:
            DTs = Price.index
        Return = Price.pct_change().iloc[-1]
        if f._QSArgs.ModelArgs["mask"]: 
            Mask, x = pd.DataFrame(x[0]==1, index=idt, columns=SectionIDs), x[1:]
            Mask = (Mask.reindex(index=DTs).astype(float).fillna(0).astype(bool) & Price.notnull())
        else:
            Mask = Price.notnull()
        Mask = Mask.shift(args["period_lookback"], axis=1).iloc[-1].astype(float).fillna(0).astype(bool)
        if f._QSArgs.ModelArgs["cat_data"]:
            CatData, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
            CatData = CatData.reindex(index=DTs)
            CatData = CatData.shift(args["period_lookback"], axis=1).iloc[-1]
            CatMask = pd.notnull(CatData)
            DummyFactorData = DummyVarTo01Var(CatData[CatMask], ignore_na=True)
        FactorData = {i: pd.DataFrame(ix, index=idt, columns=SectionIDs).reindex(index=DTs).shift(args["period_lookback"], axis=1).iloc[-1] for i, ix in enumerate(x)}
        FactorData = pd.DataFrame(FactorData).sort_index(axis=1)
        nFactor = FactorData.shape[1]
        if f._QSArgs.ModelArgs["cat_data"]:
            FactorData = pd.merge(FactorData, DummyFactorData, left_index=True, right_index=True)
        # 回归
        yData = Return[FactorData.index].values
        xData = FactorData.values
        if f._QSArgs.ModelArgs["cat_data"]:
            xData = sm.add_constant(xData, prepend=False)
            LastInds = [nFactor]
        else:
            LastInds = [nFactor + i for i in range(xData.shape[1] - nFactor)]
        Rslt = np.full(shape=(10, nFactor), fill_value=np.nan, dtype=float)
        try:
            Result = sm.OLS(yData, xData, missing="drop").fit()
            Rslt[0] = Result.params[:nFactor]
            Rslt[1] = Result.tvalues[:nFactor]
            Rslt[2] = Result.fvalue
            Rslt[3] = Result.rsquared
            Rslt[4] = Result.rsquared_adj
        except:
            pass
        for i in range(nFactor):
            iXData = xData[:, [i] + LastInds]
            try:
                Result = sm.OLS(yData, iXData, missing="drop").fit()
                Rslt[5, i] = Result.params[0]
                Rslt[6, i] = Result.tvalues[0]
                Rslt[7, i] = Result.fvalue
                Rslt[8, i] = Result.rsquared
                Rslt[9, i] = Result.rsquared_adj
            except:
                pass
        return unstructured_to_structured(Rslt.T).tolist()
        
    def __call__(self, *x:Factor, price:Factor, mask: Optional[Factor]=None, cat_data: Optional[Factor]=None, factor_name_list:Optional[List[str]]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        """将算子作用在若干个因子对象上以产生 Fama-MacBeth 回归因子

        Args:
            x: 待回归的测试因子
            price: 证券价格或者净值因子
            mask: 筛选条件因子, 每一期的 x 因子值会按照该因子是否等于 1 来筛选后再回归, None 表示不做任何筛选
            cat_data: 类别因子, 比如行业等，如果非 None 表示该因子作为哑变量参与回归
            factor_name_list: 测试因子名称列表
            factor_args: 创建回归因子时传递个它的参数集
            kwargs: 创建回归因子时传递给它的其他入参
        
        Returns:
            Fama-MacBeth 回归因子
        """
        factor_args = factor_args.copy()
        Factors = [price]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        if not x: raise __QS_Error__("测试因子列表 x 不可为空!")
        else: Factors += x
        if factor_name_list is not None:
            if factor_args.get("SectionIDs", None) is not None:
                self.Logger.warning(f"{self.__class__.__name__}.__call__: 同时指定了测试因子名称列表 factor_name_list({factor_name_list})以及因子截面ID参数 SectionIDs({factor_args['SectionIDs']}), 将使用后者作为因子的截面ID, 忽略 factor_name_list")
                factor_name_list = factor_args["SectionIDs"]
        elif factor_args.get("SectionIDs", None) is not None:
            factor_name_list = factor_args["SectionIDs"]
        else:
            factor_name_list = [iFactor.Name for iFactor in x]
            if len(set(factor_name_list)) != len(x):
                PosNum = int(np.log10(max(1, len(x) - 1))) + 1
                factor_name_list = [f"F{str(i).zfill(PosNum)}" for i in range(len(x))]
                self.Logger.info(f"测试因子的名称中有重复, 使用系统自动生成的测试因子名称列表: {factor_name_list}")
        if len(set(factor_name_list)) != len(x):
            raise __QS_Error__(f"测试因子的名称列表 : {factor_name_list} 长度不等于测试因子列表 x 的长度或者有重复!")
        else:
            SortedIdx = np.argsort(factor_name_list)
            if not np.all(SortedIdx == np.arange(len(factor_name_list))):
                self.Logger.warning(f"{self.__class__.__name__}.__call__: 测试因子的名称列表({factor_name_list})不是升序排列，将按照升序重新排列测试因子")
                x, factor_name_list = [x[i] for i in SortedIdx], [factor_name_list[i] for i in SortedIdx]
        factor_args["SectionIDs"] = factor_name_list
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None)}
        kwargs["operator_kwargs"] =  {"descriptor_ids": self._QSArgs.DescriptorSection[0], "lookback": self._QSArgs.LookBack[0]} | kwargs.get("operator_kwargs", {})
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class FamaMacBethRegression(BTNode):
    """Fama-MacBeth 回归"""
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="Fama-MacBeth 回归", frozen=True, title="名称")
        FactorNameList: Optional[List[str]] = Field(default=None, frozen=True, title="因子列表")
        RollingAvgPeriod: int = Field(default=12, frozen=True, title="移动平均期数")
        
    def __init__(self, fmr: Factor, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=[fmr], args=args, config_file=config_file, **kwargs)
    
    @staticmethod
    def genMatplotlibFig(output:dict, file_path:Optional[str]=None) -> Figure:
        nRow, nCol = 1, 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        xData = np.arange(0, output["统计数据"].shape[0])
        xTickLabels = [str(iInd) for iInd in output["统计数据"].index]
        iAxes = Fig.add_subplot(nRow, nCol, 1)
        iAxes.yaxis.set_major_formatter(PercentageFormatter)
        iAxes.bar(xData, output["统计数据"]["年化收益率(Raw)"].values, width=-0.25, align="edge", color="indianred", label="年化收益率(Raw)")
        iAxes.bar(xData, output["统计数据"]["年化收益率(Pure)"].values, width=0.25, align="edge", color="steelblue", label="年化收益率(Pure)")
        iAxes.set_xticks(xData)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc='best')
        iAxes.set_title("年化收益率")
        iAxes = Fig.add_subplot(nRow, nCol, 2)
        iAxes.yaxis.set_major_formatter(FloatFormatter)
        iAxes.bar(xData, output["统计数据"]["t统计量(Raw)"].values, width=-0.25, align="edge", color="indianred", label="t统计量(Raw)")
        iAxes.bar(xData, output["统计数据"]["t统计量(Pure)"].values, width=0.25, align="edge", color="steelblue", label="t统计量(Pure)")
        iAxes.set_xticks(xData)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc='best')
        iAxes.set_title("t统计量")
        iAxes = Fig.add_subplot(nRow, nCol, 3)
        iAxes.yaxis.set_major_formatter(PercentageFormatter)
        iAxes.bar(xData, output["统计数据"]["年化收益率(Pure-Raw)"].values, color="steelblue", label="年化收益率(Pure-Raw)")
        iAxes.set_xticks(xData)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc='upper left')
        iAxes.set_title("Pure-Raw")
        RAxes = iAxes.twinx()
        RAxes.yaxis.set_major_formatter(FloatFormatter)
        RAxes.plot(xData, output["统计数据"]["t统计量(Pure-Raw)"].values, color="indianred", lw=2.5, label="t统计量(Pure-Raw)")
        RAxes.legend(loc='upper right')
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig

    @staticmethod
    def genOutputReport(output:dict) -> str:
        HTML = ""
        FloatFormatFun = lambda x:'{0:.2f}'.format(x)
        Formatters = [_QS_formatPandasPercentage] * 2 + [FloatFormatFun, _QS_formatPandasPercentage, FloatFormatFun]
        Formatters += [_QS_formatPandasPercentage] * 2 + [FloatFormatFun, _QS_formatPandasPercentage, FloatFormatFun]
        Formatters += [_QS_formatPandasPercentage] * 2 + [FloatFormatFun, _QS_formatPandasPercentage, FloatFormatFun]
        iHTML = output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        HTML += '<div align="left" style="font-size:1em"><strong>回归统计量</strong></div>'
        iHTML = output["回归统计量均值"].to_html(formatters=[FloatFormatFun] * 8)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = FamaMacBethRegression.genMatplotlibFig(output=output)
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML

    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if isinstance(getattr(self.Deps[0], "Operator", None), CalcFamaMacBethRegression):
            ModelArgs = self.Deps[0].Operator._QSArgs.ModelArgs
            HTML += f"<li>回溯期数: {ModelArgs['period_lookback']}</li>"
        if self.Deps[0]._QSArgs.CalcDTRuler:
            HTML += "<li>计算时点: 自定义时点</li>"
        else:
            HTML += "<li>计算时点: 所有时点</li>"
        HTML += f"<li>移动平均期数: {self._QSArgs.RollingAvgPeriod}</li>"
        HTML += "</ul>"
        HTML += "\n" + FamaMacBethRegression.genOutputReport(output=output)
        return HTML

    def init_compute(self, path: List[str], init_data: DTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=None) for i, iInitData in enumerate(InitData)]
    
    def forward_compute(self, path: List[str], fwd_data: DTLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], DTLocalContext]:
        return [FactorLocalContext(DTs=fwd_data.DTs, IDs=context.NodeState[iDep.QSID]["section_ids"], PIDs=context.PIDList) for iDep in self.Deps], DTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[DTLocalContext]=None) -> dict:
        BwdData = bwd_data_list[0].dropna(how="all", axis=0)
        PureReturn, RawReturn = BwdData.map(lambda x: x[0] if pd.notnull(x) else np.nan), BwdData.map(lambda x: x[5] if pd.notnull(x) else np.nan)
        if self._QSArgs.FactorNameList:
            FactorNameList = self._QSArgs.FactorNameList
        else:
            FactorNameList = PureReturn.columns.tolist()
        PureReturn.columns = RawReturn.columns = FactorNameList
        PureReturn = PureReturn.dropna(how="all", axis=0)
        RawReturn = RawReturn.reindex(index=PureReturn.index)
        ColMapping = {Col: FactorNameList[i] for i, Col in enumerate(bwd_data_list[0].columns)}
        Output = {"Pure Return": PureReturn, "Raw Return": RawReturn}
        Output["回归t统计量(Pure)"] = BwdData.map(lambda x: x[1] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        Output["回归t统计量(Raw)"] = BwdData.map(lambda x: x[6] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        Output["回归F统计量(Pure)"] = BwdData.map(lambda x: x[2] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        Output["回归F统计量(Raw)"] = BwdData.map(lambda x: x[7] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        Output["回归R平方(Pure)"] = bwd_data_list[0].map(lambda x: x[3] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        Output["回归R平方(Raw)"] = bwd_data_list[0].map(lambda x: x[8] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        Output["回归调整R平方(Pure)"] = bwd_data_list[0].map(lambda x: x[4] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        Output["回归调整R平方(Raw)"] = bwd_data_list[0].map(lambda x: x[9] if pd.notnull(x) else np.nan).reindex(index=PureReturn.index).rename(columns=ColMapping)
        # 计算滚动t统计量
        nDT = PureReturn.shape[0]
        Output["滚动t统计量(Pure)"] = pd.DataFrame(np.nan, index=PureReturn.index, columns=FactorNameList)
        Output["滚动t统计量(Raw)"] = pd.DataFrame(np.nan, index=PureReturn.index, columns=FactorNameList)
        for i in range(nDT):
            if i < self._QSArgs.RollingAvgPeriod-1: continue
            iReturn = PureReturn.iloc[i-self._QSArgs.RollingAvgPeriod+1:i+1, :]
            Output["滚动t统计量(Pure)"].iloc[i] = iReturn.mean(axis=0) / iReturn.std(axis=0) * pd.notnull(iReturn).sum(axis=0)**0.5
            iReturn = RawReturn.iloc[i-self._QSArgs.RollingAvgPeriod+1:i+1, :]
            Output["滚动t统计量(Raw)"].iloc[i] = iReturn.mean(axis=0) / iReturn.std(axis=0) * pd.notnull(iReturn).sum(axis=0)**0.5
        nYear = (PureReturn.index[-1] - PureReturn.index[0]).days / 365
        Output["统计数据"] = pd.DataFrame(index=PureReturn.columns)
        Output["统计数据"]["年化收益率(Pure)"] = ((1 + PureReturn).prod())**(1/nYear) - 1
        Output["统计数据"]["跟踪误差(Pure)"] = PureReturn.std() * np.sqrt(nDT/nYear)
        Output["统计数据"]["信息比率(Pure)"] = Output["统计数据"]["年化收益率(Pure)"] / Output["统计数据"]["跟踪误差(Pure)"]
        Output["统计数据"]["胜率(Pure)"] = (PureReturn > 0).sum() / nDT
        Output["统计数据"]["t统计量(Pure)"] = PureReturn.mean() / PureReturn.std() * np.sqrt(nDT)
        Output["统计数据"]["年化收益率(Raw)"] = (1 + RawReturn).prod()**(1/nYear) - 1
        Output["统计数据"]["跟踪误差(Raw)"] = RawReturn.std() * np.sqrt(nDT/nYear)
        Output["统计数据"]["信息比率(Raw)"] = Output["统计数据"]["年化收益率(Raw)"] / Output["统计数据"]["跟踪误差(Raw)"]
        Output["统计数据"]["胜率(Raw)"] = (RawReturn > 0).sum() / nDT
        Output["统计数据"]["t统计量(Raw)"] = RawReturn.mean() / RawReturn.std() * np.sqrt(nDT)
        Output["统计数据"]["年化收益率(Pure-Raw)"] = (1 + PureReturn - RawReturn).prod()**(1/nYear) - 1
        Output["统计数据"]["跟踪误差(Pure-Raw)"] = (PureReturn - RawReturn).std() * np.sqrt(nDT/nYear)
        Output["统计数据"]["信息比率(Pure-Raw)"] = Output["统计数据"]["年化收益率(Pure-Raw)"] / Output["统计数据"]["跟踪误差(Pure-Raw)"]
        Output["统计数据"]["胜率(Pure-Raw)"] = (PureReturn - RawReturn > 0).sum() / nDT
        Output["统计数据"]["t统计量(Pure-Raw)"] = (PureReturn - RawReturn).mean() / (PureReturn - RawReturn).std() * np.sqrt(nDT)
        Output["回归统计量均值"] = pd.DataFrame(index=FactorNameList)
        Output["回归统计量均值"]["t统计量(Raw)"] = Output["回归t统计量(Raw)"].mean()
        Output["回归统计量均值"]["t统计量(Pure)"] = Output["回归t统计量(Pure)"].mean()
        Output["回归统计量均值"]["F统计量(Raw)"] = Output["回归F统计量(Raw)"].mean()
        Output["回归统计量均值"]["F统计量(Pure)"] = Output["回归F统计量(Pure)"].mean()
        Output["回归统计量均值"]["R平方(Raw)"] = Output["回归R平方(Raw)"].mean()
        Output["回归统计量均值"]["R平方(Pure)"] = Output["回归R平方(Pure)"].mean()
        Output["回归统计量均值"]["调整R平方(Raw)"] = Output["回归调整R平方(Raw)"].mean()
        Output["回归统计量均值"]["调整R平方(Pure)"] = Output["回归调整R平方(Pure)"].mean()
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
