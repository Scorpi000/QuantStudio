# coding=utf-8
"""Brinson 绩效分析模型"""
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, List, Any, Tuple

import pandas as pd
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import DTInitData, DTLocalContext
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData, FactorLocalContext
from QuantStudio.Factor.FactorOperation import PanelOperator, PanelOperation
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatPandasPercentage, _QS_formatMatplotlibPercentage


class CalcBrinsonModel(PanelOperator):
    """Brinson 绩效分析算子
    前提条件:
    * 投资组合的权重之和为 1, 与 1 的差值部分归为现金
    * 两个计算时点之间没有调整策略持仓
    """

    def __init__(self, section_ids:List[str], descriptor_ids:List[str], lookback:int = 31, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 Brinson 绩效分析算子

        Args:
            section_ids: 绩效分析因子的截面 ID 序列, 如果使用行业作为资产分类, 该截面应该为所有行业的列表
            descriptor_ids: 依赖因子的截面 ID 序列
            lookback: 在时间标尺上的回溯期数, 即回溯多久的数据来完成计算
        """
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcBrinsonModel"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DataType": "object"}
        Args["ModelArgs"] = {"section_ids": section_ids} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        Args["CompoundType"] = [("BMK", "double"), ("BMKR", "double"), ("TP", "double"), ("TPR", "double"), ("AA", "double"), ("SS", "double"), ("IN", "double"), ("AAA", "double")]
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Portfolio, Price, CatData = pd.DataFrame(x[0], index=idt, columns=SectionIDs), pd.DataFrame(x[1], index=idt, columns=SectionIDs), pd.DataFrame(x[2], index=idt, columns=SectionIDs)
        if f._QSArgs.ModelArgs["bmk"]: 
            Bmk = pd.DataFrame(x[-1], index=idt, columns=SectionIDs)
        else:
            Bmk = pd.DataFrame(0, index=idt, columns=SectionIDs)
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Portfolio, Price, CatData, Bmk = Portfolio.reindex(index=DTs), Price.reindex(index=DTs), CatData.reindex(index=DTs), Bmk.reindex(index=DTs)
        Return = Price.pct_change().values
        Portfolio, CatData, Bmk = Portfolio.shift(1).values, CatData.shift(1).values, Bmk.shift(1).values
        AllCats = CatData[1:].flatten()
        Mask = pd.notnull(AllCats)
        AllCats = sorted(pd.unique(AllCats[Mask]))
        if not np.all(Mask): AllCats = sorted(AllCats + ["None"])
        BMK = np.full(shape=(Return.shape[0], len(AllCats)), fill_value=np.nan, dtype=float)
        BMKR, TP, TPR, AA, SS, IN = BMK.copy(), BMK.copy(), BMK.copy(), BMK.copy(), BMK.copy(), BMK.copy()
        for i, iCat in enumerate(AllCats):
            if iCat == "None": iMask = pd.isnull(CatData)
            else: iMask = (CatData == iCat)
            iPortfolio, iBmkPortfolio, iReturn = np.where(iMask, Portfolio, np.nan), np.where(iMask, Bmk, np.nan), np.where(iMask, Return, np.nan)
            iCatWeight, iBmkCatWeight = np.nansum(iPortfolio, axis=1), np.nansum(iBmkPortfolio, axis=1)
            BMK[:, i], TP[:, i] = iBmkCatWeight, iCatWeight
            BMKR[:, i] = np.nansum(iBmkPortfolio * iReturn, axis=1) / iBmkCatWeight
            TPR[:, i] = np.nansum(iPortfolio * iReturn , axis=1) / iCatWeight
            iP2R, iP3R = iCatWeight * BMKR[:, i], iBmkCatWeight * TPR[:, i]
            AA[:, i] = iP2R - BMKR[:, i]
            SS[:, i] = iP3R - BMKR[:, i]
            IN[:, i] = TPR[:, i] - iP3R - iP2R + BMKR[:, i]
        AAA = (TP - BMK) * np.nansum(BMKR, axis=1, keepdims=True)
        Rslt = unstructured_to_structured(np.array([BMK, BMKR, TP, TPR, AA, SS, IN, AAA]).swapaxes(0, -1), dtype=self._QSArgs.CompoundType).T.astype("O")
        Rslt = pd.DataFrame(Rslt, index=DTs, columns=AllCats).reindex(index=idt, columns=iid)
        return Rslt.values[self._QSArgs.LookBack[0]:]
    
    def __call__(self, p:Factor, price: Factor, cat_data: Factor, bmk:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        """将算子作用在若干个因子对象上以产生 Brinson 绩效分析因子

        Args:
            p: 待分析的投资组合因子
            price: 证券价格或者净值因子
            cat_data: 类别因子, 比如行业等
            bmk: 基准投资组合因子，如果为 None 表示没有基准，考察绝对收益
            factor_args: 创建 IC 因子时传递个它的参数集
            kwargs: 创建 IC 因子时传递给它的其他入参

        Returns:
            Brinson 绩效分析因子，该因子为复合因子，包含的子因子有:
            * BMK: 业绩基准组合在各大类资产上的权重
            * BMKR: 业绩基准组合在各大类资产上的收益
            * TP: 目标投资组合在各大类资产上的权重
            * TPR: 目标投资组合在各大类资产上的收益
            * AA: 资产配置收益(Return of Asset Allocation)
            * SS: 个券选择收益(Return of Stock Selection)
            * IN: 交互作用(Interaction)
            * AAA: 调整的资产配置收益
        """
        Factors = [p, price, cat_data]
        if bmk is not None: Factors.append(bmk)
        if "SectionIDs" not in factor_args:
            factor_args["SectionIDs"] = self._QSArgs.ModelArgs["section_ids"]
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"bmk": (bmk is not None)}
        kwargs["operator_kwargs"] =  {"descriptor_ids": self._QSArgs.DescriptorSection[0], "lookback": self._QSArgs.LookBack[0], "section_ids": factor_args["SectionIDs"]} | kwargs.get("operator_kwargs", {})
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


class BrinsonModel(BTNode):
    """Brinson 绩效分析模型"""

    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="Brinson 绩效分析模型", frozen=True, title="名称")
        
    def __init__(self, brinson: Factor, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=[brinson], args=args, config_file=config_file, **kwargs)
    
    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if self.Deps[0]._QSArgs.CalcDTRuler:
            HTML += "<li>计算时点: 自定义时点</li>"
        else:
            HTML += "<li>计算时点: 所有时点</li>"
        HTML += "</ul>"
        Formatters = [_QS_formatPandasPercentage] * 8
        iHTML = output["多期综合"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        # Fig = self.genMatplotlibFig(output)
        # # figure 保存为二进制文件
        # Buffer = BytesIO()
        # Fig.savefig(Buffer, bbox_inches='tight')
        # PlotData = Buffer.getvalue()
        # # 图像数据转化为 HTML 格式
        # ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        # HTML += ('<img src="%s">' % ImgStr)
        return HTML

    def init_compute(self, path: List[str], init_data: DTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=None) for iInitData in InitData]
    
    def forward_compute(self, path: List[str], fwd_data: DTLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], DTLocalContext]:
        return [FactorLocalContext(DTs=fwd_data.DTs, IDs=context.NodeState[iDep.QSID]["section_ids"], PIDs=context.PIDList) for iDep in self.Deps], DTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[DTLocalContext]=None) -> dict:
        Brinson = bwd_data_list[0].dropna(how="all", axis=0)
        DTs, IDs, Brinson = Brinson.index, Brinson.columns, Brinson.values
        if np.any(pd.isnull(Brinson)):
            DefaultData = np.array([[None]], dtype="O")
            DefaultData[0, 0] = (np.nan,) * 8
            DefaultData = DefaultData.repeat(Brinson.shape[0], axis=0).repeat(Brinson.shape[1], axis=1)
            Brinson = np.where(pd.notnull(Brinson), Brinson, DefaultData)
        Brinson = Brinson.astype(np.dtype([("BMK", float), ("BMKR", float), ("TP", float), ("TPR", float), ("AA", float), ("SS", float), ("IN", float), ("AAA", float)]))
        Output = {}
        Output["策略组合资产权重"] = pd.DataFrame(Brinson["TP"], index=DTs, columns=IDs)
        Output["基准组合资产权重"] = pd.DataFrame(Brinson["BMK"], index=DTs, columns=IDs)
        Output["策略组合资产收益"] = pd.DataFrame(Brinson["TPR"], index=DTs, columns=IDs)
        Output["基准组合资产收益"] = pd.DataFrame(Brinson["BMKR"], index=DTs, columns=IDs)
        Output["主动资产配置超额收益"] = pd.DataFrame(Brinson["AA"], index=DTs, columns=IDs)
        Output["主动个券选择超额收益"] = pd.DataFrame(Brinson["SS"], index=DTs, columns=IDs)
        Output["主动资产配置组合收益"] = Output["主动资产配置超额收益"] + Output["基准组合资产收益"]
        Output["主动个券选择组合收益"] = Output["主动个券选择超额收益"] + Output["基准组合资产收益"]
        Output["交互作用超额收益"] = pd.DataFrame(Brinson["IN"], index=DTs, columns=IDs)
        Output["总超额收益"] = Output["策略组合资产收益"] - Output["基准组合资产收益"]
        Output["主动资产配置组合收益(修正)"] = pd.DataFrame(Brinson["AAA"], index=DTs, columns=IDs)
        Output["总计"] = pd.DataFrame(Output["策略组合资产权重"].sum(axis=1), columns=["策略组合资产权重"])
        Output["总计"]["基准组合资产权重"] = Output["基准组合资产权重"].sum(axis=1)
        Output["总计"]["策略组合收益"] = Output["策略组合资产收益"].sum(axis=1)
        Output["总计"]["基准组合收益"] = Output["基准组合资产收益"].sum(axis=1)
        Output["总计"]["主动资产配置组合收益"] = Output["主动资产配置组合收益"].sum(axis=1)
        Output["总计"]["主动资产配置组合收益(修正)"] = Output["主动资产配置组合收益(修正)"].sum(axis=1)
        Output["总计"]["主动个券选择组合收益"] = Output["主动个券选择组合收益"].sum(axis=1)
        Output["总计"]["主动资产配置超额收益"] = Output["主动资产配置超额收益"].sum(axis=1)
        Output["总计"]["主动个券选择超额收益"] = Output["主动个券选择超额收益"].sum(axis=1)
        Output["总计"]["交互作用超额收益"] = Output["交互作用超额收益"].sum(axis=1)
        Output["总计"]["总超额收益"] = Output["总超额收益"].sum(axis=1)
        Output["多期综合"] = pd.DataFrame(dtype=float)
        Output["多期综合"]["策略组合资产收益"] = (Output["策略组合资产收益"] + 1).prod(axis=0) - 1
        Output["多期综合"]["基准组合资产收益"] = (Output["基准组合资产收益"] + 1).prod(axis=0) - 1
        Output["多期综合"]["主动资产配置组合收益"] = (Output["主动资产配置组合收益"] + 1).prod() - 1
        Output["多期综合"]["主动个券选择组合收益"] = (Output["主动个券选择组合收益"] + 1).prod() - 1
        Output["多期综合"]["主动资产配置超额收益"] = Output["多期综合"]["主动资产配置组合收益"] - Output["多期综合"]["基准组合资产收益"]
        Output["多期综合"]["主动个券选择超额收益"] = Output["多期综合"]["主动个券选择组合收益"] - Output["多期综合"]["基准组合资产收益"]
        Output["多期综合"]["交互作用超额收益"] = Output["多期综合"]["策略组合资产收益"] - Output["多期综合"]["主动资产配置组合收益"] - Output["多期综合"]["主动个券选择组合收益"] + Output["多期综合"]["基准组合资产收益"]
        Output["多期综合"]["总超额收益"] = Output["多期综合"]["策略组合资产收益"] - Output["多期综合"]["基准组合资产收益"]
        Output["多期综合"].loc["总计"] = (Output["总计"] + 1).prod(axis=0) - 1
        k_t = (np.log(1 + Output["总计"]["策略组合收益"]) - np.log(1 + Output["总计"]["基准组合收益"])) / (Output["总计"]["策略组合收益"] - Output["总计"]["基准组合收益"])
        k_t[pd.isnull(k_t)] = 1.0
        if Output["多期综合"].loc["总计", "策略组合资产收益"] != Output["多期综合"].loc["总计", "基准组合资产收益"]:
            k = (np.log(Output["多期综合"].loc["总计", "策略组合资产收益"] + 1) - np.log(Output["多期综合"].loc["总计", "基准组合资产收益"]+1)) / (Output["多期综合"].loc["总计", "策略组合资产收益"] - Output["多期综合"].loc["总计", "基准组合资产收益"])
        else:
            k = 1.0
        Output["多期综合"].loc["总计", "主动资产配置超额收益"] = (Output["总计"]["主动资产配置超额收益"] * k_t).sum() / k
        Output["多期综合"].loc["总计", "主动个券选择超额收益"] = (Output["总计"]["主动个券选择超额收益"] * k_t).sum() / k
        Output["多期综合"].loc["总计", "交互作用超额收益"] = (Output["总计"]["交互作用超额收益"] * k_t).sum() / k
        Output["多期综合"].loc["总计", "总超额收益"] = Output["多期综合"].loc["总计", "策略组合资产收益"] - Output["多期综合"].loc["总计", "基准组合资产收益"]
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
