# coding=utf-8
"""Bias Test"""
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, List, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import DTInitData, DTLocalContext
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData, FactorLocalContext, DataFactor
from QuantStudio.Factor.BasicOperator import rename
import QuantStudio.Factor.FactorOperator as fo
from QuantStudio.Factor.FactorOperation import SectionOperation, SectionOperator
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatPandasPercentage
from QuantStudio.Risk.RiskTable import RiskTable
from QuantStudio.BackTest.SectionFactor.Portfolio import CalcPortfolioReturn


class PortfolioVolatility(SectionOperation):
    def init_compute(self, path, init_data, context):
        InitData = super().init_compute(path, init_data, context)
        return InitData[:-1] + [FactorInitData(DTRange=InitData[-1].DTRange, SectionIDs=self._Operator._QSArgs.DescriptorSection[0])]
    
    def forward_compute(self, path, fwd_data, context):
        FwdData, LocalContext = super().forward_compute(path, fwd_data, context)
        return FwdData[:-1] + [FactorLocalContext(IDs=self._QS_getDescriptorSectionIDs(0, context), DTs=FwdData[-1].DTs, PIDs=context.PIDList)], LocalContext

class CalcPortfolioVolatility(SectionOperator):
    """投资组合波动率计算算子"""

    def __init__(self, descriptor_ids:List[str], args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化投资组合波动率计算算子

        Args:
            descriptor_ids: 依赖因子的截面 ID 序列
            args: 参数集
            config_file: 配置文件地址
        """
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcPortfolioVolatility"} | args | {"DTMode": "单时点", "DataType": "double"}
        Args["ModelArgs"] = {} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: dt.datetime, iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        CovMatrix = x[-1].loc[idt].reindex(index=SectionIDs, columns=SectionIDs)
        Rslt = np.full(shape=(len(iid), ), fill_value=np.nan, dtype=float)
        if CovMatrix.isnull().all().all(): return Rslt
        CovMatrix = CovMatrix.fillna(0.0)
        for i, ix in enumerate(x[:-1]):
            iPortfolio = np.where(pd.notnull(ix), ix, 0.0)
            Rslt[i] = np.dot(np.dot(iPortfolio, CovMatrix), iPortfolio) ** 0.5
        return Rslt

    def __call__(self, *portfolio:Factor, risk_table:RiskTable, portfolio_name_list:Optional[List[str]]=None, factor_args:dict={}, **kwargs) -> PortfolioVolatility:
        """将算子作用在若干个投资组合因子对象上以产生波动率因子

        Args:
            portfolio: 待计算收益率的投资组合因子, 因子值是每个时点投资于某个证券的资金权重，如果某个时点的因子值全部为 NaN 表示改时点没有信号，不进行调仓
            risk_table: 风险表, 用于提供证券的协方差矩阵
            portfolio_name_list: 投资组合的名称列表, None 表示由系统自动生成, 非 None 时将作为收益率因子的截面 ID 序列，所以不能有重复
            factor_args: 创建波动率因子时传递个它的参数集
            kwargs: 创建波动率因子时传递给它的其他入参

        Returns:
            投资组合波动率因子
        """
        if not portfolio: raise __QS_Error__("投资组合因子不能为空!")
        if portfolio_name_list is not None:
            if factor_args.get("SectionIDs", None) is not None:
                self.Logger.warning(f"CalcPortfolioNV.__call__: 同时指定了投资组合名称列表 portfolio_name_list({portfolio_name_list})以及因子截面ID参数 SectionIDs({factor_args['SectionIDs']}), 将使用后者作为因子的截面ID, 忽略 portfolio_name_list")
                portfolio_name_list = factor_args["SectionIDs"]
        elif factor_args.get("SectionIDs", None) is not None:
            portfolio_name_list = factor_args["SectionIDs"]
        else:
            portfolio_name_list = [iFactor.Name for iFactor in portfolio]
            if len(set(portfolio_name_list)) != len(portfolio):
                PosNum = int(np.log10(max(1, len(portfolio) - 1))) + 1
                portfolio_name_list = [f"P{str(i).zfill(PosNum)}" for i in range(len(portfolio))]
                self.Logger.info(f"投资组合因子的名称中有重复, 使用系统自动生成的投资组合名称列表: {portfolio_name_list}")
        if len(set(portfolio_name_list)) != len(portfolio):
            raise __QS_Error__(f"投资组合的名称列表 : {portfolio_name_list} 长度不等于投资组合因子列表 portfolio 的长度或者有重复!")
        else:
            SortedIdx = np.argsort(portfolio_name_list)
            if not np.all(SortedIdx == np.arange(len(portfolio_name_list))):
                self.Logger.warning(f"CalcPortfolioNV.__call__: 投资组合的名称列表({portfolio_name_list})不是升序排列，将按照升序重新排列投资组合")
                portfolio, SortedPortfolioNameList = [portfolio[i] for i in SortedIdx], [portfolio_name_list[i] for i in SortedIdx]
            else:
                SortedPortfolioNameList = portfolio_name_list
        factor_args["SectionIDs"] = SortedPortfolioNameList
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"portfolio_name_list": portfolio_name_list}
        Operator = self._QS_validate(*portfolio, descriptor_ids=self._QSArgs.DescriptorSection[0])
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for iFactor in portfolio]
        return PortfolioVolatility(descriptors=Descriptors, extra_deps=[risk_table], args={"Operator": Operator, **factor_args}, **kwargs)


class CalcRandomPortfolio(SectionOperator):
    """随机投资组合生成算子"""

    def __init__(self, target_num:int=20, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化随机投资组合生成算子

        Args:
            target_num: 每一期的目标持仓数量
        """
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcRandomPortfolio"} | args | {"DTMode": "单时点", "DataType": "double"}
        Args["ModelArgs"] = {"target_num": target_num} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [None])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Weight = x[0]
        Mask = pd.notnull(Weight)
        if f._QSArgs.ModelArgs["mask"]: Mask = ((x[1] == 1) & Mask)
        nPos = np.sum(Mask)
        KeepPos = np.random.choice(nPos, size=min(args["target_num"], nPos), replace=False)
        Portfolio = np.zeros((nPos, ))
        Portfolio[KeepPos] = Weight[Mask][KeepPos]
        Portfolio = Portfolio / np.nansum(Portfolio)
        Rslt = np.full_like(Weight, fill_value=np.nan)
        Rslt[Mask] = Portfolio
        return Rslt

    def __call__(self, weight: Factor, mask: Optional[Factor]=None, factor_args:dict={}, **kwargs) -> SectionOperation:
        Factors = [weight]
        if mask is not None: Factors.append(mask)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


class BiasTest(BTNode):
    """Bias Test"""

    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="Bias Test", frozen=True, title="名称")
        IndustryList: List[str] = Field(default=[], frozen=True, title="行业列表")
        RandomNums: List[int] = Field(default=[20, 50, 100, 200], frozen=True, title="随机组合持仓数量")
        RebalanceDTs: Optional[List[dt.datetime]] = Field(default=None, title="再平衡时点", frozen=True)
        RollingAvgPeriod: int = Field(default=12, frozen=True, title="移动平均期数")
    
    def _genPortfolio(self, descriptor_ids, industry_list, random_nums, mask, weight_dict, style_dict, industry, industry_neutral_factor_dict, extra_portfolio_dict):
        if industry is not None:
            AllIndustries = sorted(industry_list)
        notnull, where, aggr_sum, section_rank = fo.NotNull(), fo.Where(), fo.Aggregate(aggr_func=np.nansum, descriptor_ids=descriptor_ids), fo.SectionRank()
        PortfolioDict = {}
        for iWeightName, iWeightFactor in weight_dict.items():
            if mask is not None:
                iMask = (notnull(iWeightFactor) & (iWeightFactor != 0) & mask)
            else:
                iMask = (notnull(iWeightFactor) & (iWeightFactor != 0))
            # 全部 ID 组合
            PortfolioDict[f"全体{iWeightName}加权组合"] = rename(iWeightFactor / aggr_sum(abs(iWeightFactor), mask=iMask), factor_name=f"全体{iWeightName}加权组合")
            if industry is not None:
                # 行业组合
                for jIndustry in AllIndustries:
                    ijMask = ((industry == jIndustry) & iMask)
                    ijWeightFactor = where(iWeightFactor, mask=ijMask, other=np.nan)
                    PortfolioDict[f"{jIndustry}行业{iWeightName}加权组合"] = rename(ijWeightFactor / aggr_sum(abs(ijWeightFactor)), factor_name=f"{jIndustry}行业{iWeightName}加权组合")
                # 行业中性组合
                for jFactorName, jFactor in industry_neutral_factor_dict.items():
                    jRank = section_rank(jFactor, mask=iMask, cat_data=industry)
                    ijWeightFactor = where(iWeightFactor, mask=(jRank > 0.5), other=np.nan)
                    PortfolioDict[f"{jFactorName}行业中性Top{iWeightName}加权组合"] = rename(ijWeightFactor / aggr_sum(abs(ijWeightFactor)), factor_name=f"{jFactorName}行业中性Top{iWeightName}加权组合")
                    ijWeightFactor = where(iWeightFactor, mask=(jRank <= 0.5), other=np.nan)
                    PortfolioDict[f"{jFactorName}行业中性Bottom{iWeightName}加权组合"] = rename(ijWeightFactor / aggr_sum(abs(ijWeightFactor)), factor_name=f"{jFactorName}行业中性Bottom{iWeightName}加权组合")
            # 风格因子组合
            for jStyleName, jStyle in style_dict.items():
                jRank = section_rank(jStyle, mask=iMask)
                ijWeightFactor = where(iWeightFactor, mask=(jRank >= 0.8), other=np.nan)
                PortfolioDict[f"{jStyleName}风格Top{iWeightName}加权组合"] = rename(ijWeightFactor / aggr_sum(abs(ijWeightFactor)), factor_name=f"{jStyleName}风格Top{iWeightName}加权组合")
                ijWeightFactor = where(iWeightFactor, mask=(jRank <= 0.2), other=np.nan)
                PortfolioDict[f"{jStyleName}风格Bottom{iWeightName}加权组合"] = rename(ijWeightFactor / aggr_sum(abs(ijWeightFactor)), factor_name=f"{jStyleName}风格Bottom{iWeightName}加权组合")
            # 随机组合
            for jNum in random_nums:
                PortfolioDict[f"随机{jNum}{iWeightName}加权组合"] = CalcRandomPortfolio(target_num=jNum)(iWeightFactor, mask=iMask, factor_args={"Name": f"随机{jNum}{iWeightName}加权组合"})
        PortfolioDict.update(extra_portfolio_dict)
        return [PortfolioDict[key] for key in sorted(PortfolioDict.keys())]

    def __init__(self, descriptor_ids:List[str], price: Factor, risk_table: RiskTable, mask:Optional[Factor]=None, weight_list: List[Factor]=[DataFactor(1, args={"Name": "等权"})], style_list: List[Factor]=[], industry:Optional[Factor]=None, industry_neutral_factor_list:List[Factor]=[], extra_portfolio_list:List[Factor]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 Bias Test 回测节点

        Args:
            descriptor_ids: 依赖因子的截面 ID
            price: 证券价格或者净值因子
            weight_list: 权重因子列表, 默认元素为等权因子
            style_list: 风格因子列表
            industry: 行业因子
            industry_neutral_factor_list: 行业中性因子列表
            extra_portfolio_list: 其他投资组合因子列表
            args: 参数集
            config_file: 配置文件
        """
        if (industry is not None) and (not args.get("IndustryList", [])): raise __QS_Error__(f"指定了行业因子 {industry} 则行业列表参数不能为空")
        WeightDict = {iFactor.Name: iFactor for iFactor in weight_list}
        if len(WeightDict) != len(weight_list): raise __QS_Error__(f"指定的权重因子列表有重名: {[iFactor.Name for iFactor in weight_list]}")
        StyleDict = {iFactor.Name: iFactor for iFactor in style_list}
        if len(StyleDict) != len(style_list): raise __QS_Error__(f"指定的风格因子列表有重名: {[iFactor.Name for iFactor in style_list]}")
        IndustryNeutralFactorDict = {iFactor.Name: iFactor for iFactor in industry_neutral_factor_list}
        if len(IndustryNeutralFactorDict) != len(industry_neutral_factor_list): raise __QS_Error__(f"指定的行业中性因子列表有重名: {[iFactor.Name for iFactor in industry_neutral_factor_list]}")
        ExtraPortfolioDict = {iFactor.Name: iFactor for iFactor in extra_portfolio_list}
        if len(set(ExtraPortfolioDict)) != len(extra_portfolio_list): raise __QS_Error__(f"额外指定的投资组合因子列表有重名: {[iFactor.Name for iFactor in extra_portfolio_list]}")
        PortfolioList = self._genPortfolio(descriptor_ids, args.get("IndustryList", []), args.get("RandomNums", [20, 50, 100, 200]), mask, WeightDict, StyleDict, industry, IndustryNeutralFactorDict, ExtraPortfolioDict)
        if not PortfolioList: raise __QS_Error__("在给定的因子条件下生成的投资组合为空!")
        self._PortfolioNameList = [iFactor.Name for iFactor in PortfolioList]

        CalcDTRuler = args.get("RebalanceDTs", None)
        if not CalcDTRuler: LookBack = 1
        else: LookBack = max(d.days for d in np.diff(CalcDTRuler))
        PortfolioReturn = CalcPortfolioReturn(descriptor_ids=descriptor_ids, lookback=LookBack)(*PortfolioList, price=price, factor_args={"CalcDTRuler": CalcDTRuler})
        PortfolioVolatility = CalcPortfolioVolatility(descriptor_ids=descriptor_ids)(*PortfolioList, risk_table=risk_table, factor_args={"CalcDTRuler": CalcDTRuler})
        ZScore = PortfolioReturn / fo.Lag(lag_period=1, window=LookBack)(PortfolioVolatility, factor_args={"CalcDTRuler": CalcDTRuler})
        super().__init__(deps=[ZScore], args=args, config_file=config_file, **kwargs)

    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        HTML += f"<li>行业列表: {self._QSArgs.IndustryList}</li>"
        HTML += f"<li>随机组合持仓数量: {self._QSArgs.RandomNums}</li>"
        if self._QSArgs.RebalanceDTs:
            HTML += "<li>再平衡时点: 自定义时点</li>"
        else:
            HTML += "<li>再平衡时点: 所有时点</li>"
        HTML += f"<li>移动平均期数: {self._QSArgs.RollingAvgPeriod}</li>"
        HTML += "</ul>"
        Formatters = [lambda x:'{0:.4f}'.format(x)]*2+[_QS_formatPandasPercentage]*6
        iHTML = output["汇总统计量"].to_html(formatters=Formatters)
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
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=self._PortfolioNameList) for iInitData in InitData]
    
    def forward_compute(self, path: List[str], fwd_data: DTLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], DTLocalContext]:
        return [FactorLocalContext(DTs=fwd_data.DTs, IDs=self._PortfolioNameList, PIDs=context.PIDList)] * len(self.Deps), DTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[DTLocalContext]=None) -> dict:
        ZScore = bwd_data_list[0].dropna(how="all", axis=0)
        Output = {"Z-Score": ZScore}
        Output["Robust Z-Score"] = ZScore.clip(upper=3, lower=-3)
        Output["Bias 统计量"] = ZScore.rolling(window=self._QSArgs.RollingAvgPeriod).std()
        Output["Robust Bias 统计量"] = Output["Robust Z-Score"].rolling(window=self._QSArgs.RollingAvgPeriod).std()
        Output["汇总统计量"] = pd.DataFrame(index=Output["Bias 统计量"].columns)
        Output["汇总统计量"]["RAD 统计量"] = (Output["Bias 统计量"] - 1).abs().mean()
        Output["汇总统计量"]["Robust RAD 统计量"] = (Output["Robust Bias 统计量"] - 1).abs().mean()
        Output["Bias 统计量"].insert(0, "95%置信下界", 1 - (2 / self._QSArgs.RollingAvgPeriod) ** 0.5)
        Output["Bias 统计量"].insert(0, "95%置信上界", 1 + (2 / self._QSArgs.RollingAvgPeriod) ** 0.5)
        Output["Robust Bias 统计量"].insert(0, "95%置信下界", 1 - (2 / self._QSArgs.RollingAvgPeriod) ** 0.5)
        Output["Robust Bias 统计量"].insert(0, "95%置信上界", 1 + (2 / self._QSArgs.RollingAvgPeriod) ** 0.5)
        Stats = Output["Bias 统计量"].iloc[:, 2:]
        SampleNum = pd.notnull(Stats).sum(axis=0)
        Output["汇总统计量"]["Bias 统计量高估比例"] = (Stats.T < Output["Bias 统计量"]["95%置信下界"]).sum(axis=1) / SampleNum
        Output["汇总统计量"]["Bias 统计量低估比例"] = (Stats.T > Output["Bias 统计量"]["95%置信上界"]).sum(axis=1) / SampleNum
        Output["汇总统计量"]["Bias 统计量准确度"] = 1 - Output["汇总统计量"]["Bias 统计量高估比例"]  - Output["汇总统计量"]["Bias 统计量低估比例"]
        Stats = Output["Robust Bias 统计量"].iloc[:, 2:]
        SampleNum = pd.notnull(Stats).sum(axis=0)
        Output["汇总统计量"]["Robust Bias 统计量高估比例"] = (Stats.T<Output["Robust Bias 统计量"]["95%置信下界"]).sum(axis=1) / SampleNum
        Output["汇总统计量"]["Robust Bias 统计量低估比例"] = (Stats.T>Output["Robust Bias 统计量"]["95%置信上界"]).sum(axis=1) / SampleNum
        Output["汇总统计量"]["Robust Bias 统计量准确度"] = 1 - Output["汇总统计量"]["Robust Bias 统计量高估比例"]  - Output["汇总统计量"]["Robust Bias 统计量低估比例"]
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
