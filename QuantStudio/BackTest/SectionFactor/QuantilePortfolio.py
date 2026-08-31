# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, List, Any, Tuple

import numpy as np
import pandas as pd
from pydantic import Field
import statsmodels.api as sm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import DTLocalContext, DTInitData
import QuantStudio.Factor.FactorOperator as fo
from QuantStudio.Factor.FactorOperation import SectionOperation
from QuantStudio.Factor.Factor import Factor, FactorInitData, FactorContext, FactorLocalContext
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.BackTest.Strategy.AllocationStrategy import CalcMaskPortfolio
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.Tools.StrategyTestFun import calcMaxDrawdownRate, calcLSYield


def _QS_plotStatistics(axes, x_data, x_ticklabels, left_data, left_formatter, right_data=None, right_formatter=None, right_axes=True, title=None):
    axes.yaxis.set_major_formatter(left_formatter)
    axes.bar(x_data, left_data.values, label=left_data.name, color="steelblue")
    if right_data is not None:
        if right_axes:
            axes.legend(loc='upper left')
            right_axes = axes.twinx()
            right_axes.yaxis.set_major_formatter(right_formatter)
            right_axes.plot(x_data, right_data.values, label=right_data.name, color="indianred", lw=2.5)
            right_axes.legend(loc="upper right")
        else:
            axes.plot(x_data, right_data.values, label=right_data.name, color="indianred", lw=2.5)
            axes.legend(loc='best')
    else:
        axes.legend(loc='best')
    axes.set_xticks(x_data)
    axes.set_xticklabels(x_ticklabels)
    if title is not None:
        axes.set_title(title)
    return axes


def makeQuantilePortfolio(factor:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, weight:Optional[Factor]=None, descriptor_ids:Optional[List[str]]=None, rebalance_dts:Optional[List[dt.datetime]]=None, ascending:bool=False, group_num:int=5, **kwargs) -> List[SectionOperation]:
    """创建分位数组合

    Args:
        factor: 用于分组的因子对象
        mask: 用于初始筛选的因子对象
        cat_data: 分类因子, 如果非 None 表示在每个类别里进行单独分组再汇总形成总的投资组合
        weight: 权重因子, 用于给投资组合里的证券设置权重
        descriptor_ids: 截面 ID 序列
        rebalance_dts: 组合再平衡时点序列
        ascending: 因子排序是否为升序
        group_num: 分组数
    
    Returns:
        创建的分位数组合因子对象列表
    """
    rank = fo.SectionRank(ascending=ascending, uniformization=True)
    Rank = rank(factor, mask=mask, cat_data=cat_data, factor_args={"CalcDTRuler": rebalance_dts})
    if weight is None: weight = 1
    calcMaskPortfolio = CalcMaskPortfolio(descriptor_ids=descriptor_ids, **kwargs)
    Portfolio = []
    for i in range(group_num):
        iLeft, iRight = i / group_num, (i+1) / group_num
        iPortfolio = calcMaskPortfolio((Rank >= iLeft) & (Rank < iRight), weight=weight)
        Portfolio.append(iPortfolio)
    return Portfolio


class MultiPortfolio(BTNode):
    """多组合对比"""

    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="多组合对比", frozen=True, title="名称")
        LSPairs: List[Tuple[str, str]] = Field(default=[], title="多空组合对", frozen=True, description="构造多空组合的投资组合对, 比如 [('P0', 'P1')] 表示 P0 组合和 P1 组合构成一个多空组合, 将考察它的表现")
        RebalanceDTs: Optional[List[dt.datetime]] = Field(default=None, title="再平衡时点", frozen=True)
        PortfolioSection: Optional[List[List[str]]] = Field(default=None, frozen=True, title="多组合截面ID")
        BmkSection: Optional[List[str]] = Field(default=None, frozen=True, title="基准组合截面ID")

    def __init__(self, 
        nv:Factor, 
        bmk_nv:Optional[Factor]=None, 
        portfolio_list:Optional[List[Factor]]=None, 
        bmk_portfolio:Optional[Factor]=None, 
        args:dict={}, config_file:Optional[str]=None, **kwargs
    ):
        """初始化多组合对比回测模块

        Args:
            nv: 多组合策略净值因子对象
            bmk_nv: 基准策略净值因子对象
            portfolio_list: 各组合策略对应的投资组合因子对象
            bmk_portfolio: 基准策略的投资组合因子对象
        """
        self._NV = nv
        self._BmkNV = bmk_nv
        self._PortfolioList = portfolio_list
        self._BmkPortfolio = bmk_portfolio
        Deps = [nv]
        if bmk_nv is not None: Deps.append(bmk_nv)
        if portfolio_list:
            if (nv._QSArgs.SectionIDs is not None) and len(nv._QSArgs.SectionIDs) != len(portfolio_list):
                raise __QS_Error__("净值因子的截面 ID 长度与传入的投资组合列表(portfolio_list)长度不一致!")
            Deps += portfolio_list
        if (bmk_nv is not None) and (bmk_portfolio is not None): Deps.append(bmk_portfolio)
        super().__init__(deps=Deps, args=args, config_file=config_file, **kwargs)
    
    @staticmethod
    def genMatplotlibFig(output:dict, file_path:Optional[str]=None) -> Figure:
        GroupNum = output["超额净值"].shape[1]
        nLS = output["净值"].shape[1] - 1 - GroupNum
        nRow, nCol = 3 + int(0 if nLS <= 0 else (nLS - 1) // 3 + 1), 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(1, GroupNum + 1)
        xTickLabels = [str(iInd) for iInd in output["统计数据"].index[:GroupNum]]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 1), xData, xTickLabels, output["统计数据"]["年化超额收益率"].iloc[:GroupNum], PercentageFormatter, output["统计数据"]["胜率"].iloc[:GroupNum], PercentageFormatter)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 2), xData, xTickLabels, output["统计数据"]["信息比率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 3), xData, xTickLabels, output["统计数据"]["超额最大回撤率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 4), xData, xTickLabels, output["统计数据"]["年化收益率"].iloc[:GroupNum], PercentageFormatter, pd.Series(output["统计数据"].loc["基准", "年化收益率"], index=output["统计数据"].index[:GroupNum], name="基准"), PercentageFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 5), xData, xTickLabels, output["统计数据"]["Sharpe比率"].iloc[:GroupNum], FloatFormatter, pd.Series(output["统计数据"].loc["基准", "Sharpe比率"], index=output["统计数据"].index[:GroupNum], name="基准"), FloatFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 6), xData, xTickLabels, output["统计数据"]["平均换手率"].iloc[:GroupNum], PercentageFormatter, None)
        Axes = Fig.add_subplot(nRow, nCol, 7)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum):
            Axes.plot(output["超额净值"].index, output["超额净值"].iloc[:, i].values, label=str(output["超额净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("超额净值")
        Axes = Fig.add_subplot(nRow, nCol, 8)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum+1):
            Axes.plot(output["净值"].index, output["净值"].iloc[:, i].values, label=str(output["净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("多头净值")
        Axes = Fig.add_subplot(nRow, nCol, 9)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum):
            iName = str(output["净值"].columns[i])
            iNum = (output["投资组合"][iName]>0).sum(axis=1)
            Axes.plot(iNum.index, iNum.values, label=f"{iName}: {round(iNum.mean(),2)}", lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("持仓数量")
        for i in range(nLS):
            Axes = Fig.add_subplot(nRow, nCol, 10+i)
            xData = np.arange(0, output["净值"].shape[0])
            xTicks = np.arange(0, output["净值"].shape[0], max(1, int(output["净值"].shape[0]/8)))
            xTickLabels = [output["净值"].index[i].strftime("%Y-%m-%d") for i in xTicks]
            iLSName = output["净值"].columns[GroupNum+1+i]
            Axes.plot(xData, output["净值"][iLSName].values, label="多空净值", color="indianred", lw=2.5)
            Axes.legend(loc='upper left')
            RAxes = Axes.twinx()
            RAxes.yaxis.set_major_formatter(PercentageFormatter)
            RAxes.bar(xData, output["收益率"][iLSName].values, label="多空收益率", color="steelblue")
            RAxes.legend(loc="upper right")
            Axes.set_xticks(xTicks)
            Axes.set_xticklabels(xTickLabels)
            Axes.set_title(iLSName)
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig

    @staticmethod
    def genOutputReport(output:dict) -> str:
        HTML = ""
        Formatters = [_QS_formatPandasPercentage] * 3 + [lambda x:'{0:.2f}'.format(x)] * 2 + [_QS_formatPandasPercentage] * 2 + [lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"] * 2
        Formatters += [_QS_formatPandasPercentage] * 3 + [lambda x:'{0:.2f}'.format(x)] * 2 + [_QS_formatPandasPercentage] * 2 + [lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        Formatters += [lambda x:'{0:.2f}'.format(x)] * 2
        iHTML = output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = MultiPortfolio.genMatplotlibFig(output)
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
        HTML += f"<li>多空组合对: {self._QSArgs.LSPairs}</li>"
        if self._QSArgs.RebalanceDTs is not None:
            HTML += "<li>再平衡时点: 自定义时点</li>"
        else:
            HTML += "<li>再平衡时点: 所有时点</li>"
        HTML += "</ul>"
        HTML += "\n" + MultiPortfolio.genOutputReport(output=output)
        return HTML

    def _QS_calcStats(self, output):
        nDT = output["净值"].shape[0] - 1
        nDays = (output["净值"].index[-1] - output["净值"].index[0]).days
        nYear = nDays / 365
        TotalReturn = output["净值"].iloc[-1, :] / output["净值"].iloc[0, :] - 1
        output["统计数据"] = pd.DataFrame(index=TotalReturn.index)
        output["统计数据"]["总收益率"] = TotalReturn
        output["统计数据"]["年化收益率"] = (1 + TotalReturn) ** (1 / nYear) - 1
        output["统计数据"]["波动率"] = output["收益率"].std() * np.sqrt(nDT / nYear)
        output["统计数据"]["Sharpe比率"] = output["统计数据"]["年化收益率"] / output["统计数据"]["波动率"]
        output["统计数据"]["t统计量(Sharpe比率)"] = (output["统计数据"]["Sharpe比率"] - output["统计数据"]["Sharpe比率"]["基准"]) / np.sqrt(2/nYear)
        output["统计数据"]["平均换手率"] = output["换手率"].mean()
        output["统计数据"]["最大回撤率"] = pd.Series(np.nan,index=output["统计数据"].index)
        output["统计数据"]["最大回撤开始时间"] = pd.Series(index=output["统计数据"].index,dtype="O")
        output["统计数据"]["最大回撤结束时间"] = pd.Series(index=output["统计数据"].index,dtype="O")
        for iCol in output["净值"].columns:
            iMaxDD,iStartPos,iEndPos = calcMaxDrawdownRate(output["净值"].loc[:, iCol].values)
            output["统计数据"].loc[iCol, "最大回撤率"] = abs(iMaxDD)
            output["统计数据"].loc[iCol, "最大回撤开始时间"] = (output["净值"].index[iStartPos] if iStartPos is not None else None)
            output["统计数据"].loc[iCol, "最大回撤结束时间"] = (output["净值"].index[iEndPos] if iEndPos is not None else None)
        output["统计数据"]["超额收益率"] = output["超额净值"].iloc[-1, :] - 1
        output["统计数据"]["年化超额收益率"] = (1 + output["统计数据"]["超额收益率"]) ** (1/nYear) - 1
        output["统计数据"]["跟踪误差"] = output["超额收益率"].std() * np.sqrt(nDT/nYear)
        output["统计数据"]["信息比率"] = output["统计数据"]["年化超额收益率"] / output["统计数据"]["跟踪误差"]
        output["统计数据"]["t统计量(信息比率)"] = output["统计数据"]["信息比率"] * np.sqrt(nYear)
        output["统计数据"]["胜率"] = (output["超额收益率"] > 0).sum() / nDT
        output["统计数据"]["超额最大回撤率"] = pd.Series(np.nan, index=output["统计数据"].index)
        output["统计数据"]["超额最大回撤开始时间"] = pd.Series(index=output["统计数据"].index, dtype="O")
        output["统计数据"]["超额最大回撤结束时间"] = pd.Series(index=output["统计数据"].index, dtype="O")
        for iCol in output["超额净值"].columns:
            iMaxDD, iStartPos, iEndPos = calcMaxDrawdownRate(output["超额净值"].loc[:, iCol].values)
            output["统计数据"].loc[iCol, "超额最大回撤率"] = abs(iMaxDD)
            output["统计数据"].loc[iCol, "超额最大回撤开始时间"] = (output["超额净值"].index[iStartPos] if iStartPos is not None else None)
            output["统计数据"].loc[iCol, "超额最大回撤结束时间"] = (output["超额净值"].index[iEndPos] if iEndPos is not None else None)
        output["统计数据"]["CAPM Alpha"], output["统计数据"]["CAPM Beta"] = 0.0, 0.0
        xData = sm.add_constant(output["收益率"]["基准"].values, prepend=True)
        for iCol in output["收益率"].columns:
            yData = output["收益率"][iCol].values
            try:
                Result = sm.OLS(yData, xData, missing="drop").fit()
                output["统计数据"].loc[iCol, "CAPM Beta"] = Result.params[1]
                output["统计数据"].loc[iCol, "CAPM Alpha"] = Result.params[0]
            except:
                output["统计数据"].loc[iCol, "CAPM Beta"] = np.nan
                output["统计数据"].loc[iCol, "CAPM Alpha"] = np.nan
        return output

    def init_compute(self, path: List[str], init_data: DTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        InitDataList = [FactorInitData(DTRange=InitData[0].DTRange, SectionIDs=None)]
        if self._BmkNV is not None: InitDataList.append(FactorInitData(DTRange=InitData[1].DTRange, SectionIDs=None))
        if self._PortfolioList is not None: SectionList = self._QSArgs.PortfolioSection if self._QSArgs.PortfolioSection is not None else [None] * len(self._PortfolioList)
        if self._BmkPortfolio is not None: SectionList.append(self._QSArgs.BmkSection)
        InitDataList += [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=SectionList[i]) for i, iInitData in enumerate(InitData[len(InitDataList):])]
        return InitDataList
    
    def forward_compute(self, path: List[str], fwd_data: DTLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], DTLocalContext]:
        SectionList = [None]
        if self._BmkNV is not None: SectionList.append(None)
        if self._PortfolioList is not None: SectionList += (self._QSArgs.PortfolioSection if self._QSArgs.PortfolioSection is not None else [None] * len(self._PortfolioList))
        if self._BmkPortfolio is not None: SectionList.append(self._QSArgs.BmkSection)
        return [FactorLocalContext(DTs=fwd_data.DTs, IDs=SectionList[i] or iDep.Args.SectionIDs or context.SectionIDs, PIDs=context.PIDList, SectionIDs=SectionList[i] or iDep.Args.SectionIDs or context.SectionIDs) for i, iDep in enumerate(self.Deps)], DTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[DTLocalContext]=None) -> dict:
        PortfolioNV = bwd_data_list[0].astype(float)
        BmkNV = (bwd_data_list[1].iloc[:, 0] if self._BmkNV is not None else pd.Series(1, index=PortfolioNV.index))
        BmkNV = BmkNV.ffill().bfill()
        StartIdx = 1 + int(self._BmkNV is not None)
        Portfolio = {PortfolioNV.columns[i]: iBwdData for i, iBwdData in enumerate(bwd_data_list[StartIdx:StartIdx+PortfolioNV.shape[1]])}
        if "portfolio_name_list" in self._NV._QSArgs.ModelArgs:
            PortfolioNV = PortfolioNV.reindex(columns=self._NV._QSArgs.ModelArgs["portfolio_name_list"])
        if self._QSArgs.RebalanceDTs is None:
            RebalanceIdx = None
        else:
            RebalanceDTs = sorted(PortfolioNV.index.intersection(self._QSArgs.RebalanceDTs))
            Portfolio = {iName: iPortfolio.reindex(index=RebalanceDTs) for iName, iPortfolio in Portfolio.items()}
            RebalanceIdx = pd.Series(np.arange(PortfolioNV.shape[0]), index=PortfolioNV.index)
            RebalanceIdx = sorted(RebalanceIdx[RebalanceDTs])
        Output = {"投资组合": Portfolio}
        Output["换手率"] = pd.DataFrame({iName: iPortfolio.diff().abs().sum(axis=1) for iName, iPortfolio in Portfolio.items()}, columns=PortfolioNV.columns)
        Output["净值"] = PortfolioNV
        Output["净值"]["基准"] = BmkNV / BmkNV.iloc[0]
        Output["收益率"] = Output["净值"].pct_change(fill_method=None)
        Output["超额收益率"] = Output["收益率"].iloc[:, :-1].copy()
        Output["超额净值"] = Output["超额收益率"].copy()
        for iCol in Output["超额收益率"].columns:
            Output["超额收益率"][iCol] = calcLSYield(Output["超额收益率"][iCol].values, Output["收益率"]["基准"].values, rebalance_index=RebalanceIdx)
            Output["超额净值"][iCol] = (1 + Output["超额收益率"][iCol]).cumprod()
        for iLName, iSName in self._QSArgs.LSPairs:
            Output["收益率"][f"{iLName}-{iSName}"] = calcLSYield(Output["收益率"].loc[:, iLName].values, Output["收益率"].loc[:, iSName].values, rebalance_index=RebalanceIdx)
            Output["净值"][f"{iLName}-{iSName}"] = (1 + Output["收益率"][f"{iLName}-{iSName}"]).cumprod()
        Output = self._QS_calcStats(Output)
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
