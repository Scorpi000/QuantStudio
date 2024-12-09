# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from traits.api import List, Enum, List, Int, Float
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.Tools.StrategyTestFun import calcMaxDrawdownRate, calcLSYield
from QuantStudio.BackTest.BackTestModel import BaseModule
import QuantStudio.FactorDataBase.FactorOperators as fo
from QuantStudio.BackTest.SectionFactor.BTSectionOperator import makeQuantilePortfolio, PortfolioNV, MaskPortfolio
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio import __QS_Error__


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

class MultiPortfolio(BaseModule):
    """多组合对比"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="调仓时点", order=6)
        PriceMiss = Enum("沿用前值", "填充为0", arg_type="SingleOption", label="价格缺失", order=7, option_range=["沿用前值", "填充为0"])
        LSPairs = List(arg_type="List", label="多空对", order=8)# [(i, j)]
        FeeRate = Float(value=0, arg_type="Double", label="交易费率", order=9)
        
    def __init__(self, *portfolio, price:Factor, section_ids=None, name="多组合对比", sys_args={}, **kwargs):
        self._Price = price
        self._PortfolioList = portfolio
        self._SectionIDs = section_ids
        self._PortfolioIDs = [iPortfolio.Name for iPortfolio in self._PortfolioList]
        if np.unique(self._PortfolioIDs).shape[0] < len(self._PortfolioIDs):
            raise __QS_Error__(f"传给模块 '{name}' 的投资组合有重名: {self._PortfolioIDs}")
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    
    def __QS_start__(self, mdl, dts, **kwargs):
        Tasks = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        calcPortfolioNV = PortfolioNV(sys_args={"参数": {"价格缺失": self._QSArgs.PriceMiss}})
        self._PortfolioNV = {self._PortfolioIDs[i]: calcPortfolioNV(iPortfolio, self._Price, init_nv=1, fee_rate=self._QSArgs.FeeRate, descriptor_ids=self._SectionIDs) for i, iPortfolio in enumerate(self._PortfolioList)}
        SortedPortfolioIDs = sorted(self._PortfolioIDs)
        self._PortfolioNV = fo.ConcatSection(descriptor_sections=[[iID] for iID in SortedPortfolioIDs])(*[self._PortfolioNV[iID] for iID in SortedPortfolioIDs], factor_name=self.Name, factor_args={"截面ID": SortedPortfolioIDs})
        Tasks += [(iPortfolio, self._SectionIDs) for iPortfolio in self._PortfolioList] + [(self._PortfolioNV, SortedPortfolioIDs)]
        return Tasks
    
    def __QS_end__(self, factor_data):
        super().__QS_end__(factor_data)
        self._Output["净值"] = factor_data[self._PortfolioNV._QSID]
        self._Output["净值"] = self._Output["净值"].reindex(columns=self._PortfolioIDs)
        RebalanceDTs = sorted(self._Output["净值"].index.intersection(self._QSArgs.CalcDTs))
        self._Output["投资组合"] = {self._PortfolioIDs[i]: factor_data[iPortfolio._QSID].reindex(index=RebalanceDTs) for i, iPortfolio in enumerate(self._PortfolioList)}
        self._Output["换手率"] = pd.DataFrame({iID: self._Output["投资组合"][iID].diff().abs().sum(axis=1) for i, iID in enumerate(self._PortfolioIDs)}).reindex(columns=self._PortfolioIDs)
        self._Output["收益率"] = self._Output["净值"].pct_change()
        self._Output["收益率"].iloc[0] = 0
        self._Output["统计数据"] = self._QS_calcStats()
        if not self._QSArgs.LSPairs: return 0
        if not self._QSArgs.CalcDTs:
            RebalanceIdx = None
        else:
            RebalanceIdx = pd.Series(np.arange(self._Output["净值"].shape[0]), index=self._Output["净值"].index)
            RebalanceIdx = sorted(RebalanceIdx[RebalanceIdx.index.intersection(self._QSArgs.CalcDTs)])
        self._Output["多空收益率"], self._Output["多空净值"] = pd.DataFrame(index=self._Output["收益率"].index), pd.DataFrame(index=self._Output["净值"].index)
        for iLIdx, iSIdx in self._QSArgs.LSPairs:
            iLPortfolioID, iSPortfolioID = self._PortfolioIDs[iLIdx], self._PortfolioIDs[iSIdx]
            iPortfolioID = f"{iLPortfolioID}-{iSPortfolioID}"
            self._Output["多空收益率"][iPortfolioID] = calcLSYield(self._Output["收益率"].iloc[:, iLIdx].values, self._Output["收益率"].iloc[:, iSIdx].values, rebalance_index=RebalanceIdx)
            self._Output["多空净值"][iPortfolioID] = (1 + self._Output["多空收益率"][iPortfolioID]).cumprod()
        self._Output["多空统计数据"] = self._QS_calcLSStats()
        return 0
    
    def _QS_calcStats(self):
        nDT = self._Output["净值"].shape[0]
        nDays = (self._Output["净值"].index[-1] - self._Output["净值"].index[0]).days
        nYear = nDays / 365
        TotalReturn = self._Output["净值"].iloc[-1,:] - 1
        Stats = pd.DataFrame(index=TotalReturn.index)
        Stats["总收益率"] = TotalReturn
        Stats["年化收益率"] = (1 + TotalReturn) ** (1/nYear) - 1
        Stats["年化波动率"] = self._Output["收益率"].std() * np.sqrt(nDT/nYear)
        Stats["Sharpe比率"] = Stats["年化收益率"] / Stats["年化波动率"]
        Stats["平均换手率"] = self._Output["换手率"].mean()
        Stats["最大回撤率"] = pd.Series(np.nan, index=Stats.index)
        Stats["最大回撤开始时间"] = pd.Series(index=Stats.index, dtype="O")
        Stats["最大回撤结束时间"] = pd.Series(index=Stats.index, dtype="O")
        for iCol in self._Output["净值"].columns:
            iMaxDD, iStartPos, iEndPos = calcMaxDrawdownRate(self._Output["净值"].loc[:, iCol].values)
            Stats.loc[iCol, "最大回撤率"] = abs(iMaxDD)
            Stats.loc[iCol, "最大回撤开始时间"] = (self._Output["净值"].index[iStartPos] if iStartPos is not None else None)
            Stats.loc[iCol, "最大回撤结束时间"] = (self._Output["净值"].index[iEndPos] if iEndPos is not None else None)
        return Stats
    
    def _QS_calcLSStats(self):
        nDT = self._Output["多空净值"].shape[0]
        nDays = (self._Output["多空净值"].index[-1] - self._Output["多空净值"].index[0]).days
        nYear = nDays / 365
        TotalReturn = self._Output["多空净值"].iloc[-1,:] - 1
        Stats = pd.DataFrame(index=TotalReturn.index)
        Stats["总收益率"] = TotalReturn
        Stats["年化收益率"] = (1 + TotalReturn) ** (1/nYear) - 1
        Stats["年化波动率"] = self._Output["多空收益率"].std() * np.sqrt(nDT/nYear)
        Stats["信息比率"] = Stats["年化收益率"] / Stats["年化波动率"]
        Stats["胜率"] = (self._Output["多空收益率"] > 0).sum() / self._Output["多空收益率"].notnull().sum()
        Stats["最大回撤率"] = pd.Series(np.nan, index=Stats.index)
        Stats["最大回撤开始时间"] = pd.Series(index=Stats.index, dtype="O")
        Stats["最大回撤结束时间"] = pd.Series(index=Stats.index, dtype="O")
        for iCol in self._Output["多空净值"].columns:
            iMaxDD, iStartPos, iEndPos = calcMaxDrawdownRate(self._Output["多空净值"].loc[:, iCol].values)
            Stats.loc[iCol, "最大回撤率"] = abs(iMaxDD)
            Stats.loc[iCol, "最大回撤开始时间"] = (self._Output["多空净值"].index[iStartPos] if iStartPos is not None else None)
            Stats.loc[iCol, "最大回撤结束时间"] = (self._Output["多空净值"].index[iEndPos] if iEndPos is not None else None)
        return Stats
    
    def genMatplotlibFig(self):
        PortfolioNum = self._Output["净值"].shape[1]
        nRow, nCol = 2, 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(1, self._Output["统计数据"].shape[0] + 1)
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"].index]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 1), xData, xTickLabels, self._Output["统计数据"]["年化收益率"], PercentageFormatter, self._Output["统计数据"]["平均换手率"], PercentageFormatter, True)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 2), xData, xTickLabels, self._Output["统计数据"]["Sharpe比率"], FloatFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 3), xData, xTickLabels, self._Output["统计数据"]["最大回撤率"], PercentageFormatter, None)
        Axes = Fig.add_subplot(nRow, nCol, 4)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(PortfolioNum):
            Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, i].values, label=str(self._Output["净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("组合净值")
        Axes = Fig.add_subplot(nRow, nCol, 5)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(PortfolioNum):
            iName = str(self._Output["净值"].columns[i])
            iNum = (self._Output["投资组合"][iName]>0).sum(axis=1)
            Axes.plot(iNum.index, iNum.values, label=f"{iName}: {round(iNum.mean(),2)}", lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("持仓数量")
        if not self._QSArgs.LSPairs: return (Fig,)
        # 多空组合
        nLS = self._Output["多空净值"].shape[1]
        nRow, nCol = 2, 3
        LSFig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(1, self._Output["多空统计数据"].shape[0] + 1)
        xTickLabels = [str(iInd) for iInd in self._Output["多空统计数据"].index]
        _QS_plotStatistics(LSFig.add_subplot(nRow, nCol, 1), xData, xTickLabels, self._Output["多空统计数据"]["年化收益率"], PercentageFormatter, self._Output["多空统计数据"]["胜率"], PercentageFormatter, True)
        _QS_plotStatistics(LSFig.add_subplot(nRow, nCol, 2), xData, xTickLabels, self._Output["多空统计数据"]["信息比率"], PercentageFormatter, None)
        _QS_plotStatistics(LSFig.add_subplot(nRow, nCol, 3), xData, xTickLabels, self._Output["多空统计数据"]["最大回撤率"], PercentageFormatter, None)
        Axes = LSFig.add_subplot(nRow, nCol, 4)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(nLS):
            Axes.plot(self._Output["多空净值"].index, self._Output["多空净值"].iloc[:, i].values, label=str(self._Output["多空净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("多空净值")
        return Fig, LSFig
    
    def _repr_html_(self):
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                if iArgName!="调仓时点":
                    HTML += f"<li>{iArgName}: {self.Args[iArgName]}</li>"
                elif self.Args[iArgName]:
                    HTML += f"<li>{iArgName}: 自定义时点</li>"
                else:
                    HTML += f"<li>{iArgName}: 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        Figs = self.genMatplotlibFig()
        Formatters = [_QS_formatPandasPercentage]*3 + [lambda x:'{0:.2f}'.format(x)] + [_QS_formatPandasPercentage]*2 + [lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        iHTML = self._Output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Figs[0].savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        if not self._QSArgs.LSPairs: return HTML
        Formatters = [_QS_formatPandasPercentage]*3 + [lambda x:'{0:.2f}'.format(x)] + [_QS_formatPandasPercentage]*2 + [lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        iHTML = self._Output["多空统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Figs[1].savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML


class QuantilePortfolio(BaseModule):
    """分位数组合"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        FactorOrder = Enum("降序", "升序", arg_type="SingleOption", label="排序方向", order=1)
        GroupNum = Int(10, arg_type="Integer", label="分组数", order=2)
        CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="调仓时点", order=6)
        PriceMiss = Enum("沿用前值", "填充为0", arg_type="SingleOption", label="价格缺失", order=9, option_range=["沿用前值", "填充为0"])

    def __init__(self, factor:Factor, price:Factor, mask:Optional[Factor]=None, bmk_mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, weight:Optional[Factor]=None, section_ids=None, name="分位数组合", sys_args={}, **kwargs):
        self._Price = price
        self._Factor = factor
        self._Mask = mask
        self._BmkMask = bmk_mask
        self._CatData = cat_data
        self._Weight = weight
        self._SectionIDs = section_ids
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
        
    def __QS_start__(self, mdl, dts, **kwargs):
        Tasks = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        RebalanceDTs = (list(self._QSArgs.CalcDTs) if self._QSArgs.CalcDTs else None)
        self._PortfolioList = makeQuantilePortfolio(self._Factor, mask=self._Mask, cat_data=self._CatData, weight=self._Weight, descriptor_ids=self._SectionIDs, rebalance_dts=RebalanceDTs, ascending=(self._QSArgs.FactorOrder=="升序"), group_num=self._QSArgs.GroupNum)
        self._BmkPortfolio = MaskPortfolio()(mask=(self._BmkMask if self._BmkMask else self._Mask), weight=self._Weight, descriptor_ids=self._SectionIDs)
        calcPortfolioNV = PortfolioNV(sys_args={"参数": {"价格缺失": self._QSArgs.PriceMiss}})
        self._PortfolioNV = {f"P{i}": calcPortfolioNV(iPortfolio, self._Price, descriptor_ids=self._SectionIDs) for i, iPortfolio in enumerate(self._PortfolioList)}
        self._PortfolioNV["Bmk"] = calcPortfolioNV(self._BmkPortfolio, self._Price, descriptor_ids=self._SectionIDs)
        self._PortfolioIDs = sorted(self._PortfolioNV.keys())
        self._PortfolioNV = fo.ConcatSection(descriptor_sections=[[iID] for iID in self._PortfolioIDs])(*[self._PortfolioNV[iID] for iID in self._PortfolioIDs], factor_name=self.Name, factor_args={"截面ID": self._PortfolioIDs})
        Tasks += [(iPortfolio, self._SectionIDs) for iPortfolio in self._PortfolioList] + [(self._BmkPortfolio, self._SectionIDs), (self._PortfolioNV, self._PortfolioIDs)]
        return Tasks
    
    def _QS_calcStats(self):
        nDT = self._Output["净值"].shape[0]
        nDays = (self._Output["净值"].index[-1] - self._Output["净值"].index[0]).days
        nYear = nDays / 365
        TotalReturn = self._Output["净值"].iloc[-1,:]-1
        self._Output["统计数据"] = pd.DataFrame(index=TotalReturn.index)
        self._Output["统计数据"]["总收益率"] = TotalReturn
        self._Output["统计数据"]["年化收益率"] = (1+TotalReturn)**(1/nYear)-1
        self._Output["统计数据"]["波动率"] = self._Output["收益率"].std()*np.sqrt(nDT/nYear)
        self._Output["统计数据"]["Sharpe比率"] = self._Output["统计数据"]["年化收益率"]/self._Output["统计数据"]["波动率"]
        self._Output["统计数据"]["t统计量(Sharpe比率)"] = (self._Output["统计数据"]["Sharpe比率"]-self._Output["统计数据"]["Sharpe比率"]["Bmk"])/np.sqrt(2/nYear)
        self._Output["统计数据"]["平均换手率"] = self._Output["换手率"].mean()
        self._Output["统计数据"]["最大回撤率"] = pd.Series(np.nan,index=self._Output["统计数据"].index)
        self._Output["统计数据"]["最大回撤开始时间"] = pd.Series(index=self._Output["统计数据"].index,dtype="O")
        self._Output["统计数据"]["最大回撤结束时间"] = pd.Series(index=self._Output["统计数据"].index,dtype="O")
        for iCol in self._Output["净值"].columns:
            iMaxDD,iStartPos,iEndPos = calcMaxDrawdownRate(self._Output["净值"].loc[:, iCol].values)
            self._Output["统计数据"].loc[iCol, "最大回撤率"] = abs(iMaxDD)
            self._Output["统计数据"].loc[iCol, "最大回撤开始时间"] = (self._Output["净值"].index[iStartPos] if iStartPos is not None else None)
            self._Output["统计数据"].loc[iCol, "最大回撤结束时间"] = (self._Output["净值"].index[iEndPos] if iEndPos is not None else None)
        self._Output["统计数据"]["超额收益率"] = self._Output["超额净值"].iloc[-1,:]-1
        self._Output["统计数据"]["年化超额收益率"] = (1+self._Output["统计数据"]["超额收益率"])**(1/nYear)-1
        self._Output["统计数据"]["跟踪误差"] = self._Output["超额收益率"].std()*np.sqrt(nDT/nYear)
        self._Output["统计数据"]["信息比率"] = self._Output["统计数据"]["年化超额收益率"]/self._Output["统计数据"]["跟踪误差"]
        self._Output["统计数据"]["t统计量(信息比率)"] = self._Output["统计数据"]["信息比率"]*np.sqrt(nYear)
        self._Output["统计数据"]["胜率"] = (self._Output["超额收益率"]>0).sum() / nDT
        self._Output["统计数据"]["超额最大回撤率"] = pd.Series(np.nan,index=self._Output["统计数据"].index)
        self._Output["统计数据"]["超额最大回撤开始时间"] = pd.Series(index=self._Output["统计数据"].index, dtype="O")
        self._Output["统计数据"]["超额最大回撤结束时间"] = pd.Series(index=self._Output["统计数据"].index, dtype="O")
        for iCol in self._Output["超额净值"].columns:
            iMaxDD, iStartPos, iEndPos = calcMaxDrawdownRate(self._Output["超额净值"].loc[:, iCol].values)
            self._Output["统计数据"].loc[iCol, "超额最大回撤率"] = abs(iMaxDD)
            self._Output["统计数据"].loc[iCol, "超额最大回撤开始时间"] = (self._Output["超额净值"].index[iStartPos] if iStartPos is not None else None)
            self._Output["统计数据"].loc[iCol, "超额最大回撤结束时间"] = (self._Output["超额净值"].index[iEndPos] if iEndPos is not None else None)
        self._Output["统计数据"]["CAPM Alpha"], self._Output["统计数据"]["CAPM Beta"] = 0.0, 0.0
        xData = sm.add_constant(self._Output["收益率"]["Bmk"].values, prepend=True)
        for iCol in self._Output["收益率"].columns:
            yData = self._Output["收益率"][iCol].values
            try:
                Result = sm.OLS(yData, xData, missing="drop").fit()
                self._Output["统计数据"].loc[iCol, "CAPM Beta"] = Result.params[1]
                self._Output["统计数据"].loc[iCol, "CAPM Alpha"] = Result.params[0]
            except:
                self._Output["统计数据"].loc[iCol, "CAPM Beta"] = np.nan
                self._Output["统计数据"].loc[iCol, "CAPM Alpha"] = np.nan
        return 0
    
    def __QS_end__(self, factor_data):
        super().__QS_end__(factor_data)
        GroupNum = self._QSArgs.GroupNum
        self._Output["净值"] = factor_data[self._PortfolioNV._QSID]
        PIDs = sorted(self._Output["净值"].columns, key=lambda s: np.inf if s=="Bmk" else int(s[1:]))
        self._Output["净值"] = self._Output["净值"].reindex(columns=PIDs)
        RebalanceDTs = sorted(self._Output["净值"].index.intersection(self._QSArgs.CalcDTs))
        self._Output["投资组合"] = {f"P{i}": factor_data[iPortfolio._QSID].reindex(index=RebalanceDTs) for i, iPortfolio in enumerate(self._PortfolioList)}
        self._Output["投资组合"]["Bmk"] = factor_data[self._BmkPortfolio._QSID].reindex(index=RebalanceDTs)
        self._Output["收益率"] = self._Output["净值"].pct_change()
        self._Output["收益率"].iloc[0] = 0
        if not self._QSArgs.CalcDTs:
            RebalanceIdx = None
        else:
            RebalanceIdx = pd.Series(np.arange(self._Output["净值"].shape[0]), index=self._Output["净值"].index)
            RebalanceIdx = sorted(RebalanceIdx[RebalanceIdx.index.intersection(self._QSArgs.CalcDTs)])
        self._Output["收益率"]["L-S"] = calcLSYield(self._Output["收益率"].iloc[:, 0].values, self._Output["收益率"].iloc[:, -2].values, rebalance_index=RebalanceIdx)
        self._Output["净值"]["L-S"] = (1 + self._Output["收益率"]["L-S"]).cumprod()
        self._Output["换手率"] = pd.DataFrame({iID: self._Output["投资组合"][iID].diff().abs().sum(axis=1) for i, iID in enumerate(PIDs)}).reindex(columns=PIDs)
        self._Output["超额收益率"] = self._Output["收益率"].iloc[:, :GroupNum].copy()
        self._Output["超额净值"] = self._Output["超额收益率"].iloc[:, :GroupNum].copy()
        for i in self._Output["超额收益率"]:
            self._Output["超额收益率"][i] = calcLSYield(self._Output["超额收益率"][i].values, self._Output["收益率"]["Bmk"].values, rebalance_index=RebalanceIdx)
            self._Output["超额净值"][i] = (1+self._Output["超额收益率"][i]).cumprod()
        self._QS_calcStats()
        return 0
    
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = 3, 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        GroupNum = self._Output["超额净值"].shape[1]
        xData = np.arange(1, GroupNum + 1)
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"].index[:GroupNum]]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 1), xData, xTickLabels, self._Output["统计数据"]["年化超额收益率"].iloc[:GroupNum], PercentageFormatter, self._Output["统计数据"]["胜率"].iloc[:GroupNum], PercentageFormatter)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 2), xData, xTickLabels, self._Output["统计数据"]["信息比率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 3), xData, xTickLabels, self._Output["统计数据"]["超额最大回撤率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 4), xData, xTickLabels, self._Output["统计数据"]["年化收益率"].iloc[:GroupNum], PercentageFormatter, pd.Series(self._Output["统计数据"].loc["Bmk", "年化收益率"], index=self._Output["统计数据"].index[:GroupNum], name="基准"), PercentageFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 5), xData, xTickLabels, self._Output["统计数据"]["Sharpe比率"].iloc[:GroupNum], FloatFormatter, pd.Series(self._Output["统计数据"].loc["Bmk", "Sharpe比率"], index=self._Output["统计数据"].index[:GroupNum], name="基准"), FloatFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 6), xData, xTickLabels, self._Output["统计数据"]["平均换手率"].iloc[:GroupNum], PercentageFormatter, None)
        Axes = Fig.add_subplot(nRow, nCol, 7)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum):
            Axes.plot(self._Output["超额净值"].index, self._Output["超额净值"].iloc[:, i].values, label=str(self._Output["超额净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("超额净值")
        Axes = Fig.add_subplot(nRow, nCol, 8)
        xData = np.arange(0, self._Output["净值"].shape[0])
        xTicks = np.arange(0, self._Output["净值"].shape[0], max(1, int(self._Output["净值"].shape[0]/8)))
        xTickLabels = [self._Output["净值"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        Axes.plot(xData, self._Output["净值"]["L-S"].values, label="多空净值", color="indianred", lw=2.5)
        Axes.legend(loc='upper left')
        Axes.set_title("多空组合")
        RAxes = Axes.twinx()
        RAxes.yaxis.set_major_formatter(PercentageFormatter)
        RAxes.bar(xData, self._Output["收益率"]["L-S"].values, label="多空收益率", color="steelblue")
        RAxes.legend(loc="upper right")
        Axes.set_xticks(xTicks)
        Axes.set_xticklabels(xTickLabels)
        Axes = Fig.add_subplot(nRow, nCol, 9)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum+1):
            Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, i].values, label=str(self._Output["净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("多头净值")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    
    def _repr_html_(self):
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                if iArgName!="调仓时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        Formatters = [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        Formatters += [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        Formatters += [lambda x:'{0:.2f}'.format(x)]*2
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


