# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from collections import OrderedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from traits.api import List, Enum, List, Int, Str, on_trait_change, Instance
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList, distributeEqual
from QuantStudio.Tools.StrategyTestFun import calcTurnover, calcMaxDrawdownRate, calcLSYield
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage

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

class QuantilePortfolio(BaseModule):
    """分位数组合"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        #TestFactor = Enum(None, arg_type="SingleOption", label="测试因子", order=0)
        FactorOrder = Enum("降序", "升序", arg_type="SingleOption", label="排序方向", order=1)
        GroupNum = Int(10, arg_type="Integer", label="分组数", order=2)
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
        #ClassFactor = Enum("无", arg_type="SingleOption", label="类别因子", order=4)
        #WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=5)
        CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="调仓时点", order=6)
        MarketIDFilter = Str(arg_type="IDFilter", label="市场组合", order=7)
        IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=8)
        PriceMiss = Enum("沿用前值", "填充为0", arg_type="SingleOption", label="价格缺失", order=9)
        Perturbation = Enum(False, True, arg_type="Bool", label="随机微扰", order=10)
        def __QS_initArgs__(self):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("TestFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="测试因子", order=0, option_range=DefaultNumFactorList))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3, option_range=DefaultNumFactorList))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            self.add_trait("ClassFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="类别因子", order=4, option_range=["无"]+DefaultStrFactorList))
            self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=5, option_range=["等权"]+DefaultNumFactorList))
    
    def __init__(self, factor_table, name="分位数组合", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {"净值":[[1] for i in range(self._QSArgs.GroupNum)]}
        self._Output["投资组合"] = [[] for i in range(self._QSArgs.GroupNum)]
        self._Output["换手率"] = [[] for i in range(self._QSArgs.GroupNum)]
        self._Output["市场净值"] = [1]
        self._Output["调仓日"] = []
        self._Output["QP_P_CurPos"] = [pd.Series() for i in range(self._QSArgs.GroupNum)]
        self._Output["QP_P_MarketPos"] = pd.Series()
        self._Output["QP_LastPrice"] = np.nan
        self._Output["GroupNum"] = self._QSArgs.GroupNum
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        # 更新投资组合净值
        Price = self._FactorTable.readData(dts=[idt], ids=self._FactorTable.getID(ifactor_name=self._QSArgs.PriceFactor), factor_names=[self._QSArgs.PriceFactor]).iloc[0, 0, :]
        self._Output["QP_LastPrice"] = Price.where(pd.notnull(Price), self._Output["QP_LastPrice"])
        for i in range(self._QSArgs.GroupNum):
            if len(self._Output["QP_P_CurPos"][i])==0:
                iWealth = self._Output["净值"][i][-1]
            elif self._QSArgs.PriceMiss=="沿用前值":
                iWealth = (self._Output["QP_P_CurPos"][i] * self._Output["QP_LastPrice"]).sum()
            else:
                iWealth = (self._Output["QP_P_CurPos"][i] * Price).sum()
            self._Output["净值"][i].append(iWealth)
            self._Output["换手率"][i].append(0)
        if len(self._Output["QP_P_MarketPos"])==0:
            self._Output["市场净值"].append(self._Output["市场净值"][-1])
        elif self._QSArgs.PriceMiss=="沿用前值":
            self._Output["市场净值"].append((self._Output["QP_P_MarketPos"] * self._Output["QP_LastPrice"]).sum())
        else:
            self._Output["市场净值"].append((self._Output["QP_P_MarketPos"] * Price).sum())
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        # 生成本期的分位数投资组合
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self._QSArgs.IDFilter)
        FactorData = self._FactorTable.readData(dts=[idt], ids=IDs, factor_names=[self._QSArgs.TestFactor]).iloc[0, 0, :]
        FactorData = FactorData[pd.notnull(FactorData) & pd.notnull(Price[IDs])].copy()
        nID = FactorData.shape[0]
        if self._QSArgs.Perturbation and (nID>0):
            FactorData = FactorData.astype("float")
            MinDiff = np.min(np.abs(np.diff(FactorData.unique())))
            FactorData += np.random.rand(nID) * MinDiff * 0.01
        FactorData = FactorData.sort_values(ascending=(self._QSArgs.FactorOrder=="升序"), inplace=False)
        IDs = FactorData.index.to_list()
        if self._QSArgs.WeightFactor!="等权":
            WeightData = self._FactorTable.readData(dts=[idt], ids=self._FactorTable.getID(ifactor_name=self._QSArgs.WeightFactor), factor_names=[self._QSArgs.WeightFactor]).iloc[0, 0, :]
        else:
            WeightData = pd.Series(1.0, index=self._FactorTable.getID())
        if self._QSArgs.ClassFactor=="无":
            nSubID = distributeEqual(nID, self._QSArgs.GroupNum, remainder_pos="middle")
            for i in range(self._QSArgs.GroupNum):
                iWealth = self._Output["净值"][i][-1]
                iSubIDs = IDs[sum(nSubID[:i]):sum(nSubID[:i+1])]
                iPortfolio = WeightData[iSubIDs]
                iPortfolio = iPortfolio/iPortfolio.sum()
                iPortfolio = iPortfolio[pd.notnull(iPortfolio)]
                self._Output["投资组合"][i].append(iPortfolio)
                self._Output["QP_P_CurPos"][i] = iPortfolio*iWealth/Price
                self._Output["QP_P_CurPos"][i] = self._Output["QP_P_CurPos"][i][pd.notnull(self._Output["QP_P_CurPos"][i])]
                nPortfolio = len(self._Output["投资组合"][i])
                if nPortfolio>1:
                    self._Output["换手率"][i][-1] = calcTurnover(self._Output["投资组合"][i][-2],self._Output["投资组合"][i][-1])
                elif nPortfolio==1:
                    self._Output["换手率"][i][-1] = 1
        else:
            Portfolio = [{} for i in range(self._QSArgs.GroupNum)]
            IndustryData = self._FactorTable.readData(dts=[idt], ids=IDs, factor_names=[self._QSArgs.ClassFactor]).iloc[0, 0, :]
            AllIndustry = IndustryData.unique()
            for iIndustry in AllIndustry:
                iMask = (IndustryData==iIndustry)
                iIDIndex = IndustryData[iMask].index
                iIDDistribution = distributeEqual(iIDIndex.shape[0], self._QSArgs.GroupNum, remainder_pos="middle")
                for j in range(self._QSArgs.GroupNum):
                    jSubID = iIDIndex[sum(iIDDistribution[:j]):sum(iIDDistribution[:j+1])]
                    Portfolio[j].update({kID:WeightData[kID] for kID in jSubID})
            for i in range(self._QSArgs.GroupNum):
                iWealth = self._Output["净值"][i][-1]
                iPortfolio = pd.Series(Portfolio[i])
                iPortfolio = iPortfolio/iPortfolio.sum()
                iPortfolio = iPortfolio[pd.notnull(iPortfolio)]
                self._Output["投资组合"][i].append(iPortfolio)
                self._Output["QP_P_CurPos"][i] = iPortfolio*iWealth/Price
                self._Output["QP_P_CurPos"][i] = self._Output["QP_P_CurPos"][i][pd.notnull(self._Output["QP_P_CurPos"][i])]
                nPortfolio = len(self._Output["投资组合"][i])
                if nPortfolio>1:
                    self._Output["换手率"][i][-1] = calcTurnover(self._Output["投资组合"][i][-2],self._Output["投资组合"][i][-1])
                elif nPortfolio==1:
                    self._Output["换手率"][i][-1] = 1
        # 生成本期的市场投资组合
        if self._QSArgs.MarketIDFilter:
            IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self._QSArgs.MarketIDFilter)
        WeightData = WeightData[IDs]
        Price = Price[IDs]
        WeightData = WeightData[pd.notnull(WeightData) & pd.notnull(Price)]
        WeightData = WeightData / WeightData.sum()
        self._Output["QP_P_MarketPos"] = WeightData*self._Output["市场净值"][-1]/Price
        self._Output["QP_P_MarketPos"] = self._Output["QP_P_MarketPos"][pd.notnull(self._Output["QP_P_MarketPos"])]
        self._Output["调仓日"].append(idt)
        return 0
    def _QS_calcStats(self):
        nDT = len(self._Model.DateTimeSeries)
        nDays = (self._Model.DateTimeSeries[-1] - self._Model.DateTimeSeries[0]).days
        nYear = nDays / 365
        TotalReturn = self._Output["净值"].iloc[-1,:]-1
        self._Output["统计数据"] = pd.DataFrame(index=TotalReturn.index)
        self._Output["统计数据"]["总收益率"] = TotalReturn
        self._Output["统计数据"]["年化收益率"] = (1+TotalReturn)**(1/nYear)-1
        self._Output["统计数据"]["波动率"] = self._Output["收益率"].std()*np.sqrt(nDT/nYear)
        self._Output["统计数据"]["Sharpe比率"] = self._Output["统计数据"]["年化收益率"]/self._Output["统计数据"]["波动率"]
        self._Output["统计数据"]["t统计量(Sharpe比率)"] = (self._Output["统计数据"]["Sharpe比率"]-self._Output["统计数据"]["Sharpe比率"]["市场"])/np.sqrt(2/nYear)
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
        xData = sm.add_constant(self._Output["收益率"]["市场"].values, prepend=True)
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
    
    def __QS_end__(self):
        if not self._isStarted: return 0
        BaseModule.__QS_end__(self)
        self._Output.pop("QP_P_CurPos")
        self._Output.pop("QP_P_MarketPos")
        self._Output.pop("QP_LastPrice")
        GroupNum = self._Output.pop("GroupNum")
        for i in range(GroupNum):
            self._Output["净值"][i].pop(0)
        nDT = len(self._Model.DateTimeSeries)
        self._Output["净值"] = pd.DataFrame(np.array(self._Output["净值"]).T, index=self._Model.DateTimeSeries)
        self._Output["净值"]["市场"] = pd.Series(self._Output.pop("市场净值")[1:], index=self._Model.DateTimeSeries)
        self._Output["收益率"] = self._Output["净值"].iloc[1:,:].values / self._Output["净值"].iloc[:-1,:].values - 1
        self._Output["收益率"] = pd.DataFrame(np.row_stack((np.zeros((1, GroupNum+1)), self._Output["收益率"])), index=self._Model.DateTimeSeries, columns=[i for i in range(GroupNum)]+["市场"])
        if not self._QSArgs.CalcDTs:
            RebalanceIdx = None
        else:
            RebalanceIdx = pd.Series(np.arange(nDT), index=self._Model.DateTimeSeries)
            RebalanceIdx = sorted(RebalanceIdx[RebalanceIdx.index.intersection(self._QSArgs.CalcDTs)])
        self._Output["收益率"]["L-S"] = calcLSYield(self._Output["收益率"].iloc[:, 0].values, self._Output["收益率"].iloc[:, -2].values, rebalance_index=RebalanceIdx)
        self._Output["净值"]["L-S"] = (1 + self._Output["收益率"]["L-S"]).cumprod()
        self._Output["换手率"] = pd.DataFrame(np.array(self._Output["换手率"]).T, index=self._Model.DateTimeSeries)
        self._Output["投资组合"] = {str(i): pd.DataFrame(self._Output["投资组合"][i], index=self._Output["调仓日"]) for i in range(GroupNum)}
        self._Output["超额收益率"] = self._Output["收益率"].iloc[:, :GroupNum].copy()
        self._Output["超额净值"] = self._Output["超额收益率"].iloc[:, :GroupNum].copy()
        for i in self._Output["超额收益率"]:
            self._Output["超额收益率"][i] = calcLSYield(self._Output["超额收益率"][i].values, self._Output["收益率"]["市场"].values, rebalance_index=RebalanceIdx)
            self._Output["超额净值"][i] = (1+self._Output["超额收益率"][i]).cumprod()
        self._QS_calcStats()
        self._Output.pop("调仓日")
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
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 4), xData, xTickLabels, self._Output["统计数据"]["年化收益率"].iloc[:GroupNum], PercentageFormatter, pd.Series(self._Output["统计数据"].loc["市场", "年化收益率"], index=self._Output["统计数据"].index[:GroupNum], name="市场"), PercentageFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 5), xData, xTickLabels, self._Output["统计数据"]["Sharpe比率"].iloc[:GroupNum], FloatFormatter, pd.Series(self._Output["统计数据"].loc["市场", "Sharpe比率"], index=self._Output["统计数据"].index[:GroupNum], name="市场"), FloatFormatter, False)
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

class FilterPortfolio(BaseModule):
    """条件筛选组合"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        PortfolioFilters = Instance(OrderedDict, arg_type="Dict", label="组合条件", order=0)
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
        #ClassFactor = Enum("无", arg_type="SingleOption", label="类别因子", order=2)
        #WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=3)
        CalcDTs = List(dt.datetime, arg_type="DateTimeList", label="调仓时点", order=4)
        MarketIDFilter = Str(arg_type="IDFilter", label="市场组合", order=5)
        IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=6)
        PriceMiss = Enum("沿用前值", "填充为0", arg_type="SingleOption", label="价格缺失", order=7)
        LSPairs = List(arg_type="List", label="多空对", order=8)
        def __QS_initArgs__(self):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1, option_range=DefaultNumFactorList))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            self.add_trait("ClassFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="类别因子", order=2, option_range=["无"]+DefaultStrFactorList))
            self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=3, option_range=["等权"]+DefaultNumFactorList))
    
    def __init__(self, factor_table, name="条件筛选组合", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        GroupNum = len(self._QSArgs.PortfolioFilters)
        self._Output = {"净值":[[1] for i in range(GroupNum)]}
        self._Output["投资组合"] = [[] for i in range(GroupNum)]
        self._Output["换手率"] = [[] for i in range(GroupNum)]
        self._Output["市场净值"] = [1]
        self._Output["调仓日"] = []
        self._Output["QP_P_CurPos"] = [pd.Series() for i in range(GroupNum)]
        self._Output["QP_P_MarketPos"] = pd.Series()
        self._Output["QP_LastPrice"] = np.nan
        self._Output["GroupNum"] = GroupNum
        self._CurCalcInd = 0
        return (self._FactorTable, )
    
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        Price = self._FactorTable.readData(dts=[idt], ids=self._FactorTable.getID(ifactor_name=self._QSArgs.PriceFactor), factor_names=[self._QSArgs.PriceFactor]).iloc[0, 0, :]
        self._Output["QP_LastPrice"] = Price.where(pd.notnull(Price), self._Output["QP_LastPrice"])
        GroupNum = self._Output["GroupNum"]
        for i in range(GroupNum):
            if len(self._Output["QP_P_CurPos"][i])==0:
                iWealth = self._Output["净值"][i][-1]
            elif self._QSArgs.PriceMiss=="沿用前值":
                iWealth = (self._Output["QP_P_CurPos"][i] * self._Output["QP_LastPrice"]).sum()
            else:
                iWealth = (self._Output["QP_P_CurPos"][i] * Price).sum()
            self._Output["净值"][i].append(iWealth)
            self._Output["换手率"][i].append(0)
        if len(self._Output["QP_P_MarketPos"])==0:
            self._Output["市场净值"].append(self._Output["市场净值"][-1])
        elif self._QSArgs.PriceMiss=="沿用前值":
            self._Output["市场净值"].append((self._Output["QP_P_MarketPos"] * self._Output["QP_LastPrice"]).sum())
        else:
            self._Output["市场净值"].append((self._Output["QP_P_MarketPos"] * Price).sum())
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self._QSArgs.IDFilter)
        if self._QSArgs.WeightFactor!="等权":
            WeightData = self._FactorTable.readData(dts=[idt], ids=self._FactorTable.getID(ifactor_name=self._QSArgs.WeightFactor), factor_names=[self._QSArgs.WeightFactor]).iloc[0, 0, :]
        else:
            WeightData = pd.Series(1.0, index=self._FactorTable.getID())
        PortfolioFilters = list(self._QSArgs.PortfolioFilters.values())
        if self._QSArgs.ClassFactor=="无":
            for i in range(GroupNum):
                iWealth = self._Output["净值"][i][-1]
                iSubIDs = self._FactorTable.getFilteredID(idt, ids=IDs, id_filter_str=PortfolioFilters[i], args={})
                iPortfolio = WeightData[iSubIDs]
                iPortfolio = iPortfolio / iPortfolio.sum()
                iPortfolio = iPortfolio[pd.notnull(iPortfolio)]
                self._Output["投资组合"][i].append(iPortfolio)
                self._Output["QP_P_CurPos"][i] = iPortfolio * iWealth / Price
                self._Output["QP_P_CurPos"][i] = self._Output["QP_P_CurPos"][i][pd.notnull(self._Output["QP_P_CurPos"][i])]
                nPortfolio = len(self._Output["投资组合"][i])
                if nPortfolio>1:
                    self._Output["换手率"][i][-1] = calcTurnover(self._Output["投资组合"][i][-2], self._Output["投资组合"][i][-1])
                elif nPortfolio==1:
                    self._Output["换手率"][i][-1] = 1
        else:
            Portfolio = [{} for i in range(GroupNum)]
            IndustryData = self._FactorTable.readData(dts=[idt], ids=IDs, factor_names=[self._QSArgs.ClassFactor]).iloc[0, 0, :]
            AllIndustry = IndustryData.unique()
            for iIndustry in AllIndustry:
                iMask = (IndustryData==iIndustry)
                iIDs = IndustryData[iMask].index.tolist()
                for j in range(GroupNum):
                    ijSubIDs = self._FactorTable.getFilteredID(idt, ids=iIDs, id_filter_str=PortfolioFilters[j], args={})
                    Portfolio[j].update(WeightData.reindex(index=ijSubIDs).to_dict())
            for i in range(GroupNum):
                iWealth = self._Output["净值"][i][-1]
                iPortfolio = pd.Series(Portfolio[i])
                iPortfolio = iPortfolio / iPortfolio.sum()
                iPortfolio = iPortfolio[pd.notnull(iPortfolio)]
                self._Output["投资组合"][i].append(iPortfolio)
                self._Output["QP_P_CurPos"][i] = iPortfolio * iWealth / Price
                self._Output["QP_P_CurPos"][i] = self._Output["QP_P_CurPos"][i][pd.notnull(self._Output["QP_P_CurPos"][i])]
                nPortfolio = len(self._Output["投资组合"][i])
                if nPortfolio>1:
                    self._Output["换手率"][i][-1] = calcTurnover(self._Output["投资组合"][i][-2], self._Output["投资组合"][i][-1])
                elif nPortfolio==1:
                    self._Output["换手率"][i][-1] = 1
        if self._QSArgs.MarketIDFilter:
            IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self._QSArgs.MarketIDFilter)
        WeightData = WeightData[IDs]
        Price = Price[IDs]
        WeightData = WeightData[pd.notnull(WeightData) & pd.notnull(Price)]
        WeightData = WeightData / WeightData.sum()
        self._Output["QP_P_MarketPos"] = WeightData * self._Output["市场净值"][-1] / Price
        self._Output["QP_P_MarketPos"] = self._Output["QP_P_MarketPos"][pd.notnull(self._Output["QP_P_MarketPos"])]
        self._Output["调仓日"].append(idt)
        return 0
    
    def __QS_end__(self):
        if not self._isStarted: return 0
        BaseModule.__QS_end__(self)
        self._Output.pop("QP_P_CurPos")
        self._Output.pop("QP_P_MarketPos")
        self._Output.pop("QP_LastPrice")
        PortfolioNames = list(self._QSArgs.PortfolioFilters)
        GroupNum = self._Output.pop("GroupNum")
        for i in range(GroupNum):
            self._Output["净值"][i].pop(0)
        nDT = len(self._Model.DateTimeSeries)
        self._Output["净值"] = pd.DataFrame(np.array(self._Output["净值"]).T, index=self._Model.DateTimeSeries, columns=PortfolioNames)
        self._Output["净值"]["市场"] = pd.Series(self._Output.pop("市场净值")[1:], index=self._Model.DateTimeSeries)
        self._Output["收益率"] = self._Output["净值"].iloc[1:,:].values / self._Output["净值"].iloc[:-1,:].values - 1
        self._Output["收益率"] = pd.DataFrame(np.row_stack((np.zeros((1, GroupNum+1)), self._Output["收益率"])), index=self._Model.DateTimeSeries, columns=PortfolioNames+["市场"])
        if not self._QSArgs.CalcDTs:
            RebalanceIdx = None
        else:
            RebalanceIdx = pd.Series(np.arange(nDT), index=self._Model.DateTimeSeries)
            RebalanceIdx = sorted(RebalanceIdx[RebalanceIdx.index.intersection(self._QSArgs.CalcDTs)])
        self._Output["换手率"] = pd.DataFrame(np.array(self._Output["换手率"]).T, index=self._Model.DateTimeSeries, columns=PortfolioNames)
        self._Output["投资组合"] = {str(i): pd.DataFrame(self._Output["投资组合"][i], index=self._Output["调仓日"]) for i in range(GroupNum)}
        self._Output["超额收益率"] = self._Output["收益率"].iloc[:, :GroupNum].copy()
        self._Output["超额净值"] = self._Output["超额收益率"].copy()
        for i in self._Output["超额收益率"]:
            self._Output["超额收益率"][i] = calcLSYield(self._Output["超额收益率"][i].values, self._Output["收益率"]["市场"].values, rebalance_index=RebalanceIdx)
            self._Output["超额净值"][i] = (1+self._Output["超额收益率"][i]).cumprod()
        for iLPortfolio, iSPortfolio in self._QSArgs.LSPairs:
            iLIdx, iSIdx = PortfolioNames.index(iLPortfolio), PortfolioNames.index(iSPortfolio)
            self._Output["收益率"][f"{iLPortfolio}-{iSPortfolio}"] = calcLSYield(self._Output["收益率"].iloc[:, iLIdx].values, self._Output["收益率"].iloc[:, iSIdx].values, rebalance_index=RebalanceIdx)
            self._Output["净值"][f"{iLPortfolio}-{iSPortfolio}"] = (1 + self._Output["收益率"][f"{iLPortfolio}-{iSPortfolio}"]).cumprod()
        QuantilePortfolio._QS_calcStats(self)
        self._Output.pop("调仓日")
        return 0
    
    def genMatplotlibFig(self, file_path=None):
        GroupNum = self._Output["超额净值"].shape[1]
        nLS = self._Output["净值"].shape[1] - 1 - GroupNum
        nRow, nCol = 3 + int(0 if nLS <= 0 else (nLS - 1) // 3 + 1), 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(1, GroupNum + 1)
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"].index[:GroupNum]]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 1), xData, xTickLabels, self._Output["统计数据"]["年化超额收益率"].iloc[:GroupNum], PercentageFormatter, self._Output["统计数据"]["胜率"].iloc[:GroupNum], PercentageFormatter)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 2), xData, xTickLabels, self._Output["统计数据"]["信息比率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 3), xData, xTickLabels, self._Output["统计数据"]["超额最大回撤率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 4), xData, xTickLabels, self._Output["统计数据"]["年化收益率"].iloc[:GroupNum], PercentageFormatter, pd.Series(self._Output["统计数据"].loc["市场", "年化收益率"], index=self._Output["统计数据"].index[:GroupNum], name="市场"), PercentageFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 5), xData, xTickLabels, self._Output["统计数据"]["Sharpe比率"].iloc[:GroupNum], FloatFormatter, pd.Series(self._Output["统计数据"].loc["市场", "Sharpe比率"], index=self._Output["统计数据"].index[:GroupNum], name="市场"), FloatFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 6), xData, xTickLabels, self._Output["统计数据"]["平均换手率"].iloc[:GroupNum], PercentageFormatter, None)
        Axes = Fig.add_subplot(nRow, nCol, 7)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum):
            Axes.plot(self._Output["超额净值"].index, self._Output["超额净值"].iloc[:, i].values, label=str(self._Output["超额净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("超额净值")
        Axes = Fig.add_subplot(nRow, nCol, 8)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum+1):
            Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, i].values, label=str(self._Output["净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("多头净值")
        Axes = Fig.add_subplot(nRow, nCol, 9)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum+1):
            iName = str(self._Output["净值"].columns[i])
            iNum = (self._Output["投资组合"][iName]>0).sum(axis=1)
            Axes.plot(iNum.index, iNum.values, label=f"{iName}: {round(iNum.mean(),2)}", lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("持仓数量")
        for i in range(nLS):
            Axes = Fig.add_subplot(nRow, nCol, 10+i)
            xData = np.arange(0, self._Output["净值"].shape[0])
            xTicks = np.arange(0, self._Output["净值"].shape[0], max(1, int(self._Output["净值"].shape[0]/8)))
            xTickLabels = [self._Output["净值"].index[i].strftime("%Y-%m-%d") for i in xTicks]
            iLSName = self._Output["净值"].columns[GroupNum+1+i]
            Axes.plot(xData, self._Output["净值"][iLSName].values, label="多空净值", color="indianred", lw=2.5)
            Axes.legend(loc='upper left')
            RAxes = Axes.twinx()
            RAxes.yaxis.set_major_formatter(PercentageFormatter)
            RAxes.bar(xData, self._Output["收益率"][iLSName].values, label="多空收益率", color="steelblue")
            RAxes.legend(loc="upper right")
            Axes.set_xticks(xTicks)
            Axes.set_xticklabels(xTickLabels)
            Axes.set_title(iLSName)
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
    
class MultiPortfolio(BaseModule):
    """多组合对比"""
    def __init__(self, name="多组合对比", sys_args={}, **kwargs):
        super().__init__(name=name, sys_args=sys_args, **kwargs)
        self._QS_isMulti = True
        self.Modules = []
    def output(self, recalculate=False):
        if (not recalculate)  and self._Output: return self._Output
        self._Output = {
            "净值": {"L-S": pd.DataFrame(), "Top": pd.DataFrame(), "Bottom": pd.DataFrame(), "市场": pd.DataFrame()},
            "收益率": {"L-S": pd.DataFrame(), "Top": pd.DataFrame(), "Bottom": pd.DataFrame(), "市场": pd.DataFrame()},
            "超额净值": {"Top": pd.DataFrame(), "Bottom": pd.DataFrame()},
            "超额收益率": {"Top": pd.DataFrame(), "Bottom": pd.DataFrame()},
            "换手率": {"Top": pd.DataFrame(), "Bottom": pd.DataFrame()},
            "统计数据": {"Top": None, "Bottom": None, "L-S": None, "市场": None}
        }
        for i, iModule in enumerate(self.Modules):
            iOutput = iModule.output(recalculate=recalculate)
            iName = str(i)+"-"+iModule.Name
            self._Output[iName] = iOutput
            self._Output["净值"]["Top"][iName] = iOutput["净值"].iloc[:, 0]
            self._Output["净值"]["Bottom"][iName] = iOutput["净值"].iloc[:, -3]
            self._Output["净值"]["市场"][iName] = iOutput["净值"]["市场"]
            self._Output["净值"]["L-S"][iName] = iOutput["净值"]["L-S"]
            self._Output["收益率"]["Top"][iName] = iOutput["收益率"].iloc[:, 0]
            self._Output["收益率"]["Bottom"][iName] = iOutput["收益率"].iloc[:, -3]
            self._Output["收益率"]["市场"][iName] = iOutput["收益率"]["市场"]
            self._Output["收益率"]["L-S"][iName] = iOutput["收益率"]["L-S"]
            self._Output["超额净值"]["Top"][iName] = iOutput["超额净值"].iloc[:, 0]
            self._Output["超额净值"]["Bottom"][iName] = iOutput["超额净值"].iloc[:, -3]
            self._Output["超额收益率"]["Top"][iName] = iOutput["超额收益率"].iloc[:, 0]
            self._Output["超额收益率"]["Bottom"][iName] = iOutput["超额收益率"].iloc[:, -3]
            self._Output["换手率"]["Top"][iName] = iOutput["换手率"].iloc[:, 0]
            self._Output["换手率"]["Bottom"][iName] = iOutput["换手率"].iloc[:, -1]
            if self._Output["统计数据"]["Top"] is None:
                self._Output["统计数据"]["Top"] = pd.DataFrame(columns=iOutput["统计数据"].columns)
                self._Output["统计数据"]["Bottom"] = pd.DataFrame(columns=iOutput["统计数据"].columns)
                self._Output["统计数据"]["L-S"] = pd.DataFrame(columns=iOutput["统计数据"].columns)
                self._Output["统计数据"]["市场"] = pd.DataFrame(columns=iOutput["统计数据"].columns)
            self._Output["统计数据"]["Top"].loc[iName] = iOutput["统计数据"].iloc[0, :]
            self._Output["统计数据"]["Bottom"].loc[iName] = iOutput["统计数据"].iloc[-3, :]
            self._Output["统计数据"]["L-S"].loc[iName] = iOutput["统计数据"].loc["L-S", :]
            self._Output["统计数据"]["市场"].loc[iName] = iOutput["统计数据"].loc["市场", :]
        return self._Output
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = 8, 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(1, self._Output["统计数据"]["Top"].shape[0]+1)
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"]["Top"].index]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 1), xData, xTickLabels, self._Output["统计数据"]["Top"]["年化超额收益率"], PercentageFormatter, self._Output["统计数据"]["Top"]["胜率"], PercentageFormatter, True, title="Top 组合年化超额收益率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 2), xData, xTickLabels, self._Output["统计数据"]["Top"]["信息比率"], PercentageFormatter, None, title="Top 组合信息比率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 3), xData, xTickLabels, self._Output["统计数据"]["Top"]["超额最大回撤率"], PercentageFormatter, None, title="Top 组合超额最大回撤率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 4), xData, xTickLabels, self._Output["统计数据"]["Top"]["年化收益率"], PercentageFormatter, None, title="Top 组合年化收益率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 5), xData, xTickLabels, self._Output["统计数据"]["Top"]["Sharpe比率"], FloatFormatter, None, title="Top 组合 Sharpe 比率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 6), xData, xTickLabels, self._Output["统计数据"]["Top"]["平均换手率"], PercentageFormatter, None, title="Top 组合平均换手率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 7), xData, xTickLabels, self._Output["统计数据"]["Bottom"]["年化超额收益率"], PercentageFormatter, self._Output["统计数据"]["Bottom"]["胜率"], PercentageFormatter, True, title="Bottom 组合年化超额收益率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 8), xData, xTickLabels, self._Output["统计数据"]["Bottom"]["信息比率"], PercentageFormatter, None, title="Bottom 组合信息比率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 9), xData, xTickLabels, self._Output["统计数据"]["Bottom"]["超额最大回撤率"], PercentageFormatter, None, title="Bottom 组合超额最大回撤率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 10), xData, xTickLabels, self._Output["统计数据"]["Bottom"]["年化收益率"], PercentageFormatter, None, title="Bottom 组合年化收益率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 11), xData, xTickLabels, self._Output["统计数据"]["Bottom"]["Sharpe比率"], FloatFormatter, None, title="Bottom 组合 Sharpe 比率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 12), xData, xTickLabels, self._Output["统计数据"]["Bottom"]["平均换手率"], PercentageFormatter, None, title="Bottom 组合平均换手率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 13), xData, xTickLabels, self._Output["统计数据"]["L-S"]["年化收益率"], PercentageFormatter, None, title="L-S 组合年化收益率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 14), xData, xTickLabels, self._Output["统计数据"]["L-S"]["Sharpe比率"], FloatFormatter, None, title="L-S 组合 Sharpe 比率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 15), xData, xTickLabels, self._Output["统计数据"]["L-S"]["最大回撤率"], PercentageFormatter, None, title="L-S 组合最大回撤Æ率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 16), xData, xTickLabels, self._Output["统计数据"]["市场"]["年化收益率"], PercentageFormatter, None, title="市场组合年化收益率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 17), xData, xTickLabels, self._Output["统计数据"]["市场"]["Sharpe比率"], FloatFormatter, None, title="市场组合 Sharpe 比率")
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 18), xData, xTickLabels, self._Output["统计数据"]["市场"]["最大回撤率"], PercentageFormatter, None, title="市场组合最大回撤率")
        Axes = Fig.add_subplot(nRow, nCol, 19)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(self._Output["净值"]["Top"].shape[1]):
            Axes.plot(self._Output["净值"]["Top"].index, self._Output["净值"]["Top"].iloc[:, i].values, label=str(self._Output["净值"]["Top"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("Top 组合净值")
        Axes = Fig.add_subplot(nRow, nCol, 20)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(self._Output["净值"]["Bottom"].shape[1]):
            Axes.plot(self._Output["净值"]["Bottom"].index, self._Output["净值"]["Bottom"].iloc[:, i].values, label=str(self._Output["净值"]["Bottom"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("Bottom 组合净值")
        Axes = Fig.add_subplot(nRow, nCol, 21)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(self._Output["净值"]["市场"].shape[1]):
            Axes.plot(self._Output["净值"]["市场"].index, self._Output["净值"]["市场"].iloc[:, i].values, label=str(self._Output["净值"]["市场"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("市场组合净值")
        Axes = Fig.add_subplot(nRow, nCol, 22)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(self._Output["超额净值"]["Top"].shape[1]):
            Axes.plot(self._Output["超额净值"]["Top"].index, self._Output["超额净值"]["Top"].iloc[:, i].values, label=str(self._Output["超额净值"]["Top"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("Top 组合超额净值")
        Axes = Fig.add_subplot(nRow, nCol, 23)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(self._Output["超额净值"]["Bottom"].shape[1]):
            Axes.plot(self._Output["超额净值"]["Bottom"].index, self._Output["超额净值"]["Bottom"].iloc[:, i].values, label=str(self._Output["超额净值"]["Bottom"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("Bottom 组合超额净值")
        Axes = Fig.add_subplot(nRow, nCol, 24)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(self._Output["净值"]["L-S"].shape[1]):
            Axes.plot(self._Output["净值"]["L-S"].index, self._Output["净值"]["L-S"].iloc[:, i].values, label=str(self._Output["净值"]["L-S"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("L-S 组合净值")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                if iArgName=="对比模块":
                    HTML += "<li>"+iArgName+": "+",".join([iModule.Name for iModule in self.Modules])+"</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        Formatters = [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        Formatters += [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        Formatters += [lambda x:'{0:.2f}'.format(x)]*2
        Output = self.output()
        for iKey in Output["统计数据"]:
            HTML += iKey+" 组合: "
            iHTML = Output["统计数据"][iKey].to_html(formatters=Formatters)
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