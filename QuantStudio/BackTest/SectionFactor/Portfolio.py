# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, Int, Str, Instance, Dict, Bool
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList, distributeEqual
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.Tools.StrategyTestFun import calcPortfolioReturn, calcTurnover, calcMaxDrawdownRate
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.PortfolioConstructor import BasePC
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage

def _QS_plotStatistics(axes, x_data, x_ticklabels, left_data, left_formatter, right_data=None, right_formatter=None, right_axes=True):
    axes.yaxis.set_major_formatter(left_formatter)
    axes.bar(x_data, left_data.values, label=left_data.name, color="b")
    if right_data is not None:
        if right_axes:
            axes.legend(loc='upper left')
            right_axes = axes.twinx()
            right_axes.yaxis.set_major_formatter(right_formatter)
            right_axes.plot(x_data, right_data.values, label=right_data.name, color="r", alpha=0.6, lw=3)
            right_axes.legend(loc="upper right")
        else:
            axes.plot(x_data, right_data.values, label=right_data.name, color="r", alpha=0.6, lw=3)
            axes.legend(loc='best')
    else:
        axes.legend(loc='best')
    axes.set_xticks(x_data)
    axes.set_xticklabels(x_ticklabels)
    return axes

class QuantilePortfolio(BaseModule):
    """分位数组合"""
    #TestFactor = Enum(None, arg_type="SingleOption", label="测试因子", order=0)
    FactorOrder = Enum("降序", "升序", arg_type="SingleOption", label="排序方向", order=1)
    GroupNum = Int(10, arg_type="Integer", label="分组数", order=2)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    #IndustryFactor = Enum("无", arg_type="SingleOption", label="行业因子", order=4)
    #WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=5)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="调仓时点", order=6)
    MarketIDFilter = Str(arg_type="IDFilter", label="市场组合", order=7)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=8)
    Perturbation = Bool(False, arg_type="Bool", label="随机微扰", order=9)
    def __init__(self, factor_table, name="分位数组合", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="测试因子", order=0))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=4))
        self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=5))
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {"净值":[[1] for i in range(self.GroupNum)]}
        self._Output["投资组合"] = [[] for i in range(self.GroupNum)]
        self._Output["换手率"] = [[] for i in range(self.GroupNum)]
        self._Output["市场净值"] = [1]
        self._Output["调仓日"] = []
        self._Output["QP_P_CurPos"] = [pd.Series() for i in range(self.GroupNum)]
        self._Output["QP_P_MarketPos"] = pd.Series()
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        Price = self._FactorTable.readData(dts=[idt], ids=self._FactorTable.getID(ifactor_name=self.PriceFactor), factor_names=[self.PriceFactor]).iloc[0, 0, :]
        for i in range(self.GroupNum):
            if len(self._Output["QP_P_CurPos"][i])==0:
                self._Output["净值"][i].append(self._Output["净值"][i][-1])
            else:
                iWealth = (self._Output["QP_P_CurPos"][i]*Price).sum()
                if pd.notnull(iWealth):
                    self._Output["净值"][i].append(iWealth)
                else:
                    self._Output["净值"][i].append(0)
            self._Output["换手率"][i].append(0)
        if len(self._Output["QP_P_MarketPos"])==0:
            self._Output["市场净值"].append(self._Output["市场净值"][-1])
        else:
            self._Output["市场净值"].append((self._Output["QP_P_MarketPos"]*Price).sum())
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.IDFilter)
        FactorData = self._FactorTable.readData(dts=[idt], ids=IDs, factor_names=[self.TestFactor]).iloc[0, 0, :]
        FactorData = FactorData[pd.notnull(FactorData) & pd.notnull(Price[IDs])].copy()
        nID = FactorData.shape[0]
        if self.Perturbation and (nID>0):
            FactorData = FactorData.astype("float")
            MinDiff = np.min(np.abs(np.diff(FactorData.unique())))
            FactorData += np.random.rand(nID) * MinDiff * 0.01
        FactorData = FactorData.sort_values(ascending=(self.FactorOrder=="升序"), inplace=False)
        IDs = list(FactorData.index)
        if self.WeightFactor!="等权":
            WeightData = self._FactorTable.readData(dts=[idt], ids=self._FactorTable.getID(ifactor_name=self.WeightFactor), factor_names=[self.WeightFactor]).iloc[0, 0, :]
        else:
            WeightData = pd.Series(1.0, index=self._FactorTable.getID())
        if self.IndustryFactor=="无":
            nSubID = distributeEqual(nID, self.GroupNum, remainder_pos="middle")
            for i in range(self.GroupNum):
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
            Portfolio = [{} for i in range(self.GroupNum)]
            IndustryData = self._FactorTable.readData(dts=[idt], ids=IDs, factor_names=[self.IndustryFactor]).iloc[0, 0, :]
            AllIndustry = IndustryData.unique()
            for iIndustry in AllIndustry:
                iMask = (IndustryData==iIndustry)
                iIDIndex = IndustryData[iMask].index
                iIDDistribution = distributeEqual(iIDIndex.shape[0], self.GroupNum, remainder_pos="middle")
                for j in range(self.GroupNum):
                    jSubID = iIDIndex[sum(iIDDistribution[:j]):sum(iIDDistribution[:j+1])]
                    Portfolio[j].update({kID:WeightData[kID] for kID in jSubID})
            for i in range(self.GroupNum):
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
        if self.MarketIDFilter:
            IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.MarketIDFilter)
        WeightData = WeightData[IDs]
        Price = Price[IDs]
        WeightData = WeightData[pd.notnull(WeightData) & pd.notnull(Price)]
        WeightData = WeightData / WeightData.sum()
        self._Output["QP_P_MarketPos"] = WeightData*self._Output["市场净值"][-1]/Price
        self._Output["QP_P_MarketPos"] = self._Output["QP_P_MarketPos"][pd.notnull(self._Output["QP_P_MarketPos"])]
        self._Output["调仓日"].append(idt)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        self._Output.pop("QP_P_CurPos")
        self._Output.pop("QP_P_MarketPos")
        for i in range(self.GroupNum):
            self._Output["净值"][i].pop(0)
        self._Output["净值"] = pd.DataFrame(np.array(self._Output["净值"]).T, index=self._Model.DateTimeSeries)
        self._Output["净值"]["市场"] = pd.Series(self._Output.pop("市场净值")[1:], index=self._Model.DateTimeSeries)
        self._Output["收益率"] = self._Output["净值"].iloc[1:,:].values / self._Output["净值"].iloc[:-1,:].values - 1
        self._Output["收益率"] = pd.DataFrame(np.row_stack((np.zeros((1, self.GroupNum+1)), self._Output["收益率"])), index=self._Model.DateTimeSeries, columns=[i for i in range(self.GroupNum)]+["市场"])
        self._Output["收益率"]["L-S"] = self._Output["收益率"].iloc[:, 0] - self._Output["收益率"].iloc[:, -2]
        self._Output["净值"]["L-S"] = (1 + self._Output["收益率"]["L-S"]).cumprod()
        self._Output["换手率"] = pd.DataFrame(np.array(self._Output["换手率"]).T, index=self._Model.DateTimeSeries)
        self._Output["投资组合"] = {str(i): pd.DataFrame(self._Output["投资组合"][i], index=self._Output["调仓日"]) for i in range(self.GroupNum)}
        self._Output["超额收益率"] = self._Output["收益率"].copy()
        self._Output["超额净值"] = self._Output["超额收益率"].copy()
        for i in self._Output["超额收益率"]:
            self._Output["超额收益率"][i] = self._Output["超额收益率"][i]-self._Output["收益率"]["市场"]
            self._Output["超额净值"][i] = (1+self._Output["超额收益率"][i]).cumprod()
        nDate = self._Output["净值"].shape[0]
        nDays = (self._Model.DateTimeSeries[-1] - self._Model.DateTimeSeries[0]).days
        nYear = nDays / 365
        TotalReturn = self._Output["净值"].iloc[-1,:]-1
        self._Output["统计数据"] = pd.DataFrame(index=TotalReturn.index)
        self._Output["统计数据"]["总收益率"] = TotalReturn
        self._Output["统计数据"]["年化收益率"] = (1+TotalReturn)**(1/nYear)-1
        self._Output["统计数据"]["波动率"] = self._Output["收益率"].std()*np.sqrt(nDate/nYear)
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
        self._Output["统计数据"]["跟踪误差"] = self._Output["超额收益率"].std()*np.sqrt(nDate/nYear)
        self._Output["统计数据"]["信息比率"] = self._Output["统计数据"]["年化超额收益率"]/self._Output["统计数据"]["跟踪误差"]
        self._Output["统计数据"]["t统计量(信息比率)"] = self._Output["统计数据"]["信息比率"]*np.sqrt(nYear)
        self._Output["统计数据"]["胜率"] = (self._Output["超额收益率"]>0).sum() / nDate
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
        self._Output.pop("调仓日")
        return 0
    def genMatplotlibFig(self, file_path=None):
        nRow, nCol = 3, 3
        Fig = plt.figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        AxesGrid = gridspec.GridSpec(nRow, nCol)
        xData = np.arange(1, self._Output["统计数据"].shape[0]-1)
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"].index[:-2]]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(plt.subplot(AxesGrid[0, 0]), xData, xTickLabels, self._Output["统计数据"]["年化超额收益率"].iloc[:-2], PercentageFormatter, self._Output["统计数据"]["胜率"].iloc[:-2], PercentageFormatter)
        _QS_plotStatistics(plt.subplot(AxesGrid[0, 1]), xData, xTickLabels, self._Output["统计数据"]["信息比率"].iloc[:-2], PercentageFormatter, None)
        _QS_plotStatistics(plt.subplot(AxesGrid[0, 2]), xData, xTickLabels, self._Output["统计数据"]["超额最大回撤率"].iloc[:-2], PercentageFormatter, None)
        _QS_plotStatistics(plt.subplot(AxesGrid[1, 0]), xData, xTickLabels, self._Output["统计数据"]["年化收益率"].iloc[:-2], PercentageFormatter, pd.Series(self._Output["统计数据"].loc["市场", "年化收益率"], index=self._Output["统计数据"].index[:-2], name="市场"), PercentageFormatter, False)
        _QS_plotStatistics(plt.subplot(AxesGrid[1, 1]), xData, xTickLabels, self._Output["统计数据"]["Sharpe比率"].iloc[:-2], FloatFormatter, pd.Series(self._Output["统计数据"].loc["市场", "Sharpe比率"], index=self._Output["统计数据"].index[:-2], name="市场"), FloatFormatter, False)
        _QS_plotStatistics(plt.subplot(AxesGrid[1, 2]), xData, xTickLabels, self._Output["统计数据"]["平均换手率"].iloc[:-2], PercentageFormatter, None)
        Axes = plt.subplot(AxesGrid[2, 0])
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, 0].values, label=str(self._Output["净值"].iloc[:, 0].name), color="r", alpha=0.6, lw=3)
        Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, -3].values, label=str(self._Output["净值"].iloc[:, -3].name), color="b", alpha=0.6, lw=3)
        Axes.plot(self._Output["净值"].index, self._Output["净值"]["市场"].values, label="市场", color="g", alpha=0.6, lw=3)
        Axes.legend(loc='best')
        Axes = plt.subplot(AxesGrid[2, 1])
        xData = np.arange(0, self._Output["净值"].shape[0])
        xTicks = np.arange(0, self._Output["净值"].shape[0], max(1, int(self._Output["净值"].shape[0]/8)))
        xTickLabels = [self._Output["净值"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        Axes.plot(xData, self._Output["净值"]["L-S"].values, label="多空净值", color="r", alpha=0.6, lw=3)
        Axes.legend(loc='upper left')
        RAxes = Axes.twinx()
        RAxes.yaxis.set_major_formatter(PercentageFormatter)
        RAxes.bar(xData, self._Output["收益率"]["L-S"].values, label="多空收益率", color="b")
        RAxes.legend(loc="upper right")
        Axes.set_xticks(xTicks)
        Axes.set_xticklabels(xTickLabels)
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self.ArgNames:
                if iArgName!="调仓时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        Formatters = [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d")]*2
        Formatters += [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d")]*2
        Formatters += [lambda x:'{0:.2f}'.format(x)]*2
        iHTML = self._Output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = self.genMatplotlibFig()
        # figure 保存为二进制文件
        Buffer = BytesIO()
        plt.savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML