# coding=utf-8
import datetime as dt
from copy import deepcopy
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import Enum, List, Int, Str, Float, Dict, ListStr, on_trait_change
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from scipy import stats

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import summaryStrategy, formatStrategySummary, summaryTimingStrategy, formatTimingStrategySummary
from QuantStudio.BackTest.BackTestModel import BaseModule


# 给定仓位水平的择时信号回测
# 测试因子: 每期的目标仓位水平, (-inf, inf) 的仓位水平或者 nan 表示维持目前仓位
class PositionSignal(BaseModule):
    """仓位信号回测"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    def __init__(self, factor_table, name="仓位信号回测", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ["价","Price","price"])
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        IDs = self._FactorTable.getID()
        nDT, nID, nFactor = len(dts), len(IDs), len(self.TestFactors)
        self._Output["标的ID"] = IDs
        self._Output["标的价格"] = np.zeros(shape=(nDT, nID))
        self._Output["目标仓位"] = np.full(shape=(nDT, nID, nFactor), fill_value=np.nan)
        self._Output["仓位"] = np.full(shape=(nDT, nID, nFactor), fill_value=np.nan)
        self._Output["总计"] = np.ones(shape=(nDT, nID, nFactor))
        self._Output["标的"] = np.zeros(shape=(nDT, nID, nFactor))
        self._Output["现金"] = np.ones(shape=(nDT, nID, nFactor))
        self._Output["换手率"] = np.zeros(shape=(nDT, nID, nFactor))
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        DTIdx = self._Model.DateTimeIndex
        iPosition = self._FactorTable.readData(factor_names=list(self.TestFactors), ids=self._Output["标的ID"], dts=[self._Model.DateTimeSeries[DTIdx]]).iloc[:, 0, :].values
        Price = self._FactorTable.readData(factor_names=[self.PriceFactor], ids=self._Output["标的ID"], dts=[self._Model.DateTimeSeries[DTIdx]]).iloc[0, 0, :].values
        self._Output["标的价格"][DTIdx] = Price
        if DTIdx==0:
            iPosition[pd.isnull(iPosition)] = 0
            self._Output["目标仓位"][0] = iPosition
            self._Output["标的"][0] = iPosition * self._Output["总计"][0]
            self._Output["现金"][0] = self._Output["总计"][0] - self._Output["标的"][0]
            self._Output["仓位"][0] = self._Output["标的"][0] / np.abs(self._Output["总计"][0])
            self._Output["换手率"][0] = np.abs(self._Output["总计"][0] - self._Output["现金"][0]) / np.abs(self._Output["总计"][0])
        else:
            iReturn = self._Output["标的价格"][DTIdx] / self._Output["标的价格"][DTIdx-1] - 1
            iReturn[pd.isnull(iReturn)] = 0
            self._Output["目标仓位"][DTIdx] = iPosition
            iAmount = (self._Output["标的"][DTIdx-1].T * (1+iReturn)).T
            self._Output["总计"][DTIdx] = iAmount + self._Output["现金"][DTIdx-1]
            iPosition[self._Output["总计"][DTIdx]<=0] = 0
            iMask = pd.notnull(iPosition)
            iAmount[iMask] = (iPosition * self._Output["总计"][DTIdx])[iMask]
            self._Output["现金"][DTIdx] = self._Output["总计"][DTIdx] - iAmount
            self._Output["标的"][DTIdx] = iAmount
            self._Output["仓位"][DTIdx] = self._Output["标的"][DTIdx] / np.abs(self._Output["总计"][DTIdx])
            self._Output["换手率"][DTIdx] = np.abs(self._Output["现金"][DTIdx] - self._Output["现金"][DTIdx-1]) / np.abs(self._Output["总计"][DTIdx])
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        DTs = self._Model.DateTimeSeries
        nDT = len(DTs)
        nYear = (DTs[-1] - DTs[0]).days / 365
        self._Output["标的净值"] = pd.DataFrame(self._Output.pop("标的价格"), index=DTs, columns=self._Output["标的ID"])
        self._Output["标的净值"] = self._Output["标的净值"].fillna(method="ffill")
        self._Output["标的净值"] = self._Output["标的净值"] / self._Output["标的净值"].fillna(method="bfill").iloc[0]
        self._Output["时间序列"] = {}
        TargetPosition, self._Output["时间序列"]["目标仓位"] = self._Output.pop("目标仓位"), {}
        Position, self._Output["时间序列"]["仓位"] = self._Output.pop("仓位"), {}
        TotalAmt, self._Output["时间序列"]["总计"] = self._Output.pop("总计"), {}
        Amt, self._Output["时间序列"]["标的"] = self._Output.pop("标的"), {}
        Cash, self._Output["时间序列"]["现金"] = self._Output.pop("现金"), {}
        Turnover, self._Output["时间序列"]["换手率"] = self._Output.pop("换手率"), {}
        self._Output["统计数据"] = {}
        self._Output["择时统计"] = {"多空统计": {}, "多头统计": {}, "空头统计": {}}
        for i, iFactor in enumerate(self.TestFactors):
            self._Output["时间序列"]["目标仓位"][iFactor] = pd.DataFrame(TargetPosition[:, :, i], index=DTs, columns=self._Output["标的ID"])
            self._Output["时间序列"]["仓位"][iFactor] = pd.DataFrame(Position[:, :, i], index=DTs, columns=self._Output["标的ID"])
            self._Output["时间序列"]["总计"][iFactor] = pd.DataFrame(TotalAmt[:, :, i], index=DTs, columns=self._Output["标的ID"])
            self._Output["时间序列"]["标的"][iFactor] = pd.DataFrame(Amt[:, :, i], index=DTs, columns=self._Output["标的ID"])
            self._Output["时间序列"]["现金"][iFactor] = pd.DataFrame(Cash[:, :, i], index=DTs, columns=self._Output["标的ID"])
            self._Output["时间序列"]["换手率"][iFactor] = pd.DataFrame(Turnover[:, :, i], index=DTs, columns=self._Output["标的ID"])
            self._Output["统计数据"][iFactor] = summaryStrategy(self._Output["时间序列"]["总计"][iFactor].values, self._Output["时间序列"]["总计"][iFactor].index.tolist(), risk_free_rate=0.0)
            self._Output["统计数据"][iFactor].columns = self._Output["时间序列"]["总计"][iFactor].columns
            self._Output["择时统计"]["多空统计"][iFactor], self._Output["择时统计"]["多头统计"][iFactor], self._Output["择时统计"]["空头统计"][iFactor], _, _ = summaryTimingStrategy(self._Output["时间序列"]["目标仓位"][iFactor].values, self._Output["标的净值"].values, n_per_year=nDT/nYear)
            self._Output["择时统计"]["多空统计"][iFactor].columns = self._Output["择时统计"]["多头统计"][iFactor].columns = self._Output["择时统计"]["空头统计"][iFactor].columns = self._Output["时间序列"]["总计"][iFactor].columns
        self._Output.pop("标的ID")
        return 0
    def genMatplotlibFig(self, file_path=None, target_factor=None):
        if target_factor is None: target_factor = self.TestFactors[0]
        iNV = self._Output["时间序列"]["总计"][target_factor]
        iPos = self._Output["时间序列"]["仓位"][target_factor]
        iTargetNV = self._Output["标的净值"]
        nID = iTargetNV.shape[1]
        xData = np.arange(0, iNV.shape[0])
        xTicks = np.arange(0, iNV.shape[0], int(iNV.shape[0]/10))
        xTickLabels = [iNV.index[i].strftime("%Y-%m-%d") for i in xTicks]
        nRow, nCol = nID//3+(nID%3!=0), min(3, nID)
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        for j, jID in enumerate(iTargetNV.columns):
            iAxes = Fig.add_subplot(nRow, nCol, j+1)
            ijPos = iPos[jID].copy()
            ijPos[~(ijPos>=0)] = 0
            iAxes.bar(xData, ijPos.values, label="多头仓位", color="indianred", alpha=0.5)
            ijPos = iPos[jID].copy()
            ijPos[~(ijPos<0)] = 0
            iAxes.bar(xData, ijPos.values, label="空头仓位", color="forestgreen", alpha=0.5)
            iRAxes = iAxes.twinx()
            iRAxes.plot(xData, iTargetNV[jID].values, label=str(jID), color="steelblue", lw=2.5)
            iRAxes.plot(xData, iNV[jID].values, label="策略净值", color="k", lw=2.5)
            iRAxes.legend()
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend()
            iAxes.set_title(target_factor+" - "+str(jID)+" : 仓位信号策略净值")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self.ArgNames:
                HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        for iFactor in self.TestFactors:
            iOutput = self._Output["统计数据"][iFactor]
            iHTML = iFactor+" - 统计数据 : "
            iHTML += formatStrategySummary(iOutput).to_html()
            Pos = iHTML.find(">")
            HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            iOutput = self._Output["择时统计"]["多空统计"][iFactor]
            iHTML = iFactor+" - 多空统计 : "
            iHTML += formatTimingStrategySummary(iOutput).to_html()
            Pos = iHTML.find(">")
            HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            iOutput = self._Output["择时统计"]["多头统计"][iFactor]
            iHTML = iFactor+" - 多头统计 : "
            iHTML += formatTimingStrategySummary(iOutput).to_html()
            Pos = iHTML.find(">")
            HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            iOutput = self._Output["择时统计"]["空头统计"][iFactor]
            iHTML = iFactor+" - 空头统计 : "
            iHTML += formatTimingStrategySummary(iOutput).to_html()
            Pos = iHTML.find(">")
            HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            Fig = self.genMatplotlibFig(target_factor=iFactor)
            # figure 保存为二进制文件
            Buffer = BytesIO()
            Fig.savefig(Buffer)
            PlotData = Buffer.getvalue()
            # 图像数据转化为 HTML 格式
            ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
            HTML += ('<img src="%s">' % ImgStr)
        return HTML

class QuantileTiming(BaseModule):
    """分位数择时"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    FactorOrder = Dict(key_trait=Str(), value_trait=Enum("降序", "升序"), arg_type="ArgDict", label="排序方向", order=1)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=3)
    SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=4)
    MinSummaryWindow = Int(3, arg_type="Integer", label="最小统计窗口", order=5)
    GroupNum = Int(3, arg_type="Integer", label="分组数", order=6)
    def __init__(self, factor_table, name="分位数择时", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ["价","Price","price"])
    @on_trait_change("TestFactors[]")
    def _on_TestFactors_changed(self, obj, name, old, new):
        self.FactorOrder = {iFactorName:self.FactorOrder.get(iFactorName, "降序") for iFactorName in self.TestFactors}
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        SecurityIDs = self._FactorTable.getID()
        nDT, nSecurityID, nFactor = len(dts), len(SecurityIDs), len(self.TestFactors)
        self._Output["标的ID"] = SecurityIDs
        self._Output["因子符号"] = 2 * np.array([(self.FactorOrder.get(iFactor, "降序")=="升序") for iFactor in self.TestFactors]).astype(float) - 1
        self._Output["标的收益率"] = np.zeros(shape=(nDT, nSecurityID))
        self._Output["因子值"] = np.full(shape=(nDT, nSecurityID, nFactor), fill_value=np.nan)
        self._Output["最新信号"] = np.full(shape=(nSecurityID, nFactor), fill_value=np.nan)
        self._Output["策略信号"] = np.full(shape=(nDT, nSecurityID, nFactor, self.GroupNum), fill_value=np.nan)
        self._Output["策略收益率"] = np.full(shape=(nDT, nSecurityID, nFactor, self.GroupNum), fill_value=np.nan)
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        DTIdx = self._Model.DateTimeIndex
        _, nSecurityID, nFactor = self._Output["因子值"].shape
        if DTIdx>0:
            Price = self._FactorTable.readData(dts=self._Model.DateTimeSeries[DTIdx-1:DTIdx+1], ids=self._Output["标的ID"], factor_names=[self.PriceFactor]).iloc[0, :, :].values
            Return = Price[1] / Price[0] - 1
            Return[pd.isnull(Return)] = 0
            self._Output["标的收益率"][DTIdx, :] = Return
            Mask = pd.notnull(self._Output["最新信号"])
            if np.any(Mask):
                SecurityIdx, FactorIdx = np.where(Mask)
                self._Output["策略收益率"][DTIdx, SecurityIdx, FactorIdx, self._Output["最新信号"][Mask].astype(int)] = np.repeat(np.reshape(Return, (nSecurityID, 1)), nFactor, axis=1)[Mask]
        self._Output["因子值"][DTIdx] = self._FactorTable.readData(dts=self._Model.DateTimeSeries[DTIdx:DTIdx+1], ids=self._Output["标的ID"], factor_names=list(self.TestFactors)).iloc[:, 0, :].values
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        if DTIdx+1<self.MinSummaryWindow: return 0
        FactorData = self._Output["因子值"][int(max(0, DTIdx + 1 - self.SummaryWindow)):DTIdx+1] * self._Output["因子符号"]
        Quantiles = np.nanpercentile(FactorData, np.arange(1, self.GroupNum+1) / self.GroupNum * 100, axis=0)
        Signal = np.sum(Quantiles<FactorData[-1, :], axis=0).astype(float)
        Signal[pd.isnull(FactorData[-1, :])] = np.nan
        self._Output["最新信号"] = Signal
        Mask = pd.notnull(Signal)
        if np.any(Mask):
            SecurityIdx, FactorIdx = np.where(Mask)
            self._Output["策略信号"][DTIdx, SecurityIdx, FactorIdx, Signal[Mask].astype(int)] = 1
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        DTs = self._Model.DateTimeSeries
        self._Output["标的收益率"] = pd.DataFrame(self._Output["标的收益率"], index=DTs, columns=self._Output["标的ID"])
        self._Output["标的净值"] = (self._Output["标的收益率"] + 1).cumprod()
        self._Output["因子值"] = {iFactor: pd.DataFrame(self._Output["因子值"][:, :, i], index=DTs, columns=self._Output["标的ID"]) for i, iFactor in enumerate(self.TestFactors)}
        Groups = np.arange(1, self.GroupNum+1).astype(str)
        Signal = self._Output["策略信号"]
        SignalReturn = self._Output["策略收益率"]
        self._Output["策略信号"] = {}
        self._Output["策略收益率"] = {}
        self._Output["策略净值"] = {}
        self._Output["统计数据"] = {}
        for i, iFactor in enumerate(self.TestFactors):
            self._Output["策略信号"][iFactor] = {}
            self._Output["策略收益率"][iFactor] = {}
            self._Output["策略净值"][iFactor] = {}
            self._Output["统计数据"][iFactor] = {}
            for j, jID in enumerate(self._Output["标的ID"]):
                ijSignal = pd.DataFrame(Signal[:, j, i, :], index=DTs, columns=Groups)
                ijDTs = pd.notnull(ijSignal).any(axis=1)
                ijDTs = ijDTs[ijDTs].index
                if ijDTs.shape[0]>0:
                    self._Output["策略信号"][iFactor][jID] = ijSignal.loc[ijDTs[0]:]
                else:
                    self._Output["策略信号"][iFactor][jID] = pd.DataFrame(columns=Groups)
                ijSignalReturn = pd.DataFrame(SignalReturn[:, j, i, :], index=DTs, columns=Groups).loc[self._Output["策略信号"][iFactor][jID].index]
                ijSignalReturn["L-S"] = ijSignalReturn.iloc[:, 0].fillna(0) - ijSignalReturn.iloc[:, -1].fillna(0)
                if ijSignalReturn.shape[0]==0:
                    self._Output["策略收益率"][iFactor][jID] = ijSignalReturn
                    self._Output["策略净值"][iFactor][jID] = ijSignalReturn
                    self._Output["统计数据"][iFactor][jID] = pd.DataFrame(index=ijSignalReturn.columns)
                else:
                    ijNV = (1+ijSignalReturn.fillna(0)).cumprod()
                    self._Output["策略收益率"][iFactor][jID] = ijSignalReturn
                    self._Output["策略净值"][iFactor][jID] = ijNV
                    ijStat = summaryStrategy(ijNV.values, ijNV.index.tolist(), risk_free_rate=0.0)
                    ijStat.columns = ijNV.columns
                    self._Output["统计数据"][iFactor][jID] = ijStat.T
                    # t 检验
                    tStats, pVal = stats.ttest_1samp(ijSignalReturn.values, 0, axis=0, nan_policy="omit")
                    tStats[-1], pVal[-1] = stats.ttest_ind(ijSignalReturn.iloc[:, 0].values, ijSignalReturn.iloc[:, -2], equal_var=True, nan_policy="omit")
                    self._Output["统计数据"][iFactor][jID]["t统计量"] = tStats
                    self._Output["统计数据"][iFactor][jID]["p值"] = pVal
        self._Output.pop("最新信号")
        self._Output.pop("因子符号")
        self._Output.pop("标的ID")
        return 0
    def genMatplotlibFig(self, file_path=None, target_factor=None):
        if target_factor is None: target_factor = self.TestFactors[0]
        iOutput = self._Output["策略净值"][target_factor]
        nID = len(iOutput)
        nRow, nCol = nID//3+(nID%3!=0), min(3, nID)
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        for j, jID in enumerate(sorted(iOutput.keys())):
            iAxes = Fig.add_subplot(nRow, nCol, j+1)
            iOutput[jID].iloc[:, :-1].plot(ax=iAxes, lw=2.5, title=target_factor+" - "+str(jID)+" : 分位数择时净值")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self.ArgNames:
                if iArgName!="计算时点":
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        FloatFormatFun = np.vectorize(lambda x: ("%.4f" % (x, )))
        for iFactor in self.TestFactors:
            iOutput = self._Output["统计数据"][iFactor]
            for jID in sorted(iOutput.keys()):
                iHTML = iFactor+" - "+str(jID)+" : "
                iFormat = formatStrategySummary(iOutput[jID].T)
                iFormat.iloc[12:] = FloatFormatFun(iOutput[jID].T.iloc[12:, :].values)
                iHTML += iFormat.T.to_html()
                Pos = iHTML.find(">")
                HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            Fig = self.genMatplotlibFig(target_factor=iFactor)
            # figure 保存为二进制文件
            Buffer = BytesIO()
            Fig.savefig(Buffer)
            PlotData = Buffer.getvalue()
            # 图像数据转化为 HTML 格式
            ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
            HTML += ('<img src="%s">' % ImgStr)
        return HTML