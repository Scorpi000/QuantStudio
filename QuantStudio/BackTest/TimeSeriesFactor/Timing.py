# coding=utf-8
import datetime as dt
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from traits.api import Enum, List, Int, Str, Float, Dict, ListStr, on_trait_change, ListInt, Bool
from matplotlib.figure import Figure
from matplotlib import cm
from scipy import stats
  
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import testTimingStrategy, summaryStrategy, formatStrategySummary, summaryTimingStrategy, formatTimingStrategySummary, summaryTrade, formatTradeSummary
from QuantStudio.Tools import CashFlowCalculator
from QuantStudio.BackTest.BackTestModel import BaseModule


# 给定仓位水平的择时信号回测
# 测试因子: 每期的目标仓位水平, (-inf, inf) 的仓位水平或者 nan 表示维持目前仓位
class TargetPositionSignal(BaseModule):
    """目标仓位信号回测"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
        def __QS_initArgs__(self):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ["价","Price","price"])
    
    def __init__(self, factor_table, name="目标仓位信号回测", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        IDs = self._FactorTable.getID()
        nDT, nID, nFactor = len(dts), len(IDs), len(self._QSArgs.TestFactors)
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
        iPosition = self._FactorTable.readData(factor_names=list(self._QSArgs.TestFactors), ids=self._Output["标的ID"], dts=[self._Model.DateTimeSeries[DTIdx]]).iloc[:, 0, :].values
        Price = self._FactorTable.readData(factor_names=[self._QSArgs.PriceFactor], ids=self._Output["标的ID"], dts=[self._Model.DateTimeSeries[DTIdx]]).iloc[0, 0, :].values
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
        self._Output["统计数据"] = {"标的": summaryStrategy(self._Output["标的净值"].values, self._Output["标的净值"].index.tolist(), risk_free_rate=0.0)}
        self._Output["统计数据"]["标的"].columns = self._Output["标的净值"].columns
        self._Output["择时统计"] = {"多空统计": {}, "多头统计": {}, "空头统计": {}}
        for i, iFactor in enumerate(self._QSArgs.TestFactors):
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
        if target_factor is None: target_factor = self._QSArgs.TestFactors[0]
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
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        for iFactor in self._QSArgs.TestFactors:
            iHTML = iFactor+" - 统计数据 : "
            iOutput = formatStrategySummary(self._Output["统计数据"][iFactor]).reset_index()
            iOutput.columns = ["Statistics"]+iOutput.columns.tolist()[1:]
            iHTML += iOutput.to_html(index=False, notebook=True)
            Pos = iHTML.find(">")
            HTML += ('<div style="float:left;width:49%%;overflow:scroll">%s</div>' % (iHTML[:Pos]+' align="center"'+iHTML[Pos:],))
            iHTML = "标的 - 统计数据 : "
            iOutput = formatStrategySummary(self._Output["统计数据"]["标的"]).reset_index()
            iOutput.columns = ["Statistics"]+iOutput.columns.tolist()[1:]
            iHTML += iOutput.to_html(index=False, notebook=True)
            Pos = iHTML.find(">")
            HTML += ('<div style="float:left;width:49%%;overflow:scroll">%s</div>' % (iHTML[:Pos]+' align="center"'+iHTML[Pos:],))
            iHTML = iFactor+" - 多空统计 : "
            iOutput = formatTimingStrategySummary(self._Output["择时统计"]["多空统计"][iFactor]).reset_index()
            iOutput.columns = ["Statistics"]+iOutput.columns.tolist()[1:]
            iHTML += iOutput.to_html(index=False, notebook=True)
            Pos = iHTML.find(">")
            HTML += ('<div style="float:left;width:33%%;overflow:scroll">%s</div>' % (iHTML[:Pos]+' align="center"'+iHTML[Pos:],))
            iHTML = iFactor+" - 多头统计 : "
            iOutput = formatTimingStrategySummary(self._Output["择时统计"]["多头统计"][iFactor]).reset_index()
            iOutput.columns = ["Statistics"]+iOutput.columns.tolist()[1:]
            iHTML += iOutput.to_html(index=False, notebook=True)
            Pos = iHTML.find(">")
            HTML += ('<div style="float:left;width:33%%;overflow:scroll">%s</div>' % (iHTML[:Pos]+' align="center"'+iHTML[Pos:],))
            iHTML = iFactor+" - 空头统计 : "
            iOutput = formatTimingStrategySummary(self._Output["择时统计"]["空头统计"][iFactor]).reset_index()
            iOutput.columns = ["Statistics"]+iOutput.columns.tolist()[1:]
            iHTML += iOutput.to_html(index=False, notebook=True)
            Pos = iHTML.find(">")
            HTML += ('<div style="float:left;width:33%%;overflow:scroll">%s</div>' % (iHTML[:Pos]+' align="center"'+iHTML[Pos:],))
            Fig = self.genMatplotlibFig(target_factor=iFactor)
            # figure 保存为二进制文件
            Buffer = BytesIO()
            Fig.savefig(Buffer)
            PlotData = Buffer.getvalue()
            # 图像数据转化为 HTML 格式
            ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
            HTML += ('<img src="%s">' % ImgStr)
        return HTML

# 给定买卖金额的交易信号回测
# 测试因子: 每期的买卖金额, (-inf, inf) 的买卖金额, 0 表示清仓或者 nan 表示无交易
class TradeSignal(BaseModule):
    """交易信号回测"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
        CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=2)
        EndClear = Bool(True, arg_type="Bool", label="结束清仓", order=3)
        CalcIRR = Bool(False, arg_type="Bool", label="计算IRR", order=4)
        def __QS_initArgs__(self):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ["价","Price","price"])
            
    def __init__(self, factor_table, name="交易信号回测", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        IDs = self._FactorTable.getID()
        nDT, nID, nFactor = len(dts), len(IDs), len(self._QSArgs.TestFactors)
        self._Output["标的ID"] = IDs
        self._Output["标的价格"] = np.zeros(shape=(nDT, nID))
        self._Output["交易信号"] = np.full(shape=(nDT, nID, nFactor), fill_value=np.nan)
        self._Output["现金流"] = np.zeros(shape=(nDT, nID, nFactor))
        self._Output["现金余额"] = np.zeros(shape=(nDT, nID, nFactor))
        self._Output["标的数量"] = np.zeros(shape=(nDT, nID, nFactor))
        self._Output["标的金额"] = np.zeros(shape=(nDT, nID, nFactor))
        self._Output["账户余额"] = np.zeros(shape=(nDT, nID, nFactor))
        self._Output["收益率"] = np.zeros(shape=(nDT, nID, nFactor))
        self._Output["交易记录"] = {iFactorName: pd.DataFrame(columns=["ID", "开仓时点", "开仓价格", "交易数量", "平仓时点", "平仓价格", "盈亏金额", "收益率"]) for iFactorName in self._QSArgs.TestFactors}
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        DTIdx = self._Model.DateTimeIndex
        Price = self._FactorTable.readData(factor_names=[self._QSArgs.PriceFactor], ids=self._Output["标的ID"], dts=[self._Model.DateTimeSeries[DTIdx]]).iloc[0, 0, :]
        if DTIdx>0:
            Mask = pd.isnull(Price).values
            Price[Mask] = self._Output["标的价格"][DTIdx-1][Mask]
        self._Output["标的价格"][DTIdx] = Price.values
        PriceVal = np.repeat(np.reshape(Price.values, (-1, 1)), len(self._QSArgs.TestFactors), axis=1)
        # 交易前账户结算
        if DTIdx>0:
            self._Output["现金余额"][DTIdx] = self._Output["现金余额"][DTIdx-1]
            self._Output["标的数量"][DTIdx] = self._Output["标的数量"][DTIdx-1]
            UnderlyingAmt = self._Output["标的数量"][DTIdx] * PriceVal
            Mask = pd.isnull(UnderlyingAmt)
            UnderlyingAmt[Mask] = self._Output["标的金额"][DTIdx-1][Mask]
            self._Output["标的金额"][DTIdx] = UnderlyingAmt
            self._Output["账户余额"][DTIdx] = UnderlyingAmt + self._Output["现金余额"][DTIdx]
            self._Output["收益率"][DTIdx] = (self._Output["账户余额"][DTIdx] - self._Output["账户余额"][DTIdx-1]) / np.abs(self._Output["账户余额"][DTIdx-1])
            self._Output["收益率"][DTIdx][self._Output["账户余额"][DTIdx-1]==0] = 0
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        # 交易
        TradeAmt = self._FactorTable.readData(factor_names=list(self._QSArgs.TestFactors), ids=self._Output["标的ID"], dts=[self._Model.DateTimeSeries[DTIdx]]).iloc[:, 0, :].values
        self._Output["交易信号"][DTIdx] = TradeAmt
        UnderlyingNum = self._Output["标的数量"][DTIdx].copy()
        UnderlyingAmt = self._Output["标的金额"][DTIdx].copy()
        Cash = self._Output["现金余额"][DTIdx].copy()
        # 清仓
        IDs = np.array(self._Output["标的ID"])
        ClearMask = (pd.notnull(TradeAmt) & (np.sign(TradeAmt) != np.sign(UnderlyingNum)))
        CashFlow = np.zeros(UnderlyingAmt.shape)
        CashFlow[ClearMask] = - (UnderlyingAmt[ClearMask] + Cash[ClearMask])
        Cash[ClearMask] = 0
        UnderlyingNum[ClearMask] = 0
        UnderlyingAmt[ClearMask] = 0
        for i, iFactorName in enumerate(self._QSArgs.TestFactors):
            if np.any(ClearMask[:, i]):
                iClearMask = (pd.isnull(self._Output["交易记录"][iFactorName]["平仓时点"]) & (self._Output["交易记录"][iFactorName]["ID"].isin(IDs[ClearMask[:, i]])))
                self._Output["交易记录"][iFactorName].loc[iClearMask, "平仓时点"] = idt
                self._Output["交易记录"][iFactorName].loc[iClearMask, "平仓价格"] = Price.reindex(index=self._Output["交易记录"][iFactorName].loc[iClearMask, "ID"]).values
                self._Output["交易记录"][iFactorName].loc[iClearMask, "盈亏金额"] = (self._Output["交易记录"][iFactorName].loc[iClearMask, "平仓价格"] - self._Output["交易记录"][iFactorName].loc[iClearMask, "开仓价格"]) * self._Output["交易记录"][iFactorName].loc[iClearMask, "交易数量"]
                self._Output["交易记录"][iFactorName].loc[iClearMask, "收益率"] = self._Output["交易记录"][iFactorName].loc[iClearMask, "盈亏金额"] / (self._Output["交易记录"][iFactorName].loc[iClearMask, "开仓价格"] * self._Output["交易记录"][iFactorName].loc[iClearMask, "交易数量"]).abs()
        # 开仓
        TradeNum = TradeAmt / PriceVal
        Mask = pd.isnull(TradeNum)
        TradeAmt[Mask] = 0
        TradeNum[Mask] = 0
        self._Output["现金流"][DTIdx] = CashFlow + np.abs(TradeAmt)
        self._Output["标的金额"][DTIdx] = UnderlyingAmt + TradeAmt
        self._Output["标的数量"][DTIdx] = UnderlyingNum + TradeNum
        self._Output["现金余额"][DTIdx] = Cash - 2 * np.clip(TradeAmt, -np.inf, 0)
        self._Output["账户余额"][DTIdx] = self._Output["现金余额"][DTIdx] + self._Output["标的金额"][DTIdx]
        Mask = (TradeNum!=0)
        for i, iFactorName in enumerate(self._QSArgs.TestFactors):
            if np.any(Mask[:, i]):
                iTradeRecord = pd.DataFrame({"ID": IDs[Mask[:, i]], "开仓时点": idt, "开仓价格": PriceVal[Mask[:, i], i], "交易数量": TradeNum[Mask[:, i], i]})
                self._Output["交易记录"][iFactorName] = self._Output["交易记录"][iFactorName].append(iTradeRecord, ignore_index=True)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        DTs, IDs = self._Model.DateTimeSeries, self._Output.pop("标的ID")
        nDT, nID = len(DTs), len(IDs)
        nYear = (DTs[-1] - DTs[0]).days / 365
        Price = self._FactorTable.readData(factor_names=[self._QSArgs.PriceFactor], ids=IDs, dts=[self._Model.DateTimeSeries[-1]]).iloc[0, 0, :]
        self._Output["时间序列"] = {}
        self._Output["时间序列"]["标的净值"] = pd.DataFrame(self._Output.pop("标的价格"), index=DTs, columns=IDs)
        self._Output["时间序列"]["标的净值"] = self._Output["时间序列"]["标的净值"].fillna(method="ffill")
        self._Output["时间序列"]["标的净值"] = self._Output["时间序列"]["标的净值"] / self._Output["时间序列"]["标的净值"].fillna(method="bfill").iloc[0]
        self._Output["统计数据"] = {}
        self._Output["交易统计"] = {"所有交易": {}, "多头交易": {}, "空头交易": {}}
        for i, iFactorName in enumerate(self._QSArgs.TestFactors):
            self._Output["时间序列"].setdefault("交易信号", {})[iFactorName] = pd.DataFrame(self._Output["交易信号"][:, :, i], index=DTs, columns=IDs)
            iCashFlow = self._Output["现金流"][:, :, i]
            self._Output["时间序列"].setdefault("现金流", {})[iFactorName] = pd.DataFrame(iCashFlow, index=DTs, columns=IDs)
            self._Output["时间序列"].setdefault("现金余额", {})[iFactorName] = pd.DataFrame(self._Output["现金余额"][:, :, i], index=DTs, columns=IDs)
            self._Output["时间序列"].setdefault("标的数量", {})[iFactorName] = pd.DataFrame(self._Output["标的数量"][:, :, i], index=DTs, columns=IDs)
            self._Output["时间序列"].setdefault("标的金额", {})[iFactorName] = pd.DataFrame(self._Output["标的金额"][:, :, i], index=DTs, columns=IDs)
            self._Output["时间序列"].setdefault("账户余额", {})[iFactorName] = pd.DataFrame(self._Output["账户余额"][:, :, i], index=DTs, columns=IDs)
            iReturn = self._Output["收益率"][:, :, i]
            self._Output["时间序列"].setdefault("收益率", {})[iFactorName] = pd.DataFrame(iReturn, index=DTs, columns=IDs)
            self._Output["时间序列"].setdefault("净值", {})[iFactorName] = (self._Output["时间序列"]["收益率"][iFactorName] + 1).cumprod()
            iStrategyStats = summaryStrategy(self._Output["时间序列"]["净值"][iFactorName].values, DTs)
            iStrategyStats.columns = IDs
            self._Output["统计数据"][iFactorName] = iStrategyStats
            iTradeRecord = self._Output["交易记录"][iFactorName].loc[:, ["ID", "开仓时点", "开仓价格", "交易数量", "平仓时点", "平仓价格", "盈亏金额", "收益率"]]
            if self._QSArgs.EndClear:# 清仓
                iClearMask = pd.isnull(iTradeRecord["平仓时点"])
                iTradeRecord.loc[iClearMask, "平仓时点"] = self._Model.DateTimeSeries[-1]
                iTradeRecord.loc[iClearMask, "平仓价格"] = Price.reindex(index=iTradeRecord.loc[iClearMask, "ID"]).values
                iTradeRecord.loc[iClearMask, "盈亏金额"] = (iTradeRecord.loc[iClearMask, "平仓价格"] - iTradeRecord.loc[iClearMask, "开仓价格"]) * iTradeRecord.loc[iClearMask, "交易数量"]
                iTradeRecord.loc[iClearMask, "收益率"] = iTradeRecord.loc[iClearMask, "盈亏金额"] / (iTradeRecord.loc[iClearMask, "开仓价格"] * iTradeRecord.loc[iClearMask, "交易数量"]).abs()
                iTradeRecord = iTradeRecord[iTradeRecord["平仓时点"]!=iTradeRecord["开仓时点"]]
            self._Output["交易记录"][iFactorName] = iTradeRecord
            iTradeStats = summaryTrade(iTradeRecord[pd.notnull(iTradeRecord["平仓时点"])])
            if self._QSArgs.CalcIRR:
                iIRR = (1 + CashFlowCalculator.rate(-iCashFlow, pv=0, fv=self._Output["时间序列"]["账户余额"][iFactorName].iloc[-1].values)) ** (nDT / nYear) - 1
                iTradeStats.loc["年化IRR"] = np.r_[(1 + CashFlowCalculator.rate(-np.sum(iCashFlow, axis=1), pv=0, fv=np.sum(self._Output["时间序列"]["账户余额"][iFactorName].iloc[-1].values))) ** (nDT / nYear) - 1, iIRR]
            self._Output["交易统计"]["所有交易"][iFactorName] = iTradeStats
            self._Output["交易统计"]["多头交易"][iFactorName] = summaryTrade(iTradeRecord[pd.notnull(iTradeRecord["平仓时点"]) & (iTradeRecord["交易数量"]>0)])
            self._Output["交易统计"]["空头交易"][iFactorName] = summaryTrade(iTradeRecord[pd.notnull(iTradeRecord["平仓时点"]) & (iTradeRecord["交易数量"]<0)])
        self._Output.pop("交易信号")
        self._Output.pop("现金流")
        self._Output.pop("现金余额")
        self._Output.pop("标的数量")
        self._Output.pop("标的金额")
        self._Output.pop("账户余额")
        self._Output.pop("收益率")
        return 0
    def genMatplotlibFig(self, file_path=None, target_factor=None):
        if target_factor is None: target_factor = self._QSArgs.TestFactors[0]
        iSignal = self._Output["时间序列"]["交易信号"][target_factor]
        iTargetNV = self._Output["时间序列"]["标的净值"]
        nID = iTargetNV.shape[1]
        xData = np.arange(0, iTargetNV.shape[0])
        xTicks = np.arange(0, iTargetNV.shape[0], int(iTargetNV.shape[0]/10))
        xTickLabels = [iTargetNV.index[i].strftime("%Y-%m-%d") for i in xTicks]
        nRow, nCol = nID//3+(nID%3!=0), min(3, nID)
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        for j, jID in enumerate(iTargetNV.columns):
            ijTargetNV = iTargetNV[jID].values
            ijSignal = iSignal[jID].values
            iAxes = Fig.add_subplot(nRow, nCol, j+1)
            h = iAxes.plot(xData, ijTargetNV, label=str(jID), color="steelblue", lw=1.5)
            h[0].set_zorder(0)
            ijMask = (ijSignal>0)
            if np.any(ijMask):
                iAxes.scatter(xData[ijMask], ijTargetNV[ijMask], 20, marker="^", c=ijSignal[ijMask] / np.nanmax(ijSignal[ijMask]), cmap=cm.get_cmap("Reds_r"))
            ijMask = (ijSignal<0)
            if np.any(ijMask):
                iAxes.scatter(xData[ijMask], ijTargetNV[ijMask], 20, marker="v", c=ijSignal[ijMask] / np.nanmin(ijSignal[ijMask]), cmap=cm.get_cmap("Greens_r"))
            ijMask = (ijSignal==0)
            if np.any(ijMask):
                iAxes.scatter(xData[ijMask], ijTargetNV[ijMask], 20, marker="o", color="y")
            iAxes.legend()
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend()
            iAxes.set_title(target_factor+" - "+str(jID)+" : 交易信号")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                if iArgName not in ("计算时点",):
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        PercentageFormatFun = np.vectorize(lambda x: ("%.2f%%" % (x*100, )))
        for iFactor in self._QSArgs.TestFactors:
            for jKey in ["所有交易", "多头交易", "空头交易"]:
                iHTML = iFactor+f" - {jKey} : "
                iOutput = self._Output["交易统计"][jKey][iFactor]
                iFormat = formatTradeSummary(iOutput)
                if iOutput.shape[0]>=14:
                    iFormat.iloc[13:] = PercentageFormatFun(iOutput.iloc[13:, :].values)
                iFormat = iFormat.reset_index()
                iFormat.columns = ["Statistics"]+iFormat.columns.tolist()[1:]
                iHTML += iFormat.to_html(index=False, notebook=True)
                Pos = iHTML.find(">")
                HTML += ('<div style="float:left;width:33%%;overflow:scroll">%s</div>' % (iHTML[:Pos]+' align="center"'+iHTML[Pos:],))
            #iOutput = self._Output["统计数据"][iFactor]
            #iHTML = iFactor+" - 净值统计 : "
            #iHTML += formatStrategySummary(iOutput).to_html()
            #Pos = iHTML.find(">")
            #HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
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
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
        FactorOrder = Dict(key_trait=Str(), value_trait=Enum("降序", "升序"), arg_type="ArgDict", label="排序方向", order=1)
        #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
        CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=3)
        SampleDTs = List(dt.datetime, arg_type="DateList", label="样本时点", order=4)
        SummaryWindow = Float(np.inf, arg_type="Integer", label="统计窗口", order=5)
        MinSummaryWindow = Int(3, arg_type="Integer", label="最小统计窗口", order=6)
        GroupNum = Int(3, arg_type="Integer", label="分组数", order=7)
        LSClearGroups = ListInt(arg_type="Integer", label="多空平仓组", order=8)
        def __QS_initArgs__(self):
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._Owner._FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
            self.TestFactors.append(DefaultNumFactorList[0])
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ["价","Price","price"])
        
        @property
        def ObservedArgs(self):
            return super().ObservedArgs + ("测试因子",)

        @on_trait_change("TestFactors[]")
        def _on_TestFactors_changed(self, obj, name, old, new):
            self.FactorOrder = {iFactorName:self.FactorOrder.get(iFactorName, "降序") for iFactorName in self.TestFactors}
    
    def __init__(self, factor_table, name="分位数择时", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Output = {}
        SecurityIDs = self._FactorTable.getID()
        nDT, nSecurityID, nFactor = len(dts), len(SecurityIDs), len(self._QSArgs.TestFactors)
        self._Output["标的ID"] = SecurityIDs
        self._Output["因子符号"] = 2 * np.array([(self._QSArgs.FactorOrder.get(iFactor, "降序")=="升序") for iFactor in self._QSArgs.TestFactors]).astype(float) - 1
        self._Output["标的收益率"] = np.zeros(shape=(nDT, nSecurityID))
        self._Output["因子值"] = np.full(shape=(nDT, nSecurityID, nFactor), fill_value=np.nan)
        self._Output["最新信号"] = np.full(shape=(nSecurityID, nFactor), fill_value=np.nan)
        self._Output["策略信号"] = np.full(shape=(nDT, nSecurityID, nFactor, self._QSArgs.GroupNum), fill_value=np.nan)
        self._Output["策略收益率"] = np.full(shape=(nDT, nSecurityID, nFactor, self._QSArgs.GroupNum), fill_value=np.nan)
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        DTIdx = self._Model.DateTimeIndex
        _, nSecurityID, nFactor = self._Output["因子值"].shape
        if DTIdx>0:
            Price = self._FactorTable.readData(dts=self._Model.DateTimeSeries[DTIdx-1:DTIdx+1], ids=self._Output["标的ID"], factor_names=[self._QSArgs.PriceFactor]).iloc[0, :, :].values
            Return = Price[1] / Price[0] - 1
            Return[pd.isnull(Return)] = 0
            self._Output["标的收益率"][DTIdx, :] = Return
            Mask = pd.notnull(self._Output["最新信号"])
            if np.any(Mask):
                SecurityIdx, FactorIdx = np.where(Mask)
                self._Output["策略收益率"][DTIdx, SecurityIdx, FactorIdx, self._Output["最新信号"][Mask].astype(int)] = np.repeat(np.reshape(Return, (nSecurityID, 1)), nFactor, axis=1)[Mask]
        self._Output["因子值"][DTIdx] = self._FactorTable.readData(dts=self._Model.DateTimeSeries[DTIdx:DTIdx+1], ids=self._Output["标的ID"], factor_names=list(self._QSArgs.TestFactors)).iloc[:, 0, :].values
        if self._QSArgs.CalcDTs:
            if idt not in self._QSArgs.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self._QSArgs.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        if DTIdx+1<self._QSArgs.MinSummaryWindow: return 0
        StartIdx = int(max(0, DTIdx + 1 - self._QSArgs.SummaryWindow))
        FactorData = self._Output["因子值"][StartIdx:DTIdx+1] * self._Output["因子符号"]
        if self._QSArgs.SampleDTs:
            DTSeries = pd.Series(np.arange(0, DTIdx+1-StartIdx), index=self._Model.DateTimeSeries[StartIdx:DTIdx+1])
            FactorData = FactorData[sorted(DTSeries.loc[DTSeries.index.intersection(self._QSArgs.SampleDTs)])]
        if FactorData.shape[0]<self._QSArgs.GroupNum: return 0
        Quantiles = np.nanpercentile(FactorData, np.arange(1, self._QSArgs.GroupNum+1) / self._QSArgs.GroupNum * 100, axis=0)
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
        self._Output["因子值"] = {iFactor: pd.DataFrame(self._Output["因子值"][:, :, i], index=DTs, columns=self._Output["标的ID"]) for i, iFactor in enumerate(self._QSArgs.TestFactors)}
        Groups = np.arange(1, self._QSArgs.GroupNum+1).astype(str)
        Signal = self._Output["策略信号"]
        SignalReturn = self._Output["策略收益率"]
        self._Output["策略信号"] = {}
        self._Output["策略收益率"] = {}
        self._Output["策略净值"] = {}
        self._Output["统计数据"] = {}
        for i, iFactor in enumerate(self._QSArgs.TestFactors):
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
                ijSignalReturn = pd.DataFrame(SignalReturn[:, j, i, :], index=DTs, columns=Groups).reindex(index=self._Output["策略信号"][iFactor][jID].index)
                # 计算多空组合收益率
                #ijSignalReturn["L-S"] = ijSignalReturn.iloc[:, 0].fillna(0) - ijSignalReturn.iloc[:, -1].fillna(0)
                ijLSSignal = ijSignal.iloc[:, 0].where(pd.notnull(ijSignal.iloc[:, 0]), -ijSignal.iloc[:, -1])
                for k in self._QSArgs.LSClearGroups:
                    ijLSSignal = (ijSignal.iloc[:, k] * 0).where(pd.notnull(ijSignal.iloc[:, k]), ijLSSignal)
                if ijDTs.shape[0]>0:
                    ijLSNV, _, _ = testTimingStrategy(ijSignal.loc[ijDTs[0]:].values.reshape((-1, 1)), self._Output["标的净值"].loc[ijDTs[0]:].iloc[:, j].values.reshape((-1, 1)))
                    ijLSReturn = ijLSNV[:, 0] / np.r_[ijLSNV[:1, 0], ijLSNV[:-1, 0]] - 1
                    ijLSReturn[np.isinf(ijLSReturn)] = 0
                    ijSignalReturn["L-S"] = ijLSReturn
                else:
                    ijSignalReturn["L-S"] = 0
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
        if target_factor is None: target_factor = self._QSArgs.TestFactors[0]
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
        if len(self._QSArgs.ArgNames)>0:
            HTML = "参数设置: "
            HTML += '<ul align="left">'
            for iArgName in self._QSArgs.ArgNames:
                if iArgName not in ("计算时点", "样本时点"):
                    HTML += "<li>"+iArgName+": "+str(self.Args[iArgName])+"</li>"
                elif self.Args[iArgName]:
                    HTML += "<li>"+iArgName+": 自定义时点</li>"
                else:
                    HTML += "<li>"+iArgName+": 所有时点</li>"
            HTML += "</ul>"
        else:
            HTML = ""
        FloatFormatFun = np.vectorize(lambda x: ("%.4f" % (x, )))
        for iFactor in self._QSArgs.TestFactors:
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