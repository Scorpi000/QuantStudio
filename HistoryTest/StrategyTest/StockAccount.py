# coding=utf-8
"""股票账户"""
import os
import datetime as dt
from copy import deepcopy

import pandas as pd
import numpy as np
from traits.api import Enum, Either, List, ListStr, Int, Float, Str, Bool, Dict, Instance, on_trait_change

from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio.Tools import DateTimeFun
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.HistoryTest.StrategyTest.StrategyTestModule import Account, cutDateTime

def _getDefaultNontradableIDFilter(ft=None, nonbuyable=True, qs_env=None):
    if ft is None:
        return None
    if nonbuyable:
        DefaultNontradableIDFilter = readNestedDictFromHDF5(qs_env.SysArgs['LibPath']+os.sep+'IDFilter.hdf5', "/默认限买条件")
    else:
        DefaultNontradableIDFilter = readNestedDictFromHDF5(qs_env.SysArgs['LibPath']+os.sep+'IDFilter.hdf5', "/默认限卖条件")
    if DefaultNontradableIDFilter is not None:
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(DefaultNontradableIDFilter, ft.FactorNames)
        if CompiledIDFilterStr is not None:
            return DefaultNontradableIDFilter
    AllFactorNames = set(ft.FactorNames)
    if {'交易状态','涨跌停'}.issubset(AllFactorNames):
        if nonbuyable:
            DefaultNontradableIDFilter = "(@交易状态==0) | (@交易状态==-2) | pd.isnull(@交易状态) | (@涨跌停==1)"
        else:
            DefaultNontradableIDFilter = "(@交易状态==0) | (@交易状态==-2) | pd.isnull(@交易状态) | (@涨跌停==-1)"
    elif '交易状态' in AllFactorNames:
        DefaultNontradableIDFilter = "(@交易状态==0) | (@交易状态==-2) | pd.isnull(@交易状态)"
    elif {'是否在市','涨跌停'}.issubset(AllFactorNames):
        if nonbuyable:
            DefaultNontradableIDFilter = "(@是否在市!=1) | (@涨跌停==1)"
        else:
            DefaultNontradableIDFilter = "(@是否在市!=1) | (@涨跌停==-1)"
    elif '是否在市' in AllFactorNames:
        DefaultNontradableIDFilter = "(@是否在市!=1)"
    elif '涨跌停' in AllFactorNames:
        if nonbuyable:
            DefaultNontradableIDFilter = "(@涨跌停==1)"
        else:
            DefaultNontradableIDFilter = "(@涨跌停==-1)"
    else:
        DefaultNontradableIDFilter = None
    return DefaultNontradableIDFilter


class _TradeLimit(__QS_Object__):
    """交易限制"""
    LimitIDFilter = Str(arg_type="IDFilter", label="禁止条件", order=0)
    TradeFee = Float(0.003, arg_type="Double", label="交易费率", order=1)
    MinUnit = Int(0, arg_type="Integer", label="最小单位", order=2)
    MarketOrderVolumeLimit = Float(0.1, arg_type="Double", label="市价单成交量限比", order=3)
    LimitOrderVolumeLimit = Float(0.1, arg_type="Double", label="限价单成交量限比", order=4)
    def __init__(self, direction, sys_args={}, **kwargs):
        self._Direction = direction
        return super().__init__(sys_args=sys_args, **kwargs)
class _AdjustFactorMap(__QS_Object__):
    """复权因子映照"""
    def __init__(self, adjust_ft=None, sys_args={}, **kwargs):
        self._AdjustFT = adjust_ft
        return super().__init__(sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        if self._AdjustFT is not None:
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._AdjustFT.getFactorMetaData(key="DataType")))
            DefaultNumFactorList.insert(0, None)
        else:
            DefaultNumFactorList = [None]
        self.add_trait("AdjustFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="复权因子", order=0))
        self.add_trait("StockDividend", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="每股送转", order=1))
        self.add_trait("CashDividend", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="每股派息", order=2))
        self.add_trait("StockPayDate", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="红股上市日", order=3))
        self.add_trait("CashPayDate", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="派息日", order=4))

class _BarFactorMap(__QS_Object__):
    """Bar 因子映照"""
    def __init__(self, market_ft, sys_args={}, **kwargs):
        self._MarketFT = market_ft
        return super().__init__(sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._MarketFT.getFactorMetaData(key="DataType")))
        DefaultNumFactorList.insert(0, None)
        self.add_trait("Open", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="开盘价", order=0))
        self.add_trait("High", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="最高价", order=1))
        self.add_trait("Low", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="最低价", order=2))
        self.add_trait("Last", Enum(*DefaultNumFactorList[1:], arg_type="SingleOption", label="最新价", order=3))
        self.add_trait("Vol", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="成交量", order=4))
        self.add_trait("TradePrice", Enum(*DefaultNumFactorList[1:], arg_type="SingleOption", label="成交价", order=5))
        self.TradePrice = self.Last = searchNameInStrList(DefaultNumFactorList[1:], ['新','收','Last','last','close','Close'])

# 基于 Bar 数据的股票账户
# 行情因子表: 开盘价(非必需), 最高价(非必需), 最低价(非必需), 最新价, 成交量(非必需). 最新价用于记录账户价值变化; 成交价: 用于模拟市价单的成交.
# 复权因子表: 复权因子或者每股送转(日期索引为股权登记日), 每股派息(税后, 日期索引为股权登记日), 派息日(日期索引为股权登记日), 红股上市日(日期索引为股权登记日)
# 市价单根据当前时段的成交价和成交量的情况成交; 限价单根据最高价, 最低价和成交量的情况成交, 假定成交量在最高价和最低价之间的分布为均匀分布, 如果没有指定最高价和最低价, 则最高价和最低价相等且等于成交价
# TODO: 加入交易状态, 涨跌停信息
class TimeBarAccount(Account):
    """基于 Bar 数据的股票账户"""
    Delay = Bool(True, arg_type="Bool", label="交易延迟", order=2)
    TargetIDs = ListStr(arg_type="IDList", label="目标ID", order=3)
    BuyLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="买入限制", order=4)
    SellLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="卖出限制", order=5)
    MarketFactorMap = Instance(_BarFactorMap, arg_type="ArgObject", label="行情因子", order=6)
    AdjustFactorMap = Instance(_AdjustFactorMap, arg_type="ArgObject", label="复权因子", order=7)
    def __init__(self, market_ft, adjust_ft=None, sys_args={}, **kwargs):
        # 继承自 Account 的属性
        #self._Cash = np.array([])# 剩余现金, 等于时间点长度+1, >=0
        #self._Debt = np.array([])# 负债, 等于时间点长度+1, >=0
        #self._CashRecord = pd.DataFrame(columns=["时间点", "现金流"])# 现金流记录, pd.DataFrame(columns=["日期", "时间点", "现金流"]), 现金流入为正, 现金流出为负
        #self._DebtRecord = pd.DataFrame(columns=["时间点", "融资"])# 融资记录, pd.DataFrame(columns=["日期", "时间点", "融资"]), 增加负债为正, 减少负债为负
        #self._TradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])# 交易记录
        self._IDs = []# 本账户支持交易的证券 ID, []
        self._Position = np.array([])# 仓位, array(index=dts+1, columns=self._IDs)
        self._PositionAmount = np.array([])# 持仓金额, array(index=dts+1, columns=self._IDs)
        self._EquityRecord = pd.DataFrame(columns=["时间点", "ID", "进出数量", "进出金额", "备注"])# 证券进出流水记录, 提取为负, 增加为正
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])# 当前接收到的订单
        self._LastPrice = np.array([])# 最新价, array(len(self._IDs))
        self._MarketFT = market_ft# 行情因子表对象
        self._AdjustFT = adjust_ft# 复权因子表对象
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = "StockAccount"
    def __QS_initArgs__(self):
        self.MarketFactorMap = _BarFactorMap(self._MarketFT)
        self.AdjustFactorMap = _AdjustFactorMap(self._AdjustFT)
        self.BuyLimit = _TradeLimit(direction="Buy")
        self.SellLimit = _TradeLimit(direction="Sell")
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        self._IDs = list(self.TargetIDs)
        if not self._IDs: self._IDs = list(self._MarketFT.getID(ifactor_name=self.MarketFactorMap.TradePrice))
        nDT, nID = len(dts), len(self._IDs)
        #self._Cash = np.zeros(nDT+1)
        #self._Debt = np.zeros(nDT+1)
        self._Position = np.zeros((nDT+1, nID))
        self._PositionAmount = np.zeros((nDT+1, nID))
        self._EquityRecord = pd.DataFrame(columns=["时间点", "ID", "进出数量", "进出金额", "备注"])
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        self._LastPrice = None# 最新价
        if self.AdjustFactorMap.AdjustFactor is not None:
            self._AdjustFactor = np.ones(nID)# 当前的复权因子
            Rslt += (self._MarketFT, self._AdjustFT)
        elif self.AdjustFactorMap.CashDividend is not None:
            self._CashDividend = pd.DataFrame(index=self._IDs)# 现金股息, columns=[派息日]
            self._StockDividend = pd.DataFrame(index=self._IDs)# 红股, columns=[红股上市日]
            Rslt += (self._MarketFT, self._AdjustFT)
        else:
            self._AdjustFT = None
            Rslt += (self._MarketFT, )
        self._iTradingRecord = []# 暂存的交易记录
        self._iDate = None# 当前所处的日期
        return Rslt
    def __QS_move__(self, idt, *args, **kwargs):
        super().__QS_move__(idt, *args, **kwargs)
        # 更新当前的账户信息
        self._LastPrice = self._MarketFT.readData(factor_names=[self.MarketFactorMap.Last], ids=self._IDs, dts=[idt]).iloc[0, 0].values
        iIndex = self._Model.DateTimeIndex
        self._Position[iIndex+1] = self._Position[iIndex]
        iDate = idt.date()
        if iDate!=self._iDate:# 进入新交易日
            self._addDividendInfo(self._iDate)# 更新上一个交易日的分红信息
            self._iDate = iDate
            self._handleDividend(iDate)# 处理分红送转
        if self.Delay:# 撮合成交
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)),
                                         columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
        else:
            TradingRecord = self._iTradingRecord
        self._PositionAmount[iIndex+1] = self._Position[iIndex+1]*self._LastPrice
        return TradingRecord
    def __QS_after_move__(self, idt, *args, **kwargs):
        super().__QS_after_move__(self, idt, *args, **kwargs)
        if not self.Delay:# 撮合成交
            iIndex = self._Model.DateTimeIndex
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)),
                                         columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
            self._iTradingRecord = TradingRecord
            self._PositionAmount[iIndex+1] = self._Position[iIndex+1]*self._LastPrice
        return 0
    def __QS_end__(self):
        super().__QS_end__()
        self._Output["持仓"] = self.getPositionSeries()
        self._Output["持仓金额"] = self.getPositionAmountSeries()
        self._Output["证券进出记录"] = self._EquityRecord
        return 0
    # 添加新的分红送转信息
    def _addDividendInfo(self, idate):
        if (self._AdjustFT is None) or (self.AdjustFactorMap.AdjustFactor is not None): return 0
        Position = self.Position
        IDs = Position[Position!=0].index
        if IDs.shape[0]==0: return 0
        iDT = dt.datetime.combine(idate, dt.time(23,59,59,999999))
        Dividend = self._AdjustFT.readData(factor_names=[self.AdjustFactorMap.CashDividend, self.AdjustFactorMap.StockDividend], ids=IDs, dts=[iDT]).iloc[:, 0, :]
        CashDvd = Dividend.iloc[:, 0]
        IDs = list(CashDvd[CashDvd>0].index)
        if IDs:# 有现金红利
            CashDvdDate = self._AdjustFT.readData(factor_names=[self.AdjustFactorMap.CashPayDate], ids=IDs, dts=[iDT]).iloc[0, 0, :]
            for i, iID in IDs:
                iDate = CashDvdDate.iloc[i]
                if pd.notnull(iDate):
                    iDate = dt.date(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]))
                    self._CashDividend[iDate] = self._CashDividend.get(iDate, 0)
                    self._CashDividend.loc[iID, iDate] += CashDvd[iID]*Position[iID]
        StockDvd = Dividend.iloc[:, 1]
        IDs = list(StockDvd[StockDvd>0].index)
        if IDs:# 有红股
            StockDvdDate = self._AdjustFT.readData(factor_names=[self.AdjustFactorMap.StockPayDate], ids=IDs, dts=[iDT]).iloc[0, 0, :]
            for i, iID in IDs:
                iDate = StockDvdDate.iloc[i]
                if pd.notnull(iDate):
                    iDate = dt.date(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]))
                    self._StockDividend[iDate] = self._StockDividend.get(iDate, 0)
                    self._StockDividend.loc[iID, iDate] += int(StockDvd[iID]*Position[iID])
        return 0
    # 处理分红送转
    def _handleDividend(self, idate):
        if self._AdjustFT is None: return 0
        AdjustFactor = self.AdjustFactorMap.AdjustFactor
        if self.AdjustFactorMap.AdjustFactor is not None:
            AdjustFactor = self._AdjustFT.readData(factor_names=[self.AdjustFactorMap.AdjustFactor], ids=self._IDs, dts=[dt.datetime.combine(idate, dt.time(23,59,59,999999))]).values.squeeze()
            AdjustFactor = np.where(np.isnan(AdjustFactor), self._AdjustFactor, AdjustFactor)
            iIndex = self._Model.DateTimeIndex
            self._Position[iIndex+1] *= AdjustFactor/self._AdjustFactor
            self._AdjustFactor = AdjustFactor
            return 0
        if idate in self._CashDividend:
            iDividend = self._CashDividend.pop(idate)
            iDividend = iDividend[iDividend>0]
            for i, iID in enumerate(iDividend.index):
                self.addCash(iDividend.iloc[i], "%s: 股息发放" % iID)
        if idate in self._StockDividend:
            iDividend = self._StockDividend.pop(idate)
            iDividend = iDividend[iDividend>0]
            for i, iID in enumerate(iDividend.index):
                self.addEquity(iID, iDividend.iloc[i], "%s: 红股发放" % iID)
        return 0
    # 提取证券
    def fetchEquity(self, target_id, num, remark=""):
        iIndex = self._Model.DateTimeIndex
        TargetIDIndex = self._IDs.index(target_id)
        Position = self._Position[iIndex, TargetIDIndex]
        EquityNum = min((num, Position))
        if EquityNum<=0: return 0
        self._Position[iIndex, TargetIDIndex] -= Position - EquityNum
        AmountDelta = EquityNum / Position * self._PositionAmount[iIndex, TargetIDIndex]
        self._PositionAmount[iIndex, TargetIDIndex] -= AmountDelta
        self._EquityRecord.loc[self._EquityRecord.shape[0]] = (self._Model.DateTime, target_id, -EquityNum, -AmountDelta, remark)
        return EquityNum
    # 增加证券
    def addEquity(self, target_id, num, remark=""):
        if num<=0: return 0
        iIndex = self._Model.DateTimeIndex
        TargetIDIndex = self._IDs.index(target_id)
        OldPosition = self._Position[iIndex, TargetIDIndex]
        self._Position[iIndex, TargetIDIndex] = OldPosition + num
        if OldPosition>0:
            AmountDelta = self._PositionAmount[iIndex, TargetIDIndex] / OldPosition * num
        else:
            LastPrice = self.LastPrice.get(target_id, np.nan)
            AmountDelta = LastPrice * num
        self._PositionAmount[iIndex, TargetIDIndex] = self._PositionAmount[iIndex, TargetIDIndex] + AmountDelta
        self._EquityRecord.loc[self._EquityRecord.shape[0]] = (self._Model.DateTime, target_id, num, AmountDelta, remark)
        return 0
    # 当前账户价值, float
    @property
    def AccountValue(self):
        return super().AccountValue + np.nansum(self._PositionAmount[self._Model.DateTimeIndex+1])
    # 当前账户的持仓数量, Series(index=[ID])
    @property
    def Position(self):
        return pd.Series(self._Position[self._Model.DateTimeIndex+1], index=self._IDs)
    # 当前账户的持仓金额, Series(index=[ID])
    @property
    def PositionAmount(self):
        return pd.Series(self._PositionAmount[self._Model.DateTimeIndex+1], index=self._IDs)
    # 本账户支持交易的证券 ID, [ID]
    @property
    def IDs(self):
        return self._IDs
    # 当前账户中还未成交的订单, DataFrame(index=[int], columns=["ID", "数量", "目标价"])
    @property
    def Orders(self):
        return self._Orders
    # 当前最新价, Series(index=[ID])
    @property
    def LastPrice(self):
        return pd.Series(self._LastPrice, index=self._IDs)
    # 当前的 Bar 数据, DataFrame(float, index=[ID], columns=["开盘价", "最高价", "最低价", "最新价", "成交量"])
    @property
    def Bar(self):
        return self._MarketFT.readData(factor_names=[self.MarketFactorMap.Open, self.MarketFactorMap.High, self.MarketFactorMap.Low, self.MarketFactorMap.Last, self.MarketFactorMap.Vol], 
                                       dts=[self._Model.DateTime], ids=self._IDs).iloc[:,0,:]
    # 获取持仓数量的历史序列, 以时间点为索引, 返回: DataFrame(float, index=[时间点], columns=[ID])
    def getPositionSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.DataFrame(self._Position[1:self._Model.DateTimeIndex+2], index=self._Model.DateTimeSeries, columns=self._IDs)
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取持仓金额的历史序列, 以时间点为索引, 返回: DataFrame(float, index=[时间点], columns=[ID])
    def getPositionAmountSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.DataFrame(self._PositionAmount[1:self._Model.DateTimeIndex+2], index=self._Model.DateTimeSeries, columns=self._IDs)
        Data = cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
        return Data.fillna(value=0.0)
    # 获取账户价值的历史序列, 以时间点为索引, 返回: Series(float, index=[时间点], columns=[ID])
    def getAccountValueSeries(self, dts=None, start_dt=None, end_dt=None):
        CashSeries = self.getCashSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        PositionAmountSeries = self.getPositionAmountSeries(dts=dts, start_dt=start_dt, end_dt=end_dt).sum(axis=1)
        DebtSeries = self.getDebtSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        return CashSeries + PositionAmountSeries - DebtSeries
    # 基本的下单函数, 执行给定数量的证券委托单
    # 单个订单: target_id: 目标证券 ID, num: 待买卖的数量, target_price: nan 表示市价单
    # 组合订单: combined_order: DataFrame(columns=[ID, 数量, 目标价])
    def order(self, target_id=None, num=1, target_price=np.nan, combined_order=None):
        if target_id is not None:
            if (num>0) and (self.BuyLimit.MinUnit>0):
                num = np.fix(num / self.BuyLimit.MinUnit) * self.BuyLimit.MinUnit
            elif (num<0) and (self.SellLimit.MinUnit>0):
                num = np.fix(num / self.SellLimit.MinUnit) * self.SellLimit.MinUnit
            self._Orders.loc[self._Orders.shape[0]] = (target_id, num, target_price)
            return (target_id, num, target_price)
        if combined_order is not None:
            if self.BuyLimit.MinUnit>0:
                Mask = (combined_order["数量"]>0)
                combined_order["数量"][Mask] = np.fix(combined_order["数量"][Mask] / self.BuyLimit.MinUnit) * self.BuyLimit.MinUnit
            if self.SellLimit.MinUnit>0:
                Mask = (combined_order["数量"]<0)
                combined_order["数量"][Mask] = np.fix(combined_order["数量"][Mask] / self.SellLimit.MinUnit) * self.SellLimit.MinUnit
            combined_order.index = np.arange(self._Orders.shape[0], self._Orders.shape[0]+combined_order.shape[0])
            self._Orders = self._Orders.append(combined_order)
        return combined_order
    # 撤销订单, order_ids 是订单在 self.Orders 中的 index
    def cancelOrder(self, order_ids):
        self._Orders = self._Orders.loc[self._Orders.index.difference(set(order_ids))]
        self._Orders.sort_index(axis=0, inplace=True)
        self._Orders.index = np.arange(self._Orders.shape[0])
        return 0
    # 更新账户信息
    def _updateAccount(self, cash_changed, position):
        iIndex = self._Model.DateTimeIndex
        DebtDelta = max((- cash_changed - self.Cash, 0))
        if self.DeltLimit>0:
            self._Debt[iIndex+1] += DebtDelta
            if DebtDelta>0: self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, DebtDelta)
            self._Cash[iIndex+1] -= min((- cash_changed, self.Cash))
        else:
            if DebtDelta>0: self._Cash[iIndex+1] = 0
            else: self._Cash[iIndex+1] -= min((- cash_changed, self.Cash))
        position[position.abs()<1e-6] = 0.0
        self._Position[iIndex+1] = position.values
        return 0
    # 撮合成交订单
    def _matchOrder(self, idt):
        MarketOrderMask = pd.isnull(self._Orders["目标价"])
        MarketOrders = self._Orders[MarketOrderMask]
        LimitOrders = self._Orders[~MarketOrderMask]
        MarketOrders = MarketOrders.groupby(by=["ID"]).sum()["数量"]
        if LimitOrders.shape[0]>0:
            LimitOrders = LimitOrders.groupby(by=["ID", "目标价"]).sum()
            LimitOrders.sort_index(ascending=False)
            LimitOrders = LimitOrders.reset_index(level=1)
        else:
            LimitOrders = pd.DataFrame(columns=["ID","数量","目标价"])
        # 先执行卖出交易
        TradingRecord = self._matchMarketSellOrder(idt, MarketOrders[MarketOrders<0])
        iTradingRecord, LimitSellOrders = self._matchLimitSellOrder(idt, LimitOrders[LimitOrders["数量"]<0])
        TradingRecord.extend(iTradingRecord)
        # 再执行买入交易
        TradingRecord.extend(self._matchMarketBuyOrder(idt, MarketOrders[MarketOrders>0]))
        iTradingRecord, LimitBuyOrders = self._matchLimitBuyOrder(idt, LimitOrders[LimitOrders["数量"]>0])
        TradingRecord.extend(iTradingRecord)
        self._Orders = LimitSellOrders.append(LimitBuyOrders)
        self._Orders.index = np.arange(self._Orders.shape[0])
        return TradingRecord
    # 撮合成交卖出市价单
    # 以成交价完成成交, 成交量满足卖出限制要求
    # 未成交的市价单自动撤销
    def _matchMarketSellOrder(self, idt, sell_orders):
        sell_orders = -sell_orders
        Position = self.Position
        sell_orders = np.minimum(sell_orders, Position.loc[sell_orders.index])# 卖出数量不能超过当前持仓数量
        sell_orders = sell_orders[sell_orders>0]
        if sell_orders.shape[0]==0: return []
        CashChanged = 0.0
        TradingRecord = []
        IDs = list(sell_orders.index)
        SellPrice = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketFactorMap.TradePrice]).iloc[0, 0]
        SellLimit = pd.Series(np.zeros(sell_orders.shape[0])+np.inf, index=IDs)
        SellLimit[pd.isnull(SellPrice)] = 0.0# 成交价缺失的不能卖出
        if self.SellLimit.LimitIDFilter: SellLimit[self._MarketFT.getIDMask(idt, ids=IDs, id_filter_str=self.SellLimit.LimitIDFilter)] = 0.0# 满足禁止条件的不能卖出, TODO
        if self.MarketFactorMap.Vol: SellLimit = np.minimum(SellLimit, self._MarketFT.readData(factor_names=[self.MarketFactorMap.Vol], ids=IDs, dts=[idt]).iloc[0, 0] * self.SellLimit.MarketOrderVolumeLimit)# 成交量限制
        SellNums = np.minimum(SellLimit, sell_orders)
        if self.SellLimit.MinUnit!=0.0: SellNums = (SellNums / self.SellLimit.MinUnit).astype("int") * self.SellLimit.MinUnit# 最小交易单位限制
        SellAmounts = SellNums * SellPrice
        SellFees = SellAmounts * self.SellLimit.TradeFee
        Mask = (SellNums>0)
        Position[SellNums.index] -= SellNums
        CashChanged += SellAmounts.sum() - SellFees.sum()
        TradingRecord.extend(list(zip([idt]*Mask.sum(), SellNums[Mask].index, -SellNums[Mask], SellPrice[Mask], SellFees[Mask], (SellAmounts-SellFees)[Mask], ["close"]*Mask.sum())))
        if TradingRecord: self._updateAccount(CashChanged, Position)# 更新账户信息
        return TradingRecord
    # 撮合成交买入市价单
    # 以成交价完成成交, 成交量满足买入限制要求
    # 未成交的市价单自动撤销
    def _matchMarketBuyOrder(self, idt, buy_orders):
        if buy_orders.shape[0]==0: return []
        CashChanged = 0.0
        TradingRecord = []
        IDs = list(buy_orders.index)
        BuyPrice = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketFactorMap.TradePrice]).iloc[0, 0]
        BuyLimit = pd.Series(np.zeros(buy_orders.shape[0])+np.inf, index=IDs)
        BuyLimit[pd.isnull(BuyPrice)] = 0.0# 成交价缺失的不能买入
        if self.BuyLimit.LimitIDFilter: BuyLimit[self._MarketFT.getIDMask(idt, ids=IDs, id_filter_str=self.BuyLimit.LimitIDFilter)] = 0.0# 满足禁止条件的不能卖出
        if self.MarketFactorMap.Vol: BuyLimit = np.minimum(BuyLimit, self._MarketFT.readData(dts=[idt], factor_names=[self.MarketFactorMap.Vol], ids=IDs).iloc[0, 0] * self.BuyLimit.MarketOrderVolumeLimit)# 成交量限制
        BuyAmounts = np.minimum(BuyLimit*BuyPrice, buy_orders*BuyPrice)
        TotalBuyAmounts = BuyAmounts.sum()
        if (TotalBuyAmounts>0):
            BuyAmounts = min((TotalBuyAmounts * (1+self.BuyLimit.TradeFee), self.AvailableCash)) * BuyAmounts / TotalBuyAmounts / (1+self.BuyLimit.TradeFee)
            if self.BuyLimit.MinUnit!=0.0:
                BuyNums = (BuyAmounts / (self.BuyLimit.MinUnit*BuyPrice)).astype("int") * self.BuyLimit.MinUnit
                BuyAmounts = BuyNums * BuyPrice
            else:
                BuyNums = BuyAmounts / BuyPrice
            BuyFees = BuyAmounts * self.BuyLimit.TradeFee
            Mask = (BuyNums>0)
            Position = self.Position
            Position[BuyNums.index] += BuyNums
            CashChanged -= BuyAmounts.sum() + BuyFees.sum()
            TradingRecord.extend(list(zip([idt]*Mask.sum(), BuyNums[Mask].index, BuyNums[Mask], BuyPrice[Mask], BuyFees[Mask], -(BuyAmounts+BuyFees)[Mask], ["open"]*Mask.sum())))
        if TradingRecord: self._updateAccount(CashChanged, Position)
        return TradingRecord
    # 撮合成交卖出限价单
    # 如果最高价和最低价未指定, 则检验成交价是否优于目标价, 是就以目标价完成成交, 且成交量满足卖出限制中的限价单成交量限比, 否则无法成交
    # 如果指定了最高价和最低价, 则假设成交价服从[最低价, 最高价]中的均匀分布, 据此确定可完成成交的数量
    # 未成交的限价单继续保留
    def _matchLimitSellOrder(self, idt, sell_orders):
        if sell_orders.shape[0]==0: return ([], sell_orders)
        IDs = list(sell_orders.index.unique())
        if self.MarketFactorMap.Vol:
            Volume = self._MarketFT.readData(factor_names=[self.MarketFactorMap.Vol], ids=IDs, dts=[idt]).iloc[0, 0]*self.BuyLimit.LimitOrderVolumeLimit
        else:
            Volume = pd.Series(np.inf, index=IDs)
        if self.MarketFactorMap.High and self.MarketFactorMap.Low:
            High = self._MarketFT.readData(factor_names=[self.MarketFactorMap.High, self.MarketFactorMap.Low], ids=IDs, dts=[idt]).iloc[:, 0]
            Low, High = High.iloc[1], High.iloc[0]
        else:
            High = Low = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketFactorMap.TradePrice]).iloc[0, 0]
        Position = self.Position
        SellNums = np.zeros(Volume.shape)
        SellAmounts = np.zeros(Volume.shape)
        for i, iID in enumerate(IDs):
            iPosition = Position[iID]
            if iPosition<=0: continue
            iSellOrders = sell_orders.loc[iID]
            iVolume = ((High.iloc[i] - iSellOrders["目标价"]) / (High.iloc[i] - Low.iloc[i])).clip(0, 1) * Volume.iloc[i]
            for j, jNum in enumerate(reversed(iSellOrders["数量"])):
                j = iSellOrders.shape[0] - 1 - j
                ijSellNum = min((iVolume.iloc[j], -jNum, iPosition))
                if self.SellLimit.MinUnit!=0.0: ijSellNum = (ijSellNum / self.SellLimit.MinUnit).astype("int") * self.SellLimit.MinUnit
                SellNums[i] += ijSellNum
                SellAmounts[i] += ijSellNum*iSellOrders["目标价"].iloc[j]
                iPosition -= ijSellNum
                iSellOrders["数量"].iloc[j] += ijSellNum
                iVolume = iVolume.clip(0, iVolume.iloc[j]-ijSellNum)
            sell_orders.loc[iID] = iSellOrders
            Position[iID] = iPosition
        sell_orders = sell_orders[sell_orders["数量"]<0]
        sell_orders = sell_orders.reset_index().loc[:, ["ID", "数量", "目标价"]]
        SellFees = SellAmounts*self.SellLimit.TradeFee
        CashChanged = SellAmounts.sum() - SellFees.sum()
        Mask = (SellNums>0)
        TradingRecord = list(zip([idt]*Mask.sum(), Volume.index[Mask], -SellNums[Mask], (SellAmounts/SellNums)[Mask], SellFees[Mask], (SellAmounts-SellFees)[Mask], ["close"]*Mask.sum()))
        if TradingRecord: self._updateAccount(CashChanged, Position)
        return (TradingRecord, sell_orders)
    # 撮合成交买入限价单
    # 如果最高价和最低价未指定, 则检验成交价是否优于目标价, 是就以目标价完成成交, 且成交量满足买入限制要求
    # 如果指定了最高价和最低价, 则假设成交价服从[最低价, 最高价]中的均匀分布, 据此确定可完成成交的数量, 同时满足买入限制要求
    # 未成交的限价单继续保留
    def _matchLimitBuyOrder(self, idt, buy_orders):
        if buy_orders.shape[0]==0: return ([], buy_orders)
        buy_orders = buy_orders.sort_values(by=["目标价"], ascending=False)
        IDs = list(buy_orders.index.unique())
        if self.MarketFactorMap.Vol:
            Volume = self._MarketFT.readData(factor_names=[self.MarketFactorMap.Vol], ids=IDs, dts=[idt]).iloc[0, 0]*self.BuyLimit.LimitOrderVolumeLimit
        else:
            Volume = pd.Series(np.inf, index=IDs)
        if self.MarketFactorMap.High and self.MarketFactorMap.Low:
            High = self._MarketFT.readData(factor_names=[self.MarketFactorMap.High, self.MarketFactorMap.Low], ids=IDs, dts=[idt]).iloc[:, 0]
            Low, High = High.iloc[1], High.iloc[0]
        else:
            High = Low = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketFactorMap.TradePrice]).iloc[0, 0]
        TargetAmounts = (buy_orders["目标价"]*buy_orders["数量"]).groupby(level=0).sum()
        TotalAmount = TargetAmounts.sum()
        AllocatedAmounts = np.min((TotalAmount*(1+self.BuyLimit.TradeFee), self.AvailableCash))*TargetAmounts/TotalAmount
        Position = self.Position
        BuyNums = np.zeros(Volume.shape)
        BuyAmounts = np.zeros(Volume.shape)
        for i, iID in enumerate(IDs):
            iPosition = 0
            iAmount = AllocatedAmounts.loc[iID]
            iBuyOrders = buy_orders.loc[iID]
            iVolume = ((iBuyOrders["目标价"] - Low.iloc[i]) / (High.iloc[i] - Low.iloc[i])).clip(0, 1) * Volume.iloc[i]
            for j, jNum in enumerate(iBuyOrders["数量"]):
                ijBuyNum = min((iVolume.iloc[j], jNum, int(iAmount/iBuyOrders["目标价"].iloc[j])))
                if self.BuyLimit.MinUnit!=0.0:
                    ijBuyNum = (ijBuyNum / self.BuyLimit.MinUnit).astype("int") * self.BuyLimit.MinUnit
                BuyNums[i] += ijBuyNum
                BuyAmounts[i] += ijBuyNum*iBuyOrders["目标价"].iloc[j]
                iAmount -= ijBuyNum*iBuyOrders["目标价"].iloc[j]*(1+self.BuyLimit.TradeFee)
                iPosition += ijBuyNum
                iBuyOrders["数量"].iloc[j] -= ijBuyNum
                iVolume = iVolume.clip(0, iVolume.iloc[j]-ijBuyNum)
            buy_orders.loc[iID] = iBuyOrders
            Position[iID] += iPosition
        buy_orders = buy_orders[buy_orders["数量"]>0]
        buy_orders = buy_orders.reset_index().loc[:, ["ID", "数量", "目标价"]]
        Fees = BuyAmounts * self.BuyLimit.TradeFee
        CashChanged = - BuyAmounts.sum() - Fees.sum()
        Mask = (BuyNums>0)
        TradingRecord = list(zip([idt]*Mask.sum(), Volume.index[Mask], BuyNums[Mask], (BuyAmounts/BuyNums)[Mask], Fees[Mask], (-BuyAmounts-Fees)[Mask], ["close"]*Mask.sum()))
        if TradingRecord: self._updateAccount(CashChanged, Position)
        return (TradingRecord, buy_orders)


class _TickFactorMap(_BarFactorMap):
    """Tick 因子映照"""
    def __init__(self, market_ft, sys_args={}, **kwargs):
        self._MarketFT = market_ft
        return super().__init__(sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.add_trait("Bid", Str(arg_type="String", label="买盘价格因子", order=7))
        self.add_trait("BidVol", Str(arg_type="String", label="买盘数量因子", order=8))
        self.add_trait("Ask", Str(arg_type="String", label="卖盘价格因子", order=9))
        self.add_trait("AskVol", Str(arg_type="String", label="买盘数量因子", order=10))
# 基于 Tick 数据的股票账户
# 行情因子表: 开盘价, 最高价, 最低价, 最新价, 成交价, 成交量, 买价1..., 卖价1..., 买量1..., 卖量1... 最新价用于记录账户价值变化; 买卖盘: 用于模拟市价单的成交
# 复权因子表: 复权因子或者每股送转(日期索引为股权登记日), 每股派息(税后, 日期索引为股权登记日), 派息日(日期索引为股权登记日), 红股上市日(日期索引为股权登记日)
# 市价单根据当前时点买卖盘的情况成交; 限价单根据最高价, 最低价和成交量的情况成交, 假定成交量在最高价和最低价之间的分布为均匀分布, 如果没有指定最高价和最低价, 则以成交价作为最高价和最低价
class TickAccount(TimeBarAccount):
    """基于 Tick 数据的股票账户"""
    def __QS_initArgs__(self):
        self.add_trait("MarketFactorMap", Instance(_TickFactorMap, arg_type="ArgObject", label="行情因子", order=6))
        self.MarketFactorMap = _TickFactorMap(self._MarketFT)
        self.AdjustFactorMap = _AdjustFactorMap(self._AdjustFT)
        self.BuyLimit = _TradeLimit(direction="Buy")
        self.SellLimit = _TradeLimit(direction="Sell")
    def __QS_start__(self, dts=None, dates=None, times=None):
        super().__QS_start__(dts=dts, dates=dates, times=times)
        self._Bid = []# 买盘价格因子
        self._BidVol = []# 买盘挂单量因子
        self._Ask = []# 卖盘价格因子
        self._AskVol = []# 卖盘挂单量因子
        for iFactorName in self._MarketFT.FactorNames:
            if iFactorName.find(self.MarketFactorMap.Bid)!=-1: self._Bid.append(iFactorName)
            elif iFactorName.find(self.MarketFactorMap.BidVol)!=-1: self._BidVol.append(iFactorName)
            elif iFactorName.find(self.MarketFactorMap.Ask)!=-1: self._Ask.append(iFactorName)
            elif iFactorName.find(self.MarketFactorMap.AskVol)!=-1: self._AskVol.append(iFactorName)
        self._Bid.sort()
        self._BidVol.sort()
        self._Ask.sort()
        self._AskVol.sort()
        self._Depth = len(self._Bid)# 行情深度
        return 0
    # 撮合成交卖出市价单
    def _matchMarketSellOrder(self, idt, sell_orders):
        sell_orders = -sell_orders
        Position = self.Position
        sell_orders = np.minimum(sell_orders, Position.loc[sell_orders.index])# 卖出数量不能超过当前持仓数量
        sell_orders = sell_orders[sell_orders>0]
        if sell_orders.shape[0]==0: return []
        Bid = self._MarketFT.readData(factor_names=self._Bid+self._BidVol, ids=list(sell_orders.index), dts=[idt]).iloc[:, 0, :]
        BidVol = Bid.iloc[:, -self._Depth:]
        Bid = Bid.iloc[:, :self._Depth]
        BidVol.fillna(0, inplace=True)
        BidVol = (BidVol.values - np.clip(BidVol.values.cumsum(axis=1)-sell_orders.values.reshape((sell_orders.shape[0], 1)), 0, np.inf)).clip(0, np.inf)
        SellNums = np.nansum(BidVol, axis=1)
        Mask = (SellNums>0)
        Bid = Bid[Mask]
        BidVol = BidVol[Mask]
        SellNums = SellNums[Mask]
        SellAmounts = (Bid*BidVol).sum(axis=1)
        SellFees = SellAmounts * self.SellLimit.TradeFee
        Position[sell_orders.index] -= SellNums
        CashChanged = SellAmounts.sum() - SellFees.sum()
        TradingRecord = list(zip([idt]*SellNums.shape[0], SellAmounts.index, -SellNums, SellAmounts/SellNums, SellFees, SellAmounts-SellFees, ["close"]*SellNums.shape[0]))
        if TradingRecord: self._updateAccount(CashChanged, Position)# 更新账户信息
        return TradingRecord
    # 撮合成交买入市价单
    def _matchMarketBuyOrder(self, idt, buy_orders):
        if buy_orders.shape[0]==0: return []
        # 分配可用资金
        Ask = self._MarketFT.readData(factor_names=self._Ask+self._AskVol, ids=list(buy_orders.index), dts=[idt]).iloc[:, 0, :]
        AskVol = Ask.iloc[:, -self._Depth:]
        Ask = Ask.iloc[:, :self._Depth]
        AskVol.fillna(0, inplace=True)
        MatchedAskVol = (AskVol.values - np.clip(AskVol.values.cumsum(axis=1)-buy_orders.values.reshape((buy_orders.shape[0], 1)), 0, np.inf)).clip(0, np.inf)
        TargetBuyAmounts = (Ask*MatchedAskVol).sum(axis=1)
        TargetTotalAmount = TargetBuyAmounts.sum() * (1+self.BuyLimit.TradeFee)
        if TargetTotalAmount==0: return []
        Position = self.Position
        BuyAmounts = TargetBuyAmounts / TargetTotalAmount * min((TargetTotalAmount, self.AvailableCash))# 买入总额不能超过当前的可用资金
        AskAmount = Ask.values*AskVol.values
        MatchedAskAmount = (AskAmount - np.clip(AskAmount.cumsum(axis=1)-BuyAmounts.values.reshape((BuyAmounts.shape[0], 1)), 0, np.inf)).clip(0, np.inf)
        BuyNums = (MatchedAskAmount/Ask).astype("int")
        BuyAmounts = (BuyNums*Ask).sum(axis=1)
        BuyNums = BuyNums.sum(axis=1)
        Mask = (BuyNums>0)
        BuyNums = BuyNums[Mask]
        BuyAmounts = BuyAmounts[Mask]
        BuyFees = BuyAmounts*self.BuyLimit.TradeFee
        Position[BuyNums.index] += BuyNums
        CashChanged = -(BuyAmounts.sum() + BuyFees.sum())
        TradingRecord = list(zip([idt]*BuyNums.shape[0], BuyNums.index, BuyNums, BuyAmounts/BuyNums, BuyFees, -(BuyAmounts+BuyFees), ["open"]*BuyNums.shape[0]))
        if TradingRecord: self._updateAccount(CashChanged, Position)
        return TradingRecord

if __name__=="__main__":
    testC = _TradeLimit(direction="Buy")
    testC.setArgs()