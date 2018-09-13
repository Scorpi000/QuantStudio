# coding=utf-8
"""基于净值数据的账户(TODO)"""
import os
import datetime as dt

import pandas as pd
import numpy as np
from traits.api import Enum, Either, List, ListStr, Int, Float, Str, Bool, Dict, Instance, on_trait_change

from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList, match2Series
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio.Tools import DateTimeFun
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.BackTest.Strategy.StrategyModule import Account, cutDateTime

class _TradeLimit(__QS_Object__):
    """交易限制"""
    LimitIDFilter = Str(arg_type="IDFilter", label="禁止条件", order=0)
    TradeFee = Float(0.003, arg_type="Double", label="交易费率", order=1)
    def __init__(self, direction, sys_args={}, **kwargs):
        self._Direction = direction
        return super().__init__(sys_args=sys_args, **kwargs)

# 基于净值数据的账户
# ft: 提供净值或者收益率数据的因子表, 时间频率任意
class NAVAccount(Account):
    """净值账户"""
    Delay = Bool(True, arg_type="Bool", label="交易延迟", order=2)
    TargetIDs = ListStr(arg_type="IDList", label="目标ID", order=3)
    NAV = Enum(None, arg_type="SingleOption", label="复权净值", order=4)
    Return = Enum(None, arg_type="SingleOption", label="收益率", order=5)
    BuyLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="买入限制", order=6)
    SellLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="卖出限制", order=7)
    def __init__(self, ft, sys_args={}, **kwargs):
        # 继承自 Account 的属性
        #self._Cash = None# 剩余现金, >=0,  array(shape=(nDT+1,))
        #self._FrozenCash = 0# 当前被冻结的现金, >=0, float
        #self._Debt = None# 负债, >=0, array(shape=(nDT+1,))
        #self._CashRecord = None# 现金流记录, 现金流入为正, 现金流出为负, DataFrame(columns=["时间点", "现金流", "备注"])
        #self._DebtRecord = None# 融资记录, 增加负债为正, 减少负债为负, DataFrame(columns=["时间点", "融资", "备注"])
        #self._TradingRecord = None# 交易记录, DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])
        self._IDs = []# 本账户支持交易的证券 ID, []
        self._Position = None# 持仓数量, DataFrame(index=[时间点]+1, columns=self._IDs)
        self._PositionAmount = None# 持仓金额, DataFrame(index=[时间点]+1, columns=self._IDs)
        self._Orders = None# 当前接收到的订单, DataFrame(columns=["ID", "数量", "目标价"])
        self._OpenPosition = None# 当前未平仓的持仓信息, DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        self._ClosedPosition = None# 已平仓的交易信息, DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        self._LastPrice = None# 最新价, Series(index=self._IDs)
        self._FT = ft# 提供净值或者收益率数据的因子表
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = "NAVAccount"
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FT.getFactorMetaData(key="DataType")))
        DefaultNumFactorList.insert(0, None)
        self.add_trait("NAV", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="复权净值", order=4))
        self.NAV = DefaultNumFactorList[-1]
        self.add_trait("Return", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="收益率", order=5))
        self.BuyLimit = _TradeLimit(direction="Buy")
        self.SellLimit = _TradeLimit(direction="Sell")
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        if (self.NAV is None) and (self.Return is None): raise __QS_Error__("复权净值或者收益率必须指定其一!")
        self._IDs = list(self.TargetIDs)
        if not self._IDs:
            if self.NAV is not None: self._IDs = list(self._FT.getID(ifactor_name=self.NAV))
            self._IDs = list(self._FT.getID(ifactor_name=self.Return))
        nDT, nID = len(dts), len(self._IDs)
        #self._Cash = np.zeros(nDT+1)
        #self._FrozenCash = 0
        #self._Debt = np.zeros(nDT+1)
        self._Position = pd.DataFrame(np.zeros((nDT+1, nID)), index=[dts[0]-dt.timedelta(1)]+dts, columns=self._IDs)
        self._PositionAmount = self._Position.copy()
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        self._LastPrice = pd.Series(1, index=self._IDs)
        self._OpenPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        self._ClosedPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        self._iTradingRecord = []# 暂存的交易记录
        self._nDT = nDT
        return Rslt + (self._FT, )
    def __QS_move__(self, idt, *args, **kwargs):
        super().__QS_move__(idt, *args, **kwargs)
        # 更新当前的账户信息
        iIndex = self._Model.DateTimeIndex
        if self.NAV is not None: self._LastPrice = self._FT.readData(factor_names=[self.NAV], ids=self._IDs, dts=[idt]).iloc[0, 0]
        else:
            iReturn = self._FT.readData(factor_names=[self.Return], ids=self._IDs, dts=[idt]).iloc[0, 0]
            iReturn[pd.isnull(iReturn)] = 0.0
            self._LastPrice *= (1 + iReturn)
        self._Position.iloc[iIndex+1] = self._Position.iloc[iIndex]# 初始化持仓
        if self.Delay:# 撮合成交
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
        else:
            TradingRecord = self._iTradingRecord
        self._QS_updatePosition()
        return TradingRecord
    def __QS_after_move__(self, idt, *args, **kwargs):
        super().__QS_after_move__(self, idt, *args, **kwargs)
        if not self.Delay:# 撮合成交
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
            self._iTradingRecord = TradingRecord
            self._QS_updatePosition()
        return 0
    def __QS_end__(self):
        super().__QS_end__()
        self._Output["持仓"] = self.getPositionSeries()
        self._Output["持仓金额"] = self.getPositionAmountSeries()
        self._Output["平仓记录"] = self._ClosedPosition
        self._Output["未平仓持仓"] = self._OpenPosition
        return 0
    # 当前账户价值
    @property
    def AccountValue(self):
        return super().AccountValue + np.nansum(self._PositionAmount.iloc[self._Model.DateTimeIndex+1])
    # 当前账户的持仓数量
    @property
    def Position(self):
        return self._Position.iloc[self._Model.DateTimeIndex+1]
    # 当前账户的持仓金额, 即保证金
    @property
    def PositionAmount(self):
        self.PositionAmount.iloc[self._Model.DateTimeIndex+1]
    # 本账户支持交易的证券 ID
    @property
    def IDs(self):
        return self._IDs
    # 当前账户中还未成交的订单, DataFrame(index=[int], columns=["ID", "数量", "目标价"])
    @property
    def Orders(self):
        return self._Orders
    # 当前最新价
    @property
    def LastPrice(self):
        return self._LastPrice
    # 获取持仓的历史序列, 以时间点为索引, 返回: pd.DataFrame(持仓, index=[时间点], columns=[ID])
    def getPositionSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = self._Position.iloc[1:self._Model.DateTimeIndex+2]
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取持仓证券的金额历史序列, 以时间点为索引, 返回: pd.DataFrame(持仓金额, index=[时间点], columns=[ID])
    def getPositionAmountSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = self._PositionAmount[1:self._Model.DateTimeIndex+2]
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取账户价值的历史序列, 以时间点为索引
    def getAccountValueSeries(self, dts=None, start_dt=None, end_dt=None):
        CashSeries = self.getCashSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        PositionAmountSeries = self.getPositionAmountSeries(dts=dts, start_dt=start_dt, end_dt=end_dt).sum(axis=1)
        DebtSeries = self.getDebtSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        return CashSeries + PositionAmountSeries - DebtSeries
    # 执行给定数量的证券委托单, target_id: 目标证券 ID, num: 待买卖的数量, target_price: nan 表示市价单, 
    # combined_order: 组合订单, DataFrame(index=[ID], columns=[数量, 目标价])
    # 基本的下单函数, 必须实现
    def order(self, target_id=None, num=0, target_price=np.nan, combined_order=None):
        if target_id is not None:
            self._Orders.loc[self._Orders.shape[0]] = (target_id, num, target_price)
            return (target_id, num, target_price)
        if combined_order is not None:
            combined_order.index = np.arange(self._Orders.shape[0], self._Orders.shape[0]+combined_order.shape[0])
            self._Orders = self._Orders.append(combined_order)
        return combined_order
    # 撤销订单, order_ids 是订单在 self.Orders 中的 index
    def cancelOrder(self, order_ids):
        self._Orders = self._Orders.loc[self._Orders.index.difference(set(order_ids))]
        self._Orders.sort_index(axis=0, inplace=True)
        self._Orders.index = np.arange(self._Orders.shape[0])
        return 0
    # 更新仓位信息
    def _QS_updatePosition(self):
        iIndex = self._Model.DateTimeIndex
        iNum = self._OpenPosition.groupby(by=["ID"])["数量"].sum()
        iPosition = pd.Series(0, index=self._IDs)
        iPosition[iNum.index] += iNum
        self._Position.iloc[iIndex+1] = iPosition.values
        LastPrice = self._LastPrice[self._OpenPosition["ID"].tolist()].values
        self._OpenPosition["浮动盈亏"] = self._OpenPosition["数量"] * (LastPrice - self._OpenPosition["开仓价格"])
        iPositionAmount = pd.Series(self._OpenPosition["数量"].values * LastPrice, index=self._OpenPosition["ID"].values).groupby(axis=0, level=0).sum()
        self._PositionAmount.iloc[iIndex+1] = 0.0
        self._PositionAmount.iloc[iIndex+1][iPositionAmount.index] = iPositionAmount
        return 0
    # 撮合成交订单
    def _matchOrder(self, idt):
        MarketOrderMask = pd.isnull(self._Orders["目标价"])
        MarketOrders = self._Orders[MarketOrderMask]
        MarketOrders = MarketOrders.groupby(by=["ID"]).sum()["数量"]# Series(数量, index=[ID])
        # 先执行平仓交易
        TradingRecord, MarketOpenOrders = self._matchMarketCloseOrder(idt, MarketOrders)
        # 再执行开仓交易
        TradingRecord.extend(self._matchMarketOpenOrder(idt, MarketOpenOrders))
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        return TradingRecord
    # 撮合成交市价平仓单
    # 以成交价完成成交, 成交量满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketCloseOrder(self, idt, orders):
        orders = orders[orders!=0]
        if orders.shape[0]==0: return ([], pd.Series())
        IDs = orders.index.tolist()
        TradePrice = self.LastPrice[IDs]
        # 过滤限制条件
        orders[pd.isnull(TradePrice)] = 0.0# 成交价缺失的不能交易
        if self.SellLimit.LimitIDFilter:# 满足卖出禁止条件的不能卖出
            Mask = (self._FT.getIDMask(idt, ids=IDs, id_filter_str=self.SellLimit.LimitIDFilter) & (orders<0))
            orders[Mask] = orders[Mask].clip_lower(0)
        if self.BuyLimit.LimitIDFilter:# 满足买入禁止条件的不能买入
            Mask = (self._FT.getIDMask(idt, ids=IDs, id_filter_str=self.BuyLimit.LimitIDFilter) & (orders>0))
            orders[Mask] = orders[Mask].clip_upper(0)
        # 分离平仓单和开仓单
        Position = self.Position[IDs]
        CloseOrders = orders.clip(lower=(-Position).clip_upper(0), upper=(-Position).clip_lower(0))# 平仓单
        OpenOrders = orders - CloseOrders# 开仓单
        CloseOrders = CloseOrders[CloseOrders!=0]
        if CloseOrders.shape[0]==0: return ([], OpenOrders)
        # 处理平仓单
        TradePrice = TradePrice[CloseOrders.index]
        TradeAmounts = CloseOrders * TradePrice
        Fees = TradeAmounts.clip_lower(0) * self.BuyLimit.TradeFee + TradeAmounts.clip_upper(0).abs() * self.SellLimit.TradeFee
        iOpenPosition = self._OpenPosition.set_index(["ID"])
        iClosedPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        iNewPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        CashChanged = np.zeros(CloseOrders.shape[0])
        for j, jID in enumerate(CloseOrders.index):
            jNum = CloseOrders.iloc[j]
            ijPosition = iOpenPosition.loc[[jID]]
            if jNum<0: ijClosedNum = (ijPosition["数量"] - (ijPosition["数量"].cumsum()+jNum).clip_lower(0)).clip_lower(0)
            else: ijClosedNum = (ijPosition["数量"] - (ijPosition["数量"].cumsum()+jNum).clip_upper(0)).clip_upper(0)
            iPNL = ijClosedNum * (TradePrice[jID] - ijPosition["开仓价格"])
            CashChanged[j] = np.abs(ijClosedNum) * ijPosition["开仓价格"] + iPNL - Fees[jID]
            ijClosedPosition = ijPosition.copy()
            ijClosedPosition["数量"] = ijClosedNum
            ijClosedPosition["平仓时点"] = idt
            ijClosedPosition["平仓价格"] = TradePrice[jID]
            ijClosedPosition["平仓交易费"] = Fees[jID]
            ijClosedPosition["平仓盈亏"] = iPNL
            ijPosition["数量"] -= ijClosedNum
            ijClosedPosition[ijClosedPosition["数量"]!=0]
            ijClosedPosition.pop("浮动盈亏")
            iClosedPosition = iClosedPosition.append(ijClosedPosition.reset_index())
            iNewPosition = iNewPosition.append(ijPosition[ijPosition["数量"]!=0].reset_index())
        iClosedPosition.index = np.arange(self._ClosedPosition.shape[0], self._ClosedPosition.shape[0]+iClosedPosition.shape[0])
        self._ClosedPosition = self._ClosedPosition.append(iClosedPosition)
        iOpenPosition = iOpenPosition.loc[iOpenPosition.index.difference(CloseOrders.index)].reset_index()
        iNewPosition.index = np.arange(iOpenPosition.shape[0], iOpenPosition.shape[0]+iNewPosition.shape[0])
        self._OpenPosition = iOpenPosition.append(iNewPosition)
        TradingRecord = list(zip([idt]*CloseOrders.shape[0], CloseOrders.index, CloseOrders, TradePrice, Fees, CashChanged, ["close"]*CloseOrders.shape[0]))
        if TradingRecord: self._QS_updateCashDebt(CashChanged.sum())
        return (TradingRecord, OpenOrders)
    # 撮合成交市价开仓单
    # 以成交价完成成交, 成交量满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketOpenOrder(self, idt, orders):
        orders = orders[orders!=0]
        if orders.shape[0]==0: return []
        IDs = orders.index.tolist()
        TradePrice = self.LastPrice[IDs]
        # 处理开仓单
        TradeAmounts = orders * TradePrice
        FeesAcquired = TradeAmounts.clip_lower(0) * self.BuyLimit.TradeFee + TradeAmounts.clip_upper(0).abs() * self.SellLimit.TradeFee
        CashAcquired = TradeAmounts.abs() + FeesAcquired
        CashAllocated = min(CashAcquired.sum(), self.AvailableCash) * CashAcquired / CashAcquired.sum()
        orders = CashAllocated / TradePrice / (1 + self.BuyLimit.TradeFee * (orders>0) + self.SellLimit.TradeFee * (orders<0)) * np.sign(orders)
        orders = orders[pd.notnull(orders) & (orders!=0)]
        if orders.shape[0]==0: return []
        TradePrice = TradePrice[orders.index]
        TradeAmounts = orders * TradePrice
        Fees = TradeAmounts.clip_lower(0) * self.BuyLimit.TradeFee + TradeAmounts.clip_upper(0).abs() * self.SellLimit.TradeFee
        CashChanged = - TradeAmounts.abs() - Fees
        TradingRecord = list(zip([idt]*orders.shape[0], orders.index, orders, TradePrice, Fees, CashChanged, ["open"]*orders.shape[0]))
        iNewPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        iNewPosition["ID"] = orders.index
        iNewPosition["数量"] = orders.values
        iNewPosition["开仓时点"] = idt
        iNewPosition["开仓价格"] = TradePrice.values
        iNewPosition["开仓交易费"] = Fees.values
        iNewPosition["浮动盈亏"] = 0.0
        iNewPosition.index = np.arange(self._OpenPosition.shape[0], self._OpenPosition.shape[0]+iNewPosition.shape[0])
        self._OpenPosition = self._OpenPosition.append(iNewPosition)
        self._QS_updateCashDebt(CashChanged.sum())
        return TradingRecord