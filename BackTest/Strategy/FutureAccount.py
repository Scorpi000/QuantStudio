# coding=utf-8
"""期货账户"""
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
from QuantStudio.BackTest.Strategy.StockAccount import _TradeLimit

class _TradeLimit(__QS_Object__):
    """交易限制"""
    TradeFee = Float(0.0, arg_type="Double", label="交易费率", order=0)

# Bar 因子: 开盘价(非必需), 最高价(非必需), 最低价(非必需), 最新价, 成交价, 成交量(非必需). 最新价用于记录账户价值变化; 成交价用于模拟市价单的成交.
class _BarFactorMap(__QS_Object__):
    """Bar 因子映照"""
    Open = Enum(None, arg_type="SingleOption", label="开盘价", order=0)
    High = Enum(None, arg_type="SingleOption", label="最高价", order=1)
    Low = Enum(None, arg_type="SingleOption", label="最低价", order=2)
    Last = Enum(None, arg_type="SingleOption", label="最新价", order=3)
    Vol = Enum(None, arg_type="SingleOption", label="成交量", order=4)
    TradePrice = Enum(None, arg_type="SingleOption", label="成交价", order=5)
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

# 证券信息: 初始保证金率, 维持保证金率, 合约乘数, 合约映射, 结算价
class _SecurityInfo(__QS_Object__):
    InitMarginRate = Float(0.15, arg_type="Double", label="初始保证金率", order=0)
    MaintenanceMarginRate = Float(0.08, arg_type="Double", label="维持保证金率", order=1)
    Multiplier = Float(1, arg_type="Double", label="合约乘数", order=2)
    ContractMapping = Enum(None, arg_type="SingleOption", label="合约映射", order=3)
    SettlementPrice = Enum(None, arg_type="SingleOption", label="结算价", order=4)
    def __init__(self, ft, sys_args={}, **kwargs):
        self._FT = ft
        return super().__init__(sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FT.getFactorMetaData(key="DataType")))
        self.add_trait("ContractMapping", Enum(*DefaultStrFactorList, arg_type="SingleOption", label="合约映射", order=3))
        self.ContractMapping = searchNameInStrList(DefaultStrFactorList, ['映射','map','Map'])
        self.add_trait("SettlementPrice", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="结算价", order=4))
        self.SettlementPrice = searchNameInStrList(DefaultNumFactorList, ['结算','价','settle','Settle','price','Price'])
        

# 基于 Bar 数据的期货账户
# 基于连续合约的行情交易
# market_ft: 行情因子表, 时间频率任意; security_ft: 证券信息因子表, 时间频率日级以下
# 市价单根据当前时段的成交价和成交量的情况成交; 限价单根据最高价, 最低价和成交量的情况成交, 假定成交量在最高价和最低价之间的分布为均匀分布, 如果没有指定最高价和最低价, 则最高价和最低价相等且等于成交价
class TimeBarAccount(Account):
    """基于 Bar 数据的期货账户"""
    Delay = Bool(True, arg_type="Bool", label="交易延迟", order=2)
    TargetIDs = ListStr(arg_type="IDList", label="目标ID", order=3)
    BuyLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="买入限制", order=4)
    SellLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="卖出限制", order=5)
    MarketFactorMap = Instance(_BarFactorMap, arg_type="ArgObject", label="行情因子", order=6)
    SecurityInfo = Instance(_SecurityInfo, arg_type="ArgObject", label="证券信息", order=7)
    def __init__(self, market_ft, security_ft, sys_args={}, **kwargs):
        # 继承自 Account 的属性
        #self._Cash = None# 剩余现金, >=0,  array(shape=(nDT+1,))
        #self._Debt = None# 负债, >=0, array(shape=(nDT+1,))
        #self._CashRecord = None# 现金流记录, 现金流入为正, 现金流出为负, DataFrame(columns=["时间点", "现金流", "备注"])
        #self._DebtRecord = None# 融资记录, 增加负债为正, 减少负债为负, DataFrame(columns=["时间点", "融资", "备注"])
        #self._TradingRecord = None# 交易记录, DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])
        self._IDs = []# 本账户支持交易的证券 ID, []
        self._Position = None# 持仓数量, DataFrame(index=[时间点]+1, columns=self._IDs)
        self._PositionAmount = None# 持仓金额, 保证金 - 浮动盈亏, DataFrame(index=[时间点]+1, columns=self._IDs)
        self._Orders = None# 当前接收到的订单, DataFrame(columns=["ID", "数量", "目标价"])
        self._OpenPosition = None# 当前未平仓的持仓信息, DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "保证金", "浮动盈亏"])
        self._ClosedPosition = None# 已平仓的持仓信息, DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "保证金", "平仓时点", "平仓价格", "平仓盈亏"])
        self._LastPrice = None# 最新价, Series(index=self._IDs)
        self._MarketFT = market_ft# 行情因子表对象
        self._SecurityFT = security_ft# 证券信息因子表对象
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = "FutureAccount"
    def __QS_initArgs__(self):
        self.MarketFactorMap = _BarFactorMap(self._MarketFT)
        self.SecurityInfo = _SecurityInfo(self._SecurityFT)
        self.BuyLimit = _TradeLimit(direction="Buy")
        self.SellLimit = _TradeLimit(direction="Sell")
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        self._IDs = list(self.TargetIDs)
        if not self._IDs: self._IDs = list(self._MarketFT.getID(ifactor_name=self.MarketFactorMap.TradePrice))
        nDT, nID = len(dts), len(self._IDs)
        #self._Cash = np.zeros(nDT+1)
        #self._Debt = np.zeros(nDT+1)
        self._Position = pd.DataFrame(np.zeros((nDT+1, nID)), index=[dts[0]-dt.timedelta(1)]+dts, columns=self._IDs)
        self._PositionAmount = self._Position.copy()
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        self._LastPrice = pd.Series(np.nan, index=self._IDs)# 最新价
        self._OpenPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "保证金", "浮动盈亏"])
        self._ClosedPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "保证金", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        self._iTradingRecord = []# 暂存的交易记录
        self._nDT = nDT
        return Rslt + (self._MarketFT, self._SecurityFT)
    def __QS_move__(self, idt, *args, **kwargs):
        super().__QS_move__(idt, *args, **kwargs)
        # 更新当前的账户信息
        iIndex = self._Model.DateTimeIndex
        self._Position.iloc[iIndex+1] = self._Position.iloc[iIndex]# 初始化持仓
        self._LastPrice = self._MarketFT.readData(factor_names=[self.MarketFactorMap.Last], ids=self._IDs, dts=[idt]).iloc[0, 0]
        if self.Delay:# 撮合成交
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
        else:
            TradingRecord = self._iTradingRecord
        self._OpenPosition["浮动盈亏"] = self._OpenPosition["数量"] * (self._LastPrice[self._OpenPosition["ID"].tolist()].values - self._OpenPosition["开仓价格"]) * self.SecurityInfo.Multiplier
        iPositionAmount = pd.Series((self._OpenPosition["浮动盈亏"]+self._OpenPosition["保证金"]).values, index=self._OpenPosition["ID"].values).groupby(axis=0, level=0).sum()
        self._PositionAmount.iloc[iIndex+1] = 0.0
        self._PositionAmount.iloc[iIndex+1][iPositionAmount.index] = iPositionAmount
        return TradingRecord
    def __QS_after_move__(self, idt, *args, **kwargs):
        super().__QS_after_move__(self, idt, *args, **kwargs)
        iIndex = self._Model.DateTimeIndex
        if not self.Delay:# 撮合成交
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
            self._iTradingRecord = TradingRecord
            self._OpenPosition["浮动盈亏"] = self._OpenPosition["数量"] * (self._LastPrice[self._OpenPosition["ID"].tolist()].values - self._OpenPosition["开仓价格"]) * self.SecurityInfo.Multiplier
            iPositionAmount = pd.Series((self._OpenPosition["浮动盈亏"]+self._OpenPosition["保证金"]).values, index=self._OpenPosition["ID"].values).groupby(axis=0, level=0).sum()
            self._PositionAmount.iloc[iIndex+1] = 0.0
            self._PositionAmount.iloc[iIndex+1][iPositionAmount.index] = iPositionAmount
        if iIndex<self._nDT-1:
            iNextDate = self._Model._QS_TestDateTimes[iIndex+1].date()
            if iNextDate!=idt.date():# 当前是该交易日的最后一个时点
                self._handleRollover(idt)# 处理展期
                self._handleSettlement(idt)# 处理结算
        return 0
    def __QS_end__(self):
        super().__QS_end__()
        self._Output["持仓"] = self.getPositionSeries()
        self._Output["持仓金额"] = self.getPositionAmountSeries()
        self._Output["平仓记录"] = self._ClosedPosition
        self._Output["未平仓持仓"] = self._OpenPosition
        return 0
    # 处理结算, 对于不满足保证金要求的仓位进行强平
    def _handleSettlement(self, idt):
        if self._OpenPosition.shape[0]==0: return 0
        idt = dt.datetime.combine(idt.date(), dt.time(23,59,59,999999))
        SettlementPrice = self._SecurityFT.readData(factor_names=[self.SecurityInfo.SettlementPrice], dts=[idt], ids=self._IDs).iloc[0,0,:][self._OpenPosition["ID"].tolist()]
        self._OpenPosition["浮动盈亏"] = self._OpenPosition["数量"] * (SettlementPrice.values - self._OpenPosition["开仓价格"]) * self.SecurityInfo.Multiplier
        MaintenanceMargin = self._OpenPosition["数量"].abs() * self.SecurityInfo.Multiplier * SettlementPrice.values * self.SecurityInfo.MaintenanceMarginRate# 维持保证金
        iGap = (MaintenanceMargin - self._OpenPosition["保证金"] - self._OpenPosition["浮动盈亏"]).clip_lower(0)# 当前保证金和维持保证金之间的缺口
        iTotalGap = np.nansum(iGap)
        if iTotalGap<=0: return 0
        AvailableCash = self.AvailableCash
        iIndex = self._Model.DateTimeIndex
        if AvailableCash>=iTotalGap:
            self._updateAccount(-iTotalGap)
            self._OpenPosition["保证金"] += iGap
            iPositionAmount = pd.Series((self._OpenPosition["浮动盈亏"]+self._OpenPosition["保证金"]).values, index=self._OpenPosition["ID"].values).groupby(axis=0, level=0).sum()
            self._PositionAmount.iloc[iIndex+1][iPositionAmount.index] = iPositionAmount
            return 0
        self._OpenPosition["保证金"] += (iGap / iTotalGap) * AvailableCash# 将账户中的现金按照缺口金额大小分配给各个 ID
        iAllowedPosition = self._OpenPosition["保证金"] / self.SecurityInfo.Multiplier / SettlementPrice.values / self.MaintenanceMarginRate# 计算当前保证金水平下允许的持仓数量
        # 对于保证金不足的仓位进行平仓
        iNum = (self._OpenPosition["数量"].abs() - iAllowedPosition).clip_lower(0)
        iOrders = pd.Series(-iNum.values*np.sign(self._OpenPosition["数量"].values), index=self._OpenPosition["ID"].values).groupby(axis=0, level=0).sum()
        iOrders = iOrders[iOrders!=0]
        iBookOrders = self._Orders.groupby(by=["ID"])["数量"].sum()
        if iBookOrders.shape[0]>0: iBookOrders = iBookOrders.loc[iOrders.index].fillna(0)
        else: iBookOrders = pd.Series(0, index=iOrders.index)
        iMask = (iOrders>0)
        iOrders[iMask] = (iOrders[iMask] - iBookOrders[iMask]).clip_lower(0)
        iMask = (iOrders<0)
        iOrders[iMask] = (iOrders[iMask] - iBookOrders[iMask]).clip_upper(0)
        iOrders = iOrders[iOrders!=0]
        if iOrders.shape[0]==0: return 0
        iOrders = pd.DataFrame(iOrders, columns=["数量"]).reset_index()
        iOrders["目标价"] = np.nan
        self.order(combined_order=iOrders)
        return 0
    # 处理展期
    def _handleRollover(self, idt):
        if self._OpenPosition.shape[0]==0: return 0
        iIndex = self._Model.DateTimeIndex
        idt = dt.datetime.combine(idt.date(), dt.time(23,59,59,999999))
        iNextDT = dt.datetime.combine(self._Model._QS_TestDateTimes[iIndex+1].date(), dt.time(23,59,59,999999))
        IDMapping = self._SecurityFT.readData(factor_names=[self.SecurityInfo.ContractMapping], ids=self._IDs, dts=[idt, iNextDT]).iloc[0]
        SettlementPrice = self._SecurityFT.readData(factor_names=[self.SecurityInfo.SettlementPrice], dts=[idt], ids=self._IDs).iloc[0,0,:]
        isRollover = False
        for iID in self._OpenPosition["ID"].unique():
            iCurID, iNextID = IDMapping[iID].iloc[0], IDMapping[iID].iloc[1]
            if iCurID==iNextID: continue
            isRollover = True
            iNextIDPrice = SettlementPrice[IDMapping.iloc[0]==iNextID].iloc[0]
            iCurIDPrice = SettlementPrice[iID]
            iMask = (self._OpenPosition["ID"]==iID)
            iNewNum = self._OpenPosition["数量"][iMask] * iCurIDPrice / iNextIDPrice
            self._OpenPosition["开仓价格"][iMask] = self._OpenPosition["数量"][iMask] * self._OpenPosition["开仓价格"][iMask] / iNewNum
            self._OpenPosition["数量"][iMask] = iNewNum
        if isRollover: self._updateAccount(cash_changed=0.0)
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
    # combined_order: 组合订单, DataFrame(index=[ID],columns=[数量, 目标价])
    # 基本的下单函数, 必须实现, 目前只实现了市价单
    def order(self, target_id=None, num=0, target_price=np.nan, combined_order=None):
        if target_id is not None:
            #if (num>0) and (self.BuyLimit.MinUnit>0):
                #num = np.fix(num / self.BuyLimit.MinUnit) * self.BuyLimit.MinUnit
            #elif (num<0) and (self.SellLimit.MinUnit>0):
                #num = np.fix(num / self.SellLimit.MinUnit) * self.SellLimit.MinUnit
            self._Orders.loc[self._Orders.shape[0]] = (target_id, num, target_price)
            return (target_id, num, target_price)
        if combined_order is not None:
            #if self.BuyLimit.MinUnit>0:
                #Mask = (combined_order["数量"]>0)
                #combined_order["数量"][Mask] = np.fix(combined_order["数量"][Mask] / self.BuyLimit.MinUnit) * self.BuyLimit.MinUnit
            #if self.SellLimit.MinUnit>0:
                #Mask = (combined_order["数量"]<0)
                #combined_order["数量"][Mask] = np.fix(combined_order["数量"][Mask] / self.SellLimit.MinUnit) * self.SellLimit.MinUnit
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
    def _updateAccount(self, cash_changed):
        iIndex = self._Model.DateTimeIndex
        DebtDelta = max(- cash_changed - self.Cash, 0)
        if self.DeltLimit>0:
            self._Debt[iIndex+1] += DebtDelta
            if DebtDelta>0: self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, DebtDelta, "")
            self._Cash[iIndex+1] -= min(- cash_changed, self.Cash)
        else:
            if DebtDelta>0: self._Cash[iIndex+1] = 0
            else: self._Cash[iIndex+1] -= min(- cash_changed, self.Cash)
        iNum = self._OpenPosition.groupby(by=["ID"])["数量"].sum()
        iPosition = pd.Series(0, index=self._IDs)
        iPosition[iNum.index] += iNum
        self._Position.iloc[iIndex+1] = iPosition.values
        return 0
    # 撮合成交订单
    def _matchOrder(self, idt):
        MarketOrderMask = pd.isnull(self._Orders["目标价"])
        MarketOrders = self._Orders[MarketOrderMask]
        LimitOrders = self._Orders[~MarketOrderMask]
        MarketOrders = MarketOrders.groupby(by=["ID"]).sum()["数量"]# Series(数量, index=[ID])
        if LimitOrders.shape[0]>0:
            LimitOrders = LimitOrders.groupby(by=["ID", "目标价"]).sum()
            LimitOrders.sort_index(ascending=False)
            LimitOrders = LimitOrders.reset_index(level=1)
        else:
            LimitOrders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        # 先执行平仓交易
        TradingRecord, MarketOpenOrders = self._matchMarketCloseOrder(idt, MarketOrders)
        iTradingRecord, LimitOrders = self._matchLimitCloseOrder(idt, LimitOrders)
        TradingRecord.extend(iTradingRecord)
        # 再执行开仓交易
        TradingRecord.extend(self._matchMarketOpenOrder(idt, MarketOpenOrders))
        iTradingRecord, LimitOrders = self._matchLimitOpenOrder(idt, LimitOrders)
        TradingRecord.extend(iTradingRecord)
        self._Orders = LimitOrders
        self._Orders.index = np.arange(self._Orders.shape[0])
        return TradingRecord
    # 撮合成交市价平仓单
    # 以成交价完成成交, 成交量满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketCloseOrder(self, idt, orders):
        orders = orders[orders!=0]
        if orders.shape[0]==0: return ([], pd.Series())
        IDs = orders.index.tolist()
        TradePrice = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketFactorMap.TradePrice]).iloc[0, 0]# 成交价
        # 过滤限制条件
        orders[pd.isnull(TradePrice)] = 0.0# 成交价缺失的不能交易
        #if self.SellLimit.LimitIDFilter:# 满足卖出禁止条件的不能卖出
            #Mask = (self._MarketFT.getIDMask(idt, ids=IDs, id_filter_str=self.SellLimit.LimitIDFilter) & (orders<0))
            #orders[Mask] = orders[Mask].clip_lower(0)
        #if self.BuyLimit.LimitIDFilter:# 满足买入禁止条件的不能买入
            #Mask = (self._MarketFT.getIDMask(idt, ids=IDs, id_filter_str=self.BuyLimit.LimitIDFilter) & (orders>0))
            #orders[Mask] = orders[Mask].clip_upper(0)
        #if self.MarketFactorMap.Vol:# 成交量限制
            #VolLimit = self._MarketFT.readData(factor_names=[self.MarketFactorMap.Vol], ids=IDs, dts=[idt]).iloc[0, 0]
            #orders = np.maximum(orders, -VolLimit * self.SellLimit.MarketOrderVolumeLimit)
            #orders = np.minimum(orders, VolLimit * self.BuyLimit.MarketOrderVolumeLimit)
        #if self.SellLimit.MinUnit!=0.0:# 最小卖出交易单位限制
            #Mask = (orders<0)
            #orders[Mask] = (orders[Mask] / self.SellLimit.MinUnit).astype("int") * self.SellLimit.MinUnit
        #if self.BuyLimit.MinUnit!=0.0:# 最小买入交易单位限制
            #Mask = (orders>0)
            #orders[Mask] = (orders[Mask] / self.BuyLimit.MinUnit).astype("int") * self.BuyLimit.MinUnit
        # 分离平仓单和开仓单
        Position = self.Position[IDs]
        CloseOrders = orders.clip(lower=(-Position).clip_upper(0), upper=(-Position).clip_lower(0))# 平仓单
        OpenOrders = orders - CloseOrders# 开仓单
        CloseOrders = CloseOrders[CloseOrders!=0]
        if CloseOrders.shape[0]==0: return ([], OpenOrders)
        # 处理平仓单
        TradePrice = TradePrice[CloseOrders.index]
        TradeAmounts = CloseOrders * TradePrice * self.SecurityInfo.Multiplier
        Fees = TradeAmounts.clip_lower(0) * self.BuyLimit.TradeFee + TradeAmounts.clip_upper(0).abs() * self.SellLimit.TradeFee
        iOpenPosition = self._OpenPosition.set_index(["ID"])
        iClosedPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "保证金", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        iNewPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "保证金", "浮动盈亏"])
        CashChanged = np.zeros(CloseOrders.shape[0])
        for j, jID in enumerate(CloseOrders.index):
            jNum = CloseOrders.iloc[j]
            ijPosition = iOpenPosition.loc[[jID]]
            if jNum<0: ijClosedNum = (ijPosition["数量"] - (ijPosition["数量"].cumsum()+jNum).clip_lower(0)).clip_lower(0)
            else: ijClosedNum = (ijPosition["数量"] - (ijPosition["数量"].cumsum()+jNum).clip_upper(0)).clip_upper(0)
            CashChanged[j] = ijClosedNum * (TradePrice[jID] - ijPosition["开仓价格"]) + ijPosition["保证金"] - Fees[jID]
            ijClosedPosition = ijPosition.copy()
            ijClosedPosition["数量"] = ijClosedNum
            ijClosedPosition["平仓时点"] = idt
            ijClosedPosition["平仓价格"] = TradePrice[jID]
            ijClosedPosition["平仓交易费"] = Fees[jID]
            ijClosedPosition["平仓盈亏"] = ijClosedNum * (TradePrice[jID] - ijPosition["开仓价格"]) * self.SecurityInfo.Multiplier
            ijClosedPosition["保证金"] = ijClosedPosition["保证金"] * (ijClosedPosition["数量"] / ijPosition["数量"])
            ijPosition["保证金"] = ijPosition["保证金"] - ijClosedPosition["保证金"]
            ijPosition["数量"] -= ijClosedNum
            CashChanged[j] = ijClosedPosition["平仓盈亏"].sum() - Fees[jID] + ijClosedPosition["保证金"].sum()
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
        if TradingRecord: self._updateAccount(CashChanged.sum())# 更新账户信息
        return (TradingRecord, OpenOrders)
    # 撮合成交市价开仓单
    # 以成交价完成成交, 成交量满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketOpenOrder(self, idt, orders):
        orders = orders[orders!=0]
        if orders.shape[0]==0: return []
        IDs = orders.index.tolist()
        TradePrice = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketFactorMap.TradePrice]).iloc[0, 0]# 成交价
        # 处理开仓单
        TradeAmounts = orders * TradePrice * self.SecurityInfo.Multiplier
        MarginAcquired = TradeAmounts.abs() * self.SecurityInfo.InitMarginRate
        FeesAcquired = TradeAmounts.clip_lower(0) * self.BuyLimit.TradeFee + TradeAmounts.clip_upper(0).abs() * self.SellLimit.TradeFee
        CashAcquired = MarginAcquired + FeesAcquired
        CashAllocated = min(CashAcquired.sum(), self.AvailableCash) * CashAcquired / CashAcquired.sum()
        orders = CashAllocated / TradePrice / self.SecurityInfo.Multiplier / (self.SecurityInfo.InitMarginRate + self.BuyLimit.TradeFee * (orders>0) + self.SellLimit.TradeFee * (orders<0)) * np.sign(orders)
        #if self.SellLimit.MinUnit!=0.0:# 最小卖出交易单位限制
            #Mask = (orders<0)
            #orders[Mask] = (orders[Mask] / self.SellLimit.MinUnit).astype("int") * self.SellLimit.MinUnit
        #if self.BuyLimit.MinUnit!=0.0:# 最小买入交易单位限制
            #Mask = (orders>0)
            #orders[Mask] = (orders[Mask] / self.BuyLimit.MinUnit).astype("int") * self.BuyLimit.MinUnit
        orders = orders[pd.notnull(orders) & (orders!=0)]
        if orders.shape[0]==0: return []
        TradePrice = TradePrice[orders.index]
        TradeAmounts = orders * TradePrice * self.SecurityInfo.Multiplier
        Fees = TradeAmounts.clip_lower(0) * self.BuyLimit.TradeFee + TradeAmounts.clip_upper(0).abs() * self.SellLimit.TradeFee
        MarginAcquired = TradeAmounts.abs() * self.SecurityInfo.InitMarginRate
        CashChanged = - (MarginAcquired + Fees).sum()
        TradingRecord = list(zip([idt]*orders.shape[0], orders.index, orders, TradePrice, Fees, -(MarginAcquired+Fees), ["open"]*orders.shape[0]))
        iNewPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "保证金", "浮动盈亏"])
        iNewPosition["ID"] = orders.index
        iNewPosition["数量"] = orders.values
        iNewPosition["开仓时点"] = idt
        iNewPosition["开仓价格"] = TradePrice.values
        iNewPosition["开仓交易费"] = Fees.values
        iNewPosition["保证金"] = MarginAcquired.values
        iNewPosition["浮动盈亏"] = 0.0
        iNewPosition.index = np.arange(self._OpenPosition.shape[0], self._OpenPosition.shape[0]+iNewPosition.shape[0])
        self._OpenPosition = self._OpenPosition.append(iNewPosition)
        self._updateAccount(CashChanged)# 更新账户信息
        return TradingRecord
    # 撮合成交卖出限价单
    # 如果最高价和最低价未指定, 则检验成交价是否优于目标价, 是就以目标价完成成交, 且成交量满足卖出限制中的限价单成交量限比, 否则无法成交
    # 如果指定了最高价和最低价, 则假设成交价服从[最低价, 最高价]中的均匀分布, 据此确定可完成成交的数量
    # 未成交的限价单继续保留, TODO
    def _matchLimitCloseOrder(self, idt, sell_orders):
        return ([], pd.DataFrame(columns=["ID", "数量", "目标价"]))
    # 撮合成交买入限价单
    # 如果最高价和最低价未指定, 则检验成交价是否优于目标价, 是就以目标价完成成交, 且成交量满足买入限制要求
    # 如果指定了最高价和最低价, 则假设成交价服从[最低价, 最高价]中的均匀分布, 据此确定可完成成交的数量, 同时满足买入限制要求
    # 未成交的限价单继续保留, TODO
    def _matchLimitOpenOrder(self, idt, buy_orders):
        return ([], pd.DataFrame(columns=["ID", "数量", "目标价"]))