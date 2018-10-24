# coding=utf-8
"""简单账户"""
import datetime as dt

import pandas as pd
import numpy as np
from traits.api import Enum, ListStr, Float, Str, Int, Bool, Instance, on_trait_change

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.BackTest.Strategy.StrategyModule import Account, cutDateTime, _QS_MinPositionNum
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList

class _TradeLimit(__QS_Object__):
    """交易限制"""
    LimitIDFilter = Str(arg_type="IDFilter", label="禁止条件", order=0)
    TradeFee = Float(0.003, arg_type="Double", label="交易费率", order=1)
    AmtLimitRatio = Float(0.1, arg_type="Double", label="成交额限比", order=2)
    def __init__(self, direction, sys_args={}, config_file=None, **kwargs):
        self._Direction = direction
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        if self._Direction=="Sell":
            self.add_trait("ShortAllowed", Bool(True, arg_type="SingleOption", label="允许卖空", order=3))

class _MarketInfo(__QS_Object__):
    """行情信息"""
    Last = Enum(None, arg_type="SingleOption", label="最新价", order=0)
    TradePrice = Enum(None, arg_type="SingleOption", label="成交价", order=1)
    Amt = Enum(None, arg_type="SingleOption", label="成交额", order=2)
    def __init__(self, ft, sys_args={}, config_file=None, **kwargs):
        self._MarketFT = ft
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._MarketFT.getFactorMetaData(key="DataType")))
        DefaultNumFactorList.insert(0, None)
        self.add_trait("Last", Enum(*DefaultNumFactorList[1:], arg_type="SingleOption", label="最新价", order=0))
        self.add_trait("TradePrice", Enum(*DefaultNumFactorList[1:], arg_type="SingleOption", label="成交价", order=1))
        self.add_trait("Amt", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="成交量", order=2))
        self.TradePrice = self.Last = searchNameInStrList(DefaultNumFactorList[1:], ['新','收','Last','last','close','Close'])

# 基于价格和成交额数据的简单账户, 只支持市价单
# market_ft: 提供价格数据的因子表, 时间频率任意
# 市价单的成交价为当前时段的指定成交价, 如果指定了成交量, 则成交量上限为当前时段成交量的一定比例, 否则无上限
class SimpleAccount(Account):
    """简单证券账户"""
    Delay = Bool(True, arg_type="Bool", label="交易延迟", order=2)
    TargetIDs = ListStr(arg_type="IDList", label="目标ID", order=3)
    BuyLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="买入限制", order=4)
    SellLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="卖出限制", order=5)
    MarketInfo = Instance(_MarketInfo, allow_none=False, arg_type="ArgObject", label="行情信息", order=6)
    def __init__(self, market_ft, name="简单证券账户", sys_args={}, config_file=None, **kwargs):
        # 继承自 Account 的属性
        #self._Cash = None# 剩余现金, >=0,  array(shape=(nDT+1,))
        #self._FrozenCash = 0# 当前被冻结的现金, >=0, float
        #self._Debt = None# 负债, >=0, array(shape=(nDT+1,))
        #self._CashRecord = None# 现金流记录, 现金流入为正, 现金流出为负, DataFrame(columns=["时间点", "现金流", "备注"])
        #self._DebtRecord = None# 融资记录, 增加负债为正, 减少负债为负, DataFrame(columns=["时间点", "融资", "备注"])
        #self._TradingRecord = None# 交易记录, DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])
        self._IDs = []# 本账户支持交易的证券 ID, []
        self._PositionNum = None# 持仓数量, DataFrame(index=[时间点]+1, columns=self._IDs)
        self._PositionAmount = None# 持仓金额, DataFrame(index=[时间点]+1, columns=self._IDs)
        self._Orders = None# 当前接收到的订单, DataFrame(columns=["ID", "数量", "目标价"])
        self._OpenPosition = None# 当前未平仓的持仓信息, DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        self._ClosedPosition = None# 已平仓的交易信息, DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        self._LastPrice = None# 最新价, Series(index=self._IDs)
        self._TradePrice = None# 成交价, Series(index=self._IDs)
        self._SellVolLimit = None# 卖出成交量限制, Series(index=self._IDs)
        self._BuyVolLimit = None# 买入成交量限制, Series(index=self._IDs)
        self._MarketFT = market_ft# 提供净值或者收益率数据的因子表
        return super().__init__(name=name, sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.BuyLimit = _TradeLimit(direction="Buy")
        self.SellLimit = _TradeLimit(direction="Sell")
        self.MarketInfo = _MarketInfo(ft=self._MarketFT)
    def __QS_start__(self, mdl, dts, **kwargs):
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._IDs = list(self.TargetIDs)
        if not self._IDs: self._IDs = self._MarketFT.getID(ifactor_name=self.MarketInfo.Last)
        nDT, nID = len(dts), len(self._IDs)
        #self._Cash = np.zeros(nDT+1)
        #self._FrozenCash = 0
        #self._Debt = np.zeros(nDT+1)
        self._PositionNum = pd.DataFrame(np.zeros((nDT+1, nID)), index=[dts[0]-dt.timedelta(1)]+dts, columns=self._IDs)
        self._PositionAmount = self._PositionNum.copy()
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        self._LastPrice = self._TradePrice = None
        self._OpenPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        self._ClosedPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        self._iTradingRecord = []# 暂存的交易记录
        self._SellVolLimit, self._BuyVolLimit = pd.Series(np.inf, index=self._IDs), pd.Series(np.inf, index=self._IDs)
        self._nDT = nDT
        return Rslt + (self._MarketFT, )
    def __QS_move__(self, idt, **kwargs):
        super().__QS_move__(idt, **kwargs)
        # 更新当前的账户信息
        iIndex = self._Model.DateTimeIndex
        self._LastPrice = self._MarketFT.readData(factor_names=[self.MarketInfo.Last], ids=self._IDs, dts=[idt]).iloc[0, 0]
        self._TradePrice = self._MarketFT.readData(factor_names=[self.MarketInfo.TradePrice], ids=self._IDs, dts=[idt]).iloc[0, 0]
        self._PositionNum.iloc[iIndex+1] = self._PositionNum.iloc[iIndex]# 初始化持仓
        if self.Delay:# 撮合成交
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
        else:
            TradingRecord = self._iTradingRecord
        self._QS_updatePosition()
        return TradingRecord
    def __QS_after_move__(self, idt, **kwargs):
        super().__QS_after_move__(idt, **kwargs)
        if not self.Delay:# 撮合成交
            TradingRecord = self._matchOrder(idt)
            TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(TradingRecord)
            self._iTradingRecord = TradingRecord
            self._QS_updatePosition()
        return 0
    def __QS_end__(self):
        super().__QS_end__()
        self._Output["持仓数量"] = self.getPositionNumSeries()
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
    def PositionNum(self):
        return self._PositionNum.iloc[self._Model.DateTimeIndex+1]
    # 当前账户的持仓金额
    @property
    def PositionAmount(self):
        return self._PositionAmount.iloc[self._Model.DateTimeIndex+1]
    # 本账户支持交易的证券 ID
    @property
    def IDs(self):
        return self._IDs
    # 当前最新价
    @property
    def LastPrice(self):
        return self._LastPrice
    # 当前账户中还未成交的订单, DataFrame(index=[int], columns=["ID", "数量", "目标价"])
    @property
    def Orders(self):
        return self._Orders
    # 获取持仓的历史序列, 以时间点为索引, 返回: pd.DataFrame(持仓, index=[时间点], columns=[ID])
    def getPositionNumSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = self._PositionNum.iloc[1:self._Model.DateTimeIndex+2]
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
    # 执行给定数量的证券委托单, target_id: 目标证券 ID, num: 待买卖的数量, target_price: nan 表示市价单, 返回: (订单ID, 
    # combined_order: 组合订单, DataFrame(index=[ID], columns=[数量, 目标价])
    # 基本的下单函数, 必须实现
    def order(self, target_id=None, num=0, target_price=np.nan, combined_order=None):
        if target_id is not None:
            self._Orders.loc[self._Orders.shape[0]] = (target_id, num, target_price)
            return (self._Orders.shape[0], target_id, num, target_price)
        if combined_order is not None:
            combined_order.index.name = "ID"
            combined_order = combined_order.reset_index()
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
        self._PositionNum.iloc[iIndex+1] = iPosition.values
        LastPrice = self._LastPrice[self._OpenPosition["ID"].tolist()].values
        self._OpenPosition["浮动盈亏"] = self._OpenPosition["数量"] * (LastPrice - self._OpenPosition["开仓价格"])
        iPositionAmount = pd.Series(self._OpenPosition["数量"].values * LastPrice, index=self._OpenPosition["ID"].values).groupby(axis=0, level=0).sum()
        self._PositionAmount.iloc[iIndex+1] = 0.0
        self._PositionAmount.iloc[iIndex+1][iPositionAmount.index] = iPositionAmount
        return 0
    # 更新交易限制条件
    def _updateTradeLimit(self, idt):
        self._SellVolLimit[:] = self._BuyVolLimit[:] = np.inf
        if self.SellLimit.LimitIDFilter:# 满足卖出禁止条件的不能卖出
            Mask = self._MarketFT.getIDMask(idt, ids=self._IDs, id_filter_str=self.SellLimit.LimitIDFilter)
            self._SellVolLimit[Mask] = 0.0
        if self.BuyLimit.LimitIDFilter:# 满足买入禁止条件的不能买入
            Mask = self._MarketFT.getIDMask(idt, ids=self._IDs, id_filter_str=self.BuyLimit.LimitIDFilter)
            self._BuyVolLimit[Mask] = 0.0
        if self.MarketInfo.Amt is not None:# 指定了成交额, 成交额满足限制要求
            Amount = self._MarketFT.readData(factor_names=[self.MarketInfo.Amt], ids=self._IDs, dts=[idt]).iloc[0,0,:]
            self._SellVolLimit = self._SellVolLimit.clip_upper(Amount * self.SellLimit.AmtLimitRatio / self._TradePrice)
            self._BuyVolLimit = self._BuyVolLimit.clip_upper(Amount * self.BuyLimit.AmtLimitRatio / self._TradePrice)
        Mask = (pd.isnull(self._TradePrice) | (self._TradePrice<=0))
        self._SellVolLimit[Mask] = self._BuyVolLimit[Mask] = 0.0# 成交价缺失的不能交易
        if not self.SellLimit.ShortAllowed:
            PositionNum = self._PositionNum.iloc[self._Model.DateTimeIndex+1]
            self._SellVolLimit = self._SellVolLimit.clip_upper(PositionNum.clip_lower(0.0))
        return 0
    # 撮合成交订单
    def _matchOrder(self, idt):
        if self._Orders.shape[0]==0: return []
        self._updateTradeLimit(idt)
        MarketOrderMask = pd.isnull(self._Orders["目标价"])
        MarketOrders = self._Orders[MarketOrderMask]
        MarketOrders = MarketOrders.groupby(by=["ID"]).sum()["数量"]# Series(数量, index=[ID])
        # 先执行平仓交易
        TradingRecord, MarketOpenOrders = self._matchMarketCloseOrder(idt, MarketOrders)
        LimitOrders = self._Orders[~MarketOrderMask]
        #if LimitOrders.shape[0]>0:
            #LimitOrders = LimitOrders.groupby(by=["ID", "目标价"]).sum()
            #LimitOrders = LimitOrders.reset_index(level=1)
            #iTradingRecord, LimitOrders = self._matchLimitCloseOrder(idt, LimitOrders)
            #TradingRecord.extend(iTradingRecord)
        # 再执行开仓交易
        TradingRecord.extend(self._matchMarketOpenOrder(idt, MarketOpenOrders))
        #if LimitOrders.shape[0]>0:
            #iTradingRecord, LimitOrders = self._matchLimitOpenOrder(idt, LimitOrders)
            #TradingRecord.extend(iTradingRecord)
            #LimitOrders = LimitOrders.reset_index()
        self._Orders = LimitOrders
        self._Orders.index = np.arange(self._Orders.shape[0])
        return TradingRecord
    # 撮合成交市价平仓单
    # 以成交价完成成交, 满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketCloseOrder(self, idt, orders):
        IDs = orders.index.tolist()
        orders = orders.clip(upper=self._BuyVolLimit.loc[IDs], lower=-self._SellVolLimit.loc[IDs])# 过滤限制条件
        orders = orders[orders!=0]
        if orders.shape[0]==0: return ([], pd.Series())
        # 分离平仓单和开仓单
        PositionNum = self.PositionNum[orders.index]
        CloseOrders = orders.clip(lower=(-PositionNum).clip_upper(0), upper=(-PositionNum).clip_lower(0))# 平仓单
        OpenOrders = orders - CloseOrders# 开仓单
        CloseOrders = CloseOrders[CloseOrders!=0]
        if CloseOrders.shape[0]==0: return ([], OpenOrders)
        # 处理平仓单
        TradePrice = self._TradePrice[CloseOrders.index]
        TradeAmounts = CloseOrders * TradePrice
        Fees = TradeAmounts.clip_lower(0) * self.BuyLimit.TradeFee + TradeAmounts.clip_upper(0).abs() * self.SellLimit.TradeFee
        OpenPosition = self._OpenPosition.set_index(["ID"])
        ClosedPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        RemainderPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        CashChanged = np.zeros(CloseOrders.shape[0])
        for i, iID in enumerate(CloseOrders.index):
            iNum = CloseOrders.iloc[i]
            iOpenPosition = OpenPosition.loc[[iID]]
            if iNum<0: iClosedNum = (iOpenPosition["数量"] - (iOpenPosition["数量"].cumsum() + iNum).clip_lower(0)).clip_lower(0)
            else: iClosedNum = (iOpenPosition["数量"] - (iOpenPosition["数量"].cumsum() + iNum).clip_upper(0)).clip_upper(0)
            iPNL = iClosedNum * (TradePrice[iID] - iOpenPosition["开仓价格"])
            CashChanged[i] = (np.abs(iClosedNum) * iOpenPosition["开仓价格"] + iPNL).sum() - Fees[iID]
            iClosedPosition = iOpenPosition.copy()
            iClosedPosition["数量"] = iClosedNum
            iClosedPosition["平仓时点"] = idt
            iClosedPosition["平仓价格"] = TradePrice[iID]
            iClosedPosition["平仓交易费"] = Fees[iID]
            iClosedPosition["平仓盈亏"] = iPNL
            iOpenPosition["数量"] -= iClosedNum
            iClosedPosition = iClosedPosition[iClosedPosition["数量"]!=0]
            iClosedPosition.pop("浮动盈亏")
            ClosedPosition = ClosedPosition.append(iClosedPosition.reset_index())
            self._SellVolLimit.loc[iID] -= np.clip(iClosedNum.sum(), 0, np.inf)# 调整卖出限制条件
            self._BuyVolLimit.loc[iID] += np.clip(iClosedNum.sum(), -np.inf, 0)# 调整买入限制条件
            RemainderPosition = RemainderPosition.append(iOpenPosition[iOpenPosition["数量"]!=0].reset_index())
        ClosedPosition.index = np.arange(self._ClosedPosition.shape[0], self._ClosedPosition.shape[0]+ClosedPosition.shape[0])
        self._ClosedPosition = self._ClosedPosition.append(ClosedPosition)
        OpenPosition = OpenPosition.loc[OpenPosition.index.difference(CloseOrders.index)].reset_index()
        RemainderPosition.index = np.arange(OpenPosition.shape[0], OpenPosition.shape[0]+RemainderPosition.shape[0])
        self._OpenPosition = OpenPosition.append(RemainderPosition)
        if self.SellLimit.ShortAllowed:
            self._OpenPosition = self._OpenPosition[self._OpenPosition["数量"].abs()>_QS_MinPositionNum]
        else:
            self._OpenPosition = self._OpenPosition[self._OpenPosition["数量"]>_QS_MinPositionNum]
        TradingRecord = list(zip([idt]*CloseOrders.shape[0], CloseOrders.index, CloseOrders, TradePrice, Fees, CashChanged, ["close"]*CloseOrders.shape[0]))
        if TradingRecord: self._QS_updateCashDebt(CashChanged.sum())
        return (TradingRecord, OpenOrders)
    # 撮合成交市价开仓单
    # 以成交价完成成交, 满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketOpenOrder(self, idt, orders):
        orders = orders[orders!=0]
        if orders.shape[0]==0: return []
        TradePrice = self._TradePrice[orders.index]
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
        NewPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        NewPosition["ID"] = orders.index
        NewPosition["数量"] = orders.values
        NewPosition["开仓时点"] = idt
        NewPosition["开仓价格"] = TradePrice.values
        NewPosition["开仓交易费"] = Fees.values
        NewPosition["浮动盈亏"] = 0.0
        NewPosition.index = np.arange(self._OpenPosition.shape[0], self._OpenPosition.shape[0]+NewPosition.shape[0])
        self._OpenPosition = self._OpenPosition.append(NewPosition)
        self._QS_updateCashDebt(CashChanged.sum())
        return TradingRecord
    # 撮合成交限价平仓单, TODO
    # 限价单的报价如果优于当前时段的成交价则可成交, 否则继续挂单
    def _matchLimitCloseOrder(self, idt, orders):
        if orders.shape[0]==0: return ([], orders)
        IDs = orders.index.unique().tolist()
        PositionNum = self.PositionNum
        RemainderOrders = pd.DataFrame(columns=["目标价", "数量"])
        OpenPosition = self._OpenPosition.set_index(["ID"])
        CashChanged = zeros(len(IDs))
        ClosedPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "平仓时点", "平仓价格", "平仓交易费", "平仓盈亏"])
        NewPosition = pd.DataFrame(columns=["ID", "数量", "开仓时点", "开仓价格", "开仓交易费", "浮动盈亏"])
        for i, iID in enumerate(IDs):
            iOpenPosition = OpenPosition.loc[[iID]]
            iPositionNum = iOpenPosition["数量"].sum()
            iOrders = orders.loc[[iID]]
            iTradePrice = self._TradePrice.loc[iID]
            if (iPositionNum==0) or pd.isnull(iTradePrice):# 当前无仓位
                RemainderOrders = RemainderOrders.append(iOrders)
                continue
            iSellOrders = iOrders[iOrders["数量"]<0].sort_values(by=["目标价"], ascending=True)
            iBuyOrders = iOrders[iOrders["数量"]>0].sort_values(by=["目标价"], ascending=False)
            iClosedPosition = iOpenPosition.copy()
            if (iPositionNum>0) and (iSellOrders.shape[0]>0):# 当前多头仓位, 卖出订单为平仓单
                iClosedNum = min(iSellOrders["数量"][iSellOrders["目标价"]<=iTradePrice].abs().sum(), self._SellVolLimit.loc[iID], iPositionNum)
                
                iRemainder = (iSellOrders["数量"].cumsum() + iClosedNum).clip_upper(0.0)
                iRemainder.iloc[1:] = iRemainder.diff().iloc[1:]
                iSellOrders["数量"] = iRemainder.values
                RemainderOrders = RemainderOrders.append(iSellOrders[iSellOrders["数量"]!=0])
                self._SellVolLimit.loc[iID] -= iClosedNum# 调整卖出限制条件
                iClosedNum = (iOpenPosition["数量"] - (iOpenPosition["数量"].cumsum() - iClosedNum).clip_lower(0)).clip_lower(0)
                iFee = iClosedNum * iTradePrice * self.SellLimit.TradeFee
            elif (iPositionNum<0) and (iBuyOrders.shape[0]>0):
                iClosedNum = min(iBuyOrders["数量"][iBuyOrders["目标价"]>=iTradePrice].sum(), self._BuyVolLimit.loc[iID], -iPositionNum)
                iRemainder = (iBuyOrders["数量"].cumsum() - iClosedNum).clip_lower(0.0)
                iRemainder.iloc[1:] = iRemainder.diff().iloc[1:]
                iBuyOrders["数量"] = iRemainder.values
                RemainderOrders = RemainderOrders.append(iBuyOrders[iBuyOrders["数量"]!=0])
                self._BuyVolLimit.loc[iID] -= iClosedNum# 调整卖出限制条件
                iClosedNum = (iOpenPosition["数量"] - (iOpenPosition["数量"].cumsum() + iClosedNum).clip_upper(0)).clip_upper(0)
                iFee = iClosedNum.abs() * iTradePrice * self.BuyLimit.TradeFee
            iPNL = iClosedNum * (iTradePrice - iOpenPosition["开仓价格"])
            CashChanged[i] = (np.abs(iClosedNum) * iOpenPosition["开仓价格"] + iPNL).sum() - iFee
            iClosedPosition["数量"] = iClosedNum
            iClosedPosition["平仓时点"] = idt
            iClosedPosition["平仓价格"] = iTradePrice
            iClosedPosition["平仓交易费"] = iFee
            iClosedPosition["平仓盈亏"] = iPNL
            iOpenPosition["数量"] -= iClosedNum
            iClosedPosition = iClosedPosition[iClosedPosition["数量"]!=0]
            iClosedPosition.pop("浮动盈亏")
            ClosedPosition = ClosedPosition.append(iClosedPosition.reset_index())
            NewPosition = NewPosition.append(iOpenPosition[iOpenPosition["数量"]!=0].reset_index())
        ClosedPosition.index = np.arange(self._ClosedPosition.shape[0], self._ClosedPosition.shape[0]+ClosedPosition.shape[0])
        self._ClosedPosition = self._ClosedPosition.append(ClosedPosition)
        OpenPosition = OpenPosition.loc[OpenPosition.index.difference(set(IDs))].reset_index()
        NewPosition.index = np.arange(OpenPosition.shape[0], OpenPosition.shape[0]+NewPosition.shape[0])
        self._OpenPosition = OpenPosition.append(NewPosition)
        if self.SellLimit.ShortAllowed:
            self._OpenPosition = self._OpenPosition[self._OpenPosition["数量"].abs()>_QS_MinPositionNum]
        else:
            self._OpenPosition = self._OpenPosition[self._OpenPosition["数量"]>_QS_MinPositionNum]
        TradingRecord = list(zip([idt]*CloseOrders.shape[0], CloseOrders.index, CloseOrders, TradePrice, Fees, CashChanged, ["close"]*CloseOrders.shape[0]))
        if TradingRecord: self._QS_updateCashDebt(CashChanged.sum())
        return (TradingRecord, RemainderOrders)
    # 撮合成交限价开仓单, TODO
    # 限价单的报价如果优于当前时段的成交价则可成交, 否则继续挂单
    def _matchLimitOpenOrder(self, idt, orders):
        return ([], pd.DataFrame(columns=["ID", "数量", "目标价"]))

