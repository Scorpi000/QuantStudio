# coding=utf-8
"""账户"""
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

# 基于价格和成交额数据的简单账户
# market_ft: 提供价格数据的因子表, 时间频率任意
# 市价单的成交价为当前时段的指定成交价, 如果指定了成交量, 则成交量上限为当前时段成交量的一定比例, 否则无上限
# 限价单的报价如果优于当前时段的指定成交价则可成交, 否则继续挂单, 如果指定了成交量, 则成交量上限为当前时段成交量的一定比例; (TODO)
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


# 基于 Bar 数据的账户(TODO)
# 行情因子表: 开盘价(非必需), 最高价(非必需), 最低价(非必需), 最新价(必需), 成交量(非必需), 成交价(必需). 最新价用于记录账户价值变化; 成交价: 用于模拟市价单的成交.
# 市价单的成交价为当前时段的指定成交价, 如果指定了成交量, 则成交量上限为当前时段成交量的一定比例;
# 限价单根据最高价, 最低价和成交量的情况成交, 假定成交量在最高价和最低价之间的分布为均匀分布, 如果没有指定最高价和最低价, 则以成交价作为最高价和最低价
class _BarMarketInfo(_MarketInfo):
    """Bar 行情信息"""
    Open = Enum(None, arg_type="SingleOption", label="开盘价", order=3)
    High = Enum(None, arg_type="SingleOption", label="最高价", order=4)
    Low = Enum(None, arg_type="SingleOption", label="最低价", order=5)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._MarketFT.getFactorMetaData(key="DataType")))
        DefaultNumFactorList.insert(0, None)
        self.add_trait("Open", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="开盘价", order=3))
        self.add_trait("High", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="最高价", order=4))
        self.add_trait("Low", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="最低价", order=5))

class TimeBarAccount(SimpleAccount):
    """基于 Bar 数据的账户"""
    def __init__(self, market_ft, name="TimeBarAccount", sys_args={}, config_file=None, **kwargs):
        return super().__init__(market_ft=market_ft, name=name, sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.MarketInfo = _BarMarketInfo(self._MarketFT)
    # 当前的 Bar 数据, DataFrame(float, index=[ID], columns=["开盘价", "最高价", "最低价", "最新价", "成交量"])
    @property
    def Bar(self):
        return self._MarketFT.readData(factor_names=[self.MarketInfo.Open, self.MarketInfo.High, self.MarketInfo.Low, self.MarketInfo.Last, self.MarketInfo.Vol], 
                                       dts=[self._Model.DateTime], ids=self._IDs).iloc[:,0,:]
    # 更新账户信息
    def _QS_updateAccount(self, cash_changed, position):
        self._QS_updateCashDebt(cash_changed)
        position[position.abs()<1e-6] = 0.0
        self._PositionNum[iIndex] = position.values
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
        PositionNum = self.PositionNum
        sell_orders = np.minimum(sell_orders, PositionNum.loc[sell_orders.index])# 卖出数量不能超过当前持仓数量
        sell_orders = sell_orders[sell_orders>0]
        if sell_orders.shape[0]==0: return []
        CashChanged = 0.0
        TradingRecord = []
        IDs = list(sell_orders.index)
        SellPrice = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketInfo.TradePrice]).iloc[0, 0]
        SellLimit = pd.Series(np.zeros(sell_orders.shape[0])+np.inf, index=IDs)
        SellLimit[pd.isnull(SellPrice)] = 0.0# 成交价缺失的不能卖出
        if self.SellLimit.LimitIDFilter: SellLimit[self._MarketFT.getIDMask(idt, ids=IDs, id_filter_str=self.SellLimit.LimitIDFilter)] = 0.0# 满足禁止条件的不能卖出, TODO
        if self.MarketInfo.Vol: SellLimit = np.minimum(SellLimit, self._MarketFT.readData(factor_names=[self.MarketInfo.Vol], ids=IDs, dts=[idt]).iloc[0, 0] * self.SellLimit.MarketOrderVolumeLimit)# 成交量限制
        SellNums = np.minimum(SellLimit, sell_orders)
        if self.SellLimit.MinUnit!=0.0: SellNums = (SellNums / self.SellLimit.MinUnit).astype("int") * self.SellLimit.MinUnit# 最小交易单位限制
        SellAmounts = SellNums * SellPrice
        SellFees = SellAmounts * self.SellLimit.TradeFee
        Mask = (SellNums>0)
        PositionNum[SellNums.index] -= SellNums
        CashChanged += SellAmounts.sum() - SellFees.sum()
        TradingRecord.extend(list(zip([idt]*Mask.sum(), SellNums[Mask].index, -SellNums[Mask], SellPrice[Mask], SellFees[Mask], (SellAmounts-SellFees)[Mask], ["close"]*Mask.sum())))
        if TradingRecord: self._QS_updateAccount(CashChanged, PositionNum)# 更新账户信息
        return TradingRecord
    # 撮合成交买入市价单
    # 以成交价完成成交, 成交量满足买入限制要求
    # 未成交的市价单自动撤销
    def _matchMarketBuyOrder(self, idt, buy_orders):
        if buy_orders.shape[0]==0: return []
        CashChanged = 0.0
        TradingRecord = []
        IDs = list(buy_orders.index)
        BuyPrice = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketInfo.TradePrice]).iloc[0, 0]
        BuyLimit = pd.Series(np.zeros(buy_orders.shape[0])+np.inf, index=IDs)
        BuyLimit[pd.isnull(BuyPrice)] = 0.0# 成交价缺失的不能买入
        if self.BuyLimit.LimitIDFilter: BuyLimit[self._MarketFT.getIDMask(idt, ids=IDs, id_filter_str=self.BuyLimit.LimitIDFilter)] = 0.0# 满足禁止条件的不能卖出
        if self.MarketInfo.Vol: BuyLimit = np.minimum(BuyLimit, self._MarketFT.readData(dts=[idt], factor_names=[self.MarketInfo.Vol], ids=IDs).iloc[0, 0] * self.BuyLimit.MarketOrderVolumeLimit)# 成交量限制
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
            PositionNum = self.PositionNum
            PositionNum[BuyNums.index] += BuyNums
            CashChanged -= BuyAmounts.sum() + BuyFees.sum()
            TradingRecord.extend(list(zip([idt]*Mask.sum(), BuyNums[Mask].index, BuyNums[Mask], BuyPrice[Mask], BuyFees[Mask], -(BuyAmounts+BuyFees)[Mask], ["open"]*Mask.sum())))
        if TradingRecord: self._QS_updateAccount(CashChanged, PositionNum)
        return TradingRecord
    # 撮合成交卖出限价单
    # 如果最高价和最低价未指定, 则检验成交价是否优于目标价, 是就以目标价完成成交, 且成交量满足卖出限制中的限价单成交量限比, 否则无法成交
    # 如果指定了最高价和最低价, 则假设成交价服从[最低价, 最高价]中的均匀分布, 据此确定可完成成交的数量
    # 未成交的限价单继续保留
    def _matchLimitSellOrder(self, idt, sell_orders):
        if sell_orders.shape[0]==0: return ([], sell_orders)
        IDs = list(sell_orders.index.unique())
        if self.MarketInfo.Vol:
            Volume = self._MarketFT.readData(factor_names=[self.MarketInfo.Vol], ids=IDs, dts=[idt]).iloc[0, 0]*self.BuyLimit.LimitOrderVolumeLimit
        else:
            Volume = pd.Series(np.inf, index=IDs)
        if self.MarketInfo.High and self.MarketInfo.Low:
            High = self._MarketFT.readData(factor_names=[self.MarketInfo.High, self.MarketInfo.Low], ids=IDs, dts=[idt]).iloc[:, 0]
            Low, High = High.iloc[1], High.iloc[0]
        else:
            High = Low = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketInfo.TradePrice]).iloc[0, 0]
        PositionNum = self.PositionNum
        SellNums = np.zeros(Volume.shape)
        SellAmounts = np.zeros(Volume.shape)
        for i, iID in enumerate(IDs):
            iPosition = PositionNum[iID]
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
            PositionNum[iID] = iPosition
        sell_orders = sell_orders[sell_orders["数量"]<0]
        sell_orders = sell_orders.reset_index().loc[:, ["ID", "数量", "目标价"]]
        SellFees = SellAmounts*self.SellLimit.TradeFee
        CashChanged = SellAmounts.sum() - SellFees.sum()
        Mask = (SellNums>0)
        TradingRecord = list(zip([idt]*Mask.sum(), Volume.index[Mask], -SellNums[Mask], (SellAmounts/SellNums)[Mask], SellFees[Mask], (SellAmounts-SellFees)[Mask], ["close"]*Mask.sum()))
        if TradingRecord: self._QS_updateAccount(CashChanged, PositionNum)
        return (TradingRecord, sell_orders)
    # 撮合成交买入限价单
    # 如果最高价和最低价未指定, 则检验成交价是否优于目标价, 是就以目标价完成成交, 且成交量满足买入限制要求
    # 如果指定了最高价和最低价, 则假设成交价服从[最低价, 最高价]中的均匀分布, 据此确定可完成成交的数量, 同时满足买入限制要求
    # 未成交的限价单继续保留
    def _matchLimitBuyOrder(self, idt, buy_orders):
        if buy_orders.shape[0]==0: return ([], buy_orders)
        buy_orders = buy_orders.sort_values(by=["目标价"], ascending=False)
        IDs = list(buy_orders.index.unique())
        if self.MarketInfo.Vol:
            Volume = self._MarketFT.readData(factor_names=[self.MarketInfo.Vol], ids=IDs, dts=[idt]).iloc[0, 0]*self.BuyLimit.LimitOrderVolumeLimit
        else:
            Volume = pd.Series(np.inf, index=IDs)
        if self.MarketInfo.High and self.MarketInfo.Low:
            High = self._MarketFT.readData(factor_names=[self.MarketInfo.High, self.MarketInfo.Low], ids=IDs, dts=[idt]).iloc[:, 0]
            Low, High = High.iloc[1], High.iloc[0]
        else:
            High = Low = self._MarketFT.readData(dts=[idt], ids=IDs, factor_names=[self.MarketInfo.TradePrice]).iloc[0, 0]
        TargetAmounts = (buy_orders["目标价"]*buy_orders["数量"]).groupby(level=0).sum()
        TotalAmount = TargetAmounts.sum()
        AllocatedAmounts = np.min((TotalAmount*(1+self.BuyLimit.TradeFee), self.AvailableCash))*TargetAmounts/TotalAmount
        PositionNum = self.PositionNum
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
            PositionNum[iID] += iPosition
        buy_orders = buy_orders[buy_orders["数量"]>0]
        buy_orders = buy_orders.reset_index().loc[:, ["ID", "数量", "目标价"]]
        Fees = BuyAmounts * self.BuyLimit.TradeFee
        CashChanged = - BuyAmounts.sum() - Fees.sum()
        Mask = (BuyNums>0)
        TradingRecord = list(zip([idt]*Mask.sum(), Volume.index[Mask], BuyNums[Mask], (BuyAmounts/BuyNums)[Mask], Fees[Mask], (-BuyAmounts-Fees)[Mask], ["close"]*Mask.sum()))
        if TradingRecord: self._QS_updateAccount(CashChanged, PositionNum)
        return (TradingRecord, buy_orders)


# 基于 Tick 数据的账户(TODO)
# 行情因子表: 开盘价, 最高价, 最低价, 最新价, 成交价, 成交量, 买价1..., 卖价1..., 买量1..., 卖量1.... 最新价用于记录账户价值变化; 买卖盘: 用于模拟市价单的成交
# 市价单的成交价为当前时段的指定成交价, 如果指定了成交量, 则成交量上限为当前时段成交量的一定比例;
# 限价单根据最高价, 最低价和成交量的情况成交, 假定成交量在最高价和最低价之间的分布为均匀分布, 如果没有指定最高价和最低价, 则以成交价作为最高价和最低价
class _TickMarketInfo(_BarMarketInfo):
    """Tick 因子映照"""
    Bid = Str(arg_type="String", label="买盘价格", order=6)
    BidVol = Str(arg_type="String", label="买盘数量", order=7)
    Ask = Str(arg_type="String", label="卖盘价格", order=8)
    AskVol = Str(arg_type="String", label="卖盘数量", order=9)

class TickAccount(TimeBarAccount):
    """基于 Tick 数据的账户"""
    def __init__(self, market_ft, name="TickAccount", sys_args={}, config_file=None, **kwargs):
        return super().__init__(market_ft=market_ft, name=name, sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.MarketInfo = _TickMarketInfo(self._MarketFT)
    def __QS_start__(self, mdl, dts, **kwargs):
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._Bid = []# 买盘价格因子
        self._BidVol = []# 买盘挂单量因子
        self._Ask = []# 卖盘价格因子
        self._AskVol = []# 卖盘挂单量因子
        for iFactorName in self._MarketFT.FactorNames:
            if iFactorName.find(self.MarketInfo.Bid)!=-1: self._Bid.append(iFactorName)
            elif iFactorName.find(self.MarketInfo.BidVol)!=-1: self._BidVol.append(iFactorName)
            elif iFactorName.find(self.MarketInfo.Ask)!=-1: self._Ask.append(iFactorName)
            elif iFactorName.find(self.MarketInfo.AskVol)!=-1: self._AskVol.append(iFactorName)
        self._Bid.sort()
        self._BidVol.sort()
        self._Ask.sort()
        self._AskVol.sort()
        self._Depth = len(self._Bid)# 行情深度
        return 0
    # 撮合成交卖出市价单
    def _matchMarketSellOrder(self, idt, sell_orders):
        sell_orders = -sell_orders
        PositionNum = self.PositionNum
        sell_orders = np.minimum(sell_orders, PositionNum.loc[sell_orders.index])# 卖出数量不能超过当前持仓数量
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
        PositionNum[sell_orders.index] -= SellNums
        CashChanged = SellAmounts.sum() - SellFees.sum()
        TradingRecord = list(zip([idt]*SellNums.shape[0], SellAmounts.index, -SellNums, SellAmounts/SellNums, SellFees, SellAmounts-SellFees, ["close"]*SellNums.shape[0]))
        if TradingRecord: self._QS_updateAccount(CashChanged, PositionNum)# 更新账户信息
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
        PositionNum = self.PositionNum
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
        PositionNum[BuyNums.index] += BuyNums
        CashChanged = -(BuyAmounts.sum() + BuyFees.sum())
        TradingRecord = list(zip([idt]*BuyNums.shape[0], BuyNums.index, BuyNums, BuyAmounts/BuyNums, BuyFees, -(BuyAmounts+BuyFees), ["open"]*BuyNums.shape[0]))
        if TradingRecord: self._QS_updateAccount(CashChanged, PositionNum)
        return TradingRecord
