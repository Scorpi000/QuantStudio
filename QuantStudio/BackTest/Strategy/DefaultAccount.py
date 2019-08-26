# coding=utf-8
"""默认账户"""
import datetime as dt

import pandas as pd
import numpy as np
from traits.api import Enum, ListStr, Float, Str, Int, Bool, Instance

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.BackTest.Strategy.StrategyModule import Account, cutDateTime
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList

class _TradeLimit(__QS_Object__):
    """交易限制"""
    #TradePrice = Enum(None, arg_type="SingleOption", label="成交价", order=0)
    LimitIDFilter = Str(arg_type="IDFilter", label="禁止条件", order=1)
    TradeFee = Float(0.003, arg_type="Double", label="交易费率", order=2)
    #Amt = Enum(None, arg_type="SingleOption", label="成交额", order=3)
    AmtLimitRatio = Float(0.1, arg_type="Double", label="成交额限比", order=4)
    def __init__(self, account, direction, sys_args={}, config_file=None, **kwargs):
        self._Account = account
        self._Direction = direction
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList = [None] + self._Account.trait("Last").option_range
        self.add_trait("TradePrice", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="成交价", order=0))
        self.add_trait("Amt", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="成交额", order=3))
        if self._Direction=="Sell": self.add_trait("ShortAllowed", Bool(False, arg_type="SingleOption", label="允许卖空", order=5))

# 基于价格和成交额数据的简单账户, 只支持市价单
# market_ft: 提供价格数据的因子表, 时间频率任意
# 市价单的成交价为当前时段的指定成交价, 如果指定了成交额, 则成交额上限为当前时段成交额的一定比例, 否则无上限
class DefaultAccount(Account):
    """默认证券账户"""
    Delay = Bool(True, arg_type="Bool", label="交易延迟", order=2)
    TargetIDs = ListStr(arg_type="IDList", label="目标ID", order=3)
    BuyLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="买入限制", order=4)
    SellLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="卖出限制", order=5)
    #Last = Enum(None, arg_type="SingleOption", label="最新价", order=6)
    def __init__(self, market_ft, name="默认证券账户", sys_args={}, config_file=None, **kwargs):
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
        self._Turnover = None# 换手率, Series(index=[时间点])
        self._Orders = None# 当前接收到的订单, DataFrame(columns=["ID", "数量", "目标价"])
        self._LastPrice = None# 最新价, Series(index=self._IDs)
        self._BuyPrice = None# 买入成交价, Series(index=self._IDs)
        self._SellPrice = None# 卖出成交价, Series(index=self._IDs)
        self._SellVolLimit = None# 卖出成交量限制, Series(index=self._IDs)
        self._BuyVolLimit = None# 买入成交量限制, Series(index=self._IDs)
        self._MarketFT = market_ft# 提供净值或者收益率数据的因子表
        self._TempData = {}# 临时数据
        return super().__init__(name=name, sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._MarketFT.getFactorMetaData(key="DataType")))
        self.add_trait("Last", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="最新价", order=6, option_range=DefaultNumFactorList))
        self.Last = searchNameInStrList(DefaultNumFactorList, ['新','收','Last','last','close','Close'])
        self.BuyLimit = _TradeLimit(account=self, direction="Buy")
        self.SellLimit = _TradeLimit(account=self, direction="Sell")
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._IDs = list(self.TargetIDs)
        if not self._IDs: self._IDs = self._MarketFT.getID(ifactor_name=self.Last)
        nDT, nID = len(dts), len(self._IDs)
        #self._Cash = np.zeros(nDT+1)
        #self._FrozenCash = 0
        #self._Debt = np.zeros(nDT+1)
        self._PositionNum = pd.DataFrame(np.zeros((nDT+1, nID)), index=[dts[0]-dt.timedelta(1)]+dts, columns=self._IDs)
        self._PositionAmount = self._PositionNum.copy()
        self._Turnover = pd.Series(np.zeros((nDT, )), index=dts)
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        self._TempData = {}
        if self.BuyLimit.TradePrice is None: self.BuyLimit.TradePrice, self._TempData["BuyPrice"] = self.Last, None
        if self.SellLimit.TradePrice is None: self.SellLimit.TradePrice, self._TempData["SellPrice"] = self.Last, None
        self._LastPrice = self._BuyPrice = self._SellPrice = None
        self._iTradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])# 暂存的交易记录
        self._SellVolLimit, self._BuyVolLimit = pd.Series(np.inf, index=self._IDs), pd.Series(np.inf, index=self._IDs)
        self._nDT = nDT
        return Rslt + (self._MarketFT, )
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return self._iTradingRecord
        super().__QS_move__(idt, **kwargs)
        # 更新当前的账户信息
        iIndex = self._Model.DateTimeIndex
        self._LastPrice = self._MarketFT.readData(factor_names=[self.Last], ids=self._IDs, dts=[idt]).iloc[0, 0]
        self._PositionNum.iloc[iIndex+1] = self._PositionNum.iloc[iIndex]# 初始化持仓
        if self.Delay:# 撮合成交
            self._iTradingRecord = self._matchOrder(idt)
            self._iTradingRecord = pd.DataFrame(self._iTradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(self._iTradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(self._iTradingRecord)
        self._PositionAmount.iloc[iIndex+1] = self._PositionNum.iloc[iIndex+1] * self._LastPrice
        return self._iTradingRecord
    def __QS_after_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        super().__QS_after_move__(idt, **kwargs)
        if not self.Delay:# 撮合成交
            self._iTradingRecord = self._matchOrder(idt)
            self._iTradingRecord = pd.DataFrame(self._iTradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(self._iTradingRecord)), columns=self._TradingRecord.columns)
            self._TradingRecord = self._TradingRecord.append(self._iTradingRecord)
            iIndex = self._Model.DateTimeIndex
            self._PositionAmount.iloc[iIndex+1] = self._PositionNum.iloc[iIndex+1] * self._LastPrice
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        super().__QS_end__()
        self.BuyLimit.TradePrice = self._TempData.pop("BuyPrice", self.BuyLimit.TradePrice)
        self.SellLimit.TradePrice = self._TempData.pop("SellPrice", self.SellLimit.TradePrice)
        self._Output["持仓数量"] = self.getPositionNumSeries()
        self._Output["持仓金额"] = self.getPositionAmountSeries()
        self._Output["换手率"] = pd.DataFrame(self._Turnover.values, index=self._Turnover.index, columns=["换手率"])
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
            if pd.notnull(target_price): self._QS_Logger.warning("账户: '%s' 不支持限价单, 限价单将自动转为市价单!" % self.Name)
            return (self._Orders.shape[0], target_id, num, target_price)
        if combined_order is not None:
            if pd.notnull(combined_order["目标价"]).sum()>0: self._QS_Logger.warning("本账户: '%s' 不支持限价单, 限价单将自动转为市价单!" % self.Name)
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
    # 更新交易限制条件
    def _updateTradeLimit(self, idt):
        self._SellVolLimit[:] = self._BuyVolLimit[:] = np.inf
        TradePrice = self._MarketFT.readData(factor_names=[self.BuyLimit.TradePrice, self.SellLimit.TradePrice], ids=self._IDs, dts=[idt]).iloc[:,0,:]
        self._BuyPrice, self._SellPrice = TradePrice.iloc[:, 0], TradePrice.iloc[:, 1]
        self._BuyVolLimit[pd.isnull(self._BuyPrice) | (self._BuyPrice<=0)] = 0.0# 买入成交价缺失的不能买入
        self._SellVolLimit[pd.isnull(self._SellPrice) | (self._SellPrice<=0)] = 0.0# 卖出成交价缺失的不能卖出
        if self.SellLimit.LimitIDFilter:# 满足卖出禁止条件的不能卖出
            Mask = self._MarketFT.getIDMask(idt, ids=self._IDs, id_filter_str=self.SellLimit.LimitIDFilter)
            self._SellVolLimit[Mask] = 0.0
        if self.BuyLimit.LimitIDFilter:# 满足买入禁止条件的不能买入
            Mask = self._MarketFT.getIDMask(idt, ids=self._IDs, id_filter_str=self.BuyLimit.LimitIDFilter)
            self._BuyVolLimit[Mask] = 0.0
        if self.BuyLimit.Amt is not None:# 指定了买入成交额, 成交额满足限制要求
            Amount = self._MarketFT.readData(factor_names=[self.BuyLimit.Amt], ids=self._IDs, dts=[idt]).iloc[0,0,:]
            self._BuyVolLimit = self._BuyVolLimit.clip_upper(Amount * self.BuyLimit.AmtLimitRatio / self._BuyPrice)
        if self.SellLimit.Amt is not None:# 指定了卖出成交额, 成交额满足限制要求
            Amount = self._MarketFT.readData(factor_names=[self.SellLimit.Amt], ids=self._IDs, dts=[idt]).iloc[0,0,:]
            self._SellVolLimit = self._SellVolLimit.clip_upper(Amount * self.SellLimit.AmtLimitRatio / self._SellPrice)
        if not self.SellLimit.ShortAllowed:
            PositionNum = self._PositionNum.iloc[self._Model.DateTimeIndex+1]
            self._SellVolLimit = self._SellVolLimit.clip_upper(PositionNum.clip_lower(0.0))
        return 0
    # 撮合成交订单
    def _matchOrder(self, idt):
        if self._Orders.shape[0]==0: return []
        self._updateTradeLimit(idt)
        MarketOrders = self._Orders.groupby(by=["ID"]).sum()["数量"]# Series(数量, index=[ID])
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        return self._matchMarketOrder(idt, MarketOrders)
    # 撮合成交市价单
    # 以成交价完成成交, 满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketOrder(self, idt, orders):
        IDs = orders.index.tolist()
        orders = orders.clip(upper=self._BuyVolLimit.loc[IDs], lower=-self._SellVolLimit.loc[IDs])# 过滤限制条件
        orders = orders[orders!=0]
        if orders.shape[0]==0: return []
        # 先执行卖出交易
        SellPrice = self._SellPrice[orders.index]
        SellAmounts = (SellPrice * orders).clip_upper(0).abs()
        Fees = SellAmounts * self.SellLimit.TradeFee# 卖出交易费
        CashChanged = SellAmounts - Fees
        Mask = (SellAmounts>0)
        SellNums = orders.clip_upper(0)
        TradingRecord = list(zip([idt]*Mask.sum(), orders.index[Mask], SellNums[Mask], SellPrice[Mask], Fees[Mask], CashChanged[Mask], ["sell"]*Mask.sum()))
        # 再执行买入交易
        BuyPrice = self._BuyPrice[orders.index]
        BuyAmounts = (BuyPrice * orders).clip_lower(0)
        CashAcquired = BuyAmounts * (1 + self.BuyLimit.TradeFee)
        TotalCashAcquired = CashAcquired.sum()
        if TotalCashAcquired>0:
            AvailableCash = self.AvailableCash + CashChanged.sum()
            CashAllocated = min(AvailableCash, TotalCashAcquired) * CashAcquired / TotalCashAcquired
            BuyAmounts = CashAllocated / (1 + self.BuyLimit.TradeFee)
            Fees = BuyAmounts * self.BuyLimit.TradeFee
            BuyNums = BuyAmounts / BuyPrice
            Mask = (BuyAmounts>0)
            TradingRecord.extend((zip([idt]*Mask.sum(), orders.index[Mask], BuyNums[Mask], BuyPrice[Mask], Fees[Mask], -CashAllocated[Mask], ["buy"]*Mask.sum())))
        else:
            CashAllocated = BuyNums = pd.Series(0, index=BuyAmounts.index)
        # 更新持仓数量和现金
        iIndex = self._Model.DateTimeIndex
        iPosition = self._PositionNum.iloc[iIndex+1].copy()
        TotalAmount = (self._BuyPrice * iPosition.clip_upper(0) + self._SellPrice * iPosition.clip_lower(0)).sum()
        self._Turnover.iloc[iIndex] = (SellAmounts.sum() + BuyAmounts.sum()) / (TotalAmount + super().AccountValue)
        iPosition[orders.index] += BuyNums + SellNums
        self._PositionNum.iloc[iIndex+1] = iPosition
        self._QS_updateCashDebt(CashChanged.sum() - CashAllocated.sum())
        return TradingRecord