# -*- coding: utf-8 -*-
"""新旧共享现金账户复用的交易限制、撮合和订单转换数值函数。"""
import numpy as np
import pandas as pd


def trade_limits(position_num, last_price, buy_price, buy_limit, buy_amt_limit, sell_price, sell_limit, sell_amt_limit, *, short_allowed=False):
    """将报价、交易禁止条件和成交额上限转换为数量上限。"""
    SellVolLimit, BuyVolLimit = np.full(shape=last_price.shape, fill_value=np.inf), np.full(shape=last_price.shape, fill_value=np.inf)
    BuyVolLimit[~ (buy_price > 0)] = 0.0# 买入成交价缺失的不能买入
    SellVolLimit[~ (sell_price > 0)] = 0.0# 卖出成交价缺失的不能卖出
    if buy_limit is not None:# 满足买入禁止条件的不能买入
        BuyVolLimit[buy_limit == 1] = 0.0
    if sell_limit is not None:# 满足卖出禁止条件的不能卖出
        SellVolLimit[sell_limit == 1] = 0.0
    if buy_amt_limit is not None:# 指定了买入成交额, 成交额满足限制要求
        BuyVolLimit = np.clip(BuyVolLimit, a_min=None, a_max=buy_amt_limit / buy_price)
    if sell_amt_limit is not None:# 指定了卖出成交额, 成交额满足限制要求
        SellVolLimit = np.clip(SellVolLimit, a_min=None, a_max=sell_amt_limit / sell_price)
    if not short_allowed:
        SellVolLimit = np.clip(SellVolLimit, a_min=None, a_max=np.clip(position_num, a_min=0.0, a_max=None))
    return BuyVolLimit, SellVolLimit


def match_market_orders(available_cash, position_num, order_num, buy_price, buy_vol_limit, buy_fee, sell_price, sell_vol_limit, sell_fee):
    """共享现金账户按既有规则先卖后买，资金不足时按比例分配。"""
    order_num = np.clip(order_num, a_max=buy_vol_limit, a_min=-sell_vol_limit)# 过滤限制条件
    # 先执行卖出交易
    SellAmounts = np.abs(np.clip(sell_price * order_num, a_min=None, a_max=0))
    SellFees = SellAmounts * sell_fee# 卖出交易费
    CashChanged = SellAmounts - SellFees
    Mask = (SellAmounts > 0)
    SellNums = np.clip(order_num, a_min=None, a_max=0)
    TradeNum, TradePrice, Fee = np.where(Mask, SellNums, 0), np.where(Mask, sell_price, np.nan), np.where(Mask, SellFees, 0)
    # 再执行买入交易
    BuyAmounts = np.clip(buy_price * order_num, a_min=0, a_max=None)
    CashAcquired = BuyAmounts * (1 + buy_fee)
    TotalCashAcquired = np.nansum(CashAcquired)
    if TotalCashAcquired > 0:
        AvailableCash = available_cash + np.nansum(CashChanged)
        CashAllocated = min(AvailableCash, TotalCashAcquired) * CashAcquired / TotalCashAcquired
        BuyAmounts = CashAllocated / (1 + buy_fee)
        BuyFees = BuyAmounts * buy_fee
        BuyNums = BuyAmounts / buy_price
        Mask = (BuyAmounts > 0)
        TradeNum, TradePrice, Fee = np.where(Mask, BuyNums, TradeNum), np.where(Mask, buy_price, TradePrice), np.where(Mask, BuyFees, Fee)
    else:
        CashAllocated = BuyNums = np.zeros(shape=BuyAmounts.shape)
    # 更新持仓数量和现金
    BuyNums[np.isnan(BuyNums)] = 0
    SellNums[np.isnan(SellNums)] = 0
    PositionNum = position_num.copy()
    # TotalAmount = np.nansum(buy_price * np.clip(PositionNum, a_min=None, a_max=0) + sell_price * np.clip(PositionNum, a_min=0, a_max=None))
    # Turnover = (np.nansum(SellAmounts) + np.nansum(BuyAmounts)) / (TotalAmount + available_cash)
    PositionNum = PositionNum + BuyNums + SellNums
    Cash = available_cash + np.nansum(CashChanged) - np.nansum(CashAllocated)
    return Cash, PositionNum, TradeNum, TradePrice, Fee


def target_weight_orders(signal:np.ndarray, price:np.ndarray, cash:float, position_amt:np.ndarray) -> np.ndarray:
    """沿用既有目标权重转数量规则。"""
    AccountValue = abs(cash + np.nansum(position_amt))
    NaMask = pd.isnull(signal)
    if np.all(NaMask): return np.full_like(signal, np.nan)
    return (np.where(NaMask, 0, signal) * AccountValue - np.where(pd.isnull(position_amt), 0, position_amt)) / price
