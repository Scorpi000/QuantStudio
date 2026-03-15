# -*- coding: utf-8 -*-
import base64
from io import BytesIO
import datetime as dt
from typing import Optional, Literal, List

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured
from pydantic import Field
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from QuantStudio.Core import __QS_Error__
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.FactorOperation import SectionOperation, SectionOperator, PanelOperation, PanelOperator
from QuantStudio.BackTest.BackTestModel import BTInitData, BTLocalContext, BTNode
from QuantStudio.Tools.StrategyTestFun import summaryStrategy, calcYieldSeq, calcLSYield, formatStrategySummary


def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x * 100, )

_QS_MinPositionNum = 1e-8# 会被忽略掉的最小持仓数量
_QS_MinCash = 1e-8# 会被忽略掉的最小现金量

class CalcSimpleAccount(PanelOperator):
    """简单账户计算算子"""

    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=3, title="入参数", frozen=True)

    def __init__(self, signal_type:Literal["买卖数量", "目标权重"]="买卖数量", short_allowed:bool=False, start_dt:Optional[dt.datetime]=None, args:dict = {}, config_file:Optional[str] = None, **kwargs):
        """初始化简单账户计算算子

        Args:
            signal_type: 信号类型
            short_allowed: 是否允许卖空
            start_dt: 净值开始日, 如果为 None, 表示从计算的第一个时点开始
        """
        Arity = args.get("Arity", None) or 3
        Args = {"Name": "calcSimpleAccount"} | args | {"DTMode": "单时点", "OutputMode": "全截面", "DataType": "object", "iInitFactor": 0}
        Args["ModelArgs"] = {"short_allowed": short_allowed, "signal_type": signal_type} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [None])[0]] * Arity
        Args["LookBack"] = [1, 0, 0] + [0] * max(0, Arity - 3)
        Args["LookBackMode"] = ["扩张窗口"] + ["滚动窗口"] * max(0, Arity - 1)
        Args["StartDT"] = [start_dt] + [None] * max(0, Arity - 1)
        Args["CompoundType"] = [("Cash", float), ("Position", float), ("Amount", float), ("Turnover", float)]
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    # 更新交易限制条件
    def _getTradeLimit(self, position_num, last_price, buy_price, buy_limit, buy_amt_limit, sell_price, sell_limit, sell_amt_limit):
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
        if not self._QSArgs.ModelArgs["short_allowed"]:
            SellVolLimit = np.clip(SellVolLimit, a_min=None, a_max=np.clip(position_num, a_min=0.0, a_max=None))
        return BuyVolLimit, SellVolLimit

    # 撮合成交市价单
    # 以成交价完成成交, 满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketOrder(self, available_cash, position_num, ids, order_num, buy_price, buy_vol_limit, buy_fee, sell_price, sell_vol_limit, sell_fee):
        order_num = np.clip(order_num, a_max=buy_vol_limit, a_min=-sell_vol_limit)# 过滤限制条件
        # 先执行卖出交易
        SellAmounts = np.abs(np.clip(sell_price * order_num, a_min=None, a_max=0))
        Fees = SellAmounts * sell_fee# 卖出交易费
        CashChanged = SellAmounts - Fees
        Mask = (SellAmounts > 0)
        SellNums = np.clip(order_num, a_min=None, a_max=0)
        SellTradingRecord = {
            "ID": ids[Mask],
            "Num": SellNums[Mask],
            "Price": sell_price[Mask],
            "Fee": Fees[Mask],
            "CashChanged": CashChanged[Mask],
            "Direction": ["sell"] * np.sum(Mask)
        }
        # 再执行买入交易
        BuyAmounts = np.clip(buy_price * order_num, a_min=0, a_max=None)
        CashAcquired = BuyAmounts * (1 + buy_fee)
        TotalCashAcquired = np.nansum(CashAcquired)
        if TotalCashAcquired > 0:
            AvailableCash = available_cash + np.nansum(CashChanged)
            CashAllocated = min(AvailableCash, TotalCashAcquired) * CashAcquired / TotalCashAcquired
            BuyAmounts = CashAllocated / (1 + buy_fee)
            Fees = BuyAmounts * buy_fee
            BuyNums = BuyAmounts / buy_price
            Mask = (BuyAmounts > 0)
            BuyTradingRecord = {
                "ID": ids[Mask],
                "Num": BuyNums[Mask],
                "Price": buy_price[Mask],
                "Fee": Fees[Mask],
                "CashChanged": - CashAllocated[Mask],
                "Direction": ["buy"] * np.sum(Mask)
            }
        else:
            CashAllocated = BuyNums = np.zeros(shape=BuyAmounts.shape)
        # 更新持仓数量和现金
        PositionNum = position_num.copy()
        TotalAmount = np.nansum(buy_price * np.clip(PositionNum, a_min=None, a_max=0) + sell_price * np.clip(PositionNum, a_min=0, a_max=None))
        Turnover = (np.nansum(SellAmounts) + np.nansum(BuyAmounts)) / (TotalAmount + available_cash)
        PositionNum = PositionNum + BuyNums + SellNums
        Cash = available_cash + np.nansum(CashChanged) - np.nansum(CashAllocated)
        return Cash, PositionNum, Turnover, BuyTradingRecord, SellTradingRecord

    # 目标权重信号转买卖数量
    def _TargetWeightSignal2OrderNum(self, signal:np.ndarray, price:np.ndarray, cash:float, position_amt:np.ndarray) -> np.ndarray:
        AccountValue = abs(cash + np.nansum(position_amt))
        NaMask = pd.isnull(signal)
        if np.all(NaMask): return np.full_like(signal, np.nan)
        return (signal * AccountValue - position_amt) / price

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        ModelArgs = f._QSArgs.ModelArgs
        LastAccount = x[0].astype(self._QSArgs.CompoundType)[0]
        Cash, PositionNum = LastAccount["Cash"][0], LastAccount["Position"]
        LastPrice, Signal = x[1][0], x[2][0]
        if not np.any(np.abs(Signal) > 0):# 没有交易信号
            Rslt = np.array([np.full(shape=PositionNum.shape, fill_value=Cash), PositionNum, PositionNum * LastPrice, np.zeros_like(PositionNum)]).T
            return unstructured_to_structured(Rslt, dtype=np.dtype(self._QSArgs.CompoundType)).astype("O")
        if self._QSArgs.ModelArgs["signal_type"]=="买卖数量":
            OrderNum = Signal
        elif self._QSArgs.ModelArgs["signal_type"]=="目标权重":
            OrderNum = self._TargetWeightSignal2OrderNum(Signal, LastPrice, Cash, PositionNum * LastPrice)
        else:
            raise __QS_Error__(f"不支持的信号类型: {self._QSArgs.ModelArgs['signal_type']}")
        _, x = ((x[3][0], x[4:]) if ModelArgs["target_price"] else (None, x[3:]))
        BuyPrice, x = (x[0][0], x[1:]) if ModelArgs["buy_price"] else (LastPrice, x)
        BuyLimit, x = (x[0][0], x[1:]) if ModelArgs["buy_limit"] else (None, x)
        BuyFee, x = (x[0][0], x[1:]) if ModelArgs["buy_fee"] else (None, x)
        BuyAmtLimit, x = (x[0][0], x[1:]) if ModelArgs["buy_amt_limit"] else (None, x)
        SellPrice, x = (x[0][0], x[1:]) if ModelArgs["sell_price"] else (LastPrice, x)
        SellLimit, x = (x[0][0], x[1:]) if ModelArgs["sell_limit"] else (None, x)
        SellFee, x = (x[0][0], x[1:]) if ModelArgs["sell_fee"] else (None, x)
        SellAmtLimit, x = (x[0][0], x[1:]) if ModelArgs["sell_amt_limit"] else (None, x)
        # 撮合成交
        BuyVolLimit, SellVolLimit = self._getTradeLimit(PositionNum, LastPrice, BuyPrice, BuyLimit, BuyAmtLimit, SellPrice, SellLimit, SellAmtLimit)
        Cash, PositionNum, Turnover, BuyTradingRecord, SellTradingRecord = self._matchMarketOrder(Cash, PositionNum, np.array(iid), OrderNum, BuyPrice, BuyVolLimit, BuyFee, SellPrice, SellVolLimit, SellFee)
        PositionNum[np.abs(PositionNum) < _QS_MinPositionNum] = 0
        if abs(Cash) < _QS_MinCash: Cash = 0
        BuyTradingRecord, SellTradingRecord = pd.DataFrame(BuyTradingRecord), pd.DataFrame(SellTradingRecord)
        SellTradingRecord["trading_dt"] = BuyTradingRecord["trading_dt"] = idt[-1]
        f.UserData["TradingRecord"] = pd.concat([f.UserData.get("TradingRecord", pd.DataFrame()), SellTradingRecord, BuyTradingRecord])
        Rslt = np.array([np.full_like(PositionNum, Cash), PositionNum, PositionNum * LastPrice, np.full_like(PositionNum, Turnover)]).T
        return unstructured_to_structured(Rslt, dtype=np.dtype(self._QSArgs.CompoundType)).astype("O")

    def __call__(self, init_account: Factor,
        last_price: Factor, signal: Factor, target_price: Optional[Factor]=None, 
        buy_price: Optional[Factor]=None, buy_limit: Optional[Factor]=None, buy_fee: float | Factor=0, buy_amt_limit: Optional[Factor]=None,
        sell_price: Optional[Factor]=None, sell_limit: Optional[Factor]=None, sell_fee: float | Factor=0, sell_amt_limit: Optional[Factor]=None,
        factor_args:dict={}, **kwargs
    ) -> PanelOperation:
        """将算子作用在若干个因子对象上以产生简单账户因子

        Args:
            init_account: 初始账户因子, 复合因子, [("Cash", float), ("Position", float), ("Amount", float), ("Turnover", float)]
            last_price: 最新价因子
            signal: 信号因子, 默认是买卖数量
            target_price: 目标价因子
            buy_price: 买入成交价因子
            buy_limit: 禁止买入条件因子, 该因子值等于 1 的 ID 禁止买入
            buy_fee: 买入交易费率因子
            buy_amt_limit: 买入成交额限制因子, 该期买入额不能超过该因子值
            sell_price: 卖出成交价因子
            sell_limit: 禁止卖出条件因子, 该因子值等于 1 的 ID 禁止卖出
            sell_fee: 卖出交易费率因子
            sell_amt_limit: 卖出成交额限制因子, 该期卖出额不能超过该因子值
            factor_args: 创建 IC 因子时传递个它的参数集
            kwargs: 创建 IC 因子时传递给它的其他入参

        Returns:
            简单账户因子
        """
        Factors = [init_account, last_price, signal]
        if target_price is not None: Factors.append(target_price)
        if buy_price is not None: Factors.append(buy_price)
        if buy_limit is not None: Factors.append(buy_limit)
        if buy_fee is not None: Factors.append(buy_fee)
        if buy_amt_limit is not None: Factors.append(buy_amt_limit)
        if sell_price is not None: Factors.append(sell_price)
        if sell_limit is not None: Factors.append(sell_limit)
        if sell_fee is not None: Factors.append(sell_fee)
        if sell_amt_limit is not None: Factors.append(sell_amt_limit)
        ModelArgs = {
            "target_price": (target_price is not None), 
            "buy_price": (buy_price is not None), 
            "buy_limit": (buy_limit is not None),
            "buy_fee": (buy_fee is not None),
            "buy_amt_limit": (buy_amt_limit is not None),
            "sell_price": (sell_price is not None),
            "sell_limit": (sell_limit is not None),
            "sell_fee": (sell_fee is not None),
            "sell_amt_limit": (sell_amt_limit is not None)
        }
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | ModelArgs
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


class PortfolioSignal2Order(SectionOperator):
    """投资组合信号转订单算子"""

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=3, title="入参数", frozen=True)
    
    def __init__(self, args:dict = {}, config_file:Optional[str] = None, **kwargs):
        Args = {"Name": "chgPortfolioSignal2Order"} | args | {"Arity": 3, "DTMode": "单时点", "OutputMode": "全截面", "DataType": "double"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Signal, Account, LastPrice = x
        Account = Account.astype(np.dtype([("Cash", float), ("Position", float), ("Amount", float), ("Turnover", float)]))
        Cash, PositionAmount = Account["Cash"][0], Account["Amount"]
        AccountValue = abs(Cash + np.nansum(PositionAmount))
        NaMask = pd.isnull(Signal)
        if np.all(NaMask): return np.full_like(Signal, np.nan)
        return (Signal * AccountValue - PositionAmount) / LastPrice

    def __call__(self, signal: Factor, account: Factor, last_price: Factor, factor_args:dict={}, **kwargs) -> SectionOperation:
        """将算子作用在若干个因子对象上以产生简单账户因子

        Args:
            signal: 投资组合信号因子, 因子值是每个时点在每个证券上的目标投资比例
            account: 账户因子, 复合因子, [("Cash", float), ("Position", float), ("Amount", float), ("Turnover", float)]
            last_price: 最新价因子
            factor_args: 创建 IC 因子时传递个它的参数集
            kwargs: 创建 IC 因子时传递给它的其他入参

        Returns:
            订单因子
        """
        return super().__call__(signal, account, last_price, factor_args=factor_args, **kwargs)