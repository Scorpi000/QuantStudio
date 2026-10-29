# -*- coding: utf-8 -*-
"""使用实际成交持仓的日频连续优化账户，收盘决策、下一交易日开盘执行。"""
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Identity import ExecutionEffect
from QuantStudio.Core.Input import DependencyModel
from QuantStudio.Factor.Factor import DataFactor, Factor
from QuantStudio.PortfolioConstructor.Nodes import PortfolioExecutionNode, PortfolioProblem
from QuantStudio.PortfolioConstructor.Numerical import solve_period
from QuantStudio.PortfolioConstructor.Result import PortfolioWeights
from QuantStudio.PortfolioConstructor.Types import AccountSnapshot, SolveResult, StrategyResult
from .AccountExecution import trade_limits, match_market_orders, target_weight_orders


class OptimizedStrategy(PortfolioExecutionNode):
    """共享现金、仅做多的连续优化账户。

    外层图保持无环。目标与约束先准备外生数据，账户在每个收盘决策时
    绑定真实持仓和现金；下一交易日开盘执行目标，未成交订单当天到期。
    StartDT 必须显式指定，查询后半段仍从账户起点完整重放。
    """

    __QS_ExecutionEffect__ = ExecutionEffect.STATEFUL

    class __QS_DepClass__(DependencyModel):
        problem: PortfolioProblem = Field(title="优化问题")
        mark_price: Factor = Field(title="收盘估值价格")
        execution_price: Factor = Field(title="下一交易日开盘成交价格")
        buy_limit: Factor = Field(default_factory=lambda: DataFactor(False), title="禁止买入")
        sell_limit: Factor = Field(default_factory=lambda: DataFactor(False), title="禁止卖出")
        buy_fee: Factor = Field(default_factory=lambda: DataFactor(0.), title="买入费率")
        sell_fee: Factor = Field(default_factory=lambda: DataFactor(0.), title="卖出费率")
        buy_amt_limit: Factor = Field(default_factory=lambda: DataFactor(np.inf), title="买入金额上限")
        sell_amt_limit: Factor = Field(default_factory=lambda: DataFactor(np.inf), title="卖出金额上限")

    class __QS_ArgClass__(PortfolioExecutionNode.__QS_ArgClass__):
        StartDT: datetime = Field(frozen=True, title="账户起点")
        InitialCash: float = Field(default=1000000., ge=0, allow_inf_nan=False, frozen=True, title="初始现金")
        InitialPositions: dict[str, float] = Field(default_factory=dict, frozen=True, title="初始持仓数量")
        OnFailure: Literal["raise", "hold"] = Field(default="raise", title="求解失败处理")

    def section_ids(self, data, context):
        ids = super().section_ids(data, context)
        return tuple(dict.fromkeys((*ids, *self.Args.InitialPositions)))

    def init_compute(self, path, init_data, context):
        if self.Args.StartDT not in context.DTRuler or self.Args.StartDT > init_data.DTRange[1]:
            raise ValueError("账户起点必须在运行标尺内，且不晚于查询结束日")
        if any(not np.isfinite(value) or value < 0 for value in self.Args.InitialPositions.values()):
            raise ValueError("首版账户初始持仓数量必须有限且非负")
        request = init_data.model_copy(update={"DTRange": (self.Args.StartDT, init_data.DTRange[1])})
        return super().init_compute(path, request, context)

    def backward_compute(self, path, inputs, context, local_context=None):
        cached = self.cached_result(context, local_context)
        if cached is not None:
            return cached
        if len(context.PIDList) > 1 and context.PID != context.PIDList[0]:
            return self.save_result(None, context, local_context)
        dates, ids = local_context.DTs, tuple(local_context.IDs)
        rows = len(dates)
        weights = pd.DataFrame(np.nan, index=dates, columns=ids)
        realized = weights.copy()
        orders = pd.DataFrame(0., index=dates, columns=ids)
        trades, fees = orders.copy(), orders.copy()
        account_rows, diagnostics, snapshots, execution_diagnostics = [], {}, {}, {}
        cash = self.Args.InitialCash
        positions = np.array([self.Args.InitialPositions.get(id_, 0.) for id_ in ids])
        prices = np.full(len(ids), np.nan)
        pending = None
        snapshot = None

        def vector(name, date):
            return inputs[name].loc[date].reindex(ids).to_numpy(dtype=float)

        for index, date in enumerate(dates):
            context.emit(self, "account", "running", datetime=date.isoformat(), progress=index / rows)
            trade_numbers = np.zeros(len(ids))
            trade_prices = np.full(len(ids), np.nan)
            actual_fees = np.zeros(len(ids))
            executed_signal = np.full(len(ids), np.nan)
            if pending is not None:
                quote = vector("execution_price", date)
                valid_quote = np.isfinite(quote) & (quote > 0)
                open_marks = np.where(valid_quote, quote, prices)
                opening = AccountSnapshot(date, ids, cash, positions.copy(), open_marks,
                                          positions.copy(), version=index)
                opening.wealth
                buy_fee, sell_fee = vector("buy_fee", date), vector("sell_fee", date)
                buy_amount, sell_amount = vector("buy_amt_limit", date), vector("sell_amt_limit", date)
                buy_limit, sell_limit = vector("buy_limit", date), vector("sell_limit", date)
                if (not np.isfinite(buy_fee).all() or not np.isfinite(sell_fee).all()
                        or np.any(buy_fee < 0) or np.any(sell_fee < 0) or np.any(sell_fee >= 1)):
                    raise ValueError("交易费率必须有限、非负，卖出费率必须小于 1")
                if (np.isnan(buy_amount).any() or np.isnan(sell_amount).any()
                        or np.any(buy_amount < 0) or np.any(sell_amount < 0)):
                    raise ValueError("成交金额上限必须非负，且不能缺失")
                if not np.isfinite(buy_limit).all() or not np.isfinite(sell_limit).all():
                    raise ValueError("买卖限制不能缺失")
                safe_quote = np.where(valid_quote, quote, 1.)
                amounts = opening.weights * opening.wealth
                requested = target_weight_orders(pending, safe_quote, cash, amounts)
                requested[~valid_quote] = 0.
                buy_volume, sell_volume = trade_limits(
                    positions, safe_quote, safe_quote, buy_limit, buy_amount,
                    safe_quote, sell_limit, sell_amount)
                buy_volume[~valid_quote] = sell_volume[~valid_quote] = 0.
                cash, positions, trade_numbers, trade_prices, actual_fees = match_market_orders(
                    cash, positions, requested, safe_quote, buy_volume, buy_fee,
                    safe_quote, sell_volume, sell_fee)
                if cash < -self.Args.Tolerance or np.any(positions < -self.Args.Tolerance):
                    raise ValueError("撮合后出现负现金或负持仓")
                cash = max(0., cash)
                positions = np.maximum(positions, 0.)
                orders.loc[date], trades.loc[date], fees.loc[date] = requested, trade_numbers, actual_fees
                execution_diagnostics[date] = {
                    "unfilled": requested - trade_numbers,
                    "missing_quotes": [id_ for id_, valid in zip(ids, valid_quote) if not valid],
                }
                executed_signal = pending.copy()
                pending = None
            close = vector("mark_price", date)
            valid_close = np.isfinite(close) & (close > 0)
            prices = np.where(valid_close, close, prices)
            snapshot = AccountSnapshot(date, ids, cash, positions.copy(), prices.copy(), positions.copy(), index)
            actual_weights = snapshot.weights
            realized.loc[date] = actual_weights
            snapshots[date] = snapshot
            decision_due = self.Args.RebalanceDTs is None or date in self.Args.RebalanceDTs
            if decision_due:
                buy_limit, sell_limit = vector("buy_limit", date), vector("sell_limit", date)
                if not np.isfinite(buy_limit).all() or not np.isfinite(sell_limit).all():
                    raise ValueError("决策时买卖限制不能缺失")
                result = solve_period(
                    inputs.problem[date], snapshot=snapshot, optim_option=self.Args.OptimOption,
                    tolerance=self.Args.Tolerance, accept_inaccurate=self.Args.AcceptInaccurate,
                    logger=self.Logger, buy_allowed=(buy_limit != 1) & valid_close,
                    sell_allowed=sell_limit != 1)
                diagnostics[date] = result
                if result.weights is None:
                    if self.Args.OnFailure == "raise":
                        raise __QS_Error__(f"账户在 {date} 求解失败: {result.info}")
                    self.Logger.warning(f"账户在 {date} 未产生新计划: {result.info}")
                else:
                    pending = result.weights.copy()
                    weights.loc[date] = pending
                context.emit(self, "decision", "done", datetime=date.isoformat(), solve_status=result.status)
            else:
                diagnostics[date] = SolveResult("skipped", None)
            amounts = actual_weights * snapshot.wealth
            account_rows.append([tuple(values) for values in zip(
                np.full(len(ids), cash), positions, amounts, executed_signal,
                trade_numbers, trade_prices, actual_fees)])
            context.emit(self, "account", "done", datetime=date.isoformat(), progress=(index + 1) / rows)
        account = pd.DataFrame(account_rows, index=dates, columns=ids, dtype=object)
        result = StrategyResult(weights, realized, account, diagnostics, snapshots, orders, trades, fees,
                                execution_diagnostics, snapshot, pending)
        return self.save_result(result, context, local_context)

    def merge_period_results(self, parts):
        """仅账户拥有进程发布连续状态，其余进程只提供上游计算结果。"""
        results = [part for part in parts if part is not None]
        if len(results) != 1:
            raise ValueError("连续账户必须且只能由一个进程推进")
        return results[0]


class StrategyAccountFactor(PortfolioWeights):
    """现有 AccountStats 可读取的七字段复合账户因子。"""

    ResultField = "account"

    def getMetaData(self, key=None):
        data = {"DataType": "object", "CompoundType": [(name, "double") for name in (
            "Cash", "Position", "Amount", "Signal", "TradeNum", "TradePrice", "Fee")]}
        return data.get(key) if key is not None else pd.Series(data)


class StrategyTargetWeights(PortfolioWeights):
    """决策时生成的目标权重，包含尚未执行的最后计划。"""


class StrategyRealizedWeights(PortfolioWeights):
    """每天收盘按实际数量和现金估值的证券权重。"""

    ResultField = "realized_weights"
