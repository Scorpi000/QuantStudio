# -*- coding: utf-8 -*-
"""连续优化使用真实成交、现金和价格漂移，且重复投影不重复交易。"""
import datetime as dt
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Node import DTLocalContext
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Risk.api import DataRiskTable
from QuantStudio.PortfolioConstructor.api import (
    MeanVarianceObjectiveNode, BudgetConstraintNode, WeightConstraintNode,
    TurnoverConstraintNode, PortfolioProblem, PortfolioDiagnostics,
)
from QuantStudio.PortfolioConstructor.Numerical import solve_period
from QuantStudio.BackTest.Strategy.api import (
    OptimizedStrategy, StrategyAccountFactor, StrategyTargetWeights, StrategyRealizedWeights, AccountStats,
)


class TestOptimizedStrategy(unittest.TestCase):
    def setUp(self):
        self.ids = ["A", "B"]
        self.dates = [dt.datetime(2026, 1, day) for day in (1, 2, 3, 4)]
        self.risk = DataRiskTable(pd.DataFrame(np.eye(2), index=self.ids, columns=self.ids))
        self.mu = self.factor([[.6, .2], [-1., 2.], [-1., 2.], [-1., 2.]])
        self.close = self.factor([[10., 10.], [12., 10.], [12., 10.], [12., 10.]])

    def factor(self, values):
        return DataFactor(pd.DataFrame(values, index=self.dates, columns=self.ids))

    def problem(self, constraints=None, mask=None):
        objective = MeanVarianceObjectiveNode.template(args={"ExpectedReturnCoef": 1.}).bind(
            expected_return=self.mu, risk_table=self.risk).build()
        if constraints is None:
            constraints = (BudgetConstraintNode.template(args={"UpLimit": .8, "DownLimit": .8}).build(),
                           WeightConstraintNode.template().build())
        values = {"objective": objective, "constraints": tuple(constraints)}
        if mask is not None:
            values["mask"] = mask
        return PortfolioProblem.template().bind(**values).build()

    def strategy(self, problem=None, args=None, **inputs):
        values = {"problem": self.problem() if problem is None else problem,
                  "mark_price": self.close, "execution_price": DataFactor(10.), **inputs}
        return OptimizedStrategy.template(args={"StartDT": self.dates[0], "InitialCash": 1000.,
                                               "SectionIDs": self.ids, **(args or {})}).bind(**values).build()

    def run_strategy(self, strategy, dates=None, ids=None, event_handler=None):
        context = FactorContext(DTRuler=self.dates, SectionIDs=self.ids, EventHandler=event_handler)
        local = FactorLocalContext(DTs=dates or self.dates, IDs=ids or self.ids)
        result, = Engine().run([strategy], context, fwd_data_list=[local])
        return result

    def test_partial_fills_and_price_drift_bind_actual_holdings(self):
        turnover = TurnoverConstraintNode.template(args={
            "HoldingSource": "账户状态", "AmtMultiple": 0., "UpLimit": 1.,
        }).build()
        problem = self.problem(constraints=(
            BudgetConstraintNode.template(args={"UpLimit": .8, "DownLimit": .8}).build(),
            WeightConstraintNode.template().build(), turnover))
        result = self.run_strategy(self.strategy(problem, buy_amt_limit=DataFactor(200.)))
        np.testing.assert_allclose(result.weights.iloc[0], [.6, .2], atol=1e-6)
        np.testing.assert_allclose(result.trades.iloc[1], [20., 20.], atol=1e-6)
        np.testing.assert_allclose(result.snapshots[self.dates[1]].weights, [240/1040, 200/1040], atol=1e-6)
        self.assertAlmostEqual(result.snapshots[self.dates[1]].cash, 600., places=5)
        # 如果错误地使用上期目标 60%/20%，总换手 1 会阻止本期直接达到 0%/80%。
        np.testing.assert_allclose(result.weights.iloc[1], [0., .8], atol=1e-5)
        np.testing.assert_allclose(result.trades.iloc[0], [0., 0.])

    def test_multiple_outputs_do_not_repeat_decisions(self):
        strategy = self.strategy()
        nodes = [StrategyAccountFactor.template().bind(source=strategy).build(),
                 StrategyRealizedWeights.template().bind(source=strategy).build(),
                 PortfolioDiagnostics.template().bind(source=strategy).build()]
        with patch("QuantStudio.BackTest.Strategy.OptimizedStrategy.solve_period", wraps=solve_period) as solve:
            values = Engine().run(nodes, FactorContext(DTRuler=self.dates, SectionIDs=self.ids),
                                  fwd_data_list=[FactorLocalContext(DTs=self.dates, IDs=self.ids)] * len(nodes))
        self.assertEqual(solve.call_count, len(self.dates))
        self.assertEqual(len(values[0].iat[1, 0]), 7)

    def test_suffix_query_replays_from_start(self):
        strategy = self.strategy()
        full = self.run_strategy(strategy)
        later = self.run_strategy(strategy, dates=self.dates[2:], ids=["B"])
        pd.testing.assert_frame_equal(full.realized_weights, later.realized_weights)
        factor = StrategyRealizedWeights.template().bind(source=strategy).build()
        subset = factor.readData(ids=["B"], section_ids=self.ids, dts=self.dates[2:], dt_ruler=self.dates)
        pd.testing.assert_frame_equal(subset, full.realized_weights.loc[self.dates[2:], ["B"]])

    def test_future_quotes_and_returns_do_not_change_previous_decisions(self):
        before = self.run_strategy(self.strategy())
        future = self.close.Args.Data.copy()
        future.iloc[-1] = [100., 100.]
        self.close = DataFactor(future)
        mu = self.mu.Args.Data.copy()
        mu.iloc[-1] = [100., -100.]
        self.mu = DataFactor(mu)
        after = self.run_strategy(self.strategy())
        pd.testing.assert_frame_equal(before.weights.iloc[:-1], after.weights.iloc[:-1])
        pd.testing.assert_frame_equal(before.trades, after.trades)

    def test_fees_are_deducted_once(self):
        self.mu = DataFactor(0.)
        budget = BudgetConstraintNode.template().build()
        result = self.run_strategy(self.strategy(self.problem((budget, WeightConstraintNode.template().build())),
                                                 args={"RebalanceDTs": [self.dates[0]]}, buy_fee=DataFactor(.01)))
        state = result.snapshots[self.dates[1]]
        fee = result.fees.iloc[1].sum()
        opening_cost = (result.trades.iloc[1] * 10).sum()
        self.assertAlmostEqual(state.cash + opening_cost + fee, 1000., places=6)
        self.assertAlmostEqual(fee, opening_cost * .01, places=6)
        self.assertTrue(result.trades.iloc[2:].eq(0.).all().all())

    def test_out_of_pool_frozen_holding_is_not_erased(self):
        problem = self.problem((BudgetConstraintNode.template(args={"DownLimit": 0., "UpLimit": 1.}).build(),),
                               mask=DataFactor(False))
        sell_limit = self.factor([[True, False], [True, False], [False, False], [False, False]])
        result = self.run_strategy(self.strategy(problem, args={"InitialCash": 0., "InitialPositions": {"A": 10.}},
                                                 sell_limit=sell_limit))
        self.assertAlmostEqual(result.snapshots[self.dates[1]].positions[0], 10., places=5)
        self.assertAlmostEqual(result.weights.iloc[0, 0], 1., places=5)
        self.assertAlmostEqual(result.snapshots[self.dates[-1]].positions[0], 0., places=5)

    def test_initial_budget_turnover_conflict_is_not_exempted(self):
        turnover = TurnoverConstraintNode.template(args={
            "HoldingSource": "账户状态", "AmtMultiple": 0., "UpLimit": .2,
        }).build()
        problem = self.problem((BudgetConstraintNode.template().build(), turnover))
        with self.assertRaises(__QS_Error__):
            self.run_strategy(self.strategy(problem))
        result = self.run_strategy(self.strategy(problem, args={"OnFailure": "hold"}))
        self.assertTrue(result.trades.eq(0.).all().all())
        self.assertTrue(all(item.status == "failed" for item in result.diagnostics.values()))

    def test_missing_quote_does_not_trade_and_missing_held_mark_fails(self):
        quote = self.factor([[10., 10.], [np.nan, 10.], [10., 10.], [10., 10.]])
        result = self.run_strategy(self.strategy(execution_price=quote))
        self.assertEqual(result.trades.iloc[1, 0], 0.)
        self.assertEqual(result.execution_diagnostics[self.dates[1]]["missing_quotes"], ["A"])
        self.close = self.factor([[np.nan, 10.]] * 4)
        with self.assertRaisesRegex(ValueError, "估值价格"):
            self.run_strategy(self.strategy(args={"InitialPositions": {"A": 1.}}))

    def test_cancelled_run_publishes_no_partial_result(self):
        strategy = self.strategy()
        def cancel(event):
            if event.get("stage") == "account" and event.get("progress", 0) >= .25:
                raise RuntimeError("取消测试")
        with self.assertRaisesRegex(RuntimeError, "取消测试"):
            self.run_strategy(strategy, event_handler=cancel)
        result = self.run_strategy(strategy)
        self.assertEqual(len(result.weights), len(self.dates))

    def test_account_factor_feeds_existing_account_statistics(self):
        strategy = self.strategy()
        account = StrategyAccountFactor.template().bind(source=strategy).build()
        stats = AccountStats(account, args={"AccountSection": self.ids})
        result, = Engine().run([stats], FactorContext(DTRuler=self.dates, SectionIDs=self.ids),
                               fwd_data_list=[DTLocalContext(DTs=self.dates[1:])])
        reference = self.run_strategy(strategy)
        self.assertAlmostEqual(result["初始资金"], 1000., places=5)
        self.assertAlmostEqual(result["时间序列"]["账户价值"].iloc[-1], reference.final_state.wealth, places=5)

    def test_existing_account_api_remains_available_without_optional_solver(self):
        process = subprocess.run([sys.executable, "-c", "import sys; sys.modules['cvxpy'] = None; "
                                  "from QuantStudio.BackTest.Strategy.api import MakeAccount; "
                                  "assert MakeAccount is not None"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
