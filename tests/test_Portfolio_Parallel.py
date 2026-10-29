# -*- coding: utf-8 -*-
"""完整截面节点的多进程日期分工、账户拥有权及失败同步回归。"""
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Factor.Factor import DataFactor, Factor, FactorContext, FactorLocalContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.FactorOperator import SectionRank
from QuantStudio.Risk.api import DataRiskTable, RollingCovarianceRiskTable
from QuantStudio.PortfolioConstructor.api import (
    MeanVarianceObjectiveNode, BudgetConstraintNode, WeightConstraintNode,
    PortfolioProblem, CVXPortfolio, PortfolioWeights, PortfolioDiagnostics,
)
from QuantStudio.BackTest.Strategy.OptimizedStrategy import (
    OptimizedStrategy, StrategyTargetWeights, StrategyRealizedWeights, StrategyAccountFactor,
)


class _FailOneProcess(Factor):
    """让某个进程在前向阶段失败，覆盖同步依赖的错误传播。"""

    def forward_compute(self, path, fwd_data, context):
        if context.PID == '1':
            raise ValueError('定向进程失败')
        return [], fwd_data

    def backward_compute(self, path, inputs, context, local_context=None):
        return pd.DataFrame(1., index=local_context.DTs, columns=local_context.IDs)


class _VersionedReturn(Factor):
    """随运行快照改变收益，检查共享缓存没有跨运行复用旧结果。"""

    def backward_compute(self, path, inputs, context, local_context=None):
        values = [.3, 0., 0.] if context.DataVersion == 'v1' else [0., .3, 0.]
        return pd.DataFrame([values] * len(local_context.DTs), index=local_context.DTs, columns=local_context.IDs)


class TestPortfolioParallel(unittest.TestCase):
    """使用真实子进程和文件缓存验证完整图，不在测试中替换求解器。"""

    def setUp(self):
        self.ids = ['A', 'B', 'C']
        self.dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(9)]
        rng = np.random.default_rng(17)
        self.prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, .02, (9, 3)), axis=0),
                                   index=self.dates, columns=self.ids)
        self.price = DataFactor(self.prices) * 1.
        self.risk = DataRiskTable(pd.DataFrame(np.diag([1., 2., 4.]), index=self.ids, columns=self.ids))

    def problem(self, risk=None, expected_return=None):
        expected_return = expected_return if expected_return is not None else SectionRank()(self.price) * .1
        objective = MeanVarianceObjectiveNode.template(args={'ExpectedReturnCoef': 1.}).bind(
            risk_table=self.risk if risk is None else risk, expected_return=expected_return).build()
        return PortfolioProblem.template().bind(objective=objective, constraints=(
            BudgetConstraintNode.template().build(), WeightConstraintNode.template().build())).build()

    def run_nodes(self, nodes, *, parallel=True, count=3, dates=None, ids=None, fail_mode='raise', requests=None):
        """启动真实引擎，事件在主进程收集，缓存目录由本次测试独占。"""
        dates, ids = self.dates if dates is None else dates, self.ids if ids is None else ids
        events = []
        pids = [str(i) for i in range(count)] if parallel else ['0']
        with TemporaryDirectory(prefix='qs-portfolio-parallel-') as directory:
            cache = FeatherFactorCache(args={'DTRuler': self.dates, 'PIDs': pids, 'CacheDir': directory, 'StartMode': 'new'})
            with cache, FactorContext(DTRuler=self.dates, SectionIDs=self.ids, PIDList=pids, DataCache=cache,
                                      EventHandler=events.append) as context:
                engine = ParallelEngine(args={'FailMode': fail_mode}) if parallel else Engine(args={'FailMode': fail_mode})
                with engine:
                    values = engine.run(nodes, context, fwd_data_list=(requests if requests is not None else
                        [FactorLocalContext(DTs=dates, IDs=ids) for _ in nodes]))
        return values, events

    def test_independent_solver_and_multiple_projections_solve_once(self):
        solver = CVXPortfolio.template().bind(problem=self.problem()).build()
        nodes = [solver, PortfolioWeights.template().bind(source=solver).build(),
                 PortfolioDiagnostics.template().bind(source=solver).build()]
        expected, _ = self.run_nodes(nodes, parallel=False)
        actual, events = self.run_nodes(nodes)
        pd.testing.assert_frame_equal(actual[0].weights, expected[0].weights)
        pd.testing.assert_frame_equal(actual[1], expected[1])
        self.assertEqual(list(actual[2]), self.dates)
        self.assertEqual(len([event for event in events if event['stage'] == 'solve' and event['instance_id'] == solver.InstanceID]), len(self.dates))
        self.assertFalse(actual[0].diagnostics[self.dates[0]].weights.flags.writeable)

    def test_empty_date_partitions_and_root_subset_requests(self):
        solver = CVXPortfolio.template().bind(problem=self.problem()).build()
        expected, _ = self.run_nodes([solver], parallel=False, dates=self.dates[-2:])
        actual, events = self.run_nodes([solver], count=4, dates=self.dates[-2:], ids=['C'])
        pd.testing.assert_frame_equal(actual[0].weights, expected[0].weights)
        self.assertEqual(len([event for event in events if event['stage'] == 'solve']), 2)

    def test_identical_solver_instances_keep_distinct_ranges_and_barriers(self):
        first = CVXPortfolio.template().bind(problem=self.problem()).build()
        second = first.new()
        self.assertEqual(first.QSID, second.QSID)
        self.assertNotEqual(first.InstanceID, second.InstanceID)
        requests = [FactorLocalContext(DTs=dates, IDs=self.ids) for dates in (self.dates[:2], self.dates[-2:])]
        expected, _ = self.run_nodes([first, second], parallel=False, requests=requests)
        actual, events = self.run_nodes([first, second], requests=requests)
        for result, baseline in zip(actual, expected):
            pd.testing.assert_frame_equal(result.weights, baseline.weights)
        self.assertEqual(len([event for event in events if event['stage'] == 'solve']), 4)

    def test_rolling_covariance_sparse_dates_and_downstream_factor(self):
        risk = RollingCovarianceRiskTable(self.price, args={'Window': 3})
        dates = [self.dates[4], self.dates[8]]
        expected, _ = self.run_nodes([risk], parallel=False, dates=dates, ids=['C', 'A'])
        actual, _ = self.run_nodes([risk], count=4, dates=dates, ids=['C', 'A'])
        np.testing.assert_allclose(actual[0].values, expected[0].values, equal_nan=True)
        self.assertEqual(actual[0].major_axis.tolist(), ['C', 'A'])
        solver = CVXPortfolio.template().bind(problem=self.problem(risk=risk, expected_return=DataFactor(0.))).build()
        weights = PortfolioWeights.template().bind(source=solver).build() * 2.
        expected, _ = self.run_nodes([weights], parallel=False, dates=self.dates[4:])
        actual, _ = self.run_nodes([weights], dates=self.dates[4:])
        pd.testing.assert_frame_equal(actual[0], expected[0])

    def test_continuous_account_has_one_owner_and_shared_projections(self):
        strategy = OptimizedStrategy.template(args={'StartDT': self.dates[0], 'InitialCash': 1000.}).bind(
            problem=self.problem(), mark_price=self.price, execution_price=self.price,
            buy_amt_limit=DataFactor(200.), sell_fee=DataFactor(.001)).build()
        nodes = [strategy, StrategyTargetWeights.template().bind(source=strategy).build(),
                 StrategyRealizedWeights.template().bind(source=strategy).build(),
                 StrategyAccountFactor.template().bind(source=strategy).build()]
        expected, _ = self.run_nodes(nodes, parallel=False, dates=self.dates[-3:])
        actual, events = self.run_nodes(nodes, dates=self.dates[-3:])
        for field in ('weights', 'realized_weights', 'account', 'orders', 'trades', 'fees'):
            pd.testing.assert_frame_equal(getattr(actual[0], field), getattr(expected[0], field))
        for index in (1, 2, 3):
            pd.testing.assert_frame_equal(actual[index], expected[index])
        self.assertFalse(actual[0].final_state.positions.flags.writeable)
        self.assertEqual(len([event for event in events if event['stage'] == 'account' and event['status'] == 'done']), len(self.dates))

    def test_failed_upstream_releases_barriers_and_preserves_independent_root(self):
        broken = SectionRank()(_FailOneProcess())
        solver = CVXPortfolio.template().bind(problem=self.problem(expected_return=broken)).build()
        with self.assertRaises(Exception):
            self.run_nodes([solver])
        actual, _ = self.run_nodes([solver, DataFactor(7.)], fail_mode='skip')
        self.assertIsNone(actual[0])
        np.testing.assert_allclose(actual[1], 7.)

    def test_repeated_runs_and_projection_first_do_not_reuse_stale_results(self):
        solver = CVXPortfolio.template().bind(problem=self.problem(expected_return=_VersionedReturn())).build()
        nodes = [PortfolioWeights.template().bind(source=solver).build(), solver]
        dates, pids, events = self.dates[-2:], ['0', '1', '2'], []
        requests = [FactorLocalContext(DTs=dates, IDs=self.ids) for _ in nodes]
        with TemporaryDirectory(prefix='qs-portfolio-rerun-') as directory:
            cache = FeatherFactorCache(args={'DTRuler': self.dates, 'PIDs': pids, 'CacheDir': directory, 'StartMode': 'new'})
            with cache, FactorContext(DTRuler=self.dates, SectionIDs=self.ids, PIDList=pids, DataCache=cache,
                                      DataVersion='v1', EventHandler=events.append) as context, ParallelEngine() as engine:
                first = engine.run(nodes, context, fwd_data_list=requests)
                first_run = context.RunID
                context.DataVersion = 'v2'
                second = engine.run(nodes, context, fwd_data_list=requests)
                self.assertNotEqual(first_run, context.RunID)
        for result in (first, second):
            pd.testing.assert_frame_equal(result[0], result[1].weights)
        self.assertGreater(first[0].iloc[0]['A'], second[0].iloc[0]['A'])
        self.assertEqual(len([event for event in events if event['stage'] == 'solve']), 2 * len(dates))

    def test_solver_failure_and_non_rebalance_dates_release_empty_partitions(self):
        base = self.problem()
        problem = base.new(deps={
            'objective': base.Dependencies.objective,
            'constraints': (BudgetConstraintNode.template(args={'UpLimit': 4., 'DownLimit': 4.}).build(),
                            WeightConstraintNode.template().build()),
        })
        solver = CVXPortfolio.template(args={'OnFailure': 'nan', 'RebalanceDTs': self.dates[:3:2]}).bind(problem=problem).build()
        actual, _ = self.run_nodes([solver], count=4, dates=self.dates[:3])
        self.assertTrue(actual[0].weights.isna().all().all())
        self.assertEqual([item.status for item in actual[0].diagnostics.values()], ['failed', 'skipped', 'failed'])
        actual, _ = self.run_nodes([solver.new(args={'OnFailure': 'raise'}), DataFactor(7.)],
                                   count=4, dates=self.dates[:3], fail_mode='skip')
        self.assertIsNone(actual[0])
        np.testing.assert_allclose(actual[1], 7.)


if __name__ == '__main__':
    unittest.main()
