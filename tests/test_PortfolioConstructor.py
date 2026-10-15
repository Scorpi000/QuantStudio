# -*- coding: utf-8 -*-
"""组合求解松弛回归及均值方差权重因子的计算图验收。"""
import datetime as dt
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import dill
import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.QSObject import Panel
from QuantStudio.BackTest.Strategy.AllocationStrategy import CalcPortfolioNV
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.FactorOperation import FactorOperator
from QuantStudio.PortfolioConstructor.BasePC import (
    MeanVarianceObjective, BudgetConstraint, WeightConstraint,
)
from QuantStudio.PortfolioConstructor.CVXPC import CVXPC
from QuantStudio.PortfolioConstructor.api import MakeMeanVariancePortfolio, MeanVariancePortfolio
from QuantStudio.Risk.RiskTable import RiskTable


class CountingPC(CVXPC):
    """记录真实求解次数，验证求解成功后不会继续丢弃约束。"""

    def _solve(self, *args):
        self.calls = getattr(self, "calls", 0) + 1
        return super()._solve(*args)


class MemoryRiskTable(RiskTable):
    """以带标签的内存协方差模拟图中风险表，不连接数据库。"""

    class __QS_ArgClass__(RiskTable.__QS_ArgClass__):
        Covariances: dict = Field(title="测试协方差")

    def readCov(self, dts, ids=None):
        return Panel({date: self.Args.Covariances[date] for date in dts},
                     items=dts, major_axis=ids, minor_axis=ids)


class TestConstraintRelaxation(unittest.TestCase):
    def make_pc(self, *constraints):
        mask = np.ones(2, dtype=bool)
        return CountingPC(mask, MeanVarianceObjective(mask, cov=np.eye(2)),
                          [BudgetConstraint(mask), *constraints])

    def weight_constraint(self, up_limit, priority=-1):
        return WeightConstraint(np.ones(2, dtype=bool), up_limit=up_limit,
                                down_limit=0., args={"DropPriority": priority})

    def test_initial_success_preserves_droppable_constraints(self):
        pc = self.make_pc(self.weight_constraint(.8, 0))
        weights, info = pc.solve()
        np.testing.assert_allclose(weights, [.5, .5], atol=1e-6)
        self.assertEqual(pc.calls, 1)
        self.assertEqual(info["status"], 1)
        self.assertEqual(info["ReleasedConstraint"], [])

    def test_release_until_success_preserves_remaining_constraints(self):
        pc = self.make_pc(self.weight_constraint(.4, 0), self.weight_constraint(.8, 1))
        weights, info = pc.solve()
        np.testing.assert_allclose(weights, [.5, .5], atol=1e-6)
        self.assertEqual(pc.calls, 2)
        self.assertEqual(info["ReleasedConstraint"], ["1-WeightConstraint"])
        self.assertEqual(info["status"], 1)

    def test_same_priority_releases_all_constraints_in_group(self):
        pc = self.make_pc(self.weight_constraint(.4, 0), self.weight_constraint(.3, 0))
        weights, info = pc.solve()
        np.testing.assert_allclose(weights, [.5, .5], atol=1e-6)
        self.assertEqual(pc.calls, 2)
        self.assertEqual(info["ReleasedConstraint"], ["1-WeightConstraint", "2-WeightConstraint"])

    def test_mandatory_constraints_are_never_released(self):
        pc = self.make_pc(self.weight_constraint(.4), self.weight_constraint(.8, 0))
        weights, info = pc.solve()
        self.assertIsNone(weights)
        self.assertEqual(pc.calls, 2)
        self.assertEqual(info["status"], 0)
        self.assertEqual(info["ReleasedConstraint"], ["2-WeightConstraint"])


class TestMeanVariancePortfolio(unittest.TestCase):
    def setUp(self):
        self.ids = ["A", "B"]
        self.dts = [dt.datetime(2026, 1, day) for day in (1, 2, 3)]
        self.cov = pd.DataFrame(np.diag([1., 4.]), index=self.ids, columns=self.ids)
        self.returns = DataFactor(pd.DataFrame([[.1, .2], [.3, .1], [.2, .4]],
                                               index=self.dts, columns=self.ids))

    def read(self, factor, ids=None):
        return factor.readData(ids=self.ids if ids is None else ids, dts=self.dts,
                               section_ids=self.ids, dt_ruler=self.dts)

    def risk_table(self):
        return MemoryRiskTable(rdb=None, args={"Covariances": {
            self.dts[0]: self.cov,
            self.dts[1]: pd.DataFrame(np.diag([4., 1.]), index=self.ids, columns=self.ids),
            self.dts[2]: self.cov.iloc[::-1, ::-1],
        }})

    def test_minimum_variance_factor_and_subset_query(self):
        factor = MakeMeanVariancePortfolio(cov=self.cov)()
        self.assertIsInstance(factor, MeanVariancePortfolio)
        np.testing.assert_allclose(self.read(factor), [[.8, .2]] * 3, atol=1e-5)
        np.testing.assert_allclose(self.read(factor, ids=["B"]), [[.2]] * 3, atol=1e-5)

    def test_factor_section_and_context_section_with_engine(self):
        factor = MakeMeanVariancePortfolio(cov=self.cov)(factor_args={"SectionIDs": self.ids})
        values, = Engine().run([factor], FactorContext(DTRuler=self.dts, SectionIDs=self.ids),
                              fwd_data_list=[FactorLocalContext(DTs=self.dts, IDs=["B"])])
        np.testing.assert_allclose(values, [[.2]] * 3, atol=1e-5)

    def test_expected_return_matches_direct_solve(self):
        factor = MakeMeanVariancePortfolio(cov=self.cov, expected_return_coef=1.)(self.returns)
        expected = []
        for mu in self.returns.readData(ids=self.ids, dts=self.dts).values:
            mask = np.ones(2, dtype=bool)
            objective = MeanVarianceObjective(mask, expected_return=mu, cov=self.cov.values,
                                              args={"ExpectedReturnCoef": 1.})
            weights, _ = CVXPC(mask, objective, [BudgetConstraint(mask),
                               WeightConstraint(mask, up_limit=1., down_limit=0.)]).solve()
            expected.append(weights)
        np.testing.assert_allclose(self.read(factor), expected, atol=1e-6)
        self.assertIs(factor.Dependencies.expected_return, self.returns)

    def test_dynamic_risk_table_uses_full_section_and_aligns_labels(self):
        risk = self.risk_table()
        factor = MakeMeanVariancePortfolio()(risk_table=risk)
        with patch.object(risk, "readCov", wraps=risk.readCov) as read_cov:
            np.testing.assert_allclose(self.read(factor, ids=["A"]), [[.8], [.2], [.8]], atol=1e-5)
            self.assertEqual(read_cov.call_args.kwargs["ids"], self.ids)
        self.assertIs(factor.Dependencies.extra_deps[0], risk)

    def test_covariance_labels_override_storage_order(self):
        factor = MakeMeanVariancePortfolio(cov=self.cov.iloc[::-1, ::-1])()
        np.testing.assert_allclose(self.read(factor), [[.8, .2]] * 3, atol=1e-5)

    def test_mask_missing_and_excluded_assets_have_zero_weight(self):
        mask = DataFactor(pd.DataFrame([[1., 0.], [np.nan, 1.], [1., 1.]],
                                      index=self.dts, columns=self.ids))
        factor = MakeMeanVariancePortfolio(cov=self.cov)(mask=mask)
        np.testing.assert_allclose(self.read(factor), [[1., 0.], [0., 1.], [.8, .2]], atol=1e-5)

    def test_factor_weight_bounds(self):
        cap = DataFactor(pd.DataFrame([[.6, 1.], [.7, 1.], [1., 1.]],
                                     index=self.dts, columns=self.ids))
        factor = MakeMeanVariancePortfolio(cov=self.cov)(up_limit=cap)
        np.testing.assert_allclose(self.read(factor), [[.6, .4], [.7, .3], [.8, .2]], atol=1e-5)

    def test_benchmark_relative_risk_and_absolute_budget(self):
        factor = MakeMeanVariancePortfolio(cov=self.cov, benchmark=True)(
            bmk=DataFactor(pd.DataFrame([[.3, .7]] * 3, index=self.dts, columns=self.ids)))
        np.testing.assert_allclose(self.read(factor), [[.3, .7]] * 3, atol=1e-5)

    def test_pure_return_objective_does_not_require_covariance(self):
        factor = MakeMeanVariancePortfolio(expected_return_coef=1., risk_aversion_coef=0.)(self.returns)
        np.testing.assert_allclose(self.read(factor), [[0., 1.], [1., 0.], [0., 1.]], atol=1e-5)

    def test_rebalance_dates_skip_other_dates(self):
        factor = MakeMeanVariancePortfolio(cov=self.cov)(factor_args={"CalcDTRuler": [self.dts[1]]})
        values = self.read(factor)
        self.assertTrue(values.iloc[[0, 2]].isna().all().all())
        np.testing.assert_allclose(values.iloc[1], [.8, .2], atol=1e-5)

    def test_weights_feed_portfolio_net_value_graph(self):
        weights = MakeMeanVariancePortfolio(cov=self.cov)()
        prices = DataFactor(pd.DataFrame([[100., 100.], [101., 102.], [102.01, 104.04]],
                                        index=self.dts, columns=self.ids))
        nv = CalcPortfolioNV(descriptor_ids=self.ids, start_dt=self.dts[1])(
            weights, price=prices, portfolio_name_list=["Portfolio"])
        values, = Engine().run([nv], FactorContext(DTRuler=self.dts, SectionIDs=self.ids),
                              fwd_data_list=[FactorLocalContext(DTs=self.dts[1:], IDs=["Portfolio"])])
        np.testing.assert_allclose(values.to_numpy(dtype=float)[:, 0], [1., 1.012], atol=1e-6)

    def test_infeasible_problem_failure_policy(self):
        with self.assertRaises(__QS_Error__):
            self.read(MakeMeanVariancePortfolio(cov=self.cov)(up_limit=.4))
        values = self.read(MakeMeanVariancePortfolio(cov=self.cov, on_failure="nan")(up_limit=.4))
        self.assertTrue(values.isna().all().all())

    def test_empty_universe_is_not_silently_treated_as_success(self):
        with self.assertRaises(__QS_Error__):
            self.read(MakeMeanVariancePortfolio(cov=self.cov)(mask=False))
        values = self.read(MakeMeanVariancePortfolio(cov=self.cov, budget=0.)(mask=False))
        np.testing.assert_array_equal(values, 0.)

    def test_missing_risk_and_conflicting_risk_sources(self):
        with self.assertRaises(__QS_Error__):
            MakeMeanVariancePortfolio()()
        with self.assertRaises(__QS_Error__):
            MakeMeanVariancePortfolio(cov=self.cov)(risk_table=self.risk_table())

    def test_invalid_covariance_is_rejected(self):
        for cov in (self.cov.loc[["A"], ["A"]], self.cov.assign(B=[1., 4.]),
                    pd.DataFrame([[1., 2.], [2., 1.]], index=self.ids, columns=self.ids)):
            with self.subTest(cov=cov), self.assertRaises(__QS_Error__):
                self.read(MakeMeanVariancePortfolio(cov=cov)())

    def test_binding_copy_identity_and_serialization(self):
        op = MakeMeanVariancePortfolio(cov=self.cov, expected_return_coef=1.)
        direct = op(self.returns)
        bound = op.bind(expected_return=self.returns)()
        self.assertEqual(direct.QSID, bound.QSID)
        self.assertNotEqual(direct.QSID, op.new(args={"ModelArgs": {"risk_aversion_coef": 2.}})(self.returns).QSID)
        self.assertNotEqual(direct.QSID, MakeMeanVariancePortfolio(cov=self.cov * 2., expected_return_coef=1.)(self.returns).QSID)
        restored_op = FactorOperator.deserialize(json.loads(json.dumps(op.serialize())))
        self.assertEqual(op.QSID, restored_op.QSID)
        for factor in (direct.new(), bound, restored_op(self.returns), dill.loads(dill.dumps(direct))):
            np.testing.assert_allclose(self.read(factor), self.read(direct), atol=1e-6)

    def test_independent_dates_can_use_thread_executor(self):
        factor = MakeMeanVariancePortfolio()(risk_table=self.risk_table())
        with ThreadPoolExecutor(max_workers=2) as executor:
            context = FactorContext(DTRuler=self.dts, SectionIDs=self.ids,
                                    TaskExecutor=executor, MaxWorkers=2)
            values, = Engine().run([factor], context,
                                  fwd_data_list=[FactorLocalContext(DTs=self.dts, IDs=self.ids)])
        np.testing.assert_allclose(values, [[.8, .2], [.2, .8], [.8, .2]], atol=1e-5)

    def test_cached_dynamic_risk_results_are_reused(self):
        factor = MakeMeanVariancePortfolio()(risk_table=self.risk_table())
        with tempfile.TemporaryDirectory() as cache_dir:
            with FeatherFactorCache(args={"CacheDir": cache_dir, "DTRuler": self.dts}) as cache:
                context = FactorContext(DTRuler=self.dts, SectionIDs=self.ids, DataCache=cache,
                                        DataVersion="portfolio-test-v1")
                request = FactorLocalContext(DTs=self.dts, IDs=self.ids)
                engine = Engine()
                first, = engine.run([factor], context, fwd_data_list=[request])
                with patch.object(CVXPC, "_solve", side_effect=AssertionError("重复求解")) as solve:
                    repeated, = engine.run([factor], context, fwd_data_list=[request])
                    solve.assert_not_called()
                pd.testing.assert_frame_equal(first, repeated)
                np.testing.assert_allclose(first, [[.8, .2], [.2, .8], [.8, .2]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
