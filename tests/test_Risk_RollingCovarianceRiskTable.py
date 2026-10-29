# -*- coding: utf-8 -*-
"""滚动风险表的收益率、当期掩码、历史请求、图串联和边界回归。"""
import datetime as dt
import tempfile
import unittest

import dill
import numpy as np
import pandas as pd

from QuantStudio.Core.Identity import ExecutionEffect
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorInitData
from QuantStudio.Factor.HDF5DB import HDF5DB
from QuantStudio.Risk.api import RiskTable, RollingCovarianceRiskTable


class TestRollingCovarianceRiskTable(unittest.TestCase):
    """使用明确的收益率路径，验证真实引擎和独立数值结果。"""

    def setUp(self):
        self.ids = ['A', 'B', 'C']
        self.dates = [dt.datetime(2026, 1, 1) + dt.timedelta(days=day) for day in range(9)]
        self.returns = np.array([[.01, .02, -.03], [.03, -.01, .02], [-.02, .04, .01],
                                 [.04, .03, -.01], [-.01, -.02, .04], [.02, .01, -.02],
                                 [.05, -.03, .03], [-.03, .05, .02]])
        self.prices = pd.DataFrame(np.vstack([np.ones(3), np.cumprod(1. + self.returns, axis=0)]) * 100.,
                                   index=self.dates, columns=self.ids)
        self.mask = pd.DataFrame(1., index=self.dates, columns=self.ids)

    def risk(self, prices=None, mask=None, **args):
        return RollingCovarianceRiskTable(DataFactor(self.prices if prices is None else prices),
            DataFactor(self.mask if mask is None else mask), args={'Window': 3, **args})

    def test_simple_returns_covariance_uses_window_plus_one_prices(self):
        risk = self.risk()
        self.assertIsInstance(risk, RiskTable)
        self.assertEqual(risk.ExecutionEffect, ExecutionEffect.PURE)
        self.assertIsNone(risk.RiskDB)
        self.assertEqual(len(risk.Deps), 2)
        values = risk.readCov(self.dates[3:])
        for index, date in enumerate(self.dates[3:], start=3):
            expected = np.cov(self.returns[index - 3:index], rowvar=False, ddof=1)
            np.testing.assert_allclose(values.loc[date], expected, atol=1e-14)
        self.assertFalse(np.allclose(values.loc[self.dates[3]], self.prices.iloc[:4].cov()))

    def test_optional_mask_defaults_to_all_assets_for_constructor_and_template(self):
        price = DataFactor(self.prices)
        expected = self.risk().readCov(self.dates[3:]).values
        implicit = RollingCovarianceRiskTable(price, args={'Window': 3})
        nodes = (implicit, RollingCovarianceRiskTable(price, mask=None, args={'Window': 3}),
                 RollingCovarianceRiskTable.template(args={'Window': 3}).bind(price=price).build(),
                 implicit.new(), dill.loads(dill.dumps(implicit)))
        for node in nodes:
            with self.subTest(node=node.InstanceID):
                np.testing.assert_allclose(node.readCov(self.dates[3:]).values, expected, atol=1e-14)
        self.assertFalse(RollingCovarianceRiskTable.__QS_DepClass__.model_fields['mask'].is_required())

    def test_current_mask_is_exactly_one_and_leaves_nan_rows_and_columns(self):
        mask = self.mask.copy()
        mask.iloc[3] = [1, 0, np.nan]
        mask.iloc[4] = [1, 1, 0]
        mask.iloc[5] = [0, 0, 0]
        mask.iloc[6] = [2, 1, -1]
        mask.iloc[:3] = 0  # 过去是否入选不影响当期可用资产的价格历史。
        values = self.risk(mask=mask).readCov(self.dates[3:7])
        for index, active in ((3, ['A']), (4, ['A', 'B']), (5, []), (6, ['B'])):
            matrix = values.loc[self.dates[index]]
            inactive = [id_ for id_ in self.ids if id_ not in active]
            self.assertTrue(matrix.loc[inactive].isna().all().all())
            self.assertTrue(matrix[inactive].isna().all().all())
            if active:
                expected = pd.DataFrame(self.returns[index - 3:index], columns=self.ids)[active].cov()
                np.testing.assert_allclose(matrix.loc[active, active], expected, atol=1e-14)

    def test_warmup_and_minimum_common_samples(self):
        strict = self.risk().readCov(self.dates[:4])
        self.assertTrue(np.isnan(strict.values[:3]).all())
        self.assertTrue(np.isfinite(strict.values[3]).all())
        relaxed = self.risk(MinPeriods=2).readCov(self.dates[:4])
        self.assertTrue(np.isnan(relaxed.values[:2]).all())
        np.testing.assert_allclose(relaxed.values[2], np.cov(self.returns[:2], rowvar=False), atol=1e-14)

    def test_missing_invalid_prices_are_not_filled_or_bridged(self):
        prices = self.prices.copy()
        prices.iloc[3, 0] = np.nan
        prices.iloc[3, 1] = 0.
        prices.iloc[3, 2] = np.inf
        risk = self.risk(prices=prices, Window=5, MinPeriods=2)
        values = risk.readCov(self.dates[5:6]).iloc[0]
        # 第 3、4 期收益率作废；共同有效的历史只剩第 1、2、5 期。
        np.testing.assert_allclose(values, np.cov(self.returns[[0, 1, 4]], rowvar=False), atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(values).min(), -1e-14)
        strict = self.risk(prices=prices, Window=5).readCov(self.dates[5:6])
        self.assertTrue(np.isnan(strict.values).all())
        prices.iloc[3, 1] = -5.
        np.testing.assert_allclose(self.risk(prices=prices, Window=5, MinPeriods=2).readCov(self.dates[5:6]).iloc[0], values)

    def test_inactive_assets_with_missing_history_do_not_reduce_samples(self):
        prices = self.prices.copy()
        prices['C'] = np.nan
        mask = self.mask.copy()
        mask['C'] = 0
        values = self.risk(prices=prices, mask=mask).readCov(self.dates[3:4]).iloc[0]
        np.testing.assert_allclose(values.loc[['A', 'B'], ['A', 'B']], np.cov(self.returns[:3, :2], rowvar=False), atol=1e-14)
        self.assertTrue(values['C'].isna().all())

    def test_sparse_dates_preserve_full_history_and_output_order(self):
        dates = [self.dates[8], self.dates[4]]
        values = self.risk().readCov(dates, ids=['C', 'A'])
        self.assertEqual(values.items.tolist(), dates)
        self.assertEqual(values.major_axis.tolist(), ['C', 'A'])
        for date, index in zip(dates, (8, 4)):
            expected = np.cov(self.returns[index - 3:index], rowvar=False)[np.ix_([2, 0], [2, 0])]
            np.testing.assert_allclose(values.loc[date], expected, atol=1e-14)

    def test_subset_projection_keeps_full_estimation_universe(self):
        prices = self.prices.copy()
        prices.iloc[3, 2] = np.nan
        risk = self.risk(prices=prices, Window=5, MinPeriods=2)
        full = risk.readCov(self.dates[5:6]).iloc[0]
        subset = risk.readCov(self.dates[5:6], ids=['A', 'unknown']).iloc[0]
        self.assertAlmostEqual(subset.loc['A', 'A'], full.loc['A', 'A'])
        self.assertNotAlmostEqual(full.loc['A', 'A'], np.var(self.returns[:5, 0], ddof=1))
        self.assertTrue(subset['unknown'].isna().all())
        self.assertTrue(subset.loc['unknown'].isna().all())

    def test_future_prices_cannot_change_earlier_covariances(self):
        changed = self.prices.copy()
        changed.iloc[5:] *= np.array([100., 7., 10.])
        original = self.risk().readCov(self.dates[3:5])
        new = self.risk(prices=changed).readCov(self.dates[3:5])
        np.testing.assert_allclose(new.values, original.values)

    def test_named_dependencies_templates_clone_and_dill(self):
        risk = self.risk()
        template = RollingCovarianceRiskTable.template(args={'Window': 3}).bind(
            price=risk.Dependencies.price, mask=risk.Dependencies.mask)
        expected = risk.readCov(self.dates[3:]).values
        for copy in (template.build(), risk.new(), dill.loads(dill.dumps(risk))):
            self.assertEqual(copy.QSID, risk.QSID)
            np.testing.assert_allclose(copy.readCov(self.dates[3:]).values, expected)
        self.assertNotEqual(risk.new(args={'Window': 4}).QSID, risk.QSID)

    def test_engine_requests_history_from_real_factor_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = HDF5DB(args={'MainDir': directory}).connect()
            try:
                database.writeData(Panel({'price': self.prices, 'mask': self.mask}), 'sample')
                table = database.getTable('sample')
                risk = RollingCovarianceRiskTable(table.getFactor('price'), table.getFactor('mask'), args={'Window': 3})
                values = risk.readCov(self.dates[6:7], ids=self.ids)
                np.testing.assert_allclose(values.iloc[0], np.cov(self.returns[3:6], rowvar=False), atol=1e-14)
            finally:
                database.disconnect()

    def test_derived_price_factor_uses_explicit_history_ruler(self):
        from QuantStudio.Factor.FactorOperator import RollingMean
        smooth = RollingMean(window=2)(DataFactor(self.prices))
        risk = RollingCovarianceRiskTable(smooth, DataFactor(self.mask), args={'Window': 3})
        values = risk.readCov(self.dates[6:7], ids=self.ids, dt_ruler=self.dates)
        prices = self.prices.rolling(2, min_periods=1).mean()
        expected = prices.pct_change(fill_method=None).iloc[4:7].cov()
        np.testing.assert_allclose(values.iloc[0], expected, atol=1e-14)

    def test_graph_roundtrip_and_portfolio_solver_consumption(self):
        from QuantStudio.PortfolioConstructor.api import (
            MeanVarianceObjectiveNode, BudgetConstraintNode, WeightConstraintNode,
            PortfolioProblem, CVXPortfolio, PortfolioWeights,
        )
        mask = self.mask.copy()
        mask.iloc[6:, 2] = 0.
        risk = self.risk(mask=mask, Window=5)
        objective = MeanVarianceObjectiveNode.template().bind(risk_table=risk).build()
        problem = PortfolioProblem.template().bind(objective=objective, mask=risk.Dependencies.mask, constraints=(
            BudgetConstraintNode.template().build(), WeightConstraintNode.template().build())).build()
        weights = PortfolioWeights.template().bind(source=CVXPortfolio.template().bind(problem=problem).build()).build()
        dates = self.dates[6:8]
        values = weights.readData(ids=self.ids, dts=dates, dt_ruler=self.dates)
        np.testing.assert_allclose(values.sum(axis=1), 1., atol=1e-6)
        self.assertTrue((values >= -1e-6).all().all())
        np.testing.assert_allclose(values['C'], 0., atol=1e-6)
        try:
            from QSExt.Graph.api import create_default_registry, graph_from_nodes, compile_graph, GraphDocument
            from QSExt.Graph.Components import register_component
        except ImportError:
            self.skipTest('图往返验证需要 QSExt')
        registry = create_default_registry()
        register_component(registry, RollingCovarianceRiskTable)
        document = graph_from_nodes({'risk': risk}, registry)
        reopened = GraphDocument.model_validate_json(document.model_dump_json())
        restored = compile_graph(reopened, registry).outputs['risk']
        np.testing.assert_allclose(restored.readCov(dates).values, risk.readCov(dates).values, atol=1e-14)

    def test_invalid_arguments_dates_and_parallel_context(self):
        for args in ({'Window': 1}, {'MinPeriods': 1}, {'MinPeriods': 4}, {'SectionIDs': []},
                     {'SectionIDs': ['A', 'A']}, {'SourceRDB': object()}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.risk(**args)
        with self.assertRaisesRegex(TypeError, '数据库'):
            RollingCovarianceRiskTable(DataFactor(1), DataFactor(True), rdb=None)
        self.assertEqual(self.risk().readCov([]).shape, (0, 3, 3))
        self.assertEqual(self.risk().getDateTime(start_dt=self.dates[2], end_dt=self.dates[4]), self.dates[2:5])
        self.assertEqual(self.risk().getID(), self.ids)
        with self.assertRaisesRegex(ValueError, '请求时点不能重复'):
            self.risk().readCov([self.dates[5]] * 2)
        with self.assertRaisesRegex(ValueError, '标尺'):
            self.risk().readCov(self.dates[3:4], dt_ruler=self.dates[::-1])
        context = FactorContext(DTRuler=self.dates, SectionIDs=self.ids, PIDList=['0', '1'])
        with self.assertRaisesRegex(ValueError, '共享缓存'):
            self.risk().init_compute([], FactorInitData(DTRange=(self.dates[5], self.dates[6])), context)


if __name__ == '__main__':
    unittest.main()
