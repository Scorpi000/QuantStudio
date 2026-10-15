# -*- coding: utf-8 -*-
"""可组合模型的数值、依赖、后端能力及序列化回归。"""
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

from QuantStudio.Core import __QS_Error__, __QS_Args__
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Node import Node
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.FactorOperation import FactorOperator
from QuantStudio.PortfolioConstructor.api import (
    OptimizationModel, ObjectiveSpec, ConstraintSpec, MakeOptimizedPortfolio,
    MeanVarianceObjective, BudgetConstraint, WeightConstraint, TurnoverConstraint,
    FactorExposeConstraint, ExpectedReturnConstraint, RiskBudgetObjective, MaxDiversificationObjective,
)
from QuantStudio.PortfolioConstructor.BasePC import OptimizationObjective, PortfolioConstructor
from QuantStudio.Risk.RiskTable import RiskTable, FactorRT


class MemoryRisk(RiskTable):
    class __QS_ArgClass__(RiskTable.__QS_ArgClass__):
        Covariances: dict = Field(title="协方差")

    def readCov(self, dts, ids=None):
        return Panel({date: self.Args.Covariances[date] for date in dts}, items=dts,
                     major_axis=ids, minor_axis=ids)


class MemoryFactorRisk(FactorRT):
    def readFactorCov(self, dts):
        return Panel({date: pd.DataFrame([[1.]], index=["F"], columns=["F"]) for date in dts}, items=dts)

    def readFactorData(self, dts, ids=None):
        return Panel({"F": pd.DataFrame([[1., 0.]] * len(dts), index=dts, columns=["A", "B"])},
                     major_axis=dts, minor_axis=ids)

    def readSpecificRisk(self, dts, ids=None):
        return pd.DataFrame([[0., 2.]] * len(dts), index=dts, columns=["A", "B"]).reindex(columns=ids)


class FixedObjective(OptimizationObjective):
    """验证用户定义的新目标及新后端，无需修改通用图算子。"""
    def __init__(self, mask, target, args=None, config_file=None, **kwargs):
        super().__init__(args=args or {}, config_file=config_file, **kwargs)
        self.weights = np.asarray(target)[mask]

    def genObjective(self):
        return {"weights": self.weights}


class FixedBackend(PortfolioConstructor):
    def _solve(self, prepared_objective, prepared_constraints, prepared_option):
        return prepared_objective["weights"], {"status": 1}


class TestOptimizationModel(unittest.TestCase):
    def setUp(self):
        self.ids = ["A", "B"]
        self.dts = [dt.datetime(2026, 1, day) for day in (1, 2)]
        self.cov = pd.DataFrame(np.diag([1., 4.]), index=self.ids, columns=self.ids)
        self.mu = DataFactor(pd.DataFrame([[.1, .2], [.3, .1]], index=self.dts, columns=self.ids))

    def constraints(self):
        return [ConstraintSpec(BudgetConstraint),
                ConstraintSpec(WeightConstraint, inputs={"up_limit": 1., "down_limit": 0.})]

    def model(self, constraints=None, args=None, **inputs):
        return OptimizationModel(ObjectiveSpec(MeanVarianceObjective, args=args,
                                               inputs={"cov": self.cov, **inputs}),
                                 self.constraints() if constraints is None else constraints)

    def read(self, factor, ids=None):
        return factor.readData(ids=ids or self.ids, section_ids=self.ids, dts=self.dts, dt_ruler=self.dts)

    def risk(self):
        return MemoryRisk(rdb=None, args={"Covariances": {
            self.dts[0]: self.cov, self.dts[1]: self.cov.iloc[::-1, ::-1] * 2.}})

    def test_composed_model_matches_convenience_example(self):
        model = self.model(args={"ExpectedReturnCoef": 1.}, expected_return=self.mu)
        factor = MakeOptimizedPortfolio(model)()
        np.testing.assert_allclose(self.read(factor), [[.78, .22], [.84, .16]], atol=1e-5)
        self.assertIs(factor.Dependencies.factors[0], self.mu)
        np.testing.assert_allclose(self.read(factor, ids=["B"]), [[.22], [.16]], atol=1e-5)

    def test_turnover_and_buy_sell_limits(self):
        p0 = DataFactor(pd.DataFrame([[0., 1.]] * 2, index=self.dts, columns=self.ids))
        for kind, limit, weight in (("总换手限制", .2, .1), ("总买入限制", .1, .1), ("总卖出限制", .1, .1)):
            with self.subTest(kind=kind):
                constraints = self.constraints() + [ConstraintSpec(TurnoverConstraint,
                    args={"AmtMultiple": 0., "UpLimit": limit, "ConstraintType": kind}, inputs={"p0": p0})]
                values = self.read(MakeOptimizedPortfolio(self.model(constraints))())
                np.testing.assert_allclose(values, [[weight, 1 - weight]] * 2, atol=1e-5)

    def test_penalties_reduce_objective_for_maximization(self):
        p0 = pd.Series([0., 1.], index=self.ids)
        for penalty in ("TurnoverPenaltyCoef", "BuyPenaltyCoef", "SellPenaltyCoef"):
            with self.subTest(penalty=penalty):
                model = self.model(args={penalty: 10.}, p0=p0)
                np.testing.assert_allclose(self.read(MakeOptimizedPortfolio(model)()), [[0., 1.]] * 2, atol=1e-5)

    def test_multiple_exposure_rows_do_not_broadcast_constraints(self):
        ids = ["A", "B", "C"]
        exposure = pd.DataFrame([[1., 0.], [0., 1.], [1., 1.]], index=ids, columns=["F1", "F2"])
        model = OptimizationModel(ObjectiveSpec(MeanVarianceObjective, inputs={"cov": np.eye(3)}),
            self.constraints() + [ConstraintSpec(FactorExposeConstraint,
                args={"UpLimit": .6, "DownLimit": .6}, inputs={"factor_data": exposure})])
        result = MakeOptimizedPortfolio(model)().readData(ids=ids, dts=self.dts)
        np.testing.assert_allclose(result, [[.4, .4, .2]] * 2, atol=1e-5)

    def test_exposure_factor_collection_and_return_constraint(self):
        exposure = DataFactor(pd.DataFrame([[1., 0.]] * 2, index=self.dts, columns=self.ids))
        constraints = self.constraints() + [ConstraintSpec(FactorExposeConstraint,
            args={"UpLimit": .5, "DownLimit": .5}, inputs={"factor_data": (exposure,)})]
        np.testing.assert_allclose(self.read(MakeOptimizedPortfolio(self.model(constraints))()), [[.5, .5]] * 2, atol=1e-5)
        constraints = self.constraints() + [ConstraintSpec(ExpectedReturnConstraint,
            args={"DownLimit": .5}, inputs={"expected_return": exposure})]
        result = self.read(MakeOptimizedPortfolio(self.model(constraints))())
        self.assertTrue((result.A >= .5 - 1e-6).all())

    def test_relaxation_is_available_in_component_parameters(self):
        constraints = self.constraints() + [ConstraintSpec(WeightConstraint,
            args={"DropPriority": 0.}, inputs={"up_limit": .4})]
        np.testing.assert_allclose(self.read(MakeOptimizedPortfolio(self.model(constraints))()), [[.8, .2]] * 2, atol=1e-5)

    def test_categorical_exposure_and_universe_mask(self):
        ids = ["A", "B", "C"]
        model = OptimizationModel(ObjectiveSpec(MeanVarianceObjective, inputs={"cov": np.eye(3)}),
            self.constraints() + [ConstraintSpec(FactorExposeConstraint,
                args={"FactorType": "类别型", "DownLimit": 0., "UpLimit": .6},
                inputs={"factor_data": pd.Series(["Tech", "Finance", "Tech"], index=ids)})])
        values = MakeOptimizedPortfolio(model)().readData(ids=ids, dts=self.dts)
        np.testing.assert_allclose(values, [[.3, .4, .3]] * 2, atol=1e-5)
        mask = DataFactor(pd.DataFrame([[1., np.nan], [0., 1.]], index=self.dts, columns=self.ids))
        np.testing.assert_allclose(self.read(MakeOptimizedPortfolio(self.model())(mask)), [[1., 0.], [0., 1.]], atol=1e-5)

    def test_model_and_input_layout_cannot_be_overridden_without_rebinding(self):
        model = self.model()
        with self.assertRaises(ValueError):
            MakeOptimizedPortfolio(model, args={"ModelArgs": {"model": model}})
        operator = MakeOptimizedPortfolio(model)
        with self.assertRaises(ValueError):
            operator.new(args={"ModelArgs": {"model": model}})
        with self.assertRaises(ValueError):
            operator(risk_tables=(self.risk(),))

    def test_dynamic_risks_and_repeated_sources_are_explicit_and_deduplicated(self):
        risk = self.risk()
        constraints = self.constraints() + [ConstraintSpec(ExpectedReturnConstraint,
            args={"DownLimit": 0.}, inputs={"expected_return": self.mu})]
        model = self.model(constraints, args={"ExpectedReturnCoef": 1.}, expected_return=self.mu, cov=risk)
        operator = MakeOptimizedPortfolio(model)
        factor = operator()
        self.assertEqual(factor.Dependencies.factors, (self.mu,))
        self.assertEqual(factor.Dependencies.extra_deps, (risk,))
        def no_nodes(value):
            self.assertNotIsInstance(value, Node)
            if isinstance(value, __QS_Args__):
                for item in value.values():
                    no_nodes(item)
            elif isinstance(value, dict):
                for item in value.values():
                    no_nodes(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    no_nodes(item)
        no_nodes(operator.Args.ModelArgs)
        np.testing.assert_allclose(self.read(factor, ids=["A"]), [[.78], [.82]], atol=1e-5)

    def test_two_risk_nodes_receive_full_section(self):
        from QuantStudio.PortfolioConstructor.BasePC import VolatilityConstraint
        risk1, risk2 = self.risk(), self.risk()
        constraints = self.constraints() + [ConstraintSpec(VolatilityConstraint,
            args={"UpLimit": 10.}, inputs={"cov": risk2})]
        factor = MakeOptimizedPortfolio(self.model(constraints, cov=risk1))()
        with patch.object(risk1, "readCov", wraps=risk1.readCov) as read1, patch.object(risk2, "readCov", wraps=risk2.readCov) as read2:
            np.testing.assert_allclose(self.read(factor, ids=["B"]), [[.2], [.2]], atol=1e-5)
            self.assertEqual(read1.call_args.kwargs["ids"], self.ids)
            self.assertEqual(read2.call_args.kwargs["ids"], self.ids)

    def test_factor_risk_fields_are_requested_together(self):
        risk = MemoryFactorRisk(rdb=None)
        model = OptimizationModel(ObjectiveSpec(MeanVarianceObjective,
            inputs={"factor_cov": risk, "factor_data": risk, "specific_risk": risk}), self.constraints())
        factor = MakeOptimizedPortfolio(model)()
        self.assertEqual(len(factor.Dependencies.extra_deps), 1)
        np.testing.assert_allclose(self.read(factor), [[.8, .2]] * 2, atol=1e-5)

    def test_special_objectives_and_unsupported_constraints(self):
        for objective in (RiskBudgetObjective, MaxDiversificationObjective):
            with self.subTest(objective=objective):
                model = OptimizationModel(ObjectiveSpec(objective, inputs={"cov": self.cov}))
                np.testing.assert_allclose(self.read(MakeOptimizedPortfolio(model)()), [[2/3, 1/3]] * 2, atol=2e-4)
                with self.assertRaises(__QS_Error__):
                    OptimizationModel(model.objective, self.constraints())

    def test_custom_objective_and_backend(self):
        with self.assertRaises(__QS_Error__):
            OptimizationModel(ObjectiveSpec(FixedObjective, inputs={"target": [.3, .7]}))
        model = OptimizationModel(ObjectiveSpec(FixedObjective, inputs={"target": [.3, .7]}), backend=FixedBackend)
        np.testing.assert_allclose(self.read(MakeOptimizedPortfolio(model)()), [[.3, .7]] * 2)

    def test_component_validation_reuses_native_parameter_model(self):
        for build in (lambda: ObjectiveSpec(MeanVarianceObjective, args={"Unknown": 1}),
                      lambda: ConstraintSpec(WeightConstraint, inputs={"unknown": 1}),
                      lambda: ConstraintSpec(TurnoverConstraint)):
            with self.assertRaises(ValueError):
                build()
        with self.assertRaises(TypeError):
            MakeOptimizedPortfolio(self.model(expected_return=Node()))

    def test_snapshot_model_can_be_used_outside_graph(self):
        model = self.model(expected_return=pd.Series([.1, .2], index=self.ids))
        weights, info = model.solve(np.ones(2, dtype=bool), self.ids)
        np.testing.assert_allclose(weights, [.8, .2], atol=1e-6)
        self.assertEqual(info["status"], 1)

    def test_bound_operator_copy_and_serialization(self):
        operator = MakeOptimizedPortfolio(self.model(args={"ExpectedReturnCoef": 1.}, expected_return=self.mu))
        factor = operator()
        bound = operator.bind(mask=DataFactor(True))
        for op in (operator.new(), bound, FactorOperator.deserialize(json.loads(json.dumps(operator.serialize())))):
            np.testing.assert_allclose(self.read(op()), self.read(factor), atol=1e-6)
        np.testing.assert_allclose(self.read(dill.loads(dill.dumps(factor))), self.read(factor), atol=1e-6)
        np.testing.assert_allclose(self.read(factor.new()), self.read(factor), atol=1e-6)
        self.assertEqual(factor.QSID, operator.new()().QSID)
        self.assertNotEqual(factor.QSID, MakeOptimizedPortfolio(self.model(expected_return=self.mu))().QSID)

    def test_risk_binding_can_be_replaced(self):
        model = self.model(cov=self.risk())
        operator = MakeOptimizedPortfolio(model)
        other = MemoryRisk(rdb=None, args={"Covariances": {date: self.cov.iloc[::-1, ::-1].rename(
            index={"A": "B", "B": "A"}, columns={"A": "B", "B": "A"}) for date in self.dts}})
        replaced = operator.bind(risk_tables=(other,))()
        np.testing.assert_allclose(self.read(replaced), [[.2, .8]] * 2, atol=1e-5)

    def test_rebalance_and_failure_policy(self):
        factor = MakeOptimizedPortfolio(self.model())(factor_args={"CalcDTRuler": [self.dts[1]]})
        values = self.read(factor)
        self.assertTrue(values.iloc[0].isna().all())
        constraints = [ConstraintSpec(BudgetConstraint), ConstraintSpec(WeightConstraint, inputs={"up_limit": .4})]
        values = self.read(MakeOptimizedPortfolio(self.model(constraints), on_failure="nan")())
        self.assertTrue(values.isna().all().all())

    def test_cache_and_thread_execution(self):
        factor = MakeOptimizedPortfolio(self.model(cov=self.risk()))()
        with tempfile.TemporaryDirectory() as cache_dir, ThreadPoolExecutor(max_workers=2) as executor:
            with FeatherFactorCache(args={"CacheDir": cache_dir, "DTRuler": self.dts}) as cache:
                context = FactorContext(DTRuler=self.dts, SectionIDs=self.ids, DataCache=cache,
                    DataVersion="model-test", TaskExecutor=executor, MaxWorkers=2)
                request = FactorLocalContext(DTs=self.dts, IDs=self.ids)
                first, = Engine().run([factor], context, fwd_data_list=[request])
                with patch.object(factor.Args.Operator.Args.ModelArgs.model.backend, "_solve", side_effect=AssertionError("重复求解")):
                    repeated, = Engine().run([factor], context, fwd_data_list=[request])
                pd.testing.assert_frame_equal(first, repeated)
                np.testing.assert_allclose(first, [[.8, .2]] * 2, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
