# -*- coding: utf-8 -*-
"""组合优化原生节点的数值、维度、风险、图往返和结果复用回归。"""
import datetime as dt
import unittest
from unittest.mock import patch

import dill
import numpy as np
import pandas as pd

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Risk.api import DataRiskTable
from QuantStudio.PortfolioConstructor.api import (
    MeanVarianceObjectiveNode, RiskBudgetObjectiveNode,
    BudgetConstraintNode, WeightConstraintNode, FactorExposeConstraintNode,
    ExpectedReturnConstraintNode, TurnoverConstraintNode, PortfolioProblem,
    CVXPortfolio, PortfolioWeights, PortfolioDiagnostics, PORTFOLIO_COMPONENTS,
)
from QuantStudio.PortfolioConstructor.Nodes import (
    OptimizationObjectiveNode, ConstraintNode, PortfolioExecutionNode,
)
from QuantStudio.PortfolioConstructor.Numerical import solve_period
from QuantStudio.Risk.RiskTable import RiskTable, FactorRT
from QuantStudio.Core.QSObject import Panel


class _FactorRisk(FactorRT):
    """以因子模型形式提供与对角协方差等价的测试风险表。"""

    def readFactorCov(self, dts):
        return Panel({date: pd.DataFrame([[1.]], index=["F"], columns=["F"]) for date in dts}, items=dts)

    def readFactorData(self, dts, ids=None):
        return Panel({"F": pd.DataFrame([[1., 0.]] * len(dts), index=dts, columns=["A", "B"])},
                     major_axis=dts, minor_axis=ids)

    def readSpecificRisk(self, dts, ids=None):
        return pd.DataFrame([[0., 2.]] * len(dts), index=dts, columns=["A", "B"]).reindex(columns=ids)


class TestPortfolioNodes(unittest.TestCase):
    def setUp(self):
        self.ids = ["A", "B"]
        self.dates = [dt.datetime(2026, 1, day) for day in (1, 2, 3)]
        self.cov = pd.DataFrame(np.diag([1., 4.]), index=self.ids, columns=self.ids)
        self.risk = DataRiskTable(self.cov)
        self.mu = self.factor([[.1, .2], [.3, .1], [.1, .2]])

    def factor(self, values):
        return DataFactor(pd.DataFrame(values, index=self.dates, columns=self.ids))

    def problem(self, constraints=None, objective=None, mask=None):
        if objective is None:
            objective = MeanVarianceObjectiveNode.template(args={"ExpectedReturnCoef": 1.}).bind(
                expected_return=self.mu, risk_table=self.risk).build()
        if constraints is None:
            constraints = (BudgetConstraintNode.template().build(), WeightConstraintNode.template().build())
        values = {"objective": objective, "constraints": tuple(constraints)}
        if mask is not None:
            values["mask"] = mask
        return PortfolioProblem.template().bind(**values).build()

    def solver(self, problem=None, **args):
        return CVXPortfolio.template(args={"SectionIDs": self.ids, **args}).bind(
            problem=self.problem() if problem is None else problem).build()

    def read(self, source, ids=None):
        factor = PortfolioWeights.template().bind(source=source).build()
        return factor.readData(ids=ids or self.ids, section_ids=self.ids, dts=self.dates, dt_ruler=self.dates)

    def test_mean_variance_and_subset_query(self):
        solver = self.solver()
        np.testing.assert_allclose(self.read(solver), [[.78, .22], [.84, .16], [.78, .22]], atol=1e-6)
        np.testing.assert_allclose(self.read(solver, ["B"]), [[.22], [.16], [.22]], atol=1e-6)

    def test_weights_continue_as_factor(self):
        factor = PortfolioWeights.template().bind(source=self.solver()).build() * 2
        values = factor.readData(ids=self.ids, dts=self.dates, section_ids=self.ids)
        np.testing.assert_allclose(values.iloc[0], [1.56, .44], atol=1e-6)

    def test_rebalance_dates_and_failure_are_distinct(self):
        solver = self.solver(RebalanceDTs=[self.dates[1]])
        values = self.read(solver)
        self.assertTrue(values.iloc[[0, 2]].isna().all().all())
        np.testing.assert_allclose(values.iloc[1], [.84, .16], atol=1e-6)
        impossible = self.problem(constraints=(BudgetConstraintNode.template(args={"UpLimit": 3., "DownLimit": 3.}).build(),
                                               WeightConstraintNode.template().build()))
        self.assertTrue(self.read(self.solver(impossible, OnFailure="nan")).isna().all().all())
        with self.assertRaises(__QS_Error__):
            self.read(self.solver(impossible))

    def test_multiple_outputs_solve_once_per_date(self):
        solver = self.solver()
        weights = PortfolioWeights.template().bind(source=solver).build()
        diagnostic = PortfolioDiagnostics.template().bind(source=solver).build()
        local = FactorLocalContext(DTs=self.dates, IDs=self.ids)
        with patch("QuantStudio.PortfolioConstructor.Nodes.solve_period", wraps=solve_period) as solve:
            values = Engine().run([weights, diagnostic, weights * 2],
                                  FactorContext(DTRuler=self.dates, SectionIDs=self.ids),
                                  fwd_data_list=[local] * 3)
        self.assertEqual(solve.call_count, len(self.dates))
        self.assertTrue(all(result.status == "success" for result in values[1].values()))

    def test_risk_labels_and_node_serialization(self):
        self.risk = DataRiskTable(self.cov.iloc[::-1, ::-1])
        solver = dill.loads(dill.dumps(self.solver()))
        np.testing.assert_allclose(self.read(solver).iloc[0], [.78, .22], atol=1e-6)
        restored = DataRiskTable.template(args=self.risk.Args.model_dump(mode='json')).build()
        np.testing.assert_allclose(restored.readCov([self.dates[0]], self.ids).iloc[0], self.cov)

    def test_mask_limits_and_exposure_collection(self):
        constraints = (BudgetConstraintNode.template().build(), WeightConstraintNode.template().build(),
                       FactorExposeConstraintNode.template(args={"UpLimit": .4, "DownLimit": .4}).bind(
                           factor_data=(DataFactor(pd.Series([1., 0.], index=self.ids)),)).build())
        np.testing.assert_allclose(self.read(self.solver(self.problem(constraints))).iloc[0], [.4, .6], atol=1e-6)
        mask = self.factor([[False, True]] * 3)
        np.testing.assert_allclose(self.read(self.solver(self.problem(mask=mask))), [[0., 1.]] * 3, atol=1e-6)

    def test_external_holdings_and_account_requirement(self):
        turnover = TurnoverConstraintNode.template(args={
            "HoldingSource": "外部因子", "AmtMultiple": 0., "UpLimit": 0.,
        }).bind(p0=self.factor([[.5, .5]] * 3)).build()
        problem = self.problem(constraints=(BudgetConstraintNode.template().build(), turnover))
        np.testing.assert_allclose(self.read(self.solver(problem)), [[.5, .5]] * 3, atol=1e-5)
        turnover = TurnoverConstraintNode.template(args={
            "HoldingSource": "账户状态", "AmtMultiple": 0., "UpLimit": 1.,
        }).build()
        with self.assertRaisesRegex(ValueError, "实际账户"):
            self.read(self.solver(self.problem(constraints=(turnover,))))

    def test_special_objectives_and_backend_capabilities(self):
        objective = RiskBudgetObjectiveNode.template().bind(risk_table=self.risk).build()
        np.testing.assert_allclose(self.read(self.solver(self.problem((), objective))), [[2/3, 1/3]] * 3, atol=2e-4)
        with self.assertRaisesRegex(__QS_Error__, "外部约束"):
            self.read(self.solver(self.problem(objective=objective)))

    def test_factor_risk_fields_are_aligned_and_used(self):
        self.risk = _FactorRisk(rdb=None)
        np.testing.assert_allclose(self.read(self.solver()).iloc[0], [.78, .22], atol=1e-6)

    def test_memory_risk_source_supplies_dated_covariances(self):
        self.risk = DataRiskTable({self.dates[0]: self.cov, self.dates[1]: self.cov.iloc[::-1, ::-1],
                                  self.dates[2]: self.cov})
        # 反转存储顺序保留证券标签，因此当期数值仍应与固定风险一致。
        np.testing.assert_allclose(self.read(self.solver()), [[.78, .22], [.84, .16], [.78, .22]], atol=1e-6)

    def test_released_constraints_are_reported_and_remaining_constraints_hold(self):
        soft = WeightConstraintNode.template(args={"DropPriority": 1}).bind(up_limit=DataFactor(.2)).build()
        solver = self.solver(self.problem((BudgetConstraintNode.template().build(), soft)))
        result, = Engine().run([solver], FactorContext(DTRuler=self.dates, SectionIDs=self.ids),
                               fwd_data_list=[FactorLocalContext(DTs=self.dates, IDs=self.ids)])
        self.assertTrue(all(item.status == "relaxed" for item in result.diagnostics.values()))
        np.testing.assert_allclose(result.weights.sum(axis=1), 1., atol=1e-6)
        self.assertEqual(result.diagnostics[self.dates[0]].info["ReleasedConstraint"], ["1-WeightConstraint"])

    def test_conflicting_state_sources_and_invalid_risk_are_input_errors(self):
        turnover = TurnoverConstraintNode.template(args={"HoldingSource": "账户状态", "AmtMultiple": 0.}).bind(
            p0=DataFactor(0.)).build()
        with self.assertRaisesRegex(ValueError, "同时绑定"):
            self.read(self.solver(self.problem((turnover,))))
        self.risk = DataRiskTable(pd.DataFrame([[1., 2.], [2., 1.]], index=self.ids, columns=self.ids))
        with self.assertRaisesRegex(ValueError, "半正定"):
            self.read(self.solver(OnFailure="nan"))

    def test_missing_risk_and_wrong_sections_are_rejected(self):
        objective = MeanVarianceObjectiveNode.template().build()
        with self.assertRaises(__QS_Error__):
            self.read(self.solver(self.problem(objective=objective)))
        objective = MeanVarianceObjectiveNode.template(args={"SectionIDs": ["B"]}).bind(risk_table=self.risk).build()
        with self.assertRaisesRegex(ValueError, "截面不同"):
            self.read(self.solver(self.problem(objective=objective)))

    def test_graph_document_roundtrip(self):
        try:
            from QSExt.Graph.api import GraphDocument, compile_graph, graph_from_nodes
            from QSExt.Graph.Components import register_component
            from QSExt.Graph.Definitions import create_default_registry
        except ImportError:
            self.skipTest("图往返检查需要 QSExt")
        registry = create_default_registry()
        for cls in (OptimizationObjectiveNode, ConstraintNode, PortfolioExecutionNode, RiskTable):
            registry.types.register(f"{cls.__module__}.{cls.__qualname__}", cls)
        for cls in (DataRiskTable, *PORTFOLIO_COMPONENTS, PortfolioWeights, PortfolioDiagnostics):
            register_component(registry, cls)
        weights = PortfolioWeights.template().bind(source=self.solver()).build()
        document = graph_from_nodes({"weights": weights}, registry)
        reopened = GraphDocument.model_validate_json(document.model_dump_json())
        factor = compile_graph(reopened, registry).outputs["weights"]
        np.testing.assert_allclose(factor.readData(ids=self.ids, dts=self.dates), self.read(self.solver()), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
