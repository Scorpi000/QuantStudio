# -*- coding: utf-8 -*-
"""内存风险数据来源的标签、日期、身份、序列化和叶子节点回归。"""
import datetime as dt
import json
import subprocess
import sys
import unittest

import dill
import numpy as np
import pandas as pd

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Identity import ExecutionEffect
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import FactorContext, FactorLocalContext
from QuantStudio.Risk.api import DataRiskTable, RiskTable


class TestDataRiskTable(unittest.TestCase):
    def setUp(self):
        self.ids = ["A", "B"]
        self.dates = [dt.datetime(2026, 1, day) for day in (1, 2, 3)]
        self.cov = pd.DataFrame([[1., .2], [.2, 4.]], index=self.ids, columns=self.ids)

    def test_fixed_matrix_is_a_general_risk_data_source(self):
        risk = DataRiskTable(self.cov)
        self.assertIsInstance(risk, RiskTable)
        self.assertEqual(risk.ExecutionEffect, ExecutionEffect.PURE)
        self.assertEqual(risk.Deps, [])
        self.assertIsNone(risk.RiskDB)
        self.assertEqual(risk.getDateTime(), [])
        self.assertEqual(risk.getID(), self.ids)
        self.assertEqual(risk.getID(self.dates[-1]), self.ids)
        for frame in risk.readCov(self.dates).values:
            np.testing.assert_allclose(frame, self.cov)

    def test_label_alignment_unknown_ids_and_input_snapshot(self):
        data = self.cov.iloc[::-1, :].copy()
        risk = DataRiskTable(data)
        data.iloc[:, :] = 100.
        values = risk.readCov(self.dates[:1], ids=["B", "A", "unknown"]).iloc[0]
        np.testing.assert_allclose(values.iloc[:2, :2], self.cov.loc[["B", "A"], ["B", "A"]])
        self.assertTrue(values.loc["unknown"].isna().all())
        self.assertTrue(values["unknown"].isna().all())

    def test_dated_mapping_exact_dates_without_forward_fill(self):
        risk = DataRiskTable({self.dates[2]: self.cov * 2, self.dates[0]: self.cov})
        self.assertEqual(risk.getDateTime(), [self.dates[0], self.dates[2]])
        self.assertEqual(risk.getDateTime(start_dt=self.dates[1]), [self.dates[2]])
        self.assertEqual(risk.getDateTime(end_dt=self.dates[1]), [self.dates[0]])
        values = risk.readCov(self.dates)
        np.testing.assert_allclose(values.iloc[0], self.cov)
        self.assertTrue(values.iloc[1].isna().all().all())
        np.testing.assert_allclose(values.iloc[2], self.cov * 2)
        self.assertEqual(risk.getID(self.dates[1]), [])

    def test_panel_input_and_changing_universe_mapping(self):
        panel = Panel({date: self.cov * (index + 1) for index, date in enumerate(self.dates)}, items=self.dates)
        risk = DataRiskTable(panel)
        np.testing.assert_allclose(risk.readCov(self.dates).values, panel.values)
        smaller = self.cov.loc[["B"], ["B"]]
        varying = DataRiskTable({self.dates[0]: smaller, self.dates[1]: self.cov})
        self.assertEqual(varying.getID(), ["B", "A"])
        self.assertEqual(varying.getID(self.dates[0]), ["B"])
        self.assertTrue(np.isnan(varying.readCov(self.dates[:1], self.ids).iloc[0].loc["A", "A"]))

    def test_engine_leaf_needs_no_io_prepare_task(self):
        risk = DataRiskTable(self.cov)
        context = FactorContext(DTRuler=self.dates, SectionIDs=self.ids)
        values, = Engine().run([risk], context,
                               fwd_data_list=[FactorLocalContext(DTs=self.dates, IDs=self.ids)])
        self.assertEqual(context.PrepareNodeDict, {})
        np.testing.assert_allclose(values.iloc[0], self.cov)

    def test_identity_clone_template_and_serialization(self):
        risk = DataRiskTable({self.dates[0]: self.cov, self.dates[1]: self.cov * 2})
        payload = json.loads(json.dumps(risk.serialize()))
        restored = DataRiskTable.deserialize(payload)
        template = DataRiskTable.template(args=risk.Args.model_dump(mode="json")).build()
        for copy in (restored, template, risk.new(), dill.loads(dill.dumps(risk))):
            self.assertEqual(copy.QSID, risk.QSID)
            np.testing.assert_allclose(copy.readCov(self.dates[:2]).values, risk.readCov(self.dates[:2]).values)
        changed = risk.new(data=self.cov * 3)
        self.assertNotEqual(changed.QSID, risk.QSID)
        np.testing.assert_allclose(changed.readCov(self.dates[:1]).iloc[0], self.cov * 3)

    def test_invalid_data_and_database_inputs_are_rejected(self):
        bad_data = [np.eye(2), {}, self.cov.rename(columns={"A": "C"}),
                    self.cov.set_axis(["A", "A"]), self.cov.assign(A=[np.nan, .2]),
                    pd.DataFrame([[1., 2.], [0., 1.]], index=self.ids, columns=self.ids)]
        for data in bad_data:
            with self.subTest(data=type(data).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    DataRiskTable(data)
        with self.assertRaisesRegex(TypeError, "数据库"):
            DataRiskTable(self.cov, rdb=None)
        with self.assertRaisesRegex(TypeError, "数据库"):
            DataRiskTable(self.cov, args={"SourceRDB": object()})
        entry = {"datetime": self.dates[0], "ids": self.ids, "values": self.cov.values.tolist()}
        with self.assertRaisesRegex(ValueError, "日期必须唯一"):
            DataRiskTable(args={"Data": [entry, entry]})

    def test_risk_api_does_not_require_portfolio_solver(self):
        code = "import sys; sys.modules['cvxpy'] = None; from QuantStudio.Risk.api import DataRiskTable"
        process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
