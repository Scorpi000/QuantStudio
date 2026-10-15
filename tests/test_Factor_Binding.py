# -*- coding: utf-8 -*-
"""算子部分绑定的数值、身份和序列化验收。"""
import datetime as dt
import unittest

import dill
import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Input import InputError
from QuantStudio.Factor.Factor import Factor, DataFactor
from QuantStudio.Factor.FactorOperation import FactorOperator, BoundFactorOperator, FactorInputModel, makeFactorOperator
from QuantStudio.Factor import FactorOperator as fo


class TestFactorBinding(unittest.TestCase):
    def setUp(self):
        self.dts = [dt.datetime(2025, 1, day) for day in range(1, 6)]
        self.ids = ["A", "B"]
        self.data = pd.DataFrame(np.arange(1., 11.).reshape(5, 2), index=self.dts, columns=self.ids)
        self.factor = DataFactor(self.data)
        self.other = DataFactor(0)
        self.mask = DataFactor(pd.DataFrame([[1, 0]] * 5, index=self.dts, columns=self.ids))

    def read(self, factor):
        return factor.readData(ids=self.ids, dts=self.dts, dt_ruler=self.dts)

    def test_bind_always_returns_operator_call_always_returns_factor(self):
        op = fo.Where()
        partial = op.bind(mask=self.mask)
        complete = partial.bind(f=self.factor, other=self.other)
        self.assertIsInstance(complete, FactorOperator)
        self.assertNotIsInstance(complete, Factor)
        self.assertIsInstance(complete(), Factor)
        self.assertFalse(op.input_info()["mask"]["bound"])
        with self.assertRaises(InputError):
            partial(f=self.factor)

    def test_direct_and_bound_application_have_same_identity(self):
        direct = fo.Where()(self.factor, self.mask, self.other)
        bound = fo.Where().bind(mask=self.mask)(self.factor, self.other)
        self.assertEqual(direct.QSID, bound.QSID)
        self.assertIs(bound.Dependencies.mask, self.mask)
        pd.testing.assert_frame_equal(self.read(direct), self.read(bound))

    def test_replace_unbind_and_duplicate_call(self):
        bound = fo.Where().bind(mask=self.mask)
        with self.assertRaises(__QS_Error__):
            bound(self.factor, self.other, mask=self.mask)
        changed = bound.bind(mask=DataFactor(1))
        np.testing.assert_allclose(self.read(changed(self.factor, self.other)), self.data)
        reopened = bound.unbind("mask")
        self.assertFalse(reopened.input_info()["mask"]["bound"])
        self.assertTrue(bound.input_info()["mask"]["bound"])

    def test_collection_binding_replaces_and_call_appends(self):
        bound = fo.Sum().bind(x=(self.factor,))
        with self.assertRaises(InputError):
            bound()
        result = bound(self.other, self.factor)
        np.testing.assert_allclose(self.read(result), self.data * 2)
        self.assertEqual(len(result.Dependencies.x), 3)
        replaced = bound.bind(x=(self.other, self.other))
        np.testing.assert_allclose(self.read(replaced()), 0)

    def test_optional_gap_retains_semantic_slot(self):
        category = DataFactor(1)
        direct = fo.SectionRank()(self.factor, cat_data=category)
        bound = fo.SectionRank().bind(cat_data=category)(self.factor)
        self.assertEqual(direct.QSID, bound.QSID)
        self.assertIsNone(bound.Dependencies.mask)
        self.assertIs(bound.Dependencies.cat_data, category)
        pd.testing.assert_frame_equal(self.read(direct), self.read(bound))

    def test_multiple_grouped_collections(self):
        component = DataFactor(pd.DataFrame([[['A'], ['B']]] * 5, index=self.dts, columns=self.ids))
        data = DataFactor(pd.DataFrame([[[1.], [2.]]] * 5, index=self.dts, columns=self.ids))
        op = fo.AggregateComponent(descriptor_ids=self.ids)
        direct = op(component, [data], [self.factor])
        bound = op.bind(component=component, component_data=(data,))(exog=(self.factor,))
        self.assertEqual(direct.QSID, bound.QSID)
        pd.testing.assert_frame_equal(self.read(direct), self.read(bound))

    def test_copy_and_serialization_preserve_bindings(self):
        bound = fo.Log(base=2).bind(x=self.factor)
        changed = bound.new(args={"ModelArgs": {"base": 10}})
        np.testing.assert_allclose(self.read(changed()), np.log10(self.data))
        for restored in (dill.loads(dill.dumps(bound)), BoundFactorOperator.deserialize(bound.serialize())):
            self.assertEqual(restored().QSID, bound().QSID)
            pd.testing.assert_frame_equal(self.read(restored()), self.read(bound()))

    def test_factory_uses_typed_input_model(self):
        class Inputs(FactorInputModel):
            left: Factor = Field(title="左输入")
            right: Factor = Field(title="右输入")

        op = makeFactorOperator(lambda f, idt, iid, x, args: x[0] + x[1], "Point",
                                input_model=Inputs, args={"DTMode": "多时点", "IDMode": "多ID"})
        factor = op.bind(left=self.factor)(right=self.other)
        self.assertIs(factor.Dependencies.left, self.factor)
        np.testing.assert_allclose(self.read(factor), self.data)

    def test_factory_preserves_multiple_collection_groups_after_binding(self):
        """函数式算子预绑定后保留各集合边界，再在计算入口展开。"""
        class Inputs(FactorInputModel):
            left: tuple[Factor, ...] = Field(min_length=2, max_length=2, title="左输入组")
            right: tuple[Factor, ...] = Field(min_length=1, max_length=1, title="右输入组")

        op = makeFactorOperator(lambda f, idt, iid, x, args: x[0] + x[1] - x[2], "Point",
                                input_model=Inputs, args={"DTMode": "多时点", "IDMode": "多ID"})
        left, right = (self.factor, self.other), (self.mask,)
        direct = op(left=left, right=right)
        partial = op.bind(left=left)
        complete = partial.bind(right=right)
        for factor in (partial(right=right), complete(), complete.new()()):
            self.assertEqual(factor.QSID, direct.QSID)
            self.assertEqual(factor.Dependencies.left, left)
            self.assertEqual(factor.Dependencies.right, right)
            pd.testing.assert_frame_equal(self.read(factor), self.data - self.read(self.mask))
        with self.assertRaises(InputError):
            partial(right=())

    def test_data_factor_template_and_empty_dependencies(self):
        template = DataFactor.template(args={"Data": self.data})
        factor = template.build()
        self.assertFalse(factor.dependency_info())
        pd.testing.assert_frame_equal(self.read(factor), self.data)
        self.assertEqual(DataFactor.template(args={"Data": "文本"}).build().Args.DataType, "string")

    def test_bind_copies_operator_parameter_containers(self):
        op = fo.RollingMean(window=2)
        bound = op.bind(x=self.factor)
        op.Args.LookBack.append(99)
        self.assertEqual(bound.Args.LookBack, [1])


if __name__ == "__main__":
    unittest.main()
