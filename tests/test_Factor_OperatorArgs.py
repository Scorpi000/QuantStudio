# -*- coding: utf-8 -*-
"""算子声明式参数改造的验收与回归测试。

对应 docs/plans/factor_operator_args.md，覆盖参数自省、旧接口兼容、
元数约束、QSID、复制及序列化，以及审阅发现的计算和文档示例缺陷。
使用确定性内存数据，不连接数据库，也不执行 Notebook 的数据加载单元。

使用方法:
    在仓库根目录使用项目虚拟环境运行::

        python -m unittest discover -s tests -p "test_Factor_OperatorArgs.py" -v
"""
import contextlib
import datetime as dt
import io
import json
import math
from pathlib import Path
import unittest

import dill
import numpy as np
import pandas as pd
from pydantic import Field, ValidationError

from QuantStudio.Core import __QS_Error__
from QuantStudio.Factor.Factor import DataFactor
from QuantStudio.Factor.FactorOperation import (
    FactorOperator,
    FactorOperatorized,
    OperandSpec,
    OperatorModelArgs,
    makeFactorOperator,
)
from QuantStudio.Factor import FactorOperator as fo


class ScaleArgs(OperatorModelArgs):
    """供函数式算子验证类型、默认值和参数约束。"""

    scale: float = Field(default=2.0, gt=0, title="缩放系数", description="结果的倍数")


def scale_data(f, idt, iid, x, args):
    """用声明的模型参数缩放输入数据。"""
    return x[0] * args.scale


def sum_data(f, idt, iid, x, args):
    """对数量可变的输入求和，兼容旧的字典读法。"""
    return np.sum(x, axis=0) * args.get("scale", 1.0)


class TestOperatorArgs(unittest.TestCase):
    """声明式参数及计算行为的回归测试。"""

    def setUp(self):
        self.dts = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(6)]
        self.ids = ["A", "B", "C"]
        self.data = pd.DataFrame(
            np.arange(1.0, 19.0).reshape(6, 3), index=self.dts, columns=self.ids
        )
        self.factor = DataFactor(data=self.data)
        self.mask = DataFactor(data=pd.DataFrame(
            [[1, 0, 1]] * len(self.dts), index=self.dts, columns=self.ids
        ))

    def read(self, factor, start=0):
        """按完整时间尺读取因子，start 指定预留的回溯期数。"""
        return factor.readData(ids=self.ids, dts=self.dts[start:], dt_ruler=self.dts)

    def scale_operator(self, scale=2.0):
        """构建显式声明参数与操作数的函数式算子。"""
        args = {"DTMode": "多时点", "IDMode": "多ID", "ModelArgs": {"scale": scale}}
        return makeFactorOperator(
            scale_data, "Point", args=args, model_args=ScaleArgs,
            operand_info=(OperandSpec(Name="x", Title="输入因子"),),
        )

    def test_builtin_parameter_metadata_and_legacy_access(self):
        """内置参数保留属性和字典读法，并提供元信息。"""
        args = fo.Log(base=2).Args.ModelArgs
        self.assertIsInstance(args, fo.LogArgs)
        self.assertEqual(args.base, 2)
        self.assertEqual(args["base"], 2)
        self.assertEqual(args.meta().loc["base", "title"], "底数")
        self.assertIn("base", args.info())

    def test_builtin_operands_are_declared_only_in_argument_models(self):
        """内置操作数声明统一存放于参数字段，不再具有算子类属性。"""
        for cls in vars(fo).values():
            if not isinstance(cls, type) or not issubclass(cls, FactorOperator):
                continue
            if cls.__module__ != fo.__name__:
                continue
            with self.subTest(operator=cls.__name__):
                self.assertNotIn("OperandInfo", cls.__dict__)
                self.assertNotIn("OperandInfo", cls.__QS_ArgClass__.model_fields)
                self.assertTrue(cls.__QS_InputClass__.model_fields)
                self.assertTrue(cls.__QS_InputClass__.model_config["frozen"])

    def test_builtin_operand_override_survives_copy(self):
        """显式操作数声明覆盖默认值，复制后仍保持实例声明。"""
        specs = (OperandSpec(Name="custom", Title="自定义输入"),)
        op = fo.Log(args={"OperandInfo": specs})
        self.assertEqual(op.input_specs[0].Name, specs[0].Name)
        self.assertEqual(op.new().input_specs, op.input_specs)
        self.assertEqual(op.QSID, fo.Log().QSID)

    def test_parameter_validation(self):
        """声明的类型、范围和未知字段均应被校验。"""
        for values in ({"base": 0}, {"base": "无效"}, {"unknown": 1}):
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    fo.Log(args={"ModelArgs": values})

    def test_builtin_accepts_parameter_instance(self):
        """内置算子可接收对应的参数模型实例。"""
        op = fo.Log(args={"ModelArgs": fo.LogArgs(base=2)})
        np.testing.assert_allclose(self.read(op(self.factor)), np.log2(self.data))

    def test_parameter_assignment_is_frozen(self):
        """内置及自定义参数拒绝属性和下标赋值，保持参数与 QSID 稳定。"""
        for op, name, value in ((fo.Log(base=2), "base", 10),
                                (self.scale_operator(), "scale", 3)):
            for access in ("attribute", "item"):
                with self.subTest(operator=type(op).__name__, access=access):
                    original_id = op.QSID
                    args = op.Args.ModelArgs
                    with self.assertRaises(ValidationError):
                        if access == "attribute":
                            setattr(args, name, value)
                        else:
                            args[name] = value
                    self.assertEqual(getattr(args, name), 2)
                    self.assertEqual(op.QSID, original_id)

    def test_new_changes_parameter_without_mutating_original(self):
        """通过 new 改参产生新身份和新结果，原算子可继续复用。"""
        original = fo.Log(base=2)
        changed = original.new(args={"ModelArgs": {"base": 10}})
        self.assertNotEqual(original.QSID, changed.QSID)
        self.assertEqual(original.Args.ModelArgs.base, 2)
        np.testing.assert_allclose(self.read(changed(self.factor)), np.log10(self.data))

    def test_factory_parameters_participate_in_qsid(self):
        """函数式算子的子模型字段必须参与算子和因子的身份计算。"""
        first = self.scale_operator(scale=2)
        second = self.scale_operator(scale=5)
        self.assertNotEqual(first.QSID, second.QSID)
        self.assertNotEqual(first(self.factor).QSID, second(self.factor).QSID)
        self.assertEqual(first.QSID, self.scale_operator(scale=2).QSID)

    def test_factory_auto_arity_preserves_parameter_type_and_result(self):
        """首次调用确定元数时保留参数子类和数值。"""
        op = self.scale_operator(scale=3)
        factor = op(self.factor)
        self.assertIsInstance(factor.Operator.Args.ModelArgs, ScaleArgs)
        self.assertEqual(factor.Operator.Args.ModelArgs.scale, 3)
        np.testing.assert_allclose(self.read(factor), self.data * 3)

    def test_factory_clone_preserves_operand_declarations(self):
        """操作数声明不参与 QSID，但在复制时仍需保留供下游自省。"""
        op = self.scale_operator()
        copied = op.new()
        self.assertEqual(copied.input_specs, op.input_specs)
        self.assertEqual(op(self.factor).Operator.input_specs, op.input_specs)

    def test_factory_new_parameter_dictionary_preserves_type(self):
        """函数式算子通过字典改参仍保留原参数模型的类型和校验。"""
        op = self.scale_operator(scale=2)
        changed = op.new(args={"ModelArgs": {"scale": 3}})
        self.assertIsInstance(changed.Args.ModelArgs, ScaleArgs)
        self.assertEqual(op.Args.ModelArgs.scale, 2)
        self.assertNotEqual(changed.QSID, op.QSID)
        np.testing.assert_allclose(self.read(changed(self.factor)), self.data * 3)
        with self.assertRaises(ValidationError):
            op.new(args={"ModelArgs": {"scale": 0}})

    def test_factory_clone_parameter_models_are_independent(self):
        """复制后的模型及其可变参数不与模板共享。"""
        op = makeFactorOperator(sum_data, "Point", args={"ModelArgs": {"weights": [1, 2]}})
        copied = op.new()
        self.assertIsNot(copied.Args.ModelArgs, op.Args.ModelArgs)
        self.assertIsNot(copied.Args.ModelArgs.weights, op.Args.ModelArgs.weights)
        copied.Args.ModelArgs.weights.append(3)
        self.assertEqual(op.Args.ModelArgs.weights, [1, 2])

    def test_new_operator_with_required_constructor_arguments(self):
        """完整参数重建不要求再次提供 func 等便捷构造函数的必填参数。"""
        for op in (fo.ApplyArrayFunc(func=np.floor), fo.ChgSection(old_ids=self.ids),
                   fo.Disaggregate(aggr_ids=self.ids)):
            with self.subTest(operator=type(op).__name__):
                restored = op.new()
                self.assertEqual(type(restored), type(op))
                self.assertEqual(restored.QSID, op.QSID)
                self.assertIsNot(restored.Args, op.Args)
        copied = fo.ApplyArrayFunc(func=np.floor).new()
        np.testing.assert_allclose(self.read(copied(self.factor)), np.floor(self.data))

    def test_new_preserves_module_parameter(self):
        """复制容器时保留模块对象引用，不能对模块执行 deepcopy。"""
        op = makeFactorOperator(sum_data, "Point", args={"ModelArgs": {"module": math}})
        copied = op.new()
        self.assertIs(copied.Args.ModelArgs.module, math)

    def test_new_updates_window_and_preserves_other_model_values(self):
        """字典增量改参保留未修改字段，并更新依赖窗口的配置。"""
        original = fo.RollingApply(func=np.nanmean, window=2, min_periods=2)
        changed = original.new(args={"ModelArgs": {"window": 3}})
        self.assertIs(changed.Args.ModelArgs.func, np.nanmean)
        self.assertEqual(changed.Args.ModelArgs.min_periods, 2)
        self.assertEqual(changed.Args.LookBack, [2])
        result = self.read(changed(self.factor), start=2)
        np.testing.assert_allclose(result, self.data.rolling(3, min_periods=2).mean().iloc[2:])

    def test_public_where_operand_order(self):
        """Where 的声明与公开 f/mask/other 顺序一致，而非内部条件优先。"""
        op = fo.Where()
        self.assertEqual([spec.Name for spec in op.input_specs], ["f", "mask", "other"])
        mask = self.mask
        np.testing.assert_allclose(self.read(op(self.factor, mask, 0)),
                                   np.where(mask.readData(ids=self.ids, dts=self.dts), self.data, 0))

    def test_optional_operand_validation_precedes_flattening(self):
        """省略掩码后传入分组因子，不能被误识别为必填掩码。"""
        specs = (OperandSpec(Name="f"), OperandSpec(Name="mask"),
                 OperandSpec(Name="cat_data", Required=False))
        for cls in (fo.SectionRank, fo.Aggregate, fo.AggregatePanel):
            with self.subTest(operator=cls.__name__):
                op = cls(args={"OperandInfo": specs})
                with self.assertRaises(__QS_Error__):
                    op(self.factor, cat_data=self.factor, factor_args={})
                op(self.factor, mask=self.factor, factor_args={})

    def test_new_lag_window_uses_full_lookback(self):
        """Lag 的窗口表示回溯期数，不采用滚动算子的 window-1。"""
        original = fo.Lag(lag_period=1, window=1)
        changed = original.new(args={"ModelArgs": {"window": 3}})
        self.assertEqual(changed.Args.LookBack, [3])
        result = self.read(changed(self.factor), start=3)
        np.testing.assert_allclose(result, self.data.shift(1).iloc[3:])
        custom = fo.Lag(args={"LookBack": [4]})
        self.assertEqual(custom.new().QSID, custom.QSID)

    def test_new_point_regression_output_updates_metadata(self):
        """逐点回归改为标量输出时同步清除复合类型。"""
        original = fo.Regress()
        changed = original.new(args={"ModelArgs": {"output": "beta"}})
        self.assertEqual(changed.Args.DataType, "double")
        self.assertEqual(changed.Args.CompoundType, [])
        np.testing.assert_allclose(self.read(changed(self.factor, self.factor * 2)), self.data)

    def test_factory_keyword_only_operand_binding(self):
        """工厂算子的关键字因子在公开接口绑定后转成描述子。"""
        op = makeFactorOperator(sum_data, "Point", args={"DTMode": "多时点", "IDMode": "多ID"},
            operand_info=(OperandSpec(Name="x"), OperandSpec(Name="other", KeywordOnly=True)))
        np.testing.assert_allclose(self.read(op(self.factor, other=self.factor)), self.data * 2)
        with self.assertRaises(__QS_Error__):
            op(self.factor, self.factor)
        with self.assertRaises(__QS_Error__):
            op(self.factor)

    def test_grouped_operands_are_checked_individually(self):
        """总元数合法仍需分别满足两个公开列表参数的个数约束。"""
        op = makeFactorOperator(sum_data, "Point", args={"DTMode": "多时点", "IDMode": "多ID"},
            operand_info=(OperandSpec(Name="head"),
                OperandSpec(Name="first", Variadic=True, Grouped=True, MinNum=1, MaxNum=2),
                OperandSpec(Name="second", Variadic=True, Grouped=True, MinNum=1, MaxNum=2)))
        np.testing.assert_allclose(self.read(op(self.factor, [self.factor], second=[self.factor])), self.data * 3)
        for first, second in (([], [self.factor, self.factor]),
                              ([self.factor, self.factor], []),
                              ([self.factor], [self.factor] * 3)):
            with self.subTest(first=len(first), second=len(second)):
                with self.assertRaises(__QS_Error__):
                    op(self.factor, first, second)

    def test_component_counts_are_call_level_arguments(self):
        """两种成分聚合的分组数量在因子侧，模板身份及调用方字典保持稳定。"""
        for cls in (fo.AggregateComponent, fo.AggregateComponentPanel):
            with self.subTest(operator=cls.__name__):
                op = cls(descriptor_ids=self.ids)
                original_id = op.QSID
                factor_args = {"ModelArgs": {"note": "保留"}}
                first = op(self.factor, [self.factor], [self.factor], factor_args=factor_args)
                second = op(self.factor, [self.factor, self.factor], [], factor_args=factor_args)
                self.assertEqual(len(first.Dependencies.component_data), 1)
                self.assertEqual(len(first.Dependencies.exog), 1)
                self.assertEqual(len(second.Dependencies.component_data), 2)
                self.assertEqual(len(second.Dependencies.exog), 0)
                self.assertEqual(first.Args.ModelArgs["note"], "保留")
                self.assertEqual(factor_args, {"ModelArgs": {"note": "保留"}})
                self.assertNotIn("endog_num", op.Args.ModelArgs | {})
                self.assertNotIn("exog_num", op.Args.ModelArgs | {})
                self.assertEqual(op.QSID, original_id)
                self.assertNotEqual(first.QSID, second.QSID)

    def test_component_group_bounds_apply_to_specialized_calls(self):
        """成分聚合自定义公开接口也执行各组校验，而非只校验展开后的总数。"""
        specs = (OperandSpec(Name="component"),
                 OperandSpec(Name="component_data", Variadic=True, Grouped=True, MinNum=1, MaxNum=1),
                 OperandSpec(Name="exog", Variadic=True, Grouped=True, MinNum=1, MaxNum=1))
        for cls in (fo.AggregateComponent, fo.AggregateComponentPanel):
            with self.subTest(operator=cls.__name__):
                op = cls(args={"OperandInfo": specs})
                op(self.factor, [self.factor], [self.factor])
                with self.assertRaises(__QS_Error__):
                    op(self.factor, [], [self.factor, self.factor])

    def test_regression_clone_binds_compound_output(self):
        """重建入口在绑定元数时更新复合字段，并在改输出时更新类型。"""
        for cls in (fo.RollingRegress, fo.SectionRegress, fo.PanelRegress):
            with self.subTest(operator=cls.__name__):
                op = cls()
                self.assertEqual(op.new().QSID, op.QSID)
                bound = op(self.factor, self.factor).Operator
                fields = [name for name, dtype in bound.Args.CompoundType]
                self.assertEqual(fields.count("beta0"), 1)
                self.assertNotIn("beta1", fields)
                scalar = bound.new(args={"ModelArgs": {"output": "alpha"}})
                self.assertEqual(scalar.Args.DataType, "double")
                self.assertEqual(scalar.Args.CompoundType, [])

    def test_factory_explicit_arity_computation(self):
        """显式元数下，模型实例也必须传入真实计算过程。"""
        np.testing.assert_allclose(
            self.read(self.scale_operator(scale=3)(self.factor)), self.data * 3
        )

    def test_decorator_with_declared_parameters(self):
        """装饰器透传模型类和操作数声明，未指定元数时仍可计算。"""
        op = FactorOperatorized(
            "Point", args={"DTMode": "多时点", "IDMode": "多ID"},
            model_args=ScaleArgs, operand_info=(OperandSpec(Name="x"),),
        )(scale_data)
        np.testing.assert_allclose(self.read(op(self.factor)), self.data * 2)

    def test_legacy_factory_dictionary(self):
        """未声明模型类的旧工厂写法仍可读取任意字典参数。"""
        op = makeFactorOperator(
            sum_data, "Point", args={"DTMode": "多时点", "IDMode": "多ID",
                                     "ModelArgs": {"scale": 3}},
            operand_info=(OperandSpec(Name="x"),),
        )
        np.testing.assert_allclose(self.read(op(self.factor)), self.data * 3)

    def test_variadic_operator_is_reusable(self):
        """不同调用可确定不同元数，模板算子的元数和 QSID 保持稳定。"""
        op = makeFactorOperator(
            sum_data, "Point", args={"DTMode": "多时点", "IDMode": "多ID"},
            operand_info=(OperandSpec(Name="x", Variadic=True, MinNum=2),),
        )
        original_id = op.QSID
        for count in (2, 3):
            with self.subTest(count=count):
                factor = op(*([self.factor] * count))
                self.assertEqual(len(factor.Descriptors), count)
                np.testing.assert_allclose(self.read(factor), self.data * count)
        self.assertNotIn("Arity", type(op.Args).model_fields)
        self.assertEqual(op.QSID, original_id)
        with self.assertRaises((__QS_Error__, ValidationError)):
            op(self.factor)

    def test_variadic_max_counts_only_tail(self):
        """一个固定前缀加一至两个尾部输入应允许总元数二至三。"""
        op = makeFactorOperator(
            sum_data, "Point", args={"DTMode": "多时点", "IDMode": "多ID"},
            operand_info=(OperandSpec(Name="head"),
                          OperandSpec(Name="tail", Variadic=True, MinNum=1, MaxNum=2)),
        )
        for count in (2, 3):
            with self.subTest(count=count):
                np.testing.assert_allclose(self.read(op(*([self.factor] * count))), self.data * count)
        for count in (1, 4):
            with self.subTest(invalid_count=count):
                with self.assertRaises((__QS_Error__, ValidationError)):
                    op(*([self.factor] * count))

    def test_tolist_mask_filters_and_does_not_become_list_item(self):
        """掩码为零的位置为空，为一的位置仅包含输入数据。"""
        mask = self.mask
        result = self.read(fo.ToList()(self.factor, mask=mask))
        self.assertEqual(result.iloc[0, 0], [1.0])
        self.assertTrue(result.iloc[:, 1].isna().all())
        self.assertEqual(result.iloc[-1, 2], [18.0])

    def test_operand_declarations_replace_arity(self):
        """参数及序列化结果不保存 Arity，固定和可变输入均由声明校验。"""
        op = self.scale_operator()
        self.assertNotIn("Arity", type(op.Args).model_fields)
        self.assertNotIn("Arity", op.serialize()["__qsargs__"])
        with self.assertRaises(__QS_Error__):
            op(self.factor, self.factor)
        specs = (OperandSpec(Name="head"),
                 OperandSpec(Name="tail", Variadic=True, MinNum=2, MaxNum=3))
        op = makeFactorOperator(sum_data, "Point", operand_info=specs)
        for count in (1, 2, 5):
            with self.subTest(count=count), self.assertRaises(__QS_Error__):
                op(*([self.factor] * count))
        for count in (3, 4):
            with self.subTest(count=count):
                self.assertEqual(len(op(*([self.factor] * count)).Descriptors), count)

    def test_binding_resizes_configs_and_regression_fields(self):
        """每次调用独立生成描述子配置及回归字段，模板保持可复用。"""
        for cls in (fo.RollingRegress, fo.SectionRegress, fo.PanelRegress):
            op = cls()
            original_id = op.QSID
            for count in (2, 3, 2):
                with self.subTest(operator=cls.__name__, count=count):
                    factor = op(*([self.factor] * count))
                    args = factor.Operator.Args
                    for field in ("LookBack", "StartDT", "DescriptorSection"):
                        if hasattr(args, field):
                            self.assertEqual(len(getattr(args, field)), count)
                    beta = [name for name, _ in args.CompoundType if name.startswith("beta")]
                    self.assertEqual(beta, [f"beta{i}" for i in range(count - 1)])
            self.assertEqual(op.QSID, original_id)

    def test_removed_arity_is_rejected(self):
        """已移除的数量参数按未知字段处理，不再转换为操作数声明。"""
        with self.assertRaises(ValidationError):
            makeFactorOperator(sum_data, "Point", args={"Arity": 2})
        data = fo.Log().serialize()
        data["__qsargs__"]["Arity"] = 1
        with self.assertRaises(ValidationError):
            FactorOperator.deserialize(data)

    def test_arity_property_is_read_only_and_derived_from_operands(self):
        """只读元数由声明推导，可变、分组和可选输入返回 None。"""
        cases = ((fo.Log(), 1), (fo.Where(), 3), (fo.Sum(), None),
                 (fo.Disaggregate(aggr_ids=self.ids), None),
                 (fo.AggregateComponent(), None),
                 (makeFactorOperator(sum_data, "Point"), None))
        for op, expected in cases:
            with self.subTest(operator=type(op).__name__):
                self.assertEqual(op.Arity, expected)
                with self.assertRaises(AttributeError):
                    op.Arity = 2
                self.assertEqual(op.Arity, expected)
        self.assertIsNone(fo.Sum()(self.factor, self.factor).Operator.Arity)
        self.assertEqual(fo.Log()(self.factor).Operator.Arity, 1)

    def test_bound_config_validation_uses_actual_operand_count(self):
        """绑定后的配置长度和自身迭代索引仍按实际输入数量校验。"""
        bound = fo.RollingMean(window=2)(self.factor).Operator
        for field, values in (("LookBack", [1, 1]), ("StartDT", [None, None])):
            with self.subTest(field=field), self.assertRaises(__QS_Error__):
                bound.new(args={field: values})(self.factor)
        with self.assertRaises(__QS_Error__):
            bound.new(args={"iInitFactor": 1})(self.factor)

    def test_bound_regression_json_preserves_custom_output_fields(self):
        """已绑定回归算子经 JSON 恢复后保留显式配置的复合字段。"""
        op = fo.RollingRegress()(self.factor, self.factor).Operator
        fields = [("custom_alpha", "double"), ("custom_beta", "double"), ("custom_r2", "double")]
        op = op.new(args={"CompoundType": fields})
        restored = FactorOperator.deserialize(json.loads(json.dumps(op.serialize())))
        self.assertEqual(restored.QSID, op.QSID)
        self.assertEqual(restored(self.factor, self.factor).Operator.Args.CompoundType, fields)

    def test_disaggregate_keeps_optional_category_operand(self):
        """反聚合绑定分组因子时保留第二个描述子及其截面。"""
        values = pd.DataFrame([[10., 20.]] * len(self.dts),
                              index=self.dts, columns=["x", "y"])
        categories = pd.DataFrame([["x", "y", "x"]] * len(self.dts),
                                  index=self.dts, columns=self.ids)
        op = fo.Disaggregate(aggr_ids=["x", "y"], disaggr_ids=self.ids)
        factor = op(DataFactor(data=values), cat_data=DataFactor(data=categories), factor_args={})
        self.assertEqual(len(factor.Descriptors), 2)
        self.assertEqual(factor.Operator.Args.DescriptorSection, [["x", "y"], self.ids])
        np.testing.assert_allclose(self.read(factor), [[10., 20., 10.]] * len(self.dts))

    def test_tolist_mask_changes_factor_qsid(self):
        """同一描述子序列在掩码和普通输入两种语义下必须具有不同身份。"""
        mask = DataFactor(data=1)
        self.assertNotEqual(
            fo.ToList()(self.factor, mask=mask).QSID,
            fo.ToList()(mask, self.factor).QSID,
        )

    def test_point_regressions_accept_more_than_two_operands(self):
        """逐点回归沿三个以上输入回归，而非固定两个语义槽位。"""
        for op in (fo.Regress(output="beta"), fo.RegressChangeRate()):
            with self.subTest(operator=type(op).__name__):
                result = self.read(op(self.factor, self.factor * 2, self.factor * 3))
                expected = self.data if isinstance(op, fo.Regress) else np.full(self.data.shape, 0.5)
                np.testing.assert_allclose(result, expected)

    def test_rolling_regress_indexed_beta_output(self):
        """滚动回归保留 beta0 索引输出，并返回正确斜率。"""
        op = fo.RollingRegress(window=3, min_periods=3, output="beta0")
        result = self.read(op(self.factor * 2 + 1, self.factor), start=2)
        np.testing.assert_allclose(result, 2.0, atol=1e-12)

    def test_reduction_output_dtype(self):
        """显式 dtype 同时决定算子和衍生因子的输出类型。"""
        for cls in (fo.Sum, fo.Max, fo.Min):
            for dtype in ("double", "string", "object"):
                with self.subTest(operator=cls.__name__, dtype=dtype):
                    op = cls(dtype=dtype)
                    self.assertEqual(op.Args.DataType, dtype)
                    self.assertEqual(op(self.factor, self.factor).getMetaData(key="DataType"), dtype)

    def test_section_optional_operands_remain_call_level_arguments(self):
        """截面算子的掩码和分组仍由因子侧字典记录，不修改算子模板。"""
        mask = self.mask
        categories = DataFactor(data=pd.DataFrame([["x", "y", "x"]] * 6,
                                                  index=self.dts, columns=self.ids))
        op = fo.Aggregate()
        original_id = op.QSID
        factor = op(self.factor, mask=mask, cat_data=categories, factor_args={})
        self.assertIsInstance(factor.Args.ModelArgs, dict)
        self.assertTrue(factor.Dependencies.mask is not None)
        self.assertTrue(factor.Dependencies.cat_data is not None)
        result = self.read(factor)
        np.testing.assert_allclose(result.iloc[:, 0], self.data.iloc[:, 0] + self.data.iloc[:, 2])
        self.assertTrue(result.iloc[:, 1].isna().all())
        self.assertEqual(op.QSID, original_id)

    def test_qsid_distinguishes_inputs_and_builtin_parameters(self):
        """不同输入和不同内置参数产生不同因子身份，同构造可复现。"""
        first = fo.Log(base=2)(self.factor)
        self.assertEqual(first.QSID, fo.Log(base=2)(self.factor).QSID)
        self.assertNotEqual(first.QSID, fo.Log(base=10)(self.factor).QSID)
        self.assertNotEqual(first.QSID, fo.Log(base=2)(DataFactor(data=self.data + 1)).QSID)

    def test_json_roundtrip(self):
        """JSON 往返保留内置、函数式及函数参数算子的类型、声明和计算。"""
        cases = (
            (fo.Log(base=2), np.log2(self.data), 0),
            (self.scale_operator(scale=3), self.data * 3, 0),
            (fo.ApplyArrayFunc(func=np.floor), np.floor(self.data).iloc[2:], 2),
            (fo.RollingApply(func=np.nanmean, window=3, min_periods=3),
             self.data.rolling(3).mean().iloc[2:], 2),
        )
        for op, expected, start in cases:
            with self.subTest(operator=type(op).__name__):
                restored = FactorOperator.deserialize(json.loads(json.dumps(op.serialize())))
                self.assertIs(type(restored), type(op))
                self.assertIs(type(restored.Args.ModelArgs), type(op.Args.ModelArgs))
                self.assertEqual(restored.input_specs, op.input_specs)
                self.assertEqual(restored.QSID, op.QSID)
                np.testing.assert_allclose(self.read(restored(self.factor), start), expected)

    def test_json_roundtrip_tolist_operator_declarations(self):
        """ToList 算子恢复声明，掩码标记仍只属于因子。"""
        op = fo.ToList()(self.factor, mask=DataFactor(data=1)).Operator
        restored = FactorOperator.deserialize(json.loads(json.dumps(op.serialize())))
        self.assertNotIn("mask", restored.Args.ModelArgs | {})
        self.assertEqual(restored.input_specs, op.input_specs)

    def test_tolist_mask_is_call_level_and_template_is_reusable(self):
        """调用期标记位于因子侧，不改变模板身份或调用方的参数字典。"""
        op = fo.ToList()
        original_id = op.QSID
        factor_args = {"ModelArgs": {"note": "保留"}}
        masked = op(self.factor, mask=DataFactor(data=1), factor_args=factor_args)
        plain = op(self.factor)
        self.assertTrue(masked.Dependencies.mask is not None)
        self.assertFalse(plain.Dependencies.mask is not None)
        self.assertEqual(masked.Args.ModelArgs["note"], "保留")
        self.assertEqual(factor_args, {"ModelArgs": {"note": "保留"}})
        self.assertNotIn("mask", op.Args.ModelArgs | {})
        self.assertEqual(op.QSID, original_id)
        self.assertNotIn("Arity", type(op.Args).model_fields)
        self.assertTrue(any(spec.Name == "mask" and not spec.Required for spec in op.input_specs))

    def test_tolist_requires_data_operand_even_with_mask(self):
        """掩码不能充当必需的待合并因子。"""
        for mask in (None, DataFactor(data=1)):
            with self.subTest(mask=mask):
                with self.assertRaises(__QS_Error__):
                    fo.ToList()(mask=mask)

    def test_deserialize_plain_operator_parameters(self):
        """普通模型参数字典恢复为声明的模型类型。"""
        op = fo.Log(base=2)
        data = op.serialize()
        data["__qsargs__"]["ModelArgs"] = {"base": 2}
        restored = FactorOperator.deserialize(json.loads(json.dumps(data)))
        self.assertEqual(restored.Args.ModelArgs.base, 2)
        np.testing.assert_allclose(self.read(restored(self.factor)), np.log2(self.data))

    def test_dill_roundtrip(self):
        """多进程传输用的 dill 保留参数、身份和计算结果。"""
        for op in (fo.Log(base=2), self.scale_operator(scale=3)):
            with self.subTest(operator=type(op).__name__):
                restored = dill.loads(dill.dumps(op))
                self.assertEqual(type(restored.Args.ModelArgs), type(op.Args.ModelArgs))
                self.assertEqual(restored.QSID, op.QSID)
                np.testing.assert_allclose(self.read(restored(self.factor)), self.read(op(self.factor)))

    def quickstart_example(self, function_name, result_name):
        """仅执行真实 Notebook 中指定函数所在示例单元。

        Args:
            function_name: 要定位的示例函数名。
            result_name: 示例单元生成的因子变量名。

        Returns:
            示例因子按测试时间尺读取的结果。
        """
        path = Path(__file__).resolve().parents[1] / "docs" / "因子框架" / "QuickStart.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8-sig"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"]
                 if cell["cell_type"] == "code"
                 and f"def {function_name}(" in "".join(cell["source"])]
        self.assertEqual(len(cells), 1, "示例函数必须能唯一定位，避免文档变更后静默漏测")
        example_dts = self.dts[4:] if function_name == "MAFunc" else self.dts
        namespace = {"np": np, "pd": pd, "dt": dt, "Close": self.factor,
                     "IDs": self.ids, "DTs": example_dts}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cells[0], str(path), "exec"), namespace)
        return namespace[result_name].readData(ids=self.ids, dts=example_dts, dt_ruler=self.dts)

    def test_quickstart_moving_average_example(self):
        """真实均线示例应返回逐日结果，模式和输出形状一致。"""
        result = self.quickstart_example("MAFunc", "MA5")
        pd.testing.assert_frame_equal(result, self.data.rolling(5).mean().iloc[4:])

    def test_quickstart_zscore_example(self):
        """真实 Z-Score 示例应按各时点截面标准化。"""
        result = self.quickstart_example("calcZScore", "Close_ZScore")
        expected = self.data.sub(self.data.mean(axis=1), axis=0).div(self.data.std(axis=1, ddof=0), axis=0)
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
