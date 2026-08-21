# -*- coding: utf-8 -*-
"""因子 SectionIDs 变体机制测试.

测试覆盖:
    - 相同 SectionIDs 不产生变体
    - 不同 SectionIDs 触发变体创建
    - 变体因子具有不同的 QSID
    - 变体因子的依赖正确初始化
    - 菱形依赖场景下的变体处理
"""

import unittest
import datetime as dt
from typing import List, Any

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Node import Node, Context
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData


class _SimpleFactor(Factor):
    """简单因子: 用于测试变体机制, 不依赖 FactorTable.

    记录 init_compute 的调用次数和接收到的 SectionIDs.
    backward_compute 返回一个简单的 DataFrame.
    """

    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Name: str = Field(default="_SimpleFactor", frozen=True, title="名称")

    def __init__(self, name="_SimpleFactor", descriptors=None, args=None, **kwargs):
        super().__init__(descriptors=descriptors or [], args=args or {"Name": name}, **kwargs)
        self.init_call_count = 0
        self.init_section_ids_received = []

    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        self.init_call_count += 1
        if init_data and init_data.SectionIDs:
            self.init_section_ids_received.append(init_data.SectionIDs)
        return super().init_compute(path=path, init_data=init_data, context=context)

    def backward_compute(self, path: List[str], bwd_data_list: List[Any],
                         context: Context, local_context: Any = None) -> Any:
        # 返回一个简单的 DataFrame
        IDs = context.NodeState[self.QSID]["section_ids"]
        DTs = context.getDateTime(context.NodeState[self.QSID]["dt_range"])
        return pd.DataFrame(np.ones((len(DTs), len(IDs))), index=DTs, columns=IDs)


class TestSectionIDVariants(unittest.TestCase):
    """SectionIDs 变体机制测试."""

    def _make_context(self, section_ids, dt_ruler):
        """创建测试用的 FactorContext."""
        return FactorContext(DTRuler=dt_ruler, SectionIDs=section_ids)

    def _make_dt_range(self, dt_ruler):
        """创建时点范围."""
        return (dt_ruler[0], dt_ruler[-1])

    def test_same_section_ids_no_variant(self):
        """相同 SectionIDs 不应产生变体."""
        dt_ruler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(10)]
        section_ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        context = self._make_context(section_ids, dt_ruler)

        factor = _SimpleFactor(name="F1")
        dt_range = self._make_dt_range(dt_ruler)

        # 两次调用使用相同的 SectionIDs
        init_data1 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids)
        init_data2 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids)

        engine = Engine()
        engine.init([factor], context, init_data_list=[init_data1])

        # 第二次调用
        factor.init_compute(path=[factor.QSID], init_data=init_data2, context=context)

        # 不应产生变体
        self.assertNotIn("_QS_FactorSectionIDVariants", context.ExtraData)
        # 因子应该被初始化了两次
        self.assertEqual(factor.init_call_count, 2)

    def test_different_section_ids_creates_variant(self):
        """不同 SectionIDs 应触发变体创建."""
        dt_ruler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(10)]
        section_ids1 = ["000001.SZ", "000002.SZ"]
        section_ids2 = ["000001.SZ", "000003.SZ"]
        context = self._make_context(section_ids1, dt_ruler)

        factor = _SimpleFactor(name="F1")
        dt_range = self._make_dt_range(dt_ruler)

        # 第一次调用使用 section_ids1
        init_data1 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids1)
        engine = Engine()
        engine.init([factor], context, init_data_list=[init_data1])

        # 第二次调用使用不同的 section_ids2
        init_data2 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids2)
        factor.init_compute(path=[factor.QSID], init_data=init_data2, context=context)

        # 应该记录了变体信息
        self.assertIn("_QS_FactorSectionIDVariants", context.ExtraData)
        variants = context.ExtraData["_QS_FactorSectionIDVariants"]
        self.assertEqual(len(variants), 1)

        # 变体信息应包含正确的 SectionIDs
        variant_key, variant_info = variants[0]
        self.assertEqual(variant_info["section_ids"], section_ids2)

    def test_variant_has_different_qsid(self):
        """变体因子应具有不同的 QSID."""
        dt_ruler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(10)]
        section_ids1 = ["000001.SZ", "000002.SZ"]
        section_ids2 = ["000001.SZ", "000003.SZ"]
        context = self._make_context(section_ids1, dt_ruler)

        factor = _SimpleFactor(name="F1")
        dt_range = self._make_dt_range(dt_ruler)

        # 通过 Engine.run 触发完整的变体处理流程
        init_data = FactorInitData(DTRange=dt_range, SectionIDs=section_ids1)
        engine = Engine()

        # 手动执行 init, 然后再次调用 init_compute 产生变体
        engine.init([factor], context, init_data_list=[init_data])

        # 产生变体
        init_data2 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids2)
        factor.init_compute(path=[factor.QSID], init_data=init_data2, context=context)

        # 手动处理变体
        engine._processSectionIDVariants(context)

        # 原因子的 QSID
        original_qsid = factor.QSID

        # 检查变体是否被创建并注册
        variant_found = False
        for qsid, node in context.NodeDict.items():
            if qsid != original_qsid and isinstance(node, _SimpleFactor):
                variant_found = True
                # 变体应有不同的 QSID
                self.assertNotEqual(qsid, original_qsid)
                # 变体的 SectionIDs 应为 section_ids2
                variant_state = context.NodeState.get(qsid, {})
                self.assertEqual(variant_state.get("section_ids"), section_ids2)
                break

        self.assertTrue(variant_found, "变体因子未被创建或注册")

    def test_variant_inherits_parameters(self):
        """变体因子应继承原因子的所有参数 (除 SectionIDs 外)."""
        section_ids1 = ["000001.SZ", "000002.SZ"]
        section_ids2 = ["000001.SZ", "000003.SZ"]

        # 直接使用 Factor 基类测试 (避免 _SimpleFactor 的构造函数干扰)
        factor = Factor(descriptors=[], args={"Name": "TestFactor", "SectionIDs": section_ids1, "CacheEnabled": False})
        variant = factor.new(args={"SectionIDs": section_ids2})

        # 变体应有不同的 QSID
        self.assertNotEqual(factor.QSID, variant.QSID)
        # 变体的 Name 应与原因子相同
        self.assertEqual(variant._QSArgs.Name, "TestFactor")
        # 变体的 SectionIDs 应为新的值
        self.assertEqual(variant._QSArgs.SectionIDs, section_ids2)
        # 变体的 CacheEnabled 应与原因子相同
        self.assertEqual(variant._QSArgs.CacheEnabled, False)

    def test_diamond_dependency_variant(self):
        """菱形依赖场景: 两个因子共享同一个依赖, 传递不同的 SectionIDs."""
        dt_ruler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(10)]
        section_ids1 = ["000001.SZ", "000002.SZ"]
        section_ids2 = ["000001.SZ", "000003.SZ"]
        context = self._make_context(section_ids1, dt_ruler)

        # 创建依赖结构: root -> [left, right], left -> leaf, right -> leaf
        leaf = _SimpleFactor(name="leaf")
        left = _SimpleFactor(name="left", descriptors=[leaf])
        right = _SimpleFactor(name="right", descriptors=[leaf])
        root = _SimpleFactor(name="root", descriptors=[left, right])

        dt_range = self._make_dt_range(dt_ruler)

        # 第一次初始化使用 section_ids1
        init_data1 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids1)
        engine = Engine()
        engine.init([root], context, init_data_list=[init_data1])

        # 验证 leaf 的 SectionIDs
        leaf_state = context.NodeState.get(leaf.QSID, {})
        self.assertEqual(leaf_state.get("section_ids"), section_ids1)

        # 第二次调用 root 使用不同的 section_ids2
        init_data2 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids2)
        root.init_compute(path=[root.QSID], init_data=init_data2, context=context)

        # 处理变体
        engine._processSectionIDVariants(context)

        # 检查是否为 leaf 创建了变体
        leaf_variants = [qsid for qsid in context.NodeDict
                         if qsid != leaf.QSID
                         and isinstance(context.NodeDict[qsid], _SimpleFactor)
                         and context.NodeDict[qsid].Name == "leaf"]

        self.assertTrue(len(leaf_variants) > 0, "leaf 的变体未被创建")

        # leaf 变体的 SectionIDs 应为 section_ids2
        for qsid in leaf_variants:
            variant_state = context.NodeState.get(qsid, {})
            self.assertEqual(variant_state.get("section_ids"), section_ids2)

    def test_no_variant_for_none_section_ids(self):
        """init_data.SectionIDs 为 None 时不应产生变体."""
        dt_ruler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(10)]
        section_ids = ["000001.SZ", "000002.SZ"]
        context = self._make_context(section_ids, dt_ruler)

        factor = _SimpleFactor(name="F1")
        dt_range = self._make_dt_range(dt_ruler)

        # 第一次调用使用明确的 SectionIDs
        init_data1 = FactorInitData(DTRange=dt_range, SectionIDs=section_ids)
        engine = Engine()
        engine.init([factor], context, init_data_list=[init_data1])

        # 第二次调用使用 None (应使用 context.SectionIDs)
        init_data2 = FactorInitData(DTRange=dt_range, SectionIDs=None)
        factor.init_compute(path=[factor.QSID], init_data=init_data2, context=context)

        # 不应产生变体 (因为 None 会回退到 context.SectionIDs, 与已记录的一致)
        self.assertNotIn("_QS_FactorSectionIDVariants", context)


if __name__ == "__main__":
    unittest.main()
