# -*- coding: utf-8 -*-
"""因子框架改造测试：计算维度分离

测试内容:
    test_make_cache_key_basic          — make_cache_key 相同参数生成相同键
    test_make_cache_key_different      — make_cache_key 不同参数生成不同键
    test_make_cache_key_order          — make_cache_key 截面顺序无关
    test_make_cache_key_section_only   — make_cache_key 仅传截面
    test_make_cache_key_dtruler_only   — make_cache_key 仅传时点标尺
    test_resolve_calc_params_default   — _resolve_calc_params 默认取 ids/dts
    test_resolve_calc_params_factor    — _resolve_calc_params 因子配置优先
    test_resolve_calc_params_caller    — _resolve_calc_params 调用者参数最高
    test_readData_section_ids          — readData 显式 section_ids 参数
    test_readData_dt_ruler             — readData 显式 dt_ruler 参数
    test_readData_different_sections   — 同一因子不同截面调用不冲突
    test_descriptor_dtruler_basic      — TimeOperation DescriptorDTRuler 基本功能
    test_descriptor_dtruler_multi_freq — 混合频率描述子

使用方法:
    python tests/test_Factor_DimensionSeparation.py
"""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor, Factor, FactorContext, FactorInitData
from QuantStudio.Factor.FactorCache import make_cache_key
import QuantStudio.Factor.FactorOperator as fo


class TestMakeCacheKey(unittest.TestCase):
    """make_cache_key 缓存键生成测试"""

    def test_make_cache_key_basic(self):
        """相同参数生成相同键"""
        qsid = "test_qsid_123"
        section = ["000001.SZ", "000002.SZ", "000003.SZ"]
        dtruler = [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
        key1 = make_cache_key(qsid, section, dtruler)
        key2 = make_cache_key(qsid, section, dtruler)
        self.assertEqual(key1, key2)

    def test_make_cache_key_different(self):
        """不同参数生成不同键"""
        qsid = "test_qsid_123"
        section_a = ["000001.SZ", "000002.SZ"]
        section_b = ["000001.SZ", "000003.SZ"]
        key_a = make_cache_key(qsid, section_a)
        key_b = make_cache_key(qsid, section_b)
        self.assertNotEqual(key_a, key_b)

    def test_make_cache_key_order(self):
        """截面顺序无关（排序后哈希）"""
        qsid = "test_qsid"
        section1 = ["000002.SZ", "000001.SZ", "000003.SZ"]
        section2 = ["000003.SZ", "000001.SZ", "000002.SZ"]
        key1 = make_cache_key(qsid, section1)
        key2 = make_cache_key(qsid, section2)
        self.assertEqual(key1, key2)

    def test_make_cache_key_section_only(self):
        """仅传截面，不传时点标尺"""
        qsid = "test_qsid"
        section = ["000001.SZ"]
        key = make_cache_key(qsid, section_ids=section)
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)

    def test_make_cache_key_dtruler_only(self):
        """仅传时点标尺，不传截面"""
        qsid = "test_qsid"
        dtruler = [dt.datetime(2025, 1, 1), dt.datetime(2025, 2, 1)]
        key = make_cache_key(qsid, dtruler=dtruler)
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)

    def test_make_cache_key_no_dims(self):
        """不传任何维度，仅用 QSID"""
        qsid = "test_qsid"
        key = make_cache_key(qsid)
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)

    def test_make_cache_key_isolation(self):
        """不同维度组合自动隔离"""
        qsid = "test_qsid"
        s1 = ["000001.SZ", "000002.SZ"]
        s2 = ["000001.SZ", "000003.SZ"]
        d1 = [dt.datetime(2025, 1, 1)]
        d2 = [dt.datetime(2025, 2, 1)]
        keys = [
            make_cache_key(qsid),
            make_cache_key(qsid, s1),
            make_cache_key(qsid, s2),
            make_cache_key(qsid, dtruler=d1),
            make_cache_key(qsid, dtruler=d2),
            make_cache_key(qsid, s1, d1),
            make_cache_key(qsid, s1, d2),
            make_cache_key(qsid, s2, d1),
        ]
        # 所有键应互不相同
        self.assertEqual(len(keys), len(set(keys)))


class TestResolveCalcParams(unittest.TestCase):
    """Factor._resolve_calc_params 计算参数解析测试"""

    @classmethod
    def setUpClass(cls):
        cls.ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        cls.dts = [dt.datetime(2025, 1, i) for i in range(1, 11)]

    def test_resolve_calc_params_default(self):
        """默认：不传 section_ids/dt_ruler，因子也无配置 → 使用 ids/dts"""
        factor = DataFactor(data=1.0, args={"Name": "test"})
        section, dtruler = factor._resolve_calc_params(self.ids, self.dts)
        self.assertEqual(section, self.ids)
        self.assertEqual(dtruler, self.dts)

    def test_resolve_calc_params_factor_config(self):
        """因子配置优先于 ids/dts"""
        fixed_section = ["000010.SZ", "000011.SZ"]
        factor = DataFactor(data=1.0, args={"Name": "test", "SectionIDs": fixed_section})
        section, dtruler = factor._resolve_calc_params(self.ids, self.dts)
        self.assertEqual(section, fixed_section)
        self.assertEqual(dtruler, self.dts)

    def test_resolve_calc_params_caller_wins(self):
        """调用者参数优先于因子配置"""
        fixed_section = ["000010.SZ", "000011.SZ"]
        caller_section = ["000020.SZ", "000021.SZ"]
        caller_dtruler = [dt.datetime(2025, 6, 1)]
        factor = DataFactor(data=1.0, args={"Name": "test", "SectionIDs": fixed_section})
        section, dtruler = factor._resolve_calc_params(
            self.ids, self.dts,
            section_ids=caller_section, dt_ruler=caller_dtruler
        )
        self.assertEqual(section, caller_section)
        self.assertEqual(dtruler, caller_dtruler)


class TestReadDataDimensionParams(unittest.TestCase):
    """Factor.readData 维度参数测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        nDT, nID = 20, 5
        cls.AllIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
        cls.SubIDs = cls.AllIDs[:3]
        cls.DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
        cls.DTs = cls.DTRuler[-5:]
        data = np.random.rand(nDT, nID)
        cls.DataFrame = pd.DataFrame(data, index=cls.DTRuler, columns=cls.AllIDs)

    def test_readData_section_ids(self):
        """readData 显式 section_ids 参数：返回 ids 子集"""
        factor = DataFactor(data=self.DataFrame, args={"Name": "f1"})
        # 返回 SubIDs，但计算截面是 AllIDs
        result = factor.readData(ids=self.SubIDs, dts=self.DTs, section_ids=self.AllIDs)
        self.assertEqual(result.shape, (len(self.DTs), len(self.SubIDs)))
        self.assertListEqual(list(result.columns), self.SubIDs)

    def test_readData_dt_ruler(self):
        """readData 显式 dt_ruler 参数"""
        factor = DataFactor(data=self.DataFrame, args={"Name": "f2"})
        # dt_ruler 是完整时点标尺，dts 是请求返回的子集
        result = factor.readData(ids=self.AllIDs, dts=self.DTs, dt_ruler=self.DTRuler)
        self.assertEqual(result.shape, (len(self.DTs), len(self.AllIDs)))
        self.assertListEqual(list(result.index), self.DTs)

    def test_readData_default_behavior(self):
        """readData 不传新参数时行为不变"""
        factor = DataFactor(data=self.DataFrame, args={"Name": "f3"})
        result_old = factor.readData(ids=self.SubIDs, dts=self.DTs)
        result_new = factor.readData(ids=self.SubIDs, dts=self.DTs, section_ids=None, dt_ruler=None)
        pd.testing.assert_frame_equal(result_old, result_new)

    def test_readData_different_sections(self):
        """同一因子不同截面调用不冲突"""
        factor = DataFactor(data=self.DataFrame, args={"Name": "f4"})
        # 第一次调用：全截面
        result_all = factor.readData(ids=self.AllIDs, dts=self.DTs)
        # 第二次调用：子截面
        result_sub = factor.readData(ids=self.SubIDs, dts=self.DTs)
        self.assertEqual(result_all.shape, (len(self.DTs), len(self.AllIDs)))
        self.assertEqual(result_sub.shape, (len(self.DTs), len(self.SubIDs)))
        # 子截面的结果应与全截面中对应列一致
        pd.testing.assert_frame_equal(
            result_sub,
            result_all.loc[:, self.SubIDs]
        )


class TestDataFactorDimension(unittest.TestCase):
    """DataFactor 维度分离测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        nDT, nID = 30, 8
        cls.AllIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
        cls.DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
        cls.Data = pd.DataFrame(
            np.random.rand(nDT, nID),
            index=cls.DTRuler,
            columns=cls.AllIDs
        )

    def test_dataFactor_scalar(self):
        """DataFactor 标量数据，任意截面/时点均可返回"""
        factor = DataFactor(data=1.0, args={"Name": "scalar"})
        ids = ["A.SZ", "B.SZ"]
        dts = [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
        result = factor.readData(ids=ids, dts=dts)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue((result.values == 1.0).all())

    def test_dataFactor_dataframe_reindex(self):
        """DataFactor DataFrame 数据，reindex 到请求的 ids/dts"""
        factor = DataFactor(data=self.Data, args={"Name": "df_factor"})
        subset_ids = self.AllIDs[:4]
        subset_dts = self.DTRuler[:10]
        result = factor.readData(ids=subset_ids, dts=subset_dts)
        self.assertEqual(result.shape, (10, 4))
        self.assertListEqual(list(result.columns), subset_ids)

    def test_dataFactor_section_override(self):
        """DataFactor 使用 section_ids 覆盖因子配置"""
        fixed_section = self.AllIDs[:5]
        factor = DataFactor(data=self.Data, args={"Name": "f", "SectionIDs": fixed_section})
        # 调用时传入更大的截面
        result = factor.readData(ids=self.AllIDs, dts=self.DTRuler[:5], section_ids=self.AllIDs)
        self.assertEqual(result.shape, (5, len(self.AllIDs)))


class TestDescriptorDTRuler(unittest.TestCase):
    """TimeOperation DescriptorDTRuler 测试"""

    def test_descriptor_dtruler_basic(self):
        """TimeOperation 基本功能：DescriptorDTRuler 不影响单频率场景"""
        np.random.seed(99)
        nDT = 20
        dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
        dts = dtr[-5:]
        ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        data = pd.DataFrame(np.random.rand(nDT, 3), index=dtr, columns=ids)
        factor = DataFactor(data=data, args={"Name": "price"})

        # RollingApply 是 TimeOperator 的子类，不指定 DescriptorDTRuler
        op = fo.RollingApply(func=np.nanmean, window=5, min_periods=3)
        result = op(factor, factor_args={"Name": "ma5"})
        out = result.readData(ids=ids, dts=dts, dt_ruler=dtr)
        self.assertEqual(out.shape, (len(dts), len(ids)))

    def test_descriptor_dtruler_multi_freq(self):
        """混合频率：日度因子使用月度描述子"""
        np.random.seed(88)
        # 月度数据
        monthly_dtr = [dt.datetime(2025, m, 1) for m in range(1, 13)]
        monthly_data = pd.DataFrame(
            np.random.rand(12, 3),
            index=monthly_dtr,
            columns=["A.SZ", "B.SZ", "C.SZ"]
        )
        monthly_factor = DataFactor(data=monthly_data, args={"Name": "monthly"})

        # 日度时点标尺
        daily_dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(60)]
        daily_dts = daily_dtr[-10:]
        ids = ["A.SZ", "B.SZ", "C.SZ"]

        # 使用 operator_kwargs 传入 DescriptorDTRuler 指定描述子是月度频率
        op = fo.RollingApply(func=np.nanmean, window=2, min_periods=1)
        result = op(monthly_factor, factor_args={"Name": "monthly_ma"},
                    operator_kwargs={"DescriptorDTRuler": [monthly_dtr]})
        out = result.readData(ids=ids, dts=daily_dts, dt_ruler=daily_dtr)
        self.assertEqual(out.shape, (len(daily_dts), len(ids)))

    def test_descriptor_dtruler_field_on_operator(self):
        """TimeOperator DescriptorDTRuler 字段正确初始化"""
        op = fo.RollingApply(func=np.nanmean, window=3, min_periods=1)
        # 默认应为 [None]（Arity=1）
        self.assertEqual(len(op._QSArgs.DescriptorDTRuler), 1)
        self.assertIsNone(op._QSArgs.DescriptorDTRuler[0])

    def test_descriptor_dtruler_auto_pad(self):
        """DescriptorDTRuler 长度不足时自动填充 None"""
        from QuantStudio.Factor.FactorOperation import TimeOperator
        # 手动构造一个 DescriptorDTRuler 长度不足的算子
        op = TimeOperator(
            args={
                "OperatorType": "Time",
                "Name": "test_op",
                "Arity": 3,
                "LookBack": [1, 1, 1],
                "StartDT": [None, None, None],
                "DescriptorDTRuler": [None, None],  # 少传一个
            }
        )
        # 应自动填充到长度 3
        self.assertEqual(len(op._QSArgs.DescriptorDTRuler), 3)
        self.assertIsNone(op._QSArgs.DescriptorDTRuler[2])


class TestSectionConsistencyRemoved(unittest.TestCase):
    """截面一致性检查移除测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(77)
        nDT, nID = 20, 10
        cls.AllIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
        cls.DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
        cls.Data = pd.DataFrame(
            np.random.rand(nDT, nID),
            index=cls.DTRuler,
            columns=cls.AllIDs
        )

    def test_same_factor_different_sections(self):
        """同一因子可以被不同截面调用，不抛异常"""
        factor = DataFactor(data=self.Data, args={"Name": "f"})
        sub1 = self.AllIDs[:5]
        sub2 = self.AllIDs[5:]
        dts = self.DTRuler[-5:]
        # 两次调用不同截面，不应抛异常
        r1 = factor.readData(ids=sub1, dts=dts)
        r2 = factor.readData(ids=sub2, dts=dts)
        self.assertEqual(r1.shape, (5, 5))
        self.assertEqual(r2.shape, (5, 5))

    def test_section_merge_in_init_compute(self):
        """Factor.init_compute 中截面合并：不同调用者的截面取并集"""
        # DataFactor 的 init_compute 不走基类逻辑，
        # 用一个带描述子的衍生因子来验证截面合并
        np.random.seed(55)
        data = pd.DataFrame(
            np.random.rand(len(self.DTRuler), len(self.AllIDs)),
            index=self.DTRuler,
            columns=self.AllIDs
        )
        base = DataFactor(data=data, args={"Name": "base"})
        # 使用 Applymap（PointOperator）创建衍生因子，触发 Factor.init_compute
        op = fo.Applymap(func=lambda x: x)
        factor = op(base, factor_args={"Name": "derived"})

        sub1 = self.AllIDs[:3]
        sub2 = self.AllIDs[2:6]
        dts = self.DTRuler[-5:]
        context = FactorContext(DTRuler=self.DTRuler, SectionIDs=self.AllIDs)

        # 第一次初始化
        init1 = FactorInitData(DTRange=(dts[0], dts[-1]), SectionIDs=sub1)
        factor.init_compute(path=[factor.QSID], init_data=init1, context=context)
        state = context.NodeState.get(factor.QSID, {})
        self.assertIn("section_ids", state)
        self.assertEqual(sorted(state["section_ids"]), sorted(sub1))

        # 第二次初始化，截面不同 → 应合并（不抛异常）
        init2 = FactorInitData(DTRange=(dts[0], dts[-1]), SectionIDs=sub2)
        factor.init_compute(path=[factor.QSID], init_data=init2, context=context)
        state = context.NodeState.get(factor.QSID, {})
        expected = sorted(set(sub1 + sub2))
        self.assertEqual(sorted(state["section_ids"]), expected)


if __name__ == "__main__":
    unittest.main()
