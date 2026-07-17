# -*- coding: utf-8 -*-
"""QuantStudio 因子算子测试模块

使用方法:
    * 运行全部测试: python tests/test_Factor_FactorOperator.py
    * 运行指定测试: python -m unittest tests.test_Factor_FactorOperator.TestFactorOperator.test_rollingApply
    * 通过 TestSuite 指定测试 (取消文件末尾的注释并修改):

        Suite = unittest.TestSuite()
        Suite.addTest(TestFactorOperator("test_rollingApply"))
        Runner = unittest.TextTestRunner()
        Runner.run(Suite)

    注意: 测试使用随机生成的数据, 不依赖外部数据库。

测试方法:
    test_rollingApply              — RollingApply 算子基本功能, 验证输出形状和 NaN 处理
    test_rollingApply_minPeriods   — RollingApply 窗口内有效数据不足时返回 NaN
    test_aggregate_basic           — Aggregate 无 cat_data 时截面聚合, 广播标量到所有 ID
    test_aggregate_with_mask       — Aggregate 带 mask 时仅对掩码为 1 的数据聚合
    test_aggregate_with_catData    — Aggregate 带 cat_data 时按类别分组聚合
    test_aggregateComponent_basic  — AggregateComponent 按成分 ID 列表聚合值
    test_aggregateComponent_noMatch — AggregateComponent 成分 ID 不在 SectionIDs 中时返回 NaN
    test_aggregateComponent_with_weight — AggregateComponent 带权重输入的加权聚合
"""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor
import QuantStudio.Factor.FactorOperator as fo


class TestFactorOperator(unittest.TestCase):
    """因子算子基本功能测试"""

    @classmethod
    def setUpClass(cls):
        """生成测试用的随机数据和因子对象"""
        np.random.seed(0)
        nDT, nID = 100, 10
        cls.SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
        cls.IDs = cls.SectionIDs[:3]
        cls.DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
        cls.DTs = cls.DTRuler[-5:]

        cls.InitData = DataFactor(data=1, args={"Name": "init"})
        F1 = np.random.rand(nDT, nID)
        F1[-8:-5, 0] = np.nan
        F1[:, 2] = np.nan
        cls.F1 = DataFactor(data=pd.DataFrame(F1, index=cls.DTRuler, columns=cls.SectionIDs), args={"Name": "F1"})
        cls.Open = DataFactor(
            data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=cls.DTRuler, columns=cls.SectionIDs),
            args={"Name": "open"},
        )
        cls.Close = DataFactor(
            data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=cls.DTRuler, columns=cls.SectionIDs),
            args={"Name": "close"},
        )

    # ==================== RollingApply 测试 ====================

    def test_rollingApply(self):
        """测试 RollingApply 算子: 输出形状正确, NaN 列全为 NaN"""
        rollingSum = fo.RollingApply(func=np.nanmean, window=5, min_periods=3)
        TestF = rollingSum(self.F1, factor_args={"Name": "TestF"})
        TestData = TestF.readData(ids=self.IDs, dts=self.DTs, dt_ruler=self.DTRuler)

        self.assertIsInstance(TestData, pd.DataFrame)
        self.assertEqual(TestData.shape, (len(self.DTs), len(self.IDs)))
        # 全 NaN 列 (SectionIDs[2]) 应保持全 NaN
        self.assertTrue(TestData.iloc[:, 2].isna().all())

    def test_rollingApply_minPeriods(self):
        """测试 RollingApply min_periods: 窗口内有效数据不足时返回 NaN"""
        # 构造只有前 2 行有值、其余全 NaN 的数据, window=5, min_periods=3
        nDT, nID = 20, 3
        dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
        dts = dtr[-5:]  # 取末尾 5 个时点, 保证 lookback 足够
        ids = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
        data = np.full((nDT, nID), np.nan)
        data[0, 0] = 1.0
        data[1, 0] = 2.0
        factor = DataFactor(data=pd.DataFrame(data, index=dtr, columns=ids), args={"Name": "sparse"})

        op = fo.RollingApply(func=np.nansum, window=5, min_periods=3)
        result = op(factor, factor_args={"Name": "sparse_result"})
        out = result.readData(ids=ids, dts=dts, dt_ruler=dtr)
        # 有效数据不足 min_periods, 结果应全 NaN
        self.assertTrue(out.iloc[:, 0].isna().all())

    def test_rollingApply_outputShape(self):
        """测试 RollingApply 输出形状与请求的 dts/ids 一致"""
        op = fo.RollingApply(func=np.nanmean, window=3, min_periods=2)
        TestF = op(self.F1, factor_args={"Name": "shapeTest"})
        TestData = TestF.readData(ids=self.SectionIDs, dts=self.DTs, dt_ruler=self.DTRuler)
        self.assertEqual(TestData.shape, (len(self.DTs), len(self.SectionIDs)))

    # ==================== Aggregate 测试 ====================

    def test_aggregate_basic(self):
        """测试 Aggregate 无 cat_data: 截面聚合后广播标量到所有 ID"""
        ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(3)]
        dts = dtr[-1:]
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        F = DataFactor(data=pd.DataFrame(data, index=dtr, columns=ids), args={"Name": "agg_in"})

        TestF = fo.Aggregate(aggr_func=np.nansum)(F, factor_args={"Name": "agg_out"})
        TestData = TestF.readData(ids=ids, dts=dts, dt_ruler=dtr)

        self.assertEqual(TestData.shape, (1, 3))
        # 7+8+9=24, 广播到所有 ID
        np.testing.assert_array_equal(TestData.values, [[24.0, 24.0, 24.0]])

    def test_aggregate_with_mask(self):
        """测试 Aggregate 带 mask: 仅对掩码为 1 的数据聚合"""
        ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(3)]
        dts = dtr[-1:]
        data = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        mask_data = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        F = DataFactor(data=pd.DataFrame(data, index=dtr, columns=ids), args={"Name": "mask_in"})
        Mask = DataFactor(data=pd.DataFrame(mask_data, index=dtr, columns=ids), args={"Name": "mask"})

        TestF = fo.Aggregate(aggr_func=np.nansum)(F, Mask, factor_args={"Name": "mask_agg_out"})
        TestData = TestF.readData(ids=ids, dts=dts, dt_ruler=dtr)

        self.assertEqual(TestData.shape, (1, 3))
        # 10+0+30=40, 广播到所有 ID
        np.testing.assert_array_equal(TestData.values, [[40.0, 40.0, 40.0]])

    def test_aggregate_with_catData(self):
        """测试 Aggregate 带 cat_data: 按类别分组聚合"""
        ids = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
        dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(3)]
        dts = dtr[-1:]
        data = np.array([[1.0, 2.0, 3.0, 4.0]] * 3)
        cat_data = np.array([["A", "B", "A", "B"]] * 3)
        F = DataFactor(data=pd.DataFrame(data, index=dtr, columns=ids), args={"Name": "cat_in"})
        Cat = DataFactor(data=pd.DataFrame(cat_data, index=dtr, columns=ids), args={"Name": "cat"})

        TestF = fo.Aggregate(aggr_func=np.nansum)(F, cat_data=Cat, factor_args={"Name": "cat_agg_out"})
        TestData = TestF.readData(ids=ids, dts=dts, dt_ruler=dtr)

        self.assertEqual(TestData.shape, (1, 4))
        # A 组: 1+3=4, B 组: 2+4=6
        np.testing.assert_array_equal(TestData.values, [[4.0, 6.0, 4.0, 6.0]])

    # ==================== AggregateComponent 测试 ====================

    def test_aggregateComponent_basic(self):
        """测试 AggregateComponent: 按成分 ID 列表从 Value 中取值并聚合"""
        section_ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        component_ids = ["000016.SH", "000300.SH"]
        dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(3)]
        dts = dtr[-1:]
        # Value: 按 section_ids 索引
        val_data = np.array([[10.0, 20.0, 30.0]] * 3)
        F = DataFactor(data=pd.DataFrame(val_data, index=dtr, columns=section_ids), args={"Name": "val"})
        # Component: columns=section_ids, 每个 cell 是该 section 所属的 component 列表
        comp_data = pd.DataFrame({
            "000001.SZ": [["000016.SH", "000300.SH"]] * 3,
            "000002.SZ": [["000300.SH"]] * 3,
            "000003.SZ": [["000016.SH", "000300.SH"]] * 3,
        }, index=dtr)
        Comp = DataFactor(data=comp_data, args={"Name": "comp"})

        op = fo.AggregateComponent(aggr_func=np.nanmean, descriptor_ids=section_ids)
        TestF = op(F, Comp, factor_args={"Name": "comp_out"})
        TestData = TestF.readData(ids=component_ids, dts=dts, dt_ruler=dtr)

        self.assertEqual(TestData.shape, (1, 2))
        # 000016: sections=[000001, 000003] → mean(10, 30)=20; 000300: sections=[000001, 000002, 000003] → mean(10, 20, 30)=20
        np.testing.assert_array_almost_equal(TestData.values, [[20.0, 20.0]])

    def test_aggregateComponent_with_weight(self):
        """测试 AggregateComponent: 带权重输入的加权聚合"""
        def _weighted_mean(v, w):
            return np.nansum(v * w) / np.nansum(w)

        section_ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        component_ids = ["000016.SH", "000300.SH"]
        dtr = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(3)]
        dts = dtr[-1:]
        val_data = np.array([[10.0, 20.0, 30.0]] * 3)
        F = DataFactor(data=pd.DataFrame(val_data, index=dtr, columns=section_ids), args={"Name": "wval"})
        # Component: columns=section_ids, 每个 cell 是该 section 所属的 component 列表
        comp_data = pd.DataFrame({
            "000016.SH": [["000001.SZ", "000003.SZ"]] * 3,
            "000300.SH": [["000001.SZ", "000002.SZ", "000003.SZ"]] * 3,
        }, index=dtr)
        Comp = DataFactor(data=comp_data, args={"Name": "wcomp"})
        # Weight: 与 Component 结构一致, 每个 cell 是权重列表
        weight_data = pd.DataFrame({
            "000016.SH": [[1.0, 2.0]] * 3,
            "000300.SH": [[3.0, 3.0, 2.0]] * 3,
        }, index=dtr)
        Weight = DataFactor(data=weight_data, args={"Name": "ww"})

        op = fo.AggregateComponent(aggr_func=_weighted_mean, descriptor_ids=section_ids)
        TestF = op(F, Comp, Weight, factor_args={"Name": "wcomp_out"})
        TestData = TestF.readData(ids=component_ids, dts=dts, dt_ruler=dtr)

        self.assertEqual(TestData.shape, (1, 2))
        # 000016: sections=[000001, 000003] → v=[10,30], w=[1,2] → (10*1+30*2)/(1+2)=70/3
        # 000300: sections=[000001, 000002, 000003] → v=[10,20,30], w=[3,3,2] → (10*3+20*3+30*2)/(3+3+2)=150/8
        np.testing.assert_array_almost_equal(TestData.values, [[70.0 / 3, 150.0 / 8]])


if __name__ == "__main__":
    # unittest.main()
    Suite = unittest.TestSuite()
    Suite.addTest(TestFactorOperator("test_aggregateComponent_with_weight"))
    Runner = unittest.TextTestRunner(verbosity=2)
    Runner.run(Suite)
