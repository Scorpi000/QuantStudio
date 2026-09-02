# -*- coding: utf-8 -*-
"""测试 BackTest.PerformanceAnalysis 模块中的算子和回测节点

测试覆盖:
- CalcBrinsonModel: 构造参数、__call__ 返回类型、calculate 核心计算
- BrinsonModel: backward_compute 输出结构和统计逻辑
- BrinsonModelReport: 报告生成和 HTML 输出
"""
import unittest
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.FactorOperation import PanelOperation
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.BackTest.PerformanceAnalysis.BrinsonModel import (
    CalcBrinsonModel, BrinsonModel, BrinsonModelReport
)


def _run_factor_engine(factor, dt_ruler, section_ids, lookback=0):
    """通过 Engine 执行因子计算并返回结果

    Args:
        factor: 待计算的因子节点
        dt_ruler: 计算时点序列 (不含 lookback 补充时点)
        section_ids: 输出截面 ID
        lookback: 回溯期数, 用于在 dt_ruler 前添加额外时点到 DTRuler
    """
    extra_dts = [dt_ruler[0] - dt.timedelta(days=lookback - i) for i in range(lookback)]
    full_dtruler = extra_dts + list(dt_ruler)
    fwd_data = FactorLocalContext(DTs=dt_ruler, IDs=section_ids, SectionIDs=section_ids)
    with FeatherFactorCache(args={"DTRuler": full_dtruler, "StartMode": "new", "CacheDir": None}) as Cache:
        with FactorContext(PID="0", PIDList=["0"], DTRuler=full_dtruler, SectionIDs=section_ids, DataCache=Cache) as Context:
            with Engine() as ExecEngine:
                Output = ExecEngine.run([factor], Context, fwd_data_list=[fwd_data])
    return Output[0]


def _make_brinson_backward_data(dts, cats):
    """构造 BrinsonModel.backward_compute 所需的结构化输入数据

    Args:
        dt_ruler: 时点序列
        cats: 资产类别列表

    Returns:
        包含 8 个字段 (BMK, BMKR, TP, TPR, AA, SS, IN, AAA) 的 DataFrame
    """
    n_dt, n_cat = len(dts), len(cats)
    dtype = np.dtype([("BMK", float), ("BMKR", float), ("TP", float), ("TPR", float),
                      ("AA", float), ("SS", float), ("IN", float), ("AAA", float)])
    data = np.empty((n_dt, n_cat), dtype="O")
    # 构造简单的 Brinson 归因数据:
    # BMK=0.5, TP=0.5, BMKR=0.01, TPR=0.02, AA=0.005, SS=0.005, IN=0.0, AAA=0.005
    for i in range(n_dt):
        for j in range(n_cat):
            data[i, j] = (0.5, 0.01, 0.5, 0.02, 0.005, 0.005, 0.0, 0.005)
    return pd.DataFrame(data, index=dts, columns=cats)


class TestCalcBrinsonModelInit(unittest.TestCase):
    """测试 CalcBrinsonModel 构造函数参数设置"""

    def test_basic_init(self):
        """验证基本参数初始化: DTMode、DataType、CompoundType、LookBack"""
        section_ids = ["行业A", "行业B", "行业C"]
        descriptor_ids = ["000001.SZ", "000002.SZ"]
        calc = CalcBrinsonModel(section_ids=section_ids, descriptor_ids=descriptor_ids, lookback=10)

        self.assertEqual(calc._QSArgs.DTMode, "多时点")
        self.assertEqual(calc._QSArgs.DataType, "object")
        self.assertEqual(calc._QSArgs.LookBack[0], 10)
        self.assertEqual(calc._QSArgs.DescriptorSection[0], descriptor_ids)
        # CompoundType 应包含 8 个字段
        ct = calc._QSArgs.CompoundType
        field_names = [f[0] for f in ct]
        self.assertEqual(field_names, ["BMK", "BMKR", "TP", "TPR", "AA", "SS", "IN", "AAA"])

    def test_default_lookback(self):
        """验证默认 lookback=31"""
        calc = CalcBrinsonModel(section_ids=["A"], descriptor_ids=["X"])
        self.assertEqual(calc._QSArgs.LookBack[0], 31)

    def test_model_args_section_ids(self):
        """验证 section_ids 存入 ModelArgs"""
        section_ids = ["化工", "电子", "医药"]
        calc = CalcBrinsonModel(section_ids=section_ids, descriptor_ids=["X"])
        self.assertEqual(calc._QSArgs.ModelArgs["section_ids"], section_ids)


class TestCalcBrinsonModelCall(unittest.TestCase):
    """测试 CalcBrinsonModel.__call__ 返回类型和因子连接"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_dt = 5
        cls.n_ids = 4
        cls.DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(cls.n_dt)]
        cls.IDs = [f"00000{i}.SZ" for i in range(1, cls.n_ids + 1)]
        cls.Cats = ["行业A", "行业B"]

    def _make_factors(self):
        """创建价格、权重、分类因子"""
        PriceDF = pd.DataFrame(np.ones((self.n_dt, self.n_ids)) * 100,
                               index=self.DTs, columns=self.IDs)
        WeightDF = pd.DataFrame(np.ones((self.n_dt, self.n_ids)) / self.n_ids,
                                index=self.DTs, columns=self.IDs)
        CatDF = pd.DataFrame(
            np.tile(["行业A", "行业A", "行业B", "行业B"], (self.n_dt, 1)),
            index=self.DTs, columns=self.IDs
        )
        PriceFactor = DataFactor(data=PriceDF, args={"Name": "Price"})
        WeightFactor = DataFactor(data=WeightDF, args={"Name": "Weight"})
        CatFactor = DataFactor(data=CatDF, args={"Name": "Category"})
        return WeightFactor, PriceFactor, CatFactor

    def test_returns_panel_operation(self):
        """验证 __call__ 返回 PanelOperation 类型"""
        calc = CalcBrinsonModel(section_ids=self.Cats, descriptor_ids=self.IDs, lookback=1)
        WeightFactor, PriceFactor, CatFactor = self._make_factors()
        result = calc(WeightFactor, PriceFactor, CatFactor)
        self.assertIsInstance(result, PanelOperation)

    def test_section_ids_propagation(self):
        """验证 SectionIDs 从算子传递到输出因子"""
        calc = CalcBrinsonModel(section_ids=self.Cats, descriptor_ids=self.IDs, lookback=1)
        WeightFactor, PriceFactor, CatFactor = self._make_factors()
        result = calc(WeightFactor, PriceFactor, CatFactor)
        self.assertEqual(result._QSArgs.SectionIDs, self.Cats)

    def test_with_benchmark(self):
        """验证带基准组合的因子创建"""
        calc = CalcBrinsonModel(section_ids=self.Cats, descriptor_ids=self.IDs, lookback=1)
        WeightFactor, PriceFactor, CatFactor = self._make_factors()
        BmkDF = pd.DataFrame(np.ones((self.n_dt, self.n_ids)) / self.n_ids,
                             index=self.DTs, columns=self.IDs)
        BmkFactor = DataFactor(data=BmkDF, args={"Name": "Bmk"})
        result = calc(WeightFactor, PriceFactor, CatFactor, bmk=BmkFactor)
        self.assertIsInstance(result, PanelOperation)
        self.assertTrue(result._QSArgs.ModelArgs["bmk"])


class TestCalcBrinsonModelCalculate(unittest.TestCase):
    """测试 CalcBrinsonModel 通过 Engine 执行的端到端计算"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_dt = 5
        cls.n_ids = 4
        # 多加 2 个时点用于 lookback=1
        cls.DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(cls.n_dt + 2)]
        cls.IDs = [f"00000{i}.SZ" for i in range(1, cls.n_ids + 1)]
        cls.Cats = ["行业A", "行业B"]
        cls.ComputeDTs = cls.DTs[1:]  # 跳过第0天, 从第1天开始计算

    def test_basic_brinson_output(self):
        """验证 Brinson 算子输出包含所有子因子"""
        # 价格: 每天涨 1%
        PriceData = np.ones((len(self.DTs), self.n_ids))
        for t in range(1, len(self.DTs)):
            PriceData[t] = PriceData[t-1] * 1.01
        PriceDF = pd.DataFrame(PriceData, index=self.DTs, columns=self.IDs)

        WeightDF = pd.DataFrame(
            np.ones((len(self.DTs), self.n_ids)) / self.n_ids,
            index=self.DTs, columns=self.IDs
        )
        CatDF = pd.DataFrame(
            np.tile(["行业A", "行业A", "行业B", "行业B"], (len(self.DTs), 1)),
            index=self.DTs, columns=self.IDs
        )

        PriceFactor = DataFactor(data=PriceDF, args={"Name": "Price"})
        WeightFactor = DataFactor(data=WeightDF, args={"Name": "Weight"})
        CatFactor = DataFactor(data=CatDF, args={"Name": "Category"})

        calc = CalcBrinsonModel(section_ids=self.Cats, descriptor_ids=self.IDs, lookback=1)
        BrinsonFactor = calc(WeightFactor, PriceFactor, CatFactor)

        Rslt = _run_factor_engine(BrinsonFactor, self.ComputeDTs,
                                  section_ids=self.Cats, lookback=1)
        self.assertIsInstance(Rslt, pd.DataFrame)
        self.assertGreater(len(Rslt), 0)
        # 输出应有 行业A 和 行业B 两列
        self.assertIn("行业A", Rslt.columns)
        self.assertIn("行业B", Rslt.columns)

    def test_with_benchmark(self):
        """验证带基准的 Brinson 归因计算"""
        PriceData = np.ones((len(self.DTs), self.n_ids))
        for t in range(1, len(self.DTs)):
            PriceData[t] = PriceData[t-1] * 1.01
        PriceDF = pd.DataFrame(PriceData, index=self.DTs, columns=self.IDs)

        WeightDF = pd.DataFrame(
            np.ones((len(self.DTs), self.n_ids)) / self.n_ids,
            index=self.DTs, columns=self.IDs
        )
        BmkDF = pd.DataFrame(
            np.tile([0.3, 0.3, 0.2, 0.2], (len(self.DTs), 1)).astype(float),
            index=self.DTs, columns=self.IDs
        )
        CatDF = pd.DataFrame(
            np.tile(["行业A", "行业A", "行业B", "行业B"], (len(self.DTs), 1)),
            index=self.DTs, columns=self.IDs
        )

        PriceFactor = DataFactor(data=PriceDF, args={"Name": "Price"})
        WeightFactor = DataFactor(data=WeightDF, args={"Name": "Weight"})
        BmkFactor = DataFactor(data=BmkDF, args={"Name": "Bmk"})
        CatFactor = DataFactor(data=CatDF, args={"Name": "Category"})

        calc = CalcBrinsonModel(section_ids=self.Cats, descriptor_ids=self.IDs, lookback=1)
        BrinsonFactor = calc(WeightFactor, PriceFactor, CatFactor, bmk=BmkFactor)

        Rslt = _run_factor_engine(BrinsonFactor, self.ComputeDTs,
                                  section_ids=self.Cats, lookback=1)
        self.assertIsInstance(Rslt, pd.DataFrame)
        self.assertGreater(len(Rslt), 0)


class TestBrinsonModelNode(unittest.TestCase):
    """测试 BrinsonModel 回测节点的 backward_compute 方法"""

    def _make_brinson_node(self, section_ids=None):
        """创建 BrinsonModel 节点实例 (绕过 __init__)"""
        node = BrinsonModel.__new__(BrinsonModel)
        cats = section_ids or ["行业A", "行业B"]
        node._QSArgs = type('Args', (), {
            'Name': 'Brinson 绩效分析模型',
            'SectionIDs': cats,
        })()
        node.Deps = [type('Dep', (), {
            'Args': type('Args', (), {'SectionIDs': cats})(),
            '_QSArgs': type('Args', (), {'CalcDTRuler': None})(),
        })()]
        return node

    def test_backward_compute_output_keys(self):
        """验证 backward_compute 输出包含所有必需的键"""
        dts = [dt.datetime(2025, 1, i+1) for i in range(3)]
        cats = ["行业A", "行业B"]
        BwdData = _make_brinson_backward_data(dts, cats)

        node = self._make_brinson_node(section_ids=cats)
        Output = node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

        ExpectedKeys = [
            "策略组合资产权重", "基准组合资产权重",
            "策略组合资产收益", "基准组合资产收益",
            "主动资产配置超额收益", "主动个券选择超额收益",
            "交互作用超额收益", "总超额收益",
            "主动资产配置组合收益", "主动个券选择组合收益",
            "主动资产配置组合收益(修正)",
            "总计", "多期综合",
        ]
        for key in ExpectedKeys:
            self.assertIn(key, Output, f"缺少输出键: {key}")

    def test_backward_compute_total_summary(self):
        """验证 '总计' 行为: 策略组合资产权重之和, 收益之和等"""
        dts = [dt.datetime(2025, 1, i+1) for i in range(3)]
        cats = ["行业A", "行业B"]
        BwdData = _make_brinson_backward_data(dts, cats)

        node = self._make_brinson_node(section_ids=cats)
        Output = node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

        总计 = Output["总计"]
        self.assertIsInstance(总计, pd.DataFrame)
        # 每行 BMK=0.5, TP=0.5, 两个类别 → 总和=1.0
        self.assertTrue((总计["策略组合资产权重"] - 1.0).abs().max() < 1e-10)
        self.assertTrue((总计["基准组合资产权重"] - 1.0).abs().max() < 1e-10)

    def test_backward_compute_excess_return(self):
        """验证总超额收益 = 策略组合收益 - 基准组合收益"""
        dts = [dt.datetime(2025, 1, i+1) for i in range(3)]
        cats = ["行业A", "行业B"]
        BwdData = _make_brinson_backward_data(dts, cats)

        node = self._make_brinson_node(section_ids=cats)
        Output = node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

        总超额 = Output["总超额收益"]
        策略收益 = Output["策略组合资产收益"]
        基准收益 = Output["基准组合资产收益"]
        diff = (策略收益 - 基准收益 - 总超额).abs().max().max()
        self.assertLess(diff, 1e-10)

    def test_backward_compute_multi_period(self):
        """验证 '多期综合' 包含累计收益和 k 因子调整"""
        dts = [dt.datetime(2025, 1, i+1) for i in range(3)]
        cats = ["行业A", "行业B"]
        BwdData = _make_brinson_backward_data(dts, cats)

        node = self._make_brinson_node(section_ids=cats)
        Output = node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

        多期综合 = Output["多期综合"]
        self.assertIsInstance(多期综合, pd.DataFrame)
        self.assertIn("总计", 多期综合.index)
        # 多期综合应包含关键列
        for col in ["策略组合收益", "基准组合收益", "总超额收益",
                     "主动资产配置超额收益", "主动个券选择超额收益", "交互作用超额收益"]:
            self.assertIn(col, 多期综合.columns, f"多期综合缺少列: {col}")
        # 总超额收益 = 策略 - 基准 (在总计行)
        total_excess = 多期综合.loc["总计", "总超额收益"]
        total_strat = 多期综合.loc["总计", "策略组合收益"]
        total_bmk = 多期综合.loc["总计", "基准组合收益"]
        self.assertAlmostEqual(total_excess, total_strat - total_bmk, places=10)

    def test_backward_compute_decomposition_sums(self):
        """验证 Brinson 分解项与组合收益的关系
        总超额收益 = 策略组合收益 - 基准组合收益 (这是 backward_compute 中的定义)
        主动资产配置组合收益 = AA + 基准收益
        主动个券选择组合收益 = SS + 基准收益
        """
        dts = [dt.datetime(2025, 1, i+1) for i in range(3)]
        cats = ["行业A", "行业B"]
        BwdData = _make_brinson_backward_data(dts, cats)

        node = self._make_brinson_node(section_ids=cats)
        Output = node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

        # 验证组合收益的构造关系
        AA_combo = Output["主动资产配置组合收益"]
        SS_combo = Output["主动个券选择组合收益"]
        AA = Output["主动资产配置超额收益"]
        SS = Output["主动个券选择超额收益"]
        BMKR = Output["基准组合资产收益"]

        diff_aa = (AA_combo - AA - BMKR).abs().max().max()
        diff_ss = (SS_combo - SS - BMKR).abs().max().max()
        self.assertLess(diff_aa, 1e-10)
        self.assertLess(diff_ss, 1e-10)


class TestBrinsonModelReport(unittest.TestCase):
    """测试 BrinsonModelReport 报告生成节点"""

    def _make_report_output(self, dts, cats):
        """构造 BrinsonModel.backward_compute 的输出字典用于报告测试"""
        BwdData = _make_brinson_backward_data(dts, cats)
        node = BrinsonModel.__new__(BrinsonModel)
        node._QSArgs = type('Args', (), {
            'Name': 'Brinson 绩效分析模型',
            'SectionIDs': cats,
        })()
        node.Deps = [type('Dep', (), {
            'Args': type('Args', (), {'SectionIDs': cats})(),
            '_QSArgs': type('Args', (), {'CalcDTRuler': None})(),
        })()]
        return node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

    def _make_report_node(self):
        """创建 BrinsonModelReport 节点实例 (绕过 __init__)"""
        report_node = BrinsonModelReport.__new__(BrinsonModelReport)
        report_node._QSArgs = type('Args', (), {
            'Name': 'Brinson绩效分析报告',
            'ReportKey': 'Report',
        })()
        # Deps[0] 是 BrinsonModel, 其 Deps[0] 是 Brinson 因子
        report_node.Deps = [type('BrinsonNode', (), {
            'Deps': [type('BrinsonFactor', (), {
                '_QSArgs': type('Args', (), {'CalcDTRuler': None})(),
            })()],
        })()]
        return report_node

    def test_gen_output_report_html(self):
        """验证 genOutputReport 生成有效的 HTML 表格"""
        dts = [dt.datetime(2025, 1, i+1) for i in range(3)]
        cats = ["行业A", "行业B"]
        Output = self._make_report_output(dts, cats)

        HTML = BrinsonModelReport.genOutputReport(Output)
        self.assertIsInstance(HTML, str)
        self.assertIn("<table", HTML)
        self.assertIn("align=\"center\"", HTML)
        # 应包含百分比格式的数据
        self.assertIn("%", HTML)

    def test_backward_compute_adds_report(self):
        """验证 backward_compute 将报告写入 ReportKey"""
        dts = [dt.datetime(2025, 1, i+1) for i in range(3)]
        cats = ["行业A", "行业B"]
        Output = self._make_report_output(dts, cats)

        report_node = self._make_report_node()
        Result = report_node.backward_compute(path=[], bwd_data_list=[Output], context=None)

        self.assertIn("Report", Result)
        self.assertIsInstance(Result["Report"], str)
        self.assertIn("<table", Result["Report"])
        # 应包含参数设置信息
        self.assertIn("参数设置", Result["Report"])

    def test_report_contains_all_categories(self):
        """验证报告 HTML 包含所有资产类别"""
        dts = [dt.datetime(2025, 1, i+1) for i in range(2)]
        cats = ["消费", "科技", "金融"]
        BwdData = _make_brinson_backward_data(dts, cats)

        node = BrinsonModel.__new__(BrinsonModel)
        node._QSArgs = type('Args', (), {'Name': '', 'SectionIDs': cats})()
        node.Deps = [type('Dep', (), {
            'Args': type('Args', (), {'SectionIDs': cats})(),
            '_QSArgs': type('Args', (), {'CalcDTRuler': None})(),
        })()]
        Output = node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

        report_node = self._make_report_node()
        Result = report_node.backward_compute(path=[], bwd_data_list=[Output], context=None)

        for cat in cats:
            self.assertIn(cat, Result["Report"], f"报告中缺少类别: {cat}")


if __name__ == "__main__":
    unittest.main()
