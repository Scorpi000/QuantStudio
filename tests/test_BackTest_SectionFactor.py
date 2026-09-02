# -*- coding: utf-8 -*-
"""测试 BackTest.SectionFactor 模块中的算子和回测节点

测试覆盖:
- CalcIC: IC 算子（Spearman/Pearson 相关系数、多因子、带 mask）
- CalcSectionCorrelation: 截面相关性算子
- CalcFactorTurnover: 因子换手率算子
- CalcFamaMacBethRegression: Fama-MacBeth 回归算子
- IC.backward_compute: IC 统计聚合（均值、标准差、IR、t 统计、移动平均）
- MultiPortfolio.backward_compute: 多组合统计（收益率、波动率、Sharpe、最大回撤）
- makeQuantilePortfolio: 分位数组合工厂函数
"""
import sys
_src = r"D:\HST\QuantStudio"
if _src not in sys.path:
    sys.path.insert(0, _src)
sys.path = [p for p in sys.path if not (p != _src and p.endswith("QuantStudio") and "QuantStudio" in p)]

import unittest
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.BackTest.SectionFactor.IC import CalcIC, IC
from QuantStudio.BackTest.SectionFactor.Correlation import (
    CalcSectionCorrelation, CalcFactorTurnover, SectionCorrelation, FactorTurnover
)
from QuantStudio.BackTest.SectionFactor.ReturnDecomposition import (
    CalcFamaMacBethRegression, FamaMacBethRegression
)
from QuantStudio.BackTest.SectionFactor.QuantilePortfolio import makeQuantilePortfolio, MultiPortfolio


def _run_factor_engine(factor, dt_ruler, section_ids, lookback=0):
    """通过 Engine 执行因子计算并返回结果

    Args:
        factor: 待计算的因子节点
        dt_ruler: 计算时点序列 (不含 lookback 补充时点)
        section_ids: 输出截面 ID
        lookback: 回溯期数, 用于在 dt_ruler 前添加额外时点到 DTRuler
    """
    # 在 dt_ruler 前添加 lookback 个额外时点作为 DTRuler 的前缀
    extra_dts = [dt_ruler[0] - dt.timedelta(days=lookback - i) for i in range(lookback)]
    full_dtruler = extra_dts + list(dt_ruler)
    # init_data.DTRange 从 dt_ruler[0] 开始 (不含额外时点), DTRuler 含额外时点
    # 这样 init_compute 中 StartIdx = lookback, StartIdx - LookBack >= 0
    init_data = FactorInitData(DTRange=(dt_ruler[0], dt_ruler[-1]), SectionIDs=section_ids)
    fwd_data = FactorLocalContext(DTs=dt_ruler, IDs=section_ids)
    with FeatherFactorCache(args={"DTRuler": full_dtruler, "StartMode": "new", "CacheDir": None}) as Cache:
        with FactorContext(PID="0", PIDList=["0"], DTRuler=full_dtruler,
                          SectionIDs=section_ids, DataCache=Cache) as Context:
            with Engine() as ExecEngine:
                Output = ExecEngine.run(
                    [factor], Context,
                    fwd_data_list=[fwd_data],
                    init_data_list=[init_data]
                )
    return Output[0]


class TestCalcIC(unittest.TestCase):
    """测试 CalcIC 算子"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_dt = 30
        cls.n_ids = 10
        # 多加1个时点用于 lookback
        cls.DTs = [dt.datetime(2024, 12, 31) + dt.timedelta(days=i) for i in range(cls.n_dt + 1)]
        cls.IDs = [f"00000{i}.SZ" for i in range(1, cls.n_ids + 1)]
        # 价格数据: 股票0~4有正收益, 股票5~9有负收益
        cls.PriceData = np.ones((cls.n_dt + 1, cls.n_ids)) * 100
        for t in range(1, cls.n_dt + 1):
            cls.PriceData[t, :5] = cls.PriceData[t-1, :5] * 1.01
            cls.PriceData[t, 5:] = cls.PriceData[t-1, 5:] * 0.99
        cls.PriceDF = pd.DataFrame(cls.PriceData, index=cls.DTs, columns=cls.IDs)
        # 用于计算的时点 (不含第0天)
        cls.ComputeDTs = cls.DTs[1:]

    def test_basic_spearman_ic(self):
        """验证 Spearman IC 基本计算: 因子值与收益率的秩相关性"""
        PriceFactor = DataFactor(data=self.PriceDF, args={"Name": "Price"})
        FactorData = pd.DataFrame(
            np.tile([1]*5 + [0]*5, (self.n_dt + 1, 1)),
            index=self.DTs, columns=self.IDs
        )
        TestFactor = DataFactor(data=FactorData, args={"Name": "TestFactor"})

        calc_ic = CalcIC(descriptor_ids=self.IDs, lookback=1, period_lookback=1,
                         corr_method="spearman")
        IC_Factor = calc_ic(TestFactor, price=PriceFactor,
                            factor_name_list=["TestFactor"])

        Rslt = _run_factor_engine(IC_Factor, self.ComputeDTs,
                                  section_ids=["TestFactor"], lookback=1)
        self.assertIsInstance(Rslt, pd.DataFrame)
        # CalcIC 输出为 (IC, Breadth) 元组, 提取 IC 值
        IC_vals = Rslt["TestFactor"].dropna().map(lambda x: x[0] if isinstance(x, tuple) else x)
        self.assertGreater(len(IC_vals), 0)
        self.assertTrue(all(abs(v) <= 1.0 for v in IC_vals))

    def test_pearson_ic(self):
        """验证 Pearson 相关系数方法"""
        PriceFactor = DataFactor(data=self.PriceDF, args={"Name": "Price"})
        FactorData = pd.DataFrame(
            np.random.randn(self.n_dt + 1, self.n_ids),
            index=self.DTs, columns=self.IDs
        )
        TestFactor = DataFactor(data=FactorData, args={"Name": "TestFactor"})

        calc_ic = CalcIC(descriptor_ids=self.IDs, lookback=1, period_lookback=1,
                         corr_method="pearson")
        IC_Factor = calc_ic(TestFactor, price=PriceFactor,
                            factor_name_list=["TestFactor"])

        Rslt = _run_factor_engine(IC_Factor, self.ComputeDTs,
                                  section_ids=["TestFactor"], lookback=1)
        self.assertIsInstance(Rslt, pd.DataFrame)
        IC_vals = Rslt["TestFactor"].dropna().map(lambda x: x[0] if isinstance(x, tuple) else x)
        self.assertGreater(len(IC_vals), 0)
        self.assertTrue(all(abs(v) <= 1.0 for v in IC_vals))

    def test_multiple_factors(self):
        """验证多因子同时计算 IC"""
        PriceFactor = DataFactor(data=self.PriceDF, args={"Name": "Price"})
        F1 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt + 1, self.n_ids),
                                          index=self.DTs, columns=self.IDs),
                        args={"Name": "Factor1"})
        F2 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt + 1, self.n_ids),
                                          index=self.DTs, columns=self.IDs),
                        args={"Name": "Factor2"})

        calc_ic = CalcIC(descriptor_ids=self.IDs, lookback=1, period_lookback=1)
        IC_Factor = calc_ic(F1, F2, price=PriceFactor,
                            factor_name_list=["Factor1", "Factor2"])

        Rslt = _run_factor_engine(IC_Factor, self.ComputeDTs,
                                  section_ids=["Factor1", "Factor2"], lookback=1)
        self.assertIn("Factor1", Rslt.columns)
        self.assertIn("Factor2", Rslt.columns)

    def test_with_mask(self):
        """验证带筛选条件的 IC 计算"""
        PriceFactor = DataFactor(data=self.PriceDF, args={"Name": "Price"})
        FactorData = pd.DataFrame(
            np.random.randn(self.n_dt + 1, self.n_ids),
            index=self.DTs, columns=self.IDs
        )
        TestFactor = DataFactor(data=FactorData, args={"Name": "TestFactor"})
        MaskData = pd.DataFrame(
            np.tile([1]*5 + [0]*5, (self.n_dt + 1, 1)),
            index=self.DTs, columns=self.IDs
        )
        MaskFactor = DataFactor(data=MaskData, args={"Name": "Mask"})

        calc_ic = CalcIC(descriptor_ids=self.IDs, lookback=1, period_lookback=1)
        IC_Factor = calc_ic(TestFactor, price=PriceFactor, mask=MaskFactor,
                            factor_name_list=["TestFactor"])

        Rslt = _run_factor_engine(IC_Factor, self.ComputeDTs,
                                  section_ids=["TestFactor"], lookback=1)
        self.assertIsInstance(Rslt, pd.DataFrame)
        IC_vals = Rslt["TestFactor"].dropna()
        self.assertGreater(len(IC_vals), 0)


class TestCalcSectionCorrelation(unittest.TestCase):
    """测试 CalcSectionCorrelation 算子"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_dt = 20
        cls.n_ids = 10
        cls.DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(cls.n_dt)]
        cls.IDs = [f"00000{i}.SZ" for i in range(1, cls.n_ids + 1)]

    def test_basic_correlation(self):
        """验证两个因子的截面相关性"""
        F1 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt, self.n_ids),
                                          index=self.DTs, columns=self.IDs),
                        args={"Name": "FactorA"})
        F2 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt, self.n_ids),
                                          index=self.DTs, columns=self.IDs),
                        args={"Name": "FactorB"})

        calc_corr = CalcSectionCorrelation(descriptor_ids=self.IDs)
        Corr_Factor = calc_corr(F1, F2, factor_name_list=["FactorA", "FactorB"])

        output_ids = Corr_Factor._QSArgs.SectionIDs
        Rslt = _run_factor_engine(Corr_Factor, self.DTs, section_ids=output_ids)
        self.assertIsInstance(Rslt, pd.DataFrame)
        self.assertIn("FactorA-FactorB", Rslt.columns)
        Corr_vals = Rslt["FactorA-FactorB"].dropna()
        self.assertGreater(len(Corr_vals), 0)
        self.assertTrue((Corr_vals.abs() <= 1.0).all())

    def test_multiple_factors(self):
        """验证多因子两两组合的截面相关性"""
        F1 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt, self.n_ids),
                                          index=self.DTs, columns=self.IDs), args={"Name": "A"})
        F2 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt, self.n_ids),
                                          index=self.DTs, columns=self.IDs), args={"Name": "B"})
        F3 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt, self.n_ids),
                                          index=self.DTs, columns=self.IDs), args={"Name": "C"})

        calc_corr = CalcSectionCorrelation(descriptor_ids=self.IDs)
        Corr_Factor = calc_corr(F1, F2, F3, factor_name_list=["A", "B", "C"])

        output_ids = Corr_Factor._QSArgs.SectionIDs
        Rslt = _run_factor_engine(Corr_Factor, self.DTs, section_ids=output_ids)
        self.assertEqual(len(Rslt.columns), 3)
        self.assertIn("A-B", Rslt.columns)
        self.assertIn("A-C", Rslt.columns)
        self.assertIn("B-C", Rslt.columns)


class TestCalcFactorTurnover(unittest.TestCase):
    """测试 CalcFactorTurnover 算子"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_dt = 30
        cls.n_ids = 10
        cls.DTs = [dt.datetime(2024, 12, 31) + dt.timedelta(days=i) for i in range(cls.n_dt + 1)]
        cls.IDs = [f"00000{i}.SZ" for i in range(1, cls.n_ids + 1)]
        cls.ComputeDTs = cls.DTs[1:]

    def test_basic_turnover(self):
        """验证基本因子换手率计算"""
        FactorData = pd.DataFrame(np.random.randn(self.n_dt + 1, self.n_ids),
                                  index=self.DTs, columns=self.IDs)
        TestFactor = DataFactor(data=FactorData, args={"Name": "TestFactor"})

        calc_turnover = CalcFactorTurnover(descriptor_ids=self.IDs, lookback=1,
                                           period_lookback=1)
        Turnover_Factor = calc_turnover(TestFactor,
                                        factor_name_list=["TestFactor"])

        Rslt = _run_factor_engine(Turnover_Factor, self.ComputeDTs,
                                  section_ids=["TestFactor"], lookback=1)
        self.assertIsInstance(Rslt, pd.DataFrame)
        self.assertIn("TestFactor", Rslt.columns)
        Turnover_vals = Rslt["TestFactor"].dropna()
        self.assertGreater(len(Turnover_vals), 0)
        self.assertTrue((Turnover_vals.abs() <= 1.0).all())

    def test_high_turnover(self):
        """验证随机因子的换手率较低"""
        np.random.seed(42)
        FactorData = pd.DataFrame(np.random.randn(self.n_dt + 1, self.n_ids),
                                  index=self.DTs, columns=self.IDs)
        TestFactor = DataFactor(data=FactorData, args={"Name": "NoisyFactor"})

        calc_turnover = CalcFactorTurnover(descriptor_ids=self.IDs, lookback=1,
                                           period_lookback=1)
        Turnover_Factor = calc_turnover(TestFactor,
                                        factor_name_list=["NoisyFactor"])

        Rslt = _run_factor_engine(Turnover_Factor, self.ComputeDTs,
                                  section_ids=["NoisyFactor"], lookback=1)
        Turnover_vals = Rslt["NoisyFactor"].dropna()
        if len(Turnover_vals) > 0:
            self.assertLess(abs(Turnover_vals.mean()), 0.8)


class TestCalcFamaMacBethRegression(unittest.TestCase):
    """测试 CalcFamaMacBethRegression 算子"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_dt = 30
        cls.n_ids = 10
        cls.DTs = [dt.datetime(2024, 12, 31) + dt.timedelta(days=i) for i in range(cls.n_dt + 1)]
        cls.IDs = [f"00000{i}.SZ" for i in range(1, cls.n_ids + 1)]
        cls.PriceData = np.ones((cls.n_dt + 1, cls.n_ids)) * 100
        for t in range(1, cls.n_dt + 1):
            cls.PriceData[t] = cls.PriceData[t-1] * (1 + np.random.randn(cls.n_ids) * 0.02)
        cls.PriceDF = pd.DataFrame(cls.PriceData, index=cls.DTs, columns=cls.IDs)
        cls.ComputeDTs = cls.DTs[1:]

    def test_basic_regression(self):
        """验证基本 Fama-MacBeth 回归计算"""
        PriceFactor = DataFactor(data=self.PriceDF, args={"Name": "Price"})
        FactorData = pd.DataFrame(np.random.randn(self.n_dt + 1, self.n_ids),
                                  index=self.DTs, columns=self.IDs)
        TestFactor = DataFactor(data=FactorData, args={"Name": "TestFactor"})

        calc_fmr = CalcFamaMacBethRegression(descriptor_ids=self.IDs, lookback=1,
                                              period_lookback=1)
        FMR_Factor = calc_fmr(TestFactor, price=PriceFactor,
                              factor_name_list=["TestFactor"])

        Rslt = _run_factor_engine(FMR_Factor, self.ComputeDTs,
                                  section_ids=["TestFactor"], lookback=1)
        self.assertIsInstance(Rslt, pd.DataFrame)
        self.assertIn("TestFactor", Rslt.columns)

    def test_multiple_factors_regression(self):
        """验证多因子 Fama-MacBeth 回归"""
        PriceFactor = DataFactor(data=self.PriceDF, args={"Name": "Price"})
        F1 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt + 1, self.n_ids),
                                          index=self.DTs, columns=self.IDs),
                        args={"Name": "Factor1"})
        F2 = DataFactor(data=pd.DataFrame(np.random.randn(self.n_dt + 1, self.n_ids),
                                          index=self.DTs, columns=self.IDs),
                        args={"Name": "Factor2"})

        calc_fmr = CalcFamaMacBethRegression(descriptor_ids=self.IDs, lookback=1,
                                              period_lookback=1)
        FMR_Factor = calc_fmr(F1, F2, price=PriceFactor,
                              factor_name_list=["Factor1", "Factor2"])

        Rslt = _run_factor_engine(FMR_Factor, self.ComputeDTs,
                                  section_ids=["Factor1", "Factor2"], lookback=1)
        self.assertIn("Factor1", Rslt.columns)
        self.assertIn("Factor2", Rslt.columns)


class TestICNode(unittest.TestCase):
    """测试 IC 回测节点的 backward_compute 方法"""

    def _make_ic_result(self, ic_values, breadth_values):
        """构造 IC 算子输出格式的 DataFrame"""
        n_dt, n_factors = ic_values.shape
        data = np.empty((n_dt, n_factors), dtype=object)
        for i in range(n_dt):
            for j in range(n_factors):
                data[i, j] = (ic_values[i, j], breadth_values[i, j])
        return pd.DataFrame(data, columns=[f"F{i}" for i in range(n_factors)])

    def _make_ic_node(self, factor_name_list=None, rolling_avg_period=3):
        """创建 IC 节点实例 (绕过 __init__)"""
        ic_node = IC.__new__(IC)
        ic_node._QSArgs = type('Args', (), {
            'FactorNameList': factor_name_list,
            'RollingAvgPeriod': rolling_avg_period,
            'SectionIDs': None,
        })()
        ic_node.Deps = [type('Dep', (), {
            'Args': type('Args', (), {'SectionIDs': None})()
        })()]
        return ic_node

    def test_backward_compute_basic(self):
        """验证 IC 统计聚合: 均值、标准差、IR、t 统计量"""
        ic_values = np.array([
            [0.8, 0.5],
            [0.6, 0.3],
            [0.4, 0.7],
            [0.5, 0.2],
        ])
        breadth_values = np.full_like(ic_values, 10.0)
        BwdData = self._make_ic_result(ic_values, breadth_values)

        ic_node = self._make_ic_node(rolling_avg_period=3)
        Output = ic_node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)

        self.assertIn("IC", Output)
        self.assertIn("统计数据", Output)
        self.assertIn("IC的移动平均", Output)
        self.assertIn("截面宽度", Output)

        IC_df = Output["IC"]
        self.assertEqual(IC_df.shape, (4, 2))
        np.testing.assert_array_almost_equal(IC_df["F0"].values, [0.8, 0.6, 0.4, 0.5])
        np.testing.assert_array_almost_equal(IC_df["F1"].values, [0.5, 0.3, 0.7, 0.2])

        Stats = Output["统计数据"]
        self.assertAlmostEqual(Stats.loc["F0", "平均值"], 0.575, places=10)
        self.assertAlmostEqual(Stats.loc["F1", "平均值"], (0.5+0.3+0.7+0.2)/4, places=10)
        self.assertAlmostEqual(Stats.loc["F0", "标准差"], np.std([0.8, 0.6, 0.4, 0.5], ddof=1), places=10)
        self.assertAlmostEqual(Stats.loc["F0", "IC_IR"],
                               Stats.loc["F0", "平均值"] / Stats.loc["F0", "标准差"], places=10)
        self.assertEqual(Stats.loc["F0", "有效期数"], 4.0)
        self.assertAlmostEqual(Stats.loc["F0", "t统计量"],
                               2.0 * Stats.loc["F0", "IC_IR"], places=10)

    def test_rolling_average(self):
        """验证 IC 移动平均计算"""
        ic_values = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        breadth_values = np.full_like(ic_values, 10.0)
        BwdData = self._make_ic_result(ic_values, breadth_values)

        ic_node = self._make_ic_node(rolling_avg_period=2)
        Output = ic_node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)
        MA = Output["IC的移动平均"]

        self.assertTrue(pd.isna(MA.iloc[0, 0]))
        self.assertAlmostEqual(MA.iloc[1, 0], 2.0, places=10)
        self.assertAlmostEqual(MA.iloc[2, 0], 4.0, places=10)

    def test_with_factor_name_list(self):
        """验证自定义因子名称列表"""
        ic_values = np.array([[0.5, 0.3]])
        breadth_values = np.array([[10.0, 10.0]])
        BwdData = self._make_ic_result(ic_values, breadth_values)

        ic_node = self._make_ic_node(factor_name_list=["Alpha", "Beta"], rolling_avg_period=1)
        Output = ic_node.backward_compute(path=[], bwd_data_list=[BwdData], context=None)
        self.assertIn("Alpha", Output["IC"].columns)
        self.assertIn("Beta", Output["IC"].columns)


class TestMultiPortfolioNode(unittest.TestCase):
    """测试 MultiPortfolio 回测节点的 backward_compute 方法"""

    def _make_mp_node(self, ls_pairs=None, rebalance_dts=None):
        """创建 MultiPortfolio 节点实例 (绕过 __init__)"""
        mp_node = MultiPortfolio.__new__(MultiPortfolio)
        mp_node._QSArgs = type('Args', (), {
            'LSPairs': ls_pairs or [],
            'RebalanceDTs': rebalance_dts,
            'PortfolioSection': None,
            'BmkSection': None,
        })()
        mp_node._NV = type('NV', (), {
            '_QSArgs': type('Args', (), {'ModelArgs': {}, 'SectionIDs': None})()
        })()
        mp_node._BmkNV = None
        return mp_node

    def test_backward_compute_stats(self):
        """验证净值统计: 收益率、波动率、Sharpe、最大回撤"""
        DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(4)]

        PortfolioNV = pd.DataFrame({
            "P0": [1.0, 1.1, 1.05, 1.15],
            "P1": [1.0, 0.95, 1.0, 1.08],
        }, index=DTs)

        mp_node = self._make_mp_node()
        Output = mp_node.backward_compute(
            path=[], bwd_data_list=[PortfolioNV], context=None
        )

        self.assertIn("净值", Output)
        self.assertIn("收益率", Output)
        self.assertIn("统计数据", Output)
        self.assertIn("超额收益率", Output)
        self.assertIn("超额净值", Output)
        self.assertIn("基准", Output["净值"].columns)

        Stats = Output["统计数据"]
        nDays = (DTs[-1] - DTs[0]).days
        nYear = nDays / 365
        nDT = len(DTs) - 1

        self.assertAlmostEqual(Stats.loc["P0", "总收益率"], 0.15, places=10)
        expected_annual = (1 + 0.15) ** (1 / nYear) - 1
        self.assertAlmostEqual(Stats.loc["P0", "年化收益率"], expected_annual, places=5)
        p0_returns = PortfolioNV["P0"].pct_change().dropna()
        expected_vol = p0_returns.std() * np.sqrt(nDT / nYear)
        self.assertAlmostEqual(Stats.loc["P0", "波动率"], expected_vol, places=5)
        if expected_vol != 0:
            self.assertAlmostEqual(Stats.loc["P0", "Sharpe比率"],
                                   expected_annual / expected_vol, places=5)
        self.assertGreaterEqual(Stats.loc["P0", "最大回撤率"], 0)
        self.assertGreaterEqual(Stats.loc["P1", "最大回撤率"], 0)

    def test_excess_return(self):
        """验证超额收益率和信息比率"""
        DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(5)]

        PortfolioNV = pd.DataFrame({
            "P0": [1.0, 1.02, 1.05, 1.08, 1.12],
        }, index=DTs)

        mp_node = self._make_mp_node()
        Output = mp_node.backward_compute(
            path=[], bwd_data_list=[PortfolioNV], context=None
        )

        Stats = Output["统计数据"]
        self.assertIn("超额收益率", Stats.columns)
        self.assertIn("年化超额收益率", Stats.columns)
        self.assertIn("信息比率", Stats.columns)
        self.assertIn("胜率", Stats.columns)
        self.assertGreaterEqual(Stats.loc["P0", "胜率"], 0)
        self.assertLessEqual(Stats.loc["P0", "胜率"], 1.0)


class TestMakeQuantilePortfolio(unittest.TestCase):
    """测试 makeQuantilePortfolio 工厂函数"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_dt = 10
        cls.n_ids = 10
        cls.DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(cls.n_dt)]
        cls.IDs = [f"00000{i}.SZ" for i in range(1, cls.n_ids + 1)]

    def test_basic_quantile(self):
        """验证基本分位数组合创建"""
        FactorData = pd.DataFrame(
            np.arange(self.n_ids, dtype=float).reshape(1, -1).repeat(self.n_dt, axis=0),
            index=self.DTs, columns=self.IDs
        )
        Factor = DataFactor(data=FactorData, args={"Name": "RankFactor"})

        Portfolios = makeQuantilePortfolio(
            factor=Factor, descriptor_ids=self.IDs, group_num=5
        )

        self.assertEqual(len(Portfolios), 5)
        from QuantStudio.Factor.FactorOperation import SectionOperation
        for p in Portfolios:
            self.assertIsInstance(p, SectionOperation)

    def test_ascending(self):
        """验证升序分组"""
        FactorData = pd.DataFrame(
            np.arange(self.n_ids, dtype=float).reshape(1, -1).repeat(self.n_dt, axis=0),
            index=self.DTs, columns=self.IDs
        )
        Factor = DataFactor(data=FactorData, args={"Name": "RankFactor"})

        Portfolios = makeQuantilePortfolio(
            factor=Factor, descriptor_ids=self.IDs,
            group_num=2, ascending=True
        )
        self.assertEqual(len(Portfolios), 2)

    def test_custom_group_num(self):
        """验证自定义分组数"""
        FactorData = pd.DataFrame(
            np.random.randn(self.n_dt, self.n_ids),
            index=self.DTs, columns=self.IDs
        )
        Factor = DataFactor(data=FactorData, args={"Name": "RankFactor"})

        Portfolios = makeQuantilePortfolio(
            factor=Factor, descriptor_ids=self.IDs, group_num=3
        )
        self.assertEqual(len(Portfolios), 3)


if __name__ == "__main__":
    unittest.main()
