# coding=utf-8
"""业绩分析工具函数测试。"""
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Tools.PerformanceAnalysisFun import calcBrinsonSignlePeriodExt, calcCarinoAdjustFactor


class TestBrinsonSinglePeriodExt(unittest.TestCase):
    """验证外部收益输入格式与归因恒等式。"""

    def test_selection_uses_benchmark_weight(self):
        """选券收益使用基准权重，并与总超额收益勾稽。"""
        data = pd.DataFrame({
            "p_weight": [0.6, 0.4], "bmk_weight": [0.5, 0.5],
            "p_return": [0.1, 0.02], "bmk_return": [0.04, 0.03],
        }, index=["股票", "债券"])

        result = calcBrinsonSignlePeriodExt(data)

        self.assertAlmostEqual(result.loc["股票", "selection"], 0.03)
        np.testing.assert_allclose(result["portfolio"] - result["benchmark"],
                                   result["allocation"] + result["selection"] + result["interaction"])


class TestBrinsonMultiPeriodAdjustFactor(unittest.TestCase):
    """验证 Carino 调整因子及边界情形。"""

    def test_linked_effects_match_compounded_excess_return(self):
        """调整后的每期超额收益之和等于多期总超额收益。"""
        data = pd.DataFrame({
            "portfolio": [0.10, 0.02, -0.04],
            "benchmark": [0.05, 0.02, -0.01],
        }, index=["一期", "二期", "三期"])

        factor = calcCarinoAdjustFactor(data)
        expected = np.prod(1 + data["portfolio"]) - np.prod(1 + data["benchmark"])

        self.assertEqual(factor.index.tolist(), data.index.tolist())
        self.assertAlmostEqual(((data["portfolio"] - data["benchmark"]) * factor).sum(), expected)
        self.assertTrue(np.isfinite(factor).all())

    def test_equal_returns_use_logarithmic_limit(self):
        """各期组合与基准相同时，差商使用连续极限。"""
        data = pd.DataFrame({"portfolio": [0.10, -0.05], "benchmark": [0.10, -0.05]})

        factor = calcCarinoAdjustFactor(data)
        total = np.prod(1 + data["benchmark"])

        np.testing.assert_allclose(factor, total / (1 + data["benchmark"]))

    def test_equal_compounded_returns_use_logarithmic_limit(self):
        """累计收益相同但单期超额收益不同时仍可完成连接。"""
        data = pd.DataFrame({"portfolio": [0.10, 0.0], "benchmark": [0.0, 0.10]})

        factor = calcCarinoAdjustFactor(data)

        self.assertTrue(np.isfinite(factor).all())
        self.assertAlmostEqual(((data["portfolio"] - data["benchmark"]) * factor).sum(), 0)

    def test_invalid_return_raises(self):
        """无法计算对数收益率时给出明确错误。"""
        data = pd.DataFrame({"portfolio": [-1.0], "benchmark": [0.0]})
        with self.assertRaises(ValueError):
            calcCarinoAdjustFactor(data)


if __name__ == "__main__":
    unittest.main()
