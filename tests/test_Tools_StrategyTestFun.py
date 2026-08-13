# -*- coding: utf-8 -*-
import unittest
from typing import Tuple

import numpy as np
import pandas as pd

import QuantStudio.Tools.StrategyTestFun as STF
from QuantStudio.Tools.DataPreprocessingFun import numpy_ffill


def _backtestPortfolioStrategy_v0(portfolio: np.ndarray, price: np.ndarray, fee: float | np.ndarray = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """backtestPortfolioStrategy 的 v0 参考实现，用于交叉验证。"""
    Mask = (~np.all(np.isnan(portfolio), axis=1))
    MaskedReturn, MaskedTurnover = STF.backtestPortfolioStrategyWithMargin(portfolio[Mask], price[Mask], fee=fee)
    if np.all(Mask):
        return np.cumprod(1 + MaskedReturn), MaskedTurnover
    MaskedNV = np.full(shape=Mask.shape, fill_value=np.nan)
    MaskedNV[Mask] = np.cumprod(1 + MaskedReturn)
    MaskedNV = numpy_ffill(MaskedNV)
    MaskedNV[np.isnan(MaskedNV)] = 1
    portfolio = np.r_[np.zeros((1, portfolio.shape[1])), portfolio]
    portfolio = numpy_ffill(portfolio, axis=0, limit=None)
    MaskedPrice = price.copy()
    MaskedPrice[~Mask] = np.nan
    MaskedPrice[0] = price[0]
    MaskedPrice = numpy_ffill(MaskedPrice, axis=0, limit=None)
    MaskedReturn = np.zeros_like(MaskedPrice)
    MaskedReturn[1:] = price[1:] / MaskedPrice[:-1] - 1
    MaskedReturn = np.where(np.isinf(MaskedReturn), np.sign(MaskedReturn), MaskedReturn)
    MaskedReturn[np.isnan(MaskedReturn)] = 0.0
    PortfolioNV = 1 + np.nansum(portfolio[:-1] * MaskedReturn, axis=1)
    PortfolioNV = MaskedNV * PortfolioNV
    Turnover = np.zeros(shape=PortfolioNV.shape)
    Turnover[Mask] = MaskedTurnover
    return PortfolioNV, Turnover


class TestCalcMD(unittest.TestCase):
    """测试 calcMD 最大回撤函数"""

    @classmethod
    def setUpClass(cls):
        cls.p = np.array([0.6, 0.8, 0.9, 0.85, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
                          0.67, 0.66, 0.6, 0.8, 1.2, 1.0, 1.4, 1.5, 1.2, 1.0,
                          0.8, 0.9, 1.0, 0.7, 0.9])

    def test_calcMD(self):
        """calcMD 应返回正确的最大回撤比例"""
        self.assertAlmostEqual(STF.calcMD(self.p), 0.7 / 1.5 - 1)

    def test_calcMaxDrawdownRate(self):
        """calcMaxDrawdownRate 应返回正确的回撤比例、起始和结束位置"""
        self.assertTupleEqual(STF.calcMaxDrawdownRate(self.p), (0.7 / 1.5 - 1, 17, 23))


class TestBacktestPortfolioStrategy(unittest.TestCase):
    """测试投资组合回测函数：v0 参考实现、pandas 版本、numpy 版本的一致性"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        nDT, nID = 1000, 3000
        Portfolio = np.random.rand(nDT, nID)
        Portfolio = Portfolio / np.sum(Portfolio, axis=1, keepdims=True)
        Mask = (np.random.randn(nDT) > 0)
        Portfolio[Mask] = np.nan
        Price = np.cumprod(np.exp(0.03 * np.random.randn(nDT, nID)), axis=0)
        cls.Portfolio = Portfolio
        cls.Price = Price

    def test_numpy_vs_v0_final_nav(self):
        """numpy 版本与 v0 参考实现的最终净值应一致"""
        nv_v0, _ = _backtestPortfolioStrategy_v0(self.Portfolio, self.Price)
        nv_np, _ = STF.backtestPortfolioStrategy(self.Portfolio, self.Price)
        self.assertAlmostEqual(nv_np[-1], nv_v0[-1], places=10)

    def test_pandas_vs_v0_final_nav(self):
        """pandas 版本与 v0 参考实现的最终净值应一致"""
        nv_v0, _ = _backtestPortfolioStrategy_v0(self.Portfolio, self.Price)
        portfolio_df = pd.DataFrame(self.Portfolio).dropna(how="all", axis=0)
        nv_pd = STF.backtestPortfolioStrategy_pd(portfolio_df, pd.DataFrame(self.Price))
        self.assertAlmostEqual(nv_pd.iloc[-1], nv_v0[-1], places=10)

    def test_numpy_turnover_non_negative(self):
        """换手率应全部非负"""
        _, turnover = STF.backtestPortfolioStrategy(self.Portfolio, self.Price)
        self.assertTrue(np.all(turnover >= 0))

    def test_numpy_initial_nav_is_one(self):
        """初始净值应为 1"""
        nv, _ = STF.backtestPortfolioStrategy(self.Portfolio, self.Price)
        self.assertAlmostEqual(nv[0], 1.0)

    # vectorbt 版本对比
    def test_vectorbt_vs_numpy_final_nav(self):
        """vectorbt 版本与 numpy 版本的最终净值应一致"""
        import sys
        sys.path.append(r"D:\Project\vectorbt")
        import vectorbt as vbt
        from vectorbt.portfolio.enums import SizeType
        price_df = pd.DataFrame(self.Price)
        weights_df = pd.DataFrame(self.Portfolio)
        pf = vbt.Portfolio.from_orders(
            price_df,
            size=weights_df,
            size_type=SizeType.TargetPercent,
            cash_sharing=True,
            call_seq='auto'
        )
        total_return = pf.total_return()
        nv_np, _ = STF.backtestPortfolioStrategy(self.Portfolio, self.Price)
        self.assertAlmostEqual(total_return, nv_np[-1] - 1, places=6)


if __name__ == "__main__":
    # unittest.main()
    Suite = unittest.TestSuite()
    Suite.addTest(TestBacktestPortfolioStrategy("test_vectorbt_vs_numpy_final_nav"))
    Runner = unittest.TextTestRunner()
    Runner.run(Suite)