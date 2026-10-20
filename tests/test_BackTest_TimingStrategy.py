# -*- coding: utf-8 -*-
"""验证择时组合权重、清仓行为及两种回测方式的一致性。"""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.BackTest.Strategy.api import TimingStrategy
from QuantStudio.BackTest.Strategy.AllocationStrategy import CalcPortfolioNV
from QuantStudio.BackTest.Strategy.Strategy import MakeAccount
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorInitData, FactorLocalContext


class TestTimingStrategy(unittest.TestCase):
    """通过计算引擎验证真实因子与账户输出。"""

    def setUp(self):
        self.ids = ["A", "B", "C", "D"]
        self.dates = [dt.datetime(2026, 1, 1) + dt.timedelta(days=i) for i in range(7)]
        self.signal_data = np.array([
            [1, 1, 1, 1],
            [1, 0, -1, 1],
            [0, 1, 1, 1],
            [-1, -1, np.nan, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ])
        self.signal = self._factor(self.signal_data)
        self.mask = self._factor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [np.nan, 1, 0, 0],
        ])
        self.expected = np.array([
            [1/3, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/4, 1/4, 1/4, 1/4],
            [0, 1, 0, 0],
        ])

    def _factor(self, values):
        """将确定性测试数据包装为因子。

        Args:
            values: 时点与标的组成的二维数据。

        Returns:
            数据因子。
        """
        return DataFactor(pd.DataFrame(values, index=self.dates, columns=self.ids))

    def _run(self, factor, ids=None):
        """使用完整标尺运行因子，预留一个账户初始化时点。

        Args:
            factor: 待运行因子。
            ids: 输出标的，None 使用全部标的。

        Returns:
            引擎计算结果。
        """
        ids = self.ids if ids is None else ids
        with FactorContext(DTRuler=self.dates, SectionIDs=self.ids) as context:
            with Engine() as engine:
                result, = engine.run(
                    [factor], context,
                    fwd_data_list=[FactorLocalContext(DTs=self.dates[1:], IDs=ids)],
                    init_data_list=[FactorInitData(DTRange=(self.dates[1], self.dates[-1]))],
                )
        return result

    def test_equal_budget_and_empty_pool(self):
        """分母包含池内中性和看空标的，空池与缺失信号输出零仓位。"""
        portfolio = TimingStrategy.CalcTimingPortfolio()(self.signal, mask=self.mask)
        np.testing.assert_allclose(self._run(portfolio).values, self.expected)

    def test_default_mask_and_boolean_mask(self):
        """省略标的池时等分整个截面，也接受布尔池因子。"""
        operator = TimingStrategy.CalcTimingPortfolio()
        portfolio = operator(self.signal)
        np.testing.assert_allclose(self._run(portfolio).values,
                                   (self.signal_data[1:] == 1) / 4)
        mask = self._factor(np.ones((7, 4), dtype=bool))
        np.testing.assert_allclose(self._run(operator(self.signal, mask)).values,
                                   self._run(portfolio).values)

    def test_descriptor_section_and_binding(self):
        """输出部分标的时仍按完整池分配，部分绑定保持公开输入语义。"""
        operator = TimingStrategy.CalcTimingPortfolio(descriptor_ids=self.ids)
        portfolio = operator.bind(mask=self.mask)(self.signal)
        np.testing.assert_allclose(self._run(portfolio, ids=["B", "A"]).values,
                                   self.expected[:, [1, 0]])

    def test_trade_dates_keep_nontrading_rows_missing(self):
        """指定交易时点时仅在交易时点给出目标权重。"""
        portfolio = TimingStrategy.CalcTimingPortfolio()(
            self.signal, self.mask, factor_args={"CalcDTRuler": self.dates[1::2]}
        )
        expected = self.expected.copy()
        expected[1::2] = np.nan
        np.testing.assert_allclose(self._run(portfolio).values, expected, equal_nan=True)

    def test_account_liquidates_and_matches_vector_backtest(self):
        """信号转中性、看空或移出池时清仓，并与向量化回测净值一致。"""
        price_data = 100 * np.power([1.1, 1.2, 1.3, 1.4], np.arange(7).reshape(-1, 1))
        price = self._factor(price_data)
        portfolio = TimingStrategy.CalcTimingPortfolio()(self.signal, self.mask)
        account = MakeAccount(init_cash=1200, short_allowed=False, start_dt=self.dates[1])(
            last_price=price, signal=portfolio
        )
        result = self._run(account)
        amount = result.map(lambda value: value[2]).to_numpy(dtype=float)
        cash = result.iloc[:, 0].map(lambda value: value[0]).to_numpy(dtype=float)
        value = cash + amount.sum(axis=1)
        np.testing.assert_allclose(value, [1200, 1240, 1364, 1364, 1364, 1705])
        np.testing.assert_allclose(amount / value[:, None], self.expected, atol=1e-12)
        np.testing.assert_allclose(cash, value * (1 - self.expected.sum(axis=1)), atol=1e-10)
        vector = CalcPortfolioNV(descriptor_ids=self.ids, start_dt=self.dates[1])(
            portfolio, price=price, init_nv=1200, portfolio_name_list=["择时"]
        )
        np.testing.assert_allclose(self._run(vector, ids=["择时"]).iloc[:, 0].to_numpy(dtype=float), value)


if __name__ == "__main__":
    unittest.main()
