"""多组合净值的截面与数值回归，组合名称独立于证券 ID。"""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorInitData, FactorLocalContext
from QuantStudio.BackTest.Strategy.AllocationStrategy import CalcPortfolioNV


class TestPortfolioNV(unittest.TestCase):
    """用确定收益率和多种集合长度验证实际计算。"""

    def setUp(self):
        self.ids = ["A", "B"]
        self.dates = [dt.datetime(2026, 1, 1) + dt.timedelta(days=index) for index in range(7)]
        prices = 100 * np.power([1.01, 1.02], np.arange(7).reshape(-1, 1))
        self.price = DataFactor(data=pd.DataFrame(prices, index=self.dates, columns=self.ids), args={"Name": "价格"})
        self.portfolios = [DataFactor(data=pd.DataFrame(np.tile([weight, 1 - weight], (7, 1)),
            index=self.dates, columns=self.ids), args={"Name": f"P{index}"})
            for index, weight in enumerate(np.linspace(0, 1, 5))]

    def test_multiple_portfolio_counts_keep_stock_sections(self):
        operator = CalcPortfolioNV(descriptor_ids=self.ids, start_dt=self.dates[1])
        for count in (1, 2, 5):
            with self.subTest(count=count):
                factor = operator(*self.portfolios[:count], price=self.price)
                context = FactorContext(DTRuler=self.dates, SectionIDs=self.ids)
                values, = Engine().run([factor], context,
                    fwd_data_list=[FactorLocalContext(DTs=self.dates[1:], IDs=factor.Args.SectionIDs)],
                    init_data_list=[FactorInitData(DTRange=(self.dates[1], self.dates[-1]))])
                weights = np.linspace(0, 1, 5)[:count]
                expected = np.power(1 + weights * .01 + (1 - weights) * .02, np.arange(6).reshape(-1, 1))
                np.testing.assert_allclose(values.to_numpy(dtype=float), expected, rtol=1e-12)
                self.assertEqual(values.columns.tolist(), [f"P{index}" for index in range(count)])
        self.assertEqual(len(operator.Args.DescriptorSection), 4)

    def test_bound_collection_expansion_keeps_price_section(self):
        operator = CalcPortfolioNV(descriptor_ids=self.ids, start_dt=self.dates[1])
        bound = operator.bind(portfolio=tuple(self.portfolios[:2]), price=self.price)
        factor = bound(*self.portfolios[2:])
        context = FactorContext(DTRuler=self.dates, SectionIDs=self.ids)
        values, = Engine().run([factor], context,
            fwd_data_list=[FactorLocalContext(DTs=self.dates[1:], IDs=factor.Args.SectionIDs)],
            init_data_list=[FactorInitData(DTRange=(self.dates[1], self.dates[-1]))])
        expected = np.power(1 + np.linspace(.02, .01, 5), np.arange(6).reshape(-1, 1))
        np.testing.assert_allclose(values.to_numpy(dtype=float), expected, rtol=1e-12)

    def test_copy_keeps_explicit_per_input_settings(self):
        operator = CalcPortfolioNV(descriptor_ids=self.ids, start_dt=self.dates[1])
        factor = operator(*self.portfolios, price=self.price)
        lookback = [3] + [0] * 7
        start_dates = [self.dates[0]] + [self.dates[1]] * 7
        sections = [None] + [self.ids] * 6 + [None]
        configured = factor._Operator.new(args={"LookBack": lookback, "StartDT": start_dates, "DescriptorSection": sections})
        copied = configured.new()
        self.assertEqual(copied.Args.LookBack, lookback)
        self.assertEqual(copied.Args.StartDT, start_dates)
        self.assertEqual(copied.Args.DescriptorSection, sections)


if __name__ == "__main__":
    unittest.main()
