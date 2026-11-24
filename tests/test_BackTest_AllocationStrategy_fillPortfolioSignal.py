# -*- coding: utf-8 -*-
"""测试目标权重信号经账户成交后填充为实际持仓权重。"""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.BackTest.Strategy.AllocationStrategy import fillPortfolioSignal
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache


class TestFillPortfolioSignal(unittest.TestCase):
    def test_weights_drift_without_signal_and_rebalance_on_next_signal(self):
        """无信号时沿用持仓并随价格漂移，再按新目标权重调仓。"""
        dts = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(5)]
        ids = ["A", "B"]
        signal = DataFactor(pd.DataFrame([
            [np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan],
            [np.nan, np.nan], [0.5, 0.5],
        ], index=dts, columns=ids))
        price = DataFactor(pd.DataFrame([
            [10, 10], [10, 10], [20, 10], [30, 10], [30, 10],
        ], index=dts, columns=ids))
        portfolio = fillPortfolioSignal(signal, price)

        with FeatherFactorCache(args={"DTRuler": dts, "StartMode": "new", "CacheDir": None}) as cache:
            with FactorContext(DTRuler=dts, SectionIDs=ids, DataCache=cache) as context:
                with Engine() as engine:
                    result = engine.run(
                        [portfolio], context,
                        fwd_data_list=[FactorLocalContext(DTs=dts[1:], IDs=ids, SectionIDs=ids)],
                    )[0]

        expected = pd.DataFrame([
            [0.5, 0.5], [2 / 3, 1 / 3], [0.75, 0.25], [0.5, 0.5],
        ], index=dts[1:], columns=ids)
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
