# -*- coding: utf-8 -*-
"""因子时间轴回溯填充的回归测试。"""
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.FactorUtils import adjustDataDTID


class TestAdjustDataDTID(unittest.TestCase):

    def setUp(self):
        self.ids = ["000001.SZ", "000002.SZ"]
        raw_dts = pd.to_datetime(["2026-08-01", "2026-08-03", "2026-08-06"])
        raw = pd.DataFrame([[10.0, 20.0], [30.0, np.nan], [60.0, 80.0]],
                           index=raw_dts, columns=self.ids)
        # Panel 合并时间轴后保留 object 索引，正是数据库因子使用的路径。
        self.data = Panel({"value": raw})
        self.assertEqual(self.data.major_axis.dtype, np.dtype("O"))
        self.dts = pd.date_range("2026-08-02", periods=8).to_pydatetime().tolist()

    def test_only_lookback_nontarget_with_object_time_axis(self):
        """仅非目标日可用于填充，目标日已有数据及缺失值保持原样。"""
        actual = adjustDataDTID(self.data, 2, ["value"], self.ids, self.dts,
                                only_lookback_nontarget=True).iloc[0]
        expected = pd.DataFrame([[10.0, 20.0], [30.0, np.nan], [np.nan, np.nan],
                                 [np.nan, np.nan], [60.0, 80.0], [np.nan, np.nan],
                                 [np.nan, np.nan], [np.nan, np.nan]],
                                index=pd.Index(self.dts, dtype="O"), columns=self.ids)
        pd.testing.assert_frame_equal(actual, expected)

    def test_only_lookback_dt_with_object_time_axis(self):
        """按源数据时点整体回溯，原始时点的单列缺失值不单独填充。"""
        actual = adjustDataDTID(self.data, 2, ["value"], self.ids, self.dts,
                                only_lookback_dt=True).iloc[0]
        expected = pd.DataFrame([[10.0, 20.0], [30.0, np.nan], [30.0, np.nan],
                                 [30.0, np.nan], [60.0, 80.0], [60.0, 80.0],
                                 [60.0, 80.0], [np.nan, np.nan]],
                                index=pd.Index(self.dts, dtype="O"), columns=self.ids)
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
