# -*- coding: utf-8 -*-
import datetime as dt
import unittest

import pandas as pd

import QuantStudio.Tools.DateTimeFun as DTF


class TestTransformDateTime(unittest.TestCase):
    """测试 QuantStudio.Tools.DateTimeFun 中的 transformDateTime 函数"""
    @classmethod
    def setUpClass(cls):
        """生成 2024 年全年日频时点序列和带时间的时点序列"""
        dts = pd.date_range(dt.datetime(2024, 1, 1), dt.datetime(2024, 12, 31), freq="D")
        cls.daily_dts = [d.to_pydatetime() for d in dts]
        dts_with_time = pd.date_range(dt.datetime(2024, 1, 1, 9, 30), dt.datetime(2024, 1, 31, 15, 0), freq="min")
        cls.minute_dts = [d.to_pydatetime() for d in dts_with_time]

    # ========== 月频测试 ==========

    def test_month_target_day_int_postpone(self):
        """月频：取每月 15 日，向后顺延"""
        result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day=15, postpone=True)
        self.assertEqual([r.day for r in result], [15] * 12)

    def test_month_target_day_int_prepone(self):
        """月频：取每月 15 日，向前顺延"""
        result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day=15, postpone=False)
        self.assertEqual([r.day for r in result], [15] * 12)

    def test_month_last(self):
        """月频：取每月最后一天"""
        result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day="last")
        expected_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.assertEqual([r.day for r in result], expected_days)

    def test_month_first(self):
        """月频：取每月第一天"""
        result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day="first")
        self.assertEqual([r.day for r in result], [1] * 12)

    def test_month_exact(self):
        """月频：精确取 15 日"""
        result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day=15, exact=True)
        self.assertEqual([r.day for r in result], [15] * 12)

    def test_month_target_31_prepone(self):
        """月频：取 31 日向前顺延，2 月无 31 日"""
        result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day=31, postpone=False)
        months_with_31 = {1, 3, 5, 7, 8, 10, 12}
        for r in result:
            if r.month in months_with_31:
                self.assertEqual(r.day, 31)
            else:
                self.assertLess(r.day, 31)

    def test_month_multi_n(self):
        """月频：每 2 个月取 15 日"""
        result = DTF.transformDateTime(self.daily_dts, freq="2m", target_day=15, postpone=True)
        self.assertEqual([r.month for r in result], list(range(1, 13, 2)))

    # ========== 对比已有函数 ==========

    def test_compare_getMonthDateTime_over_month_false(self):
        """与 getMonthDateTime 对比（over_period=False）"""
        for td in [1, 15, 28]:
            for pp in [True, False]:
                with self.subTest(target_day=td, postpone=pp):
                    expected = DTF.getMonthDateTime(self.daily_dts, target_day=td, postpone=pp, over_month=False)
                    result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day=td, postpone=pp, over_period=False)
                    self.assertEqual(result, expected)

    def test_compare_getMonthDateTime_over_month_true(self):
        """与 getMonthDateTime 对比（over_period=True）"""
        for td in [1, 15, 28]:
            for pp in [True, False]:
                with self.subTest(target_day=td, postpone=pp):
                    expected = DTF.getMonthDateTime(self.daily_dts, target_day=td, postpone=pp, over_month=True)
                    result = DTF.transformDateTime(self.daily_dts, freq="1m", target_day=td, postpone=pp, over_period=True)
                    self.assertEqual(result, expected)

    def test_compare_getWeekDateTime_over_week_false(self):
        """与 getWeekDateTime 对比（over_period=False）"""
        for tw in [1, 3, 5, 7]:
            for pp in [True, False]:
                with self.subTest(target_weekday=tw, postpone=pp):
                    expected = DTF.getWeekDateTime(self.daily_dts, target_weekday=tw, postpone=pp, over_week=False)
                    result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day=tw, postpone=pp, over_period=False)
                    self.assertEqual(result, expected)

    def test_compare_getWeekDateTime_over_week_true(self):
        """与 getWeekDateTime 对比（over_period=True）"""
        for tw in [1, 3, 5, 7]:
            for pp in [True, False]:
                with self.subTest(target_weekday=tw, postpone=pp):
                    expected = DTF.getWeekDateTime(self.daily_dts, target_weekday=tw, postpone=pp, over_week=True)
                    result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day=tw, postpone=pp, over_period=True)
                    self.assertEqual(result, expected)

    # ========== 周频测试 ==========

    def test_week_monday_postpone(self):
        """周频：取每周一（isoweekday=1）"""
        result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day=1, postpone=True)
        for r in result:
            self.assertEqual(r.isoweekday(), 1)

    def test_week_friday_postpone(self):
        """周频：取每周五（isoweekday=5）"""
        result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day=5, postpone=True)
        for r in result:
            self.assertEqual(r.isoweekday(), 5)

    def test_week_sunday_postpone(self):
        """周频：取每周日（isoweekday=7）"""
        result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day=7, postpone=True)
        for r in result:
            self.assertEqual(r.isoweekday(), 7)

    def test_week_last(self):
        """周频：取每周最后一天（应与 getWeekLastDateTime 一致）"""
        result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day="last")
        expected = DTF.getWeekLastDateTime(self.daily_dts)
        self.assertEqual(result, expected)

    def test_week_first(self):
        """周频：取每周第一天（应与 getWeekFirstDateTime 一致）"""
        result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day="first")
        expected = DTF.getWeekFirstDateTime(self.daily_dts)
        self.assertEqual(result, expected)

    def test_week_multi_n(self):
        """周频：每 2 周取周三"""
        result = DTF.transformDateTime(self.daily_dts, freq="2w", target_day=3, postpone=True)
        for r in result:
            self.assertEqual(r.isoweekday(), 3)
        # 相邻结果间隔恰好 14 天
        for i in range(1, len(result)):
            self.assertEqual((result[i] - result[i - 1]).days, 14)

    def test_week_exact(self):
        """周频：精确取周三"""
        result = DTF.transformDateTime(self.daily_dts, freq="1w", target_day=3, exact=True)
        for r in result:
            self.assertEqual(r.isoweekday(), 3)

    # ========== 季频测试 ==========

    def test_quarter_first(self):
        """季频：取每季度第一天"""
        result = DTF.transformDateTime(self.daily_dts, freq="1q", target_day="first")
        self.assertEqual([r.month for r in result], [1, 4, 7, 10])
        self.assertEqual([r.day for r in result], [1, 1, 1, 1])

    def test_quarter_last(self):
        """季频：取每季度最后一天"""
        result = DTF.transformDateTime(self.daily_dts, freq="1q", target_day="last")
        self.assertEqual([r.month for r in result], [3, 6, 9, 12])

    def test_quarter_mid_day(self):
        """季频：取每季度第 15 日"""
        result = DTF.transformDateTime(self.daily_dts, freq="1q", target_day=15, postpone=True)
        self.assertEqual([r.month for r in result], [1, 4, 7, 10])
        self.assertEqual([r.day for r in result], [15, 15, 15, 15])

    def test_quarter_multi_n(self):
        """季频：每 2 季度取第一天"""
        result = DTF.transformDateTime(self.daily_dts, freq="2q", target_day="first")
        self.assertEqual([r.month for r in result], [1, 7])

    # ========== 年频测试 ==========

    def test_year_first(self):
        """年频：取每年第一天"""
        result = DTF.transformDateTime(self.daily_dts, freq="1y", target_day="first")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], dt.datetime(2024, 1, 1))

    def test_year_last(self):
        """年频：取每年最后一天"""
        result = DTF.transformDateTime(self.daily_dts, freq="1y", target_day="last")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], dt.datetime(2024, 12, 31))

    def test_year_mid_day(self):
        """年频：取每年第 15 天"""
        result = DTF.transformDateTime(self.daily_dts, freq="1y", target_day=15, postpone=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], dt.datetime(2024, 1, 15))

    # ========== 日频测试 ==========

    def test_day_first(self):
        """日频：每 3 天取第一个"""
        result = DTF.transformDateTime(self.daily_dts, freq="3d", target_day="first")
        self.assertEqual(len(result), 122)  # 366 // 3

    def test_day_last(self):
        """日频：每 3 天取最后一个"""
        result = DTF.transformDateTime(self.daily_dts, freq="3d", target_day="last")
        self.assertEqual(len(result), 122)

    def test_day_index(self):
        """日频：每 3 天取第 2 个"""
        result = DTF.transformDateTime(self.daily_dts, freq="3d", target_day=2, postpone=True)
        self.assertEqual(result[0], dt.datetime(2024, 1, 2))
        self.assertEqual(result[1], dt.datetime(2024, 1, 5))

    def test_day_freq_1(self):
        """日频：n=1 返回全部"""
        result = DTF.transformDateTime(self.daily_dts, freq="1d", target_day="last")
        self.assertEqual(len(result), len(self.daily_dts))

    # ========== 边界条件测试 ==========

    def test_empty_list(self):
        """空列表返回空列表"""
        result = DTF.transformDateTime([], freq="1m", target_day=15)
        self.assertEqual(result, [])

    def test_single_element(self):
        """单元素列表"""
        single = [dt.datetime(2024, 5, 15)]
        result = DTF.transformDateTime(single, freq="1m", target_day=15)
        self.assertEqual(result, single)

    def test_over_period_true_cross_boundary(self):
        """跨周期顺延：月末无 31 日时跨到下月"""
        few_dts = [
            dt.datetime(2024, 1, 31), dt.datetime(2024, 2, 1),
            dt.datetime(2024, 2, 2), dt.datetime(2024, 3, 1)
        ]
        result = DTF.transformDateTime(few_dts, freq="1m", target_day=31, postpone=False, over_period=True)
        self.assertEqual(len(result), 3)

    def test_exact_no_match(self):
        """exact=True：目标日不存在时不返回该周期"""
        few_dts = [dt.datetime(2024, 2, 1), dt.datetime(2024, 2, 3), dt.datetime(2024, 2, 5)]
        result = DTF.transformDateTime(few_dts, freq="1m", target_day=15, exact=True)
        self.assertEqual(result, [])

    def test_invalid_freq_unit(self):
        """不支持的频率单位抛出 ValueError"""
        with self.assertRaises(ValueError):
            DTF.transformDateTime(self.daily_dts, freq="1x", target_day=15)

    # ========== 带时间的时点测试 ==========

    def test_minute_dts_month_first(self):
        """带时间序列：取每月第一个时点"""
        result = DTF.transformDateTime(self.minute_dts, freq="1m", target_day="first")
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0], self.minute_dts[0])

    def test_minute_dts_month_last(self):
        """带时间序列：取每月最后一个时点"""
        result = DTF.transformDateTime(self.minute_dts, freq="1m", target_day="last")
        self.assertGreater(len(result), 0)
        self.assertEqual(result[-1], self.minute_dts[-1])


if __name__ == "__main__":
    unittest.main()
