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


class TestLookbackDateTime(unittest.TestCase):
    """测试 QuantStudio.Tools.DateTimeFun 中的 lookbackDateTime 函数"""

    # ========== 预定义 lookback ==========

    def test_today(self):
        """today: 返回同一天，保留时间"""
        idt = dt.datetime(2025, 3, 5, 14, 30)
        self.assertEqual(DTF.lookbackDateTime(idt, "today"), idt)

    def test_yesterday(self):
        """yesterday: 返回前一天"""
        idt = dt.datetime(2025, 3, 5, 9, 0)
        self.assertEqual(DTF.lookbackDateTime(idt, "yesterday"), dt.datetime(2025, 3, 4, 9, 0))

    def test_yesterday_cross_month(self):
        """yesterday: 跨月边界"""
        idt = dt.datetime(2025, 3, 1)
        self.assertEqual(DTF.lookbackDateTime(idt, "yesterday"), dt.datetime(2025, 2, 28))

    def test_last_friday_on_friday(self):
        """last_friday: idt 是周五时返回上一个周五"""
        # 2025-08-15 是周五
        idt = dt.datetime(2025, 8, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "last_friday"), dt.datetime(2025, 8, 8))

    def test_last_friday_on_monday(self):
        """last_friday: idt 是周一"""
        # 2025-08-18 是周一
        idt = dt.datetime(2025, 8, 18)
        self.assertEqual(DTF.lookbackDateTime(idt, "last_friday"), dt.datetime(2025, 8, 15))

    def test_last_friday_on_sunday(self):
        """last_friday: idt 是周日"""
        # 2025-08-17 是周日
        idt = dt.datetime(2025, 8, 17)
        self.assertEqual(DTF.lookbackDateTime(idt, "last_friday"), dt.datetime(2025, 8, 15))

    def test_last_friday_preserves_time(self):
        """last_friday: 保留时间部分"""
        idt = dt.datetime(2025, 8, 18, 10, 30)
        self.assertEqual(DTF.lookbackDateTime(idt, "last_friday"), dt.datetime(2025, 8, 15, 10, 30))

    def test_last_month_end(self):
        """last_month_end: 返回上月最后一天"""
        idt = dt.datetime(2025, 3, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "last_month_end"), dt.datetime(2025, 2, 28))

    def test_last_month_end_jan(self):
        """last_month_end: 1月返回去年12月31日"""
        idt = dt.datetime(2025, 1, 10)
        self.assertEqual(DTF.lookbackDateTime(idt, "last_month_end"), dt.datetime(2024, 12, 31))

    def test_last_month_end_leap_feb(self):
        """last_month_end: 闰年3月返回2月29日"""
        idt = dt.datetime(2024, 3, 1)
        self.assertEqual(DTF.lookbackDateTime(idt, "last_month_end"), dt.datetime(2024, 2, 29))

    # ========== 天单位 ==========

    def test_day_1d(self):
        """1d: 回退1天"""
        idt = dt.datetime(2025, 3, 5, 9, 0)
        self.assertEqual(DTF.lookbackDateTime(idt, "1d"), dt.datetime(2025, 3, 4, 9, 0))

    def test_day_2d(self):
        """2d: 回退2天"""
        idt = dt.datetime(2025, 3, 5)
        self.assertEqual(DTF.lookbackDateTime(idt, "2d"), dt.datetime(2025, 3, 3))

    def test_day_cross_month(self):
        """天: 跨月回退"""
        idt = dt.datetime(2025, 3, 1)
        self.assertEqual(DTF.lookbackDateTime(idt, "1d"), dt.datetime(2025, 2, 28))

    def test_day_ignores_target_day(self):
        """天: 忽略 target_day 参数"""
        idt = dt.datetime(2025, 3, 10)
        r1 = DTF.lookbackDateTime(idt, "3d", target_day="first")
        r2 = DTF.lookbackDateTime(idt, "3d", target_day="last")
        r3 = DTF.lookbackDateTime(idt, "3d")
        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(r1, dt.datetime(2025, 3, 7))

    # ========== 周单位 ==========

    def test_week_first(self):
        """周: first 取周一"""
        # 2025-08-18 是周一，回退1周到 2025-08-11（周一）
        idt = dt.datetime(2025, 8, 18)
        result = DTF.lookbackDateTime(idt, "1w", target_day="first")
        self.assertEqual(result, dt.datetime(2025, 8, 11))
        self.assertEqual(result.isoweekday(), 1)

    def test_week_last(self):
        """周: last 取周日"""
        # 2025-08-18 是周一，回退1周到 2025-08-11 那周的周日 = 2025-08-17
        idt = dt.datetime(2025, 8, 18)
        result = DTF.lookbackDateTime(idt, "1w", target_day="last")
        self.assertEqual(result, dt.datetime(2025, 8, 17))
        self.assertEqual(result.isoweekday(), 7)

    def test_week_exact(self):
        """周: exact 保持同星期几"""
        # 2025-08-18 是周一，回退1周 = 2025-08-11（也是周一）
        idt = dt.datetime(2025, 8, 18, 14, 0)
        result = DTF.lookbackDateTime(idt, "1w", target_day="exact")
        self.assertEqual(result, dt.datetime(2025, 8, 11, 14, 0))

    def test_week_multi(self):
        """2w: 回退2周"""
        idt = dt.datetime(2025, 8, 18)
        result = DTF.lookbackDateTime(idt, "2w", target_day="first")
        self.assertEqual(result, dt.datetime(2025, 8, 4))

    # ========== 月单位 ==========

    def test_month_first(self):
        """月: first 取月初"""
        idt = dt.datetime(2025, 3, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", "first"), dt.datetime(2025, 2, 1))

    def test_month_last(self):
        """月: last 取月末"""
        idt = dt.datetime(2025, 3, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", "last"), dt.datetime(2025, 2, 28))

    def test_month_exact(self):
        """月: exact 保持同日"""
        idt = dt.datetime(2025, 3, 10, 9, 30)
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", "exact"), dt.datetime(2025, 2, 10, 9, 30))

    def test_month_exact_overflow(self):
        """月: exact 目标日不存在时回退到月末（3/31 -> 2/28）"""
        idt = dt.datetime(2025, 3, 31)
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", "exact"), dt.datetime(2025, 2, 28))

    def test_month_int_target_day(self):
        """月: int target_day 取指定日"""
        idt = dt.datetime(2025, 3, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", 20), dt.datetime(2025, 2, 20))

    def test_month_int_target_day_overflow(self):
        """月: int target_day 超出时回退到月末"""
        idt = dt.datetime(2025, 3, 15)
        # 2月无30日，回退到28日
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", 30), dt.datetime(2025, 2, 28))

    def test_month_multi(self):
        """月: 回退多月"""
        idt = dt.datetime(2025, 3, 10)
        self.assertEqual(DTF.lookbackDateTime(idt, "2m", "first"), dt.datetime(2025, 1, 1))

    def test_month_cross_year(self):
        """月: 跨年回退"""
        idt = dt.datetime(2025, 1, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", "last"), dt.datetime(2024, 12, 31))

    def test_month_leap_year(self):
        """月: 闰年2月"""
        idt = dt.datetime(2024, 3, 31)
        self.assertEqual(DTF.lookbackDateTime(idt, "1m", "last"), dt.datetime(2024, 2, 29))

    # ========== 季单位 ==========

    def test_quarter_first(self):
        """季: first 取季初"""
        idt = dt.datetime(2025, 5, 15)
        # 回退1季 = 2月，属于Q1，季初 = 1月1日
        result = DTF.lookbackDateTime(idt, "1q", "first")
        self.assertEqual(result, dt.datetime(2025, 1, 1))

    def test_quarter_last(self):
        """季: last 取季末"""
        idt = dt.datetime(2025, 5, 15)
        # 回退1季 = 2月，季末 = 3月31日（Q1结束月=3）
        result = DTF.lookbackDateTime(idt, "1q", "last")
        self.assertEqual(result, dt.datetime(2025, 3, 31))

    def test_quarter_exact(self):
        """季: exact 保持同日"""
        idt = dt.datetime(2025, 5, 10)
        result = DTF.lookbackDateTime(idt, "1q", "exact")
        self.assertEqual(result, dt.datetime(2025, 2, 10))

    def test_quarter_exact_overflow(self):
        """季: exact 目标日不存在时回退到月末"""
        idt = dt.datetime(2025, 8, 31)
        # 回退1季 = 5月，5月31日存在
        result = DTF.lookbackDateTime(idt, "1q", "exact")
        self.assertEqual(result, dt.datetime(2025, 5, 31))

    def test_quarter_int_target_day(self):
        """季: int target_day"""
        idt = dt.datetime(2025, 8, 15)
        # 回退1季 = 5月，取15日
        result = DTF.lookbackDateTime(idt, "1q", 15)
        self.assertEqual(result, dt.datetime(2025, 5, 15))

    # ========== 年单位 ==========

    def test_year_first(self):
        """年: first 取1月1日"""
        idt = dt.datetime(2025, 6, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "1y", "first"), dt.datetime(2024, 1, 1))

    def test_year_last(self):
        """年: last 取12月31日"""
        idt = dt.datetime(2025, 6, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "1y", "last"), dt.datetime(2024, 12, 31))

    def test_year_exact(self):
        """年: exact 保持同月日"""
        idt = dt.datetime(2025, 3, 10)
        self.assertEqual(DTF.lookbackDateTime(idt, "1y", "exact"), dt.datetime(2024, 3, 10))

    def test_year_exact_leap_day(self):
        """年: exact 2/29 回退到非闰年时取2/28"""
        idt = dt.datetime(2024, 2, 29)
        self.assertEqual(DTF.lookbackDateTime(idt, "1y", "exact"), dt.datetime(2023, 2, 28))

    def test_year_int_target_day(self):
        """年: int target_day"""
        idt = dt.datetime(2025, 6, 15)
        self.assertEqual(DTF.lookbackDateTime(idt, "1y", 20), dt.datetime(2024, 6, 20))

    # ========== 时间保留 ==========

    def test_time_preserved_month(self):
        """月: 时间部分保留"""
        idt = dt.datetime(2025, 3, 10, 14, 30, 45)
        result = DTF.lookbackDateTime(idt, "1m", "exact")
        self.assertEqual(result.hour, 14)
        self.assertEqual(result.minute, 30)
        self.assertEqual(result.second, 45)

    def test_time_preserved_day(self):
        """天: 时间部分保留"""
        idt = dt.datetime(2025, 3, 10, 9, 0, 0)
        result = DTF.lookbackDateTime(idt, "1d")
        self.assertEqual(result, dt.datetime(2025, 3, 9, 9, 0, 0))

    # ========== 错误处理 ==========

    def test_invalid_lookback(self):
        """无效 lookback 抛出 ValueError"""
        with self.assertRaises(ValueError):
            DTF.lookbackDateTime(dt.datetime(2025, 1, 1), "abc")

    def test_invalid_unit(self):
        """不支持的单位抛出 ValueError"""
        with self.assertRaises(ValueError):
            DTF.lookbackDateTime(dt.datetime(2025, 1, 1), "1x")

    def test_invalid_target_day_for_week(self):
        """周: 无效的 target_day 抛出 ValueError"""
        with self.assertRaises(ValueError):
            DTF.lookbackDateTime(dt.datetime(2025, 1, 1), "1w", target_day="invalid")


if __name__ == "__main__":
    unittest.main()
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestLookbackDateTime("test_month_last"))
    # Runner = unittest.TextTestRunner()
    # Runner.run(Suite)