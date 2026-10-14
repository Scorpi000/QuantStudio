# -*- coding: utf-8 -*-
"""验证时序及面板算子的窗口范围、自身迭代和缓存接续。"""
import datetime as dt
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.FactorOperation import OperandSpec, makeFactorOperator


def window_mean(f, idt, iid, x, args):
    """分别对各描述子的时间窗口求均值，再相加。"""
    return sum(np.mean(values, axis=0) for values in x)


def iterative_window_mean(f, idt, iid, x, args):
    """用自身历史均值与输入窗口均值递推，排除当前时点占位。"""
    return np.mean(x[0][:-1], axis=0) + window_mean(f, idt, iid, x[1:], args)


class TestFactorOperationLookBackMode(unittest.TestCase):
    """以独立的 pandas 参考计算校验窗口边界和接续结果。"""

    def setUp(self):
        self.ids = ["000001.SZ", "000002.SZ", "000003.SZ"]
        self.dts = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(23)]
        values = np.arange(69, dtype=float).reshape(23, 3)
        self.data = pd.DataFrame((values + 1) ** 2 / 100, index=self.dts, columns=self.ids)
        self.other = pd.DataFrame(np.cos(values / 7), index=self.dts, columns=self.ids)
        self.initial = pd.DataFrame(values / 10 + 1, index=self.dts, columns=self.ids)
        self.factor = DataFactor(data=self.data, args={"Name": "open"})
        self.other_factor = DataFactor(data=self.other, args={"Name": "close"})
        self.initial_factor = DataFactor(data=self.initial, args={"Name": "initial"})

    def modes(self):
        """覆盖时序算子的两种 ID 模式及面板算子。"""
        yield "Time", {"IDMode": "单ID"}
        yield "Time", {"IDMode": "多ID"}
        yield "Panel", {}

    def make_factor(self, operator_type, modes, look_back, start_dt, iterative=False):
        """声明每个描述子的回溯范围，并绑定输入因子。"""
        operands = [self.factor, self.other_factor]
        if iterative:
            operands.insert(0, self.initial_factor)
        operator = makeFactorOperator(
            iterative_window_mean if iterative else window_mean, operator_type,
            args={
                "DTMode": "单时点", **modes, "LookBack": look_back,
                "StartDT": start_dt, "iInitFactor": 0 if iterative else -1,
            },
            operand_info=tuple(OperandSpec() for _ in operands),
        )
        return operator(*operands, factor_args={"Name": "WindowFactor"})

    def expected_windows(self, look_back, start_dt):
        """根据声明的起始位置计算每个输入的窗口均值。"""
        result = pd.DataFrame(0.0, index=self.dts, columns=self.ids)
        for data, count, start in zip((self.data, self.other), look_back, start_dt):
            if start is None:
                means = data.rolling(count + 1).mean()
            else:
                begin = self.dts.index(start) - count
                means = data.iloc[begin:].expanding().mean().reindex(self.dts)
            result += means
        return result

    def expected_iteration(self, expanding, start=6):
        """从初值开始递推，扩张模式使用全部已计算的自身历史。"""
        inputs = self.expected_windows([2, 2], [self.dts[3], None])
        begin = 4 if expanding else start
        history = [self.initial.iloc[begin - 1].to_numpy()]
        result = pd.DataFrame(np.nan, index=self.dts, columns=self.ids)
        for i in range(begin, len(self.dts)):
            previous = np.mean(history, axis=0) if expanding else history[-1]
            value = previous + inputs.iloc[i].to_numpy()
            result.iloc[i] = value
            history.append(value)
        return result

    def assert_data_equal(self, actual, expected):
        """比较数值、标签和缺失值位置，允许迭代暂存表保留 object 类型。"""
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)

    def check_windows(self, look_back, start_dt):
        """同一窗口声明在各类算子中应产生相同的数值结果。"""
        expected = self.expected_windows(look_back, start_dt).iloc[6:]
        for operator_type, modes in self.modes():
            with self.subTest(operator_type=operator_type, **modes):
                factor = self.make_factor(operator_type, modes, look_back, start_dt)
                actual = factor.readData(ids=self.ids, dts=self.dts[6:], dt_ruler=self.dts)
                self.assert_data_equal(actual, expected)

    def test_rolling_windows(self):
        """滚动窗口长度固定，各描述子可使用不同的回溯期数。"""
        self.check_windows([2, 3], [None, None])

    def test_expanding_windows(self):
        """扩张窗口从 StartDT 减去 LookBack 的时点开始。"""
        self.check_windows([1, 2], [self.dts[4], self.dts[3]])

    def test_mixed_windows(self):
        """同一算子可同时包含扩张窗口和滚动窗口。"""
        self.check_windows([2, 2], [self.dts[3], None])

    def check_iteration(self, expanding):
        """校验两种自身回溯模式及混合输入窗口。"""
        expected = self.expected_iteration(expanding).iloc[6:]
        for operator_type, modes in self.modes():
            with self.subTest(operator_type=operator_type, **modes):
                factor = self.make_factor(
                    operator_type, modes, [1, 2, 2],
                    [self.dts[4] if expanding else None, self.dts[3], None],
                    iterative=True,
                )
                actual = factor.readData(ids=self.ids, dts=self.dts[6:], dt_ruler=self.dts)
                self.assert_data_equal(actual, expected)

    def test_rolling_self_iteration(self):
        """自身滚动窗口从请求开始时点的前一个初值接续。"""
        self.check_iteration(expanding=False)

    def test_expanding_self_iteration(self):
        """自身扩张窗口从固定起点递推后再截取请求范围。"""
        self.check_iteration(expanding=True)

    def test_cached_iteration_continuation(self):
        """分段计算及重复读取与无缓存的一次计算一致。"""
        expected = self.expected_iteration(expanding=True).iloc[6:]
        for operator_type, modes in self.modes():
            with self.subTest(operator_type=operator_type, **modes):
                factor = self.make_factor(
                    operator_type, modes, [1, 2, 2],
                    [self.dts[4], self.dts[3], None], iterative=True,
                )
                with tempfile.TemporaryDirectory() as cache_dir:
                    with FeatherFactorCache(args={
                        "DTRuler": self.dts, "CacheDir": cache_dir, "StartMode": "new",
                    }) as cache:
                        context = FactorContext(DTRuler=self.dts, SectionIDs=self.ids, DataCache=cache, DataVersion="fixture-v1")
                        engine = Engine()
                        first = engine.run([factor], context, fwd_data_list=[
                            FactorLocalContext(DTs=self.dts[6:-3], IDs=self.ids),
                        ])[0]
                        second = engine.run([factor], context, fwd_data_list=[
                            FactorLocalContext(DTs=self.dts[-3:], IDs=self.ids),
                        ])[0]
                        with patch.object(factor.Operator, "calculate", wraps=factor.Operator.calculate) as calculate:
                            repeated = engine.run([factor], context, fwd_data_list=[
                                FactorLocalContext(DTs=self.dts[6:], IDs=self.ids),
                            ])[0]
                            calculate.assert_not_called()
                        self.assert_data_equal(first, expected.iloc[:-3])
                        self.assert_data_equal(second, expected.iloc[-3:])
                        self.assert_data_equal(pd.concat([first, second]), expected)
                        self.assert_data_equal(repeated, expected)

    def test_insufficient_history_raises(self):
        """时间尺不足时明确报错，避免静默截短窗口。"""
        for operator_type, modes in self.modes():
            with self.subTest(operator_type=operator_type, **modes):
                factor = self.make_factor(operator_type, modes, [2, 2], [None, None])
                with self.assertRaises(__QS_Error__):
                    factor.readData(ids=self.ids, dts=self.dts[1:], dt_ruler=self.dts)


if __name__ == "__main__":
    unittest.main()
