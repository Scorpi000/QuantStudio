# -*- coding: utf-8 -*-
"""验证当前因子算子接口在不同运算模式下的数值结果。"""
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.Factor import DataFactor
from QuantStudio.Factor.FactorOperation import OperandSpec, makeFactorOperator


def divide(f, idt, iid, x, args):
    """除法适配标量和数组输入，分母为零时返回缺失值。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(x[1]) == 0, np.nan, np.asarray(x[0]) / x[1])


def rolling_mean(f, idt, iid, x, args):
    """计算完整窗口的均值，多时点输入按窗口逐次计算。"""
    window = args["window"]
    if f.Operator.Args.DTMode == "单时点":
        return np.mean(x[0], axis=0)
    return np.array([
        np.mean(x[0][i - window + 1:i + 1], axis=0)
        for i in range(window - 1, len(x[0]))
    ])


def exponential_mean(f, idt, iid, x, args):
    """以第一个描述子提供初值，并将计算结果用于后续迭代。"""
    alpha = 2 / (args["window"] + 1)
    if f.Operator.Args.DTMode == "单时点":
        # 自身描述子的末行是当前时点占位，前一行才是上次计算结果。
        return alpha * x[1][-1] + (1 - alpha) * x[0][-2]
    result = x[0].astype(float, copy=True)
    for i, value in enumerate(x[1], start=1):
        result[i] = alpha * value + (1 - alpha) * result[i - 1]
    if f.Operator.Args.OperatorType == "Time" and f.Operator.Args.IDMode == "多ID":
        # 多 ID 时序算子由框架截去初值行，其余模式由算子截去。
        return result
    return result[1:]


def standardize(f, idt, iid, x, args):
    """按截面均值和样本标准差计算标准分数。"""
    return ((x[0] - np.nanmean(x[0], axis=-1, keepdims=True))
            / np.nanstd(x[0], axis=-1, ddof=1, keepdims=True))


def section_difference(f, idt, iid, x, args):
    """计算两个不同截面的均值之差，输出一个宏观 ID。"""
    return (np.nanmean(x[0], axis=-1, keepdims=True)
            - np.nanmean(x[1], axis=-1, keepdims=True))


def panel_exponential_mean(f, idt, iid, x, args):
    """对两个截面均值之差进行自身迭代。"""
    difference = section_difference(f, idt, iid, x[1:], args)
    return exponential_mean(f, idt, iid, [x[0], difference], args)


class TestFactorOperation(unittest.TestCase):
    """使用确定性数据验证四类算子及自身迭代。"""

    @classmethod
    def setUpClass(cls):
        cls.ids = [f"{i:06d}.SZ" for i in range(10)]
        cls.group_ids = [f"Group{i}" for i in range(3)]
        cls.macro_ids = ["000000.HST"]
        cls.dts = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(30)]
        rng = np.random.default_rng(0)
        cls.data = pd.DataFrame(rng.normal(size=(30, 10)), index=cls.dts, columns=cls.ids)
        cls.denominator = pd.DataFrame(rng.normal(size=(30, 10)), index=cls.dts, columns=cls.ids)
        cls.denominator.iloc[0, 0] = 0
        cls.denominator.iloc[8, 3] = 0
        cls.group_data = pd.DataFrame(rng.normal(size=(30, 3)), index=cls.dts, columns=cls.group_ids)
        cls.factor = DataFactor(data=cls.data, args={"Name": "Factor0"})
        cls.denominator_factor = DataFactor(data=cls.denominator, args={"Name": "Factor1"})
        cls.group_factor = DataFactor(data=cls.group_data, args={"Name": "GroupFactor"})

    def assert_factor_equal(self, factor, expected, ids=None, start=0):
        """使用完整时间尺读取结果，并校验数值、标签及缺失值位置。"""
        actual = factor.readData(
            ids=self.ids if ids is None else ids,
            dts=self.dts[start:], dt_ruler=self.dts,
        )
        # 自身迭代的框架暂存表可能保留 object 类型，此处比较数值和标签。
        pd.testing.assert_frame_equal(
            actual, expected.iloc[start:], check_dtype=False, rtol=1e-12, atol=1e-12,
        )

    def make_factor(self, func, operator_type, operands, **args):
        """声明操作数及算子参数，再绑定输入因子。"""
        operator = makeFactorOperator(
            func, operator_type, args=args,
            operand_info=tuple(OperandSpec() for _ in operands),
        )
        return operator(*operands, factor_args={"Name": "TestFactor"})

    def point_modes(self):
        """枚举时点和 ID 的四种组合。"""
        for dt_mode in ("单时点", "多时点"):
            for id_mode in ("单ID", "多ID"):
                yield {"DTMode": dt_mode, "IDMode": id_mode}

    def test_point_operation(self):
        """四种单点运算模式均正确处理除零。"""
        expected = self.data.div(self.denominator).mask(self.denominator == 0)
        for modes in self.point_modes():
            with self.subTest(**modes):
                factor = self.make_factor(divide, "Point", [self.factor, self.denominator_factor], **modes)
                self.assert_factor_equal(factor, expected)

    def test_builtin_division_and_rename(self):
        """内置除法及重命名使用当前公共接口。"""
        factor = rename(self.factor / self.denominator_factor, "Division")
        self.assertEqual(factor.Name, "Division")
        self.assert_factor_equal(factor, self.data.div(self.denominator).mask(self.denominator == 0))

    def test_time_rolling_mean(self):
        """完整时间尺提供历史窗口，覆盖两种窗口长度及四种模式。"""
        for window in (3, 5):
            expected = self.data.rolling(window, min_periods=window).mean()
            for modes in self.point_modes():
                with self.subTest(window=window, **modes):
                    factor = self.make_factor(
                        rolling_mean, "Time", [self.factor], LookBack=[window - 1],
                        ModelArgs={"window": window}, **modes,
                    )
                    self.assert_factor_equal(factor, expected, start=window - 1)
                    self.assert_factor_equal(factor, expected, start=window + 1)

    def test_time_exponential_mean(self):
        """将首日数据作为初值，后续时点按 iInitFactor 自身迭代。"""
        expected = self.data.ewm(span=5, adjust=False).mean()
        for modes in self.point_modes():
            with self.subTest(**modes):
                factor = self.make_factor(
                    exponential_mean, "Time", [self.factor, self.factor],
                    LookBack=[1, 0], iInitFactor=0,
                    StartDT=[self.dts[1], None], ModelArgs={"window": 5}, **modes,
                )
                self.assert_factor_equal(factor, expected, start=1)

    def test_section_standardize(self):
        """单时点和多时点截面标准化均匹配 pandas。"""
        expected = self.data.sub(self.data.mean(axis=1), axis=0).div(self.data.std(axis=1), axis=0)
        for dt_mode in ("单时点", "多时点"):
            with self.subTest(DTMode=dt_mode):
                factor = self.make_factor(standardize, "Section", [self.factor], DTMode=dt_mode)
                self.assert_factor_equal(factor, expected)

    def test_section_difference(self):
        """显式指定描述子截面，输出截面与输入截面不同。"""
        expected = (self.data.mean(axis=1) - self.group_data.mean(axis=1)).to_frame(self.macro_ids[0])
        for dt_mode in ("单时点", "多时点"):
            with self.subTest(DTMode=dt_mode):
                factor = self.make_factor(
                    section_difference, "Section", [self.factor, self.group_factor],
                    DescriptorSection=[self.ids, self.group_ids], DTMode=dt_mode,
                )
                self.assert_factor_equal(factor, expected, ids=self.macro_ids)

    def test_panel_exponential_mean(self):
        """面板运算组合不同截面及自身迭代，覆盖两种时点模式。"""
        difference = self.data.mean(axis=1) - self.group_data.mean(axis=1)
        expected = difference.ewm(span=5, adjust=False).mean().to_frame(self.macro_ids[0])
        initial = DataFactor(data=difference.to_frame(self.macro_ids[0]))
        for dt_mode in ("单时点", "多时点"):
            with self.subTest(DTMode=dt_mode):
                factor = self.make_factor(
                    panel_exponential_mean, "Panel", [initial, self.factor, self.group_factor],
                    DescriptorSection=[self.macro_ids, self.ids, self.group_ids],
                    LookBack=[1, 0, 0], iInitFactor=0,
                    StartDT=[self.dts[1], None, None],
                    ModelArgs={"window": 5}, DTMode=dt_mode,
                )
                self.assert_factor_equal(factor, expected, ids=self.macro_ids, start=1)


if __name__ == "__main__":
    unittest.main()
