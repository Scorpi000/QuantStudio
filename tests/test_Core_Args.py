# -*- coding: utf-8 -*-
"""参数集字典合并运算的回归测试。

使用方法:
    在仓库根目录运行:
    python -m unittest discover -s tests -p "test_Core_Args.py" -v
"""
import unittest

from pydantic import ConfigDict, Field

from QuantStudio.Core import __QS_Args__


class UnionArgs(__QS_Args__):
    """覆盖隐藏、排除及嵌套原始值的参数集。"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    value: int = Field(default=1)
    hidden: int = Field(default=2, repr=False)
    excluded: int = Field(default=3, exclude=True)
    nested: __QS_Args__ = Field(default_factory=__QS_Args__)
    values_list: list[int] = Field(default_factory=lambda: [1, 2])


class TestArgsUnion(unittest.TestCase):
    """合并遵循字典的右侧优先和浅复制语义。"""

    def test_dictionary_on_right(self):
        """参数集与右侧字典合并，右侧覆盖且双方不被修改。"""
        args = UnionArgs(value=1)
        values = {"value": 5, "new": 6}
        result = args | values
        self.assertIsInstance(result, dict)
        self.assertEqual(result["value"], 5)
        self.assertEqual(result["new"], 6)
        self.assertEqual(args.value, 1)
        self.assertEqual(values, {"value": 5, "new": 6})
        self.assertIsNot(result, values)

    def test_dictionary_on_left(self):
        """dict | 参数集使用参数集的值覆盖字典默认值。"""
        args = UnionArgs(value=5)
        values = {"value": 1, "new": 6}
        result = values | args
        self.assertEqual(result["value"], 5)
        self.assertEqual(result["new"], 6)
        self.assertEqual(values, {"value": 1, "new": 6})
        self.assertEqual(args.value, 5)

    def test_two_parameter_models(self):
        """参数集之间合并也返回右侧优先的新字典。"""
        left, right = UnionArgs(value=1), UnionArgs(value=5)
        result = left | right
        self.assertIsInstance(result, dict)
        self.assertEqual(result["value"], 5)
        self.assertIs(result["nested"], right.nested)
        self.assertEqual(left.value, 1)
        self.assertEqual(right.value, 5)

    def test_all_fields_and_raw_values_are_preserved(self):
        """合并包含非展示、非序列化和额外字段，嵌套值不展开。"""
        args = UnionArgs(extra_value=4)
        for result in (args | {}, {} | args):
            with self.subTest(result_keys=list(result)):
                self.assertEqual(result["hidden"], 2)
                self.assertEqual(result["excluded"], 3)
                self.assertEqual(result["extra_value"], 4)
                self.assertIs(result["nested"], args.nested)
                self.assertIs(result["values_list"], args.values_list)
                self.assertNotIn("_Owner", result)
                self.assertNotIn("_Logger", result)

    def test_unsupported_operands(self):
        """不支持的类型交由 Python 运算符协议报 TypeError。"""
        args = UnionArgs()
        self.assertIs(args.__or__(1), NotImplemented)
        self.assertIs(args.__ror__(1), NotImplemented)
        with self.assertRaises(TypeError):
            args | 1
        with self.assertRaises(TypeError):
            1 | args


if __name__ == "__main__":
    unittest.main()
