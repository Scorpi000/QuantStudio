# -*- coding: utf-8 -*-
import unittest

from QuantStudio.Tools.Visualization import qs_help


class Parent:
    """父类说明"""

    def greet(self, name: str) -> str:
        """
        打招呼方法。

        Args:
            name: 对方的名字

        Returns:
            问候语字符串

        Examples:
            >>> p = Parent()
            >>> p.greet("Alice")
            'Hello, Alice!'
        """
        return f"Hello, {name}!"

    def farewell(self):
        """说再见"""
        pass


class Child(Parent):
    """子类说明"""

    def greet(self, name: str) -> str:
        # 重写了方法，但没有写文档
        return f"Hi, {name}!"

    def farewell(self):
        # 也没有文档
        return "Bye!"


class GrandChild(Child):
    # 完全没有文档
    def greet(self, name: str) -> str:
        return f"Hey, {name}!"


class TestQsHelp(unittest.TestCase):
    """测试 QuantStudio.Tools.Visualization 中的 qs_help 函数"""

    # ========== 方法级文档继承 ==========

    def test_child_greet_inherits_parent_doc(self):
        """子类方法无文档时，应从父类继承"""
        result = qs_help(Child.greet)
        self.assertIn("打招呼方法", result)
        self.assertIn("name", result)

    def test_bound_method_inherits_parent_doc(self):
        """实例方法（bound method）无文档时，应从父类继承"""
        child = Child()
        result = qs_help(child.greet)
        self.assertIn("打招呼方法", result)

    def test_grandchild_greet_inherits_grandparent_doc(self):
        """孙子类方法无文档时，应从祖父类继承"""
        gc = GrandChild()
        result = qs_help(gc.greet)
        self.assertIn("打招呼方法", result)

    def test_parent_greet_has_own_doc(self):
        """有文档的方法应返回自身文档"""
        result = qs_help(Parent.greet)
        self.assertIn("打招呼方法", result)
        self.assertIn("Hello, Alice!", result)

    # ========== 类级帮助 ==========

    def test_class_help(self):
        """类本身应返回格式化的帮助文档"""
        result = qs_help(Child)
        self.assertIn("子类说明", result)

    # ========== 内置函数 ==========

    def test_builtin_function_no_doc(self):
        """内置函数（C 实现）无文档时应返回提示信息"""
        result = qs_help(len)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    # ========== 返回值格式 ==========

    def test_return_type_is_str(self):
        """qs_help 应始终返回字符串"""
        self.assertIsInstance(qs_help(Parent.greet), str)
        self.assertIsInstance(qs_help(Child), str)
        self.assertIsInstance(qs_help(len), str)

    def test_child_farewell_inherits_parent_doc(self):
        """子类 farewell 无文档时，应从父类继承 '说再见'"""
        result = qs_help(Child.farewell)
        self.assertIn("说再见", result)


if __name__ == "__main__":
    unittest.main()
