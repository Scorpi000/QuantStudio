# -*- coding: utf-8 -*-

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

print("=" * 70)
print("测试 qs_help 函数")
print("=" * 70)

# 测试1：子类方法（无文档，应从父类继承）
print("\n>>> qs_help(Child.greet)")
print(qs_help(Child.greet))

# 测试2：实例方法（bound method）
child = Child()
print("\n>>> qs_help(child.greet)")
qs_help(child.greet)

# 测试3：孙子类（应从 Parent 继承文档）
print("\n>>> qs_help(GrandChild.greet)")
gc = GrandChild()
qs_help(gc.greet)

# 测试4：有文档的方法（正常使用）
print("\n>>> qs_help(Parent.greet)")
qs_help(Parent.greet)

# 测试5：类本身
print("\n>>> qs_help(Child)")
qs_help(Child)

# 测试6：内置函数（无文档的情况）
print("\n>>> qs_help(len)")
qs_help(len)