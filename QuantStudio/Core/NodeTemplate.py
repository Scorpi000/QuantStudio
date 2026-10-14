# -*- coding: utf-8 -*-
"""允许部分配置的节点构造模板，不执行节点初始化。"""
from dataclasses import dataclass

from QuantStudio.Core.Input import Binding


@dataclass(frozen=True, init=False)
class NodeTemplate:
    """保存节点类型、参数和显式依赖的不可变模板。"""

    node_class: type
    _args: Binding
    _deps: Binding

    def __init__(self, node_class, args=None, deps=None):
        object.__setattr__(self, "node_class", node_class)
        object.__setattr__(self, "_args", Binding(node_class.__QS_ArgClass__, args))
        object.__setattr__(self, "_deps", Binding(node_class.__QS_DepClass__, deps))

    @property
    def Args(self):
        """显式参数的副本。"""
        return self._args.values

    @property
    def Dependencies(self):
        """显式依赖的副本。"""
        return self._deps.values

    @property
    def missing_dependencies(self):
        """尚未提供的必填槽位。"""
        return self._deps.missing

    def bind(self, **deps):
        """指定或替换依赖并返回新模板。"""
        return type(self)(self.node_class, self.Args, self._deps.bind(**deps).values)

    def unbind(self, *names):
        """移除依赖并返回新模板。"""
        return type(self)(self.node_class, self.Args, self._deps.unbind(*names).values)

    def with_args(self, args=None, **updates):
        """更新显式参数并返回新模板。"""
        return type(self)(self.node_class, self._args.bind(**((args or {}) | updates)).values, self.Dependencies)

    def dependency_info(self):
        """读取依赖声明和绑定状态。"""
        return self._deps.input_info()

    def validate(self):
        """返回参数及依赖的完整校验诊断。"""
        return tuple({**item, "path": (prefix,) + item["path"]}
                     for prefix, binding in (("args", self._args), ("deps", self._deps))
                     for item in binding.validate())

    def build(self, **kwargs):
        """完成校验并调用节点统一构造入口。

        Args:
            kwargs: 日志等运行时构造选项。

        Returns:
            已完成参数和依赖校验的节点。
        """
        args = self._args.build()
        deps = self._deps.build()
        return self.node_class.from_dependencies(deps, args=args, **kwargs)
