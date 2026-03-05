# -*- coding: utf-8 -*-
"""基本运算"""
from typing import Any, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.CalcEngine import SimpleEngine


class Add(Node):
    def __init__(self, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        if "name" not in args: args["name"] = "add"
        return super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context:Any=None) -> Any:
        return sum(bwd_data_list)

class Prod(Node):
    def __init__(self, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        if "name" not in args: args["name"] = "prod"
        return super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context:Any=None) -> Any:
        return np.prod(bwd_data_list)

class Float(Node):
    def __init__(self, value:float, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        if "name" not in args: args["name"] = str(value)
        self._Value = value
        return super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context:Any=None) -> Any:
        return self._Value

if __name__ == "__main__1":
    Node1 = Prod([Add([Float(1), Float(2)]), Float(3)], args={"name": "(1 + 2) * 3"})
    Node2 = Add([Float(3), Prod([Float(3), Float(2)])], args={"name": "3 + 3 * 2"})

    Engine = SimpleEngine()
    NodeList = [Node1, Node2]
    Rslt = Engine.run(NodeList, Context())
    for i, iNode in enumerate(NodeList):
        print(f"{iNode.Args.name}: ", Rslt[i])

    print("===")


# class X(Node):
#     class __QS_ArgClass__(Node.__QS_ArgClass__):
#         name: str = Field(default="x", frozen=True, title="名称")
#         init_val: float = Field(frozen=True, title="初始值")
#         iter_num: int = Field(default=20, frozen=True, title="迭代次数")
#
#     def __init__(self, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
#         super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)
#         # self._Value = self.Args.init_val
#
#     def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
#         if path.count(self.QSID)>=self.Args.iter_num:
#             return [], None
#         else:
#             return [None, None], None
#
#     def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None) -> Any:
#         if bwd_data_list:
#             Value = sum(bwd_data_list)
#         else:
#             Value = self.Args.init_val
#         print(f"{self.Args.name}: ", Value)
#         return Value
#
# class Y(Node):
#     class __QS_ArgClass__(Node.__QS_ArgClass__):
#         name: str = Field(default="y", frozen=True, title="名称")
#         init_val: float = Field(frozen=True, title="初始值")
#         iter_num: int = Field(default=20, frozen=True, title="迭代次数")
#
#     def __init__(self, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
#         super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)
#         # self._Value = self.Args.init_val
#
#     def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
#         if path.count(self.QSID)>=self.Args.iter_num:
#             return [], None
#         else:
#             return [None], None
#
#     def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None) -> Any:
#         if bwd_data_list:
#             Value = bwd_data_list[0] + 1
#         else:
#             Value = self.Args.init_val
#         print(f"{self.Args.name}: ", Value)
#         return Value
#
# class Z(Node):
#     class __QS_ArgClass__(Node.__QS_ArgClass__):
#         name: str = Field(default="z", frozen=True, title="名称")
#         init_val: float = Field(frozen=True, title="初始值")
#         iter_num: int = Field(default=20, frozen=True, title="迭代次数")
#
#     def __init__(self, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
#         super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)
#         # self._Value = self.Args.init_val
#
#     def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
#         if path.count(self.QSID)>=self.Args.iter_num:
#             return [], None
#         else:
#             return [None], None
#
#     def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None) -> Any:
#         if bwd_data_list:
#             Value = bwd_data_list[0] + 2
#         else:
#             Value = self.Args.init_val
#         print(f"{self.Args.name}: ", Value)
#         return Value
#
# # 迭代运算
# if __name__=="__main__":
#     # x = y + z, y = x + 1, z = y + 2
#     x = X(deps=[], args={"name": "x", "init_val": 1, "iter_num": 2})
#     y = Y(deps=[x], args={"name": "y", "init_val": 1, "iter_num": 2})
#     z = Z(deps=[y], args={"name": "z", "init_val": 1, "iter_num": 2})
#     x.Deps = [y, z]
#
#     Engine = SimpleEngine()
#     Rslt = Engine.run([x], Context())
#     print(Rslt)
#
#     print("===")
