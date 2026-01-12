# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Dict, Tuple

from pydantic import Field, BaseModel, ConfigDict

from QuantStudio.Core import __QS_Object__, QSArgs


class Context(BaseModel):
    NodeDict: Dict[str, "Node"] = Field(default={}, description="{节点ID: Node}, 本次运算的所有 Node，由计算引擎生成")
    NodeState: Dict[str, Any] = Field(default={}, description="{节点ID: Any}, 运算中用于存储节点的临时数据，由节点生成和维护")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Node(__QS_Object__):
    """节点类，以节点为中心的计算单元"""
    class __QS_ArgClass__(QSArgs):
        Name: str = Field(frozen=True, title="名称")

    def __init__(self, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        self.Deps = deps
        return super().__init__(args=args, config_file=config_file, **kwargs)
    
    @property
    def Name(self):
        return self._QSArgs.Name

    def model_dump(self):
        if getattr(self, "_Dumped", False):
            self._Dumped = False
            return super().model_dump()
        self._Dumped = True
        d = super().model_dump()
        d["deps"] = [Node.model_dump() for i, Node in enumerate(self.Deps)]
        self._Dumped = False
        return d

    def init(self, path: List[str], init_data: Any, context: Context):
        context.NodeDict[self.QSID] = self
        InitDataList = self.init_compute(path, init_data, context)
        if not InitDataList: return
        for i, Node in enumerate(self.Deps):
            Node.init(path+[self.QSID], InitDataList[i], context)

    def compute(self, path: List[str], fwd_data: Any, context: Context) -> Any:
        FwdDataList, LocalContext = self.forward_compute(path, fwd_data, context)
        if FwdDataList:
            BwdDataList = [Node.compute(path+[self.QSID], FwdDataList[i], context) for i, Node in enumerate(self.Deps)]
        else:
            BwdDataList = []
        return self.backward_compute(path, BwdDataList, context=context, local_context=LocalContext)

    def init_compute(self, path: List[str], init_data: Any, context: Context) -> List[Any]:
        """
        按照边的方向传递数据执行初始化
        :param path: 运行至当前节点的路径, 所有上游节点 ID 的 list
        :param init_data: 上游传递的数据
        :param context: 全局上下文对象
        :return: 产生的向下游传递的数据列表, 如果返回空 list 表示终止继续向下的初始化
        """
        if self.QSID in path: return []
        return [init_data] * len(self.Deps)

    def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
        """
        按照边的方向传递数据执行运算
        :param path: 运行至当前节点的路径, 所有上游节点 ID 的 list
        :param fwd_data: 上游传递的数据
        :param context: 运算时全局上下文对象
        :return: (产生的向下游传递的数据列表, 局部运行时上下文), 如果返回空数据列表表示终止继续向下的运算, 局部运行时上下文将传递给 backward_compute 方法作为入参
        """
        return [fwd_data] * len(self.Deps), fwd_data

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None) -> Any:
        """
        按照边的反方向传递数据执行运算
        :param path: 运行至当前节点的路径, 所有上游节点 ID 的 list
        :param bwd_data_list: 下游传递的数据列表, 如果为空列表表示在 forward_compute 方法中选择了终止向下的运算
        :param context: 运算时全局上下文对象
        :param local_context: 运算时局部上下文对象
        :return: 产生的消息列表
        """
        raise NotImplementedError("子类必须实现compute方法")
