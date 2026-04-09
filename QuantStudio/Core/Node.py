# -*- coding: utf-8 -*-
import datetime as dt
from concurrent.futures import Executor
from typing import Any, List, Optional, Dict, Tuple, Literal

from pydantic import Field, ConfigDict

from QuantStudio.Core import __QS_Object__, __QS_Args__
from QuantStudio.Core.Cache import Cache


class Context(__QS_Args__):
    """节点运算时全局上下文对象"""

    Mode: Literal["PRD", "DEBUG"] = Field(default="PRD", title="运行模式")
    NodeDict: Dict[str, "Node"] = Field(default={}, title="节点集", description="{节点ID: Node}, 本次运算的所有 Node, 由计算引擎生成")
    NodeState: Dict[str, Any] = Field(default={}, title="节点状态", description="{节点ID: Any}, 运算中用于存储节点的临时数据，由节点生成和维护")
    PrepareNodeDict: Dict[str, Tuple[str, Any]] = Field(default={}, title="准备节点列表", description="{准备ID: (节点ID, Any)}, 需要执行准备操作的节点列表")
    # ----- 多进程相关 -----
    PID: str = Field(default="0", title="当前进程ID", description="当前的运行进程 ID, 默认为 '0'")
    PIDList: List[str] = Field(default=["0"], title="全部进程ID", description="所有运行进程 ID 列表")
    SplitType: Literal["连续切分", "间隔切分"] = Field(default="连续切分", title="切分方式", frozen=True)
    Event: dict = Field(default={}, title="同步Event", description="{节点ID: Event}, 用于多进程同步的 Event 数据")
    Sub2MainQueue: Optional[Any] = Field(default=None, description="用于子进程向主进程发送消息")
    # ----------
    TaskExecutor: Optional[Executor] = Field(default=None, title="并行执行器", description="给到节点用于并行计算")
    MaxWorkers: int = Field(default=1, title="最大并行数量", description="节点执行并行计算的最大并发量")
    DataCache: Optional[Cache] = Field(default=None, title="数据缓存", frozen=True)
    ExtraData: dict = Field(default={}, title="其他数据")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 并发运行后返回需要同步的内容
    def getUpdateData(self, **kwargs) -> dict:
        return {}

    # 并发运行后更新同步内容
    def updateContext(self, update_data: dict):
        return
    
    # 并发运行时切分自身成 n 份
    def split(self, n: int, **kwargs):
        return [self] * n
    
    def __enter__(self):
        __QS_Context__.append(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if __QS_Context__: __QS_Context__.pop()


class LocalContext(__QS_Args__):
    """节点运算时局部上下文对象"""

    ExtraData: dict = Field(default={}, title="其他数据")
    
    # 并发运行时切分自身成 n 份
    def split(self, n: int, context: Context, **kwargs):
        return [self] * n


class DTLocalContext(LocalContext):
    """时序运算类节点运算时局部上下文对象"""

    DTs: List[dt.datetime] = Field(title="时点序列")


class DTInitData(__QS_Args__):
    """时序运算类节点初始化数据对象"""
    
    DTRange: Tuple[dt.datetime, dt.datetime] = Field(title="时点区间")


class Node(__QS_Object__):
    """计算图中的节点, 独立的计算单元"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name: str = Field(default="Node", frozen=True, title="名称")
        Parallel: bool = Field(default=True, title="并行计算", frozen=True, exclude=True)
        TaskExecutor: Optional[Executor] = Field(default=None, title="并行执行器", description="给到节点用于并行计算", frozen=True, exclude=True)

    def __init__(self, deps:List["Node"]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化计算节点

        Args:
            deps: 该节点所依赖的节点列表
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        self.Deps = deps
        if deps: kwargs.setdefault("logger", deps[0]._QS_Logger)
        return super().__init__(args=args, config_file=config_file, **kwargs)
    
    @property
    def Name(self) -> str:
        """节点名称"""
        return self._QSArgs.Name
    
    def new(self, args={}, **kwargs) -> "Node":
        kwargs = {"deps": self.Deps} | kwargs
        return super().new(args=args, **kwargs)

    def model_dump(self) -> Dict[str, Any]:
        if getattr(self, "_Dumped", False):
            self._Dumped = False
            return super().model_dump()
        self._Dumped = True
        d = super().model_dump()
        d["deps"] = [Node.model_dump() for i, Node in enumerate(self.Deps)]
        self._Dumped = False
        return d

    def compute(self, path: List[str], fwd_data: Any, context: Context) -> Any:
        FwdDataList, LocalContext = self.forward_compute(path, fwd_data, context)
        if FwdDataList:
            BwdDataList = [iNode.compute(path + [iNode.QSID], FwdDataList[i], context) for i, iNode in enumerate(self.Deps)]
        else:
            BwdDataList = []
        return self.backward_compute(path, BwdDataList, context=context, local_context=LocalContext)

    def init_compute(self, path: List[str], init_data: Any, context: Context) -> List[Any]:
        """按照边的方向传递数据执行初始化，可以修改 context 中的全局变量，最好不要有耗时的计算

        Args:
            path: 运行至当前节点的路径, 由路径上所有节点 ID 组成的 list
            init_data: 上游传递的数据
            context: 全局上下文对象

        Returns:
            产生的向下游传递的数据列表, 如果返回空 list 表示终止继续向下的初始化
        """
        if self.QSID in path[:-1]: return []
        return [init_data] * len(self.Deps)

    def prepare_compute(self, prepare_data: Any, context: Context):
        """主逻辑计算开始前的准备计算, 不可以修改 context 中的全局变量，最好将 IO 操作在这里实现，只对 context.PrepareNodeDict 中的节点执行该操作

        Args:
            prepare_data: 执行准备计算所需的数据, 在 init_compute 时生成在 context.PrepareNodeDict 中
            context: 全局上下文对象
        """
        pass

    def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
        """按照边的方向传递数据执行运算, 即从父节点向子节点传递

        Args:
            path: 运行至当前节点的路径, 由路径上所有节点 ID 组成的 list
            fwd_data: 上游传递的数据
            context: 运算时全局上下文对象
        
        Returns: 
            (产生的向下游传递的数据列表, 局部运行时上下文), 如果返回空数据列表表示终止继续向下的运算, 局部运行时上下文将传递给 backward_compute 方法作为入参
        """
        return [fwd_data] * len(self.Deps), fwd_data

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None) -> Any:
        """按照边的反方向传递数据执行运算, 即从子节点向父节点传递

        Args:
            path: 运行至当前节点的路径, 由路径上所有节点 ID 组成的 list
            bwd_data_list: 下游传递的数据列表, 如果为空列表表示在 forward_compute 方法中选择了终止向下的运算
            context: 运算时全局上下文对象
            local_context: 运算时局部上下文对象
        
        Returns:
            计算后产生的结果
        """
        raise NotImplementedError("子类必须实现 backward_compute 方法")
    
    def merge_result(self, result_list: List[Any], context: Context) -> Any:
        """合并并行计算产生的结果

        Args:
            result_list: 并行计算产生的结果列表
            context: 运算时全局上下文对象
        
        Returns:
            合并后的结果
        """        
        return result_list


# 全局运行时环境
__QS_Context__: List[Context] = []