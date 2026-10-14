# -*- coding: utf-8 -*-
import datetime as dt
import inspect
from functools import wraps
from concurrent.futures import Executor
from typing import Any, List, Optional, Dict, Tuple, Literal

from pydantic import Field, ConfigDict

from QuantStudio.Core import __QS_Object__, __QS_Args__
from QuantStudio.Core.Cache import Cache
from QuantStudio.Core.Input import DependencyModel, DependencyValues, Binding, dependency_edges
from QuantStudio.Core.Identity import ExecutionEffect, new_instance_id, execution_key


class Context(__QS_Args__):
    """节点运算时全局上下文对象"""

    Mode: Literal["PRD", "DEBUG"] = Field(default="PRD", title="运行模式")
    RunID: str = Field(default_factory=new_instance_id, exclude=True, title="运行标识")
    DataVersion: Optional[str] = Field(default=None, exclude=True, title="数据快照版本")
    Resources: dict = Field(default_factory=dict, exclude=True, title="运行资源")
    EventHandler: Optional[Any] = Field(default=None, exclude=True, repr=False, title="执行事件处理器")
    NodeDict: Dict[str, "Node"] = Field(default={}, title="节点集", description="{节点ID: Node}, 本次运算的所有 Node, 由计算引擎生成")
    NodeState: Dict[str, Any] = Field(default={}, title="节点状态", description="{节点ID: Any}, 运算中用于存储节点的临时数据，由节点生成和维护")
    NodeErrors: Dict[str, Exception] = Field(default={}, title="节点错误", description="{节点ID: Exception}, 记录失败节点的异常信息, 由计算引擎在 FailMode='skip' 时填充")
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
    
    # 并发运行后返回需要同步的内容
    def getUpdateData(self, **kwargs) -> dict:
        return {}

    # 并发运行后更新同步内容
    def updateContext(self, update_data: dict):
        return
    
    # 并发运行时切分自身成 n 份
    def split(self, n: int, **kwargs):
        return [self] * n

    def emit(self, node, stage, status, **details):
        """发送与应用传输方式无关的执行事件。"""
        if self.EventHandler is not None:
            self.EventHandler({"run_id": self.RunID, "instance_id": node.InstanceID,
                               "qsid": node.QSID, "stage": stage, "status": status, **details})

    def begin_run(self):
        """隔离连续运行的状态，保留显式资源和数据缓存。"""
        self.RunID = new_instance_id()
        self.NodeDict.clear()
        self.NodeState.clear()
        self.NodeErrors.clear()
        self.PrepareNodeDict.clear()
        self.Event.clear()
        self.ExtraData.pop("_QS_FailSkip", None)
    
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

class DTInitData(__QS_Args__):
    """时序运算类节点初始化数据对象"""
    
    DTRange: Tuple[dt.datetime, dt.datetime] = Field(title="时点区间")

class DTLocalContext(LocalContext, DTInitData):
    """时序运算类节点运算时局部上下文对象"""

    DTs: List[dt.datetime] = Field(title="时点序列")

    def __init__(self, /, _owner=None, _logger=None, **data: Any) -> None:
        if "DTRange" not in data:
            data["DTRange"] = (data["DTs"][0], data["DTs"][-1])
        return super().__init__(_owner=_owner, _logger=_logger, **data)


class NodeDependencies(DependencyModel):
    """通用节点的有序依赖集合。"""

    nodes: tuple["Node", ...] = Field(default=(), title="依赖节点")


class _DependencyInfo:
    def __get__(self, instance, owner):
        def read():
            model = owner.__QS_DepClass__ if instance is None else type(instance.Dependencies)
            values = None if instance is None else {name: getattr(instance.Dependencies, name) for name in model.model_fields}
            return model.input_info(values)
        return read


def _lifecycle(method, stage):
    """在生命周期边界统一组织命名依赖值。"""
    parameters = inspect.signature(method).parameters

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if stage == "backward_compute":
            args = list(args)
            key = "inputs" if "inputs" in parameters else "bwd_data_list"
            other = "bwd_data_list" if key == "inputs" else "inputs"
            if other in kwargs:
                kwargs[key] = kwargs.pop(other)
            if key in kwargs:
                value = kwargs[key]
                kwargs[key] = self.dependency_values(value) if value or not self.Deps else value
            elif len(args) > 1:
                args[1] = self.dependency_values(args[1]) if args[1] or not self.Deps else args[1]
            return method(self, *args, **kwargs)
        result = method(self, *args, **kwargs)
        if stage == "forward_compute":
            requests, local = result
            return (self.dependency_values(requests) if requests else requests), local
        return self.dependency_values(result) if result else result
    return wrapped


class Node(__QS_Object__):
    """计算图中的节点, 独立的计算单元"""

    __QS_DepClass__ = NodeDependencies
    __QS_BehaviorVersion__ = 1
    __QS_ExecutionEffect__ = ExecutionEffect.STATEFUL
    __QS_DynamicDependencies__ = False
    dependency_info = _DependencyInfo()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("init_compute", "forward_compute", "backward_compute"):
            if name in cls.__dict__:
                setattr(cls, name, _lifecycle(cls.__dict__[name], name))

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        Name: str = Field(default="Node", frozen=True, exclude=True, title="名称")
        Parallel: bool = Field(default=True, title="并行计算", frozen=True, exclude=True, repr=False)
        TaskExecutor: Optional[Executor] = Field(default=None, title="并行执行器", description="给到节点用于并行计算", frozen=True, exclude=True, repr=False)

    def __init__(self, deps=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化计算节点

        Args:
            deps: 该节点所依赖的节点列表
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        if isinstance(deps, DependencyModel):
            if isinstance(deps, self.__QS_DepClass__) or self.__QS_DynamicDependencies__:
                self._Dependencies = deps
            else:
                self._Dependencies = Binding(self.__QS_DepClass__,
                                             {name: getattr(deps, name) for name in type(deps).model_fields}).build()
        else:
            if isinstance(deps, (list, tuple)):
                names = list(self.__QS_DepClass__.model_fields)
                if not deps and not names:
                    deps = {}
                elif names != ["nodes"]:
                    raise TypeError("命名依赖模型必须通过字段绑定")
                else:
                    deps = {"nodes": tuple(deps)}
            self._Dependencies = Binding(self.__QS_DepClass__, deps).build()
        self._InstanceID = new_instance_id()
        if self.Deps: kwargs.setdefault("logger", self.Deps[0]._QS_Logger)
        return super().__init__(args=args, config_file=config_file, **kwargs)

    @property
    def Dependencies(self):
        """完整且不可变的依赖模型。"""
        return self._Dependencies

    @property
    def Deps(self):
        """供调度使用的展开副本，依赖模型是唯一来源。"""
        return [edge.node for edge in self.iter_dependency_edges()]

    @property
    def InstanceID(self):
        """独立于计算语义的节点实例标识。"""
        return self._InstanceID

    @property
    def ExecutionEffect(self):
        """节点的执行副作用声明。"""
        return self.__QS_ExecutionEffect__

    def iter_dependency_edges(self):
        """按照槽位和集合顺序遍历依赖边。"""
        return dependency_edges(self.Dependencies)

    def dependency_values(self, values):
        """按当前依赖形状组织请求或结果。"""
        return DependencyValues(self.Dependencies, values)

    @classmethod
    def template(cls, args=None, deps=None):
        """创建允许部分参数和依赖的模板。"""
        from QuantStudio.Core.NodeTemplate import NodeTemplate
        return NodeTemplate(cls, args, deps)

    @classmethod
    def from_dependencies(cls, deps, args=None, **kwargs):
        """根据完整模型构造节点，不重复依赖便利构造签名。"""
        node = cls.__new__(cls)
        values = args.to_dict(repr=False) if isinstance(args, __QS_Args__) else (args or {})
        Node.__init__(node, deps=deps, args=values, **kwargs)
        node._QS_post_bind()
        return node

    def _QS_post_bind(self):
        """统一构造完成后的子类初始化钩子。"""
        return None

    def execution_key(self, request, data_version=None):
        """生成包含请求与数据版本的执行结果标识。"""
        return execution_key(self.QSID, request, data_version)
    
    @property
    def Name(self) -> str:
        """节点名称"""
        return self._QSArgs.Name
    
    def new(self, args={}, **kwargs) -> "Node":
        deps = kwargs.pop("deps", self.Dependencies)
        if not isinstance(deps, DependencyModel):
            deps = Binding(type(self.Dependencies), deps).build()
        return type(self).from_dependencies(deps, args=(self.Args | {}) | args, **({"logger": self.Logger} | kwargs))

    def model_dump(self) -> Dict[str, Any]:
        d = super().model_dump()
        d["behavior_version"] = self.__QS_BehaviorVersion__
        d["deps"] = {name: ([node.QSID for node in value] if isinstance(value, tuple) else
                            (None if value is None else value.QSID))
                     for name, value in ((name, getattr(self.Dependencies, name)) for name in type(self.Dependencies).model_fields)}
        return d

    def compute(self, path: List[str], fwd_data: Any, context: Context) -> Any:
        SkipMode = context.ExtraData.get("_QS_FailSkip", False)
        context.emit(self, "compute", "running")
        FwdDataList, LocalContext = self.forward_compute(path, fwd_data, context)
        if FwdDataList:
            BwdDataList = []
            for i, iNode in enumerate(self.Deps):
                iResult = iNode.compute(path + [iNode.InstanceID], FwdDataList[i], context)
                if context.NodeState.get(iNode.InstanceID, {}).get("__status__") in ("FAILED", "SKIPPED"):
                    context.NodeState.setdefault(self.InstanceID, {})["__status__"] = "SKIPPED"
                    context.emit(self, "compute", "skipped")
                    return None
                BwdDataList.append(iResult)
        else:
            BwdDataList = []
        try:
            result = self.backward_compute(path, BwdDataList, context=context, local_context=LocalContext)
            context.emit(self, "compute", "done")
            return result
        except Exception as e:
            context.emit(self, "compute", "failed", error=str(e))
            if not SkipMode: raise
            context.NodeState.setdefault(self.InstanceID, {})["__status__"] = "FAILED"
            context.NodeErrors[self.InstanceID] = e
            self._QS_Logger.error(f"节点 {self.Name}({self.QSID[:8]}) 计算失败: {e}")
            return None

    def init_compute(self, path: List[str], init_data: Any, context: Context) -> List[Any]:
        """按照边的方向传递数据执行初始化，可以修改 context 中的全局变量，最好不要有耗时的计算

        Args:
            path: 运行至当前节点的路径, 由路径上所有节点 ID 组成的 list
            init_data: 上游传递的数据
            context: 全局上下文对象

        Returns:
            产生的向下游传递的数据列表, 如果返回空 list 表示终止继续向下的初始化
        """
        if self.InstanceID in path[:-1]: return []
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
NodeDependencies.model_rebuild(_types_namespace={"Node": Node})
__QS_Context__: List[Context] = []
