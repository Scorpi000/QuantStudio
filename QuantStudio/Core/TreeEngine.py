# -*- coding: utf-8 -*-
import queue
import signal
import threading
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed, Executor, Future
from typing import Any, List, Optional, Callable, Literal, Dict

from pydantic import Field
from progressbar import ProgressBar
from multiprocess import Process, Queue, cpu_count

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.CalcEngine import Engine


class NodeStatus(Enum):
    UNSTARTED = auto()# 从未运行过
    PENDING = auto()# 暂时挂起
    RUNNING = auto()# 正在运行
    DONE = auto()# 运行结束

class _TaskSpec:
    """纯数据任务描述，可安全序列化"""
    __slots__ = ('fn', 'args', 'kwargs', 'task_id')
    
    def __init__(self, fn: Callable, args: tuple, kwargs: dict, task_id: int):
        # 确保函数可序列化（模块级函数或顶层函数）
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.task_id = task_id


class _ResultSpec:
    """纯数据结果描述"""
    __slots__ = ('status', 'task_id', 'payload')
    
    def __init__(self, status: str, task_id: int, payload: Any):
        self.status = status  # 'SUCCESS', 'ERROR', 'INIT_ERROR'
        self.task_id = task_id
        self.payload = payload


# ============ 工作进程 ============

def _worker_process(task_queue: Queue, result_queue: Queue, initializer=None, initargs=()):
    """
    工作进程主循环 - 完全隔离，不共享任何线程锁
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # 初始化
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException as e:
            result_queue.put(_ResultSpec('INIT_ERROR', -1, e))
            return
    
    # 主循环
    while True:
        try:
            task_spec = task_queue.get()
            if task_spec is None:  # 终止信号
                break
            
            # 在子进程中执行任务
            try:
                result = task_spec.fn(*task_spec.args, **task_spec.kwargs)
                result_queue.put(_ResultSpec('SUCCESS', task_spec.task_id, result))
            except BaseException as e:
                result_queue.put(_ResultSpec('ERROR', task_spec.task_id, e))
                
        except (EOFError, OSError, KeyboardInterrupt):
            break


# ============ 执行器 ============

class ProcessPoolExecutor(Executor):
    """
    基于 multiprocessing.Process 的进程池实现
    """
    
    def __init__(self, max_workers: Optional[int] = None,
                 initializer: Optional[Callable] = None,
                 initargs: tuple = ()):
        super().__init__()
        
        self._max_workers = max_workers or cpu_count()
        self._initializer = initializer
        self._initargs = initargs
        
        # 使用 Manager 队列或普通 Queue（spawn 模式下普通 Queue 即可）
        self._task_queue = Queue()
        self._result_queue = Queue()
        
        # Future 管理 - 仅在主进程访问
        self._futures: dict[int, Future] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        
        # 状态控制
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._result_thread: Optional[threading.Thread] = None
        self._processes: list[Process] = []
        
        # 启动
        self._start_result_thread()
        self._spawn_workers()
    
    def _start_result_thread(self):
        """启动结果分发线程"""
        self._result_thread = threading.Thread(
            target=self._result_loop,
            daemon=True,
            name="ResultDispatcher"
        )
        self._result_thread.start()
    
    def _spawn_workers(self):
        """启动工作进程"""
        for i in range(self._max_workers):
            p = Process(
                target=_worker_process,
                args=(self._task_queue, self._result_queue,
                      self._initializer, self._initargs),
                name=f"PoolWorker-{i}"
            )
            p.daemon = True
            p.start()
            self._processes.append(p)
    
    def _result_loop(self):
        """后台线程：将子进程结果路由到对应 Future"""
        while True:
            try:
                # 超时检查以便响应 shutdown
                try:
                    result = self._result_queue.get(timeout=0.1)
                except queue.Empty:
                    with self._shutdown_lock:
                        if self._shutdown and not self._futures:
                            break
                    continue
                
                # 处理结果
                if result.status == 'INIT_ERROR':
                    with self._lock:
                        for future in list(self._futures.values()):
                            future.set_exception(RuntimeError(f"Worker init failed: {result.payload}"))
                        self._futures.clear()
                    continue
                
                # 路由到对应 Future
                with self._lock:
                    future = self._futures.pop(result.task_id, None)
                
                if future is None:
                    continue  # 可能已被取消
                
                if result.status == 'SUCCESS':
                    future.set_result(result.payload)
                else:
                    future.set_exception(result.payload)
                    
            except (EOFError, OSError):
                break
    
    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        """提交任务 - 只发送纯数据，不发送 Future"""
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("Executor is shutdown")
        
        # 创建 Future（仅主进程持有）
        future = Future()
        
        # 生成任务 ID 并注册
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            self._futures[task_id] = future
        
        # 创建可序列化的任务描述
        task_spec = _TaskSpec(fn, args, kwargs, task_id)
        
        # 发送到子进程
        self._task_queue.put(task_spec)
        
        return future
    
    def map(self, fn: Callable, *iterables, timeout=None, chunksize=1):
        """并行 map 实现"""
        fs = [self.submit(fn, *args) for args in zip(*iterables)]
        
        def result_iterator():
            for future in fs:
                try:
                    yield future.result(timeout=timeout)
                except Exception:
                    future.cancel()
                    raise
        
        return result_iterator()
    
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """优雅关闭"""
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
        
        # 取消未完成任务
        if cancel_futures:
            with self._lock:
                for future in self._futures.values():
                    future.cancel()
                self._futures.clear()
        
        # 发送终止信号
        for _ in self._processes:
            self._task_queue.put(None)
        
        if wait:
            # 等待工作进程
            for p in self._processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)
            
            # 等待结果线程
            if self._result_thread.is_alive():
                self._result_thread.join(timeout=5)
            
            # 清理队列
            self._task_queue.close()
            self._result_queue.close()
            self._task_queue.join_thread()
            self._result_queue.join_thread()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown(wait=True)
        return False


def _executeNodeBwdCompute(node: Node, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None):
    Result = node.backward_compute(path=path, bwd_data_list=bwd_data_list, context=context, local_context=local_context)
    return "/".join(path), Result

class TreeEngine(Engine):
    class __QS_ArgClass__(Engine.__QS_ArgClass__):
        CalcConcurrentMode: Literal["Thread", "Process"] = Field(default="Thread", title="并发模式", frozen=True)
        CalcConcurrentNum: Optional[int] = Field(default=None, title="计算并发数", frozen=True, ge=1)

    def compute1(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        RootQSIDList = [iNode.QSID for iNode in node_list]
        FwdTaskQ, FwdDataQ, PathQ = node_list.copy(), fwd_data_list, [[iNode.QSID] for iNode in node_list]
        BwdFutureList = []
        Path2Node = self._Path2Node
        Path2LocalData = {}# {节点路径: 节点执行 backward_compute 所需要的运行时数据}
        Path2BwdDataList = {}# {节点路径: [子节点计算结果]}
        Path2DepStatus = {}# {节点路径: [子节点状态，即是否计算完成]}
        QSID2Status = {}# {节点QSID: 是否在执行}
        QSIDPendingTask = {}# {节点QSID: [由于有同样 QSID 的节点在运行而暂时挂起的任务]}
        PendingTask = []# 因为并发数量限制而暂时挂起的任务
        Rslt = [None] * len(node_list)
        Executor = ThreadPoolExecutor(max_workers=self._QSArgs.CalcConcurrentNum) if self._QSArgs.CalcConcurrentMode=="Thread" else ProcessPoolExecutor(max_workers=self._QSArgs.CalcConcurrentNum)
        with Executor:
            while FwdTaskQ:
                iNode, iPath = FwdTaskQ.pop(0), PathQ.pop(0)
                iFwdDataList, iLocalData = iNode.forward_compute(path=iPath, fwd_data=FwdDataQ.pop(0), context=context)
                iPathStr = "/".join(iPath)
                if iFwdDataList:
                    FwdTaskQ += iNode.Deps
                    FwdDataQ += iFwdDataList
                    PathQ += [iPath + [iDep.QSID] for iDep in iNode.Deps]
                    Path2LocalData[iPathStr] = iLocalData
                    Path2BwdDataList[iPathStr] = [None] * len(iNode.Deps)
                    Path2DepStatus[iPathStr] = [False] * len(iNode.Deps)
                else:
                    if not QSID2Status.get(iNode.QSID, False):
                        if len(BwdFutureList) < self._QSArgs.CalcConcurrentNum:
                            QSID2Status[iNode.QSID] = True
                            BwdFutureList.append(Executor.submit(_executeNodeBwdCompute, iNode, iPath, [], context, iLocalData))
                        else:
                            PendingTask.append((_executeNodeBwdCompute, iNode, iPath, [], context, iLocalData))
                    else:
                        QSIDPendingTask.setdefault(iNode.QSID, []).append((_executeNodeBwdCompute, iNode, iPath, [], context, iLocalData))
            with ProgressBar(max_value=len(Path2Node)) as ProgBar:
                while BwdFutureList:
                    iFuture = next(as_completed(BwdFutureList))
                    BwdFutureList.remove(iFuture)
                    ProgBar.update(ProgBar.value + 1)
                    try:
                        iPath, iRslt = iFuture.result()
                    except Exception as e:
                        self._QS_Logger.error(f"backward_compute 计算失败: {e}")
                        raise e
                    iNode = Path2Node[iPath]
                    if QSIDPendingTask.get(iNode.QSID, []):
                        BwdFutureList.append(Executor.submit(*QSIDPendingTask[iNode.QSID].pop(0)))
                    else:
                        QSID2Status[iNode.QSID] = False
                    iParentPath = "/".join(iPath.split("/")[:-1])
                    if iParentPath == "":# 根节点
                        Rslt[RootQSIDList.index(iNode.QSID)] = iRslt
                        continue
                    iParentNode = Path2Node[iParentPath]
                    iDepIdx = [iDep.QSID for iDep in iParentNode.Deps].index(iNode.QSID)
                    Path2BwdDataList[iParentPath][iDepIdx] = iRslt
                    Path2DepStatus[iParentPath][iDepIdx] = True
                    if all(Path2DepStatus[iParentPath]):
                        iTask = (_executeNodeBwdCompute, iParentNode, iParentPath.split("/"), Path2BwdDataList.pop(iParentPath), context, Path2LocalData[iParentPath])
                        if not QSID2Status.get(iParentNode.QSID, False):
                            if len(BwdFutureList) < self._QSArgs.CalcConcurrentNum:
                                QSID2Status[iParentNode.QSID] = True
                                BwdFutureList.append(Executor.submit(*iTask))
                            else:
                                PendingTask.append(iTask)
                        else:
                            QSIDPendingTask.setdefault(iParentNode.QSID, []).append(iTask)
                    while (len(BwdFutureList) < self._QSArgs.CalcConcurrentNum) and PendingTask:
                        BwdFutureList.append(Executor.submit(*PendingTask.pop(0)))
        return Rslt
    
    # 初始化
    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        Path2Node = {}# {节点路径: Node}
        NodeQ, InitDataQ, PathQ = node_list.copy(), init_data_list, [[iNode.QSID] for iNode in node_list]
        while NodeQ:
            iNode, iPath = NodeQ.pop(0), PathQ.pop(0)
            context.NodeDict[iNode.QSID] = iNode
            Path2Node["/".join(iPath)] = iNode
            iInitDataList = iNode.init_compute(path=iPath, init_data=InitDataQ.pop(0), context=context)
            if iInitDataList:
                NodeQ += iNode.Deps
                InitDataQ += iInitDataList
                PathQ += [iPath + [iDep.QSID] for iDep in iNode.Deps]
        self._Path2Node = Path2Node

    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        def handleFwdTask(iNode, iPath, iFwdData):
            nonlocal FwdTaskQ, FwdDataQ, PathQ
            if QSID2Status.get(iNode.QSID, NodeStatus.UNSTARTED) in (NodeStatus.RUNNING, NodeStatus.PENDING):# 同样 QSID 的节点正在运行, 暂停前向传播
                QSID2PendingFwdTask.setdefault(iNode.QSID, []).append((iPath, iFwdData))
                return
            iFwdDataList, iLocalData = iNode.forward_compute(path=iPath, fwd_data=iFwdData, context=context)
            iPathStr = "/".join(iPath)
            if iFwdDataList:
                QSID2Status[iNode.QSID] = NodeStatus.PENDING
                FwdTaskQ += iNode.Deps
                FwdDataQ += iFwdDataList
                PathQ += [iPath + [iDep.QSID] for iDep in iNode.Deps]
                Path2LocalData[iPathStr] = iLocalData
                Path2BwdDataList[iPathStr] = [None] * len(iNode.Deps)
                Path2DepDone[iPathStr] = [False] * len(iNode.Deps)
            else:
                if QSID2Status.get(iNode.QSID, NodeStatus.UNSTARTED) in (NodeStatus.UNSTARTED, NodeStatus.DONE):
                    if len(BwdFutureList) < self._QSArgs.CalcConcurrentNum:
                        QSID2Status[iNode.QSID] = NodeStatus.RUNNING
                        BwdFutureList.append(Executor.submit(_executeNodeBwdCompute, iNode, iPath, [], context, iLocalData))
                    else:
                        QSID2Status[iNode.QSID] = NodeStatus.PENDING
                        PendingBwdTask.append((_executeNodeBwdCompute, iNode, iPath, [], context, iLocalData))
                else:
                    raise __QS_Error__("理论上不应该走到这里!")

        RootQSIDList = [iNode.QSID for iNode in node_list]
        FwdTaskQ, FwdDataQ, PathQ = node_list.copy(), fwd_data_list, [[iNode.QSID] for iNode in node_list]
        BwdFutureList = []
        Path2LocalData = {}# {节点路径: 节点执行 backward_compute 所需要的运行时数据}
        Path2BwdDataList = {}# {节点路径: [子节点计算结果]}
        Path2DepDone = {}# {节点路径: [子节点是否计算完成]}
        QSID2Status: Dict[str, NodeStatus] = {}# {节点QSID: 节点状态}
        QSID2PendingFwdTask: Dict[str, list] = {}# {节点QSID: [由于有同样 QSID 的节点在运行而暂时挂起的前向传播任务]}
        # QSID2PendingBwdTask = {}# {节点QSID: [由于有同样 QSID 的节点在运行而暂时挂起的后向传播任务]}
        PendingBwdTask = []# 因为并发数量限制而暂时挂起的任务
        Rslt = [None] * len(node_list)
        Executor = ThreadPoolExecutor(max_workers=self._QSArgs.CalcConcurrentNum) if self._QSArgs.CalcConcurrentMode=="Thread" else ProcessPoolExecutor(max_workers=self._QSArgs.CalcConcurrentNum)
        with Executor:
            with ProgressBar(max_value=len(self._Path2Node)) as ProgBar:
                while BwdFutureList or FwdTaskQ:
                    # 处理前向传播
                    while FwdTaskQ:
                        iNode, iPath, iFwdData = FwdTaskQ.pop(0), PathQ.pop(0), FwdDataQ.pop(0)
                        handleFwdTask(iNode, iPath, iFwdData)
                    # 处理后向传播
                    iFuture = next(as_completed(BwdFutureList))
                    BwdFutureList.remove(iFuture)
                    ProgBar.update(ProgBar.value + 1)
                    try:
                        iPath, iRslt = iFuture.result()
                    except Exception as e:
                        self._QS_Logger.error(f"backward_compute 计算失败: {e}")
                        raise e
                    iNode = self._Path2Node[iPath]
                    QSID2Status[iNode.QSID] = NodeStatus.DONE
                    # 处理挂起的前向传播任务
                    if QSID2PendingFwdTask.get(iNode.QSID, []):
                        handleFwdTask(iNode, *QSID2PendingFwdTask[iNode.QSID].pop(0))
                    # # 处理挂起的后向传播任务
                    # if QSID2PendingBwdTask.get(iNode.QSID, []):
                    #     BwdFutureList.append(Executor.submit(*QSID2PendingBwdTask[iNode.QSID].pop(0)))
                    # else:
                    #     QSID2Status[iNode.QSID] = False
                    # 处理父节点的后向传播任务
                    iParentPath = "/".join(iPath.split("/")[:-1])
                    if iParentPath == "":# 根节点
                        Rslt[RootQSIDList.index(iNode.QSID)] = iRslt
                        continue
                    iParentNode = self._Path2Node[iParentPath]
                    iDepIdx = [iDep.QSID for iDep in iParentNode.Deps].index(iNode.QSID)
                    Path2BwdDataList[iParentPath][iDepIdx] = iRslt
                    Path2DepDone[iParentPath][iDepIdx] = True
                    if all(Path2DepDone[iParentPath]):
                        iTask = (_executeNodeBwdCompute, iParentNode, iParentPath.split("/"), Path2BwdDataList.pop(iParentPath), context, Path2LocalData[iParentPath])
                        if QSID2Status.get(iParentNode.QSID, NodeStatus.UNSTARTED) in (NodeStatus.UNSTARTED, NodeStatus.DONE, NodeStatus.PENDING):
                            if len(BwdFutureList) < self._QSArgs.CalcConcurrentNum:
                                QSID2Status[iParentNode.QSID] = NodeStatus.RUNNING
                                BwdFutureList.append(Executor.submit(*iTask))
                            else:
                                QSID2Status[iParentNode.QSID] = NodeStatus.PENDING
                                PendingBwdTask.append(iTask)
                        else:
                            # QSID2PendingBwdTask.setdefault(iParentNode.QSID, []).append(iTask)
                            raise __QS_Error__(f"理论上不应该走到这里")
                    # 处理挂起的后向传播任务
                    while (len(BwdFutureList) < self._QSArgs.CalcConcurrentNum) and PendingBwdTask:
                        BwdFutureList.append(Executor.submit(*PendingBwdTask.pop(0)))
        return Rslt