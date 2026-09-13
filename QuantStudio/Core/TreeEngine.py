# -*- coding: utf-8 -*-
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Literal, Dict

from pydantic import Field
from progressbar import ProgressBar

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Tools.ProcessPoolExecutor import ProcessPoolExecutor


class NodeStatus(Enum):
    UNSTARTED = auto()# 从未运行过
    PENDING = auto()# 暂时挂起
    RUNNING = auto()# 正在运行
    DONE = auto()# 运行结束
    FAILED = auto()# 计算失败
    SKIPPED = auto()# 因上游失败被跳过

def _executeNodeBwdComputeThread(node: Node, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None):
    Result = node.backward_compute(path=path, bwd_data_list=bwd_data_list, context=context, local_context=local_context)
    return "/".join(path), Result, context.getUpdateData(node_id_list=[node.QSID])

__PROC_CONTEXT__: Context = None

def _initProcess(context):
    global __PROC_CONTEXT__
    __PROC_CONTEXT__ = context

def _executeNodeBwdComputeProcess(node_id: str, path: List[str], bwd_data_list: List[Any], updated_data: dict, local_context: Any=None):
    global __PROC_CONTEXT__
    __PROC_CONTEXT__.updateContext(updated_data)
    Node = __PROC_CONTEXT__.NodeDict[node_id]
    Result = Node.backward_compute(path=path, bwd_data_list=bwd_data_list, context=__PROC_CONTEXT__, local_context=local_context)
    return "/".join(path), Result, __PROC_CONTEXT__.getUpdateData(node_id_list=[Node.QSID])

class TreeEngine(Engine):
    class __QS_ArgClass__(Engine.__QS_ArgClass__):
        CalcConcurrentMode: Literal["Thread", "Process"] = Field(default="Process", title="并发模式", frozen=True)
        CalcConcurrentNum: Optional[int] = Field(default=None, title="计算并发数", frozen=True, ge=1)

    # 初始化
    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        if init_data_list is None: init_data_list = [None] * len(node_list)
        Path2Node = {}# {节点路径: Node}
        NodeQ, InitDataQ, PathQ = node_list.copy(), init_data_list.copy(), [[iNode.QSID] for iNode in node_list]
        SkipMode = self._QSArgs.FailMode == "skip"
        while NodeQ:
            iNode, iPath = NodeQ.pop(0), PathQ.pop(0)
            context.NodeDict[iNode.QSID] = iNode
            Path2Node["/".join(iPath)] = iNode
            try:
                iInitDataList = iNode.init_compute(path=iPath, init_data=InitDataQ.pop(0), context=context)
            except Exception as e:
                if not SkipMode: raise
                context.NodeState.setdefault(iNode.QSID, {})["__status__"] = "FAILED"
                context.NodeErrors[iNode.QSID] = e
                self._QS_Logger.error(f"节点 {iNode.Name}({iNode.QSID[:8]}) 初始化失败: {e}")
                continue
            if iInitDataList:
                NodeQ += iNode.Deps
                InitDataQ += iInitDataList
                PathQ += [iPath + [iDep.QSID] for iDep in iNode.Deps]
        self._Path2Node = Path2Node

    def _compute_thread(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if fwd_data_list is None: fwd_data_list = [None] * len(node_list)
        SkipMode = self._QSArgs.FailMode == "skip"
        def handleFwdTask(iNode, iPath, iFwdData):
            nonlocal FwdTaskQ, FwdDataQ, PathQ
            if SkipMode and context.NodeState.get(iNode.QSID, {}).get("__status__") in ("FAILED", "SKIPPED"):
                QSID2Status[iNode.QSID] = NodeStatus.SKIPPED
                iPathStr = "/".join(iPath)
                parentPath = "/".join(iPath.split("/")[:-1]) if "/" in iPathStr else ""
                if parentPath == "":
                    Rslt[RootQSIDList.index(iNode.QSID)] = None
                else:
                    _markFwdFailedDep(iNode, iPathStr)
                return
            if QSID2Status.get(iNode.QSID, NodeStatus.UNSTARTED) in (NodeStatus.RUNNING, NodeStatus.PENDING):# 同样 QSID 的节点正在运行, 暂停前向传播
                QSID2PendingFwdTask.setdefault(iNode.QSID, []).append((iPath, iFwdData))
                return
            try:
                iFwdDataList, iLocalData = iNode.forward_compute(path=iPath, fwd_data=iFwdData, context=context)
            except Exception as e:
                if not SkipMode: raise
                QSID2Status[iNode.QSID] = NodeStatus.FAILED
                context.NodeState.setdefault(iNode.QSID, {})["__status__"] = "FAILED"
                context.NodeErrors[iNode.QSID] = e
                self._QS_Logger.error(f"节点 {iNode.Name}({iNode.QSID[:8]}) 前向计算失败: {e}")
                # 前向计算失败, 该节点不会产生 bwd future, 需要标记其在父节点中的 dep slot 为完成
                if SkipMode:
                    _markFwdFailedDep(iNode, iPath)
                return
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
                        _f = Executor.submit(_executeNodeBwdComputeThread, iNode, iPath, [], context, iLocalData)
                        Future2Path[_f] = "/".join(iPath) if isinstance(iPath, list) else iPath
                        BwdFutureList.append(_f)
                    else:
                        QSID2Status[iNode.QSID] = NodeStatus.PENDING
                        PendingBwdTask.append((_executeNodeBwdComputeThread, iNode, iPath, [], context, iLocalData))
                else:
                    raise __QS_Error__("理论上不应该走到这里!")

        def _markFwdFailedDep(failedNode, failedPath):
            """前向计算失败时, 将该节点在父节点的 dep slot 标记为完成, 并向上传播 SKIPPED."""
            parentPath = "/".join(failedPath.split("/")[:-1])
            if parentPath == "":# 根节点
                Rslt[RootQSIDList.index(failedNode.QSID)] = None
                return
            if parentPath not in Path2DepDone:
                return# 父节点的前向尚未执行到这一步 (不太可能, 但做防御)
            parentNode = self._Path2Node[parentPath]
            depIdx = [iDep.QSID for iDep in parentNode.Deps].index(failedNode.QSID)
            Path2DepDone[parentPath][depIdx] = True
            Path2BwdDataList[parentPath][depIdx] = None
            QSID2Status[parentNode.QSID] = NodeStatus.SKIPPED
            context.NodeState.setdefault(parentNode.QSID, {})["__status__"] = "SKIPPED"
            if all(Path2DepDone[parentPath]):
                # 所有子节点完成(含失败/跳过), 继续向上传播
                propagateSkipUp(parentNode, parentPath)

        def propagateSkipUp(childNode, childPath):
            """将子节点的失败/跳过状态向上逐级传播到所有祖先节点."""
            curNode, curPath = childNode, childPath
            while True:
                parentPath = "/".join(curPath.split("/")[:-1])
                if parentPath == "":# curNode 是根节点, 记录结果
                    Rslt[RootQSIDList.index(curNode.QSID)] = None
                    break
                parentNode = self._Path2Node[parentPath]
                depIdx = [iDep.QSID for iDep in parentNode.Deps].index(curNode.QSID)
                Path2DepDone[parentPath][depIdx] = True
                Path2BwdDataList[parentPath][depIdx] = None
                QSID2Status[parentNode.QSID] = NodeStatus.SKIPPED
                context.NodeState.setdefault(parentNode.QSID, {})["__status__"] = "SKIPPED"
                if not all(Path2DepDone[parentPath]):
                    break# 尚有其他子节点未完成, 等待
                # 所有子节点完成, 继续向上
                curNode, curPath = parentNode, parentPath

        RootQSIDList = [iNode.QSID for iNode in node_list]
        FwdTaskQ, FwdDataQ, PathQ = node_list.copy(), fwd_data_list, [[iNode.QSID] for iNode in node_list]
        BwdFutureList = []
        Future2Path = {}# {future: path_str}, 用于 future 失败时获取 path
        Path2LocalData = {}# {节点路径: 节点执行 backward_compute 所需要的运行时数据}
        Path2BwdDataList = {}# {节点路径: [子节点计算结果]}
        Path2DepDone = {}# {节点路径: [子节点是否计算完成]}
        QSID2Status: Dict[str, NodeStatus] = {}# {节点QSID: 节点状态}
        QSID2PendingFwdTask: Dict[str, list] = {}# {节点QSID: [由于有同样 QSID 的节点在运行而暂时挂起的前向传播任务]}
        # QSID2PendingBwdTask = {}# {节点QSID: [由于有同样 QSID 的节点在运行而暂时挂起的后向传播任务]}
        PendingBwdTask = []# 因为并发数量限制而暂时挂起的任务
        Rslt = [None] * len(node_list)
        with ThreadPoolExecutor(max_workers=self._QSArgs.CalcConcurrentNum) as Executor:
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
                        iPath, iRslt, _ = iFuture.result()
                    except Exception as e:
                        if not SkipMode:
                            self._QS_Logger.error(f"backward_compute 计算失败: {e}")
                            raise e
                        # future 失败时 iPath 未赋值, 从 Future2Path 映射获取
                        iPath = Future2Path[iFuture]
                        iFutureNode = self._Path2Node[iPath]
                        QSID2Status[iFutureNode.QSID] = NodeStatus.FAILED
                        context.NodeState.setdefault(iFutureNode.QSID, {})["__status__"] = "FAILED"
                        context.NodeErrors[iFutureNode.QSID] = e
                        self._QS_Logger.error(f"节点 {iFutureNode.Name}({iFutureNode.QSID[:8]}) 后向计算失败: {e}")
                        iRslt = None
                    Future2Path.pop(iFuture, None)
                    iNode = self._Path2Node[iPath]
                    if SkipMode and QSID2Status.get(iNode.QSID) == NodeStatus.FAILED:
                        # 由 except 块已设置状态, 直接进入父节点传播逻辑
                        pass
                    else:
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
                    # 如果子节点失败或被跳过, 向上传播 SKIPPED
                    if SkipMode and QSID2Status.get(iNode.QSID) in (NodeStatus.FAILED, NodeStatus.SKIPPED):
                        propagateSkipUp(iNode, iPath)
                        continue
                    Path2BwdDataList[iParentPath][iDepIdx] = iRslt
                    Path2DepDone[iParentPath][iDepIdx] = True
                    if all(Path2DepDone[iParentPath]):
                        if SkipMode and QSID2Status.get(iParentNode.QSID) in (NodeStatus.FAILED, NodeStatus.SKIPPED):
                            # 父节点已被标记失败/跳过, 不再提交计算任务
                            Path2BwdDataList.pop(iParentPath, None)
                            Path2LocalData.pop(iParentPath, None)
                            continue
                        iTask = (_executeNodeBwdComputeThread, iParentNode, iParentPath.split("/"), Path2BwdDataList.pop(iParentPath), context, Path2LocalData[iParentPath])
                        if QSID2Status.get(iParentNode.QSID, NodeStatus.UNSTARTED) in (NodeStatus.UNSTARTED, NodeStatus.DONE, NodeStatus.PENDING):
                            if len(BwdFutureList) < self._QSArgs.CalcConcurrentNum:
                                QSID2Status[iParentNode.QSID] = NodeStatus.RUNNING
                                _f = Executor.submit(*iTask)
                                Future2Path[_f] = iParentPath
                                BwdFutureList.append(_f)
                            else:
                                QSID2Status[iParentNode.QSID] = NodeStatus.PENDING
                                PendingBwdTask.append(iTask)
                        else:
                            # QSID2PendingBwdTask.setdefault(iParentNode.QSID, []).append(iTask)
                            raise __QS_Error__(f"理论上不应该走到这里")
                    # 处理挂起的后向传播任务
                    while (len(BwdFutureList) < self._QSArgs.CalcConcurrentNum) and PendingBwdTask:
                        iPendingTask = PendingBwdTask.pop(0)
                        _f = Executor.submit(*iPendingTask)
                        Future2Path[_f] = "/".join(iPendingTask[2]) if isinstance(iPendingTask[2], list) else iPendingTask[2]
                        BwdFutureList.append(_f)
        return Rslt

    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if fwd_data_list is None: fwd_data_list = [None] * len(node_list)
        if self._QSArgs.CalcConcurrentMode == "Thread": return self._compute_thread(node_list=node_list, context=context, fwd_data_list=fwd_data_list)
        SkipMode = self._QSArgs.FailMode == "skip"
        def handleFwdTask(iNode, iPath, iFwdData):
            nonlocal FwdTaskQ, FwdDataQ, PathQ
            if SkipMode and context.NodeState.get(iNode.QSID, {}).get("__status__") in ("FAILED", "SKIPPED"):
                QSID2Status[iNode.QSID] = NodeStatus.SKIPPED
                iPathStr = "/".join(iPath)
                parentPath = "/".join(iPath.split("/")[:-1]) if "/" in iPathStr else ""
                if parentPath == "":
                    Rslt[RootQSIDList.index(iNode.QSID)] = None
                else:
                    _markFwdFailedDep(iNode, iPathStr)
                return
            if QSID2Status.get(iNode.QSID, NodeStatus.UNSTARTED) in (NodeStatus.RUNNING, NodeStatus.PENDING):# 同样 QSID 的节点正在运行, 暂停前向传播
                QSID2PendingFwdTask.setdefault(iNode.QSID, []).append((iPath, iFwdData))
                return
            try:
                iFwdDataList, iLocalData = iNode.forward_compute(path=iPath, fwd_data=iFwdData, context=context)
            except Exception as e:
                if not SkipMode: raise
                QSID2Status[iNode.QSID] = NodeStatus.FAILED
                context.NodeState.setdefault(iNode.QSID, {})["__status__"] = "FAILED"
                context.NodeErrors[iNode.QSID] = e
                self._QS_Logger.error(f"节点 {iNode.Name}({iNode.QSID[:8]}) 前向计算失败: {e}")
                # 前向计算失败, 该节点不会产生 bwd future, 需要标记其在父节点中的 dep slot 为完成
                if SkipMode:
                    _markFwdFailedDep(iNode, iPath)
                return
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
                        _f = Executor.submit(_executeNodeBwdComputeProcess, iNode.QSID, iPath, [], context.getUpdateData(node_id_list=[iNode.QSID]), iLocalData)
                        Future2Path[_f] = "/".join(iPath) if isinstance(iPath, list) else iPath
                        BwdFutureList.append(_f)
                    else:
                        QSID2Status[iNode.QSID] = NodeStatus.PENDING
                        PendingBwdTask.append((_executeNodeBwdComputeProcess, iNode.QSID, iPath, [], context.getUpdateData(node_id_list=[iNode.QSID]), iLocalData))
                else:
                    raise __QS_Error__("理论上不应该走到这里!")

        def _markFwdFailedDep(failedNode, failedPath):
            """前向计算失败时, 将该节点在父节点的 dep slot 标记为完成, 并向上传播 SKIPPED."""
            parentPath = "/".join(failedPath.split("/")[:-1])
            if parentPath == "":# 根节点
                Rslt[RootQSIDList.index(failedNode.QSID)] = None
                return
            if parentPath not in Path2DepDone:
                return# 父节点的前向尚未执行到这一步 (不太可能, 但做防御)
            parentNode = self._Path2Node[parentPath]
            depIdx = [iDep.QSID for iDep in parentNode.Deps].index(failedNode.QSID)
            Path2DepDone[parentPath][depIdx] = True
            Path2BwdDataList[parentPath][depIdx] = None
            QSID2Status[parentNode.QSID] = NodeStatus.SKIPPED
            context.NodeState.setdefault(parentNode.QSID, {})["__status__"] = "SKIPPED"
            if all(Path2DepDone[parentPath]):
                # 所有子节点完成(含失败/跳过), 继续向上传播
                propagateSkipUp(parentNode, parentPath)

        def propagateSkipUp(childNode, childPath):
            """将子节点的失败/跳过状态向上逐级传播到所有祖先节点."""
            curNode, curPath = childNode, childPath
            while True:
                parentPath = "/".join(curPath.split("/")[:-1])
                if parentPath == "":# curNode 是根节点, 记录结果
                    Rslt[RootQSIDList.index(curNode.QSID)] = None
                    break
                parentNode = self._Path2Node[parentPath]
                depIdx = [iDep.QSID for iDep in parentNode.Deps].index(curNode.QSID)
                Path2DepDone[parentPath][depIdx] = True
                Path2BwdDataList[parentPath][depIdx] = None
                QSID2Status[parentNode.QSID] = NodeStatus.SKIPPED
                context.NodeState.setdefault(parentNode.QSID, {})["__status__"] = "SKIPPED"
                if not all(Path2DepDone[parentPath]):
                    break# 尚有其他子节点未完成, 等待
                # 所有子节点完成, 继续向上
                curNode, curPath = parentNode, parentPath

        RootQSIDList = [iNode.QSID for iNode in node_list]
        FwdTaskQ, FwdDataQ, PathQ = node_list.copy(), fwd_data_list, [[iNode.QSID] for iNode in node_list]
        BwdFutureList = []
        Future2Path = {}# {future: path_str}, 用于 future 失败时获取 path
        Path2LocalData = {}# {节点路径: 节点执行 backward_compute 所需要的运行时数据}
        Path2BwdDataList = {}# {节点路径: [子节点计算结果]}
        Path2DepDone = {}# {节点路径: [子节点是否计算完成]}
        QSID2Status: Dict[str, NodeStatus] = {}# {节点QSID: 节点状态}
        QSID2PendingFwdTask: Dict[str, list] = {}# {节点QSID: [由于有同样 QSID 的节点在运行而暂时挂起的前向传播任务]}
        # QSID2PendingBwdTask = {}# {节点QSID: [由于有同样 QSID 的节点在运行而暂时挂起的后向传播任务]}
        PendingBwdTask = []# 因为并发数量限制而暂时挂起的任务
        Rslt = [None] * len(node_list)
        with ProcessPoolExecutor(max_workers=self._QSArgs.CalcConcurrentNum, initializer=_initProcess, initargs=(context,)) as Executor:
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
                        iPath, iRslt, iUpdateData = iFuture.result()
                    except Exception as e:
                        if not SkipMode:
                            self._QS_Logger.error(f"backward_compute 计算失败: {e}")
                            raise e
                        iPath = Future2Path[iFuture]
                        iFutureNode = self._Path2Node[iPath]
                        QSID2Status[iFutureNode.QSID] = NodeStatus.FAILED
                        context.NodeState.setdefault(iFutureNode.QSID, {})["__status__"] = "FAILED"
                        context.NodeErrors[iFutureNode.QSID] = e
                        self._QS_Logger.error(f"节点 {iFutureNode.Name}({iFutureNode.QSID[:8]}) 后向计算失败: {e}")
                        iRslt = None
                        iUpdateData = None
                    Future2Path.pop(iFuture, None)
                    if iUpdateData is not None:
                        context.updateContext(iUpdateData)
                    iNode = self._Path2Node[iPath]
                    if SkipMode and QSID2Status.get(iNode.QSID) == NodeStatus.FAILED:
                        # 由 except 块已设置状态, 直接进入父节点传播逻辑
                        pass
                    else:
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
                    # 如果子节点失败或被跳过, 向上传播 SKIPPED
                    if SkipMode and QSID2Status.get(iNode.QSID) in (NodeStatus.FAILED, NodeStatus.SKIPPED):
                        propagateSkipUp(iNode, iPath)
                        continue
                    Path2BwdDataList[iParentPath][iDepIdx] = iRslt
                    Path2DepDone[iParentPath][iDepIdx] = True
                    if all(Path2DepDone[iParentPath]):
                        if SkipMode and QSID2Status.get(iParentNode.QSID) in (NodeStatus.FAILED, NodeStatus.SKIPPED):
                            Path2BwdDataList.pop(iParentPath, None)
                            Path2LocalData.pop(iParentPath, None)
                            continue
                        iTask = (_executeNodeBwdComputeProcess, iParentNode.QSID, iParentPath.split("/"), Path2BwdDataList.pop(iParentPath), context.getUpdateData(node_id_list=[iParentNode.QSID]), Path2LocalData[iParentPath])
                        if QSID2Status.get(iParentNode.QSID, NodeStatus.UNSTARTED) in (NodeStatus.UNSTARTED, NodeStatus.DONE, NodeStatus.PENDING):
                            if len(BwdFutureList) < self._QSArgs.CalcConcurrentNum:
                                QSID2Status[iParentNode.QSID] = NodeStatus.RUNNING
                                _f = Executor.submit(*iTask)
                                Future2Path[_f] = iParentPath
                                BwdFutureList.append(_f)
                            else:
                                QSID2Status[iParentNode.QSID] = NodeStatus.PENDING
                                PendingBwdTask.append(iTask)
                        else:
                            # QSID2PendingBwdTask.setdefault(iParentNode.QSID, []).append(iTask)
                            raise __QS_Error__(f"理论上不应该走到这里")
                    # 处理挂起的后向传播任务
                    while (len(BwdFutureList) < self._QSArgs.CalcConcurrentNum) and PendingBwdTask:
                        iPendingTask = PendingBwdTask.pop(0)
                        _f = Executor.submit(*iPendingTask)
                        Future2Path[_f] = "/".join(iPendingTask[2]) if isinstance(iPendingTask[2], list) else iPendingTask[2]
                        BwdFutureList.append(_f)
        return Rslt