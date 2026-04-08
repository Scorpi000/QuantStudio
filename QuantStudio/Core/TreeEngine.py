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

    def _compute_thread(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
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
                        BwdFutureList.append(Executor.submit(_executeNodeBwdComputeThread, iNode, iPath, [], context, iLocalData))
                    else:
                        QSID2Status[iNode.QSID] = NodeStatus.PENDING
                        PendingBwdTask.append((_executeNodeBwdComputeThread, iNode, iPath, [], context, iLocalData))
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
                        iTask = (_executeNodeBwdComputeThread, iParentNode, iParentPath.split("/"), Path2BwdDataList.pop(iParentPath), context, Path2LocalData[iParentPath])
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

    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if self._QSArgs.CalcConcurrentMode == "Thread": return self._compute_thread(node_list=node_list, context=context, fwd_data_list=fwd_data_list)
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
                        BwdFutureList.append(Executor.submit(_executeNodeBwdComputeProcess, iNode.QSID, iPath, [], context.getUpdateData(node_id_list=[iNode.QSID]), iLocalData))
                    else:
                        QSID2Status[iNode.QSID] = NodeStatus.PENDING
                        PendingBwdTask.append((_executeNodeBwdComputeProcess, iNode.QSID, iPath, [], context.getUpdateData(node_id_list=[iNode.QSID]), iLocalData))
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
                        self._QS_Logger.error(f"backward_compute 计算失败: {e}")
                        raise e
                    else:
                        context.updateContext(iUpdateData)
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
                        iTask = (_executeNodeBwdComputeProcess, iParentNode.QSID, iParentPath.split("/"), Path2BwdDataList.pop(iParentPath), context.getUpdateData(node_id_list=[iParentNode.QSID]), Path2LocalData[iParentPath])
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