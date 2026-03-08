# -*- coding: utf-8 -*-
import os
import time
import queue
import concurrent.futures
from multiprocessing import Process, Queue, Manager
from typing import Any, List, Optional

from pydantic import Field
from progressbar import ProgressBar

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Node import Node, Context


class Engine(__QS_Object__):

    # 初始化
    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        if not init_data_list: init_data_list = [None] * len(node_list)
        for i, iNode in enumerate(node_list):
            iNode.init([], init_data_list[i], context)

    # 准备数据
    def prepare(self, node_list: List[Node], context: Context):
        for _, iPrepareData in context.PrepareNodeDict.items():
            iNodeID, iPrepareData = iPrepareData
            context.NodeDict[iNodeID].prepare_compute(iPrepareData, context)

    # 主计算
    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if not fwd_data_list: fwd_data_list = [None] * len(node_list)
        return [iNode.compute([], fwd_data_list[i], context) for i, iNode in enumerate(node_list)]

    def run(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None, fwd_data_list: Optional[List[Any]]=None) -> List[Any]:
        """给定节点列表, 执行所有节点的计算, 返回每个节点的计算结果

        Args:
            node_list: 待计算的节点列表
            context: 全局上下文对象
            init_data_list: 初始化数据列表
            fwd_data_list: 前向计算输入数据列表
        
        Returns:
            节点计算的结果列表
        """
        self._QS_Logger.info("开始初始化计算...")
        StartT = time.perf_counter()
        self.init(node_list=node_list, context=context, init_data_list=init_data_list)
        self._QS_Logger.info(f"初始化计算完成, 耗时 {time.perf_counter() - StartT} 秒")
        self._QS_Logger.info("开始准备计算...")
        StartT = time.perf_counter()
        self.prepare(node_list=node_list, context=context)
        self._QS_Logger.info(f"准备计算完成, 耗时 {time.perf_counter() - StartT} 秒")
        self._QS_Logger.info("开始正式计算...")
        StartT = time.perf_counter()
        Rslt = self.compute(node_list=node_list, context=context, fwd_data_list=fwd_data_list)
        self._QS_Logger.info(f"正式计算完成, 耗时 {time.perf_counter() - StartT} 秒")
        return Rslt

    def __enter__(self):
        __QS_Engine__.append(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if __QS_Engine__: __QS_Engine__.pop()

def _execute_task(task):
    NodeList, Context, FwdDataList = task["NodeList"], task["Context"], task["FwdDataList"]
    Context.Logger.info(f'子任务进程 {task["PID"]} start, PID: {os.getpid()}')
    Context.PID = task["PID"]
    for i, iNode in enumerate(NodeList):
        iRslt = iNode.compute([], FwdDataList[i], Context)
        task["Sub2MainQueue"].put((task["PID"], 1, (iNode.QSID, iRslt)))
    task["Sub2MainQueue"].put((task["PID"], -1, Context.getUpdateData()))
    Context.Logger.info(f'子任务进程 {task["PID"]} finish')


class ParallelEngine(Engine):

    class __QS_ArgClass__(Engine.__QS_ArgClass__):
        IOConcurrentNum: Optional[int] = Field(default=None, title="IO并发数", frozen=True, ge=1)

    # 初始化
    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        if os.name=="nt": self._MP_Manager = context.ExtraData["mp_manager"] = Manager()
        Rslt = super().init(node_list=node_list, context=context, init_data_list=init_data_list)
        if os.name=="nt": context.ExtraData.pop("mp_manager")
        return Rslt

    def prepare(self, node_list: List[Node], context: Context):
        if not context.PrepareNodeDict: return
        Futures = []
        IOConcurrentNum = (self._QSArgs.IOConcurrentNum if self._QSArgs.IOConcurrentNum is not None else len(context.PrepareNodeDict))
        with concurrent.futures.ThreadPoolExecutor(max_workers=IOConcurrentNum) as Executor:
            for _, iPrepareData in context.PrepareNodeDict.items():
                iNodeID, iPrepareData = iPrepareData
                iFuture = Executor.submit(context.NodeDict[iNodeID].prepare_compute, iPrepareData, context)
                Futures.append(iFuture)
            with ProgressBar(max_value=len(Futures)) as ProgBar:
                for iFuture in concurrent.futures.as_completed(Futures):
                    iFuture.result()
                    ProgBar.update(ProgBar.value + 1)

    def _safe_queue_empty(self, q):
        """跨平台安全的队列空检查"""
        try:
            # Windows: 尝试非阻塞获取再放回
            item = q.get_nowait()
        except queue.Empty:
            return True
        else:
            q.put(item)# 放回去
            return False

    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if len(node_list) != len(fwd_data_list): raise __QS_Error__("node_list 和 fwd_data_list 长度不一致!")
        nTask = len(context.PIDList)
        SplitedContext = context.split(nTask)
        SplitedFwdDataList = zip(*[(FwdData.split(nTask, context) if hasattr(FwdData, "split") else [FwdData] * nTask) for FwdData in fwd_data_list])
        if os.name=="nt":
            Sub2MainQueue = self._MP_Manager.Queue()
        else:
            Sub2MainQueue = Queue()
        Procs = {}
        for i, iFwdDataList in enumerate(SplitedFwdDataList):
            iPID = context.PIDList[i]
            iTask = {"PID": iPID, "NodeList": node_list, "Context": SplitedContext[i], "FwdDataList": iFwdDataList, "Sub2MainQueue": Sub2MainQueue}
            Procs[iPID] = Process(target=_execute_task, args=(iTask,))
            Procs[iPID].start()
        
        nProg = len(node_list) * nTask
        EventState = {iNodeID: 0 for iNodeID in context.Event}
        iProg, ContextUpdated, FinishedNum = 0, False, 0
        Data = {}
        with ProgressBar(max_value=nProg) as ProgBar:
            while True:
                nEvent = len(EventState)
                if nEvent > 0:
                    NodeIDs = tuple(EventState.keys())
                    for iNodeID in NodeIDs:
                        iQueue = context.Event[iNodeID][0]
                        # while not self._safe_queue_empty(iQueue):
                        while not iQueue.empty():
                            jInc = iQueue.get()
                            EventState[iNodeID] += jInc
                        if EventState[iNodeID] >= nTask:
                            context.Event[iNodeID][1].set()
                            EventState.pop(iNodeID)
                # while ((not self._safe_queue_empty(Sub2MainQueue)) or (nEvent == 0)) and ((iProg < nProg) or (not ContextUpdated)):
                while ((not Sub2MainQueue.empty()) or (nEvent == 0)) and ((iProg < nProg) or (not ContextUpdated)):
                    iPID, iSubProg, iMsg = Sub2MainQueue.get()
                    if iSubProg >= 0:# 接收到因子数据
                        iProg += iSubProg
                        ProgBar.update(iProg)
                        Data.setdefault(iMsg[0], []).append(iMsg[1])
                    elif not ContextUpdated:# 接收到进程结束信号
                        context.updateContext(iMsg)
                        ContextUpdated = True
                        FinishedNum += 1
                    else:
                        FinishedNum += 1
                if (iProg >= nProg) and ContextUpdated: break
        # 清空 Queue，否则子进程有可能不退出
        while FinishedNum < nTask:
            iPID, iSubProg, iMsg = Sub2MainQueue.get()
            FinishedNum += (iSubProg < 0)
        for iPID, iPrcs in Procs.items(): iPrcs.join()
        if os.name == "nt": self._MP_Manager.shutdown()
        return [iNode.merge_result(result_list=Data[iNode.QSID], context=context) for i, iNode in enumerate(node_list)]

class StackEngine(Engine):
    def run(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None, fwd_data_list: Optional[List[Any]]=None) -> List[Any]:
        if not init_data_list: init_data_list = [None] * len(node_list)
        for i, iNode in enumerate(node_list):
            iNode.init([], init_data_list[i], context)
        if not fwd_data_list: fwd_data_list = [None] * len(node_list)
        Rslt = []
        for i, iNode in enumerate(node_list):
            NodeStack, PathStack, LocalContextStack, DataStack = [], [], [], []
            iNodeList, iPathList, iFwdDataList = [iNode], [[]], [fwd_data_list[i]]
            # 前向计算
            while iNodeList:
                ijNode = iNodeList.pop(0)
                ijPath = iPathList.pop(0)
                ijFwdDataList, ijContext = ijNode.forward_compute(path=ijPath, fwd_data=iFwdDataList.pop(0), context=context)
                NodeStack.append(ijNode)
                PathStack.append(ijPath)
                if not ijFwdDataList:
                    LocalContextStack.append((ijContext, True))  # (LocalContext, 是否终止 forward)
                    continue
                else:
                    LocalContextStack.append((ijContext, False))
                assert len(ijFwdDataList) == len(ijNode.Deps)
                iNodeList = ijNode.Deps + iNodeList
                iFwdDataList = ijFwdDataList + iFwdDataList
                iPathList = [Node.QSID for Node in ijNode.Deps] + iPathList
            assert (len(iFwdDataList) == 0) and (len(iPathList) == 0)
            # 后向计算
            for j in range(len(NodeStack) - 1, -1, -1):
                ijNode, ijPath = NodeStack[j], PathStack[j]
                ijContext, ijTerminated = LocalContextStack[j]
                if ijTerminated:
                    ijDepData = []
                else:
                    ijDepData, DataStack = DataStack[len(DataStack) - len(ijNode.Deps):][::-1], DataStack[:len(DataStack) - len(ijNode.Deps)]
                ijBwdData = ijNode.backward_compute(path=ijPath, bwd_data_list=ijDepData, context=context, local_context=ijContext)
                DataStack.append(ijBwdData)
            assert len(DataStack) == 1
            iRslt = DataStack.pop()
            Rslt.append(iRslt)
        return Rslt


# 全局计算引擎
__QS_Engine__ = [Engine()]