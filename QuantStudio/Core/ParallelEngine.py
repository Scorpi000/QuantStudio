# -*- coding: utf-8 -*-
import os
from typing import Any, List, Optional

from progressbar import ProgressBar
from multiprocess import Process

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Node import Node, Context
from multiprocess import SimpleQueue as Queue
from QuantStudio.Core.CalcEngine import Engine

def _execute_task(task):
    NodeList, Context, FwdDataList = task["NodeList"], task["Context"], task["FwdDataList"]
    Context.Logger.debug(f'子任务进程 {task["PID"]} start, PID: {os.getpid()}')
    Context.PID = task["PID"]
    for i, iNode in enumerate(NodeList):
        iRslt = iNode.compute([iNode.QSID], FwdDataList[i], Context)
        Context.Sub2MainQueue.put(("Calc", task["PID"], (1, iNode.QSID, iRslt)))
    Context.Sub2MainQueue.put(("Done", task["PID"], Context.getUpdateData()))
    Context.Logger.debug(f'子任务进程 {task["PID"]} finish')

class ParallelEngine(Engine):

    def run(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None, init_data_list: Optional[List[Any]]=None) -> List[Any]:
        """执行并行计算: init (DFS) → prepare → forward (DFS) → backward.
        
        Args:
            node_list: 待计算的节点列表
            context: 全局上下文
            fwd_data_list: 与 node_list 一一对应的前向计算输入数据列表
            init_data_list: 与 node_list 一一对应的初始化数据列表, 如果为 None 则使用 fwd_data_list
        
        Returns:
            List[Any]: 每个节点的 backward_compute 返回值列表, 顺序与 node_list 一致
        """
        # self._MP_Manager = context.ExtraData["mp_manager"] = Manager()
        if context.Sub2MainQueue is None: context.Sub2MainQueue = Queue()
        Rslt = super().run(node_list, context, fwd_data_list, init_data_list)
        # self._MP_Manager.shutdown()
        return Rslt

    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if len(node_list) != len(fwd_data_list): raise __QS_Error__("node_list 和 fwd_data_list 长度不一致!")
        # context.ExtraData.pop("mp_manager")
        nTask = len(context.PIDList)
        SplitedContext = context.split(nTask)
        SplitedFwdDataList = zip(*[(FwdData.split(nTask, context) if hasattr(FwdData, "split") else [FwdData] * nTask) for FwdData in fwd_data_list])
        Procs = {}
        for i, iFwdDataList in enumerate(SplitedFwdDataList):
            iPID = context.PIDList[i]
            iTask = {"PID": iPID, "NodeList": node_list, "Context": SplitedContext[i], "FwdDataList": iFwdDataList}
            Procs[iPID] = Process(target=_execute_task, args=(iTask,))
            Procs[iPID].start()
        
        Sub2MainQueue = context.Sub2MainQueue
        nProg = len(node_list) * nTask
        EventState = {iNodeID: 0 for iNodeID in context.Event}
        iProg, ContextUpdated, FinishedNum = 0, False, 0
        Data = {}
        with ProgressBar(max_value=nProg) as ProgBar:
            while (iProg < nProg) or (not ContextUpdated):
                iMsgType, iPID, iMsg = Sub2MainQueue.get()
                if iMsgType == "Event":# 同步事件
                    iNodeID, iInc = iMsg
                    EventState[iNodeID] += iInc
                    if EventState[iNodeID] >= nTask:
                        context.Event[iNodeID].set()
                elif iMsgType == "Calc":# 单个节点计算完成
                    iSubProg, iNodeID, iRslt = iMsg
                    iProg += iSubProg
                    ProgBar.update(iProg)
                    Data.setdefault(iNodeID, []).append(iRslt)
                elif iMsgType == "Done":# 子进程结束
                    if not ContextUpdated:
                        context.updateContext(iMsg)
                        ContextUpdated = True
                    FinishedNum += 1
                else:
                    raise __QS_Error__(f"无法识别的消息类型: {iMsgType}")
        
        # 清空 Queue，否则子进程有可能不退出
        while FinishedNum < nTask:
            iMsgType, iPID, iMsg = Sub2MainQueue.get()
            FinishedNum += (iMsgType == "Done")
        for iPID, iPrcs in Procs.items(): iPrcs.join()
        return [iNode.merge_result(result_list=Data[iNode.QSID], context=context) for i, iNode in enumerate(node_list)]
