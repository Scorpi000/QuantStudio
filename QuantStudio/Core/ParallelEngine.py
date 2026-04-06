# -*- coding: utf-8 -*-
import os
from typing import Any, List, Optional

from progressbar import ProgressBar
from multiprocess import Process

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Node import Node, Context
if os.name=="nt":
    from QuantStudio.Core.QSObject import QSQueue as Queue
else:
    from multiprocess import Queue
from QuantStudio.Core.CalcEngine import Engine

def _execute_task(task):
    NodeList, Context, FwdDataList = task["NodeList"], task["Context"], task["FwdDataList"]
    Context.Logger.info(f'子任务进程 {task["PID"]} start, PID: {os.getpid()}')
    Context.PID = task["PID"]
    for i, iNode in enumerate(NodeList):
        iRslt = iNode.compute([iNode.QSID], FwdDataList[i], Context)
        task["Sub2MainQueue"].put((task["PID"], 1, (iNode.QSID, iRslt)))
    task["Sub2MainQueue"].put((task["PID"], -1, Context.getUpdateData()))
    Context.Logger.info(f'子任务进程 {task["PID"]} finish')

class ParallelEngine(Engine):

    # 初始化
    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        # self._MP_Manager = context.ExtraData["mp_manager"] = Manager()
        Rslt = super().init(node_list=node_list, context=context, init_data_list=init_data_list)
        # context.ExtraData.pop("mp_manager")
        return Rslt

    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if len(node_list) != len(fwd_data_list): raise __QS_Error__("node_list 和 fwd_data_list 长度不一致!")
        nTask = len(context.PIDList)
        SplitedContext = context.split(nTask)
        SplitedFwdDataList = zip(*[(FwdData.split(nTask, context) if hasattr(FwdData, "split") else [FwdData] * nTask) for FwdData in fwd_data_list])
        # Sub2MainQueue = self._MP_Manager.Queue()
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
        # self._MP_Manager.shutdown()
        return [iNode.merge_result(result_list=Data[iNode.QSID], context=context) for i, iNode in enumerate(node_list)]
