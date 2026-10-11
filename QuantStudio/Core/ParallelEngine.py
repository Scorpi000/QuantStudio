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
    SkipMode = task.get("SkipMode", False)
    if SkipMode:
        Context.ExtraData["_QS_FailSkip"] = True
    for i, iNode in enumerate(NodeList):
        try:
            if Context.NodeState.get(iNode.QSID, {}).get("__status__") in ("FAILED", "SKIPPED"):
                iRslt = None
            else:
                iRslt = iNode.compute([iNode.QSID], FwdDataList[i], Context)
            if iNode.QSID in Context.NodeErrors:
                Context.Sub2MainQueue.put(("CalcError", task["PID"], (1, iNode.QSID, Context.NodeErrors[iNode.QSID])))
            else:
                Context.Sub2MainQueue.put(("Calc", task["PID"], (1, iNode.QSID, iRslt)))
        except Exception as e:
            if SkipMode:
                Context.NodeState.setdefault(iNode.QSID, {})["__status__"] = "FAILED"
                Context.NodeErrors[iNode.QSID] = e
                Context.Logger.error(f"节点 {iNode.Name}({iNode.QSID[:8]}) 计算失败: {e}")
                Context.Sub2MainQueue.put(("CalcError", task["PID"], (1, iNode.QSID, e)))
            else:
                Context.Sub2MainQueue.put(("TaskError", task["PID"], e))
                return
    statuses = {qsid: state["__status__"] for qsid, state in Context.NodeState.items()
                if state.get("__status__") in ("FAILED", "SKIPPED")}
    Context.Sub2MainQueue.put(("Done", task["PID"], (Context.getUpdateData(), Context.NodeErrors, statuses)))
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
        if not node_list:
            return []
        SkipMode = self._QSArgs.FailMode == "skip"
        nTask = len(context.PIDList)
        SplitedContext = context.split(nTask)
        SplitedFwdDataList = zip(*[(FwdData.split(nTask, context) if hasattr(FwdData, "split") else [FwdData] * nTask) for FwdData in fwd_data_list])
        Procs = {}
        completed = False
        try:
            for i, iFwdDataList in enumerate(SplitedFwdDataList):
                iPID = context.PIDList[i]
                iTask = {"PID": iPID, "NodeList": node_list, "Context": SplitedContext[i], "FwdDataList": iFwdDataList, "SkipMode": SkipMode}
                Procs[iPID] = Process(target=_execute_task, args=(iTask,))
                Procs[iPID].start()

            Sub2MainQueue = context.Sub2MainQueue
            nProg = len(node_list) * nTask
            EventState = {iNodeID: 0 for iNodeID in context.Event}
            iProg, ContextUpdated = 0, False
            FinishedPIDs = set()
            Data = {}
            with ProgressBar(max_value=nProg) as ProgBar:
                while (iProg < nProg) or (len(FinishedPIDs) < nTask):
                    # SimpleQueue 不支持超时读取，轮询管道以便检查子进程退出。
                    if not Sub2MainQueue._reader.poll(0.1):
                        for pid, proc in Procs.items():
                            if (proc.exitcode is not None and pid not in FinishedPIDs
                                    and not Sub2MainQueue._reader.poll()):
                                raise __QS_Error__(f"子进程 {pid} 未发送结束消息即退出，退出码为 {proc.exitcode}")
                        continue
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
                    elif iMsgType == "CalcError":# 单个节点计算失败 (skip 模式)
                        iSubProg, iNodeID, iErr = iMsg
                        iProg += iSubProg
                        ProgBar.update(iProg)
                        context.NodeErrors.setdefault(iNodeID, iErr)
                        Data.setdefault(iNodeID, []).append(None)
                    elif iMsgType == "Done":# 子进程结束
                        if not ContextUpdated:
                            context.updateContext(iMsg[0])
                            ContextUpdated = True
                        context.NodeErrors.update(iMsg[1])
                        for qsid, status in iMsg[2].items():
                            context.NodeState.setdefault(qsid, {})["__status__"] = status
                        FinishedPIDs.add(iPID)
                    elif iMsgType == "TaskError":
                        raise iMsg
                    else:
                        raise __QS_Error__(f"无法识别的消息类型: {iMsgType}")
        
            completed = True
        finally:
            for proc in Procs.values():
                if proc.pid is not None:
                    if not completed and proc.is_alive():
                        proc.terminate()
                    proc.join()
            if not completed:
                # 异常退出后丢弃未消费的消息，避免污染下一次运行。
                context.Sub2MainQueue.close()
                context.Sub2MainQueue = None
        return [(None if context.NodeState.get(iNode.QSID, {}).get("__status__") in ("FAILED", "SKIPPED") else
                 iNode.merge_result(result_list=Data[iNode.QSID], context=context)) for iNode in node_list]
