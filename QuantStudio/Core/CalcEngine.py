# -*- coding: utf-8 -*-
import time
import concurrent.futures
from typing import Any, List, Optional

from pydantic import Field
from progressbar import ProgressBar

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Node import Node, Context


class Engine(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        IOConcurrentNum: Optional[int] = Field(default=None, title="IO并发数", frozen=True, ge=1)

    # 初始化
    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        if init_data_list is None: init_data_list = [None] * len(node_list)
        NodeQ, InitDataQ, PathQ = node_list.copy(), init_data_list, [[iNode.QSID] for iNode in node_list]
        while NodeQ:
            iNode, iPath = NodeQ.pop(0), PathQ.pop(0)
            context.NodeDict[iNode.QSID] = iNode
            iInitDataList = iNode.init_compute(path=iPath, init_data=InitDataQ.pop(0), context=context)
            if iInitDataList:
                NodeQ += iNode.Deps
                InitDataQ += iInitDataList
                PathQ += [iPath + [iDep.QSID] for iDep in iNode.Deps]

    # 准备数据
    def prepare(self, node_list: List[Node], context: Context):
        if not context.PrepareNodeDict: return
        IOConcurrentNum = (self._QSArgs.IOConcurrentNum if self._QSArgs.IOConcurrentNum is not None else len(context.PrepareNodeDict))
        if IOConcurrentNum <= 1:
            for _, iPrepareData in context.PrepareNodeDict.items():
                iNodeID, iPrepareData = iPrepareData
                context.NodeDict[iNodeID].prepare_compute(iPrepareData, context)
        else:
            Futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=IOConcurrentNum) as Executor:
                for _, iPrepareData in context.PrepareNodeDict.items():
                    iNodeID, iPrepareData = iPrepareData
                    iFuture = Executor.submit(context.NodeDict[iNodeID].prepare_compute, iPrepareData, context)
                    Futures.append(iFuture)
                with ProgressBar(max_value=len(Futures)) as ProgBar:
                    for iFuture in concurrent.futures.as_completed(Futures):
                        iFuture.result()
                        ProgBar.update(ProgBar.value + 1)

    # 主计算
    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        if not fwd_data_list: fwd_data_list = [None] * len(node_list)
        return [iNode.compute([iNode.QSID], fwd_data_list[i], context) for i, iNode in enumerate(node_list)]

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