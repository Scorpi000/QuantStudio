# -*- coding: utf-8 -*-
from typing import Any, List, Optional

from QuantStudio.Core import __QS_Object__
from QuantStudio.Core.Node import Node, Context


class SimpleEngine(__QS_Object__):
    def run(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None, fwd_data_list: Optional[List[Any]]=None) -> List[Any]:
        # 初始化
        if not init_data_list: init_data_list = [None] * len(node_list)
        for i, iNode in enumerate(node_list):
            iNode.init([], init_data_list[i], context)
        # 准备计算
        for _, iPrepareData in context.PrepareNodeDict.items():
            iNodeID, iPrepareData = iPrepareData
            context.NodeDict[iNodeID].prepare_compute(iPrepareData, context)
        # 主计算
        if not fwd_data_list: fwd_data_list = [None] * len(node_list)
        return [iNode.compute([], fwd_data_list[i], context) for i, iNode in enumerate(node_list)]

class RecursiveEngine(__QS_Object__):
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
