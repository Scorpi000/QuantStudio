# -*- coding: utf-8 -*-
"""计算引擎模块: 提供顺序执行和栈式执行两种计算图执行策略.

核心类:
    Engine: 顺序计算引擎, 按 init → prepare → compute 三阶段执行计算图
    StackEngine: 栈式引擎, 通过 forward → backward 两阶段深度优先遍历执行计算

全局变量:
    __QS_Engine__: 全局引擎栈, 支持嵌套使用 (通过上下文管理器压入/弹出)
"""
import time
import concurrent.futures
from typing import Any, List, Optional

from pydantic import Field
from progressbar import ProgressBar

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Node import Node, Context


class Engine(__QS_Object__):
    """顺序计算引擎, 按拓扑顺序依次执行计算图的初始化、准备和计算三个阶段.

    计算流程:
        1. init: 从给定节点列表出发, BFS 遍历依赖树, 注册节点到上下文并递归调用 init_compute
        2. prepare: 对上下文中注册的节点执行 prepare_compute, 支持线程池并发 IO
        3. compute: 对给定节点列表执行 compute, 返回计算结果

    支持作为上下文管理器使用, 进入时自动压入全局引擎栈.
    """

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        """Engine 的参数配置类.

        Attributes:
            IOConcurrentNum: IO 并发数, None 表示不限制上限, 根据待准备节点数量自动决定并发度.
        """
        IOConcurrentNum: Optional[int] = Field(default=None, title="IO并发数", frozen=True, ge=1, description="在准备计算阶段, 允许的最大IO并发数, None 不限制上限, 根据待准备节点数量自动决定并发度")

    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        """初始化计算图: BFS 遍历依赖树, 注册节点并递归初始化.

        从给定节点列表开始, 广度优先遍历每个节点的依赖 (Deps), 将节点注册到上下文的
        NodeDict 中, 并调用每个节点的 init_compute 方法。若某节点的 init_compute 返回了
        初始化数据列表, 则将其依赖节点加入遍历队列, 实现递归初始化。

        当因子检测到不同的 SectionIDs 时, 会创建因子变体 (具有不同的 QSID) 并注册到计算图中。

        Args:
            node_list: 待初始化的节点列表 (计算目标)
            context: 全局上下文, 初始化后的节点将注册到 context.NodeDict 中
            init_data_list: 与 node_list 一一对应的初始化数据列表, None 时使用默认值
        """
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
        # 处理因子 SectionIDs 变体
        self._processSectionIDVariants(context)

    def _processSectionIDVariants(self, context: Context):
        """处理因子 SectionIDs 变体: 为每个变体创建新的因子对象并初始化.

        当因子的 init_compute 检测到不同的 SectionIDs 时, 会将变体信息存储在
        context._QS_FactorSectionIDVariants 中。本方法处理这些变体, 创建新的因子对象,
        初始化它们的依赖, 并注册到计算图中。

        变体处理会循环执行, 直到没有新的变体产生 (因为变体的依赖可能也需要创建变体)。

        Args:
            context: 全局上下文
        """
        while True:
            Variants = context.pop("_QS_FactorSectionIDVariants", [])
            if not Variants:
                break
            for VariantKey, VariantInfo in Variants:
                Factor = VariantInfo["factor"]
                NewSectionIDs = VariantInfo["section_ids"]
                DTRange = VariantInfo["dt_range"]
                Path = VariantInfo["path"]
                InitData = VariantInfo["init_data"]
                # 创建变体因子
                VariantFactor = Factor._createSectionIDVariant(NewSectionIDs)
                VariantQSID = VariantFactor.QSID
                # 如果变体已经存在, 跳过
                if VariantQSID in context.NodeDict:
                    continue
                # 注册变体因子
                context.NodeDict[VariantQSID] = VariantFactor
                # 初始化变体因子的依赖
                # 变体因子与原因子共享相同的依赖结构, 但需要使用新的 SectionIDs 初始化
                self._initVariantDeps(VariantFactor, NewSectionIDs, DTRange, Path, context)
                # 初始化变体因子本身
                VariantFactorState = context.NodeState.setdefault(VariantQSID, {})
                VariantFactorState["dt_range"] = DTRange
                VariantFactorState["section_ids"] = NewSectionIDs
                if NewSectionIDs == context.SectionIDs:
                    VariantFactorState["pid_ids"] = context.DefaultPIDIDs
                else:
                    VariantFactorState["pid_ids"] = context.splitID(NewSectionIDs)
                # 处理变体因子的因子表
                if VariantFactor._FactorTable:
                    FactorTable = VariantFactor._FactorTable
                    PrepareID = FactorTable.PrepareID
                    if PrepareID and PrepareID in context.PrepareNodeDict:
                        _, PrepareData = context.PrepareNodeDict[PrepareID]
                        PrepareData["SectionIDs"] = sorted(set(PrepareData["SectionIDs"] + NewSectionIDs))

    def _initVariantDeps(self, factor: "Node", section_ids: List[str], dt_range: tuple, path: List[str], context: Context):
        """初始化变体因子的依赖节点.

        为变体因子的每个依赖创建对应的变体 (如果需要), 并递归初始化。

        Args:
            factor: 变体因子
            section_ids: 新的截面ID列表
            dt_range: 时点范围
            path: 初始化路径
            context: 全局上下文
        """
        from QuantStudio.Factor.Factor import FactorInitData
        # 创建用于初始化依赖的 InitData
        DepInitData = FactorInitData(DTRange=dt_range, SectionIDs=section_ids)
        # 递归初始化依赖
        for iDep in factor.Deps:
            if iDep.QSID in context.NodeState:
                # 依赖已经初始化过, 检查 SectionIDs 是否一致
                DepState = context.NodeState[iDep.QSID]
                DepSectionIDs = DepState.get("section_ids")
                if DepSectionIDs != section_ids:
                    # SectionIDs 不一致, 需要为依赖创建变体
                    if hasattr(iDep, '_createSectionIDVariant'):
                        VariantDep = iDep._createSectionIDVariant(section_ids)
                        if VariantDep.QSID not in context.NodeDict:
                            context.NodeDict[VariantDep.QSID] = VariantDep
                            # 递归初始化变体依赖的依赖
                            self._initVariantDeps(VariantDep, section_ids, dt_range, path + [VariantDep.QSID], context)
                            # 初始化变体依赖本身
                            VariantDepState = context.NodeState.setdefault(VariantDep.QSID, {})
                            VariantDepState["dt_range"] = dt_range
                            VariantDepState["section_ids"] = section_ids
                            if section_ids == context.SectionIDs:
                                VariantDepState["pid_ids"] = context.DefaultPIDIDs
                            else:
                                VariantDepState["pid_ids"] = context.splitID(section_ids)
            else:
                # 依赖未初始化, 直接初始化
                context.NodeDict[iDep.QSID] = iDep
                iDepInitDataList = iDep.init_compute(path=path + [iDep.QSID], init_data=DepInitData, context=context)
                if iDepInitDataList:
                    self._initVariantDeps(iDep, section_ids, dt_range, path + [iDep.QSID], context)

    def prepare(self, node_list: List[Node], context: Context):
        """准备计算数据: 对上下文中已注册的节点执行 prepare_compute.

        遍历 context.PrepareNodeDict 中的所有节点 (由 init 阶段注册), 逐个调用
        prepare_compute。当 IOConcurrentNum > 1 时, 使用 ThreadPoolExecutor 并发执行
        IO 操作, 并通过进度条展示完成进度。

        Args:
            node_list: 节点列表 (保留参数, 当前未直接使用)
            context: 全局上下文, 包含 PrepareNodeDict 记录待准备节点的信息
        """
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

    def compute(self, node_list: List[Node], context: Context, fwd_data_list: Optional[List[Any]]=None):
        """执行正式计算: 对每个目标节点调用 compute 方法并收集结果.

        Args:
            node_list: 待计算的节点列表
            context: 全局上下文
            fwd_data_list: 与 node_list 一一对应的前向计算输入数据, None 时使用默认值

        Returns:
            List[Any]: 每个节点 compute 的返回值列表, 顺序与 node_list 一致
        """
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
        if init_data_list is None: init_data_list = [None] * len(node_list)
        if fwd_data_list is None: fwd_data_list = [None] * len(node_list)
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
        """上下文管理器入口: 将当前引擎实例压入全局引擎栈, 支持嵌套使用."""
        __QS_Engine__.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器出口: 从全局引擎栈弹出当前引擎实例.

        Args:
            exc_type: 异常类型 (若有)
            exc_value: 异常值 (若有)
            traceback: 异常回溯 (若有)

        Note:
            不抑制任何异常, 仅清理引擎栈.
        """
        if __QS_Engine__: __QS_Engine__.pop()


class StackEngine(Engine):
    """栈式计算引擎, init 和 compute 阶段均采用深度优先遍历.

    与 Engine 的 BFS 拓扑序遍历不同, StackEngine 全流程使用 DFS:
        1. init: DFS 遍历依赖树, 注册节点并递归初始化, 顺序与 forward 阶段一致
        2. prepare: 复用 Engine.prepare, 对上下文中的节点执行 IO 准备
        3. 前向计算 (forward): 从目标节点出发沿依赖链 DFS 向下, 收集每层的局部上下文
        4. 后向计算 (backward): 从依赖链底端向上回溯, 将子节点结果逐层传递给父节点
    """

    def init(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None):
        """初始化计算图: DFS 遍历依赖树, 注册节点并递归初始化.

        使用栈进行深度优先遍历, 遍历顺序与 forward_compute 的 DFS 顺序一致:
        从目标节点出发, 沿第一个依赖深入到底后再处理其他分支.

        Args:
            node_list: 待初始化的节点列表 (计算目标)
            context: 全局上下文, 初始化后的节点将注册到 context.NodeDict 中
            init_data_list: 与 node_list 一一对应的初始化数据列表, None 时使用默认值
        """
        if init_data_list is None: init_data_list = [None] * len(node_list)
        # 反转初始列表, 使得 node_list[0] 优先出栈处理
        NodeStack = node_list.copy()[::-1]
        InitDataStack = init_data_list[::-1]
        PathStack = [[iNode.QSID] for iNode in node_list][::-1]
        while NodeStack:
            iNode = NodeStack.pop()
            iPath = PathStack.pop()
            context.NodeDict[iNode.QSID] = iNode
            iInitDataList = iNode.init_compute(path=iPath, init_data=InitDataStack.pop(), context=context)
            if iInitDataList:
                # 反转 Deps 入栈: 第一个依赖最后压入 → 最先弹出 → 优先深探
                NodeStack += iNode.Deps[::-1]
                InitDataStack += iInitDataList[::-1]
                PathStack += [iPath + [iDep.QSID] for iDep in iNode.Deps][::-1]

    def run(self, node_list: List[Node], context: Context, init_data_list: Optional[List[Any]]=None, fwd_data_list: Optional[List[Any]]=None) -> List[Any]:
        """执行栈式计算: init (DFS) → prepare → forward (DFS) → backward.

        Args:
            node_list: 待计算的节点列表
            context: 全局上下文
            init_data_list: 与 node_list 一一对应的初始化数据列表
            fwd_data_list: 与 node_list 一一对应的前向计算输入数据列表

        Returns:
            List[Any]: 每个节点的 backward_compute 返回值列表, 顺序与 node_list 一致
        """
        self._QS_Logger.info("开始初始化计算...")
        StartT = time.perf_counter()
        self.init(node_list=node_list, context=context, init_data_list=init_data_list)
        self._QS_Logger.info(f"初始化计算完成, 耗时 {time.perf_counter() - StartT} 秒")
        self._QS_Logger.info("开始准备计算...")
        StartT = time.perf_counter()
        self.prepare(node_list=node_list, context=context)
        self._QS_Logger.info(f"准备计算完成, 耗时 {time.perf_counter() - StartT} 秒")
        self._QS_Logger.info("开始正式计算 (forward/backward)...")
        StartT = time.perf_counter()
        if not fwd_data_list: fwd_data_list = [None] * len(node_list)
        Rslt = []
        for i, iNode in enumerate(node_list):
            NodeStack, PathStack, LocalContextStack, DataStack = [], [], [], []
            iNodeList, iPathList, iFwdDataList = [iNode], [[]], [fwd_data_list[i]]
            # 前向计算: 从目标节点 DFS 遍历依赖树, 收集每层的局部上下文
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
                assert len(ijFwdDataList) == len(ijNode.Deps), f"节点 {ijNode.Name} 的 forward_compute 返回数据数量 ({len(ijFwdDataList)}) 与依赖数量 ({len(ijNode.Deps)}) 不一致"
                iNodeList = ijNode.Deps + iNodeList
                iFwdDataList = ijFwdDataList + iFwdDataList
                iPathList = [iDepNode.QSID for iDepNode in ijNode.Deps] + iPathList
            assert (len(iFwdDataList) == 0) and (len(iPathList) == 0), "前向计算结束时应无剩余未处理数据"
            # 后向计算: 从依赖链底端向上回溯, 子节点结果传递给父节点
            for j in range(len(NodeStack) - 1, -1, -1):
                ijNode, ijPath = NodeStack[j], PathStack[j]
                ijContext, ijTerminated = LocalContextStack[j]
                if ijTerminated:
                    ijDepData = []
                else:
                    nDeps = len(ijNode.Deps)
                    ijDepData, DataStack = DataStack[len(DataStack) - nDeps:][::-1], DataStack[:len(DataStack) - nDeps]
                ijBwdData = ijNode.backward_compute(path=ijPath, bwd_data_list=ijDepData, context=context, local_context=ijContext)
                DataStack.append(ijBwdData)
            assert len(DataStack) == 1, f"后向计算结束后 DataStack 长度应为 1, 实际为 {len(DataStack)}"
            iRslt = DataStack.pop()
            Rslt.append(iRslt)
        self._QS_Logger.info(f"正式计算完成, 耗时 {time.perf_counter() - StartT} 秒")
        return Rslt


# 全局计算引擎栈: 默认包含一个 Engine 实例, 支持通过 with 语句压入自定义引擎
# 使用场景: with StackEngine() as engine: engine.run(...)
__QS_Engine__ = [Engine()]