# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, List, Optional, Literal

from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Tools.ProcessPoolExecutor import ProcessPoolExecutor



def _executeNodeBwdComputeThread(node: Node, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None):
    Result = node.backward_compute(path=path, bwd_data_list=bwd_data_list, context=context, local_context=local_context)
    return "/".join(path), Result, context.getUpdateData(node_id_list=[node.InstanceID])

__PROC_CONTEXT__: Context = None

def _initProcess(context):
    global __PROC_CONTEXT__
    __PROC_CONTEXT__ = context

def _executeNodeBwdComputeProcess(node_id: str, path: List[str], bwd_data_list: List[Any], updated_data: dict, local_context: Any=None):
    global __PROC_CONTEXT__
    __PROC_CONTEXT__.updateContext(updated_data)
    Node = __PROC_CONTEXT__.NodeDict[node_id]
    Result = Node.backward_compute(path=path, bwd_data_list=bwd_data_list, context=__PROC_CONTEXT__, local_context=local_context)
    return "/".join(path), Result, __PROC_CONTEXT__.getUpdateData(node_id_list=[Node.InstanceID])


class TreeEngine(Engine):
    """按依赖边的执行实例并发调度，重复引用不会覆盖其他槽位。"""

    class __QS_ArgClass__(Engine.__QS_ArgClass__):
        CalcConcurrentMode: Literal["Thread", "Process"] = Field(default="Process", title="并发模式", frozen=True)
        CalcConcurrentNum: Optional[int] = Field(default=None, title="计算并发数", frozen=True, ge=1)

    def compute(self, node_list, context, fwd_data_list=None):
        """先传播依赖请求，再按依赖完成顺序调度计算。

        Args:
            node_list: 完整执行目标。
            context: 已完成初始化及准备的运行上下文。
            fwd_data_list: 各目标的前向请求。

        Returns:
            按目标顺序组织的计算结果。
        """
        fwd_data_list = [None] * len(node_list) if fwd_data_list is None else fwd_data_list
        skip = self.Args.FailMode == "skip"
        jobs = {}
        queue = [(node, (index,), [node.InstanceID], fwd_data_list[index]) for index, node in enumerate(node_list)]
        while queue:
            node, token, path, request = queue.pop(0)
            job = jobs[token] = {"node": node, "path": path, "children": [], "done": False,
                                 "failed": False, "result": None, "local": None}
            if context.NodeState.get(node.InstanceID, {}).get("__status__") in ("FAILED", "SKIPPED"):
                job.update(done=True, failed=True)
                continue
            try:
                requests, local = node.forward_compute(path, request, context)
                job["local"] = local
            except Exception as error:
                if not skip:
                    raise
                self._fail(job, error, context)
                continue
            if requests:
                if len(requests) != len(node.Deps):
                    raise __QS_Error__("前向请求数量与依赖边数量不一致")
                for index, (dependency, value) in enumerate(zip(node.Deps, requests)):
                    child = token + (index,)
                    job["children"].append(child)
                    queue.append((dependency, child, path + [dependency.InstanceID], value))

        processes = self.Args.CalcConcurrentMode == "Process"
        if processes:
            executor = ProcessPoolExecutor(max_workers=self.Args.CalcConcurrentNum,
                                           initializer=_initProcess, initargs=(context,))
        else:
            executor = ThreadPoolExecutor(max_workers=self.Args.CalcConcurrentNum)
        futures, active = {}, set()
        capacity = self.Args.CalcConcurrentNum or 32
        with executor:
            while any(not job["done"] for job in jobs.values()):
                for token, job in jobs.items():
                    node = job["node"]
                    if job["done"] or token in futures.values() or node.InstanceID in active:
                        continue
                    children = [jobs[child] for child in job["children"]]
                    if not all(child["done"] for child in children):
                        continue
                    if any(child["failed"] for child in children):
                        job.update(done=True, failed=True)
                        context.NodeState.setdefault(node.InstanceID, {})["__status__"] = "SKIPPED"
                        context.emit(node, "compute", "skipped", request_id=token)
                        continue
                    if len(futures) >= capacity:
                        break
                    values = [child["result"] for child in children]
                    context.emit(node, "compute", "running", request_id=token)
                    if processes:
                        future = executor.submit(_executeNodeBwdComputeProcess, node.InstanceID, job["path"],
                                                 values, context.getUpdateData(node_id_list=[node.InstanceID]), job["local"])
                    else:
                        future = executor.submit(_executeNodeBwdComputeThread, node, job["path"], values, context, job["local"])
                    futures[future] = token
                    active.add(node.InstanceID)
                if not futures:
                    if all(job["done"] for job in jobs.values()):
                        break
                    raise __QS_Error__("依赖调度无法继续")
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    token = futures.pop(future)
                    job = jobs[token]
                    node = job["node"]
                    active.remove(node.InstanceID)
                    try:
                        _, result, updates = future.result()
                    except Exception as error:
                        if not skip:
                            raise
                        self._fail(job, error, context)
                    else:
                        context.updateContext(updates)
                        job.update(done=True, result=result)
                        context.emit(node, "compute", "done", request_id=token)
        return [jobs[(index,)]["result"] for index in range(len(node_list))]

    @staticmethod
    def _fail(job, error, context):
        node = job["node"]
        job.update(done=True, failed=True)
        context.NodeState.setdefault(node.InstanceID, {})["__status__"] = "FAILED"
        context.NodeErrors[node.InstanceID] = error
        context.emit(node, "compute", "failed", error=str(error))
