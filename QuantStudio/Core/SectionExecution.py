# -*- coding: utf-8 -*-
"""完整截面节点的日期分工、结构化结果交换与进程内复用。"""
import pandas as pd
from multiprocess import Event

from .FileCache import FileDTCache
from QuantStudio.Tools.AuxiliaryFun import partitionList


def section_state(node, ids, context):
    """按节点、截面、运行和数据版本隔离执行状态。"""
    key = node.execution_key({"instance": node.InstanceID, "section": tuple(ids), "run": context.RunID}, context.DataVersion)
    states = context.NodeState.setdefault(node.InstanceID, {}).setdefault("section_execution", {})
    return key, states.setdefault(key, {"results": {}})


def init_section(node, init_data, ids, context):
    """汇总计算区间，并在主进程为共享结果注册同步事件。"""
    if not context.PIDList or len(set(context.PIDList)) != len(context.PIDList):
        raise ValueError("执行进程列表必须非空且不能重复")
    key, state = section_state(node, ids, context)
    previous = state.get("range", init_data.DTRange)
    state["range"] = (min(previous[0], init_data.DTRange[0]), max(previous[1], init_data.DTRange[1]))
    if len(context.PIDList) > 1:
        if not isinstance(context.DataCache, FileDTCache) or not context.DataCache._isStarted:
            raise ValueError("并行完整截面节点需要已启动的文件共享缓存")
        if key not in context.Event:
            context.Event[key] = Event()
    return state["range"]


def section_plan(node, ids, context, *, ordered=False):
    """生成共同日期范围与本进程负责的日期，连续账户只交给首进程。"""
    event, state = section_state(node, ids, context)
    dates = context.getDateTime(state["range"])
    key = node.execution_key({"instance": node.InstanceID, "section": tuple(ids), "dates": dates, "run": context.RunID},
                             context.DataVersion)
    if len(context.PIDList) == 1:
        assigned = dates
    elif ordered:
        assigned = dates if context.PID == context.PIDList[0] else []
    else:
        assigned = partitionList(dates, len(context.PIDList))[context.PIDList.index(context.PID)]
    return {"event": event, "key": key, "dates": dates, "assigned": assigned, "ids": tuple(ids)}


def cached_section(node, plan, context):
    """读取本进程已经交换完成的完整结果。"""
    return section_state(node, plan["ids"], context)[1]["results"].get(plan["key"])


def publish_section(node, plan, result, merge, context):
    """发布本进程结果，同步后读取所有分片并只在本次运行内复用。

    Args:
        node: 当前完整截面节点。
        plan: section_plan 返回的任务分工。
        result: 当期分片；非账户拥有进程可发布 None。
        merge: 接收按 PID 顺序排列的分片并返回完整结果的函数。
        context: 运行上下文及共享缓存。

    Returns:
        可供后续多个出口共享的完整结果。
    """
    if len(context.PIDList) > 1:
        index = context.PIDList.index(context.PID)
        payload = pd.DataFrame(index=[0], columns=["result"], dtype=object)
        payload.at[0, "result"] = result
        key = lambda number: f"section-{plan['key']}-{number}"
        context.DataCache.writeData(key(index), payload, if_exists="replace", data_type="object")
        context.Sub2MainQueue.put(("Event", context.PID, (plan["event"], 1)))
        context.wait_for_event(plan["event"], node)
        parts = [context.DataCache.readData(key(number), data_type="object").iloc[0, 0]
                 for number in range(len(context.PIDList))]
        result = merge(parts)
    section_state(node, plan["ids"], context)[1]["results"][plan["key"]] = result
    return result
