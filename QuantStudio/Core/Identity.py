# -*- coding: utf-8 -*-
"""计算语义、执行请求和运行实例的身份约定。"""
from enum import Enum
from uuid import uuid4

from QuantStudio.Tools.DataTypeFun import dict2id


class ExecutionEffect(str, Enum):
    """影响调度去重及缓存范围的执行特征。"""

    PURE = "pure"
    READ = "read"
    STATEFUL = "stateful"
    WRITE = "write"


def new_instance_id():
    """生成独立于计算语义的实例标识。"""
    return uuid4().hex


def execution_key(qsid, request, data_version=None):
    """根据完整请求和数据版本生成结果身份。

    Args:
        qsid: 完整节点的计算语义身份。
        request: 时点、截面等影响结果的请求配置。
        data_version: 数据快照版本；空值不承诺跨运行缓存有效。

    Returns:
        确定性的执行结果标识。
    """
    return dict2id({"node": qsid, "request": request, "data_version": data_version})
