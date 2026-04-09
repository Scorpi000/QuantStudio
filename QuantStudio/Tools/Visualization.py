# -*- coding: utf-8 -*-
import inspect
from collections import OrderedDict
from typing import Optional, Literal, List, Any

import pandas as pd
import numpy as np

from QuantStudio.Core import __QS_Object__, __QS_Args__
from QuantStudio.Core.Node import Node, Context, LocalContext
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Cache import Cache
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.FactorTable import FactorTable
from QuantStudio.Factor.FactorDB import FactorDB
from QuantStudio.Factor.FactorOperation import FactorOperator
from QuantStudio.Risk.RiskDB import RiskDB
from QuantStudio.Risk.RiskTable import RiskTable
from QuantStudio.BackTest.BackTestModel import BTNode


# ===================== 获取对象说明信息 =====================
def qs_help(obj: Any) -> str:
    """
    智能帮助函数，当对象的 __doc__ 为 None 时，自动查找父类方法的文档。
    
    支持：函数、方法、类、模块、内置类型等
    
    Returns:
        格式化的帮助文档字符串
    """
    # 获取对象的文档（会尝试从父类继承）
    doc = _get_doc_with_inheritance(obj)
    
    if doc is None:
        # 如果还是没有文档，生成提示信息
        doc = """⚠️  未找到文档字符串（包括父类）
该对象可能：
- 是内置函数/方法（C 实现，无 __doc__）
- 确实没有文档"""

    # 返回格式化的文档字符串
    return _format_help(obj, doc)

def _get_doc_with_inheritance(obj: Any) -> Optional[str]:
    """
    获取对象的文档字符串，支持从父类继承。
    """
    # 直接获取文档
    doc = inspect.getdoc(obj)
    if doc is not None:
        return doc
    
    # 如果是 bound method（实例方法），尝试从父类查找
    if inspect.ismethod(obj):
        return _get_method_doc_from_parents(obj)
    
    # 如果是 function（未绑定），尝试在类中查找对应方法
    if inspect.isfunction(obj):
        # 尝试获取定义该函数的类
        qualname = getattr(obj, '__qualname__', '')
        if '.' in qualname:
            class_name, method_name = qualname.rsplit('.', 1)
            try:
                # 尝试获取模块并找到类
                module = inspect.getmodule(obj)
                if module and hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    if inspect.isclass(cls):
                        return _get_method_doc_from_class(cls, method_name)
            except (AttributeError, ImportError):
                pass
    
    # 如果是类，尝试合并父类文档
    if inspect.isclass(obj):
        return _get_class_doc_from_parents(obj)
    else:
        return _get_class_doc_from_parents(obj.__class__)


def _get_method_doc_from_parents(method) -> Optional[str]:
    """
    从实例方法的父类中查找文档。
    """
    # 获取 self 实例和类
    self_obj = method.__self__
    cls = type(self_obj)
    method_name = method.__name__
    
    return _get_method_doc_from_class(cls, method_name)

def _get_method_doc_from_class(cls: type, method_name: str) -> Optional[str]:
    """
    从类的 MRO(方法解析顺序)中查找方法文档。
    """
    # 遍历 MRO（父类链）
    for parent in cls.__mro__[1:]:  # 跳过自身，从父类开始
        if hasattr(parent, method_name):
            parent_method = getattr(parent, method_name)
            # 获取父类方法的文档
            parent_doc = inspect.getdoc(parent_method)
            if parent_doc:
                return f"[继承自 {parent.__name__}.{method_name}]\n\n{parent_doc}"
    
    return None

def _get_class_doc_from_parents(cls: type) -> Optional[str]:
    """
    尝试从父类获取类的文档（用于 __init__ 方法）。
    """
    # 如果类本身没有 doc，但 __init__ 可能有
    for parent in cls.__mro__[1:]:
        parent_doc = inspect.getdoc(parent)
        if parent_doc:
            return f"[类文档继承自 {parent.__name__}]\n\n{parent_doc}"
    
    return None

def _format_help(obj: Any, doc: str) -> str:
    """
    格式化帮助信息为字符串。
    """
    lines = []
    
    # 获取对象信息
    try:
        module = getattr(obj, '__module__', 'built-in')
        obj_type_name = type(obj).__name__
        name = getattr(obj, '__qualname__', getattr(obj, '__name__', str(obj)))
    except Exception:
        name = str(obj)
        module = 'unknown'
        obj_type_name = 'unknown'

    # 类型信息
    if inspect.isclass(obj):
        lines.append(f"类型: class")
        # 显示继承链
        parents = [p.__name__ for p in obj.__mro__[1:-1]]  # 排除自身和 object
        if parents: lines.append(f"继承自: {', '.join(parents)}")
    elif inspect.isfunction(obj):
        lines.append(f"类型: function")
    elif inspect.ismethod(obj):
        lines.append(f"类型: method (bound to {type(obj.__self__).__name__})")
    elif inspect.ismodule(obj):
        lines.append(f"类型: module")
    else:
        lines.append(f"类型: {obj_type_name}")
    
    if module != 'built-in':
        lines.append(f"模块: {module}")
    
    # QS 对象处理
    if isinstance(obj, FactorOperator):
        name = f"{module}.{obj_type_name}.__call__"
        lines.append("QS 对象类型: 因子算子")
    elif isinstance(obj, Factor):
        lines.append("QS 对象类型: 计算节点-因子")
    elif isinstance(obj, FactorTable):
        lines.append("QS 对象类型: 计算节点-因子表")
    elif isinstance(obj, FactorDB):
        lines.append("QS 对象类型: 因子库")
    elif isinstance(obj, RiskTable):
        lines.append("QS 对象类型: 计算节点-风险表")
    elif isinstance(obj, RiskDB):
        lines.append("QS 对象类型: 风险库")
    elif isinstance(obj, BTNode):
        lines.append("QS 对象类型: 计算节点-回测节点")
    elif isinstance(obj, Context):
        lines.append("QS 对象类型: 全局上下文")
    elif isinstance(obj, LocalContext):
        lines.append("QS 对象类型: 局部上下文")
    elif isinstance(obj, Node):
        lines.append("QS 对象类型: 计算节点")
    elif isinstance(obj, Engine):
        lines.append("QS 对象类型: 计算引擎")
    elif isinstance(obj, Cache):
        lines.append("QS 对象类型: 缓存")
    elif isinstance(obj, __QS_Args__):
        lines.append("QS 对象类型: 参数集")
    if isinstance(obj, (__QS_Args__, __QS_Object__)):
        if hasattr(obj, "Name"):
            lines.append(f"QS 对象名称: {obj.Name}")
        lines.append(f"QSID: {obj.QSID}")
    if isinstance(obj, __QS_Object__): lines.append(f"参数集:\n    {obj.Args.info(html=False).replace('\n', '\n    ')}")
    elif isinstance(obj, __QS_Args__): lines.append(f"所含参数:\n    {obj.info(html=False).replace('\n', '\n    ')}")
    
    # 显示签名（如果是可调用的）
    try:
        if inspect.isclass(obj):
            sig = inspect.signature(obj.__init__)
            lines.append(f"构造函数签名: {name}.__init__{sig}")
            init_doc = _get_doc_with_inheritance(obj.__init__)
            lines.append("构造函数文档:")
            lines.append("    " + init_doc.replace("\n", "\n    "))
        if callable(obj) and not inspect.isclass(obj):
            sig = inspect.signature(obj)
            lines.append(f"签名: {name}{sig}")
    except (ValueError, TypeError):
        pass
    lines.append("说明文档:")
    lines.append("    " + doc.replace("\n", "\n    "))
    
    return "\n".join(lines)
# =============================================================

def node2dict(node_list: List[Node]) -> dict:
    """
    Node 列表转换为嵌套字典
    
    参数:
        node_list: Node 列表

    返回:
        嵌套字典, Key 格式为 "QSID:Name", 其中 QSID 是唯一ID, Name 为 Node 的名称
    """
    ParsedNode = set()
    def traverse(node_list, target_dict):
        """递归遍历 Node 列表"""
        for iNode in node_list:
            iQSID, iName = iNode.QSID, iNode.Name
            iKey = f"{iQSID}:{iName}"
            if iKey in ParsedNode:
                target_dict[iKey] = None
            else:
                ParsedNode.add(iKey)
                if not iNode.Deps:
                    target_dict[iKey] = None
                else:
                    target_dict[iKey] = traverse(iNode.Deps, OrderedDict())
        return target_dict
    return traverse(node_list, OrderedDict())

def dict2mermaid(nested_dict: dict, direction: Literal["TD", "LR", "BT", "RL"]="TD", node_style: Optional[dict]=None) -> str:
    """
    将嵌套字典转换为 Mermaid 语法的图
    
    Args:
        nested_dict: 嵌套字典, Key 格式为 "ID:Name", 其中 ID 是唯一标识, Name 为显示文本
        direction: 图的方向，'TD'(从上到下), 'LR'(从左到右), 'BT'(从下到上), 'RL'(从右到左)
        node_style: 自定义节点样式 dict, 如 {"fill":"#f9f", "stroke":"#333"}
    
    Returns:
        Mermaid 语法的字符串
    """
    lines = [f"graph {direction}"]
    if node_style:
        style_str = ",".join([f"{k}:{v}" for k, v in node_style.items()])
        lines.append(f"    classDef default {style_str}")
    
    # 关键：使用安全ID映射，避免使用原始ID中的非法字符
    id_counter = [0]  # 使用列表实现nonlocal修改
    original_to_safe = {}  # 原始ID -> 安全ID (node1, node2...)
    safe_to_display = {}   # 安全ID -> 显示名称
    
    def get_safe_id(original_id, display_name):
        """获取或创建安全的Mermaid ID"""
        if original_id not in original_to_safe:
            id_counter[0] += 1
            safe_id = f"node{id_counter[0]}"
            original_to_safe[original_id] = safe_id
            safe_to_display[safe_id] = display_name
            # 立即添加节点定义
            lines.append(f'    {safe_id}["{display_name}"]')
        return original_to_safe[original_id]
    
    def parse_key(key):
        """解析 key，分离 ID 和显示名称"""
        if ':' in key:
            parts = key.split(':', 1)
            return parts[0], parts[1]
        return key, key
    
    def traverse(current_dict, parent_safe_id=None):
        """递归遍历字典"""
        if current_dict is None:
            return
        
        for key, value in current_dict.items():
            original_id, display_name = parse_key(key)
            
            # 获取安全的Mermaid ID，而不是直接使用原始ID
            current_safe_id = get_safe_id(original_id, display_name)
            
            # 如果有父节点，添加边（使用安全ID）
            if parent_safe_id is not None:
                lines.append(f"    {parent_safe_id} --> {current_safe_id}")
            
            # 递归处理子节点
            if isinstance(value, dict):
                traverse(value, current_safe_id)
    
    # 开始遍历
    traverse(nested_dict)
    
    return '\n'.join(lines)


# 测试 dict2mermaid
if __name__ == "__main__":
    data = {
        "ID1:Node1": {
            "ID1-1:Node1-1": {"ID1-1-1:Node1-1-1": None, "ID1-1-2:Node1-1-2": None},
            "ID1-2:Node1-2": {"ID1-1-1:Node1-1-1": None, "ID1-2-2:Node1-2-2": None}
        },
        "ID2:Node2": {"ID2-1:Node2-1": None}
    }

    # 隐藏值，从左到右布局
    print(dict2mermaid(data, direction="TD", node_style={"fill": "#f96", "stroke": "#333"}))