# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import Optional, Literal, List

import numpy as np
import pandas as pd

from QuantStudio.Core.Node import Node


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


def dict2mermaid(nested_dict: dict, direction: Literal["TD", "LR", "BT", "RL"]="TD", node_style: Optional[dict]=None):
    """
    将嵌套字典转换为 Mermaid 语法的图
    
    参数:
        nested_dict: 嵌套字典, Key 格式为 "ID:Name", 其中 ID 是唯一标识, Name 为显示文本
        direction: 图的方向，'TD'(从上到下), 'LR'(从左到右), 'BT'(从下到上), 'RL'(从右到左)
        node_style: 自定义节点样式 dict, 如 {"fill":"#f9f", "stroke":"#333"}
    
    返回:
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


if __name__=="__main__":
    data = {
        "ID1:Node1": {
            "ID1-1:Node1-1": {"ID1-1-1:Node1-1-1": None, "ID1-1-2:Node1-1-2": None},
            "ID1-2:Node1-2": {"ID1-1-1:Node1-1-1": None, "ID1-2-2:Node1-2-2": None}
        },
        "ID2:Node2": {"ID2-1:Node2-1": None}
    }

    # 隐藏值，从左到右布局
    print(dict2mermaid(data, direction="TD", node_style={"fill": "#f96", "stroke": "#333"}))