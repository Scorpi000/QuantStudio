# -*- coding: utf-8 -*-
"""聚源数据库文档目录树预抓取模块。

登录聚源数据字典平台(dd.gildata.com)，遍历所有数据库的目录树结构，
生成 tree_index.json 索引文件，供 MCP 服务快速加载使用。

使用方式:
    from mcp.jy_doc.scraper import scrape_and_save
    scrape_and_save("D:/Data/JYDBDoc", user="xxx", pwd="xxx")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .fetcher import JYDocFetcher

logger = logging.getLogger(__name__)


def build_flat_index(
    nodes: list[dict], category: str = "", parent_path: str = "/聚源数据库"
) -> list[dict]:
    """将树形目录结构展平为扁平索引列表，便于搜索。

    递归遍历所有节点，提取叶子节点（表）的关键信息。

    Gildata 目录树的节点结构：
    {
        "id": 258,
        "groupName": "公司主要财务分析指标",
        "istable": true/false,
        "description": "...",
        "nodes": [...子节点...]
    }

    Args:
        nodes: 目录树节点列表
        category: 所属数据库分类名称
        parent_path: 父级路径

    Returns:
        扁平的索引条目列表（仅包含叶子节点，即 istable=True 的表）
    """
    entries = []
    for node in nodes:
        node_name = node.get("groupName", "")
        node_id = node.get("id", 0)
        is_table = node.get("istable", False)
        current_path = f"{parent_path}/{node_name}" if node_name else parent_path

        if is_table:
            entries.append({
                "table_id": node_id,
                "table_name": node_name,
                # 目录树的叶子节点自带 tableName（物理表名），如 LC_StockArchives
                "base_table_name": node.get("tableName", ""),
                "path": current_path,
                "category": category,
                "description": node.get("description", ""),
            })

        # 递归处理子节点
        children = node.get("nodes", [])
        if children:
            entries.extend(build_flat_index(children, category, current_path))

    return entries


def scrape_and_save(
    cache_dir: str,
    user: str = "",
    pwd: str = "",
) -> dict:
    """抓取聚源数据库文档目录树并保存到本地。

    遍历所有可访问的数据库，抓取每个库的目录树结构，
    提取所有表的信息，生成 tree_index.json 索引文件。

    Args:
        cache_dir: 缓存目录路径
        user: 登录用户名，默认从环境变量 JY_DOC_USER 读取
        pwd: 登录密码，默认从环境变量 JY_DOC_PWD 读取

    Returns:
        生成的索引数据字典
    """
    os.makedirs(cache_dir, exist_ok=True)

    fetcher = JYDocFetcher(cache_dir=cache_dir, user=user, pwd=pwd)

    # 步骤 1：获取数据库列表
    logger.info("正在获取数据库列表...")
    databases = fetcher.get_database_list()
    if not databases:
        raise RuntimeError("未获取到任何数据库，请检查登录凭据和权限")

    logger.info("共 %d 个数据库", len(databases))

    index_data = {
        "version": 1,
        "databases": [],
        "categories": {},
        "flat_index": [],
    }

    # 记录库信息
    for db in databases:
        index_data["databases"].append({
            "id": db.id,
            "name": db.name,
            "description": db.description,
        })

    total_tables = 0

    # 步骤 2：遍历每个库，抓取目录树
    for db in databases:
        logger.info("正在抓取库: %s (id=%d)...", db.name, db.id)

        try:
            tree = fetcher.fetch_tree(db.id)
        except Exception as e:
            logger.warning("库 %s (id=%d) 目录树抓取异常，跳过: %s", db.name, db.id, e)
            continue

        if not tree:
            logger.warning("库 %s 目录树获取失败，跳过", db.name)
            continue

        # 保存树结构
        index_data["categories"][db.name] = {
            "id": db.id,
            "tree": tree,
        }

        # 构建扁平索引
        flat = build_flat_index(tree, category=db.name)
        index_data["flat_index"].extend(flat)

        table_count = len(flat)
        total_tables += table_count
        logger.info("库 %s: %d 张表", db.name, table_count)

        # 每个库抓完即落盘，避免中途失败导致已抓取内容全部丢失
        _save_index(cache_dir, index_data)

    # 步骤 3：保存索引文件
    index_path = _save_index(cache_dir, index_data)

    logger.info(
        "索引文件已保存: %s (共 %d 个库, %d 张表)",
        index_path, len(index_data["categories"]), total_tables,
    )
    return index_data


def _save_index(cache_dir: str, index_data: dict) -> str:
    """将索引数据写入 tree_index.json，返回文件路径。"""
    index_path = os.path.join(cache_dir, "tree_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    return index_path


def count_tables(nodes: list[dict]) -> int:
    """递归计算目录树中的表总数（叶子节点数）。"""
    count = 0
    for node in nodes:
        if node.get("istable", False):
            count += 1
        children = node.get("nodes", [])
        if children:
            count += count_tables(children)
    return count
