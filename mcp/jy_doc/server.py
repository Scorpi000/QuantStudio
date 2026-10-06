# -*- coding: utf-8 -*-
"""聚源数据库(Gildata)在线文档 MCP 服务入口。

提供 7 个 MCP 工具供 AI Agent 查询聚源数据字典文档：
- search_tables: 搜索数据库表
- get_table_detail: 获取指定表的完整字段和说明
- browse_categories: 浏览数据库分类目录
- get_database_page: 获取某库下所有表的概要列表
- search_online: 在线实时搜索（作为兜底）
- query_qs_read_data_help: 查某表在 QuantStudio 中读取数据的用法
- query_qs_get_factor_help: 查某表在 QuantStudio 中获取因子对象的用法

后两个工具的数据来自 QuantStudio/Resource/JYDBInfo.xlsx，离线生成，不连数据库。

使用方式:
    # stdio 模式（默认）
    python mcp/jy_doc/server.py

    # 指定缓存目录
    python mcp/jy_doc/server.py --cache-dir D:/Data/JYDBDoc

    # 启动前重建全部缓存（清除旧索引与表详情缓存并重新抓取）
    python mcp/jy_doc/server.py --rebuild-cache

    # 指定缓存的登录凭据（重建时需要）
    python mcp/jy_doc/server.py --rebuild-cache --user xxx --pwd xxx

缓存说明:
    本服务启动时加载 tree_index.json；若索引缺失或为空，会自动抓取构建一次
    （构建失败则中止启动，不会让客户端误判为连接成功）。本服务不会自动更新
    缓存：tree_index.json 在服务启动时加载一次，tables/<id>.json 一旦写入便
    永久复用。若聚源平台的表结构有更新，需重启服务并加上 --rebuild-cache 才会刷新。

    重建/首次构建过程需要网络与聚源平台登录凭据（验证码自动识别），耗时约
    1 分钟，期间服务不响应。可通过 --user/--pwd 参数或 JY_DOC_USER/JY_DOC_PWD
    环境变量提供凭据。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

if __package__:
    from .fetcher import JYDocFetcher
    from .qs_help import get_factor_help, read_data_help, preload as qs_help_preload
    from .scraper import count_tables, scrape_and_save
else:
    # 直接以脚本方式运行（python mcp/jy_doc/server.py）时 __package__ 为空，
    # 相对导入不可用。这里用 importlib 把 jy_doc 注册成真正的包再导入，
    # 使 fetcher/scraper 内部的相对导入（`from .models import ...`）同样解析成功。
    # 不能简单地把 mcp/ 加进 sys.path 后 `import jy_doc.xxx`——那样模块会被
    # 加载两次（__main__ 一份、包命名空间一份），isinstance 之类的判断会失效。
    import importlib.util

    _pkg_dir = Path(__file__).resolve().parent

    def _load(name: str, filename: str, is_pkg: bool = False):
        kwargs = {"submodule_search_locations": [str(_pkg_dir)]} if is_pkg else {}
        spec = importlib.util.spec_from_file_location(
            name, str(_pkg_dir / filename), **kwargs
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("jy_doc", "__init__.py", is_pkg=True)
    _load("jy_doc.models", "models.py")
    from jy_doc.fetcher import JYDocFetcher
    from jy_doc.qs_help import get_factor_help, read_data_help, preload as qs_help_preload
    from jy_doc.scraper import count_tables, scrape_and_save

# 凭据与缓存目录配置（mcp/.env，随仓库分发 .env.example）
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

# ── 全局状态 ────────────────────────────────────────────────────────

mcp = FastMCP(name="jy_doc")
fetcher: JYDocFetcher = None  # type: ignore[assignment]
flat_index: list[dict] = []
databases_data: list[dict] = []
categories_data: dict = {}


def _load_index(cache_dir: str) -> None:
    """加载本地索引数据到全局变量。"""
    global flat_index, databases_data, categories_data

    index_path = os.path.join(cache_dir, "tree_index.json")
    if not os.path.exists(index_path):
        logger.info("索引文件不存在: %s", index_path)
        flat_index = []
        databases_data = []
        categories_data = {}
        return

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_index = data.get("flat_index", [])
    databases_data = data.get("databases", [])
    categories_data = data.get("categories", {})
    logger.info(
        "索引加载完成: %d 张表, %d 个数据库",
        len(flat_index),
        len(databases_data),
    )


def _format_table_detail(detail, fmt: str = "text") -> str:
    """将 TableDetail 格式化为 Agent 可读的文本。"""
    parts = []

    if fmt == "markdown":
        parts.append(f"# {detail.table_name}")
        parts.append(f"**物理表名:** `{detail.base_table_name}`")
        if detail.path:
            parts.append(f"**文档路径:** {detail.path}")
        parts.append("")
        if detail.description:
            parts.append(f"## 说明\n{detail.description}\n")
        if detail.update_frequency:
            parts.append(f"**数据更新频率:** {detail.update_frequency}\n")
        if detail.columns:
            parts.append("## 字段列表\n")
            parts.append("| 字段名 | 中文名 | 数据类型 | 可空 | 说明 |")
            parts.append("|--------|--------|----------|------|------|")
            for col in detail.columns:
                nullable = "是" if col.is_nullable else "否"
                remark = col.remark.replace("|", "\\|").replace("\n", " ")
                parts.append(
                    f"| {col.name} | {col.chinese_name} | {col.data_type} | {nullable} | {remark} |"
                )
            parts.append("")
        if detail.unique_index:
            idx_cols = detail.unique_index.get("columnName", "")
            if idx_cols:
                parts.append(f"## 唯一索引\n字段: {idx_cols}\n")
    else:
        parts.append(f"[表名] {detail.table_name}")
        parts.append(f"[物理表名] {detail.base_table_name}")
        if detail.path:
            parts.append(f"[文档路径] {detail.path}")
        if detail.description:
            parts.append(f"\n[说明]\n{detail.description}")
        if detail.update_frequency:
            parts.append(f"\n[数据更新频率] {detail.update_frequency}")
        if detail.columns:
            parts.append(f"\n[字段列表] ({len(detail.columns)} 个字段)")
            for col in detail.columns:
                nullable = "可空" if col.is_nullable else "非空"
                line = f"  - {col.name} ({col.chinese_name}) {col.data_type} [{nullable}]"
                if col.remark:
                    # 截断过长的 remark
                    remark = col.remark[:100] + "..." if len(col.remark) > 100 else col.remark
                    line += f" — {remark}"
                parts.append(line)
        if detail.unique_index:
            idx_cols = detail.unique_index.get("columnName", "")
            if idx_cols:
                parts.append(f"\n[唯一索引] {idx_cols}")

    return "\n".join(parts)


# ── MCP 工具 ────────────────────────────────────────────────────────


@mcp.tool()
def search_tables(
    keyword: str,
    category: str = "all",
    max_results: int = 15,
) -> str:
    """搜索聚源数据库中的表。

    在本地索引中按表名、物理表名、路径、描述进行多字段加权匹配搜索。
    当本地结果不足时自动调用在线搜索补充。

    Args:
        keyword: 搜索关键词（支持中英文，如 "财务"、"MainFinancial"、"股票基本信息"）
        category: 搜索范围 — 数据库名如 "国内上市公司数据库" | "all"(全部)
        max_results: 最大返回结果数量，默认 15

    Returns:
        匹配的表列表，包含表名、物理表名、路径、描述
    """
    keyword_lower = keyword.lower()
    keywords = [kw.strip() for kw in keyword_lower.split() if kw.strip()]

    # 加权评分搜索
    scored = []
    for entry in flat_index:
        if category != "all" and entry["category"] != category:
            continue

        base_name_lower = entry.get("base_table_name", "").lower()
        table_name_lower = entry.get("table_name", "").lower()
        path_lower = entry.get("path", "").lower()
        desc_lower = entry.get("description", "").lower()

        score = 0
        for kw in keywords:
            # 物理表名完全匹配 — 最高分
            if kw == base_name_lower:
                score += 100
            # 物理表名包含关键词
            elif kw in base_name_lower:
                score += 60
            # 中文表名包含关键词
            if kw in table_name_lower:
                score += 40
            # 路径包含关键词
            if kw in path_lower:
                score += 20
            # 描述包含关键词
            if kw in desc_lower:
                score += 10

        # 关键词作为整体也尝试匹配
        if keyword_lower in base_name_lower:
            score += 30
        if keyword_lower in table_name_lower:
            score += 20

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    results = [entry for _, entry in scored[:max_results]]

    if not results:
        return (
            f"未找到与 '{keyword}' 相关的表。建议尝试：\n"
            f"1. 使用不同的关键词（中英文均可）\n"
            f"2. 使用 browse_categories 查看可用数据库分类\n"
            f"3. 使用 search_online 工具进行在线搜索"
        )

    lines = [f"找到 {len(results)} 个相关表：\n"]
    for i, r in enumerate(results, 1):
        desc = r.get("description", "")
        if len(desc) > 80:
            desc = desc[:80] + "..."
        base_name = r.get("base_table_name", "")
        name_info = f"{r['table_name']}"
        if base_name:
            name_info += f" ({base_name})"
        lines.append(f"{i}. **{name_info}**  [{r.get('category', '')}]")
        lines.append(f"   路径: {r.get('path', '')}")
        if desc:
            lines.append(f"   {desc}")
        lines.append(f"   table_id: {r.get('table_id', '')}")

    lines.append(f"\n使用 get_table_detail(table_id=<ID>) 获取表的完整字段和说明。")

    return "\n".join(lines)


@mcp.tool()
def get_table_detail(table_id: int, format: str = "text") -> str:
    """获取指定聚源数据库表的完整详情。

    通过表 ID 获取表的详细信息，包括表说明、字段列表（名称、类型、说明）、
    唯一索引、更新频率等。内容来自本地缓存或实时调用 Gildata API。

    Args:
        table_id: Gildata 平台中的表 ID（从 search_tables 或 browse_categories 结果中获取）
        format: 返回格式 — "text"(纯文本，默认) | "markdown"(Markdown格式)

    Returns:
        表的完整信息
    """
    # 先从索引中查找路径信息
    path = ""
    for entry in flat_index:
        if entry.get("table_id") == table_id:
            path = entry.get("path", "")
            break

    try:
        detail = fetcher.fetch_table_detail(table_id, use_cache=True, path=path)
    except Exception as e:
        # 底层可能抛出登录失败、请求重试耗尽等异常，统一转为 Agent 可读的提示
        logger.warning("获取表详情异常 (table_id=%s): %s: %s", table_id, type(e).__name__, e)
        detail = None

    if detail is None:
        return (
            f"获取表详情失败 (table_id: {table_id})\n"
            f"可能原因：\n1. 表 ID 不存在\n2. 网络连接失败\n3. 无权限访问该表\n"
            f"建议先使用 search_tables 搜索确认表 ID。"
        )

    return _format_table_detail(detail, format)


@mcp.tool()
def browse_categories(detail: bool = False) -> str:
    """浏览聚源数据库的分类目录。

    返回所有数据库及其表数量统计，帮助了解聚源数据库的整体结构。

    Args:
        detail: 是否显示每个库的 tree_index 中的原始 ID，默认 False

    Returns:
        数据库分类目录列表及表统计
    """
    if not databases_data:
        return "未加载分类索引。请先运行爬虫脚本：python scripts/scrape_jy_doc.py"

    lines = ["聚源数据库分类目录：\n"]

    total_count = 0
    for db in databases_data:
        db_name = db.get("name", "")
        db_id = db.get("id", 0)
        # 统计该库下的表数量
        cat_info = categories_data.get(db_name, {})
        tree = cat_info.get("tree", [])

        table_count = count_tables(tree) if tree else 0
        total_count += table_count

        line = f"  - **{db_name}** ({table_count} 张表)"
        if detail:
            line += f"  [ID: {db_id}]"
        lines.append(line)

    lines.append(f"\n共计 {len(databases_data)} 个数据库, {total_count} 张表")
    lines.append("\n使用 search_tables(keyword, category=<库名>) 搜索特定库下的表。")
    lines.append("使用 get_database_page(database=<库名>) 查看某库下的所有表。")

    return "\n".join(lines)


@mcp.tool()
def get_database_page(database: str, max_tables: int = 50) -> str:
    """获取某个聚源数据库下的所有表概要列表。

    返回指定数据库下所有表的名称和简要描述，适合在搜索前先浏览某个库下有哪些表。

    Args:
        database: 数据库名称，如 "国内上市公司数据库"。使用 browse_categories 查看可用库名
        max_tables: 最大返回表数量，默认 50

    Returns:
        该库下所有表的概要列表
    """
    if database not in categories_data:
        available = ", ".join(categories_data.keys())
        return f"未知数据库: '{database}'。可用数据库: {available}"

    cat_info = categories_data[database]
    tree = cat_info.get("tree", [])

    # 从树中提取所有叶子节点
    tables = []
    _collect_tables(tree, tables)

    if not tables:
        return f"数据库 '{database}' 下未找到任何表。"

    lines = [f"# {database}  (共 {len(tables)} 张表)\n"]

    total_len = len(lines[0])
    for t in tables[:max_tables]:
        desc = t.get("description", "")
        if len(desc) > 100:
            desc = desc[:100] + "..."
        entry = f"- **{t['name']}** (ID: {t['id']})"
        if desc:
            entry += f": {desc}"
        if total_len + len(entry) > 8000:
            lines.append(f"\n... (内容过长已截断，共 {len(tables)} 张表)")
            break
        lines.append(entry)
        total_len += len(entry)

    if len(tables) > max_tables:
        lines.append(f"\n... (仅显示前 {max_tables} 张表，共 {len(tables)} 张)")

    lines.append(f"\n使用 get_table_detail(table_id=<ID>) 获取表的完整字段和说明。")

    return "\n".join(lines)


def _collect_tables(nodes: list[dict], tables: list[dict]) -> None:
    """递归收集目录树中的叶子节点（表）。"""
    for node in nodes:
        if node.get("istable", False):
            tables.append({
                "id": node.get("id", 0),
                "name": node.get("groupName", ""),
                "description": node.get("description", ""),
            })
        children = node.get("nodes", [])
        if children:
            _collect_tables(children, tables)


@mcp.tool()
def search_online(keyword: str) -> str:
    """在线实时搜索聚源数据库表。

    在本地索引中进行实时模糊匹配搜索，结果更实时但需要索引已加载。
    当本地 search_tables 找不到结果时，建议使用此工具。

    Args:
        keyword: 搜索关键词

    Returns:
        搜索结果列表
    """
    keyword_lower = keyword.lower()

    results = []
    for entry in flat_index:
        # 搜索所有文本字段
        searchable = " ".join([
            entry.get("table_name", ""),
            entry.get("base_table_name", ""),
            entry.get("path", ""),
            entry.get("description", ""),
            entry.get("category", ""),
        ]).lower()

        if keyword_lower in searchable:
            results.append(entry)

    if not results:
        return f"在线搜索未找到与 '{keyword}' 相关的表。"

    lines = [f"在线搜索 '{keyword}' 找到 {len(results)} 条结果：\n"]
    for i, r in enumerate(results[:20], 1):
        desc = r.get("description", "")
        if len(desc) > 60:
            desc = desc[:60] + "..."
        lines.append(f"{i}. {r.get('table_name', '')} [{r.get('category', '')}]")
        if r.get("base_table_name"):
            lines.append(f"   物理表名: {r['base_table_name']}")
        lines.append(f"   ID: {r.get('table_id', '')}  |  路径: {r.get('path', '')}")
        if desc:
            lines.append(f"   {desc}")

    if len(results) > 20:
        lines.append(f"\n... 共 {len(results)} 条结果，仅显示前 20 条")
    lines.append(f"\n使用 get_table_detail(table_id=<ID>) 获取表的完整字段和说明。")

    return "\n".join(lines)


def _resolve_base_table_name(table_id: int) -> tuple[str, str]:
    """由 table_id 查出 (物理表名, 中文表名)。未找到时返回 ("", "")。"""
    for entry in flat_index:
        if entry.get("table_id") == table_id:
            return entry.get("base_table_name", ""), entry.get("table_name", "")
    return "", ""


@mcp.tool()
def query_qs_read_data_help(table_id: int) -> str:
    """查询指定表在 QuantStudio 中读取数据的用法。

    给出一段可直接运行的示例代码（getTable + readData），包含该表在
    QuantStudio 中的**内部表名**与**因子名**，以及 args 常用参数。
    数据来自 JYDBInfo.xlsx，离线生成，不连数据库。

    因子名与聚源字段中文名可能不同（如聚源“所属状态”在 QuantStudio 中是
    “所属状态_R”），本工具统一给出 QuantStudio 侧的因子名，可直接用于
    getFactor 与 readData。

    Args:
        table_id: 表 ID（从 search_tables / browse_categories 结果中获取）

    Returns:
        示例代码与参数说明；该表未被 QuantStudio 支持时返回提示
    """
    base_name, cn_name = _resolve_base_table_name(table_id)
    if not base_name:
        return (
            f"未在索引中找到 table_id={table_id} 的表，或其缺少物理表名。\n"
            f"建议先使用 search_tables 确认 table_id。"
        )
    return read_data_help(base_name)


@mcp.tool()
def query_qs_get_factor_help(table_id: int) -> str:
    """查询指定表在 QuantStudio 中获取因子对象的用法。

    给出一段可直接运行的示例代码（getTable + getFactor），包含该表在
    QuantStudio 中的**内部表名**与**因子名**，以及 args 常用参数。
    数据来自 JYDBInfo.xlsx，离线生成，不连数据库。

    Args:
        table_id: 表 ID（从 search_tables / browse_categories 结果中获取）

    Returns:
        示例代码与参数说明；该表未被 QuantStudio 支持时返回提示
    """
    base_name, cn_name = _resolve_base_table_name(table_id)
    if not base_name:
        return (
            f"未在索引中找到 table_id={table_id} 的表，或其缺少物理表名。\n"
            f"建议先使用 search_tables 确认 table_id。"
        )
    return get_factor_help(base_name)


# ── 启动 ────────────────────────────────────────────────────────────


def get_default_cache_dir() -> str:
    """获取默认缓存目录路径。"""
    return os.getenv("JY_DOC_CACHE", r"D:\Data\JYDBDoc")


def rebuild_cache(cache_dir: str, user: str = "", pwd: str = "") -> None:
    """重建全部本地缓存：清除旧缓存并重新抓取索引。

    依次清除 tables/（表详情）与 trees/（目录树）下的缓存文件，然后调用
    爬虫重新生成 tree_index.json。需要网络访问与聚源平台登录凭据。

    Args:
        cache_dir: 缓存目录路径
        user: 聚源平台用户名
        pwd: 聚源平台密码

    Raises:
        RuntimeError: 抓取失败时抛出（凭据缺失、网络异常等）
    """
    import shutil

    for sub in ("tables", "trees"):
        sub_dir = os.path.join(cache_dir, sub)
        if os.path.isdir(sub_dir):
            count = len(os.listdir(sub_dir))
            shutil.rmtree(sub_dir)
            logger.info("已清除旧缓存目录 %s/ (%d 个文件)", sub, count)

    index_path = os.path.join(cache_dir, "tree_index.json")
    if os.path.exists(index_path):
        os.remove(index_path)
        logger.info("已删除旧索引文件: %s", index_path)

    logger.info("开始重建缓存（需登录聚源平台，耗时约 1 分钟）...")
    index_data = scrape_and_save(cache_dir=cache_dir, user=user, pwd=pwd)
    logger.info(
        "缓存重建完成: %d 个数据库, %d 张表",
        len(index_data.get("categories", {})),
        len(index_data.get("flat_index", [])),
    )


def _preimport_native_deps() -> None:
    """在主线程预先导入含 C 扩展的依赖，规避 Windows 上的线程加载死锁。

    Windows 上存在 CPython 已知缺陷（bpo-33895）：工作线程首次 import 原生
    扩展时，LoadLibraryExW 会持有 GIL 去竞争 DLL 加载器锁，与 asyncio
    proactor 事件循环的线程启动/退出相互等待，造成死锁。表现为 FastMCP 工具
    调用（运行在 anyio 工作线程中）无限挂起，直到客户端断开才被解锁。

    在启动阶段（主线程、事件循环之前）先导入一次即可规避：后续在工具线程中的
    import 命中 sys.modules 缓存，不再触发原生扩展加载。

    两个来源都必须在主线程预热：
    1. ddddocr（连带 onnxruntime / numpy）——登录时识别验证码用；
       未安装时留待 _login 抛出更明确的错误提示。
    2. QuantStudio + JYDBInfo.xlsx 解析——query_qs_* 工具用。
       `_importInfo` 会拉起 pandas/numpy 等原生扩展；且 xlsx 解析本身耗时，
       放在工具线程里既可能死锁又会拖慢首次调用，故一并预热（顺带填充
       qs_help 的 lru_cache）。
    """
    try:
        import ddddocr  # noqa: F401
    except ImportError:
        pass

    try:
        from QuantStudio import __QS_MainPath__  # noqa: F401
        from QuantStudio.Factor.JYDB import _importInfo  # noqa: F401

        # 真正解析一次，把原生扩展与 lru_cache 都在主线程准备好
        qs_help_preload()
    except Exception as e:
        # QuantStudio 不可用时仅 qs_help 两个工具降级，其余工具照常
        logger.warning(
            "QuantStudio/JYDBInfo 预加载失败，query_qs_* 工具将不可用: %s: %s",
            type(e).__name__, e,
        )


def init_server(
    cache_dir: str, user: str = "", pwd: str = "", rebuild: bool = False
) -> None:
    """初始化 MCP 服务。

    若本地索引未建立（tree_index.json 缺失或为空），会自动抓取构建索引；
    构建失败则抛出异常中止启动，避免 MCP 在无索引状态下被客户端判定为连接成功。

    Args:
        cache_dir: 缓存目录路径
        user: 聚源平台用户名
        pwd: 聚源平台密码
        rebuild: 是否在启动前重建缓存（清除旧缓存并重新抓取）

    Raises:
        RuntimeError: 索引缺失且自动构建失败时抛出
    """
    _preimport_native_deps()

    global fetcher
    fetcher = JYDocFetcher(cache_dir=cache_dir, user=user, pwd=pwd)

    if rebuild:
        rebuild_cache(cache_dir, user=user, pwd=pwd)

    _load_index(cache_dir)

    # 索引未建立时自动构建（首次启动或缓存被清空时）
    if not flat_index:
        logger.info("未检测到本地索引，开始自动构建索引...")
        try:
            scrape_and_save(cache_dir=cache_dir, user=user, pwd=pwd)
        except Exception as e:
            # 构建失败则中止启动，避免 MCP 在无索引状态下被判定为连接成功
            raise RuntimeError(
                f"索引自动构建失败，MCP 服务中止启动: {type(e).__name__}: {e}"
            ) from e
        _load_index(cache_dir)

    logger.info("聚源数据库文档 MCP 服务初始化完成, cache_dir=%s", cache_dir)


if __name__ == "__main__":
    # 仓库根 .env 已在模块导入时加载，此处的 JY_DOC_* 环境变量对 argparse 默认值生效
    parser = argparse.ArgumentParser(description="聚源数据库文档 MCP 服务")
    parser.add_argument(
        "--cache-dir",
        default=get_default_cache_dir(),
        help="本地缓存目录路径 (默认: D:\\Data\\JYDBDoc 或 JY_DOC_CACHE 环境变量)",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("JY_DOC_USER", ""),
        help="聚源文档平台用户名 (默认: JY_DOC_USER 环境变量)",
    )
    parser.add_argument(
        "--pwd",
        default=os.getenv("JY_DOC_PWD", ""),
        help="聚源文档平台密码 (默认: JY_DOC_PWD 环境变量)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help=(
            "启动前重建全部缓存：清除旧索引与表详情缓存并重新抓取。"
            "需要网络与聚源平台登录凭据，耗时约 1 分钟，期间服务不会响应"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.rebuild_cache:
        try:
            init_server(
                args.cache_dir, user=args.user, pwd=args.pwd, rebuild=True
            )
        except Exception as e:
            # 重建失败不应导致服务无法启动：回退到加载现有缓存
            logger.error("缓存重建失败: %s: %s", type(e).__name__, e)
            logger.warning("回退为使用现有缓存启动（如缓存不存在则索引为空）")
            init_server(args.cache_dir, user=args.user, pwd=args.pwd)
    else:
        init_server(args.cache_dir, user=args.user, pwd=args.pwd)

    mcp.run(transport="stdio")
    # mcp.run(transport="http", host="0.0.0.0", port=58001)
