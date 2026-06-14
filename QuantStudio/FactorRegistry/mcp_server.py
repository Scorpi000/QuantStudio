# -*- coding: utf-8 -*-
"""FactorRegistry MCP Server — 因子注册中心 MCP 服务

提供三个工具：
- search_factors: 语义/关键词检索因子
- get_factor_info: 查询因子详细信息
- get_factor_code: 返回因子定义 Python 源代码
"""
import os
import json
import re
import logging
from typing import Optional

from fastmcp import FastMCP

from QuantStudio.Core import setDefaultLogLevel
setDefaultLogLevel(logging.WARNING)
from QuantStudio.Core import __QS_Logger__

mcp = FastMCP("FactorRegistry")
_FGDB = None


def _get_fgdb():
    """懒加载 FactorGraphDB 单例"""
    global _FGDB
    if _FGDB is not None:
        return _FGDB
    from .FactorGraphDB import FactorGraphDB

    # 加载 Neo4j 配置
    neo4j_cfg = _load_neo4j_config()
    if neo4j_cfg is None:
        raise RuntimeError("无法加载 Neo4j 配置: ~/QuantStudioConfig/Neo4jDBConfig.json 不存在")
    neo4j_args = {
        "Neo4jURI": f"bolt://{neo4j_cfg['IPAddr']}:{neo4j_cfg['Port']}",
        "Neo4jUser": neo4j_cfg["User"],
        "Neo4jPwd": neo4j_cfg["Pwd"],
        "Neo4jDB": neo4j_cfg.get("DBName", "neo4j"),
        "OllamaBaseURL": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "OllamaAPIKey": os.getenv("OLLAMA_API_KEY", "ollama"),
        "EmbeddingModel": os.getenv("FACTOR_EMBEDDING_MODEL", "bge-m3"),
    }
    # 根据模型设置维度
    model = neo4j_args["EmbeddingModel"]
    if model == "bge-m3":
        neo4j_args["EmbeddingDim"] = 1024
    elif model == "qwen3-embedding:8b":
        neo4j_args["EmbeddingDim"] = 4096

    _FGDB = FactorGraphDB(args=neo4j_args)
    _FGDB.connect()
    __QS_Logger__.info("FactorRegistry MCP: FactorGraphDB 已连接")
    return _FGDB


def _load_neo4j_config() -> Optional[dict]:
    """加载 Neo4j 连接配置"""
    config_path = os.path.expanduser("~/QuantStudioConfig/Neo4jDBConfig.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r",\s*([}\]])", r"\1", content)
    return json.loads(content)


def _parse_meta_json(meta_json_str: Optional[str]) -> dict:
    """安全解析 MetaJSON 字符串"""
    if not meta_json_str:
        return {}
    try:
        from ._serialization import _desanitizeFromJSON
        return _desanitizeFromJSON(json.loads(meta_json_str))
    except Exception:
        return {}


def _format_factor(f: dict, similarity: Optional[float] = None) -> dict:
    """格式化因子节点为统一输出格式"""
    result = {
        "name": f.get("Name", ""),
        "qsid": f.get("QSID", ""),
        "factor_class": f.get("FactorClass", ""),
        "data_type": f.get("DataType", ""),
        "operator_type": f.get("OperatorType", ""),
        "operator_name": f.get("OperatorName", ""),
    }
    if similarity is not None:
        result["similarity"] = similarity
    return result


# ─── MCP Tools ──────────────────────────────────────────────

@mcp.tool
def search_factors(query: str, limit: int = 20) -> list[dict]:
    """搜索因子列表。使用语义向量检索（若启用）或关键词匹配。

    Args:
        query: 查询文本，如 "动量因子"、"成交量相关"、"财务质量"
        limit: 返回结果数量上限，默认 20

    Returns:
        匹配的因子列表 [{name, qsid, factor_class, operator_type, data_type, similarity}]
    """
    fgdb = _get_fgdb()
    vector_results = []
    if fgdb._QSArgs.EmbeddingModel:
        try:
            vector_results = fgdb.searchFactorsByDescription(query, limit=limit)
        except Exception as e:
            __QS_Logger__.warning(f"向量检索失败，回退到关键词检索: {e}")
    # 关键词检索回退
    keyword_results = fgdb.searchFactors(name=query, limit=limit)
    # 合并结果：向量结果优先
    seen = set()
    formatted = []
    for r in vector_results:
        item = _format_factor(r, similarity=r.get("Similarity"))
        formatted.append(item)
        seen.add(item["qsid"])
    for r in keyword_results:
        item = _format_factor(r)
        if item["qsid"] not in seen:
            formatted.append(item)
            seen.add(item["qsid"])
    return formatted[:limit]


@mcp.tool
def get_factor_info(qsid: str) -> dict:
    """查询因子的详细信息，包括名称、描述、数据类型、算子信息、依赖关系等。

    Args:
        qsid: 因子的 QSID（唯一标识符）

    Returns:
        因子详细信息字典，包含:
        - name, qsid, factor_class, data_type, module_path
        - description: 因子描述文本
        - operator_name, operator_type: 算子信息（仅 DerivativeFactor）
        - meta: 用户自定义元信息
        - descriptors: 直接依赖因子列表 [{name, qsid}]
        - dependents: 直接下游因子列表 [{name, qsid}]
        - dependency_depth: 依赖链深度
        - tags: 标签列表
    """
    fgdb = _get_fgdb()
    node = fgdb.getFactorByQSID(qsid)
    if node is None:
        return {"error": f"未找到 QSID 为 {qsid} 的因子"}

    desc = fgdb.getDescriptors(qsid)
    deps = fgdb.getDependents(qsid, transitive=False)

    # 获取描述文本
    description = ""
    meta = _parse_meta_json(node.get("MetaJSON"))
    if isinstance(meta, dict):
        description = meta.get("Description", "")
    if not description:
        qs_args = _parse_meta_json(node.get("QSArgsJSON"))
        if isinstance(qs_args, dict):
            description = qs_args.get("Description", "")

    # 获取依赖深度
    dep_graph = fgdb.getDependencyGraph(qsid, direction="down")
    depth = max((e.get("order", 0) for e in dep_graph.get("edges", [])), default=0)

    # 获取标签
    tags = []
    try:
        tag_results = fgdb._runCypher(
            "MATCH (f:Factor {QSID: $qsid})-[:TAGGED]->(t:Tag) RETURN t.Name",
            {"qsid": qsid}
        )
        tags = [t["t.Name"] for t in tag_results]
    except Exception:
        pass

    return {
        "name": node.get("Name", ""),
        "qsid": node.get("QSID", qsid),
        "factor_class": node.get("FactorClass", ""),
        "data_type": node.get("DataType", ""),
        "module_path": node.get("ModulePath", ""),
        "description": description,
        "operator_name": node.get("OperatorName", ""),
        "operator_type": node.get("OperatorType", ""),
        "operator_qsid": node.get("OperatorQSID", ""),
        "meta": {k: str(v) for k, v in meta.items()} if isinstance(meta, dict) else {},
        "descriptors": [{"name": d.get("Name", ""), "qsid": d.get("QSID", "")} for d in desc],
        "dependents": [{"name": d.get("Name", ""), "qsid": d.get("QSID", "")} for d in deps],
        "dependency_depth": depth,
        "tags": tags,
    }


@mcp.tool
def get_factor_code(qsid: str) -> dict:
    """返回定义该因子的 Python 源代码。通过因子表追溯到 DefScriptPath 并读取文件。

    Args:
        qsid: 因子的 QSID（唯一标识符）

    Returns:
        {qsid, factor_name, script_path, source_code}
        若无法定位脚本，返回 {error: "..."}
    """
    fgdb = _get_fgdb()
    node = fgdb.getFactorByQSID(qsid)
    if node is None:
        return {"error": f"未找到 QSID 为 {qsid} 的因子"}

    factor_name = node.get("Name", "")

    def_script_path = _resolve_def_script_path(fgdb, qsid)

    if not def_script_path:
        # 通过标签推断脚本路径
        def_script_path = _infer_script_path_from_tags(fgdb, qsid)

    if not def_script_path:
        return {
            "error": f"无法定位因子 '{factor_name}' 的定义脚本",
            "qsid": qsid,
            "factor_name": factor_name,
        }

    if not os.path.exists(def_script_path):
        return {
            "error": f"定义脚本文件不存在: {def_script_path}",
            "qsid": qsid,
            "factor_name": factor_name,
            "script_path": def_script_path,
        }

    try:
        with open(def_script_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        return {
            "error": f"读取脚本文件失败: {e}",
            "qsid": qsid,
            "script_path": def_script_path,
        }

    return {
        "qsid": qsid,
        "factor_name": factor_name,
        "script_path": def_script_path,
        "source_code": source_code,
    }


def _resolve_def_script_path(fgdb, qsid: str) -> Optional[str]:
    """通过 FactorTable 的 MetaDataJSON 解析 DefScriptPath"""
    ft_results = fgdb._runCypher(
        """
        MATCH (f:Factor {QSID: $qsid})-[:BELONGS_TO]->(t:FactorTable)
        RETURN t.MetaDataJSON, t.Name
        """,
        {"qsid": qsid}
    )
    if not ft_results:
        # 尝试通过标签查找关联表
        tag_results = fgdb._runCypher(
            """
            MATCH (f:Factor {QSID: $qsid})-[:TAGGED]->(tag:Tag)
            MATCH (t:FactorTable)
            WHERE t.Name CONTAINS tag.Name OR tag.Name CONTAINS t.Name
            RETURN DISTINCT t.MetaDataJSON, t.Name
            LIMIT 1
            """,
            {"qsid": qsid}
        )
        ft_results = tag_results

    if not ft_results:
        return None

    ft_data = ft_results[0]
    meta_json_str = ft_data.get("t.MetaDataJSON", "{}")
    table_meta = _parse_meta_json(meta_json_str)
    if isinstance(table_meta, dict):
        return table_meta.get("DefScriptPath")
    return None


def _infer_script_path_from_tags(fgdb, qsid: str) -> Optional[str]:
    """通过因子标签推断定义脚本路径（标签名即模块文件名）"""
    try:
        tag_results = fgdb._runCypher(
            "MATCH (f:Factor {QSID: $qsid})-[:TAGGED]->(t:Tag) RETURN t.Name",
            {"qsid": qsid}
        )
    except Exception:
        return None

    import importlib
    for row in tag_results:
        tag_name = row["t.Name"]
        # 标签匹配因子定义模块: e.g. "stock_cn_day_bar_nafilled"
        if not re.match(r'^[a-z][a-z0-9_]*$', tag_name):
            continue
        try:
            mod = importlib.import_module(f"QSResearch.FactorDef.JY.{tag_name}")
            if hasattr(mod, '__file__') and mod.__file__:
                return mod.__file__
        except Exception:
            continue
    return None


if __name__ == "__main__":
    mcp.run(transport="stdio")
