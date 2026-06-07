# -*- coding: utf-8 -*-
"""基于 Neo4j 的因子图数据库 — 因子注册中心核心存储引擎"""
import os
import json
import html
import time
import base64
import importlib
import tempfile
import datetime as dt
from typing import Optional, Any, Dict, List, Literal, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from pydantic import Field

try:
    import neo4j
except ImportError:
    neo4j = None

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Factor.FactorDB import FactorDB
from QuantStudio.Factor.FactorTable import FactorTable
from QuantStudio.Factor.Factor import Factor, DataFactor
from QuantStudio.Factor.FactorOperation import (
    FactorOperator, DerivativeFactor,
    PointOperator, TimeOperator, SectionOperator, PanelOperator
)
from QuantStudio.FactorRegistry._serialization import (
    _sanitizeForJSON, _desanitizeFromJSON,
    serializeFactorArgs, serializeOperatorArgs, serializeOperatorCalculateRef,
    _serializeCallable, _deserializeFuncRef
)


# region Schema 定义

_SCHEMA_CONSTRAINTS = [
    "CREATE CONSTRAINT factor_qsid IF NOT EXISTS FOR (f:Factor) REQUIRE f.QSID IS UNIQUE",
    "CREATE CONSTRAINT operator_qsid IF NOT EXISTS FOR (o:FactorOperator) REQUIRE o.QSID IS UNIQUE",
    "CREATE CONSTRAINT table_qsid IF NOT EXISTS FOR (t:FactorTable) REQUIRE t.QSID IS UNIQUE",
    "CREATE CONSTRAINT fdb_name IF NOT EXISTS FOR (d:FactorDB) REQUIRE d.Name IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.Name IS UNIQUE",
]

_SCHEMA_INDEXES = [
    "CREATE INDEX factor_name IF NOT EXISTS FOR (f:Factor) ON (f.Name)",
    "CREATE INDEX factor_class IF NOT EXISTS FOR (f:Factor) ON (f.FactorClass)",
    "CREATE INDEX factor_op_name IF NOT EXISTS FOR (f:Factor) ON (f.OperatorName)",
    "CREATE INDEX factor_op_type IF NOT EXISTS FOR (f:Factor) ON (f.OperatorType)",
    "CREATE INDEX operator_name IF NOT EXISTS FOR (o:FactorOperator) ON (o.Name)",
    "CREATE INDEX operator_type IF NOT EXISTS FOR (o:FactorOperator) ON (o.OperatorType)",
    "CREATE INDEX fdb_type IF NOT EXISTS FOR (d:FactorDB) ON (d.DBType)",
]

# endregion


class FactorGraphDB(__QS_Object__):
    """基于 Neo4j 的因子图数据库

    因子注册中心的核心存储引擎，存储因子元数据、依赖关系图和数据引用。
    支持因子检索、重建计算、依赖分析和影响范围查询。

    参数通过 ~/QuantStudioConfig/FactorGraphDBConfig.json 配置或显式传入。
    """

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name: str = Field(default="FactorGraphDB", frozen=True, title="图数据库名称")
        Neo4jURI: str = Field(default="bolt://localhost:7687", frozen=True, exclude=True, title="Neo4j 连接 URI")
        Neo4jUser: str = Field(default="neo4j", frozen=True, exclude=True, title="Neo4j 用户名")
        Neo4jPwd: str = Field(default="", frozen=True, exclude=True, repr=False, title="Neo4j 密码")
        Neo4jDB: str = Field(default="neo4j", frozen=True, exclude=True, title="Neo4j 数据库名")
        DataDir: Optional[str] = Field(default=None, frozen=False, exclude=True, title="数据因子内联数据存储目录")

    def __init__(self, args: dict = {}, config_file: Optional[str] = None, **kwargs):
        if neo4j is None:
            raise ImportError("FactorGraphDB 需要 neo4j 包，请执行: pip install neo4j")
        super().__init__(
            args=args,
            config_file=(__QS_ConfigPath__ + os.sep + "FactorGraphDBConfig.json" if config_file is None else config_file),
            **kwargs
        )
        self._Driver: Optional[neo4j.Driver] = None
        self._FactorDBRegistry: Dict[str, FactorDB] = {}
        if self._QSArgs.DataDir is None:
            self._QSArgs.DataDir = os.path.join(tempfile.gettempdir(), "QS_FactorGraphDB_Data")
        os.makedirs(self._QSArgs.DataDir, exist_ok=True)

    # region 生命周期

    def connect(self) -> "FactorGraphDB":
        """连接到 Neo4j 数据库，首次连接自动创建约束和索引"""
        self._Driver = neo4j.GraphDatabase.driver(
            self._QSArgs.Neo4jURI,
            auth=(self._QSArgs.Neo4jUser, self._QSArgs.Neo4jPwd),
            database=self._QSArgs.Neo4jDB
        )
        # 验证连接
        self._Driver.verify_connectivity()
        self._initSchema()
        self._QS_Logger.info(f"FactorGraphDB 已连接到 {self._QSArgs.Neo4jURI}")
        return self

    def disconnect(self) -> int:
        """断开 Neo4j 连接"""
        if self._Driver:
            self._Driver.close()
            self._Driver = None
            self._QS_Logger.info("FactorGraphDB 已断开连接")
        return 0

    def _initSchema(self):
        """初始化数据库 schema（约束和索引）"""
        with self._Driver.session() as session:
            for stmt in _SCHEMA_CONSTRAINTS + _SCHEMA_INDEXES:
                session.run(stmt)

    def _runCypher(self, query: str, parameters: Optional[Dict] = None) -> list:
        """执行 Cypher 查询并返回结果

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            记录列表（每条记录转为 dict）
        """
        with self._Driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    # endregion

    # region 存储（Store）

    def registerFactorDB(self, fdb: FactorDB) -> str:
        """注册因子库到图数据库和内存注册表

        Args:
            fdb: QuantStudio FactorDB 实例

        Returns:
            因子库名称
        """
        props = {
            "Name": fdb.Name,
            "DBType": fdb.__class__.__name__,
            "ClassName": fdb.__class__.__name__,
            "ModulePath": fdb.__class__.__module__,
            "ConnectionJSON": self._extractFDBConnection(fdb),
            "UpdatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self._runCypher(
            """
            MERGE (d:FactorDB {Name: $name})
            ON CREATE SET d += $props, d.CreatedAt = $now
            ON MATCH SET d += $props
            """,
            {"name": fdb.Name, "props": props, "now": dt.datetime.now(dt.timezone.utc).isoformat()}
        )
        self._FactorDBRegistry[fdb.Name] = fdb
        self._QS_Logger.info(f"已注册因子库: {fdb.Name}")
        return fdb.Name

    def _extractFDBConnection(self, fdb: FactorDB) -> str:
        """提取因子库的连接信息"""
        conn_info = {"type": fdb.__class__.__name__}
        if hasattr(fdb._QSArgs, "MainDir"):
            conn_info["MainDir"] = str(fdb._QSArgs.MainDir)
        return json.dumps(conn_info, ensure_ascii=False)

    def storeFactorOperator(self, op: FactorOperator) -> str:
        """存储单个算子节点

        Args:
            op: FactorOperator 实例

        Returns:
            算子 QSID
        """
        calc_ref_json, is_custom = serializeOperatorCalculateRef(op)
        look_back = getattr(op._QSArgs, "LookBack", [])
        props = {
            "Name": op._QSArgs.Name,
            "QSID": op.QSID,
            "ClassName": op.__class__.__name__,
            "ModulePath": op.__class__.__module__,
            "OperatorType": op._QSArgs.OperatorType,
            "Arity": op._QSArgs.Arity,
            "DataType": op._QSArgs.DataType,
            "Description": op._QSArgs.Description,
            "ModelArgsJSON": serializeOperatorArgs(op),
            "LookBackJSON": json.dumps(_sanitizeForJSON(look_back), ensure_ascii=False),
            "CalculateRef": calc_ref_json,
            "IsCustom": is_custom,
            "UpdatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self._runCypher(
            """
            MERGE (o:FactorOperator {QSID: $qsid})
            ON CREATE SET o += $props, o.CreatedAt = $now
            ON MATCH SET o += $props
            """,
            {"qsid": op.QSID, "props": props, "now": dt.datetime.now(dt.timezone.utc).isoformat()}
        )
        return op.QSID

    def storeFactorTable(self, ft: FactorTable, fdb_name: Optional[str] = None) -> str:
        """存储因子表节点

        Args:
            ft: FactorTable 实例
            fdb_name: 关联的因子库名称

        Returns:
            因子表 QSID
        """
        props = {
            "Name": ft._QSArgs.Name,
            "QSID": ft.QSID,
            "FactorNamesJSON": json.dumps(ft.FactorNames, ensure_ascii=False),
            "MetaDataJSON": json.dumps(_sanitizeForJSON(ft.getMetaData(key=None).to_dict()) if hasattr(ft.getMetaData(key=None), 'to_dict') else {}, ensure_ascii=False),
            "UpdatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self._runCypher(
            """
            MERGE (t:FactorTable {QSID: $qsid})
            ON CREATE SET t += $props, t.CreatedAt = $now
            ON MATCH SET t += $props
            """,
            {"qsid": ft.QSID, "props": props, "now": dt.datetime.now(dt.timezone.utc).isoformat()}
        )
        # 建立 IN_DATABASE 关系
        actual_fdb_name = fdb_name or (ft.FactorDB.Name if ft.FactorDB else None)
        if actual_fdb_name:
            self._runCypher(
                """
                MATCH (t:FactorTable {QSID: $t_qsid})
                MATCH (d:FactorDB {Name: $fdb_name})
                MERGE (t)-[:IN_DATABASE]->(d)
                """,
                {"t_qsid": ft.QSID, "fdb_name": actual_fdb_name}
            )
        return ft.QSID

    def storeFactor(self, factor: Factor, tags: Optional[List[str]] = None) -> str:
        """递归存储因子及其完整依赖 DAG

        Args:
            factor: 根因子
            tags: 可选标签列表

        Returns:
            因子的 QSID
        """
        # 收集完整 DAG
        dag_nodes = []
        visited = set()
        self._collectDAG(factor, dag_nodes, visited)
        # 拓扑排序
        sorted_nodes = self._topologicalSort(dag_nodes)
        # 批量存储
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        for node in sorted_nodes:
            self._storeFactorNode(node, now)
        # 存储标签
        if tags:
            for tag_name in tags:
                self._runCypher(
                    """
                    MERGE (t:Tag {Name: $tag_name})
                    WITH t
                    MATCH (f:Factor {QSID: $qsid})
                    MERGE (f)-[:TAGGED]->(t)
                    """,
                    {"tag_name": tag_name, "qsid": factor.QSID}
                )
        self._QS_Logger.info(f"已存储因子: {factor._QSArgs.Name} (QSID: {factor.QSID[:12]}...)")
        return factor.QSID

    def _collectDAG(self, factor: Factor, dag_nodes: list, visited: set):
        """递归收集因子的依赖 DAG"""
        qsid = factor.QSID
        if qsid in visited:
            return
        visited.add(qsid)
        # 先收集依赖
        if factor.FactorTable:
            # FactorTableFactor: 依赖因子表
            self._collectDAGFromTable(factor.FactorTable, visited)
        for desc in factor.Descriptors:
            self._collectDAG(desc, dag_nodes, visited)
        dag_nodes.append(factor)

    def _collectDAGFromTable(self, ft: FactorTable, visited: set):
        """收集因子表及其因子库"""
        ft_qsid = ft.QSID
        if ft_qsid in visited:
            return
        visited.add(ft_qsid)
        # 存储因子表
        if ft.FactorDB:
            fdb_name = ft.FactorDB.Name
            if fdb_name not in self._FactorDBRegistry:
                self.registerFactorDB(ft.FactorDB)
            self.storeFactorTable(ft, fdb_name=fdb_name)
        else:
            self.storeFactorTable(ft)

    def _topologicalSort(self, dag_nodes: list) -> list:
        """拓扑排序（叶子节点在前）"""
        qsid_to_node = {n.QSID: n for n in dag_nodes}
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for node in dag_nodes:
            qsid = node.QSID
            in_degree.setdefault(qsid, 0)
            for desc in node.Descriptors:
                if desc.QSID in qsid_to_node:
                    adj[desc.QSID].append(qsid)
                    in_degree[qsid] += 1
        queue = [q for q, d in in_degree.items() if d == 0]
        result = []
        while queue:
            q = queue.pop(0)
            result.append(qsid_to_node[q])
            for neighbor in adj[q]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return result

    def _storeFactorNode(self, factor: Factor, now: str):
        """存储单个因子节点及其关系"""
        props = self._serializeFactor(factor)
        # 存储因子节点
        self._runCypher(
            """
            MERGE (f:Factor {QSID: $qsid})
            ON CREATE SET f += $props, f.CreatedAt = $now
            ON MATCH SET f += $props
            """,
            {"qsid": factor.QSID, "props": props, "now": now}
        )
        # 存储算子关系
        if isinstance(factor, DerivativeFactor) and factor.Operator:
            self.storeFactorOperator(factor.Operator)
            self._runCypher(
                """
                MATCH (f:Factor {QSID: $f_qsid})
                MATCH (o:FactorOperator {QSID: $o_qsid})
                MERGE (f)-[:USES_OPERATOR]->(o)
                """,
                {"f_qsid": factor.QSID, "o_qsid": factor.Operator.QSID}
            )
        # 存储因子表关系
        if factor.FactorTable:
            self._runCypher(
                """
                MATCH (f:Factor {QSID: $f_qsid})
                MATCH (t:FactorTable {QSID: $t_qsid})
                MERGE (f)-[:BELONGS_TO]->(t)
                """,
                {"f_qsid": factor.QSID, "t_qsid": factor.FactorTable.QSID}
            )
        # 存储依赖关系
        descriptors = factor.Descriptors
        for i, desc in enumerate(descriptors):
            self._runCypher(
                """
                MATCH (source:Factor {QSID: $source_qsid})
                MATCH (target:Factor {QSID: $target_qsid})
                MERGE (source)-[r:DEPENDS_ON]->(target)
                SET r.order = $order
                """,
                {"source_qsid": factor.QSID, "target_qsid": desc.QSID, "order": i}
            )

    def _serializeFactor(self, factor: Factor) -> dict:
        """序列化因子为 Neo4j 节点属性字典"""
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        props = {
            "Name": factor._QSArgs.Name,
            "QSID": factor.QSID,
            "ClassName": factor.__class__.__name__,
            "ModulePath": factor.__class__.__module__,
            "MetaJSON": json.dumps(_sanitizeForJSON(factor._QSArgs.Meta), ensure_ascii=False),
            "QSArgsJSON": serializeFactorArgs(factor),
            "UpdatedAt": now,
        }
        # 判定因子类别
        if isinstance(factor, DataFactor):
            props["FactorClass"] = "DataFactor"
            props["DataType"] = factor._QSArgs.DataType
            props["DataRef"] = self._serializeDataRef(factor)
        elif isinstance(factor, DerivativeFactor):
            props["FactorClass"] = "DerivativeFactor"
            if factor.Operator:
                props["OperatorName"] = factor.Operator._QSArgs.Name
                props["OperatorType"] = factor.Operator._QSArgs.OperatorType
                props["OperatorQSID"] = factor.Operator.QSID
                props["DataType"] = factor.Operator._QSArgs.DataType
        elif factor.FactorTable:
            props["FactorClass"] = "FactorTableFactor"
            props["FactorTableQSID"] = factor.FactorTable.QSID
            props["FactorTableName"] = factor._QSArgs.Name
            # 从因子表获取 DataType
            try:
                meta = factor.getMetaData(key="DataType")
                props["DataType"] = meta if meta else "double"
            except Exception:
                props["DataType"] = "double"
        else:
            props["FactorClass"] = "Factor"
            try:
                meta = factor.getMetaData(key="DataType")
                props["DataType"] = meta if meta else "double"
            except Exception:
                props["DataType"] = "double"
        return props

    def _serializeDataRef(self, factor: DataFactor) -> str:
        """序列化 DataFactor 的数据引用"""
        data = factor._Data
        dtype = factor._QSArgs.DataType
        content = factor._DataContent
        if content == "Value":
            ref = {"type": "scalar", "value": _sanitizeForJSON(data), "dtype": dtype}
        else:
            # 写入 HDF5 文件
            qsid = factor.QSID
            subdir = os.path.join(self._QSArgs.DataDir, qsid[:8])
            os.makedirs(subdir, exist_ok=True)
            filepath = os.path.join(subdir, f"{qsid}.hdf5")
            import h5py
            with h5py.File(filepath, "w") as f:
                if content == "Factor":
                    f.create_dataset("DateTime", data=[t.timestamp() for t in data.index])
                    f.create_dataset("ID", data=np.array(data.columns.tolist(), dtype="S"))
                    f.create_dataset("Data", data=data.values)
                elif content == "DateTime":
                    f.create_dataset("DateTime", data=[t.timestamp() for t in data.index])
                    f.create_dataset("Data", data=data.values)
                elif content == "ID":
                    f.create_dataset("ID", data=np.array(data.index.tolist(), dtype="S"))
                    f.create_dataset("Data", data=data.values)
            ref = {"type": content.lower(), "file": filepath, "dtype": dtype}
        return json.dumps(ref, ensure_ascii=False)

    # endregion

    # region 检索（Retrieve）

    def getFactorByQSID(self, qsid: str) -> Optional[Dict]:
        """按 QSID 查询因子节点"""
        results = self._runCypher(
            "MATCH (f:Factor {QSID: $qsid}) RETURN f",
            {"qsid": qsid}
        )
        return results[0]["f"] if results else None

    def searchFactors(self, name: Optional[str] = None, operator_type: Optional[str] = None,
                      operator_name: Optional[str] = None, tag: Optional[str] = None,
                      factor_class: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """多条件组合搜索因子

        Args:
            name: 因子名称（模糊匹配）
            operator_type: 算子类型（Point/Time/Section/Panel）
            operator_name: 算子名称
            tag: 标签名称
            factor_class: 因子类别（DataFactor/DerivativeFactor/FactorTableFactor）
            limit: 返回数量上限

        Returns:
            因子属性字典列表
        """
        conditions = []
        params = {"limit": limit}
        if name:
            conditions.append("f.Name CONTAINS $name")
            params["name"] = name
        if operator_type:
            conditions.append("f.OperatorType = $op_type")
            params["op_type"] = operator_type
        if operator_name:
            conditions.append("f.OperatorName = $op_name")
            params["op_name"] = operator_name
        if factor_class:
            conditions.append("f.FactorClass = $factor_class")
            params["factor_class"] = factor_class
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        if tag:
            query = f"""
                MATCH (f:Factor)-[:TAGGED]->(t:Tag {{Name: $tag}})
                {where_clause}
                RETURN f ORDER BY f.Name LIMIT $limit
            """
            params["tag"] = tag
        else:
            query = f"""
                MATCH (f:Factor)
                {where_clause}
                RETURN f ORDER BY f.Name LIMIT $limit
            """
        results = self._runCypher(query, params)
        return [r["f"] for r in results]

    def getDependencyGraph(self, qsid: str, direction: str = "both") -> Dict:
        """获取因子的依赖子图

        Args:
            qsid: 目标因子 QSID
            direction: "down"（输入）/ "up"（下游）/ "both"（双向）

        Returns:
            {"root": qsid, "nodes": [...], "edges": [...]}
        """
        nodes = {}
        edges = []
        if direction in ("down", "both"):
            results = self._runCypher(
                """
                MATCH path = (root:Factor {QSID: $qsid})-[:DEPENDS_ON*]->(leaf:Factor)
                UNWIND nodes(path) AS n
                WITH DISTINCT n
                RETURN n
                """,
                {"qsid": qsid}
            )
            for r in results:
                n = r["n"]
                nodes[n["QSID"]] = n
            # 收集边
            edge_results = self._runCypher(
                """
                MATCH (a:Factor)-[r:DEPENDS_ON]->(b:Factor)
                WHERE a.QSID = $qsid OR b.QSID = $qsid
                   OR a.QSID IN $node_ids OR b.QSID IN $node_ids
                RETURN a.QSID AS source, b.QSID AS target, r.order AS order
                """,
                {"qsid": qsid, "node_ids": list(nodes.keys())}
            )
            edges.extend(edge_results)
        if direction in ("up", "both"):
            results = self._runCypher(
                """
                MATCH (dependent:Factor)-[:DEPENDS_ON*]->(target:Factor {QSID: $qsid})
                RETURN DISTINCT dependent
                """,
                {"qsid": qsid}
            )
            for r in results:
                n = r["dependent"]
                nodes[n["QSID"]] = n
            # 补充边
            if nodes:
                edge_results = self._runCypher(
                    """
                    MATCH (a:Factor)-[r:DEPENDS_ON]->(b:Factor)
                    WHERE a.QSID IN $node_ids AND b.QSID IN $node_ids
                    RETURN a.QSID AS source, b.QSID AS target, r.order AS order
                    """,
                    {"node_ids": list(nodes.keys())}
                )
                existing = {(e["source"], e["target"]) for e in edges}
                for e in edge_results:
                    if (e["source"], e["target"]) not in existing:
                        edges.append(e)
                        existing.add((e["source"], e["target"]))
        # 确保根节点在 nodes 中
        if qsid not in nodes:
            root = self.getFactorByQSID(qsid)
            if root:
                nodes[qsid] = root
        return {"root": qsid, "nodes": list(nodes.values()), "edges": edges}

    def getDescriptors(self, qsid: str) -> List[Dict]:
        """获取因子的直接依赖因子（有序）"""
        results = self._runCypher(
            """
            MATCH (f:Factor {QSID: $qsid})-[r:DEPENDS_ON]->(d:Factor)
            RETURN d, r.order AS order
            ORDER BY r.order
            """,
            {"qsid": qsid}
        )
        return [r["d"] for r in results]

    def getDependents(self, qsid: str, transitive: bool = False) -> List[Dict]:
        """获取依赖该因子的因子

        Args:
            qsid: 目标因子 QSID
            transitive: 是否传递闭包

        Returns:
            因子属性字典列表
        """
        if transitive:
            results = self._runCypher(
                """
                MATCH (dependent:Factor)-[:DEPENDS_ON*]->(target:Factor {QSID: $qsid})
                RETURN DISTINCT dependent
                """,
                {"qsid": qsid}
            )
        else:
            results = self._runCypher(
                """
                MATCH (dependent:Factor)-[:DEPENDS_ON]->(target:Factor {QSID: $qsid})
                RETURN dependent
                """,
                {"qsid": qsid}
            )
        return [r["dependent"] for r in results]

    def findOrphanFactors(self) -> List[Dict]:
        """查找无下游依赖且不属于因子表的叶子因子"""
        results = self._runCypher(
            """
            MATCH (f:Factor)
            WHERE NOT (f)<-[:DEPENDS_ON]-()
              AND NOT (f)-[:BELONGS_TO]->(:FactorTable)
            RETURN f
            """
        )
        return [r["f"] for r in results]

    # endregion

    # region 重建（Reconstruct）

    def reconstructFactor(self, qsid: str, descriptor_map: Optional[Dict[str, Factor]] = None) -> Factor:
        """从图中的元数据重建可计算的 Factor 对象

        Args:
            qsid: 目标因子的 QSID
            descriptor_map: 可选的预构建描述子映射 {QSID: Factor}

        Returns:
            可直接在计算引擎中使用的 Factor 实例
        """
        if descriptor_map is None:
            descriptor_map = {}
        if qsid in descriptor_map:
            return descriptor_map[qsid]
        factor_data = self.getFactorByQSID(qsid)
        if factor_data is None:
            raise __QS_Error__(f"图中不存在 QSID 为 {qsid} 的因子")
        factor_class = factor_data.get("FactorClass", "Factor")
        if factor_class == "DataFactor":
            factor = self._reconstructDataFactor(factor_data)
        elif factor_class == "DerivativeFactor":
            factor = self._reconstructDerivativeFactor(factor_data, descriptor_map)
        elif factor_class == "FactorTableFactor":
            factor = self._reconstructFactorTableFactor(factor_data)
        else:
            raise __QS_Error__(f"不支持的因子类别: {factor_class}")
        descriptor_map[qsid] = factor
        return factor

    def _reconstructDataFactor(self, factor_data: dict) -> DataFactor:
        """重建 DataFactor"""
        data_ref = json.loads(factor_data.get("DataRef", "{}"))
        args = json.loads(factor_data.get("QSArgsJSON", "{}"))
        args = _desanitizeFromJSON(args)
        args["Name"] = factor_data["Name"]
        ref_type = data_ref.get("type", "")
        if ref_type == "scalar":
            data = _desanitizeFromJSON(data_ref["value"])
        elif ref_type in ("factor", "datetime", "id"):
            filepath = data_ref["file"]
            import h5py
            with h5py.File(filepath, "r") as f:
                if ref_type == "factor":
                    dts = [dt.datetime.fromtimestamp(t) for t in f["DateTime"][:]]
                    ids = [s.decode("utf-8") for s in f["ID"][:]]
                    data = pd.DataFrame(f["Data"][:], index=dts, columns=ids)
                elif ref_type == "datetime":
                    dts = [dt.datetime.fromtimestamp(t) for t in f["DateTime"][:]]
                    data = pd.Series(f["Data"][:], index=dts)
                elif ref_type == "id":
                    ids = [s.decode("utf-8") for s in f["ID"][:]]
                    data = pd.Series(f["Data"][:], index=ids)
        else:
            raise __QS_Error__(f"不支持的 DataRef 类型: {ref_type}")
        return DataFactor(data=data, args=args)

    def _reconstructDerivativeFactor(self, factor_data: dict, descriptor_map: dict) -> Factor:
        """重建 DerivativeFactor"""
        # 重建算子
        operator = self._reconstructOperatorFromData(factor_data)
        # 重建描述子
        descriptors_data = self.getDescriptors(factor_data["QSID"])
        descriptors = []
        for desc_data in descriptors_data:
            desc = self.reconstructFactor(desc_data["QSID"], descriptor_map)
            descriptors.append(desc)
        # 解析参数
        args = json.loads(factor_data.get("QSArgsJSON", "{}"))
        args = _desanitizeFromJSON(args)
        args["Name"] = factor_data["Name"]
        args["Operator"] = operator
        # 调用算子生成因子
        return operator(*descriptors, factor_args=args)

    def _reconstructOperatorFromData(self, factor_data: dict) -> FactorOperator:
        """从因子数据重建其算子"""
        op_qsid = factor_data.get("OperatorQSID")
        if not op_qsid:
            raise __QS_Error__(f"因子 {factor_data['Name']} 没有关联的算子")
        return self.reconstructOperator(op_qsid)

    def _reconstructFactorTableFactor(self, factor_data: dict) -> Factor:
        """重建 FactorTableFactor"""
        ft_qsid = factor_data.get("FactorTableQSID")
        if not ft_qsid:
            raise __QS_Error__(f"因子 {factor_data['Name']} 没有关联的因子表")
        # 查找因子表关联的因子库
        fdb_results = self._runCypher(
            """
            MATCH (t:FactorTable {QSID: $ft_qsid})-[:IN_DATABASE]->(d:FactorDB)
            RETURN d
            """,
            {"ft_qsid": ft_qsid}
        )
        if not fdb_results:
            raise __QS_Error__(f"因子表 {ft_qsid} 没有关联的因子库，请先调用 registerFactorDB")
        fdb_name = fdb_results[0]["d"]["Name"]
        if fdb_name not in self._FactorDBRegistry:
            raise __QS_Error__(f"因子库 {fdb_name} 未注册，请先调用 registerFactorDB")
        fdb = self._FactorDBRegistry[fdb_name]
        # 获取因子表名称
        ft_data = self._runCypher(
            "MATCH (t:FactorTable {QSID: $qsid}) RETURN t",
            {"qsid": ft_qsid}
        )
        ft_name = ft_data[0]["t"]["Name"] if ft_data else factor_data["FactorTableName"]
        ft = fdb.getTable(ft_name)
        args = json.loads(factor_data.get("QSArgsJSON", "{}"))
        args = _desanitizeFromJSON(args)
        args["Name"] = factor_data["FactorTableName"]
        return ft.getFactor(factor_data["FactorTableName"], args=args)

    def reconstructOperator(self, qsid: str) -> FactorOperator:
        """从图中重建算子对象"""
        results = self._runCypher(
            "MATCH (o:FactorOperator {QSID: $qsid}) RETURN o",
            {"qsid": qsid}
        )
        if not results:
            raise __QS_Error__(f"图中不存在 QSID 为 {qsid} 的算子")
        op_data = results[0]["o"]
        # 导入算子类
        module_path = op_data["ModulePath"]
        class_name = op_data["ClassName"]
        module = importlib.import_module(module_path)
        op_class = getattr(module, class_name)
        # 解析参数
        model_args = json.loads(op_data.get("ModelArgsJSON", "{}"))
        model_args = _desanitizeFromJSON(model_args)
        look_back = json.loads(op_data.get("LookBackJSON", "[]"))
        look_back = _desanitizeFromJSON(look_back)
        args = {
            "Name": op_data["Name"],
            "ModelArgs": model_args,
            "DataType": op_data.get("DataType", "double"),
            "Arity": op_data.get("Arity"),
        }
        if look_back:
            args["LookBack"] = look_back
        # 实例化算子
        op = op_class(args=args)
        # 处理自定义算子
        if op_data.get("IsCustom", False) and op_data.get("CalculateRef"):
            calc_ref = json.loads(op_data["CalculateRef"])
            if "__func_ref__" in calc_ref:
                op.calculate = _deserializeFuncRef(calc_ref["__func_ref__"])
            elif "__dill__" in calc_ref:
                import dill
                op.calculate = dill.loads(base64.b64decode(calc_ref["__dill__"]))
            elif "__numpy_func__" in calc_ref:
                op.calculate = getattr(np, calc_ref["name"])
        return op

    # endregion

    # region 管理（Manage）

    def deleteFactor(self, qsid: str, cascade: bool = False) -> int:
        """删除因子节点及其关系

        Args:
            qsid: 目标因子 QSID
            cascade: 是否级联删除孤立依赖

        Returns:
            删除的节点数
        """
        deleted = 0
        if cascade:
            # 递归删除孤立因子
            to_delete = [qsid]
            while to_delete:
                current = to_delete.pop(0)
                # 检查是否有其他因子依赖当前因子
                dependents = self.getDependents(current, transitive=False)
                if len(dependents) == 0 or current == qsid:
                    # 没有其他依赖者，可以删除
                    self._runCypher(
                        "MATCH (f:Factor {QSID: $qsid}) DETACH DELETE f",
                        {"qsid": current}
                    )
                    deleted += 1
                    # 检查该因子的描述子是否变为孤立
                    descs = self.getDescriptors(current)
                    for desc in descs:
                        desc_qsid = desc["QSID"]
                        remaining = self.getDependents(desc_qsid, transitive=False)
                        if len(remaining) == 0:
                            to_delete.append(desc_qsid)
        else:
            self._runCypher(
                "MATCH (f:Factor {QSID: $qsid}) DETACH DELETE f",
                {"qsid": qsid}
            )
            deleted = 1
        return deleted

    def updateFactorMetaData(self, qsid: str, meta: Dict) -> None:
        """更新因子的元信息"""
        existing = self.getFactorByQSID(qsid)
        if not existing:
            raise __QS_Error__(f"因子 {qsid} 不存在")
        old_meta = json.loads(existing.get("MetaJSON", "{}"))
        old_meta = _desanitizeFromJSON(old_meta)
        old_meta.update(meta)
        self._runCypher(
            "MATCH (f:Factor {QSID: $qsid}) SET f.MetaJSON = $meta",
            {"qsid": qsid, "meta": json.dumps(_sanitizeForJSON(old_meta), ensure_ascii=False)}
        )

    def updateFactorTags(self, qsid: str, add_tags: Optional[List[str]] = None,
                         remove_tags: Optional[List[str]] = None) -> None:
        """增删因子标签"""
        if add_tags:
            for tag_name in add_tags:
                self._runCypher(
                    """
                    MERGE (t:Tag {Name: $tag_name})
                    WITH t
                    MATCH (f:Factor {QSID: $qsid})
                    MERGE (f)-[:TAGGED]->(t)
                    """,
                    {"tag_name": tag_name, "qsid": qsid}
                )
        if remove_tags:
            for tag_name in remove_tags:
                self._runCypher(
                    """
                    MATCH (f:Factor {QSID: $qsid})-[r:TAGGED]->(t:Tag {Name: $tag_name})
                    DELETE r
                    """,
                    {"qsid": qsid, "tag_name": tag_name}
                )

    def renameFactor(self, qsid: str, new_name: str) -> None:
        """更新因子名称"""
        self._runCypher(
            "MATCH (f:Factor {QSID: $qsid}) SET f.Name = $name",
            {"qsid": qsid, "name": new_name}
        )

    # endregion

    # region 分析（Analyze）

    def impactAnalysis(self, qsid: str) -> List[Dict]:
        """影响范围分析，返回所有传递依赖该因子的下游因子"""
        results = self._runCypher(
            """
            MATCH (impacted:Factor)-[:DEPENDS_ON*1..]->(changed:Factor {QSID: $qsid})
            RETURN impacted,
                   length(shortestPath((impacted)-[:DEPENDS_ON*]->(changed))) AS depth
            ORDER BY depth
            """,
            {"qsid": qsid}
        )
        return results

    def findSimilarFactors(self, qsid: str, limit: int = 20) -> List[Dict]:
        """查找使用相同算子的相似因子"""
        results = self._runCypher(
            """
            MATCH (f:Factor {QSID: $qsid})-[:USES_OPERATOR]->(o:FactorOperator)
            MATCH (other:Factor)-[:USES_OPERATOR]->(o2:FactorOperator)
            WHERE o2.OperatorType = o.OperatorType
              AND o2.Name = o.Name
              AND other.QSID <> $qsid
            RETURN other, o2 LIMIT $limit
            """,
            {"qsid": qsid, "limit": limit}
        )
        return results

    def getGraphStats(self) -> Dict[str, int]:
        """返回各类节点和关系的计数统计"""
        stats = {}
        for label in ["Factor", "FactorOperator", "FactorTable", "FactorDB", "Tag"]:
            results = self._runCypher(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            stats[label] = results[0]["cnt"] if results else 0
        for rel in ["DEPENDS_ON", "USES_OPERATOR", "BELONGS_TO", "TAGGED", "IN_DATABASE"]:
            results = self._runCypher(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt")
            stats[rel] = results[0]["cnt"] if results else 0
        return stats

    # endregion

    # region 工具

    def executeCypher(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """执行原始 Cypher 查询"""
        return self._runCypher(query, parameters)

    def _repr_html_(self) -> str:
        HTML = f"<b>类</b>: FactorGraphDB<br/>"
        HTML += f"<b>Neo4j URI</b>: {html.escape(self._QSArgs.Neo4jURI)}<br/>"
        HTML += f"<b>数据库</b>: {html.escape(self._QSArgs.Neo4jDB)}<br/>"
        HTML += f"<b>连接状态</b>: {'已连接' if self._Driver else '未连接'}<br/>"
        if self._Driver:
            stats = self.getGraphStats()
            HTML += "<b>图统计</b>:<br/>"
            HTML += "<ul>"
            for key, val in stats.items():
                HTML += f"<li>{html.escape(key)}: {val}</li>"
            HTML += "</ul>"
        return HTML

    # endregion
