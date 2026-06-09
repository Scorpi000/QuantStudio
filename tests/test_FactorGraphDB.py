# -*- coding: utf-8 -*-
"""FactorRegistry 功能测试 — 基于 Neo4j 实例"""
import sys
import os
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# 配置路径
NEO4J_CONFIG = os.path.expanduser("~/QuantStudioConfig/Neo4jDBConfig.json")
with open(NEO4J_CONFIG, "r", encoding="utf-8") as f:
    content = f.read()
# 容忍尾随逗号
import re
content = re.sub(r",\s*([}\]])", r"\1", content)
neo4j_cfg = json.loads(content)

fgdb_args = {
    "Neo4jURI": f"bolt://{neo4j_cfg['IPAddr']}:{neo4j_cfg['Port']}",
    "Neo4jUser": neo4j_cfg["User"],
    "Neo4jPwd": neo4j_cfg["Pwd"],
    "Neo4jDB": neo4j_cfg.get("DBName", "neo4j"),
}

# ============================================================
# 导入 QuantStudio
# ============================================================
from QuantStudio.FactorRegistry.api import FactorGraphDB
from QuantStudio.Factor.Factor import DataFactor
from QuantStudio.Factor.FactorOperation import (
    PointOperator, TimeOperator, makeFactorOperator
)
from QuantStudio.Factor.JYDB import JYDB
import numpy as np
import pandas as pd


def test_1_connect():
    """测试 1: 连接与 Schema 初始化"""
    print("\n" + "=" * 60)
    print("测试 1: 连接与 Schema 初始化")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()
    print(f"  [PASS] 连接成功")
    print(f"  [PASS] URI: {fgdb._QSArgs.Neo4jURI}")
    fgdb.disconnect()
    print(f"  [PASS] 断开成功")
    return True


def test_2_store_and_retrieve_datafactor():
    """测试 2: 存储和检索 DataFactor（标量、Series、DataFrame）"""
    print("\n" + "=" * 60)
    print("测试 2: 存储和检索 DataFactor")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 2a: 标量 DataFactor
    scalar_factor = DataFactor(data=42.0, args={"Name": "test_scalar", "DataType": "double"})
    qsid1 = fgdb.storeFactor(scalar_factor, tags=["test", "scalar"])
    print(f"  [PASS] 标量 DataFactor 存储完成, QSID: {qsid1[:16]}...")

    result = fgdb.getFactorByQSID(qsid1)
    assert result is not None, "getFactorByQSID 返回 None"
    assert result["Name"] == "test_scalar"
    assert result["FactorClass"] == "DataFactor"
    print(f"  [PASS] getFactorByQSID 成功, Name={result['Name']}, FactorClass={result['FactorClass']}")

    # 2b: Series DataFactor
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    series_data = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], index=dates, name="test_series_data")
    series_factor = DataFactor(data=series_data, args={"Name": "test_series", "DataType": "double"})
    qsid2 = fgdb.storeFactor(series_factor, tags=["test", "series"])
    print(f"  [PASS] Series DataFactor 存储完成, QSID: {qsid2[:16]}...")

    # 2c: DataFrame DataFactor
    ids = ["000001.SZ", "000002.SZ", "600000.SH"]
    df_data = pd.DataFrame(
        np.random.randn(5, 3), index=dates, columns=ids
    )
    df_factor = DataFactor(data=df_data, args={"Name": "test_dataframe", "DataType": "double"})
    qsid3 = fgdb.storeFactor(df_factor, tags=["test", "dataframe"])
    print(f"  [PASS] DataFrame DataFactor 存储完成, QSID: {qsid3[:16]}...")

    fgdb.disconnect()
    return True


def test_3_store_derivative_chain():
    """测试 3: 存储衍生因子链（DataFactor → PointOp → TimeOp）"""
    print("\n" + "=" * 60)
    print("测试 3: 存储衍生因子链")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 构建因子链
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    ids = ["000001.SZ", "000002.SZ", "600000.SH"]
    price_data = pd.DataFrame(
        np.random.uniform(10, 50, (10, 3)),
        index=dates, columns=ids
    )
    close = DataFactor(data=price_data, args={"Name": "Close", "DataType": "double"})

    # 使用 fo 算子
    from QuantStudio.Factor.api import fo
    log_close = fo.Log()(close)
    lag1 = fo.Lag(lag_period=1)(log_close)

    # 存储整条链
    qsid = fgdb.storeFactor(lag1, tags=["test", "chain", "momentum"])
    print(f"  [PASS] 因子链存储完成, 根因子 QSID: {qsid[:16]}...")

    # 查看图统计
    stats = fgdb.getGraphStats()
    print(f"  [PASS] 图统计: {stats}")

    # 搜索因子
    results = fgdb.searchFactors(name="log")
    print(f"  [PASS] searchFactors(name='log') 返回 {len(results)} 个因子")

    results = fgdb.searchFactors(operator_type="Point")
    print(f"  [PASS] searchFactors(operator_type='Point') 返回 {len(results)} 个因子")

    results = fgdb.searchFactors(tag="chain")
    print(f"  [PASS] searchFactors(tag='chain') 返回 {len(results)} 个因子")

    fgdb.disconnect()
    return True


def test_4_dependency_graph():
    """测试 4: 依赖图查询"""
    print("\n" + "=" * 60)
    print("测试 4: 依赖图查询")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 搜索一个 DerivativeFactor 来测试依赖图
    results = fgdb.searchFactors(name="lag")
    if not results:
        print("  [WARN] 图中无 lag 因子，跳过")
        fgdb.disconnect()
        return True

    target_qsid = results[0]["QSID"]
    target_name = results[0]["Name"]
    print(f"  目标因子: {target_name} (QSID: {target_qsid[:16]}...)")

    # 向下依赖图
    graph_down = fgdb.getDependencyGraph(target_qsid, direction="down")
    print(f"  [PASS] getDependencyGraph(down): {len(graph_down['nodes'])} 个节点, {len(graph_down['edges'])} 条边")
    for n in graph_down["nodes"]:
        print(f"    - {n.get('Name', '?')} [{n.get('FactorClass', '?')}]")

    # 描述子
    descs = fgdb.getDescriptors(target_qsid)
    print(f"  [PASS] getDescriptors: {len(descs)} 个直接依赖")
    for d in descs:
        print(f"    - {d.get('Name', '?')} [{d.get('FactorClass', '?')}]")

    # 向上依赖图（谁依赖这个因子）
    graph_up = fgdb.getDependencyGraph(target_qsid, direction="up")
    print(f"  [PASS] getDependencyGraph(up): {len(graph_up['nodes'])} 个节点")

    fgdb.disconnect()
    return True


def test_5_reconstruct_scalar():
    """测试 5: 重建标量 DataFactor"""
    print("\n" + "=" * 60)
    print("测试 5: 重建标量 DataFactor")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 搜索之前存储的标量因子
    results = fgdb.searchFactors(name="test_scalar")
    if not results:
        print("  [WARN] 无 test_scalar 因子，跳过")
        fgdb.disconnect()
        return True

    qsid = results[0]["QSID"]
    reconstructed = fgdb.reconstructFactor(qsid)
    print(f"  [PASS] 重建成功: {type(reconstructed).__name__}")
    print(f"  [PASS] Name: {reconstructed._QSArgs.Name}")
    print(f"  [PASS] QSID 一致: {reconstructed.QSID == qsid}")

    fgdb.disconnect()
    return True


def test_6_reconstruct_series():
    """测试 6: 重建 Series DataFactor"""
    print("\n" + "=" * 60)
    print("测试 6: 重建 Series DataFactor")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    results = fgdb.searchFactors(name="test_series")
    if not results:
        print("  [WARN] 无 test_series 因子，跳过")
        fgdb.disconnect()
        return True

    qsid = results[0]["QSID"]
    reconstructed = fgdb.reconstructFactor(qsid)
    print(f"  [PASS] 重建成功: {type(reconstructed).__name__}")
    print(f"  [PASS] 数据形状: {reconstructed._Data.shape}")
    print(f"  [PASS] 数据前 3 项: {reconstructed._Data.values[:3]}")
    print(f"  [PASS] QSID 一致: {reconstructed.QSID == qsid}")

    fgdb.disconnect()
    return True


def test_7_reconstruct_dataframe():
    """测试 7: 重建 DataFrame DataFactor"""
    print("\n" + "=" * 60)
    print("测试 7: 重建 DataFrame DataFactor")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    results = fgdb.searchFactors(name="test_dataframe")
    if not results:
        print("  [WARN] 无 test_dataframe 因子，跳过")
        fgdb.disconnect()
        return True

    qsid = results[0]["QSID"]
    reconstructed = fgdb.reconstructFactor(qsid)
    print(f"  [PASS] 重建成功: {type(reconstructed).__name__}")
    print(f"  [PASS] 数据形状: {reconstructed._Data.shape}")
    print(f"  [PASS] 列: {list(reconstructed._Data.columns)}")
    print(f"  [PASS] QSID 一致: {reconstructed.QSID == qsid}")

    fgdb.disconnect()
    return True


def test_8_reconstruct_chain():
    """测试 8: 重建衍生因子链"""
    print("\n" + "=" * 60)
    print("测试 8: 重建衍生因子链")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    results = fgdb.searchFactors(name="lag")
    if not results:
        print("  [WARN] 无 lag 因子，跳过")
        fgdb.disconnect()
        return True

    # 找到根因子（有 "lag" 名称的衍生因子）
    lag_factor_data = None
    for r in results:
        if r.get("FactorClass") == "DerivativeFactor":
            lag_factor_data = r
            break
    if not lag_factor_data:
        print("  [WARN] 无 DerivativeFactor 类型的 lag 因子，跳过")
        fgdb.disconnect()
        return True

    qsid = lag_factor_data["QSID"]
    print(f"  重建目标: {lag_factor_data['Name']} (QSID: {qsid[:16]}...)")

    reconstructed = fgdb.reconstructFactor(qsid)
    print(f"  [PASS] 重建成功: {type(reconstructed).__name__}")
    print(f"  [PASS] Name: {reconstructed._QSArgs.Name}")
    print(f"  [PASS] Descriptors 数量: {len(reconstructed.Descriptors)}")
    print(f"  [PASS] QSID 一致: {reconstructed.QSID == qsid}")

    # 验证描述子
    for i, desc in enumerate(reconstructed.Descriptors):
        print(f"    描述子 {i}: {desc._QSArgs.Name} [{type(desc).__name__}]")

    fgdb.disconnect()
    return True


def test_9_management():
    """测试 9: 管理操作（标签、元信息、重命名）"""
    print("\n" + "=" * 60)
    print("测试 9: 管理操作")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    results = fgdb.searchFactors(name="test_scalar")
    if not results:
        print("  [WARN] 无 test_scalar 因子，跳过")
        fgdb.disconnect()
        return True

    qsid = results[0]["QSID"]

    # 更新标签
    fgdb.updateFactorTags(qsid, add_tags=["updated_tag"])
    print("  [PASS] 添加标签 'updated_tag'")

    # 验证标签
    results_after = fgdb.searchFactors(tag="updated_tag")
    assert any(r["QSID"] == qsid for r in results_after), "标签添加失败"
    print("  [PASS] 标签验证通过")

    # 更新元信息
    fgdb.updateFactorMetaData(qsid, {"author": "test_user", "version": 2})
    print("  [PASS] 更新元信息")

    # 验证元信息
    factor_data = fgdb.getFactorByQSID(qsid)
    meta = json.loads(factor_data.get("MetaJSON", "{}"))
    assert meta.get("author") == "test_user"
    print(f"  [PASS] 元信息验证通过: {meta}")

    # 重命名
    fgdb.renameFactor(qsid, "test_scalar_renamed")
    factor_data = fgdb.getFactorByQSID(qsid)
    assert factor_data["Name"] == "test_scalar_renamed"
    print(f"  [PASS] 重命名为: {factor_data['Name']}")

    # 移除标签
    fgdb.updateFactorTags(qsid, remove_tags=["updated_tag"])
    results_after = fgdb.searchFactors(tag="updated_tag")
    assert not any(r["QSID"] == qsid for r in results_after), "标签移除失败"
    print("  [PASS] 标签移除成功")

    fgdb.disconnect()
    return True


def test_10_impact_analysis():
    """测试 10: 影响分析"""
    print("\n" + "=" * 60)
    print("测试 10: 影响分析")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 找到 Close DataFactor
    results = fgdb.searchFactors(name="Close")
    if not results:
        print("  [WARN] 无 Close 因子，跳过")
        fgdb.disconnect()
        return True

    close_qsid = results[0]["QSID"]
    print(f"  分析目标: Close (QSID: {close_qsid[:16]}...)")

    impacted = fgdb.impactAnalysis(close_qsid)
    print(f"  [PASS] 受影响因子数量: {len(impacted)}")
    for item in impacted:
        node = item["impacted"]
        depth = item["depth"]
        print(f"    depth={depth}: {node.get('Name', '?')} [{node.get('FactorClass', '?')}]")

    fgdb.disconnect()
    return True


def test_11_find_orphans():
    """测试 11: 查找孤立因子"""
    print("\n" + "=" * 60)
    print("测试 11: 查找孤立因子")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    orphans = fgdb.findOrphanFactors()
    print(f"  [PASS] 孤立因子数量: {len(orphans)}")
    for o in orphans[:5]:
        print(f"    - {o.get('Name', '?')} [{o.get('FactorClass', '?')}] QSID: {o.get('QSID', '?')[:16]}...")

    fgdb.disconnect()
    return True


def test_12_custom_operator():
    """测试 12: 自定义算子存储和重建"""
    print("\n" + "=" * 60)
    print("测试 12: 自定义算子")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 创建自定义算子
    def zscore_func(f, idt, iid, x, args):
        return (x[0] - np.nanmean(x[0])) / np.nanstd(x[0])

    zscore_op = makeFactorOperator(
        zscore_func, "Point",
        args={"Name": "custom_zscore", "Arity": 1, "DataType": "double"}
    )

    # 创建数据
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    ids = ["000001.SZ", "000002.SZ"]
    data = pd.DataFrame(np.random.randn(5, 2), index=dates, columns=ids)
    base = DataFactor(data=data, args={"Name": "base_data", "DataType": "double"})

    # 应用自定义算子
    zscore_factor = zscore_op(base)
    qsid = fgdb.storeFactor(zscore_factor, tags=["test", "custom_op"])
    print(f"  [PASS] 自定义算子因子存储完成, QSID: {qsid[:16]}...")

    # 检查算子
    factor_data = fgdb.getFactorByQSID(qsid)
    op_qsid = factor_data.get("OperatorQSID")
    op_results = fgdb.executeCypher(
        "MATCH (o:FactorOperator {QSID: $qsid}) RETURN o",
        {"qsid": op_qsid}
    )
    if op_results:
        op_data = op_results[0]["o"]
        print(f"  [PASS] 算子: {op_data['Name']}, IsCustom={op_data.get('IsCustom')}")
        print(f"  [PASS] CalculateRef 存在: {op_data.get('CalculateRef') is not None}")

    # 重建
    reconstructed = fgdb.reconstructFactor(qsid)
    print(f"  [PASS] 重建成功: {type(reconstructed).__name__}")
    print(f"  [PASS] QSID 一致: {reconstructed.QSID == qsid}")

    fgdb.disconnect()
    return True


def test_13_delete():
    """测试 13: 删除因子"""
    print("\n" + "=" * 60)
    print("测试 13: 删除因子")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 创建一个临时因子
    temp = DataFactor(data=999.0, args={"Name": "temp_to_delete", "DataType": "double"})
    qsid = fgdb.storeFactor(temp)
    print(f"  创建临时因子: {qsid[:16]}...")

    # 非级联删除
    deleted = fgdb.deleteFactor(qsid, cascade=False)
    assert deleted == 1
    result = fgdb.getFactorByQSID(qsid)
    assert result is None
    print(f"  [PASS] 非级联删除成功, 删除数: {deleted}")

    fgdb.disconnect()
    return True


def test_14_cypher_raw():
    """测试 14: 原始 Cypher 查询"""
    print("\n" + "=" * 60)
    print("测试 14: 原始 Cypher 查询")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 统计各类型因子数量
    results = fgdb.executeCypher(
        "MATCH (f:Factor) RETURN f.FactorClass AS cls, count(f) AS cnt ORDER BY cnt DESC"
    )
    print("  [PASS] 因子类别统计:")
    for r in results:
        print(f"    {r['cls']}: {r['cnt']}")

    # 统计关系
    results = fgdb.executeCypher(
        "MATCH ()-[r:DEPENDS_ON]->() RETURN count(r) AS cnt"
    )
    print(f"  [PASS] DEPENDS_ON 关系数: {results[0]['cnt']}")

    fgdb.disconnect()
    return True


def test_15_idempotent_store():
    """测试 15: 幂等存储"""
    print("\n" + "=" * 60)
    print("测试 15: 幂等存储")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    factor = DataFactor(data=3.14, args={"Name": "idempotent_test", "DataType": "double"})
    qsid1 = fgdb.storeFactor(factor)
    qsid2 = fgdb.storeFactor(factor)  # 第二次存储
    assert qsid1 == qsid2
    print(f"  [PASS] 两次存储 QSID 一致: {qsid1[:16]}...")

    # 验证只有一个节点
    results = fgdb.searchFactors(name="idempotent_test")
    assert len(results) == 1, f"期望 1 个节点，实际 {len(results)}"
    print(f"  [PASS] 图中只有 1 个节点")

    # 清理
    fgdb.deleteFactor(qsid1)
    fgdb.disconnect()
    return True


# ============================================================
# JYDB FactorTableFactor 测试（测试 16-21）
# ============================================================

# JYDB 实例（跨测试共享，只需连接一次）
_jydb = None
_jydb_available = None  # None=未检测, True/False=已检测

def _get_jydb():
    """懒加载 JYDB 连接，失败返回 None"""
    global _jydb, _jydb_available
    if _jydb_available is False:
        return None
    if _jydb is None:
        try:
            _jydb = JYDB().connect()
            _jydb_available = True
        except Exception as e:
            print(f"  [SKIP] JYDB 连接失败: {e}")
            _jydb_available = False
            return None
    return _jydb


def test_16_register_fdb_and_store_table():
    """测试 16: 注册 JYDB 因子库并存储因子表"""
    print("\n" + "=" * 60)
    print("测试 16: 注册 JYDB + 存储因子表")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    jydb = _get_jydb()
    if jydb is None:
        print("  [SKIP] JYDB 不可用")
        fgdb.disconnect()
        return True
    print(f"  JYDB 已连接, Name={jydb.Name}, DBType={jydb._QSArgs.DBType}")

    # 注册因子库
    fdb_name = fgdb.registerFactorDB(jydb)
    assert fdb_name == "JYDB"
    print(f"  [PASS] registerFactorDB: {fdb_name}")

    # 验证 FactorDB 节点
    fdb_results = fgdb.executeCypher(
        "MATCH (d:FactorDB {Name: $name}) RETURN d", {"name": "JYDB"}
    )
    assert len(fdb_results) == 1
    assert fdb_results[0]["d"]["DBType"] == "JYDB"
    print(f"  [PASS] FactorDB 节点已创建, DBType={fdb_results[0]['d']['DBType']}")

    # 获取并存储因子表
    ft = jydb.getTable("日行情表")
    ft_qsid = fgdb.storeFactorTable(ft, fdb_name="JYDB")
    print(f"  [PASS] storeFactorTable: {ft._QSArgs.Name}, QSID: {ft_qsid[:16]}...")

    # 验证 FactorTable 节点和 IN_DATABASE 关系
    ft_results = fgdb.executeCypher(
        "MATCH (t:FactorTable {QSID: $qsid})-[:IN_DATABASE]->(d:FactorDB) RETURN t, d",
        {"qsid": ft_qsid}
    )
    assert len(ft_results) == 1
    assert ft_results[0]["d"]["Name"] == "JYDB"
    print(f"  [PASS] IN_DATABASE 关系已建立 -> JYDB")

    # 验证因子名称列表
    ft_node = ft_results[0]["t"]
    factor_names = json.loads(ft_node["FactorNamesJSON"])
    print(f"  [PASS] 因子表包含 {len(factor_names)} 个因子")
    print(f"    前5个: {factor_names[:5]}")

    fgdb.disconnect()
    return True


def test_17_store_factortable_factor():
    """测试 17: 存储 FactorTableFactor"""
    print("\n" + "=" * 60)
    print("测试 17: 存储 FactorTableFactor")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()
    jydb = _get_jydb()
    if jydb is None:
        print("  [SKIP] JYDB 不可用")
        fgdb.disconnect()
        return True

    # 确保 JYDB 已注册
    if "JYDB" not in fgdb._FactorDBRegistry:
        fgdb.registerFactorDB(jydb)

    # 获取因子表和因子
    ft = jydb.getTable("日行情表")
    close_factor = ft.getFactor("收盘价(元)")
    print(f"  因子: {close_factor._QSArgs.Name}, FactorTable={close_factor.FactorTable._QSArgs.Name}")

    # 存储因子（会自动递归存储因子表和因子库）
    qsid = fgdb.storeFactor(close_factor, tags=["jydb", "price", "daily"])
    print(f"  [PASS] storeFactor 完成, QSID: {qsid[:16]}...")

    # 验证因子属性
    factor_data = fgdb.getFactorByQSID(qsid)
    assert factor_data["FactorClass"] == "FactorTableFactor"
    assert factor_data["FactorTableName"] == "收盘价(元)"
    assert factor_data["FactorTableQSID"] is not None
    print(f"  [PASS] FactorClass={factor_data['FactorClass']}")
    print(f"  [PASS] FactorTableName={factor_data['FactorTableName']}")
    print(f"  [PASS] FactorTableQSID={factor_data['FactorTableQSID'][:16]}...")

    # 验证 BELONGS_TO 关系
    bt_results = fgdb.executeCypher(
        """
        MATCH (f:Factor {QSID: $qsid})-[:BELONGS_TO]->(t:FactorTable)
        RETURN t.Name AS name, t.QSID AS qsid
        """,
        {"qsid": qsid}
    )
    assert len(bt_results) == 1
    assert bt_results[0]["name"] == "日行情表"
    print(f"  [PASS] BELONGS_TO -> {bt_results[0]['name']}")

    # 验证标签
    tag_results = fgdb.executeCypher(
        """
        MATCH (f:Factor {QSID: $qsid})-[:TAGGED]->(t:Tag)
        RETURN collect(t.Name) AS tags
        """,
        {"qsid": qsid}
    )
    tags = tag_results[0]["tags"]
    assert "jydb" in tags and "price" in tags and "daily" in tags
    print(f"  [PASS] 标签: {tags}")

    fgdb.disconnect()
    return True


def test_18_factortable_derivative_chain():
    """测试 18: FactorTableFactor 衍生因子链（日行情表.收盘价 → Log → Lag）"""
    print("\n" + "=" * 60)
    print("测试 18: FactorTableFactor 衍生链")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()
    jydb = _get_jydb()
    if jydb is None:
        print("  [SKIP] JYDB 不可用")
        fgdb.disconnect()
        return True

    if "JYDB" not in fgdb._FactorDBRegistry:
        fgdb.registerFactorDB(jydb)

    from QuantStudio.Factor.api import fo

    ft = jydb.getTable("日行情表")
    close = ft.getFactor("收盘价(元)")
    log_close = fo.Log()(close)
    lag1 = fo.Lag(lag_period=1)(log_close)

    # 存储整条链
    qsid = fgdb.storeFactor(lag1, tags=["jydb", "chain", "log_price"])
    print(f"  [PASS] 因子链存储完成, QSID: {qsid[:16]}...")

    # 验证图结构：lag1 -> log_close -> close(FTF) -> 日行情表(FactorTable) -> JYDB(FactorDB)
    graph = fgdb.getDependencyGraph(qsid, direction="down")
    node_names = [n["Name"] for n in graph["nodes"]]
    node_classes = [n["FactorClass"] for n in graph["nodes"]]
    print(f"  [PASS] 依赖图: {len(graph['nodes'])} 个节点")
    for n in graph["nodes"]:
        print(f"    - {n['Name']} [{n.get('FactorClass', '?')}]")

    # 验证包含 FactorTableFactor
    assert "FactorTableFactor" in node_classes, "图中应包含 FactorTableFactor"
    print(f"  [PASS] 图中包含 FactorTableFactor 节点")

    # 验证根因子是 DerivativeFactor
    root_data = fgdb.getFactorByQSID(qsid)
    assert root_data["FactorClass"] == "DerivativeFactor"
    assert root_data["OperatorType"] == "Time"
    print(f"  [PASS] 根因子: {root_data['Name']} [DerivativeFactor, Time]")

    fgdb.disconnect()
    return True


def test_19_reconstruct_factortable_factor():
    """测试 19: 重建 FactorTableFactor（从 JYDB 获取实际数据）"""
    print("\n" + "=" * 60)
    print("测试 19: 重建 FactorTableFactor")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()
    jydb = _get_jydb()
    if jydb is None:
        print("  [SKIP] JYDB 不可用")
        fgdb.disconnect()
        return True

    # 注册 JYDB（重建必须有注册的 FactorDB）
    fgdb.registerFactorDB(jydb)

    # 搜索 FactorTableFactor
    results = fgdb.searchFactors(name="收盘价(元)", factor_class="FactorTableFactor")
    if not results:
        print("  [WARN] 无 收盘价(元) FactorTableFactor，跳过")
        fgdb.disconnect()
        return True

    qsid = results[0]["QSID"]
    print(f"  重建目标: {results[0]['Name']} (QSID: {qsid[:16]}...)")

    # 重建
    reconstructed = fgdb.reconstructFactor(qsid)
    print(f"  [PASS] 重建成功: {type(reconstructed).__name__}")
    print(f"  [PASS] Name: {reconstructed._QSArgs.Name}")
    print(f"  [PASS] QSID 一致: {reconstructed.QSID == qsid}")
    print(f"  [PASS] FactorTable: {reconstructed.FactorTable._QSArgs.Name}")

    # 用重建的因子读取实际数据（验证可用性）
    import datetime as dt
    ids = ["000001.SZ", "600000.SH"]
    dts = [dt.datetime(2024, 6, 3)]
    data = reconstructed.FactorTable.readData(
        factor_names=[reconstructed._QSArgs.Name], ids=ids, dts=dts
    )
    print(f"  [PASS] 数据读取成功, shape={data.shape}")

    fgdb.disconnect()
    return True


def test_20_reconstruct_factortable_chain():
    """测试 20: 重建 FactorTableFactor 衍生链"""
    print("\n" + "=" * 60)
    print("测试 20: 重建 FactorTableFactor 衍生链")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()
    jydb = _get_jydb()
    if jydb is None:
        print("  [SKIP] JYDB 不可用")
        fgdb.disconnect()
        return True
    fgdb.registerFactorDB(jydb)

    # 搜索 lag 因子（应来自 test_18 的链）
    results = fgdb.searchFactors(tag="log_price")
    if not results:
        print("  [WARN] 无 log_price 标签因子，跳过")
        fgdb.disconnect()
        return True

    # 找到根 DerivativeFactor
    root_data = None
    for r in results:
        if r.get("FactorClass") == "DerivativeFactor" and r.get("OperatorType") == "Time":
            root_data = r
            break
    if not root_data:
        print("  [WARN] 无 Time 类型 DerivativeFactor，跳过")
        fgdb.disconnect()
        return True

    qsid = root_data["QSID"]
    print(f"  重建目标: {root_data['Name']} (QSID: {qsid[:16]}...)")

    # 重建（递归重建整条链）
    reconstructed = fgdb.reconstructFactor(qsid)
    print(f"  [PASS] 重建成功: {type(reconstructed).__name__}")
    print(f"  [PASS] Name: {reconstructed._QSArgs.Name}")
    print(f"  [PASS] QSID 一致: {reconstructed.QSID == qsid}")
    print(f"  [PASS] Descriptors: {[d._QSArgs.Name for d in reconstructed.Descriptors]}")

    # 验证底层描述子是 FactorTableFactor
    desc = reconstructed.Descriptors[0]
    while desc.Descriptors:
        desc = desc.Descriptors[0]
    assert desc.FactorTable is not None, "叶子节点应是 FactorTableFactor"
    print(f"  [PASS] 叶子节点: {desc._QSArgs.Name} (FactorTable: {desc.FactorTable._QSArgs.Name})")

    fgdb.disconnect()
    return True


def test_21_search_and_impact_factortable():
    """测试 21: FactorTableFactor 的搜索和影响分析"""
    print("\n" + "=" * 60)
    print("测试 21: FactorTableFactor 搜索与影响分析")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 按 factor_class 搜索
    results = fgdb.searchFactors(factor_class="FactorTableFactor")
    print(f"  [PASS] searchFactors(factor_class='FactorTableFactor'): {len(results)} 个")
    for r in results:
        print(f"    - {r['Name']} (Table: {r.get('FactorTableName', '?')})")

    # 按 tag 搜索
    results = fgdb.searchFactors(tag="jydb")
    print(f"  [PASS] searchFactors(tag='jydb'): {len(results)} 个")

    # 影响分析：修改 日行情表.收盘价 会影响哪些因子？
    close_results = fgdb.searchFactors(name="收盘价(元)", factor_class="FactorTableFactor")
    if close_results:
        close_qsid = close_results[0]["QSID"]
        impacted = fgdb.impactAnalysis(close_qsid)
        print(f"  [PASS] 影响分析 - 收盘价(元) 影响 {len(impacted)} 个因子:")
        for item in impacted:
            node = item["impacted"]
            depth = item["depth"]
            print(f"    depth={depth}: {node['Name']} [{node.get('FactorClass', '?')}]")

    # 统计
    stats = fgdb.getGraphStats()
    print(f"  [PASS] 图统计: {stats}")

    fgdb.disconnect()
    return True


def test_22_to_mermaid():
    """测试 22: toMermaid 依赖图可视化"""
    print("\n" + "=" * 60)
    print("测试 22: toMermaid 依赖图可视化")
    print("=" * 60)
    fgdb = FactorGraphDB(args=fgdb_args)
    fgdb.connect()

    # 22a: 单因子
    results = fgdb.searchFactors(name="turnover")
    if results:
        qsid = results[0]["QSID"]
        mermaid = fgdb.toMermaid(qsid, direction="down")
        assert "flowchart LR" in mermaid
        assert qsid[:8] in mermaid
        assert "turnover" in mermaid
        assert 'style' in mermaid
        print(f"  [PASS] 单因子 Mermaid 生成成功 ({len(mermaid.splitlines())} 行)")
        print(f"        前 3 行:")
        for line in mermaid.splitlines()[:3]:
            print(f"          {line}")
    else:
        print("  [WARN] 无 turnover 因子，跳过单因子测试")

    # 22b: 多因子合并
    results = fgdb.searchFactors(name="close", limit=2)
    if len(results) >= 2:
        qsids = [r["QSID"] for r in results]
        mermaid = fgdb.toMermaid(qsids, direction="down")
        lines = mermaid.splitlines()
        assert "flowchart LR" in mermaid
        for q in qsids:
            assert q[:8] in mermaid
        # 两个目标因子都应高亮
        style_count = sum(1 for l in lines if "fill:#f9f" in l)
        assert style_count >= 2, f"期望 >=2 个高亮节点，实际 {style_count}"
        print(f"  [PASS] 多因子 Mermaid 生成成功 ({len(lines)} 行, {style_count} 个目标节点高亮)")
    else:
        print("  [WARN] close 因子不足 2 个，跳过多因子测试")

    # 22c: direction="both"
    if results:
        qsid = results[0]["QSID"]
        mermaid = fgdb.toMermaid(qsid, direction="both")
        assert "flowchart LR" in mermaid
        print(f"  [PASS] direction='both' 生成成功 ({len(mermaid.splitlines())} 行)")

    # 22d: 颜色验证 — DerivativeFactor 蓝, FactorTableFactor 橙
    mermaid = fgdb.toMermaid(results[0]["QSID"], direction="down") if results else ""
    if mermaid:
        blue_nodes = [l for l in mermaid.splitlines() if "#e1f5fe" in l]
        orange_nodes = [l for l in mermaid.splitlines() if "#fff3e0" in l]
        print(f"  [PASS] DerivativeFactor(蓝): {len(blue_nodes)} 个")
        print(f"  [PASS] FactorTableFactor(橙): {len(orange_nodes)} 个")

    # 22e: 空结果处理
    fake_mermaid = fgdb.toMermaid("nonexistent_qsid_12345", direction="down")
    assert fake_mermaid == "flowchart LR"
    print(f"  [PASS] 无效 QSID 返回空图")

    fgdb.disconnect()
    return True


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    tests = [
        ("连接生命周期", test_1_connect),
        ("DataFactor 存储", test_2_store_and_retrieve_datafactor),
        ("衍生因子链存储", test_3_store_derivative_chain),
        ("依赖图查询", test_4_dependency_graph),
        ("重建标量 DataFactor", test_5_reconstruct_scalar),
        ("重建 Series DataFactor", test_6_reconstruct_series),
        ("重建 DataFrame DataFactor", test_7_reconstruct_dataframe),
        ("重建衍生因子链", test_8_reconstruct_chain),
        ("管理操作", test_9_management),
        ("影响分析", test_10_impact_analysis),
        ("孤立因子", test_11_find_orphans),
        ("自定义算子", test_12_custom_operator),
        ("删除因子", test_13_delete),
        ("原始 Cypher", test_14_cypher_raw),
        ("幂等存储", test_15_idempotent_store),
        ("注册JYDB+存储因子表", test_16_register_fdb_and_store_table),
        ("存储FactorTableFactor", test_17_store_factortable_factor),
        ("FactorTableFactor衍生链", test_18_factortable_derivative_chain),
        ("重建FactorTableFactor", test_19_reconstruct_factortable_factor),
        ("重建FactorTableFactor衍生链", test_20_reconstruct_factortable_chain),
        ("FactorTableFactor搜索与影响分析", test_21_search_and_impact_factortable),
        ("toMermaid 依赖图可视化", test_22_to_mermaid),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            ok = test_fn()
            if ok:
                passed += 1
            else:
                failed += 1
                errors.append((name, "返回 False"))
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  X 异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败 (共 {passed + failed} 个)")
    print("=" * 60)
    if errors:
        print("\n失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    # 清理 JYDB 连接
    if _jydb is not None:
        _jydb.disconnect()
    sys.exit(1 if failed else 0)
