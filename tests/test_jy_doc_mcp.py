# -*- coding: utf-8 -*-
"""聚源数据库文档 MCP 服务 (mcp/jy_doc/server.py) 测试。

覆盖 _load_index 索引加载、search_tables 加权评分、get_table_detail 详情格式化、
browse_categories / get_database_page 目录浏览、search_online 在线搜索，
以及 init_server 初始化（含索引缺失自动构建）与缓存重建流程；
query_qs_* 的离线用法说明生成。共 108 个测试（mock 模式 94 个）。

使用方法:
    * 运行全部 mock 测试: python tests/test_jy_doc_mcp.py
    * 通过 unittest discover: python -m unittest tests.test_jy_doc_mcp -v
    * 运行真实环境测试: python tests/test_jy_doc_mcp.py --real -v
    * 指定缓存目录: python tests/test_jy_doc_mcp.py --real --jy-doc-cache D:/MyCache -v

默认使用 mock fetcher 与内存样例索引，无需网络、凭据或真实缓存。
传入 --real 可对真实缓存索引运行 TestRealEnvironment 集成测试。
完整的运行说明（运行单个测试、常用参数、准备真实数据、测试类一览等）
见 docs/MCP/jy_doc_mcp.md。
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

# 加载模块（mcp/ 目录与 mcp SDK 包名冲突，统一用 importlib 按路径加载，
# 并把 jy_doc 包注册为 jy_doc_pkg，使 server 内的相对导入 .fetcher/.scraper 生效）
_PKG_DIR = Path(__file__).resolve().parent.parent / "mcp" / "jy_doc"


def _load(name: str, filename: str, is_pkg: bool = False):
    """按路径加载模块，屏蔽 mcp SDK 的包名冲突。"""
    path = (_PKG_DIR / filename) if not is_pkg else (_PKG_DIR / "__init__.py")
    kwargs = {"submodule_search_locations": [str(_PKG_DIR)]} if is_pkg else {}
    spec = importlib.util.spec_from_file_location(name, str(path), **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_pkg = _load("jy_doc", "__init__.py", is_pkg=True)
models_mod = _load("jy_doc.models", "models.py")
scraper_mod = _load("jy_doc.scraper", "scraper.py")
qs_help_mod = _load("jy_doc.qs_help", "qs_help.py")
_mod = _load("jy_doc.server", "server.py")

mod = _mod  # 测试中使用的短别名

# ── CLI 参数解析（仅用于直接运行脚本时）─────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--real", action="store_true", default=False,
                     help="使用真实环境运行测试")
_parser.add_argument("--jy-doc-cache", default=None, metavar="DIR",
                     help="聚源文档缓存目录（配合 --real 使用）")
_cli_args, _remaining_argv = _parser.parse_known_args()
_USE_REAL = _cli_args.real
_REAL_CACHE_DIR_OPT = _cli_args.jy_doc_cache

REAL_CACHE_DIR = os.getenv("JY_DOC_CACHE", r"D:\Data\JYDBDoc")

SAMPLE_FLAT_INDEX = [
    {
        "table_id": 258,
        "table_name": "公司主要财务分析指标",
        "base_table_name": "LC_MainFinancialData",
        "path": "/聚源数据库/国内上市公司数据库/财务数据/公司主要财务分析指标",
        "category": "国内上市公司数据库",
        "description": "公司主要财务分析指标表",
    },
    {
        "table_id": 259,
        "table_name": "公司基本资料",
        "base_table_name": "SecuMain",
        "path": "/聚源数据库/国内上市公司数据库/基本资料/公司基本资料",
        "category": "国内上市公司数据库",
        "description": "证券主表，记录证券基本资料",
    },
    {
        "table_id": 301,
        "table_name": "基金净值表现",
        "base_table_name": "MF_NetValue",
        "path": "/聚源数据库/基金数据库/净值数据/基金净值表现",
        "category": "基金数据库",
        "description": "基金单位净值与累计净值数据",
    },
    {
        "table_id": 302,
        "table_name": "基金基本资料",
        "base_table_name": "MF_FundArchives",
        "path": "/聚源数据库/基金数据库/基本资料/基金基本资料",
        "category": "基金数据库",
        "description": "公募基金基本信息",
    },
]

SAMPLE_TREE = [
    {
        "id": 10,
        "groupName": "基本资料",
        "istable": False,
        "nodes": [
            {
                "id": 259,
                "groupName": "公司基本资料",
                "istable": True,
                "description": "证券主表，记录证券基本资料",
            },
        ],
    },
    {
        "id": 11,
        "groupName": "财务数据",
        "istable": False,
        "nodes": [
            {
                "id": 258,
                "groupName": "公司主要财务分析指标",
                "istable": True,
                "description": "公司主要财务分析指标表",
            },
        ],
    },
]

SAMPLE_DATABASES = [
    {"id": 1, "name": "国内上市公司数据库", "description": ""},
    {"id": 2, "name": "基金数据库", "description": ""},
]

SAMPLE_CATEGORIES = {
    "国内上市公司数据库": {"id": 1, "tree": SAMPLE_TREE},
    "基金数据库": {
        "id": 2,
        "tree": [
            {"id": 301, "groupName": "基金净值表现", "istable": True, "description": ""},
        ],
    },
}


# ── 辅助函数 ─────────────────────────────────────────────────────────


def _load_jy_doc_credentials() -> None:
    """尝试从 mcp/.env 加载聚源平台凭据到环境变量（缺失时静默忽略）。"""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / "mcp" / ".env"
        load_dotenv(env_path)
    except Exception:
        return
    _mod.fetcher.user = _mod.fetcher.user or os.getenv("JY_DOC_USER", "")
    _mod.fetcher.pwd = _mod.fetcher.pwd or os.getenv("JY_DOC_PWD", "")


def _reset_server_globals() -> None:
    """清空 server 全局状态。"""
    _mod.flat_index = []
    _mod.databases_data = []
    _mod.categories_data = {}
    _mod.fetcher = None


def _seed_mock_globals() -> None:
    """注入 mock fetcher 与样例索引（单元测试依赖这些固定数据）。"""
    _mod.flat_index = list(SAMPLE_FLAT_INDEX)
    _mod.databases_data = [dict(db) for db in SAMPLE_DATABASES]
    _mod.categories_data = dict(SAMPLE_CATEGORIES)
    _mod.fetcher = MagicMock()


def _make_detail(**overrides):
    """构造一个 TableDetail 实例，字段可覆盖。"""
    ColumnInfo, TableDetail = models_mod.ColumnInfo, models_mod.TableDetail

    kwargs = dict(
        table_id=258,
        table_name="公司主要财务分析指标",
        base_table_name="LC_MainFinancialData",
        path="/聚源数据库/国内上市公司数据库/财务数据",
        description="公司主要财务分析指标表",
        update_frequency="每日",
        columns=[
            ColumnInfo(
                name="SecuCode",
                chinese_name="证券代码",
                data_type="varchar(20)",
                is_nullable=False,
                remark="带后缀的证券代码",
            ),
            ColumnInfo(
                name="ROE",
                chinese_name="净资产收益率",
                data_type="decimal(18,6)",
                is_nullable=True,
                remark="",
            ),
        ],
        unique_index={"columnName": "SecuCode, EndDate"},
    )
    kwargs.update(overrides)
    return TableDetail(**kwargs)


def _seed_cache(tmpdir: str, with_tables: bool = True, with_trees: bool = True) -> None:
    """在临时目录里铺设一份"已有缓存"，模拟重建前的状态。"""
    with open(os.path.join(tmpdir, "tree_index.json"), "w", encoding="utf-8") as f:
        json.dump(SAMPLE_INDEX_PAYLOAD, f, ensure_ascii=False)

    if with_tables:
        tables_dir = os.path.join(tmpdir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        for tid in (258, 259):
            with open(os.path.join(tables_dir, f"{tid}.json"), "w", encoding="utf-8") as f:
                json.dump({"table_id": tid, "table_name": "旧数据"}, f, ensure_ascii=False)

    if with_trees:
        trees_dir = os.path.join(tmpdir, "trees")
        os.makedirs(trees_dir, exist_ok=True)
        for bid in (1, 2):
            with open(os.path.join(trees_dir, f"{bid}.json"), "w", encoding="utf-8") as f:
                json.dump([], f)


SAMPLE_INDEX_PAYLOAD = {
    "flat_index": SAMPLE_FLAT_INDEX,
    "databases": SAMPLE_DATABASES,
    "categories": SAMPLE_CATEGORIES,
}


def _fake_jydb_info():
    """构造一份内存版 (TableInfo, FactorInfo)，接口与 _importInfo 一致。

    FactorInfo 的索引是 (TableName, FieldName) 两级，与真实 JYDBInfo 相同。
    """
    import pandas as pd

    table_info = pd.DataFrame(
        [
            {"DBTableName": "QT_AdjustingFactor", "TableClass": "WideTable",
             "DefaultArgs": "{'MultiMapping':False}"},
            {"DBTableName": "LC_StockArchives", "TableClass": "FeatureTable",
             "DefaultArgs": None},
            {"DBTableName": "LC_NoFactor", "TableClass": "FeatureTable",
             "DefaultArgs": None},
        ],
        index=pd.Index(["复权因子表", "公司概况", "无因子表"], name="TableName"),
    )
    rows = [
        ("复权因子表", "除权除息日", "ExDiviDate", "datetime", "Date"),
        ("复权因子表", "证券内部编码", "InnerCode", "int", "ID"),
        ("复权因子表", "精确复权因子", "AdjustingFactor", "float", "因子"),
        ("复权因子表", "比例复权因子", "RatioAdjustingFactor", "float", "因子"),
        ("公司概况", "公司中文名称", "ChiName", "varchar(60)", "因子"),
        ("公司概况", "所属状态", "SecuState", "int", "因子"),
        # FieldType 为 NaN 的字段不成为因子
        ("无因子表", "辅助字段", "AuxField", "int", None),
    ]
    factor_info = pd.DataFrame(
        [
            {"TableName": t, "FieldName": f, "DBFieldName": d,
             "DataType": dt, "FieldType": ft}
            for t, f, d, dt, ft in rows
        ]
    ).set_index(["TableName", "FieldName"])
    return table_info, factor_info


# ── 测试基类 ─────────────────────────────────────────────────────────


class _BaseTestCase(unittest.TestCase):
    """每个测试前注入 mock 环境，测试后清空全局状态。"""

    def setUp(self):
        _seed_mock_globals()

    def tearDown(self):
        _reset_server_globals()


# ── _load_index 测试 ─────────────────────────────────────────────────


class TestLoadIndex(_BaseTestCase):
    """_load_index 索引文件加载。"""

    def test_load_valid_index(self):
        """正常加载索引文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "flat_index": SAMPLE_FLAT_INDEX,
                "databases": SAMPLE_DATABASES,
                "categories": SAMPLE_CATEGORIES,
            }
            with open(os.path.join(tmpdir, "tree_index.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            _mod._load_index(tmpdir)

            self.assertEqual(len(_mod.flat_index), 4)
            self.assertEqual(len(_mod.databases_data), 2)
            self.assertIn("国内上市公司数据库", _mod.categories_data)

    def test_load_missing_index_resets_globals(self):
        """索引文件不存在时清空全局状态而不抛异常。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _mod._load_index(tmpdir)

            self.assertEqual(_mod.flat_index, [])
            self.assertEqual(_mod.databases_data, [])
            self.assertEqual(_mod.categories_data, {})

    def test_load_index_missing_keys(self):
        """索引文件缺少可选键时使用空默认值。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "tree_index.json"), "w", encoding="utf-8") as f:
                json.dump({"flat_index": SAMPLE_FLAT_INDEX}, f, ensure_ascii=False)

            _mod._load_index(tmpdir)

            self.assertEqual(len(_mod.flat_index), 4)
            self.assertEqual(_mod.databases_data, [])
            self.assertEqual(_mod.categories_data, {})

    def test_load_index_empty_file(self):
        """索引文件为空对象时全部置空。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "tree_index.json"), "w", encoding="utf-8") as f:
                json.dump({}, f)

            _mod._load_index(tmpdir)

            self.assertEqual(_mod.flat_index, [])


# ── search_tables 测试 ───────────────────────────────────────────────


class TestSearchTables(_BaseTestCase):
    """search_tables 加权评分搜索。"""

    def test_exact_base_table_name_ranks_first(self):
        """物理表名精确匹配排在最前。"""
        result = _mod.search_tables("secumain")
        self.assertIn("SecuMain", result)
        # 精确匹配 (100分) 应排在第一位
        first_entry = result.split("\n")[2]
        self.assertIn("SecuMain", first_entry)

    def test_chinese_table_name_match(self):
        """中文表名匹配。"""
        result = _mod.search_tables("基金")
        self.assertIn("基金净值表现", result)
        self.assertIn("基金基本资料", result)

    def test_partial_base_name_match(self):
        """物理表名部分匹配。"""
        result = _mod.search_tables("MF_")
        self.assertIn("MF_NetValue", result)
        self.assertIn("MF_FundArchives", result)

    def test_description_match(self):
        """描述字段匹配。"""
        result = _mod.search_tables("证券主表")
        self.assertIn("公司基本资料", result)

    def test_path_match(self):
        """路径字段匹配。"""
        result = _mod.search_tables("财务数据")
        self.assertIn("公司主要财务分析指标", result)

    def test_category_filter(self):
        """按数据库分类过滤。"""
        result = _mod.search_tables("基金", category="基金数据库")
        self.assertIn("基金净值表现", result)

        # 换个库搜索，应无结果
        result = _mod.search_tables("基金", category="国内上市公司数据库")
        self.assertIn("未找到", result)

    def test_category_all_searches_everything(self):
        """category='all' 覆盖所有库。"""
        result = _mod.search_tables("资料", category="all")
        self.assertIn("公司基本资料", result)
        self.assertIn("基金基本资料", result)

    def test_multiple_keywords(self):
        """空格分隔的多关键词取并集加权。"""
        result = _mod.search_tables("基金 净值")
        self.assertIn("基金净值表现", result)

    def test_max_results_limit(self):
        """max_results 限制返回条数。"""
        result = _mod.search_tables("基金", max_results=1)
        result_lines = [
            line for line in result.split("\n") if line.startswith(("1.", "2.", "3."))
        ]
        self.assertEqual(len(result_lines), 1)
        self.assertIn("找到 1 个相关表", result)

    def test_no_match_returns_suggestions(self):
        """无匹配时返回引导建议。"""
        result = _mod.search_tables("zzzznonexistent")
        self.assertIn("未找到", result)
        self.assertIn("browse_categories", result)
        self.assertIn("search_online", result)

    def test_result_includes_table_id_and_hint(self):
        """结果包含 table_id 和后续操作提示。"""
        result = _mod.search_tables("SecuMain")
        self.assertIn("table_id: 259", result)
        self.assertIn("get_table_detail", result)

    def test_long_description_truncated(self):
        """过长描述被截断。"""
        _mod.flat_index = [{
            "table_id": 1,
            "table_name": "长描述表",
            "base_table_name": "LONG_DESC",
            "path": "/x",
            "category": "c",
            "description": "很长的描述" * 50,
        }]
        result = _mod.search_tables("长描述表")
        self.assertIn("...", result)

    def test_case_insensitive(self):
        """搜索大小写不敏感。"""
        upper = _mod.search_tables("SECUMAIN")
        lower = _mod.search_tables("secumain")
        self.assertIn("SecuMain", upper)
        self.assertIn("SecuMain", lower)


# ── get_table_detail 测试 ────────────────────────────────────────────


class TestGetTableDetail(_BaseTestCase):
    """get_table_detail 详情获取与格式化。"""

    def test_text_format(self):
        """默认 text 格式输出。"""
        _mod.fetcher.fetch_table_detail.return_value = _make_detail()
        result = _mod.get_table_detail(258)

        self.assertIn("[表名] 公司主要财务分析指标", result)
        self.assertIn("[物理表名] LC_MainFinancialData", result)
        self.assertIn("[说明]", result)
        self.assertIn("[数据更新频率] 每日", result)
        self.assertIn("[字段列表] (2 个字段)", result)
        self.assertIn("SecuCode (证券代码) varchar(20) [非空]", result)
        self.assertIn("ROE (净资产收益率) decimal(18,6) [可空]", result)
        self.assertIn("[唯一索引] SecuCode, EndDate", result)

    def test_markdown_format(self):
        """markdown 格式输出表格。"""
        _mod.fetcher.fetch_table_detail.return_value = _make_detail()
        result = _mod.get_table_detail(258, format="markdown")

        self.assertIn("# 公司主要财务分析指标", result)
        self.assertIn("**物理表名:** `LC_MainFinancialData`", result)
        self.assertIn("| 字段名 | 中文名 | 数据类型 | 可空 | 说明 |", result)
        self.assertIn("| SecuCode | 证券代码 | varchar(20) | 否 |", result)
        self.assertIn("## 唯一索引", result)

    def test_markdown_escapes_pipe_and_newline(self):
        """markdown 表格转义竖线和换行。"""
        ColumnInfo = models_mod.ColumnInfo

        _mod.fetcher.fetch_table_detail.return_value = _make_detail(
            columns=[ColumnInfo(
                name="C1",
                chinese_name="列1",
                data_type="text",
                remark="说明|含竖线\n含换行",
            )],
        )
        result = _mod.get_table_detail(258, format="markdown")
        self.assertIn("说明\\|含竖线 含换行", result)

    def test_text_truncates_long_remark(self):
        """text 格式下过长 remark 被截断到 100 字符。"""
        ColumnInfo = models_mod.ColumnInfo

        _mod.fetcher.fetch_table_detail.return_value = _make_detail(
            columns=[ColumnInfo(name="C1", remark="长" * 200)],
        )
        result = _mod.get_table_detail(258)
        self.assertIn("...", result)
        self.assertNotIn("长" * 200, result)

    def test_path_looked_up_from_index(self):
        """调用 fetcher 时从 flat_index 传入路径。"""
        _mod.fetcher.fetch_table_detail.return_value = _make_detail()
        _mod.get_table_detail(258)

        _, kwargs = _mod.fetcher.fetch_table_detail.call_args
        self.assertEqual(kwargs["path"], SAMPLE_FLAT_INDEX[0]["path"])
        self.assertTrue(kwargs["use_cache"])

    def test_unknown_table_id_passes_empty_path(self):
        """table_id 不在索引中时路径为空。"""
        _mod.fetcher.fetch_table_detail.return_value = _make_detail(table_id=9999)
        _mod.get_table_detail(9999)

        _, kwargs = _mod.fetcher.fetch_table_detail.call_args
        self.assertEqual(kwargs["path"], "")

    def test_fetch_failure_returns_error_message(self):
        """fetch 返回 None 时给出失败说明。"""
        _mod.fetcher.fetch_table_detail.return_value = None
        result = _mod.get_table_detail(12345)

        self.assertIn("获取表详情失败 (table_id: 12345)", result)
        self.assertIn("表 ID 不存在", result)

    def test_fetch_exception_returns_error_message(self):
        """fetch 抛异常时同样返回失败提示，而非让异常穿透。"""
        _mod.fetcher.fetch_table_detail.side_effect = RuntimeError("SESSION 获取失败")

        result = _mod.get_table_detail(12345)

        self.assertIn("获取表详情失败 (table_id: 12345)", result)
        self.assertIn("表 ID 不存在", result)

    def test_fetch_retry_error_returns_error_message(self):
        """重试耗尽的 RetryError 也被捕获（登录失败的真实表现）。"""
        import tenacity

        _mod.fetcher.fetch_table_detail.side_effect = tenacity.RetryError(
            "登录重试耗尽"
        )

        result = _mod.get_table_detail(12345)

        self.assertIn("获取表详情失败 (table_id: 12345)", result)

    def test_fetch_exception_does_not_leak(self):
        """异常被吞掉且不向外传播。"""
        _mod.fetcher.fetch_table_detail.side_effect = ValueError("意外错误")

        # 不应抛出任何异常
        result = _mod.get_table_detail(258)
        self.assertIsInstance(result, str)
        self.assertIn("获取表详情失败", result)
        self.assertIn("search_tables", result)

    def test_detail_without_optional_fields(self):
        """无描述/字段/索引时仍能正常输出。"""
        _mod.fetcher.fetch_table_detail.return_value = _make_detail(
            description="", update_frequency="", columns=[], unique_index={},
        )
        result = _mod.get_table_detail(258)

        self.assertIn("[表名]", result)
        self.assertNotIn("[字段列表]", result)
        self.assertNotIn("[唯一索引]", result)
        self.assertNotIn("[数据更新频率]", result)

    def test_unique_index_without_columns(self):
        """unique_index 存在但无 columnName 时不输出索引段。"""
        _mod.fetcher.fetch_table_detail.return_value = _make_detail(
            unique_index={"indexName": "idx_1"},
        )
        result = _mod.get_table_detail(258)
        self.assertNotIn("[唯一索引]", result)


# ── browse_categories 测试 ───────────────────────────────────────────


class TestBrowseCategories(_BaseTestCase):
    """browse_categories 分类目录浏览。"""

    def test_basic_listing(self):
        """列出所有数据库及表数量。"""
        result = _mod.browse_categories()

        self.assertIn("国内上市公司数据库", result)
        self.assertIn("(2 张表)", result)
        self.assertIn("基金数据库", result)
        self.assertIn("(1 张表)", result)
        self.assertIn("共计 2 个数据库, 3 张表", result)

    def test_detail_shows_ids(self):
        """detail=True 时显示库 ID。"""
        result = _mod.browse_categories(detail=True)
        self.assertIn("[ID: 1]", result)
        self.assertIn("[ID: 2]", result)

    def test_detail_off_hides_ids(self):
        """detail=False 时不显示库 ID。"""
        result = _mod.browse_categories(detail=False)
        self.assertNotIn("[ID: 1]", result)

    def test_hints_included(self):
        """包含后续操作提示。"""
        result = _mod.browse_categories()
        self.assertIn("search_tables", result)
        self.assertIn("get_database_page", result)

    def test_empty_categories_returns_hint(self):
        """未加载分类时提示运行爬虫。"""
        _mod.databases_data = []
        result = _mod.browse_categories()
        self.assertIn("未加载分类索引", result)
        self.assertIn("scrape_jy_doc.py", result)


# ── get_database_page 测试 ───────────────────────────────────────────


class TestGetDatabasePage(_BaseTestCase):
    """get_database_page 单库表列表。"""

    def test_list_tables(self):
        """列出指定库下的表。"""
        result = _mod.get_database_page("国内上市公司数据库")

        self.assertIn("# 国内上市公司数据库  (共 2 张表)", result)
        self.assertIn("公司基本资料", result)
        self.assertIn("公司主要财务分析指标", result)
        self.assertIn("ID: 259", result)
        self.assertIn("get_table_detail", result)

    def test_unknown_database_lists_alternatives(self):
        """未知库名时列出可用库。"""
        result = _mod.get_database_page("不存在的库")

        self.assertIn("未知数据库: '不存在的库'", result)
        self.assertIn("国内上市公司数据库", result)

    def test_max_tables_truncation(self):
        """max_tables 限制显示条数。"""
        result = _mod.get_database_page("国内上市公司数据库", max_tables=1)
        self.assertIn("仅显示前 1 张表，共 2 张", result)

    def test_empty_tree(self):
        """库下无表时给出提示。"""
        _mod.categories_data["空库"] = {"id": 99, "tree": []}
        result = _mod.get_database_page("空库")
        self.assertIn("未找到任何表", result)

    def test_nested_tree_flattened(self):
        """嵌套目录树被正确展平。"""
        _mod.categories_data["深层库"] = {
            "id": 5,
            "tree": [{
                "id": 1, "groupName": "一级", "istable": False,
                "nodes": [{
                    "id": 2, "groupName": "二级", "istable": False,
                    "nodes": [{"id": 3, "groupName": "深层表", "istable": True}],
                }],
            }],
        }
        result = _mod.get_database_page("深层库")
        self.assertIn("深层表", result)
        self.assertIn("ID: 3", result)


# ── _collect_tables 测试 ─────────────────────────────────────────────


class TestCollectTables(_BaseTestCase):
    """_collect_tables 递归收集叶子节点。"""

    def test_collects_only_leaf_tables(self):
        """只收集 istable=True 的节点。"""
        tables = []
        _mod._collect_tables(SAMPLE_TREE, tables)

        self.assertEqual(len(tables), 2)
        ids = {t["id"] for t in tables}
        self.assertEqual(ids, {258, 259})
        for t in tables:
            self.assertIn("name", t)
            self.assertIn("description", t)

    def test_empty_nodes(self):
        """空列表返回空结果。"""
        tables = []
        _mod._collect_tables([], tables)
        self.assertEqual(tables, [])

    def test_node_without_istable_skipped(self):
        """缺少 istable 字段的节点被跳过。"""
        tables = []
        _mod._collect_tables([{"id": 1, "groupName": "x"}], tables)
        self.assertEqual(tables, [])


# ── build_flat_index 测试 ────────────────────────────────────────────


class TestBuildFlatIndex(_BaseTestCase):
    """scraper.build_flat_index 目录树展平。"""

    def test_reads_physical_table_name_from_leaf(self):
        """叶子节点的 tableName 被填入 base_table_name。

        目录树叶子自带 tableName（物理表名），如 QT_DailyQuote。
        """
        tree = [{
            "id": 1, "groupName": "股票日行情", "istable": True,
            "tableName": "QT_DailyQuote", "description": "日行情数据",
        }]
        entries = scraper_mod.build_flat_index(tree, category="接口数据库")

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["base_table_name"], "QT_DailyQuote")
        self.assertEqual(entries[0]["table_name"], "股票日行情")

    def test_missing_table_name_yields_empty_string(self):
        """叶子没有 tableName 字段时退化为空字符串，不抛异常。"""
        tree = [{"id": 2, "groupName": "无物理名表", "istable": True}]
        entries = scraper_mod.build_flat_index(tree, category="c")

        self.assertEqual(entries[0]["base_table_name"], "")

    def test_nested_leaves_all_get_physical_name(self):
        """嵌套目录树中每个叶子都带上自己的物理表名。"""
        tree = [{
            "id": 10, "groupName": "一级目录", "istable": False, "nodes": [
                {"id": 11, "groupName": "表A", "istable": True, "tableName": "TBL_A"},
                {"id": 12, "groupName": "二级目录", "istable": False, "nodes": [
                    {"id": 13, "groupName": "表B", "istable": True, "tableName": "TBL_B"},
                ]},
            ],
        }]
        entries = scraper_mod.build_flat_index(tree, category="c")
        got = {e["table_name"]: e["base_table_name"] for e in entries}

        self.assertEqual(got, {"表A": "TBL_A", "表B": "TBL_B"})

    def test_path_built_from_group_names(self):
        """路径由各级 groupName 拼接。"""
        tree = [{
            "id": 1, "groupName": "财务数据", "istable": False, "nodes": [
                {"id": 2, "groupName": "指标表", "istable": True, "tableName": "T"},
            ],
        }]
        entries = scraper_mod.build_flat_index(tree, category="c", parent_path="/聚源数据库")

        self.assertEqual(entries[0]["path"], "/聚源数据库/财务数据/指标表")


# ── search_online 测试 ───────────────────────────────────────────────


class TestSearchOnline(_BaseTestCase):
    """search_online 在线实时搜索。"""

    def test_finds_by_table_name(self):
        """按中文表名搜索。"""
        result = _mod.search_online("基金净值")
        self.assertIn("基金净值表现", result)
        self.assertIn("在线搜索 '基金净值' 找到", result)

    def test_finds_by_base_table_name(self):
        """按物理表名搜索。"""
        result = _mod.search_online("SecuMain")
        self.assertIn("公司基本资料", result)
        self.assertIn("物理表名: SecuMain", result)

    def test_finds_by_category(self):
        """按分类名搜索。"""
        result = _mod.search_online("基金数据库")
        self.assertIn("基金净值表现", result)
        self.assertIn("基金基本资料", result)

    def test_finds_by_description(self):
        """按描述搜索。"""
        result = _mod.search_online("证券主表")
        self.assertIn("公司基本资料", result)

    def test_case_insensitive(self):
        """大小写不敏感。"""
        result = _mod.search_online("mF_netvalue")
        self.assertIn("基金净值表现", result)

    def test_no_result(self):
        """无结果时返回提示。"""
        result = _mod.search_online("zzzznonexistent")
        self.assertIn("在线搜索未找到", result)

    def test_empty_index(self):
        """索引为空时无结果。"""
        _mod.flat_index = []
        result = _mod.search_online("基金")
        self.assertIn("在线搜索未找到", result)

    def test_result_limit_20(self):
        """结果超过 20 条时截断并提示。"""
        _mod.flat_index = [
            {
                "table_id": i,
                "table_name": f"测试表{i}",
                "base_table_name": f"TEST_{i}",
                "path": "/x",
                "category": "c",
                "description": "",
            }
            for i in range(25)
        ]
        result = _mod.search_online("测试表")
        self.assertIn("共 25 条结果，仅显示前 20 条", result)

    def test_result_lines_include_id_and_path(self):
        """结果行包含 ID 与路径。"""
        result = _mod.search_online("SecuMain")
        self.assertIn("ID: 259", result)
        self.assertIn("路径:", result)

    def test_hint_included(self):
        """包含 get_table_detail 提示。"""
        result = _mod.search_online("SecuMain")
        self.assertIn("get_table_detail", result)

    def test_long_description_truncated(self):
        """过长描述被截断到 60 字符。"""
        _mod.flat_index = [{
            "table_id": 1,
            "table_name": "长描述表",
            "base_table_name": "LONG_DESC",
            "path": "/x",
            "category": "c",
            "description": "很长的描述" * 50,
        }]
        result = _mod.search_online("长描述表")
        self.assertIn("...", result)


# ── 服务初始化测试 ───────────────────────────────────────────────────


class TestServerInit(_BaseTestCase):
    """get_default_cache_dir / init_server。"""

    def test_default_cache_dir_from_env(self):
        """优先读取 JY_DOC_CACHE 环境变量。"""
        old = os.environ.get("JY_DOC_CACHE")
        os.environ["JY_DOC_CACHE"] = "/tmp/jycache"
        try:
            self.assertEqual(_mod.get_default_cache_dir(), "/tmp/jycache")
        finally:
            if old is not None:
                os.environ["JY_DOC_CACHE"] = old
            else:
                del os.environ["JY_DOC_CACHE"]

    def test_default_cache_dir_fallback(self):
        """未设置环境变量时使用默认目录。"""
        old = os.environ.pop("JY_DOC_CACHE", None)
        try:
            self.assertEqual(_mod.get_default_cache_dir(), r"D:\Data\JYDBDoc")
        finally:
            if old is not None:
                os.environ["JY_DOC_CACHE"] = old

    def test_init_server_loads_index(self):
        """init_server 创建 fetcher 并加载索引。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "flat_index": SAMPLE_FLAT_INDEX,
                "databases": SAMPLE_DATABASES,
                "categories": SAMPLE_CATEGORIES,
            }
            with open(os.path.join(tmpdir, "tree_index.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            _mod.init_server(tmpdir, user="u", pwd="p")

            self.assertIsInstance(_mod.fetcher, _mod.JYDocFetcher)
            self.assertEqual(_mod.fetcher.user, "u")
            self.assertEqual(_mod.fetcher.pwd, "p")
            self.assertEqual(_mod.fetcher.cache_dir, tmpdir)
            self.assertEqual(len(_mod.flat_index), 4)

    def test_init_server_without_index_auto_builds(self):
        """缓存目录无索引文件时自动构建索引。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_scrape(cache_dir, user="", pwd=""):
                with open(os.path.join(cache_dir, "tree_index.json"), "w", encoding="utf-8") as f:
                    json.dump(SAMPLE_INDEX_PAYLOAD, f, ensure_ascii=False)
                return SAMPLE_INDEX_PAYLOAD

            with patch.object(_mod, "scrape_and_save", side_effect=fake_scrape) as mock_scrape:
                _mod.init_server(tmpdir, user="u", pwd="p")

            self.assertEqual(len(_mod.flat_index), 4)
            _, kwargs = mock_scrape.call_args
            self.assertEqual(kwargs["user"], "u")
            self.assertEqual(kwargs["pwd"], "p")

    def test_init_server_auto_build_failure_raises(self):
        """索引自动构建失败时抛出异常，中止启动。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_mod, "scrape_and_save") as mock_scrape:
                mock_scrape.side_effect = RuntimeError("登录凭据未配置")
                with self.assertRaisesRegex(RuntimeError, "索引自动构建失败"):
                    _mod.init_server(tmpdir)

    def test_init_server_with_existing_index_skips_build(self):
        """已存在索引时不触发自动构建。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "tree_index.json"), "w", encoding="utf-8") as f:
                json.dump(SAMPLE_INDEX_PAYLOAD, f, ensure_ascii=False)

            with patch.object(_mod, "scrape_and_save") as mock_scrape:
                _mod.init_server(tmpdir, user="u", pwd="p")

            mock_scrape.assert_not_called()
            self.assertEqual(len(_mod.flat_index), 4)


# ── 缓存重建测试 ─────────────────────────────────────────────────────


class TestRebuildCache(_BaseTestCase):
    """rebuild_cache / init_server(rebuild=True)。"""

    def test_clears_tables_and_trees_subdirs(self):
        """重建时清除 tables/ 与 trees/ 下的旧缓存文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_cache(tmpdir)

            with patch.object(_mod, "scrape_and_save") as mock_scrape:
                mock_scrape.return_value = {"categories": {}, "flat_index": []}
                _mod.rebuild_cache(tmpdir)

            self.assertFalse(os.path.exists(os.path.join(tmpdir, "tables")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "trees")))

    def test_deletes_stale_index_before_scraping(self):
        """抓取前先删除旧索引文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_cache(tmpdir)
            index_path = os.path.join(tmpdir, "tree_index.json")
            self.assertTrue(os.path.exists(index_path))

            seen = {}

            def fake_scrape(cache_dir, user="", pwd=""):
                # 抓取被调用时，旧索引应已被删除
                seen["index_exists"] = os.path.exists(index_path)
                return {"categories": {}, "flat_index": []}

            with patch.object(_mod, "scrape_and_save", fake_scrape):
                _mod.rebuild_cache(tmpdir)

            self.assertFalse(seen["index_exists"])

    def test_passes_credentials_to_scraper(self):
        """登录凭据透传给爬虫。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_mod, "scrape_and_save") as mock_scrape:
                mock_scrape.return_value = {"categories": {}, "flat_index": []}
                _mod.rebuild_cache(tmpdir, user="alice", pwd="secret")

            _, kwargs = mock_scrape.call_args
            self.assertEqual(kwargs["cache_dir"], tmpdir)
            self.assertEqual(kwargs["user"], "alice")
            self.assertEqual(kwargs["pwd"], "secret")

    def test_rebuild_on_empty_dir(self):
        """缓存目录为空时重建不报错。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_mod, "scrape_and_save") as mock_scrape:
                mock_scrape.return_value = {"categories": {}, "flat_index": []}
                _mod.rebuild_cache(tmpdir)

            mock_scrape.assert_called_once()

    def test_scrape_failure_propagates(self):
        """抓取失败时异常向上抛出，由调用方决定如何降级。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_mod, "scrape_and_save") as mock_scrape:
                mock_scrape.side_effect = RuntimeError("登录凭据未配置")
                with self.assertRaisesRegex(RuntimeError, "登录凭据未配置"):
                    _mod.rebuild_cache(tmpdir)

    def test_init_server_with_rebuild_triggers_rebuild(self):
        """init_server(rebuild=True) 会触发重建。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_cache(tmpdir)

            def fake_scrape(cache_dir, user="", pwd=""):
                # 模拟爬虫写出新索引
                with open(os.path.join(cache_dir, "tree_index.json"), "w", encoding="utf-8") as f:
                    json.dump(SAMPLE_INDEX_PAYLOAD, f, ensure_ascii=False)
                return SAMPLE_INDEX_PAYLOAD

            with patch.object(_mod, "scrape_and_save", fake_scrape):
                _mod.init_server(tmpdir, rebuild=True)

            # 重建后索引被重新加载
            self.assertEqual(len(_mod.flat_index), 4)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "tables")))

    def test_init_server_without_rebuild_keeps_cache(self):
        """不传 rebuild 时不触碰缓存目录内容。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_cache(tmpdir)

            with patch.object(_mod, "scrape_and_save") as mock_scrape:
                _mod.init_server(tmpdir)

            mock_scrape.assert_not_called()
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tables", "258.json")))

    def test_init_server_rebuild_reflects_new_index(self):
        """重建后内存索引反映新抓取的内容，而非旧缓存。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _seed_cache(tmpdir)

            new_entry = {
                "table_id": 999,
                "table_name": "新增表",
                "base_table_name": "NEW_TABLE",
                "path": "/x",
                "category": "新库",
                "description": "",
            }

            def fake_scrape(cache_dir, user="", pwd=""):
                payload = {"flat_index": [new_entry], "databases": [], "categories": {}}
                with open(os.path.join(cache_dir, "tree_index.json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                return payload

            with patch.object(_mod, "scrape_and_save", fake_scrape):
                _mod.init_server(tmpdir, rebuild=True)

            self.assertEqual(len(_mod.flat_index), 1)
            self.assertEqual(_mod.flat_index[0]["table_name"], "新增表")


# ── QuantStudio 使用说明测试 ─────────────────────────────────────────


class TestQSHelp(_BaseTestCase):
    """qs_help 离线生成 QuantStudio 使用说明。"""

    def setUp(self):
        super().setUp()
        self._info = _fake_jydb_info()
        self._original_loader = qs_help_mod._load_jydb_info
        qs_help_mod._load_jydb_info = lambda: self._info

    def tearDown(self):
        qs_help_mod._load_jydb_info = self._original_loader
        qs_help_mod._load_jydb_info.cache_clear() if hasattr(qs_help_mod._load_jydb_info, "cache_clear") else None
        super().tearDown()

    def test_get_factor_help_contains_table_and_factor(self):
        """get_factor_help 给出 QuantStudio 内部表名与因子名。"""
        result = qs_help_mod.get_factor_help("QT_AdjustingFactor")

        self.assertIn('getTable(table_name="复权因子表"', result)
        self.assertIn("getFactor(", result)
        self.assertIn("比例复权因子", result)

    def test_read_data_help_contains_readdata_example(self):
        """read_data_help 给出 readData 示例与因子名列表。"""
        result = qs_help_mod.read_data_help("QT_AdjustingFactor")

        self.assertIn('getTable(table_name="复权因子表"', result)
        self.assertIn("readData(factor_names=", result)
        self.assertIn("Panel", result)

    def test_arg_info_follows_table_class(self):
        """args 说明取表类的参数模型：仅 repr=True 的参数，含中文名与默认值。"""
        wide = qs_help_mod.get_factor_help("QT_AdjustingFactor")
        self.assertIn("MultiMapping(多重映射): 默认 False", wide)
        self.assertIn("LookBack(回溯天数): 默认 0", wide)
        self.assertIn("{'MultiMapping':False}", wide)

        feature = qs_help_mod.get_factor_help("LC_StockArchives")
        self.assertIn("LookBack(回溯天数): 默认 inf", feature)  # FeatureTable 回溯默认无限

    def test_arg_info_only_shows_repr_fields(self):
        """repr=False 的内部参数不出现在 args 说明中。"""
        wide = qs_help_mod.get_factor_help("QT_AdjustingFactor")
        for internal in ("TaskExecutor", "IgnoreIndex", "ForceIndex", "TransformSQL"):
            self.assertNotIn(internal, wide)

        feature = qs_help_mod.get_factor_help("LC_StockArchives")
        self.assertNotIn("TargetDT", feature)  # repr=False

    def test_arg_info_falls_back_when_model_missing(self):
        """参数模型取不到时退回静态参数名清单，不抛异常。"""
        original = qs_help_mod._table_arg_fields
        qs_help_mod._table_arg_fields = lambda table_class: None
        try:
            result = qs_help_mod.get_factor_help("QT_AdjustingFactor")
        finally:
            qs_help_mod._table_arg_fields = original

        self.assertIn("LookBack", result)
        self.assertIn("{'MultiMapping':False}", result)

    def test_non_factor_fields_excluded(self):
        """FieldType 为空（不成为因子）的字段不出现在结果中。"""
        result = qs_help_mod.get_factor_help("LC_NoFactor")
        self.assertNotIn("辅助字段", result)
        self.assertIn("未配置任何因子字段", result)

    def test_unsupported_table_returns_hint(self):
        """未在 JYDBInfo 注册的表返回不支持提示。"""
        result = qs_help_mod.get_factor_help("NOT_REGISTERED")
        self.assertIn("QuantStudio 不支持使用该表 NOT_REGISTERED", result)

    def test_empty_base_table_name_returns_hint(self):
        """物理表名为空时返回不支持提示而非抛异常。"""
        self.assertIn("不支持", qs_help_mod.read_data_help(""))

    def test_factor_name_map_is_physical_to_factor(self):
        """factor_name_map 以物理字段名为键、QuantStudio 因子名为值。"""
        mapping = qs_help_mod.factor_name_map("QT_AdjustingFactor")

        self.assertEqual(mapping["RatioAdjustingFactor"], "比例复权因子")
        self.assertEqual(mapping["AdjustingFactor"], "精确复权因子")
        # 非因子字段（FieldType=ID/Date）不在映射里
        self.assertNotIn("InnerCode", mapping)
        self.assertNotIn("ExDiviDate", mapping)

    def test_table_arg_fields_unknown_class_returns_none(self):
        """未知 TableClass 取不到参数模型，返回 None 而非抛异常。"""
        self.assertIsNone(qs_help_mod._table_arg_fields("NotATableClass"))
        self.assertIsNone(qs_help_mod._table_arg_fields(""))

    def test_format_arg_default_renders_common_types(self):
        """默认值按类型渲染：字符串加引号，容器用 repr，None 显式写出。"""
        from QuantStudio.Factor.FactorUtils import SQL_WideTable

        fields = SQL_WideTable.__QS_ArgClass__.model_fields
        self.assertEqual(qs_help_mod._format_arg_default(fields["LookBack"]), "0")
        self.assertEqual(qs_help_mod._format_arg_default(fields["MultiMapping"]), "False")
        self.assertEqual(qs_help_mod._format_arg_default(fields["DTField"]), "None")
        self.assertEqual(qs_help_mod._format_arg_default(fields["OrderFields"]), "[]")
        self.assertEqual(qs_help_mod._format_arg_default(fields["TransformSQL"]), "{}")
        self.assertEqual(qs_help_mod._format_arg_default(fields["TableType"]), "'WideTable'")

    def test_jydb_info_load_failure_degrades(self):
        """JYDBInfo 加载失败时优雅降级为不支持提示。"""
        qs_help_mod._load_jydb_info.cache_clear() if hasattr(qs_help_mod._load_jydb_info, "cache_clear") else None
        qs_help_mod._load_jydb_info = lambda: None

        result = qs_help_mod.get_factor_help("ANY_TABLE")
        self.assertIn("QuantStudio 不支持使用该表 ANY_TABLE", result)


class TestQSHelpTools(_BaseTestCase):
    """server 中两个 MCP 工具的 table_id 解析与委派。"""

    def setUp(self):
        super().setUp()
        self._info = _fake_jydb_info()
        self._original_loader = qs_help_mod._load_jydb_info
        qs_help_mod._load_jydb_info = lambda: self._info
        # 把样例索引的 258 号表物理表名指向内存 JYDBInfo 里注册过的表
        _mod.flat_index = [dict(e) for e in SAMPLE_FLAT_INDEX]
        _mod.flat_index[0]["base_table_name"] = "QT_AdjustingFactor"

    def tearDown(self):
        qs_help_mod._load_jydb_info = self._original_loader
        super().tearDown()

    def test_read_data_help_resolves_table_id(self):
        """table_id 经索引解析为物理表名后委派给 qs_help。"""
        result = _mod.query_qs_read_data_help(258)
        self.assertIn('getTable(table_name="复权因子表"', result)

    def test_get_factor_help_resolves_table_id(self):
        """同上，get_factor_help 分支。"""
        result = _mod.query_qs_get_factor_help(258)
        self.assertIn('getTable(table_name="复权因子表"', result)

    def test_unsupported_table_id_degrades_to_hint(self):
        """物理表名解析成功但 JYDBInfo 未注册时，返回不支持提示。"""
        result = _mod.query_qs_get_factor_help(259)
        self.assertIn("QuantStudio 不支持使用该表 SecuMain", result)

    def test_unknown_table_id_returns_hint(self):
        """索引中不存在的 table_id 给出确认提示，不抛异常。"""
        result = _mod.query_qs_get_factor_help(999999)
        self.assertIn("未在索引中找到 table_id=999999", result)
        self.assertIn("search_tables", result)

    def test_table_without_physical_name_returns_hint(self):
        """索引条目物理表名为空时给出确认提示。"""
        _mod.flat_index = [{
            "table_id": 1, "table_name": "X", "base_table_name": "",
            "path": "/x", "category": "c", "description": "",
        }]
        result = _mod.query_qs_read_data_help(1)
        self.assertIn("未在索引中找到 table_id=1", result)


# ── MCP 工具注册测试 ─────────────────────────────────────────────────


class TestToolRegistration(_BaseTestCase):
    """MCP 工具注册完整性。"""

    def test_all_tools_registered(self):
        """7 个工具均已注册到 FastMCP 实例。"""
        tools = asyncio.run(_mod.mcp.list_tools())
        names = {t.name for t in tools}
        self.assertEqual(names, {
            "search_tables",
            "get_table_detail",
            "browse_categories",
            "get_database_page",
            "search_online",
            "query_qs_read_data_help",
            "query_qs_get_factor_help",
        })

    def test_tools_have_descriptions(self):
        """每个工具都有描述供 Agent 理解用途。"""
        tools = asyncio.run(_mod.mcp.list_tools())
        for tool in tools:
            self.assertTrue(tool.description, f"工具 {tool.name} 缺少描述")

    def test_server_name(self):
        """MCP 服务名为 jy_doc。"""
        self.assertEqual(_mod.mcp.name, "jy_doc")


class TestNativePreload(_BaseTestCase):
    """主线程原生依赖预热（Windows 加载器锁死锁规避）。"""

    def setUp(self):
        super().setUp()
        self._info = _fake_jydb_info()
        self._original_loader = qs_help_mod._load_jydb_info
        qs_help_mod._load_jydb_info = lambda: self._info

    def tearDown(self):
        qs_help_mod._load_jydb_info = self._original_loader
        super().tearDown()

    def test_qs_help_preload_callable_and_idempotent(self):
        """preload() 返回布尔值且可重复调用。"""
        first = qs_help_mod.preload()
        self.assertTrue(first)
        # lru_cache 命中后再次调用仍成功
        self.assertTrue(qs_help_mod.preload())

    def test_preload_degrades_when_jydbinfo_unavailable(self):
        """JYDBInfo 不可用时 preload 返回 False 而不抛异常。"""
        qs_help_mod._load_jydb_info = lambda: None
        self.assertFalse(qs_help_mod.preload())

    def test_preimport_native_deps_is_callable(self):
        """server._preimport_native_deps 不抛异常（已装 ddddocr 时）。"""
        _mod._preimport_native_deps()  # 不应抛异常


# ── 真实环境测试 ─────────────────────────────────────────────────────
#
# 默认跳过。传入 --real 后运行，使用真实的本地缓存索引。
#
#   python tests/test_jy_doc_mcp.py --real -v
#   python tests/test_jy_doc_mcp.py --real --jy-doc-cache D:/MyCache -v
#
# 缓存目录优先取 --jy-doc-cache，其次 JY_DOC_CACHE 环境变量，最后 D:\Data\JYDBDoc。
# 其中 test_real_fetch_table_detail 还需要网络+平台凭据(JY_DOC_USER/JY_DOC_PWD)，
# 缺少凭据时自动跳过。缓存未命中时该测试会真正访问 dd.gildata.com。


@unittest.skipUnless(_USE_REAL, "需传入 --real 以运行真实环境测试")
class TestRealEnvironment(unittest.TestCase):
    """真实环境测试：基于真实树索引验证工具行为。"""

    @classmethod
    def setUpClass(cls):
        cls.cache_dir = _REAL_CACHE_DIR_OPT or REAL_CACHE_DIR
        _mod.fetcher = _mod.JYDocFetcher(cache_dir=cls.cache_dir)
        _mod._load_index(cls.cache_dir)

    @classmethod
    def tearDownClass(cls):
        _reset_server_globals()

    def _skip_if_no_cache(self):
        index_path = os.path.join(self.cache_dir, "tree_index.json")
        if not os.path.exists(index_path):
            self.skipTest(f"真实索引文件不存在: {index_path}")

    def test_index_loaded(self):
        """真实索引成功加载。"""
        self._skip_if_no_cache()
        self.assertGreater(len(_mod.flat_index), 0)
        self.assertGreater(len(_mod.databases_data), 0)
        self.assertGreater(len(_mod.categories_data), 0)

    def test_index_entries_have_required_fields(self):
        """索引条目具备搜索所需的全部字段。"""
        self._skip_if_no_cache()
        required = {"table_id", "table_name", "base_table_name", "path", "category", "description"}
        for entry in _mod.flat_index:
            self.assertTrue(required <= set(entry), f"条目缺字段: {entry}")

    def test_index_base_table_name_is_filled(self):
        """索引条目的物理表名非空。

        回归：build_flat_index 曾把 base_table_name 写成空字符串占位
        （注释称"需要通过 fetch_table_detail 获取"），导致 search_tables 里
        物理表名匹配的 +60 分项永不触发、按物理表名搜索彻底失效。
        目录树叶子节点实际自带 tableName，应当直接取用。
        """
        self._skip_if_no_cache()
        empty = [e for e in _mod.flat_index if not e.get("base_table_name")]
        self.assertEqual(empty, [], f"{len(empty)} 个条目物理表名为空，示例: {empty[:3]}")

    def test_search_by_base_table_name_finds_exact(self):
        """按物理表名搜索时，精确匹配排在首位。"""
        self._skip_if_no_cache()
        entry = next(e for e in _mod.flat_index if e.get("base_table_name"))
        result = _mod.search_tables(entry["base_table_name"], max_results=3)
        first = next(ln for ln in result.splitlines() if ln.strip().startswith("1."))
        self.assertIn(entry["base_table_name"], first, f"精确匹配未置顶: {first}")

    def test_browse_categories_real(self):
        """真实分类目录可浏览且统计出表数量。"""
        self._skip_if_no_cache()
        result = _mod.browse_categories()
        self.assertNotIn("未加载分类索引", result)
        self.assertIn(f"共计 {len(_mod.databases_data)} 个数据库", result)

    def test_get_database_page_real(self):
        """真实库下的表列表可获取。"""
        self._skip_if_no_cache()
        db_name = _mod.databases_data[0]["name"]
        result = _mod.get_database_page(db_name, max_tables=5)
        self.assertTrue(result.startswith(f"# {db_name}"))
        self.assertIn("get_table_detail", result)

    def test_search_tables_real_returns_hits(self):
        """用真实表名搜索能命中（本地索引打分路径）。"""
        self._skip_if_no_cache()
        entry = _mod.flat_index[0]
        result = _mod.search_tables(entry["table_name"], max_results=5)
        self.assertIn("找到", result)
        self.assertNotIn("未找到", result)

    def test_search_online_real(self):
        """真实索引上的在线搜索能命中。"""
        self._skip_if_no_cache()
        entry = _mod.flat_index[0]
        result = _mod.search_online(entry["table_name"])
        self.assertNotIn("在线搜索未找到", result)

    def test_real_fetch_table_detail(self):
        """缓存未命中时能走网络从平台拉取真实表详情并格式化。

        需要网络与登录凭据（从 mcp/.env 的 JY_DOC_USER/JY_DOC_PWD 加载），
        缺失时跳过。为真实验证网络路径而非缓存路径，测试前会主动清除该表的
        本地缓存；若该表 ID 不在真实索引中则跳过。
        """
        self._skip_if_no_cache()
        table_id = 7063

        entry = next(
            (e for e in _mod.flat_index if e.get("table_id") == table_id), None
        )
        if entry is None:
            self.skipTest(f"table_id={table_id} 不在真实索引中，无法验证路径回填")

        if not (_mod.fetcher.user and _mod.fetcher.pwd):
            _load_jy_doc_credentials()
        if not (_mod.fetcher.user and _mod.fetcher.pwd):
            self.skipTest("未配置 JY_DOC_USER/JY_DOC_PWD")

        # 清除缓存，确保走真实网络请求而非缓存命中
        cache_path = _mod.fetcher._table_cache_path(table_id)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        self.assertIsNone(_mod.fetcher._load_table_cache(table_id))

        result = _mod.get_table_detail(table_id)

        self.assertNotIn(f"获取表详情失败 (table_id: {table_id})", result)
        self.assertIn("[表名]", result)
        self.assertIn("[物理表名]", result)
        # 路径由 get_table_detail 从 flat_index 回填，验证 int 型 ID 能正确匹配
        self.assertIn(entry["path"], result)
        # 网络拉取应已写回缓存
        self.assertIsNotNone(_mod.fetcher._load_table_cache(table_id))

    def test_login_twice_on_same_fetcher(self):
        """同一实例连续登录两次都应成功（回归：残留 SESSION 曾导致第二次必失败）。

        历史缺陷：fetcher 复用 http_session，首次登录后 cookie jar 中残留
        SESSION，而 /api/captcha 仅在请求未携带 SESSION 时才下发新的 SESSION，
        致使第二次登录拿不到 Set-Cookie 而抛 "SESSION 获取失败"。
        """
        self._skip_if_no_cache()
        if not (_mod.fetcher.user and _mod.fetcher.pwd):
            _load_jy_doc_credentials()
        if not (_mod.fetcher.user and _mod.fetcher.pwd):
            self.skipTest("未配置 JY_DOC_USER/JY_DOC_PWD")

        for _ in range(2):
            _mod.fetcher.session_id = None  # 模拟 session 过期后重新登录
            self.assertTrue(_mod.fetcher._login())

    def test_concurrent_fetch_table_detail(self):
        """同一实例被多线程并发抓取时全部成功。

        本对象在实际使用中是跨线程共享的单例（MCP 服务的 anyio 工作线程会并发
        调用）。并发登录会互相覆盖共享 http_session 的 SESSION cookie，并让
        self.session_id 在多线程间来回改写，触发多余的重复登录与 401 重试；
        fetcher 内部用锁串行化后应能全部成功。此用例为并发可用性验证——竞态
        窗口很窄，不一定每次都能在无锁实现上复现失败。
        """
        self._skip_if_no_cache()
        if not (_mod.fetcher.user and _mod.fetcher.pwd):
            _load_jy_doc_credentials()
        if not (_mod.fetcher.user and _mod.fetcher.pwd):
            self.skipTest("未配置 JY_DOC_USER/JY_DOC_PWD")

        # 取若干未缓存的表，确保走真实网络请求
        picks = []
        for entry in _mod.flat_index:
            table_id = entry.get("table_id")
            if not table_id:
                continue
            if os.path.exists(_mod.fetcher._table_cache_path(table_id)):
                continue
            picks.append((table_id, entry.get("path", "")))
            if len(picks) >= 3:
                break
        if len(picks) < 3:
            self.skipTest("真实索引中未找到足够的未缓存表")

        barrier = threading.Barrier(len(picks))

        def fetch(item):
            barrier.wait()  # 所有线程同一时刻发起，最大化并发登录的竞争
            table_id, path = item
            return _mod.fetcher.fetch_table_detail(table_id, use_cache=False, path=path)

        with ThreadPoolExecutor(max_workers=len(picks)) as pool:
            results = list(pool.map(fetch, picks))

        for r in results:
            self.assertIsNotNone(r)
            self.assertTrue(r.columns)

    # ── qs_help 工具（真实 JYDBInfo.xlsx，离线）──

    def test_qs_help_uses_real_jydbinfo(self):
        """真实 JYDBInfo 下，能连上的表生成完整示例。"""
        self._skip_if_no_cache()
        entry = next(
            (e for e in _mod.flat_index if e.get("base_table_name") == "QT_AdjustingFactor"),
            None,
        )
        if entry is None:
            self.skipTest("真实索引中没有 QT_AdjustingFactor")

        result = _mod.query_qs_get_factor_help(entry["table_id"])

        self.assertNotIn("QuantStudio 不支持", result)
        self.assertIn('getTable(table_name="复权因子表"', result)
        self.assertIn("getFactor(", result)

    def test_qs_help_supported_ratio(self):
        """索引中相当一部分表能被 JYDBInfo 支持（回归：连接率不应为 0）。

        覆盖率由 QuantStudio 的 JYDBInfo 配置决定（约 495 张表），
        而非 jy_doc 索引规模，故只断言一个宽松下界。
        """
        self._skip_if_no_cache()
        supported = 0
        for e in _mod.flat_index:
            if qs_help_mod.factor_name_map(e.get("base_table_name", "")):
                supported += 1
        self.assertGreater(supported, 200, f"仅 {supported} 张表能被 JYDBInfo 支持，连接可能失效")

    def test_qs_help_unsupported_for_unconfigured_table(self):
        """真实索引里未被 JYDBInfo 注册的表返回不支持提示。"""
        self._skip_if_no_cache()
        entry = next(
            (e for e in _mod.flat_index
             if e.get("base_table_name") and not qs_help_mod.factor_name_map(e["base_table_name"])),
            None,
        )
        if entry is None:
            self.skipTest("所有表都被 JYDBInfo 支持，无法验证不支持分支")

        result = _mod.query_qs_read_data_help(entry["table_id"])
        self.assertIn("QuantStudio 不支持使用该表", result)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + _remaining_argv)
