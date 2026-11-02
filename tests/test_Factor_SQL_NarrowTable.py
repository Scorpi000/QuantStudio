"""SQL 窄表因子名元信息的隔离回归测试，不连接外部数据库。"""

import unittest
from unittest.mock import patch

import pandas as pd

from QuantStudio.Factor.FactorUtils import SQL_NarrowTable
from QuantStudio.Factor.SQLDB import SQLDB


class SQLNarrowTableFactorNamesTests(unittest.TestCase):
    """验证元信息优先及缺失时的查询缓存和名称映射。"""

    def setUp(self):
        self.db = SQLDB(args={"DBType": "PostgreSQL"}, config_file={})
        self.table = SQL_NarrowTable(
            fdb=self.db,
            args={"FactorNameField": "factor_name", "FactorValueField": "factor_value"},
            table_info=pd.Series({"DBTableName": "qs_factors"}),
            factor_info=pd.DataFrame({
                "DBFieldName": ["datetime", "code", "factor_name", "factor_value"],
                "DataType": ["TIMESTAMP", "VARCHAR", "VARCHAR", "DOUBLE"],
                "FieldType": ["Date", "ID", "因子", "因子"],
                "Supplementary": ["Default", None, None, None],
            }, index=["datetime", "code", "factor_name", "factor_value"]),
        )

    def test_metadata_returns_list_without_query_including_empty_list(self):
        """列表保持元信息中的顺序，空列表也不触发数据库查询。"""
        for names in (["pb", "pe"], []):
            with self.subTest(names=names), patch.object(self.db, "fetchall") as query:
                self.table._TableInfo["FactorNames"] = names
                self.assertEqual(self.table.FactorNames, names)
                query.assert_not_called()

    def test_missing_or_null_metadata_falls_back_to_cached_distinct_query(self):
        """未配置及其他表引入的空值不影响原来的查询和缓存。"""
        for metadata in ({}, {"FactorNames": None}, {"FactorNames": float("nan")}):
            with self.subTest(metadata=metadata):
                self.table._TableInfo = pd.Series({"DBTableName": "qs_factors"} | metadata)
                self.table._QSArgs._FactorNames = None
                with patch.object(self.db, "fetchall", return_value=[("pb",), ("pe",)]) as query:
                    self.assertEqual(self.table.FactorNames, ["pb", "pe"])
                    self.assertEqual(self.table.FactorNames, ["pb", "pe"])
                    query.assert_called_once()
                    self.assertIn("SELECT DISTINCT", query.call_args.args[0])

    def test_metadata_overrides_previously_queried_names(self):
        """后来设置的表元信息优先于旧的查询缓存。"""
        with patch.object(self.db, "fetchall", return_value=[("old",)]) as query:
            self.assertEqual(self.table.FactorNames, ["old"])
            self.table._TableInfo["FactorNames"] = ["new"]
            self.assertEqual(self.table.FactorNames, ["new"])
            query.assert_called_once()

    def test_related_sql_mapping_is_preserved(self):
        """缺少元信息时保持关联 SQL 的因子名称到编码映射。"""
        self.table._FactorInfo["RelatedSQL"] = None
        self.table._TablePrefix = ""
        self.table._FactorInfo.loc["factor_name", "RelatedSQL"] = "SELECT code, name FROM factor_dict"
        with patch.object(self.db, "fetchall", return_value=[("2", "pe"), ("1", "pb")]) as query:
            self.assertEqual(self.table.FactorNames, ["pb", "pe"])
            self.assertEqual(self.table._getFactorNames("factor_name", check_list=True),
                             {"pe": "2", "pb": "1"})
            query.assert_called_once()


if __name__ == "__main__":
    unittest.main()
