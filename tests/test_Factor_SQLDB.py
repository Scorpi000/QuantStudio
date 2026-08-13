# -*- coding: utf-8 -*-
"""QuantStudio SQLDB 关系数据库因子库测试模块

测试覆盖:
    * 基本功能: 连接、表名列表、获取因子表、时点序列
    * 元数据: 侧表元数据读写、级联操作、列注释

使用方法:
    * 运行全部测试: python tests/test_Factor_SQLDB.py
    * 运行指定测试: python -m unittest tests.test_Factor_SQLDB.TestSQLDB.test_connect
    * 通过 TestSuite 指定测试 (取消文件末尾的注释并修改):

        Suite = unittest.TestSuite()
        Suite.addTest(TestSQLDB("test_connect"))
        Runner = unittest.TextTestRunner()
        Runner.run(Suite)

    注意: SQLDB 需要连接到 PostgreSQL 数据库, 测试依赖数据库可用。
"""
import unittest
import pandas as pd

from QuantStudio.Factor.SQLDB import SQLDB


# 元数据测试用临时表名
_TEST_TABLE = "_qstest_meta_demo"
_TEST_TABLE_2 = "_qstest_meta_demo_renamed"


class TestSQLDB(unittest.TestCase):
    """SQLDB 关系数据库因子库基本功能测试"""

    @classmethod
    def setUpClass(cls):
        """创建 SQLDB 实例并连接数据库"""
        cls.FDB = SQLDB(
            args={"DBType": "PostgreSQL", "InnerPrefix": "qsd_"}
        ).connect()

    # ==================== 连接测试 ====================

    def test_connect(self):
        """测试 SQLDB 构造和连接"""
        self.assertIsNotNone(self.FDB)
        self.assertEqual(self.FDB.Name, "SQLDB")
        # 连接后可再次调用 connect (幂等)
        self.FDB.connect()

    # ==================== 因子表测试 ====================

    def test_TableNames(self):
        """测试 TableNames 返回表名列表"""
        TableNames = self.FDB.TableNames
        self.assertIsInstance(TableNames, list)
        self.assertGreater(len(TableNames), 0)

    def test_getTable(self):
        """测试 getTable 获取因子表对象, 验证 FactorNames 和 getFactor"""
        TableName = self.FDB.TableNames[0]
        FT = self.FDB.getTable(TableName, args={"MultiMapping": False})
        self.assertIsNotNone(FT)
        self.assertEqual(FT.Name, TableName)
        FactorNames = FT.FactorNames
        self.assertGreater(len(FactorNames), 0)
        # 通过 getFactor 获取单个因子
        FirstFactor = FactorNames[0]
        F = FT.getFactor(FirstFactor)
        self.assertEqual(F.Name, FirstFactor)
        # 通过 __getitem__ 语法糖获取
        self.assertEqual(FT[FirstFactor].Name, FirstFactor)

    # ==================== ID 和时点测试 ====================

    def test_getDateTime(self):
        """测试获取表中时点序列"""
        TableName = self.FDB.TableNames[0]
        FT = self.FDB.getTable(TableName, args={"MultiMapping": False})
        DTs = FT.getDateTime()
        self.assertIsInstance(DTs, list)
        self.assertGreater(len(DTs), 0)
        # 时点应递增
        self.assertTrue(all(DTs[i] < DTs[i + 1] for i in range(len(DTs) - 1)))

    def test_getDateTime_range(self):
        """测试通过 start_dt 过滤时点序列"""
        TableName = self.FDB.TableNames[0]
        FT = self.FDB.getTable(TableName, args={"MultiMapping": False})
        AllDTs = FT.getDateTime()
        if len(AllDTs) < 3:
            self.skipTest("时点数量不足，跳过范围过滤测试")
        # 过滤后的时点列表应为全集的子集
        FilteredDTs = FT.getDateTime(start_dt=AllDTs[0])
        self.assertIsInstance(FilteredDTs, list)
        self.assertGreater(len(FilteredDTs), 0)
        self.assertLessEqual(len(FilteredDTs), len(AllDTs))
        self.assertTrue(all(d in AllDTs for d in FilteredDTs))


class TestSQLDBMeta(unittest.TestCase):
    """SQLDB 侧表元数据功能测试（测试按数字前缀顺序执行，有状态依赖）"""

    @classmethod
    def setUpClass(cls):
        """创建 SQLDB 实例并连接数据库，使用独立侧表名隔离测试"""
        cls.FDB = SQLDB(
            args={
                "DBType": "PostgreSQL",
                "InnerPrefix": "qsd_",
                "MetaTableName": "_qstest_meta",
                "IgnoreFields": [],
            }
        ).connect()

    @classmethod
    def tearDownClass(cls):
        """清理测试产生的元数据和临时表"""
        try:
            DBMetaTableName = cls.FDB._QSArgs.TablePrefix + "qsd__qstest_meta"
            cls.FDB.execute(f"DROP TABLE IF EXISTS {DBMetaTableName}")
        except Exception:
            pass
        try:
            for t in [_TEST_TABLE, _TEST_TABLE_2]:
                cls.FDB.execute(f"DROP TABLE IF EXISTS qsd_{t}")
        except Exception:
            pass

    # ==================== 侧表创建与加载 ====================

    def test_meta_01_createMetaTable(self):
        """connect() 自动创建元数据侧表"""
        DBMetaTableName = self.FDB._QSArgs.TablePrefix + "qsd__qstest_meta"
        SQLStr = f"""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = '{DBMetaTableName}'
        """
        Result = self.FDB.fetchall(SQLStr)
        self.assertEqual(len(Result), 1, f"元数据侧表 {DBMetaTableName} 应被自动创建")

    # ==================== 表级元数据 ====================

    def test_meta_02_setTableMetaData(self):
        """setTableMetaData 写入表级元数据"""
        self.FDB.createTable(_TEST_TABLE, {"test_factor": "DOUBLE PRECISION"})

        self.FDB.setTableMetaData(_TEST_TABLE, key="Description", value="测试描述")
        self.FDB.setTableMetaData(_TEST_TABLE, meta_data={
            "data_source": "test_source",
            "update_freq": "daily",
        })

        FT = self.FDB.getTable(_TEST_TABLE, args={"MultiMapping": False})
        Meta = FT.getMetaData()
        self.assertEqual(Meta.get("Description"), "测试描述")
        self.assertEqual(Meta.get("data_source"), "test_source")
        self.assertEqual(Meta.get("update_freq"), "daily")
        self.assertEqual(FT.getMetaData(key="Description"), "测试描述")
        self.assertIsNone(FT.getMetaData(key="nonexistent_key"))

    # ==================== 因子级元数据 ====================

    def test_meta_03_setFactorMetaData(self):
        """setFactorMetaData 写入因子级元数据"""
        self.FDB.setFactorMetaData(_TEST_TABLE, "test_factor", key="unit", value="元")
        self.FDB.setFactorMetaData(_TEST_TABLE, "test_factor", meta_data={
            "category": "基本面",
            "source_table": "income_statement",
        })

        FT = self.FDB.getTable(_TEST_TABLE, args={"MultiMapping": False})
        Meta = FT.getFactorMetaData(factor_names=["test_factor"], key=None)
        self.assertIn("unit", Meta.columns)
        self.assertEqual(Meta.loc["test_factor", "unit"], "元")
        self.assertEqual(Meta.loc["test_factor", "category"], "基本面")
        self.assertEqual(Meta.loc["test_factor", "source_table"], "income_statement")
        self.assertEqual(
            FT.getFactorMetaData(factor_names=["test_factor"], key="unit").loc["test_factor"],
            "元",
        )

    def test_meta_04_getFactorMetaData_all_keys(self):
        """getFactorMetaData(key=None) 返回所有元数据列"""
        FT = self.FDB.getTable(_TEST_TABLE, args={"MultiMapping": False})
        Meta = FT.getFactorMetaData(key=None)

        self.assertIn("DataType", Meta.columns)
        self.assertIn("unit", Meta.columns)
        self.assertIn("category", Meta.columns)
        self.assertIn(Meta.loc["test_factor", "DataType"], ("double", "string", "object"))

    # ==================== 删除元数据 ====================

    def test_meta_05_deleteMetaData(self):
        """value=None 时删除元数据"""
        self.FDB.setFactorMetaData(_TEST_TABLE, "test_factor", key="unit", value=None)

        FT = self.FDB.getTable(_TEST_TABLE, args={"MultiMapping": False})
        Result = FT.getFactorMetaData(factor_names=["test_factor"], key="unit")
        self.assertTrue(pd.isnull(Result.loc["test_factor"]))

    # ==================== 级联更新 ====================

    def test_meta_06_renameTable_cascade(self):
        """renameTable 级联更新侧表"""
        self.FDB.setTableMetaData(_TEST_TABLE, key="Description", value="重命名前")
        self.FDB.renameTable(_TEST_TABLE, _TEST_TABLE_2)

        FT = self.FDB.getTable(_TEST_TABLE_2, args={"MultiMapping": False})
        self.assertEqual(FT.getMetaData(key="Description"), "重命名前")
        self.assertEqual(
            FT.getFactorMetaData(factor_names=["test_factor"], key="category").loc["test_factor"],
            "基本面",
        )

    def test_meta_07_renameFactor_cascade(self):
        """renameFactor 级联更新侧表"""
        self.FDB.renameFactor(_TEST_TABLE_2, "test_factor", "test_factor_v2")

        FT = self.FDB.getTable(_TEST_TABLE_2, args={"MultiMapping": False})
        self.assertIn("test_factor_v2", FT.FactorNames)
        self.assertNotIn("test_factor", FT.FactorNames)
        self.assertEqual(
            FT.getFactorMetaData(factor_names=["test_factor_v2"], key="category").loc["test_factor_v2"],
            "基本面",
        )

    def test_meta_08_deleteFactor_cascade(self):
        """deleteFactor 级联删除侧表"""
        self.FDB.addFactor(_TEST_TABLE_2, {"extra_factor": "DOUBLE PRECISION"})
        self.FDB.setFactorMetaData(_TEST_TABLE_2, "extra_factor", key="tag", value="extra")

        self.FDB.deleteFactor(_TEST_TABLE_2, ["extra_factor"])

        FT = self.FDB.getTable(_TEST_TABLE_2, args={"MultiMapping": False})
        self.assertNotIn("extra_factor", FT.FactorNames)
        self.assertIn("test_factor_v2", FT.FactorNames)

    def test_meta_09_deleteTable_cascade(self):
        """deleteTable 级联删除侧表"""
        self.FDB.deleteTable(_TEST_TABLE_2)
        self.assertNotIn(_TEST_TABLE_2, self.FDB.TableNames)
        self.assertNotIn(_TEST_TABLE, self.FDB.TableNames)

    # ==================== Description 列注释 ====================

    def test_meta_10_descriptionFromColumnComment(self):
        """Description 从 SQL 列注释中读取"""
        self.FDB.createTable(_TEST_TABLE, {"factor_a": "DOUBLE PRECISION"})
        self.FDB.execute(f"COMMENT ON COLUMN qsd_{_TEST_TABLE}.factor_a IS '这是因子A的描述'")
        self.FDB.connect()

        FT = self.FDB.getTable(_TEST_TABLE, args={"MultiMapping": False})
        Desc = FT.getFactorMetaData(factor_names=["factor_a"], key="Description")
        self.assertEqual(Desc.loc["factor_a"], "这是因子A的描述")
        self.FDB.deleteTable(_TEST_TABLE)


if __name__ == "__main__":
    unittest.main()
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestSQLDB("test_connect"))
    # Runner = unittest.TextTestRunner()
    # Runner.run(Suite)
