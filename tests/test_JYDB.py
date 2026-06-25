# -*- coding: utf-8 -*-
"""QuantStudio JYDB 聚源数据库测试模块

使用方法:
    * 运行全部测试: python tests/test_JYDB.py
    * 运行指定测试: python -m unittest tests.test_JYDB.TestJYDB.test_connect
    * 通过 TestSuite 指定测试 (取消文件末尾的注释并修改):

        Suite = unittest.TestSuite()
        Suite.addTest(TestJYDB("test_connect"))
        Runner = unittest.TextTestRunner()
        Runner.run(Suite)

    注意: JYDB 需要连接到聚源 PostgreSQL 数据库, 测试依赖数据库可用。

测试方法:
    test_connect      — JYDB 对象构造和数据库连接
    test_TableNames   — 获取因子表名称列表, 验证常用表存在
    test_getTable     — 获取因子表对象, 验证 FactorNames 和 getFactor
    test_getTradeDay  — 获取交易日序列, 验证返回类型和排序
    test_getStockID   — 获取股票 ID 序列, 验证返回格式
"""
import datetime as dt
import unittest

from QuantStudio.Factor.JYDB import JYDB


class TestJYDB(unittest.TestCase):
    """JYDB 聚源数据库基本功能测试"""

    @classmethod
    def setUpClass(cls):
        """创建 JYDB 实例并连接数据库"""
        cls.FDB = JYDB().connect()

    # ==================== 连接测试 ====================

    def test_connect(self):
        """测试 JYDB 构造和连接"""
        self.assertIsNotNone(self.FDB)
        self.assertEqual(self.FDB.Name, "JYDB")
        # 连接后可再次调用 connect (幂等)
        self.FDB.connect()

    # ==================== 因子表测试 ====================

    def test_TableNames(self):
        """测试 TableNames 返回表名列表, 验证常用表存在"""
        TableNames = self.FDB.TableNames
        self.assertIsInstance(TableNames, list)
        self.assertGreater(len(TableNames), 0)
        # 常用表应存在
        self.assertIn("日行情表", TableNames)
        self.assertIn("A股证券主表", TableNames)

    def test_getTable(self):
        """测试 getTable 获取因子表对象, 验证 FactorNames 和 getFactor"""
        FT = self.FDB.getTable("A股证券主表")
        self.assertIsNotNone(FT)
        self.assertEqual(FT.Name, "A股证券主表")
        self.assertIn("证券简称", FT.FactorNames)
        # 通过 getFactor 获取单个因子
        F = FT.getFactor("证券简称")
        self.assertEqual(F.Name, "证券简称")
        # 通过 __getitem__ 语法糖获取
        self.assertEqual(FT["证券简称"].Name, "证券简称")

    # ==================== 数据提取测试 ====================

    def test_getTradeDay(self):
        """测试 getTradeDay 获取交易日序列"""
        EndDate = dt.datetime(2025, 12, 31)
        StartDate = dt.datetime(2025, 12, 25)
        DTs = self.FDB.getTradeDay(start_date=StartDate, end_date=EndDate, exchange="SSE")
        self.assertIsInstance(DTs, list)
        self.assertGreater(len(DTs), 0)
        # 交易日应在给定范围内
        self.assertGreaterEqual(DTs[0], StartDate)
        self.assertLessEqual(DTs[-1], EndDate)
        # 交易日应递增
        self.assertTrue(all(DTs[i] < DTs[i + 1] for i in range(len(DTs) - 1)))

    def test_getStockID(self):
        """测试 getStockID 获取股票 ID 序列"""
        TestDate = dt.datetime(2025, 12, 31)
        IDs = self.FDB.getStockID(exchange="SSE", date=TestDate, is_current=True)
        self.assertIsInstance(IDs, list)
        self.assertGreater(len(IDs), 0)
        # ID 应以 .SH 结尾 (上交所)
        self.assertTrue(all(id_str.endswith(".SH") for id_str in IDs))
        # 应包含常见标的
        self.assertIn("600000.SH", IDs)


if __name__ == "__main__":
    unittest.main()
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestJYDB("test_connect"))
    # Runner = unittest.TextTestRunner()
    # Runner.run(Suite)
