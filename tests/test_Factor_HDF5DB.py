# -*- coding: utf-8 -*-
"""QuantStudio HDF5DB 因子库测试模块

使用方法:
    * 运行全部测试: python tests/test_Factor_HDF5DB.py
    * 运行指定测试: python -m unittest tests.test_Factor_HDF5DB.TestHDF5DB.test_DataIO
    * 通过 TestSuite 指定测试 (取消文件末尾的注释并修改):

        Suite = unittest.TestSuite()
        Suite.addTest(TestHDF5DB("test_DataIO"))
        Runner = unittest.TextTestRunner()
        Runner.run(Suite)

测试方法:
    test_DataIO           — double 类型因子数据的写入（update/append 模式）和读取
    test_ObjectDataIO     — object 类型因子数据的读写
    test_readFactorData   — readFactorData 直接返回 DataFrame, 及 Factor.readData
    test_writeFactorData  — writeFactorData 单因子写入和 update 追加
    test_getID            — 获取表中 ID 序列, 支持指定因子名
    test_getDateTime      — 获取表中时点序列, 支持指定因子名和日期范围过滤
    test_getFactor        — getFactor 获取因子对象, 及 FT["factor"] 语法糖
    test_getitem_table    — FDB["table_name"] 语法糖
    test_renameFactor     — 因子重命名
    test_deleteFactor     — 因子删除（含删光因子后自动删表）
    test_renameTable      — 表重命名
    test_deleteTable      — 表删除
    test_TableMetaData    — 因子表元数据读写 (setTableMetaData / getMetaData)
    test_FactorMetaData   — 因子元数据读写 (setFactorMetaData / getFactorMetaData)
"""
import datetime as dt
import tempfile
import unittest

import numpy as np
import pandas as pd

from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.HDF5DB import HDF5DB


def compareDataFrame(df1, df2, dtype="double"):
    """比较两个 DataFrame，返回逐元素的差异矩阵

    Args:
        df1, df2: 待比较的 DataFrame
        dtype: 数据类型，"double" 表示数值比较，"object" 表示对象比较

    Returns:
        逐元素差异的 DataFrame，差异为 0 表示两值相等
    """
    m1, m2 = pd.isna(df1), pd.isna(df2)
    if dtype == "double":
        Err = (df1.astype(float) - df2.astype(float)).abs()
        Err[pd.isna(Err)] = 0
        return np.maximum(Err, (m1 ^ m2).astype("float"))
    else:
        Err = (df1 != df2)
        Err[m1 | m2] = False
        return np.maximum(Err.astype("float"), (m1 ^ m2).astype("float"))


class TestHDF5DB(unittest.TestCase):
    """HDF5DB 因子库测试"""

    @classmethod
    def setUpClass(cls):
        """创建临时目录和 HDF5DB 实例"""
        cls.TempDir = tempfile.TemporaryDirectory()
        cls.FDB = HDF5DB(args={"MainDir": cls.TempDir.name}).connect()

    @classmethod
    def tearDownClass(cls):
        """清理临时目录"""
        cls.TempDir.cleanup()

    # ==================== 数据读写测试 ====================

    def test_DataIO(self):
        """测试 double 类型因子数据的写入（update/append 模式）和读取"""
        TestTable = "TestTable_DataIO"
        TestFactor1 = "TestFactor1_DataIO"
        TestFactor2 = "TestFactor2_DataIO"
        DTs = [dt.datetime(2018, 1, 1) + dt.timedelta(i) for i in range(4)]
        IDs = ["00000%d.SZ" % i for i in range(3)]
        TargetData = Panel(
            np.zeros((2, len(DTs), len(IDs))),
            items=[TestFactor1, TestFactor2],
            major_axis=DTs,
            minor_axis=IDs,
        )
        # 创建因子表并写入数据
        self.FDB.writeData(TargetData.iloc[:, 0:2, 0:1], TestTable)
        FT = self.FDB.getTable(TestTable)
        TestData = FT.readData(
            factor_names=[TestFactor1, TestFactor2], ids=IDs[0:1], dts=DTs[0:2]
        )
        Err = compareDataFrame(TestData.iloc[0], TargetData.iloc[0, 0:2, 0:1], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        Err = compareDataFrame(TestData.iloc[1], TargetData.iloc[1, 0:2, 0:1], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        # 以 update 方式写入数据
        TargetData.iloc[0, 1:3, 0:2] = 1.0
        self.FDB.writeData(TargetData.iloc[:, 1:3, 0:2], TestTable, if_exists="update")
        TestData = FT.readData(
            factor_names=[TestFactor1, TestFactor2], ids=IDs[0:2], dts=DTs[0:3]
        )
        TestData.iloc[:, 0, 1] = 0
        Err = compareDataFrame(TestData.iloc[0], TargetData.iloc[0, 0:3, 0:2], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        Err = compareDataFrame(TestData.iloc[1], TargetData.iloc[1, 0:3, 0:2], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        # 以 append 方式写入数据
        TargetData.iloc[0, 2:4, 2] = 2.0
        self.FDB.writeData(TargetData.iloc[:, 2:4, 0:3], TestTable, if_exists="append")
        TestData = FT.readData(
            factor_names=[TestFactor1, TestFactor2], ids=IDs[0:3], dts=DTs[0:4]
        )
        TestData.iloc[:, 0, 1] = 0
        TestData.iloc[:, 0:2, 2] = 0
        Err = compareDataFrame(TestData.iloc[0], TargetData.iloc[0], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        Err = compareDataFrame(TestData.iloc[1], TargetData.iloc[1], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)

    def test_ObjectDataIO(self):
        """测试 object 类型因子数据的读写"""
        TestTable = "TestTable_ObjectDataIO"
        TestFactor = "TestFactor_ObjectDataIO"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        TargetData = np.full(shape=(3, 2), fill_value=None, dtype="O")
        TargetData[0, 0] = [1, 2, 3]
        TargetData[1, 1] = {"测试": {"a": ["数据"]}}
        TargetData = pd.DataFrame(TargetData, index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: TargetData}), TestTable)
        FT = self.FDB.getTable(TestTable)
        TestData = FT.readData(factor_names=[TestFactor], ids=IDs, dts=DTs).iloc[0]
        Err = compareDataFrame(TestData, TargetData, dtype="object")
        self.assertAlmostEqual(Err.max().max(), 0)

    # ==================== 单因子读写测试 ====================

    def test_readFactorData(self):
        """测试 readFactorData 直接返回 DataFrame"""
        TestTable = "TestTable_readFactorData"
        TestFactor = "TestFactor_readFactorData"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.arange(6).reshape(3, 2).astype(float), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        # 用 readFactorData 直接读取
        TestData = FT.readFactorData(ifactor_name=TestFactor, ids=IDs, dts=DTs)
        Err = compareDataFrame(TestData, Data)
        self.assertAlmostEqual(Err.max().max(), 0)
        # 用 Factor 对象的 readData 读取
        F = FT.getFactor(TestFactor)
        FactorData = F.readData(ids=IDs, dts=DTs)
        Err = compareDataFrame(FactorData, Data)
        self.assertAlmostEqual(Err.max().max(), 0)

    def test_writeFactorData(self):
        """测试 writeFactorData 写入单因子数据"""
        TestTable = "TestTable_writeFactorData"
        TestFactor = "TestFactor_writeFactorData"
        DTs = [dt.datetime(2019, 1, 4), dt.datetime(2019, 1, 5)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones((2, 2)), index=DTs, columns=IDs)
        self.FDB.writeFactorData(Data, TestTable, TestFactor, if_exists="update")
        FT = self.FDB.getTable(TestTable)
        TestData = FT.readFactorData(ifactor_name=TestFactor, ids=IDs, dts=DTs)
        Err = compareDataFrame(TestData, Data)
        self.assertAlmostEqual(Err.max().max(), 0)
        # update 追加数据
        NewData = pd.DataFrame(np.full((2, 2), 2.0), index=DTs, columns=IDs)
        self.FDB.writeFactorData(NewData, TestTable, TestFactor, if_exists="update")
        TestData = FT.readFactorData(ifactor_name=TestFactor, ids=IDs, dts=DTs)
        Err = compareDataFrame(TestData, NewData)
        self.assertAlmostEqual(Err.max().max(), 0)

    # ==================== ID 和时点读取测试 ====================

    def test_getID(self):
        """测试获取表中的 ID 序列"""
        TestTable = "TestTable_getID"
        TestFactor = "TestFactor_getID"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3, 2)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        TestIDs = FT.getID()
        self.assertListEqual(sorted(IDs), sorted(TestIDs))
        # 测试通过指定因子名获取 ID
        TestIDs2 = FT.getID(ifactor_name=TestFactor)
        self.assertListEqual(sorted(IDs), sorted(TestIDs2))

    def test_getDateTime(self):
        """测试获取表中的时点序列"""
        TestTable = "TestTable_getDateTime"
        TestFactor = "TestFactor_getDateTime"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3, 2)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        TestDTs = FT.getDateTime()
        self.assertListEqual(DTs, TestDTs)
        # 测试通过指定因子名获取时点
        TestDTs2 = FT.getDateTime(ifactor_name=TestFactor)
        self.assertListEqual(DTs, TestDTs2)
        # 测试通过日期范围过滤
        TestDTs3 = FT.getDateTime(start_dt=dt.datetime(2019, 1, 2))
        self.assertListEqual(DTs[1:], TestDTs3)

    # ==================== 因子操作测试 ====================

    def test_getFactor(self):
        """测试获取表中的因子对象"""
        TestTable = "TestTable_getFactor"
        TestFactor = "TestFactor_getFactor"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones((2, 2)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        # 通过 getFactor 获取因子对象
        F = FT.getFactor(TestFactor)
        self.assertEqual(F.Name, TestFactor)
        # 通过 __getitem__ 语法糖获取因子对象
        F2 = FT[TestFactor]
        self.assertEqual(F2.Name, TestFactor)
        # 因子对象应能读取数据
        FactorData = F.readData(ids=IDs, dts=DTs)
        Err = compareDataFrame(FactorData, Data)
        self.assertAlmostEqual(Err.max().max(), 0)

    def test_getitem_table(self):
        """测试 FDB["table_name"] 语法糖"""
        TestTable = "TestTable_getitem_table"
        TestFactor = "TestFactor_getitem_table"
        DTs = [dt.datetime(2019, 1, 1)]
        IDs = ["000001.SZ"]
        Data = pd.DataFrame(np.ones((1, 1)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        # FDB["table_name"] 等同于 FDB.getTable("table_name")
        FT1 = self.FDB.getTable(TestTable)
        FT2 = self.FDB[TestTable]
        self.assertListEqual(FT1.FactorNames, FT2.FactorNames)

    def test_renameFactor(self):
        """测试因子重命名"""
        TestTable = "TestTable_renameFactor"
        TestFactor = "TestFactor_renameFactor"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3, 2)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        self.assertTrue(TestFactor in FT.FactorNames)
        NewFactorName = "New_" + TestFactor
        self.FDB.renameFactor(TestTable, TestFactor, NewFactorName)
        self.assertFalse(TestFactor in FT.FactorNames)
        self.assertTrue(NewFactorName in FT.FactorNames)

    def test_deleteFactor(self):
        """测试因子删除，以及删除全部因子后自动删除表"""
        TestTable = "TestTable_deleteFactor"
        TestFactor1 = "TestFactor1_deleteFactor"
        TestFactor2 = "TestFactor2_deleteFactor"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3, 2)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor1: Data, TestFactor2: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        self.assertTrue(TestFactor1 in FT.FactorNames)
        self.FDB.deleteFactor(TestTable, [TestFactor1])
        self.assertFalse(TestFactor1 in FT.FactorNames)
        self.assertTrue(TestFactor2 in FT.FactorNames)
        self.FDB.deleteFactor(TestTable, [TestFactor2])
        self.assertFalse(TestTable in self.FDB.TableNames)

    # ==================== 表操作测试 ====================

    def test_renameTable(self):
        """测试表重命名"""
        TestTable = "TestTable_renameTable"
        TestFactor = "TestFactor_renameTable"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3, 2)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        self.assertTrue(TestTable in self.FDB.TableNames)
        NewTableName = "New_" + TestTable
        self.FDB.renameTable(TestTable, NewTableName)
        self.assertFalse(TestTable in self.FDB.TableNames)
        self.assertTrue(NewTableName in self.FDB.TableNames)

    def test_deleteTable(self):
        """测试表删除"""
        TestTable = "TestTable_deleteTable"
        TestFactor = "TestFactor_deleteTable"
        DTs = [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3, 2)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        self.assertTrue(TestTable in self.FDB.TableNames)
        self.FDB.deleteTable(table_name=TestTable)
        self.assertFalse(TestTable in self.FDB.TableNames)

    # ==================== 元数据测试 ====================

    def test_TableMetaData(self):
        """测试因子表元数据的读写"""
        TestTable = "TestTable_TableMetaData"
        TestFactor = "TestFactor_TableMetaData"
        DTs = [dt.datetime(2019, 1, 1)]
        IDs = ["000001.SZ"]
        Data = pd.DataFrame(np.ones((1, 1)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        # 设置单条元数据
        self.FDB.setTableMetaData(TestTable, key="description", value="测试表")
        FT = self.FDB.getTable(TestTable)
        Meta = FT.getMetaData()
        self.assertEqual(Meta.get("description"), "测试表")
        # 设置多条元数据
        self.FDB.setTableMetaData(
            TestTable, meta_data={"author": "test", "version": "1.0"}
        )
        Meta = FT.getMetaData()
        self.assertEqual(Meta.get("author"), "test")
        self.assertEqual(Meta.get("version"), "1.0")

    def test_FactorMetaData(self):
        """测试因子元数据的读写"""
        TestTable = "TestTable_FactorMetaData"
        TestFactor = "TestFactor_FactorMetaData"
        DTs = [dt.datetime(2019, 1, 1)]
        IDs = ["000001.SZ"]
        Data = pd.DataFrame(np.ones((1, 1)), index=DTs, columns=IDs)
        self.FDB.writeData(Panel({TestFactor: Data}), TestTable)
        # 设置因子元数据
        self.FDB.setFactorMetaData(
            TestTable, TestFactor, key="DataType", value="double"
        )
        self.FDB.setFactorMetaData(
            TestTable, TestFactor, key="description", value="测试因子"
        )
        FT = self.FDB.getTable(TestTable)
        # 获取单个 key
        DType = FT.getFactorMetaData(factor_names=[TestFactor], key="DataType")
        self.assertEqual(DType[TestFactor], "double")
        # 获取全部元数据
        AllMeta = FT.getFactorMetaData(factor_names=[TestFactor])
        self.assertIn("DataType", AllMeta.columns)
        self.assertIn("description", AllMeta.columns)


if __name__ == "__main__":
    unittest.main()
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestHDF5DB("test_getDateTime"))
    # Runner = unittest.TextTestRunner()
    # Runner.run(Suite)
