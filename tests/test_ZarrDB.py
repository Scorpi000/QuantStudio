# -*- coding: utf-8 -*-
import os
import datetime as dt
import tempfile
import unittest

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.ZarrDB import ZarrDB
from test_HDF5DB import compareDataFrame

class TestZarrDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TestZarrDB.TempDir = tempfile.TemporaryDirectory()
        TestZarrDB.FDB = ZarrDB(sys_args={"主目录":TestZarrDB.TempDir.name})
    # 测试数据读写
    def test_DataIO(self):
        self.FDB.connect()
        TestTable = "TestTable_DataIO"
        TestFactor1 = "TestFactor1_DataIO"
        TestFactor2 = "TestFactor2_DataIO"
        DTs =  [dt.datetime(2018,1,1)+dt.timedelta(i) for i in range(4)]
        IDs = [("00000%d.SZ" % (i,)) for i in range(3)]
        TargetData = pd.Panel(np.zeros((2, len(DTs), len(IDs))), items=[TestFactor1, TestFactor2], major_axis=DTs, minor_axis=IDs)
        # 创建因子表并写入数据
        self.FDB.writeData(TargetData.iloc[:, 0:2, 0:1], TestTable)
        FT = self.FDB.getTable(TestTable)
        TestData = FT.readData(factor_names=[TestFactor1, TestFactor2], ids=IDs[0:1], dts=DTs[0:2])
        Err = compareDataFrame(TestData.iloc[0], TargetData.iloc[0, 0:2, 0:1], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        Err = compareDataFrame(TestData.iloc[1], TargetData.iloc[1, 0:2, 0:1], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        # 以 update 方式写入数据
        TargetData.iloc[0, 1:3, 0:2] = 1.0
        self.FDB.writeData(TargetData.iloc[:, 1:3, 0:2], TestTable, if_exists="update")
        TestData = FT.readData(factor_names=[TestFactor1, TestFactor2], ids=IDs[0:2], dts=DTs[0:3])
        TestData.iloc[:, 0, 1] = 0
        Err = compareDataFrame(TestData.iloc[0], TargetData.iloc[0, 0:3, 0:2], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        Err = compareDataFrame(TestData.iloc[1], TargetData.iloc[1, 0:3, 0:2], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        # 以 append 方式写入数据
        TargetData.iloc[0, 2:4, 2] = 2.0
        self.FDB.writeData(TargetData.iloc[:, 2:4, 0:3], TestTable, if_exists="append")
        TestData = FT.readData(factor_names=[TestFactor1, TestFactor2], ids=IDs[0:3], dts=DTs[0:4])
        TestData.iloc[:, 0, 1] = 0
        TestData.iloc[:, 0:2, 2] = 0
        Err = compareDataFrame(TestData.iloc[0], TargetData.iloc[0], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
        Err = compareDataFrame(TestData.iloc[1], TargetData.iloc[1], dtype="double")
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试 object 类型的数据读写
    def test_ObjectDataIO(self):
        TestTable = "TestTable_ObjectDataIO"
        TestFactor = "TestFactor_ObjectDataIO"
        DTs = [dt.datetime(2019,1,1), dt.datetime(2019,1,2), dt.datetime(2019,1,3)]
        IDs = ["000001.SZ", "600000.SH"]
        TargetData = np.full(shape=(3,2), fill_value=None, dtype="O")
        TargetData[0,0] = [1,2,3]
        TargetData[1,1] = {"测试": {"a": ["数据"]}}
        TargetData = pd.DataFrame(TargetData, index=DTs, columns=IDs)
        self.FDB.connect()
        self.FDB.writeData(pd.Panel({TestFactor: TargetData}), TestTable)
        FT = self.FDB.getTable(TestTable)
        TestData = FT.readData(factor_names=[TestFactor], ids=IDs, dts=DTs).iloc[0]
        Err = compareDataFrame(TestData, TargetData, dtype="object")
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试 ID 读取
    def test_getID(self):
        TestTable = "TestTable_getID"
        TestFactor = "TestFactor_getID"
        DTs = [dt.datetime(2019,1,1), dt.datetime(2019,1,2), dt.datetime(2019,1,3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3,2)), index=DTs, columns=IDs)
        self.FDB.connect()
        self.FDB.writeData(pd.Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        TestIDs = FT.getID()
        self.assertListEqual(IDs, TestIDs)
    # 测试时点读取
    def test_getDateTime(self):
        TestTable = "TestTable_getDateTime"
        TestFactor = "TestFactor_getDateTime"
        DTs = [dt.datetime(2019,1,1), dt.datetime(2019,1,2), dt.datetime(2019,1,3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3,2)), index=DTs, columns=IDs)
        self.FDB.connect()
        self.FDB.writeData(pd.Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        TestDTs = FT.getDateTime()
        self.assertListEqual(DTs, TestDTs)
    # 测试因子重命名
    def test_renameFactor(self):
        TestTable = "TestTable_renameFactor"
        TestFactor = "TestFactor_renameFactor"
        DTs = [dt.datetime(2019,1,1), dt.datetime(2019,1,2), dt.datetime(2019,1,3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3,2)), index=DTs, columns=IDs)
        self.FDB.connect()
        self.FDB.writeData(pd.Panel({TestFactor: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        self.assertTrue(TestFactor in FT.FactorNames)
        NewFactorName = "New_"+TestFactor
        self.FDB.renameFactor(TestTable, TestFactor, NewFactorName)
        self.assertFalse(TestFactor in FT.FactorNames)
        self.assertTrue(NewFactorName in FT.FactorNames)
    # 测试因子删除
    def test_deleteFactor(self):
        TestTable = "TestTable_deleteFactor"
        TestFactor1 = "TestFactor1_deleteFactor"
        TestFactor2 = "TestFactor2_deleteFactor"
        DTs = [dt.datetime(2019,1,1), dt.datetime(2019,1,2), dt.datetime(2019,1,3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3,2)), index=DTs, columns=IDs)
        self.FDB.connect()
        self.FDB.writeData(pd.Panel({TestFactor1: Data, TestFactor2: Data}), TestTable)
        FT = self.FDB.getTable(TestTable)
        self.assertTrue(TestFactor1 in FT.FactorNames)
        self.FDB.deleteFactor(TestTable, [TestFactor1])
        self.assertFalse(TestFactor1 in FT.FactorNames)
        self.assertTrue(TestFactor2 in FT.FactorNames)
        self.FDB.deleteFactor(TestTable, [TestFactor2])
        self.assertFalse(TestTable in self.FDB.TableNames)
    # 测试表重命名
    def test_renameTable(self):
        TestTable = "TestTable_renameTable"
        TestFactor = "TestFactor_renameTable"
        DTs = [dt.datetime(2019,1,1), dt.datetime(2019,1,2), dt.datetime(2019,1,3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3,2)), index=DTs, columns=IDs)
        self.FDB.connect()
        self.FDB.writeData(pd.Panel({TestFactor: Data}), TestTable)
        self.assertTrue(TestTable in self.FDB.TableNames)
        NewTableName = "New_"+TestTable
        self.FDB.renameTable(TestTable, NewTableName)
        self.assertFalse(TestTable in self.FDB.TableNames)
        self.assertTrue(NewTableName in self.FDB.TableNames)
    # 测试表删除
    def test_deleteTable(self):
        TestTable = "TestTable_deleteTable"
        TestFactor = "TestFactor_deleteTable"
        DTs = [dt.datetime(2019,1,1), dt.datetime(2019,1,2), dt.datetime(2019,1,3)]
        IDs = ["000001.SZ", "600000.SH"]
        Data = pd.DataFrame(np.ones(shape=(3,2)), index=DTs, columns=IDs)
        self.FDB.connect()
        self.FDB.writeData(pd.Panel({TestFactor: Data}), TestTable)
        self.assertTrue(TestTable in self.FDB.TableNames)
        self.FDB.deleteTable(table_name=TestTable)
        self.assertFalse(TestTable in self.FDB.TableNames)


if __name__=="__main__":
    #unittest.main()
    
    Suite = unittest.TestSuite()
    Suite.addTest(TestZarrDB("test_DataIO"))
    Runner = unittest.TextTestRunner()
    Runner.run(Suite)