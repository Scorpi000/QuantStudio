# -*- coding: utf-8 -*-
import os
import datetime as dt
import tempfile
import unittest

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.HDF5DB import HDF5DB

class TestHDF5DB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TestHDF5DB.TempDir = tempfile.TemporaryDirectory()
        TestHDF5DB.TargetTable = "TestTable"
        TestHDF5DB.FactorNames = ["Factor0", "Factor1"]
        TestHDF5DB.IDs = [("00000%d.SZ" % (i,)) for i in range(3)]
        TestHDF5DB.DTs = [dt.datetime(2018,1,1)+dt.timedelta(i) for i in range(4)]
        TestHDF5DB.Data = pd.Panel(np.zeros((len(TestHDF5DB.FactorNames), len(TestHDF5DB.DTs), len(TestHDF5DB.IDs))), items=TestHDF5DB.FactorNames, major_axis=TestHDF5DB.DTs, minor_axis=TestHDF5DB.IDs)
        TestHDF5DB.FDB = HDF5DB(sys_args={"主目录":TestHDF5DB.TempDir.name})
    # 测试连接功能
    def test_1_connect(self):
        self.FDB.connect()
    # 测试写入功能
    def test_2_writeData(self):
        # 创建因子表并写入数据
        self.FDB.writeData(self.Data.iloc[:, 0:2, 0:1], self.TargetTable)
        #self.assertTrue(self.TargetTable in os.listdir(self.TempDir.name))
        #self.assertTrue({iFactor+".hdf5" for iFactor in self.FactorNames}.issubset(os.listdir(self.TempDir.name+os.sep+self.TargetTable)))
        # 以 update 方式写入数据
        self.Data.iloc[0, 1:3, 0:2] = 1.0
        self.FDB.writeData(self.Data.iloc[:, 1:3, 0:2], self.TargetTable, if_exists="update")
        # 以 append 方式写入数据
        self.Data.iloc[0, 2:4, 2] = 2.0
        self.FDB.writeData(self.Data.iloc[:, 2:4, 0:3], self.TargetTable, if_exists="append")
    # 测试读取功能
    def test_3_readData(self):
        self.assertEqual(self.FDB.TableNames, [self.TargetTable])
        FT = self.FDB.getTable(self.TargetTable)
        self.assertEqual(FT.Name, self.TargetTable)
        FactorNames = FT.FactorNames
        self.assertEqual(FactorNames, self.FactorNames)
        IDs = FT.getID()
        self.assertEqual(IDs, self.IDs)
        DTs = FT.getDateTime()
        self.assertEqual(DTs, self.DTs)
        Data = FT.readData(factor_names=FactorNames, ids=IDs, dts=DTs)
        Data.iloc[0, 0:2, 2] = 0
        Err = (Data.iloc[0] - self.Data.iloc[0]).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        self.assertAlmostEqual(Data.iloc[1].abs().sum().sum(), 0)
    # 测试因子表的操作
    def test_4_alterData(self):
        # 重命名因子
        OldFactorName, self.FactorNames[1] = self.FactorNames[1], "New"+self.FactorNames[1]
        self.FDB.renameFactor(self.TargetTable, OldFactorName, self.FactorNames[1])
        self.assertSetEqual(set(self.FDB.getTable(self.TargetTable).FactorNames), set(self.FactorNames))
        #self.assertTrue(self.FactorNames[1]+".hdf5" in os.listdir(self.TempDir.name+os.sep+self.TargetTable))
        # 删除因子
        self.FDB.deleteFactor(self.TargetTable, [self.FactorNames[1]])
        self.assertSetEqual(set(self.FDB.getTable(self.TargetTable).FactorNames), {self.FactorNames[0]})
        #self.assertTrue(self.FactorNames[1]+".hdf5" not in os.listdir(self.TempDir.name+os.sep+self.TargetTable))
        # 重命名表
        OldTargetTable, self.TargetTable = self.TargetTable, "New"+self.TargetTable
        self.FDB.renameTable(OldTargetTable, self.TargetTable)
        self.assertTrue(OldTargetTable not in self.FDB.TableNames)
        self.assertTrue(self.TargetTable in self.FDB.TableNames)
        #self.assertTrue(self.TargetTable in os.listdir(self.TempDir.name))
        # 删除表
        self.FDB.deleteTable(table_name=self.TargetTable)
        self.assertListEqual(self.FDB.TableNames, [])
        #self.assertTrue(self.TargetTable not in os.listdir(self.TempDir.name))

if __name__=="__main__":
    unittest.main()