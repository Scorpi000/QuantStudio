# -*- coding: utf-8 -*-
import os
import datetime as dt
import tempfile
import unittest

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.FactorDB import CustomFT, DataFactor
from QuantStudio.FactorDataBase.HDF5DB import HDF5DB

class TestFactorTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nDT, nID = 10, 365
        TestFactorTable.FactorNames = ["Factor0", "Factor1"]
        TestFactorTable.IDs = [("00000%d.SZ" % (i,)) for i in range(nID)]
        TestFactorTable.DTs = [dt.datetime(2018,1,1)+dt.timedelta(i) for i in range(nDT)]
        np.random.seed(0)
        TestFactorTable.Data0 = pd.DataFrame(np.random.randn(nDT, nID), index=TestFactorTable.DTs, columns=TestFactorTable.IDs)
        TestFactorTable.Factor0 = DataFactor(name=TestFactorTable.FactorNames[0], data=TestFactorTable.Data0)
        TestFactorTable.Data1 = pd.DataFrame(np.random.randn(nDT, nID), index=TestFactorTable.DTs, columns=TestFactorTable.IDs)
        TestFactorTable.Factor1 = DataFactor(name=TestFactorTable.FactorNames[1], data=TestFactorTable.Data1)
        TestFactorTable.CFT = CustomFT(name="TestFactorTable")
        TestFactorTable.CFT.addFactors(factor_list=[TestFactorTable.Factor0, TestFactorTable.Factor1])
        TestFactorTable.CFT.setID(TestFactorTable.IDs)
        TestFactorTable.CFT.setDateTime(TestFactorTable.DTs)
    # 测试随机计算
    def test_1_RandomCalc(self):
        TargetData = self.Data0.loc[self.DTs[-4:], self.IDs[-3:]]
        TestData = self.CFT.readData(factor_names=[self.FactorNames[0]], ids=self.IDs[-3:], dts=self.DTs[-4:]).iloc[0]
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试遍历计算
    def test_2_ErgodicCalc(self):
        TargetData = self.Data1.mean(axis=1) - self.Data0.mean(axis=1)
        TestData = pd.Series(np.nan, index=self.DTs)
        self.CFT["遍历模式"]["向前缓冲时点数"] = 122
        self.CFT.start(self.DTs)
        for iDT in self.DTs:
            self.CFT.move(iDT)
            iData = self.CFT.readData(factor_names=self.FactorNames, ids=self.IDs, dts=[iDT]).iloc[:, 0, :]
            TestData.loc[iDT] = iData.mean(axis=0).diff().iloc[1]
        self.CFT.end()
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max(), 0)
    # 测试批量计算
    def test_3_BatchCalc(self):
        TargetData = pd.Panel({self.FactorNames[0]:self.Data0, self.FactorNames[1]:self.Data1})
        TempDir = tempfile.TemporaryDirectory()
        FDB = HDF5DB(sys_args={"主目录": TempDir.name})
        FDB.connect()
        self.CFT.write2FDB(self.FactorNames, self.IDs, self.DTs, FDB, self.CFT.Name, if_exists="update", subprocess_num=0, dt_ruler=None, section_ids=None)
        FT = FDB.getTable(self.CFT.Name)
        TestData = FT.readData(factor_names=self.FactorNames, ids=self.IDs, dts=self.DTs)
        for iFactorName in self.FactorNames:
            Err = (TestData.loc[iFactorName] - TargetData.loc[iFactorName]).abs()
            self.assertAlmostEqual(Err.max().max(), 0)
        FDB.disconnect()

if __name__=="__main__":
    unittest.main()