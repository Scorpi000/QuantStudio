# -*- coding: utf-8 -*-
"""测试同一个因子对象多个截面同时运行"""
import unittest
import tempfile
import datetime as dt

import numpy as np

from QuantStudio.Core.QSObject import Panel
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorInitData, FactorLocalContext, FactorContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.HDF5DB import HDF5DB


class TestMultiSection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """创建临时目录和 HDF5DB 实例"""
        cls.TempDir = tempfile.TemporaryDirectory()
        cls.FDB = HDF5DB(args={"MainDir": cls.TempDir.name}).connect()

    @classmethod
    def tearDownClass(cls):
        """清理临时目录"""
        cls.TempDir.cleanup()

    def setUp(self):
        self.DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(20)]
        self.SectionIDs = [str(i).zfill(6)+".SZ" for i in range(1, 6)]
        self.Cache = FeatherFactorCache(args={"DTRuler": self.DTRuler})
        self.Context = FactorContext(Mode="DEBUG", PID="0", DTRuler=self.DTRuler, SectionIDs=self.SectionIDs, DataCache=self.Cache)
        self.Engine = Engine()

        TestTable = "TestTable"
        TestFactor1 = "TestFactor1"
        TestFactor2 = "TestFactor2"
        TargetData = Panel(
            np.zeros((2, len(DTs), len(IDs))),
            items=[TestFactor1, TestFactor2],
            major_axis=DTs,
            minor_axis=IDs,
        )
        # 创建因子表并写入数据
        self.FDB.writeData(TargetData.iloc[:, 0:2, 0:1], TestTable)

    def testDataFactor(self):
        """测试 DataFactor 在多截面时运行"""
        Factor = DataFactor(data=1, args={"Name": "TestFactor"})
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, Factor]
        InitDataList = [
            FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=self.SectionIDs[:3]),
            FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=self.SectionIDs[-3:]),
        ]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[:3]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:])
        ]
        with self.Cache:
            Rslt = self.Engine.run(
                node_list=FactorList, 
                context=self.Context,
                init_data_list=InitDataList,
                fwd_data_list=FwdDataList
            )
        print(Rslt[0])
        print(Rslt[1])

    def testFactorTableFactor(self):
        pass

    def testPointOperation(self):
        pass

    def testTimeOperation(self):
        pass

    def testSectionOperation(self):
        pass

    def testPanelOperation(self):
        pass

if __name__ == "__main__":
    unittest.main()
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestSQLDB("testDataFactor"))
    # Runner = unittest.TextTestRunner()
    # Runner.run(Suite)