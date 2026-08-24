# -*- coding: utf-8 -*-
"""测试同一个因子对象多个截面同时运行

使用方法:
    * 运行全部测试: python tests/test_Factor_MultiSection.py
    * 运行指定测试: python -m unittest tests.test_Factor_MultiSection.TestMultiSection.testContext
    * 通过 TestSuite 指定测试 (取消文件末尾的注释并修改):

        Suite = unittest.TestSuite()
        Suite.addTest(TestMultiSection("testContext"))
        Runner = unittest.TextTestRunner()
        Runner.run(Suite)
"""
import unittest
import tempfile
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Core.QSObject import Panel
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorLocalContext, FactorContext, makeFactorRunningKey
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor import FactorOperator as fo
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

        # 创建因子表并写入数据
        self.TestTable = "TestTable"
        self.TestFactor1 = "TestFactor1"
        self.TestFactor2 = "TestFactor2"
        np.random.seed(0)
        self.TestData = Panel(
            np.random.randn(2, len(self.DTRuler), len(self.SectionIDs)),
            items=[self.TestFactor1, self.TestFactor2],
            major_axis=self.DTRuler,
            minor_axis=self.SectionIDs,
        )
        self.FDB.writeData(self.TestData, self.TestTable)

    def testContext(self):
        """测试在多截面时运行时全局上下文的状态"""
        FT = self.FDB.getTable(self.TestTable)
        Factor = FT.getFactor(self.TestFactor1)
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, Factor]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[:3]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        self.assertListEqual(list(self.Context.NodeState), [Factor.QSID])
        RunningKey1 = makeFactorRunningKey(qsid=Factor.QSID, section_ids=self.SectionIDs[:3], context=self.Context)
        RunningKey2 = makeFactorRunningKey(qsid=Factor.QSID, section_ids=self.SectionIDs[-3:], context=self.Context)
        self.assertListEqual(sorted(self.Context.NodeState[Factor.QSID]), sorted([RunningKey1, RunningKey2]))
        self.assertListEqual(self.Context.NodeState[Factor.QSID][RunningKey1]["section_ids"], self.SectionIDs[:3])
        self.assertListEqual(self.Context.NodeState[Factor.QSID][RunningKey2]["section_ids"], self.SectionIDs[-3:])

    def testCache(self):
        """测试在多截面时运行时缓存的状态"""
        FT = self.FDB.getTable(self.TestTable)
        Factor = FT.getFactor(self.TestFactor1)
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, Factor]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[:3]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        RunningKey1 = makeFactorRunningKey(qsid=Factor.QSID, section_ids=self.SectionIDs[:3], context=self.Context)
        RunningKey2 = makeFactorRunningKey(qsid=Factor.QSID, section_ids=self.SectionIDs[-3:], context=self.Context)
        self.assertListEqual(sorted(self.Cache._CachedDTRange), sorted([RunningKey1, RunningKey2]))

    def testDataFactor(self):
        """测试 DataFactor 在多截面时运行"""
        np.random.seed(0)
        TestData = pd.DataFrame(np.random.randn(len(self.DTRuler), len(self.SectionIDs)), index=self.DTRuler, columns=self.SectionIDs)
        Factor = DataFactor(data=TestData, args={"Name": "TestFactor"})
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, Factor]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[:3]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        TestData1 = TestData.reindex(columns=self.SectionIDs[:3]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData1, Rslt[0])
        TestData2 = TestData.reindex(columns=self.SectionIDs[-3:]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData2, Rslt[1])

    def testFactorTableFactor(self):
        """测试因子表因子在多截面时运行"""
        FT = self.FDB.getTable(self.TestTable)
        Factor = FT.getFactor(self.TestFactor1)
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, Factor]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[:3]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        TestData1 = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[:3]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData1, Rslt[0])
        TestData2 = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[-3:]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData2, Rslt[1])

    def testMultiSectionSource(self):
        """测试不同截面来源下的多截面时运行"""
        FT = self.FDB.getTable(self.TestTable)
        TestFactor1 = FT.getFactor(self.TestFactor1, args={"SectionIDs": self.SectionIDs[1:4]})
        Factor = fo.Aggregate(func=np.nansum, descriptor_ids=self.SectionIDs[:3])(TestFactor1)
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, TestFactor1, TestFactor1]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=["000000.HST"], SectionIDs=["000000.HST"]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[:3]).sum(axis=1).reindex(index=DTs)
        pd.testing.assert_series_equal(TestData, Rslt[0].iloc[:, 0], check_names=False)
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[-3:]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData, Rslt[1])
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[1:4]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData, Rslt[2])

    def testPointOperation(self):
        """测试单点运算因子在多截面时运行"""
        FT = self.FDB.getTable(self.TestTable)
        TestFactor1 = FT.getFactor(self.TestFactor1)
        TestFactor2 = FT.getFactor(self.TestFactor2)
        Factor = TestFactor1 + TestFactor2
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, Factor]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[:3]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        TestData1 = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[:3]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        TestData2 = self.TestData[self.TestFactor2].reindex(columns=self.SectionIDs[:3]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData1 + TestData2, Rslt[0])
        TestData1 = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[-3:]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        TestData2 = self.TestData[self.TestFactor2].reindex(columns=self.SectionIDs[-3:]).reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData1 + TestData2, Rslt[1])

    def testTimeOperation(self):
        """测试时序运算因子在多截面时运行"""
        FT = self.FDB.getTable(self.TestTable)
        TestFactor1 = FT.getFactor(self.TestFactor1)
        Factor = fo.RollingApply(func=np.nansum, window=2, min_periods=2)(TestFactor1)
        DTs = self.DTRuler[-5:]
        FactorList = [Factor, Factor]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[:3]),
            FactorLocalContext(DTs=DTs, IDs=self.SectionIDs[1:4], SectionIDs=self.SectionIDs[-3:])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[:3]).rolling(window=2, min_periods=2).sum().reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData, Rslt[0])
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[-3:]).rolling(window=2, min_periods=2).sum().reindex(index=DTs, columns=self.SectionIDs[1:4])
        pd.testing.assert_frame_equal(TestData, Rslt[1])

    def testSectionOperation(self):
        """测试截面运算因子在多截面时运行"""
        FT = self.FDB.getTable(self.TestTable)
        TestFactor1 = fo.Aggregate(func=np.nansum, descriptor_ids=self.SectionIDs[:3])(FT.getFactor(self.TestFactor1))
        TestFactor2 = fo.Aggregate(func=np.nansum, descriptor_ids=self.SectionIDs[-3:])(FT.getFactor(self.TestFactor1))
        DTs = self.DTRuler[-5:]
        FactorList = [TestFactor1, TestFactor2]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=["000001.HST"], SectionIDs=["000001.HST"]),
            FactorLocalContext(DTs=DTs, IDs=["000002.HST"], SectionIDs=["000002.HST"])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[:3]).sum(axis=1).reindex(index=DTs)
        pd.testing.assert_series_equal(TestData, Rslt[0].iloc[:, 0], check_names=False)
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[-3:]).sum(axis=1).reindex(index=DTs)
        pd.testing.assert_series_equal(TestData, Rslt[1].iloc[:, 0], check_names=False)

    def testPanelOperation(self):
        """测试面板运算因子在多截面时运行"""
        FT = self.FDB.getTable(self.TestTable)
        TestFactor1 = fo.AggregatePanel(func=np.nansum, window=2, descriptor_ids=self.SectionIDs[:3])(FT.getFactor(self.TestFactor1))
        TestFactor2 = fo.AggregatePanel(func=np.nansum, window=2, descriptor_ids=self.SectionIDs[-3:])(FT.getFactor(self.TestFactor1))
        DTs = self.DTRuler[-5:]
        FactorList = [TestFactor1, TestFactor2]
        FwdDataList = [
            FactorLocalContext(DTs=DTs, IDs=["000001.HST"], SectionIDs=["000001.HST"]),
            FactorLocalContext(DTs=DTs, IDs=["000002.HST"], SectionIDs=["000002.HST"])
        ]
        with self.Cache:
            with self.Context:
                with self.Engine:
                    Rslt = self.Engine.run(
                        node_list=FactorList, 
                        context=self.Context,
                        fwd_data_list=FwdDataList
                    )
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[:3]).sum(axis=1).rolling(window=2).sum().reindex(index=DTs)
        pd.testing.assert_series_equal(TestData, Rslt[0].iloc[:, 0], check_names=False)
        TestData = self.TestData[self.TestFactor1].reindex(columns=self.SectionIDs[-3:]).sum(axis=1).rolling(window=2).sum().reindex(index=DTs)
        pd.testing.assert_series_equal(TestData, Rslt[1].iloc[:, 0], check_names=False)

if __name__ == "__main__":
    unittest.main()
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestMultiSection("testMultiSectionSource"))
    # Runner = unittest.TextTestRunner()
    # Runner.run(Suite)