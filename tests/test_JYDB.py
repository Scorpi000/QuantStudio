# -*- coding: utf-8 -*-
import os
import sys
import zipfile
import tempfile
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.JYDB import JYDB

__TestDirPath__ = os.path.split(os.path.realpath(__file__))[0]

def AnalystEstDetailTable4StockFun(f, idt, iid, x, args):
    return np.nanmean(x[0]["每股收益"].values.astype("float"))
def AnalystRatingDetailTable4StockFun(f, idt, iid, x, args):
    return np.nanmean(x["本期评级"].values.astype("float"))


class TestJYDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TestJYDB.TargetDataFile = __TestDirPath__+os.sep+"JYTestData.zip"
        with zipfile.ZipFile(TestJYDB.TargetDataFile, mode="a") as ZIPFile:
            TestJYDB.DataFileList = ZIPFile.namelist()
        TestJYDB.FDB = JYDB()
        #TestJYDB.FDB = TushareDB()
        TestJYDB.StartDT, TestJYDB.EndDT = dt.datetime(2018, 9, 5), dt.datetime(2018, 10, 31)
        TestJYDB.DTs = []
        TestJYDB.StockIDs = ["000001.SZ", "000003.SZ", "603297.SH"]
        TestJYDB.IndexIDs = ["000001.SH", "000300.SH", "000905.SH"]
        TestJYDB.MutualFundIDs = ["000001.MF", "510050.SH", "161130.SZ"]
        TestJYDB.CBondIDs = ["110030.SH", "126002.SZ", "SF5580"]
        TestJYDB.MacroIDs = ["110000001", "110002855", "601280005"]
        TestJYDB.GoldIDs = ["Au99.99.SGX", "Au99.95.SGX", "Au(T+N2).SGX"]
        TestJYDB.OptionIDs = ["510050C1811M02850", "cu1909C45000", "m1811-C-2650"]
    def _saveTargetData(self, fun_name, data):
        if fun_name+".csv" not in self.DataFileList:
            iCSVFile = __TestDirPath__+os.sep+fun_name+".csv"
            data.to_csv(iCSVFile, encoding="utf-8")
            with zipfile.ZipFile(self.TargetDataFile, mode='a') as ZIPFile:
                ZIPFile.write(iCSVFile, arcname=iCSVFile)
            os.remove(iCSVFile)
    def _readTargetData(self, fun_name, **kwargs):
        iCSVFile = fun_name+".csv"
        with zipfile.ZipFile(self.TargetDataFile, mode='r') as ZIPFile:
            ZIPFile.extract(iCSVFile, __TestDirPath__)
        Data = pd.read_csv(__TestDirPath__+os.sep+iCSVFile, index_col=0, header=0, encoding="utf-8", engine="python", **kwargs)
        os.remove(__TestDirPath__+os.sep+iCSVFile)
        return Data
    def _compareDataFrame(self, df1, df2, dtype="double"):
        m1, m2 = pd.isnull(df1), pd.isnull(df2)
        if dtype=="double":
            Err = (df1 - df2).abs()
            Err[pd.isnull(Err)] = 0
            return np.maximum(Err, (m1 ^ m2).astype("float"))
        else:
            Err = (df1!=df2)
            Err[m1 | m2] = False
            return np.maximum(Err.astype("float"), (m1 ^ m2).astype("float"))
    # 测试连接功能
    def test_001_connect(self):
        self.FDB.connect()
    # 测试交易日读取功能
    def test_002_TradeDay(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        self.DTs.extend(self.FDB.getTradeDay(start_date=self.StartDT.date(), end_date=self.EndDT.date(), exchange="SSE", output_type="datetime"))
        self._saveTargetData(FunName, pd.DataFrame(self.DTs, columns=["TradeDay"]))
        TargetData = self._readTargetData(FunName)
        TargetData = [dt.datetime.strptime(iDT, "%Y-%m-%d") for iDT in TargetData.iloc[:, 0]]
        self.assertListEqual(self.DTs, TargetData)
    # 测试 ID 读取功能
    def test_003_StockID(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        self.AllStockIDs = self.FDB.getStockID(index_id="全体A股", date=self.EndDT.date(), is_current=False)
        self._saveTargetData(FunName, pd.DataFrame(self.AllStockIDs, columns=["AllStockIDs"]))
        TargetData = self._readTargetData(FunName).iloc[:, 0].tolist()
        self.assertListEqual(self.AllStockIDs, TargetData)
    def test_003_OptionID(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        self.AllOptionIDs = self.FDB.getOptionID(option_code="510050", date=self.EndDT.date(), is_current=False)
        self._saveTargetData(FunName, pd.DataFrame(self.AllOptionIDs, columns=["AllOptionIDs"]))
        TargetData = self._readTargetData(FunName).iloc[:, 0].tolist()
        self.assertListEqual(self.AllOptionIDs, TargetData)
    def test_003_MutualFundID(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        self.AllMutualFundIDs = self.FDB.getMutualFundID(date=self.EndDT.date(), is_current=False)
        self._saveTargetData(FunName, pd.DataFrame(self.AllMutualFundIDs, columns=["AllMutualFundIDs"]))
        TargetData = self._readTargetData(FunName).iloc[:, 0].tolist()
        self.assertListEqual(self.AllMutualFundIDs, TargetData)
    # 测试特征因子表
    def test_004_FeatureTable4Macro(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("宏观指标主表")
        TestData = FT.readData(factor_names=["数据披露频率", "量纲系数"], ids=self.MacroIDs, dts=[self.EndDT]).iloc[:, 0, :]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName)
        TargetData.index = [str(iID) for iID in TargetData.index]
        Err = self._compareDataFrame(TestData, TargetData, dtype="string")
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_004_FeatureTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("A股证券主表")
        TestData = FT.readData(factor_names=["证券市场", "上市板块"], ids=self.StockIDs, dts=[self.EndDT]).iloc[:, 0, :]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName)
        Err = self._compareDataFrame(TestData, TargetData, dtype="string")
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_004_FeatureTable4Index(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("指数基本情况")
        TestData = FT.readData(factor_names=["编制机构名称", "基点(点)"], ids=self.IndexIDs, dts=[self.EndDT]).iloc[:, 0, :]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName)
        Err = self._compareDataFrame(TestData, TargetData, dtype="string")
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_004_FeatureTable4MutualFund(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("公募基金概况")
        TestData = FT.readData(factor_names=["场内申购赎回简称", "基金经理"], ids=self.MutualFundIDs, dts=[self.EndDT]).iloc[:, 0, :]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName)
        Err = self._compareDataFrame(TestData, TargetData, dtype="string")
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_004_FeatureTable4CBond(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("可转债基本信息")
        TestData = FT.readData(factor_names=["主承销商", "债券形式"], ids=self.CBondIDs, dts=[self.EndDT]).iloc[:, 0, :]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName)
        Err = self._compareDataFrame(TestData, TargetData, dtype="string")
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_004_FeatureTable4Option(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("期权合约")
        TestData = FT.readData(factor_names=["合约简称", "标的名称"], ids=self.OptionIDs, dts=[self.EndDT]).iloc[:, 0, :]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName)
        Err = self._compareDataFrame(TestData, TargetData, dtype="string")
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试行情因子表
    def test_005_MarketTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("日行情表")
        TestData = FT.readData(factor_names=["收盘价(元)"], ids=self.StockIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_005_MarketTable4Index(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("指数行情")
        TestData = FT.readData(factor_names=["收盘价(元-点)"], ids=self.IndexIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_005_MarketTable4MutualFund(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("公募基金行情历史表现")
        TestData = FT.readData(factor_names=["收盘价(元)"], ids=self.MutualFundIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_005_MarketTable4CBond(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("可转换债券行情")
        TestData = FT.readData(factor_names=["收盘价(元)"], ids=self.CBondIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_005_MarketTable4Option(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("期权每日行情")
        TestData = FT.readData(factor_names=["收盘价"], ids=self.CBondIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    def test_005_MarketTable4Gold(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("上海黄金交易所交易行情")
        TestData = FT.readData(factor_names=["收盘价(元-克)"], ids=self.GoldIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试财务因子表
    def test_006_FinancialTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("利润分配表_新会计准则")
        TestData = FT.readData(factor_names=["归属于母公司所有者的净利润"], ids=self.StockIDs, dts=self.DTs, args={"报告期":"所有", "计算方法":"TTM"}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试财务指标因子表
    def test_007_FinancialIndicatorTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("公司主要财务分析指标_新会计准则")
        TestData = FT.readData(factor_names=["每股营业收入(元-股)"], ids=self.StockIDs, dts=self.DTs, args={"报告期":"所有", "计算方法":"单季度"}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试盈利预测汇总表
    def test_008_AnalystConsensusTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("个股盈利预测")
        TestData = FT.readData(factor_names=["每股收益平均值"], ids=self.StockIDs, dts=self.DTs, args={"回溯天数":0, "统计周期时间间隔":"30", "计算方法":"Fwd12M"}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试分红因子表
    def test_009_DividendTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("公司分红")
        TestData = FT.readData(factor_names=["派现(含税-人民币元)"], ids=self.StockIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试成分因子表
    def test_010_ConstituentTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("指数成份")
        TestData = FT.readData(factor_names=["3145"], ids=self.StockIDs, dts=self.DTs).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试映射因子表
    def test_011_MappingTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("公司行业划分表")
        TestData = FT.readData(factor_names=["一级行业名称"], ids=self.StockIDs, dts=self.DTs, args={"行业划分标准":"3"}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData, dtype="string")
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试信息发布因子表
    def test_012_InfoPublTable4MutualFund(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("公募基金净值")
        TestData = FT.readData(factor_names=["单位净值(元)"], ids=self.MutualFundIDs, dts=self.DTs, args={"回溯天数":np.inf}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试宏观因子表
    def test_013_MacroTable4Macro(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("宏观基础指标数据")
        TestData = FT.readData(factor_names=["指标数据"], ids=self.MacroIDs, dts=self.DTs, args={"回溯天数":365}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试分析师盈利预测明细因子表
    def test_014_AnalystEstDetailTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("研究报告_盈利预测")
        TestData = FT.readData(factor_names=["每股收益"], ids=self.StockIDs, dts=self.DTs, args={"算子":AnalystEstDetailTable4StockFun, "向前年数":[0], "周期":180, "去重字段":["撰写机构代码"]}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试分析师评级明细因子表
    def test_015_AnalystRatingDetailTable4Stock(self):
        FunName = sys._getframe().f_code.co_name.split("_")[-1]
        FT = self.FDB.getTable("研究报告_目标价与评级")
        TestData = FT.readData(factor_names=["本期评级"], ids=self.StockIDs, dts=self.DTs, args={"算子":AnalystRatingDetailTable4StockFun, "周期":180, "去重字段":["撰写机构"]}).iloc[0]
        self._saveTargetData(FunName, TestData)
        TargetData = self._readTargetData(FunName, parse_dates=True)
        Err = self._compareDataFrame(TestData, TargetData)
        self.assertAlmostEqual(Err.max().max(), 0)

if __name__=="__main__":
    unittest.main()
    #Suite = unittest.TestSuite()
    #Suite.addTest(TestJYDB("test_001_connect"))
    #Suite.addTest(TestJYDB("test_002_TradeDay"))
    #Suite.addTest(TestJYDB("test_014_AnalystEstDetailTable4Stock"))
    #Suite.addTest(TestJYDB("test_015_AnalystRatingDetailTable4Stock"))
    #Runner = unittest.TextTestRunner()
    #Runner.run(Suite)