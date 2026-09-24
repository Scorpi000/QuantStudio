# -*- coding: utf-8 -*-
"""QuantStudio BaoStockDB 因子库测试模块

对 BaoStockDB 中每张因子表进行 readData 验证，确保 API 调用和数据解析正常。

使用方法:
    * 运行全部测试: python tests/test_Factor_BaoStockDB.py
    * 运行单个测试: python -m unittest tests.test_Factor_BaoStockDB.TestBaoStockDB.test_季度盈利能力
"""
import datetime as dt
import unittest

import numpy as np

from QuantStudio.Factor.BaoStockDB import BaoStockDB


class TestBaoStockDB(unittest.TestCase):
    """BaoStockDB 因子库测试"""

    BSDB = None

    @classmethod
    def setUpClass(cls):
        cls.BSDB = BaoStockDB().connect()

    @classmethod
    def tearDownClass(cls):
        if cls.BSDB is not None:
            cls.BSDB.disconnect()

    def _readTable(self, table_name, factor_names, ids=None, dts=None):
        """通用辅助：读取指定表的数据并做基本断言

        Args:
            table_name: 表名
            factor_names: 因子名列表
            ids: 证券代码列表，默认 ["sh.600000"]
            dts: 时点列表，默认 [2024-12-31]

        Returns:
            Panel 对象
        """
        if ids is None:
            ids = ["sh.600000"]
        if dts is None:
            dts = [dt.datetime(2024, 12, 31)]
        FT = self.BSDB.getTable(table_name)
        self.assertIsNotNone(FT, f"表 {table_name} 获取失败")
        Data = FT.readData(factor_names=factor_names, ids=ids, dts=dts)
        self.assertIsNotNone(Data, f"表 {table_name} readData 返回 None")
        self.assertEqual(len(Data.shape), 3, f"表 {table_name} 返回数据维度不为 3")
        return Data

    # ==================== DTRangeTable ====================

    def test_A股K线数据(self):
        Data = self._readTable("A股K线数据", ["close", "volume"])
        self.assertGreaterEqual(Data.shape[1], 1)

    def test_季度业绩快报(self):
        Data = self._readTable("季度业绩快报",
                               ["performanceExpressTotalAsset", "performanceExpressROEWa"],
                               dts=[dt.datetime(2024, 6, 30)])
        self.assertEqual(Data.shape[0], 2)

    def test_季度业绩预告(self):
        Data = self._readTable("季度业绩预告",
                               ["profitForcastType", "profitForcastChgPctUp"],
                               dts=[dt.datetime(2024, 6, 30)])
        self.assertEqual(Data.shape[0], 2)

    # ==================== DTTable ====================

    def test_行业分类(self):
        Data = self._readTable("行业分类", ["industry", "industryClassification"])
        self.assertEqual(Data.shape[0], 2)

    def test_每日A股K线(self):
        Data = self._readTable("每日A股K线", ["close", "volume"],
                               ids=["sh.600000", "sz.000001"],
                               dts=[dt.datetime(2024, 6, 28)])
        self.assertGreaterEqual(Data.shape[2], 1)

    def test_每日ETFK线(self):
        Data = self._readTable("每日ETF K线", ["close", "volume"],
                               ids=["sh.510050"],
                               dts=[dt.datetime(2024, 6, 28)])
        self.assertGreaterEqual(Data.shape[2], 1)

    def test_每日复权因子(self):
        Data = self._readTable("每日复权因子",
                               ["foreAdjustFactor", "backAdjustFactor"],
                               dts=[dt.datetime(2024, 6, 28)])
        self.assertEqual(Data.shape[0], 2)

    def test_某日全部证券(self):
        Data = self._readTable("某日全部证券", ["code_name", "tradeStatus"],
                               dts=[dt.datetime(2024, 6, 28)])
        self.assertEqual(Data.shape[0], 2)

    def test_上证50成分股(self):
        Data = self._readTable("上证50成分股", ["code_name"],
                               dts=[dt.datetime(2024, 6, 28)])
        self.assertEqual(Data.shape[0], 1)

    def test_沪深300成分股(self):
        Data = self._readTable("沪深300成分股", ["code_name"],
                               dts=[dt.datetime(2024, 6, 28)])
        self.assertEqual(Data.shape[0], 1)

    def test_中证500成分股(self):
        Data = self._readTable("中证500成分股", ["code_name"],
                               dts=[dt.datetime(2024, 6, 28)])
        self.assertEqual(Data.shape[0], 1)

    # ==================== QuarterTable ====================

    def test_季度盈利能力(self):
        Data = self._readTable("季度盈利能力",
                               ["roeAvg", "npMargin", "epsTTM"])
        self.assertEqual(Data.shape[0], 3)

    def test_季度营运能力(self):
        Data = self._readTable("季度营运能力",
                               ["NRTurnRatio", "INVTurnRatio", "AssetTurnRatio"])
        self.assertEqual(Data.shape[0], 3)

    def test_季度成长能力(self):
        Data = self._readTable("季度成长能力",
                               ["YOYEquity", "YOYNI", "YOYPNI"])
        self.assertEqual(Data.shape[0], 3)

    def test_季度偿债能力(self):
        Data = self._readTable("季度偿债能力",
                               ["currentRatio", "quickRatio", "liabilityToAsset"])
        self.assertEqual(Data.shape[0], 3)

    def test_季度现金流量(self):
        Data = self._readTable("季度现金流量",
                               ["CAToAsset", "CFOToOR", "CFOToNP"])
        self.assertEqual(Data.shape[0], 3)

    def test_季度杜邦指数(self):
        Data = self._readTable("季度杜邦指数",
                               ["dupontROE", "dupontAssetTurn", "dupontNitogr"])
        self.assertEqual(Data.shape[0], 3)

    # ==================== YearTable ====================

    def test_除权除息信息(self):
        Data = self._readTable("除权除息信息",
                               ["dividCashPsBeforeTax", "dividStocksPs"])
        self.assertEqual(Data.shape[0], 2)

    # ==================== StockBasicTable ====================

    def test_证券基本资料(self):
        Data = self._readTable("证券基本资料",
                               ["code_name", "ipoDate", "status"])
        self.assertEqual(Data.shape[0], 3)

    # ==================== MacroDataTable ====================

    def test_存款利率(self):
        Data = self._readTable("存款利率",
                               ["demandDepositRate", "fixedDepositRate1Year"],
                               ids=["ALL"],
                               dts=[dt.datetime(2015, 3, 1), dt.datetime(2015, 10, 24)])
        self.assertEqual(Data.shape[0], 2)

    def test_贷款利率(self):
        Data = self._readTable("贷款利率",
                               ["loanRate6Month", "loanRate1YearTo3Year"],
                               ids=["ALL"],
                               dts=[dt.datetime(2015, 3, 1), dt.datetime(2015, 10, 24)])
        self.assertEqual(Data.shape[0], 2)

    def test_存款准备金率(self):
        Data = self._readTable("存款准备金率",
                               ["bigInstitutionsRatioAfter", "mediumInstitutionsRatioAfter"],
                               ids=["ALL"],
                               dts=[dt.datetime(2015, 3, 1), dt.datetime(2015, 10, 24)])
        self.assertEqual(Data.shape[0], 2)

    def test_货币供应量(self):
        Data = self._readTable("货币供应量",
                               ["m0Month", "m1Month", "m2Month"],
                               ids=["ALL"],
                               dts=[dt.datetime(2010, 1, 1), dt.datetime(2010, 3, 1)])
        self.assertEqual(Data.shape[0], 3)

    def test_货币供应量年底余额(self):
        Data = self._readTable("货币供应量(年底余额)",
                               ["m0Year", "m1Year", "m2Year"],
                               ids=["ALL"],
                               dts=[dt.datetime(2010, 1, 1), dt.datetime(2012, 1, 1)])
        self.assertEqual(Data.shape[0], 3)


if __name__ == "__main__":
    # unittest.main()
    Suite = unittest.TestSuite()
    Suite.addTest(TestBaoStockDB("test_货币供应量年底余额"))
    Runner = unittest.TextTestRunner()
    Runner.run(Suite)
