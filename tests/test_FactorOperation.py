# -*- coding: utf-8 -*-
import os
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.FactorOperation import PointOperation, TimeOperation, SectionOperation, PanelOperation
from QuantStudio.FactorDataBase.FactorDB import DataFactor, Factorize
from QuantStudio.FactorDataBase import FactorTools as fd

# 单点运算测试算子
def TestPointOperationSingleDTSingleIDFun(f, idt, iid, x, args):
    if x[1]==0: return np.nan
    else: return x[0] / x[1]
def TestPointOperationSingleDTMultiIDFun(f, idt, iid, x, args):
    Rslt = x[0] / x[1]
    Rslt[x[1]==0] = np.nan
    return Rslt
def TestPointOperationMultiDTSingleIDFun(f, idt, iid, x, args):
    Rslt = x[0] / x[1]
    Rslt[x[1]==0] = np.nan
    return Rslt
def TestPointOperationMultiDTMultiIDFun(f, idt, iid, x, args):
    Rslt = x[0] / x[1]
    Rslt[x[1]==0] = np.nan
    return Rslt

# 滚动窗口, 非迭代时间序列运算测试算子
def TestTimeOperationRollingNoniterativeSingleDTSingleIDFun(f, idt, iid, x, args):
    return np.mean(x[0])
def TestTimeOperationRollingNoniterativeSingleDTMultiIDFun(f, idt, iid, x, args):
    return np.mean(x[0], axis=0)
def TestTimeOperationRollingNoniterativeMultiDTSingleIDFun(f, idt, iid, x, args):
    N = args["N"]
    Rslt = np.full((x[0].shape[0]-N+1, ), fill_value=np.nan)
    for i in range(4, x[0].shape[0]): Rslt[i-N+1] = np.mean(x[0][i-N+1:i+1])
    return Rslt
def TestTimeOperationRollingNoniterativeMultiDTMultiIDFun(f, idt, iid, x, args):
    N = args["N"]
    Rslt = np.full((x[0].shape[0]-N+1, x[0].shape[1]), fill_value=np.nan)
    for i in range(4, x[0].shape[0]): Rslt[i-N+1, :] = np.mean(x[0][i-N+1:i+1, :], axis=0)
    return Rslt

# 滚动窗口, 迭代时间序列运算测试算子
def TestTimeOperationRollingIterativeSingleDTSingleIDFun(f, idt, iid, x, args):
    N = args["N"]
    if x[0].shape[0]==0: return x[1][0]
    else: return 2 / (N+1) * x[1][0]  + (1 - 2 / (N+1)) * x[0][0]
def TestTimeOperationRollingIterativeSingleDTMultiIDFun(f, idt, iid, x, args):
    N = args["N"]
    if x[0].shape[0]==0: return x[1][0, :]
    else: return 2 / (N+1) * x[1][0, :] + (1 - 2 / (N+1)) * x[0][0, :]
def TestTimeOperationRollingIterativeMultiDTSingleIDFun(f, idt, iid, x, args):
    N = args["N"]
    Rslt = np.copy(x[1])
    for i in range(1, x[1].shape[0]): Rslt[i] = 2 / (N+1) * x[1][i] + (1 - 2 / (N+1)) * Rslt[i-1]
    return Rslt
def TestTimeOperationRollingIterativeMultiDTMultiIDFun(f, idt, iid, x, args):
    N = args["N"]
    Rslt = np.copy(x[1])
    for i in range(1, x[1].shape[0]): Rslt[i, :] = 2 / (N+1) * x[1][i, :] + (1 - 2 / (N+1)) * Rslt[i-1, :]
    return Rslt

# 单一截面运算测试算子
def TestSectionOperationSingleSectionSingleDTMultiIDFun(f, idt, iid, x, args):
    return (x[0] - np.nanmean(x[0])) / np.nanstd(x[0], ddof=1)
def TestSectionOperationSingleSectionMultiDTMultiIDFun(f, idt, iid, x, args):
    return ((x[0].T - np.nanmean(x[0], axis=1)) / np.nanstd(x[0], axis=1, ddof=1)).T
# 不同截面运算测试算子
def TestSectionOperationMultiSectionSingleDTMultiIDFun(f, idt, iid, x, args):
    return np.nanmean(x[0], keepdims=True) - np.nanmean(x[1], keepdims=True)
def TestSectionOperationMultiSectionMultiDTMultiIDFun(f, idt, iid, x, args):
    return np.nanmean(x[0], axis=1, keepdims=True) - np.nanmean(x[1], axis=1, keepdims=True)


# 面板运算测试算子
def TestPanelOperationMultiSectionSingleDTMultiIDFun(f, idt, iid, x, args):
    N = args["N"]
    if x[0].shape[0]==0: return np.nanmean(x[1][0, :]) - np.nanmean(x[2][0, :])
    else: return 2 / (N+1) * (np.nanmean(x[1][0, :]) - np.nanmean(x[2][0, :])) + (1 - 2 / (N+1)) * x[0][0, :]
def TestPanelOperationMultiSectionMultiDTMultiIDFun(f, idt, iid, x, args):
    N = args["N"]
    Rslt = np.full((x[1].shape[0], 1), fill_value=np.nan)
    Rslt[0, :] = np.nanmean(x[1][0, :]) - np.nanmean(x[2][0, :])
    for i in range(1, x[1].shape[0]):
        Rslt[i, :] = 2 / (N+1) * (np.nanmean(x[1][i, :]) - np.nanmean(x[2][i, :])) + (1 - 2 / (N+1)) * Rslt[i-1, :]
    return Rslt


class TestFactorOperation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TestFactorOperation.FactorNames = ["Factor0", "Factor1"]
        TestFactorOperation.IDs = [("00000%d.SZ" % (i,)) for i in range(10)]
        TestFactorOperation.GroupIDs = [("Group%d" % (i, )) for i in range(3)]
        TestFactorOperation.MacroID = "000000.HST"
        TestFactorOperation.DTs = [dt.datetime(2018,1,1)+dt.timedelta(i) for i in range(30)]
        np.random.seed(0)
        TestFactorOperation.Data0 = pd.DataFrame(np.random.randn(30, 10), index=TestFactorOperation.DTs, columns=TestFactorOperation.IDs)
        TestFactorOperation.Factor0 = DataFactor(name=TestFactorOperation.FactorNames[0], data=TestFactorOperation.Data0)
        TestFactorOperation.Data1 = pd.DataFrame(np.random.randn(30, 10), index=TestFactorOperation.DTs, columns=TestFactorOperation.IDs)
        TestFactorOperation.Factor1 = DataFactor(name=TestFactorOperation.FactorNames[1], data=TestFactorOperation.Data1)
        TestFactorOperation.GroupData = pd.DataFrame(np.random.randn(30, 3), index=TestFactorOperation.DTs, columns=TestFactorOperation.GroupIDs)
        TestFactorOperation.GroupFactor = DataFactor(name="GroupFactor", data=TestFactorOperation.GroupData)        
    # 测试单点运算功能, 测试运算: 除法
    def test_1_PointOperation(self):
        TargetData = self.Data0 / self.Data1
        TargetData[self.Data1==0] = np.nan
        # 单时点, 单 ID 模式
        TestFactor = PointOperation(name="TestFactor", descriptors=[self.Factor0, self.Factor1], 
                                    sys_args={"算子": TestPointOperationSingleDTSingleIDFun, "运算时点":"单时点", "运算ID":"单ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 单时点, 多 ID 模式
        TestFactor = PointOperation(name="TestFactor", descriptors=[self.Factor0, self.Factor1], 
                                    sys_args={"算子": TestPointOperationSingleDTMultiIDFun, "运算时点":"单时点", "运算ID":"多ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 多时点, 单 ID 模式
        TestFactor = PointOperation(name="TestFactor", descriptors=[self.Factor0, self.Factor1], 
                                    sys_args={"算子": TestPointOperationMultiDTSingleIDFun, "运算时点":"多时点", "运算ID":"单ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 多时点, 多 ID 模式
        TestFactor = PointOperation(name="TestFactor", descriptors=[self.Factor0, self.Factor1], 
                                    sys_args={"算子": TestPointOperationMultiDTMultiIDFun, "运算时点":"多时点", "运算ID":"多ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 内置算子模式
        TestFactor = Factorize(self.Factor0 / self.Factor1, factor_name="TestFactor")
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试滚动窗口, 非迭代时间序列运算功能, 测试运算: 简单移动平均
    def test_2_TimeOperationRollingNoniterative(self):
        N = 5# 移动窗口长度
        TargetData = self.Data0.rolling(window=N, min_periods=N).mean()
        # 单时点, 单 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingNoniterativeSingleDTSingleIDFun, "参数":{"N":N}, "回溯期数":[N-1], "运算时点":"单时点", "运算ID":"单ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 单时点, 多 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingNoniterativeSingleDTMultiIDFun, "参数":{"N":N}, "回溯期数":[N-1], "运算时点":"单时点", "运算ID":"多ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 多时点, 单 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingNoniterativeMultiDTSingleIDFun, "参数":{"N":N}, "回溯期数":[N-1], "运算时点":"多时点", "运算ID":"单ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 多时点, 多 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingNoniterativeMultiDTMultiIDFun, "参数":{"N":N}, "回溯期数":[N-1], "运算时点":"多时点", "运算ID":"多ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试滚动窗口, 迭代时间序列运算功能, 测试运算: 指数移动平均
    def test_3_TimeOperationRollingIterative(self):
        N = 5# 移动窗口长度
        TargetData = self.Data0.ewm(span=N, adjust=False).mean()
        # 单时点, 单 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingIterativeSingleDTSingleIDFun, "参数":{"N":N}, "回溯期数":[1-1], "自身回溯期数":1, "运算时点":"单时点", "运算ID":"单ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 单时点, 多 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingIterativeSingleDTMultiIDFun, "参数":{"N":N}, "回溯期数":[1-1], "自身回溯期数":1, "运算时点":"单时点", "运算ID":"多ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 多时点, 单 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingIterativeMultiDTSingleIDFun, "参数":{"N":N}, "回溯期数":[1-1], "自身回溯期数":1, "运算时点":"多时点", "运算ID":"单ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 多时点, 多 ID 模式
        TestFactor = TimeOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestTimeOperationRollingIterativeMultiDTMultiIDFun, "参数":{"N":N}, "回溯期数":[1-1], "自身回溯期数":1, "运算时点":"多时点", "运算ID":"多ID"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试单一截面运算功能, 测试运算: z-score 截面标准化
    def test_4_SectionOperationSingleSection(self):
        TargetData = ((self.Data0.T - self.Data0.mean(axis=1)) / self.Data0.std(axis=1, ddof=1)).T
        # 单时点, 返回值全截面模式
        TestFactor = SectionOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestSectionOperationSingleSectionSingleDTMultiIDFun, "运算时点":"单时点", "输出形式":"全截面"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
        # 多时点, 返回值全截面模式
        TestFactor = SectionOperation(name="TestFactor", descriptors=[self.Factor0], 
                                    sys_args={"算子": TestSectionOperationSingleSectionMultiDTMultiIDFun, "运算时点":"多时点", "输出形式":"全截面"})
        TestData = TestFactor.readData(ids=self.IDs, dts=self.DTs)
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max().max(), 0)
    # 测试不同截面运算功能, 测试运算: 两个截面均值的差
    def test_5_SectionOperationSingleSection(self):
        TargetData = self.Data0.mean(axis=1) - self.GroupData.mean(axis=1)
        # 单时点, 返回值全截面模式
        TestFactor = SectionOperation(name="TestFactor", descriptors=[self.Factor0, self.GroupFactor], 
                                    sys_args={"算子": TestSectionOperationMultiSectionSingleDTMultiIDFun, "描述子截面":[self.IDs, self.GroupIDs], "运算时点":"单时点", "输出形式":"全截面"})
        TestData = TestFactor.readData(ids=[self.MacroID], dts=self.DTs).iloc[:, 0]
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max(), 0)
        # 多时点, 返回值全截面模式
        TestFactor = SectionOperation(name="TestFactor", descriptors=[self.Factor0, self.GroupFactor], 
                                    sys_args={"算子": TestSectionOperationMultiSectionMultiDTMultiIDFun, "描述子截面":[self.IDs, self.GroupIDs], "运算时点":"多时点", "输出形式":"全截面"})
        TestData = TestFactor.readData(ids=[self.MacroID], dts=self.DTs).iloc[:, 0]
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max(), 0)
    # 测试面板运算功能, 测试运算: 两个截面均值的指数移动平均之差
    def test_6_PanelOperation(self):
        N = 5# 移动窗口长度
        TargetData = self.Data0.mean(axis=1).ewm(span=N, adjust=False).mean() - self.GroupData.mean(axis=1).ewm(span=N, adjust=False).mean()
        # 单时点, 返回值全截面模式
        TestFactor = PanelOperation(name="TestFactor", descriptors=[self.Factor0, self.GroupFactor], 
                                    sys_args={"算子": TestPanelOperationMultiSectionSingleDTMultiIDFun, "参数":{"N":N}, 
                                              "回溯期数":[1-1,1-1], "自身回溯期数":1, "描述子截面":[self.IDs, self.GroupIDs], 
                                              "运算时点":"单时点", "输出形式":"全截面"})
        TestData = TestFactor.readData(ids=[self.MacroID], dts=self.DTs).iloc[:, 0]
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max(), 0)
        # 多时点, 返回值全截面模式
        TestFactor = PanelOperation(name="TestFactor", descriptors=[self.Factor0, self.GroupFactor], 
                                    sys_args={"算子": TestPanelOperationMultiSectionMultiDTMultiIDFun, "参数":{"N":N}, 
                                              "回溯期数":[1-1,1-1], "自身回溯期数":1, "描述子截面":[self.IDs, self.GroupIDs], 
                                              "运算时点":"多时点", "输出形式":"全截面"})
        TestData = TestFactor.readData(ids=[self.MacroID], dts=self.DTs).iloc[:, 0]
        Err = (TestData - TargetData).abs()
        self.assertAlmostEqual(Err.max(), 0)

if __name__=="__main__":
    unittest.main()