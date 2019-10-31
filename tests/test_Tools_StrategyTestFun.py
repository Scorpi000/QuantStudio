# -*- coding: utf-8 -*-
import os
import time
import datetime as dt
import unittest

import numpy as np
import pandas as pd

import QuantStudio.Tools.StrategyTestFun as STF

class TestStrategyTestFun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 测试价格序列
        TestStrategyTestFun.p = np.array([0.6, 0.8, 0.9, 0.85, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.67, 0.66, 0.6, 0.8, 1.2, 1.0, 1.4, 1.5, 1.2, 1.0, 0.8, 0.9, 1.0, 0.7, 0.9])
        TestStrategyTestFun.r = np.diff(TestStrategyTestFun.p) / TestStrategyTestFun.p[:-1]
    def test_calcMD(self):
        self.assertAlmostEqual(STF.calcMD(self.p), 0.7 / 1.5 - 1)
    def test_calcMaxDrawdownRate(self):
        self.assertTupleEqual(STF.calcMaxDrawdownRate(self.p), (0.7 / 1.5 - 1, 17, 23))

if __name__=="__main__":
    unittest.main()