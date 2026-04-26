# coding=utf-8
"""测试因子算子"""
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor
import QuantStudio.Factor.FactorOperator as fo


np.random.seed(0)
nDT, nID = 100, 10
SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
IDs = SectionIDs[:3]
DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
DTs = DTRuler[-5:]

InitData = DataFactor(data=1, args={"Name": "init"})
F1 = np.random.rand(nDT, nID)
F1[-8:-5, 0] = np.nan
F1[:, 2] = np.nan
F1 = DataFactor(data=pd.DataFrame(F1, index=DTRuler, columns=SectionIDs), args={"Name": "F1"})
Open = DataFactor(data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTRuler, columns=SectionIDs), args={"Name": "open"})
Close = DataFactor(data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTRuler, columns=SectionIDs), args={"Name": "close"})

rollingSum = fo.RollingApply(func=np.nanmean, window=5, min_periods=3)
TestF = rollingSum(F1, factor_args={"Name": "TestF"})
TestData = TestF.readData(ids=IDs, dts=DTs, dt_ruler=DTRuler)
print(TestData)

print("===")