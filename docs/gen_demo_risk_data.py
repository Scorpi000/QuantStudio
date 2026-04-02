# -*- coding: utf-8 -*-
"""生成 Demo 风险数据"""
import os
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Core.QSObject import Panel
from QuantStudio.Risk.HDF5RDB import HDF5FRDB


TargetDir = "./data/Risk"
if not os.path.isdir(TargetDir): os.makedirs(TargetDir, exist_ok=True)
CacheDir = "./data/Cache"
if not os.path.isdir(CacheDir): os.makedirs(CacheDir, exist_ok=True)

HFRDB = HDF5FRDB(args={"MainDir": TargetDir}).connect()

np.random.seed(0)
nDT, nID = 100, 20
IDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
FactorNames = ["Size", "Beta", "Momentum", "ResidualVolatility", "NonlinearSize"]
nFactor = len(FactorNames)


# FactorData
FactorData = Panel(np.random.randn(nFactor, nDT, nID), items=FactorNames, major_axis=DTs, minor_axis=IDs)

# FactorCov
FactorCov = Panel(np.array([np.cov(np.random.randn(100, nFactor), rowvar=False) for _ in range(nDT)]), items=DTs, major_axis=FactorNames, minor_axis=FactorNames)

# SpecificRisk
SpecificRisk = pd.DataFrame(np.random.rand(nDT, nID), index=DTs, columns=IDs)

# FactorRet
FactorRet = pd.DataFrame(np.random.randn(nDT, nFactor), index=DTs, columns=FactorNames)

# SpecificRet
SpecificRet = pd.DataFrame(np.random.randn(nDT, nID), index=DTs, columns=IDs)

for iDT in DTs:
    HFRDB.writeData(
        table_name="demo_risk_table", idt=iDT, 
        factor_data=FactorData.loc[:, iDT], 
        factor_cov=FactorCov.loc[iDT], 
        specific_risk=SpecificRisk.loc[iDT], 
        factor_ret=FactorRet.loc[iDT],
        specific_ret=SpecificRet.loc[iDT]
    )
HFRDB.setTableMetaData(table_name="demo_risk_table", meta_data={"Description": "这是一张示例风险表"})

# 删除多余的表
for iTableName in HFRDB.TableNames:
    if iTableName not in ['demo_risk_table']:
        HFRDB.deleteTable(iTableName)