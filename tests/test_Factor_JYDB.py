# -*- coding: utf-8 -*-
import datetime as dt

from QuantStudio.Factor.JYDB import JYDB

# FDB = JYDB(args={"Connector": "pyodbc", "DBType": "PostgreSQL"}).connect()
FDB = JYDB(args={}).connect()

# 测试 ID 方法
IDs = FDB.getMutualFundID(type="ETF")
print(IDs)

# # 测试 FeatureTable
# IDs = ["000001.SZ", "000003.SZ", "603297.SH"]
# DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(5)]
# FT = FDB.getTable("A股证券主表")
# Data = FT.readData(factor_names=["证券简称"], ids=IDs, dts=DTs)
# print(Data.iloc[0])

print("===")