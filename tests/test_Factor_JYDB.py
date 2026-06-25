# -*- coding: utf-8 -*-
import datetime as dt

from QuantStudio.Factor.JYDB import JYDB

# FDB = JYDB(args={"Connector": "pyodbc", "DBType": "PostgreSQL"}).connect()
FDB = JYDB(args={}).connect()

# 测试时点方法
# DTs = FDB.getTradeDay(start_date=dt.datetime(2025, 1, 1), end_date=dt.datetime(2025, 12, 31))
# print(len(DTs), DTs[:10])

# 测试 ID 方法
# IDs = FDB.getMutualFundID(type="ETF")
# print(len(IDs), IDs[:10])

# IDs = FDB.getMutualFundID(type="指数基金")
# print(len(IDs), IDs[:10])

# IDs = FDB.getIndexID(type="申万一级行业指数")
# print(len(IDs), IDs[:10])

# IDs = FDB.getOptionID(option_code="510050", contract_code=True)
IDs = FDB.getOptionID(option_code="CU", contract_code=False, date=dt.datetime(2025, 11, 3))
print(len(IDs), IDs[:10])

# # 测试 FeatureTable
# IDs = ["000001.SZ", "000003.SZ", "603297.SH"]
# DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(5)]
# FT = FDB.getTable("A股证券主表")
# Data = FT.readData(factor_names=["证券简称"], ids=IDs, dts=DTs)
# print(Data.iloc[0])

print("===")