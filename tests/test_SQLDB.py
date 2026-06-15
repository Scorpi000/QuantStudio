# -*- coding: utf-8 -*-
import datetime as dt

from QuantStudio.Factor.SQLDB import SQLDB

FDB = SQLDB(args={"DBType": "PostgreSQL", "InnerPrefix": "qsd_"}).connect()
print(FDB.TableNames)


# 测试 FactorTable
IDs = ["000001.SZ", "000003.SZ", "603297.SH"]
DTs = [dt.datetime(2026, 6, 3) + dt.timedelta(i) for i in range(5)]
FT = FDB.getTable("stock_cn_day_bar", args={"MultiMapping": False})
Data = FT.readData(factor_names=["close"], ids=IDs, dts=DTs)
print(Data.iloc[0])

print("===")