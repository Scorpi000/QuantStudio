# -*- coding: utf-8 -*-
import datetime as dt

import pyodbc
print([d for d in pyodbc.drivers() if 'post' in d.lower()])

from QuantStudio.Factor.JYDB import JYDB

SDB = JYDB(args={"Connector": "pyodbc", "DBType": "PostgreSQL"}).connect()

IDs = ["000001.SZ", "000003.SZ", "603297.SH"]
DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(5)]

FT = SDB.getTable("A股证券主表")
Data = FT.readData(factor_names=["证券简称"], ids=IDs, dts=DTs)
print(Data.iloc[0])

print("===")