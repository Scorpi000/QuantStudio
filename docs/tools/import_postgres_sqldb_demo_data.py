# -*- coding: utf-8 -*-
"""生成 Demo SQLDB 因子数据"""
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.SQLDB import SQLDB


# 导入 SQLDB 数据
FDB = SQLDB().connect()

np.random.seed(0)
nDT, nID = 100, 20
IDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]

# stock_cn_day_bar
Data = {
    "open" : pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTs, columns=IDs),
    "close" : pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTs, columns=IDs),
    "volume" : pd.DataFrame(np.random.rand(nDT, nID) * 100, index=DTs, columns=IDs),
    "amount" : pd.DataFrame(np.random.rand(nDT, nID) * 1000, index=DTs, columns=IDs),
}
Data["high"] = pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTs, columns=IDs).combine(Data["open"], np.maximum).combine(Data["close"], np.maximum)
Data["low"] = pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTs, columns=IDs).combine(Data["open"], np.minimum).combine(Data["close"], np.minimum)
Data = Panel(Data)
FDB.writeData(data=Data, table_name="stock_cn_day_bar", if_exists="update")

# stock_cn_status
Data = {
    "if_listed" : pd.DataFrame(np.ones((nDT, nID)), index=DTs, columns=IDs)
}
Data["if_listed"].iloc[:, 2] = 0
Data = Panel(Data)
FDB.writeData(data=Data, table_name="stock_cn_status", if_exists="update")

# stock_cn_industry
Data = {
    "industry" : pd.DataFrame(np.repeat(np.random.choice(["Fin", "TMT", "Ind"], size=(1, nID)), axis=0, repeats=nDT), index=DTs, columns=IDs, dtype=pd.StringDtype(storage="python")),
}
Data = Panel(Data)
FDB.writeData(data=Data, table_name="stock_cn_industry", if_exists="update")

# stock_cn_factor_value
Data = {
    "ep_ttm" : pd.DataFrame(np.random.rand(nDT, nID), index=DTs, columns=IDs),
    "bp_lr" : pd.DataFrame(np.random.rand(nDT, nID), index=DTs, columns=IDs),
}
Data = Panel(Data)
FDB.writeData(data=Data, table_name="stock_cn_factor_value", if_exists="update")

# index_cn_day_bar
nDT, nIndexID = 100, 3
IndexIDs = ["000300.SH", "000905.SH", "000852.SH"]
Data = {
    "open" : pd.DataFrame(np.random.rand(nDT, nIndexID) * 10, index=DTs, columns=IndexIDs),
    "close" : pd.DataFrame(np.random.rand(nDT, nIndexID) * 10, index=DTs, columns=IndexIDs),
    "volume" : pd.DataFrame(np.random.rand(nDT, nIndexID) * 100, index=DTs, columns=IndexIDs),
    "amount" : pd.DataFrame(np.random.rand(nDT, nIndexID) * 1000, index=DTs, columns=IndexIDs),
}
Data["high"] = pd.DataFrame(np.random.rand(nDT, nIndexID) * 10, index=DTs, columns=IndexIDs).combine(Data["open"], np.maximum).combine(Data["close"], np.maximum)
Data["low"] = pd.DataFrame(np.random.rand(nDT, nIndexID) * 10, index=DTs, columns=IndexIDs).combine(Data["open"], np.minimum).combine(Data["close"], np.minimum)
Data = Panel(Data)
FDB.writeData(data=Data, table_name="index_cn_day_bar", if_exists="update")

# ==================== NarrowTable 窄表示例 ====================
# stock_cn_factor_value_narrow: (datetime, code, factor_name, factor_value)
# 窄表将因子名和因子值分别存储在两个字段中
# 窄表的主键是 (datetime, code, factor_name)，不能使用 createTable（默认主键是 datetime, code）
DemoTable = "stock_cn_factor_value_narrow"
FDB.deleteTable(DemoTable)  # 清理旧表
DBTableName = FDB._QSArgs.TablePrefix + FDB._QSArgs.InnerPrefix + DemoTable
Cursor = FDB.Connection.cursor()
SQLStr = f"""
CREATE TABLE IF NOT EXISTS {DBTableName} (
    datetime TIMESTAMP NOT NULL,
    code VARCHAR(40) NOT NULL,
    factor_name VARCHAR(40) NOT NULL,
    factor_value DOUBLE PRECISION,
    PRIMARY KEY (datetime, code, factor_name)
)
"""
Cursor.execute(SQLStr)
FDB.Connection.commit()
Cursor.close()
# 重新 connect 以刷新表信息
FDB.connect()

# 窄表数据每行一条 (时点, 代码, 因子名, 因子值)，(datetime, code) 存在重复行
# 由于 writeData 要求 (DT, ID) 唯一，这里用原始 SQL 插入
NarrowRecords = []
for iDT in DTs:
    for iID in IDs:
        for iFactor, iVal in [("pe", np.random.randn()), ("pb", np.random.randn()), ("roe", np.random.randn() / 100)]:
            NarrowRecords.append((str(iDT), iID, iFactor, float(iVal)))
DBTableName = FDB._QSArgs.TablePrefix + FDB._QSArgs.InnerPrefix + DemoTable
Cursor = FDB.Connection.cursor()
Cursor.executemany(f"INSERT INTO {DBTableName} (datetime, code, factor_name, factor_value) VALUES (%s, %s, %s, %s)", NarrowRecords)
FDB.Connection.commit()
Cursor.close()
# 重新 connect 以刷新表信息
FDB.connect()
print(f"写入 NarrowTable 示例表: {DemoTable}")

# ==================== FeatureTable 特征表示例 ====================
# stock_cn_static_info: 无时点字段，只有 ID 维度的静态属性
DemoTable = "stock_cn_static_info"
FDB.deleteTable(DemoTable)
FDB.createTable(DemoTable, field_types={
    "full_name": "VARCHAR(100)",
    "listed_date": "DATE"
})

StaticIDs = IDs
FullNames = [f"股票{i+1}" for i in range(len(StaticIDs))]
ListedDates = [dt.date(2020, 1, 1) + dt.timedelta(days=i * 30) for i in range(len(StaticIDs))]
# 特征表无时点维度，用一个固定时点写入
StaticDT = dt.datetime(2025, 1, 1)
Data = Panel({
    "full_name": pd.DataFrame([FullNames], index=[StaticDT], columns=StaticIDs, dtype=pd.StringDtype(storage="python")),
    "listed_date": pd.DataFrame([ListedDates], index=[StaticDT], columns=StaticIDs)
})
FDB.writeData(data=Data, table_name=DemoTable, if_exists="update")
print(f"写入 FeatureTable 示例表: {DemoTable}")

# ==================== TimeSeriesTable 时序表示例 ====================
# macro_indicator: 无 ID 字段，只有时点维度的宏观指标
DemoTable = "macro_indicator"
FDB.deleteTable(DemoTable)
# 时序表没有 code 字段，需要手动建表
Cursor = FDB.Connection.cursor()
# 手动建表（不含 code 列）
DBTableName = FDB._QSArgs.TablePrefix + FDB._QSArgs.InnerPrefix + DemoTable
SQLStr = f"""
CREATE TABLE IF NOT EXISTS {DBTableName} (
    datetime TIMESTAMP NOT NULL,
    interest_rate DOUBLE PRECISION,
    cpi DOUBLE PRECISION,
    gdp DOUBLE PRECISION,
    PRIMARY KEY (datetime)
)
"""
Cursor.execute(SQLStr)
FDB.Connection.commit()
Cursor.close()
# 重新 connect 以刷新表信息
FDB.connect()

MacroDTs = [dt.datetime(2020, 1, 1) + dt.timedelta(days=i * 90) for i in range(20)]
# 时序表无 ID 列，用原始 SQL 插入
MacroRecords = [(str(iDT), float(np.random.rand() * 5 + 2), float(np.random.rand() * 3 + 1), float(np.random.rand() * 10)) for iDT in MacroDTs]
DBTableName = FDB._QSArgs.TablePrefix + FDB._QSArgs.InnerPrefix + DemoTable
Cursor = FDB.Connection.cursor()
Cursor.executemany(f"INSERT INTO {DBTableName} (datetime, interest_rate, cpi, gdp) VALUES (%s, %s, %s, %s)", MacroRecords)
FDB.Connection.commit()
Cursor.close()
FDB.connect()
print(f"写入 TimeSeriesTable 示例表: {DemoTable}")

# ==================== MappingTable 映射表示例 ====================
# stock_index_mapping: 证券与指数的映射关系，含生效和失效日期
DemoTable = "stock_index_mapping"
FDB.deleteTable(DemoTable)
Cursor = FDB.Connection.cursor()
DBTableName = FDB._QSArgs.TablePrefix + FDB._QSArgs.InnerPrefix + DemoTable
SQLStr = f"""
CREATE TABLE IF NOT EXISTS {DBTableName} (
    datetime TIMESTAMP NOT NULL,
    stock_code VARCHAR(40) NOT NULL,
    index_code VARCHAR(40) NOT NULL,
    start_date DATE,
    end_date DATE,
    PRIMARY KEY (datetime, stock_code, index_code)
)
"""
Cursor.execute(SQLStr)
FDB.Connection.commit()
Cursor.close()
FDB.connect()

MappingIDs = IDs[:10]
MappingDTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(10)]
# 映射表字段名不同（stock_code 而非 code），用原始 SQL 插入
MappingRecords = []
for iDT in MappingDTs:
    for iID in MappingIDs:
        idx = np.random.choice(["000300.SH", "000905.SH", "000852.SH"])
        start = (iDT - dt.timedelta(days=np.random.randint(30, 365))).date()
        end = (iDT + dt.timedelta(days=np.random.randint(30, 365))).date()
        MappingRecords.append((str(iDT), iID, idx, start, end))
DBTableName = FDB._QSArgs.TablePrefix + FDB._QSArgs.InnerPrefix + DemoTable
Cursor = FDB.Connection.cursor()
Cursor.executemany(f"INSERT INTO {DBTableName} (datetime, stock_code, index_code, start_date, end_date) VALUES (%s, %s, %s, %s, %s)", MappingRecords)
FDB.Connection.commit()
Cursor.close()
FDB.connect()
print(f"写入 MappingTable 示例表: {DemoTable}")
