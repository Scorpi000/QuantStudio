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
