# -*- coding: utf-8 -*-
"""因子生成"""
import numpy as np
import pandas as pd

import QuantStudio.api as QS

Factors = []

WDB = QS.FactorDB.WindDB2()

FT = WDB.getTable("中国A股盈利预测汇总")
Factors.append(FT.getFactor("净利润平均值(万元)", args={"计算方法":"FY0", "回溯天数":0}, new_name="净利润_FY0_0"))
Factors.append(FT.getFactor("净利润平均值(万元)", args={"计算方法":"FY0", "回溯天数":5}, new_name="净利润_FY0_30"))
#Factors.append(FT.getFactor("净利润平均值(万元)", args={"计算方法":"Fwd12M", "回溯天数":0}, new_name="净利润_Fwd12M"))

if __name__=="__main__":
    import datetime as dt
    
    WDB.connect()
    
    CFT = QS.FactorDB.CustomFT("ElementaryFactor")
    CFT.addFactors(factor_list=Factors)

    IDs = ["000001.SZ", "600000.SH"]# FT.getID(idt=dt.datetime(2018, 2, 1))
    StartDT = dt.datetime(2007, 1, 1)
    EndDT = dt.datetime(2007, 12, 31)
    DTs = WDB.getTable("中国A股交易日历").getDateTime(iid="SSE", start_dt=StartDT, end_dt=EndDT)
    MonthDTs = DTs
    CFT.OperationMode.DateTimes = DTs
    CFT.OperationMode.IDs = IDs
    CFT.OperationMode.SubProcessNum = 0
    CFT.OperationMode.DTRuler = DTs

    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    CFT.calculate(factor_db=HDB, table_name=QS.Tools.genAvailableName("TestTable", HDB.TableNames), if_exists="append")
    
    HDB.disconnect()
    WDB.disconnect()