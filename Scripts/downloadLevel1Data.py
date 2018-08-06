# coding=utf-8
import time
import sys
import gc

from QuantStudio import QSEnv
QSE = QSEnv()

from QuantStudio.DataBase.TinySoftDB import TinySoft
TSDB = TinySoft(QSE)
TSDB.SysArgs["安装目录"] = "C:\\Program Files\\Tinysoft\\Analyse.NET"
TSDB.connect()

from QuantStudio.FactorDB.HFDB import HDF5HFDB
HFDB = HDF5HFDB(QSE)
HFDB.connect()

TargetTable = "TinySoft_ETF_Level1"
LogFilePath = "C:\\HST\\QuantStudio\\Scripts\\Log.txt"
#Dates = TSDB.getTradeDay(start_date="20100416")
Dates = ["20100416"]
#IDs = ["IF00","IF01","IF02","IF03","IF04","IC00","IC01","IC02","IC03","IC04","IH00","IH01","IH02","IH03","IH04"]
IDs = ["510050.SH","510680.SH","510800.SH"]
#IDs = ["IC00"]
StartDate = Dates[0]
StartID = IDs[0]

StartT = time.clock()
DataNum = 0
with open(LogFilePath,"a") as LogFile:
    sys.stdout = LogFile
    print("=========开始=======")
    for iDate in Dates[Dates.index(StartDate):]:
        for jID in IDs[IDs.index(StartID):]:
            ijData = TSDB.getTickData(jID, start_date=iDate, end_date=iDate)
            HFDB.writeIDDateData(TargetTable, jID, iDate, ijData)
            DataNum += 1
            print(iDate+" - "+jID)
        StartID = IDs[0]
        gc.collect()
    print("数据量: %d" % DataNum)
    print("运行时间: %f" % (time.clock()-StartT))

QSE.close()