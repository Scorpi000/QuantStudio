# coding=utf-8
"""利用分红配股数据还原复权因子的计算"""

import os
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio import QSEnv
from QuantStudio.FactorDataBase.WindDB2 import WindDB2

QSE = QSEnv()

WDB = WindDB2(QSE)
WDB.connect()

MarketFT = WDB.getTable("中国A股日行情")

IDs = MarketFT.getID()
DateTimes = MarketFT.getDateTime(start_dt=dt.datetime(2018,1,1))
MarketData = MarketFT.readData(factor_names=["昨收盘价", "收盘价"], 
                               ids=IDs, dts=DateTimes)
DvdData = WDB.getTable("中国A股分红").readData(factor_names=["每股派息(税前)", "每股送转"], 
                                              ids=IDs, dts=DateTimes)
DvdData.fillna(0, inplace=True)

RightIssueData = WDB.getTable("中国A股配股").readData(factor_names=["配股价格", "配股比例"], 
                                                     ids=IDs, dts=DateTimes)
RightIssueData.fillna(0, inplace=True)

Close = MarketData.loc["收盘价"].values[0:-1]

PreClose = np.round((Close - DvdData.loc["每股派息(税前)"].values[1:]+RightIssueData.loc["配股比例"].values[1:]*RightIssueData.loc["配股价格"].values[1:])/(1+DvdData.loc["每股送转"].values[1:]+RightIssueData.loc["配股比例"].values[1:]), 2)
Error = np.abs(PreClose - MarketData.loc["昨收盘价"].values[1:])
pass