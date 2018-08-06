# coding=utf-8
"""ETF - 期货套利策略"""

import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio import QSEnv
from QuantStudio.FactorDataBase.CloverDB import CloverDB
from QuantStudio.FactorDataBase.WindDB2 import WindDB2
from QuantStudio.FactorDataBase.MMAPFactorTable import IDCacheFT
from QuantStudio.StrategyTest import FutureAccount, StockAccount, Strategy
from QuantStudio.FunLib.DateTimeFun import combineDateTime, getTimeSeries, getDateSeries
from QuantStudio.GUI.QtGUIFun import showOutput

# 套利策略
class ArbitrageStrategy(Strategy):
    def init(self):
        self.ModelArgs["ETFID"] = "510050.SH"# ETF 目标 ID
        self.ModelArgs["FutureID"] = "IH00.CFE"# 期货目标 ID
        self.ModelArgs["a"] = -286.65480309
        self.ModelArgs["b"] = 5.01292795
        self.ModelArgs["sigma"] = 11.34432971413167
    def trade(self, idt, trading_record, signal):
        ETFPrice = self.QSEnv.STM.Accounts["股票账户"].LastPrice[self.ModelArgs["ETFID"]]
        FuturePrice = self.QSEnv.STM.Accounts["期货账户"].LastPrice[self.ModelArgs["FutureID"]]
        Residual = FuturePrice-self.ModelArgs["a"]-self.ModelArgs["b"]*ETFPrice
        Position = self.QSEnv.STM.Accounts["股票账户"].Position
        if (Position.shape[0]==0) and (Residual>self.ModelArgs["sigma"]*2):# 当前无仓位且价差超过两倍标准差, 开仓
            self.QSEnv.STM.Accounts["期货账户"].order(self.ModelArgs["FutureID"], -1)
            ETFNum = 1*self.QSEnv.STM.Accounts["期货账户"].SysArgs["合约乘数"]*self.ModelArgs["b"]
            self.QSEnv.STM.Accounts["股票账户"].order(self.ModelArgs["ETFID"], ETFNum)
        elif (Position.shape[0]>0) and (Residual<=0):# 当前有仓位且价差低于一倍标准差, 平仓
            self.QSEnv.STM.Accounts["期货账户"].order(self.ModelArgs["FutureID"], 1)
            self.QSEnv.STM.Accounts["股票账户"].order(self.ModelArgs["ETFID"], -Position[self.ModelArgs["ETFID"]])
        return 0
        

if __name__=='__main__':
    QSE = QSEnv()
    
    # 创建因子数据库
    MainDB = CloverDB(QSE)
    MainDB.connect()
    QSE.addFactorDataBase(MainDB)
    WDB = WindDB2(QSE)
    WDB.connect()
    # 创建自定义的因子表
    MainFT = IDCacheFT("MainFT", QSE)
    MainFT.addFactors("Tick 数据", ["lst"], db_name="CloverDB", args={"时间间隔":300})
    MainFT.DateTimes = list(combineDateTime(WDB.getTradeDay(dt.date(2017,11,18), dt.date(2017,12,31)), 
                                            np.append(getTimeSeries(dt.time(9,30), dt.time(11,30), dt.timedelta(minutes=5)), getTimeSeries(dt.time(13), dt.time(14,57), dt.timedelta(minutes=5)))))
    MainFT.IDs = ["510050.SH", "IH00.CFE"]
    QSE.addFactorTable(MainFT)
    
    # 添加 ETF 账户
    iAccount = StockAccount("股票账户",QSE)
    QSE.STM.addAccount(iAccount)# 将账户添加入 QS 系统的策略测试模型
    iAccount.SysArgs["负债上限"] = np.inf
    iAccount.SysArgs["初始资金"] = 0
    iAccount.SysArgs["买入限制"]["交易费率"] = 0.0
    iAccount.SysArgs["卖出限制"]["交易费率"] = 0.0
    
    # 添加期货账户
    iAccount = FutureAccount("期货账户",QSE)
    QSE.STM.addAccount(iAccount)# 将账户添加入 QS 系统的策略测试模型
    iAccount.SysArgs["负债上限"] = np.inf
    iAccount.SysArgs["初始资金"] = 0
    iAccount.SysArgs["合约乘数"] = 300
    iAccount.SysArgs["保证金率"] = 0.15
    iAccount.SysArgs["买入限制"]["交易费率"] = 0.0
    iAccount.SysArgs["卖出限制"]["交易费率"] = 0.0
    
    # 构建策略
    QSE.STM.Strategy = ArbitrageStrategy("主策略", QSE)# 将策略添加入 QS 系统的策略测试模型
    
    # 测试
    TestDateTimes = MainFT.DateTimes
    QSE.STM.run(test_dts=TestDateTimes)
    
    # 生成报告
    Rslt = QSE.STM.output()
    showOutput(QSE, Rslt)
    QSE.close()