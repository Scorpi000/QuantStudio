# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.StrategyTest import Strategy

# 简单买入并持有策略
class SimpleBuyAndHoldStrategy(Strategy):
    def init(self):
        self.ModelArgs["TargetID"] = "000001.SZ"# 进行交易的目标 ID
    def trade(self, idt, trading_record, signal):
        if self.UserData.get("Buy", True) or ((self.QSEnv.STM.Accounts["主账户"].Position!=0).sum()==0):
            self.QSEnv.STM.Accounts["主账户"].order(self.ModelArgs["TargetID"], 1)
            self.UserData["Buy"] = False
# 简单对冲策略
class SimleHedgingStrategy(Strategy):
    def init(self):
        self.ModelArgs["TargetID"] = "000001.SZ"
    def trade(self, idt, trading_record, signal):
        if self.UserData.get("Buy", True):
            self.QSEnv.STM.Accounts["主账户"].order(self.ModelArgs["TargetID"], 1)
            self.QSEnv.STM.Accounts["次账户"].order(self.ModelArgs["TargetID"], -1)
            self.UserData["Buy"] = False
# Demo 策略, 单数月份的月初做多一手股指期货, 双数月份的月初做空一手股指期货, 原有持仓在月初平仓
class DemoStrategy(Strategy):
    def init(self):
        self.UserData["LastMonth"] = None# 记录上一个月的月份
        self.ModelArgs["TargetID"] = "000001.SZ"# 进行交易的目标 ID
    def trade(self, idt, trading_record, signal):
        if self.UserData["LastMonth"]!=idt[4:6]:# 进入新的月份
            self.UserData["LastMonth"] = idt[4:6]
            Account = self.QSEnv.STM.Accounts["主账户"]
            Position = Account.Position# 当前已有持仓
            for iID in Position.index:# 先平仓
                Account.order(iID, -Position[iID])
            if int(idt[4:6]) % 2==0:# 双数月份
                Account.order(self.ModelArgs["TargetID"], -1)
            else:# 单数月份
                Account.order(self.ModelArgs["TargetID"], 1)

if __name__=='__main__':
    from QuantStudio import QSEnv
    from QuantStudio.FactorDataBase.HDF5DB import HDF5DB
    from QuantStudio.FactorDataBase.CloverDB import CloverDB
    from QuantStudio.FactorDataBase.MMAPFactorTable import FactorCacheFT, IDCacheFT
    from QuantStudio.StrategyTest.StockAccount import TimeBarAccount
    from QuantStudio.FunLib.DateTimeFun import cutDate, getTimeSeries, combineDateTime
    from QuantStudio.GUI.QtGUIFun import showOutput
    QSE = QSEnv()
    
    # 创建因子数据库
    MainDB = HDF5DB(QSE)
    MainDB.connect()
    # 创建自定义的因子表
    MainFT = FactorCacheFT("MainFT", QSE)
    FT = MainDB.getTable("ElementaryFactor")
    MainFT.addFactors(factor_table=FT, factor_names=["收盘价", "复权因子"], args={})
    MainFT.setDateTime(FT.getDateTime(ifactor_name="收盘价", start_dt=dt.datetime(2017,1,1), end_dt=dt.datetime(2018,1,1)))
    MainFT.setID(FT.getID(ifactor_name="收盘价"))
    QSE.addFactorTable(MainFT)
    
    # 添加账户
    iAccount = TimeBarAccount("主账户",QSE)
    iAccount.SysArgs["初始资金"] = 0
    iAccount.SysArgs["负债上限"] = np.inf
    iAccount.SysArgs["目标ID"] = ["000001.SZ"]
    iAccount.SysArgs["行情因子表"]["最新价"] = "收盘价"
    iAccount.SysArgs["行情因子表"]["成交价"] = "收盘价"
    #iAccount.SysArgs["复权因子表"]["复权因子"] = "复权因子"
    QSE.STM.addAccount(iAccount)# 将账户添加入 QS 系统的策略测试模型
    
    ## 添加账户
    #iAccount = FutureAccount("期货账户",QSE)
    #QSE.STM.addAccount(iAccount)# 将账户添加入 QS 系统的策略测试模型
    #iAccount.SysArgs["负债上限"] = np.inf
    #iAccount.SysArgs["初始资金"] = 0
    
    # 构建策略
    QSE.STM.Strategy = SimpleBuyAndHoldStrategy("主策略", QSE)# 将策略添加入 QS 系统的策略测试模型
    
    # 测试
    TestDateTimes = MainFT.getDateTime()
    QSE.STM.run(test_dts=TestDateTimes)
    
    # 生成报告
    Rslt = QSE.STM.output()
    showOutput(QSE, Rslt)
    QSE.close()  