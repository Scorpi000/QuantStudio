# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd

if __name__=='__main__':
    from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime
    from QuantStudio.Tools.QtGUI.QtGUIFun import showOutput
    from QuantStudio.FactorDataBase.HDF5DB import HDF5DB
    from QuantStudio.FactorDataBase.CustomDB import CustomDB, FactorCacheFT
    from QuantStudio.HistoryTest.HistoryTestModel import HistoryTestModel
    from QuantStudio.HistoryTest.SectionTest.IC import IC
    from QuantStudio.HistoryTest.SectionTest.Portfolio import QuantilePortfolio
    from QuantStudio.HistoryTest.StrategyTest.StrategyTestModule import Strategy
    from QuantStudio.HistoryTest.StrategyTest import StockAccount
    
    # 创建因子库
    MainDB = HDF5DB()
    MainDB.connect()
    
    # 创建自定义因子库
    MainFT = FactorCacheFT("MainFT")
    FT = MainDB.getTable("ElementaryFactor")
    MainFT.addFactors(factor_table=FT, factor_names=["复权收盘价", "流通市值"], args={})
    MainFT.setDateTime(FT.getDateTime(ifactor_name="复权收盘价", start_dt=dt.datetime(2017, 1, 1), end_dt=dt.datetime(2018, 1, 1)))
    MainFT.setID(FT.getID(ifactor_name="复权收盘价"))
    
    # 创建回测模型
    Model = HistoryTestModel()
    # --------因子测试模块--------
    iModule = IC(factor_table=MainFT)# IC 测试
    iModule["测试因子"] = ["流通市值"]
    iModule["计算时点"] = getMonthLastDateTime(MainFT.getDateTime()).tolist()
    Model.Modules.append(iModule)
    iModule = QuantilePortfolio(factor_table=MainFT)# 分位数组合测试
    iModule["测试因子"] = "流通市值"
    iModule["调仓时点"] = getMonthLastDateTime(MainFT.getDateTime()).tolist()
    Model.Modules.append(iModule)
    # --------策略测试模块--------
    #class DemoStrategy(Strategy):
        #def init(self):
            #self.ModelArgs["TargetID"] = "000001.SZ"# 进行交易的目标 ID
            #return ()
        #def trade(self, idt, trading_record, signal):
            #if self.UserData.get("Buy", True) or ((self.Accounts[0].Position!=0).sum()==0):
                #self.Accounts[0].order(self.ModelArgs["TargetID"], 1)
                #self.UserData["Buy"] = False
    #iModule = DemoStrategy()
    #iModule.Accounts.append(StockAccount.TimeBarAccount(market_ft=MainFT))
    #Model.Modules.append(iModule)
    
    # 设置模型参数
    #Model.setArgs()
    
    # 运行模型
    TestDateTimes = MainFT.getDateTime()
    Model.run(test_dts=TestDateTimes)
    
    # 查看结果
    Output = Model.output()
    showOutput(Output)
    