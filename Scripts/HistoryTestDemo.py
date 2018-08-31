# coding=utf-8
import os
import datetime as dt

import numpy as np
import pandas as pd

if __name__=='__main__':
    import QuantStudio.api as QS
    #from QuantStudio.HistoryTest.StrategyTest.StrategyTestModule import Strategy
    #from QuantStudio.HistoryTest.StrategyTest import StockAccount
    
    # 创建因子库
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    
    # 创建自定义因子表
    MainFT = QS.FactorDB.CustomFT("MainFT")
    FT = HDB.getTable("ElementaryFactor")
    DTs = FT.getDateTime(ifactor_name="复权收盘价", start_dt=dt.datetime(2017, 1, 1), end_dt=dt.datetime(2018, 1, 1))
    MonthLastDT = QS.Tools.DateTime.getMonthLastDateTime(DTs)
    MainFT.addFactors(factor_table=FT, factor_names=["复权收盘价", "流通市值"], args={})
    MainFT.setDateTime(MonthLastDT)
    MainFT.setID(FT.getID(ifactor_name="复权收盘价"))
    
    # 创建回测模型
    Model = QS.HistoryTest.HistoryTestModel()
    # --------因子测试模块--------
    iModule = QS.HistoryTest.Section.IC(factor_table=MainFT)# IC 测试
    iModule["测试因子"] = ["流通市值", "复权收盘价"]
    iModule["计算时点"] = MonthLastDT
    Model.Modules.append(iModule)
    #iModule = QS.HistoryTest.Section.QuantilePortfolio(factor_table=MainFT)# 分位数组合测试
    #iModule["测试因子"] = "流通市值"
    #iModule["调仓时点"] = QS.Tools.DateTime.getMonthLastDateTime(MainFT.getDateTime()).tolist()
    #Model.Modules.append(iModule)
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
    #Output = Model.output()
    #QS.Tools.QtGUI.showOutput(Output)
    
    Fig = iModule.genMatplotlibFigReport(file_name=QS.Tools.File.getWindowsDesktopPath()+os.sep+"aha.png")
    Fig.show(block=True)