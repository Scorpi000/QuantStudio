# coding=utf-8
import datetime as dt

if __name__=='__main__':
    import QuantStudio.api as QS
    
    # 创建因子库
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    
    # 创建自定义因子表
    MainFT = QS.FactorDB.CustomFT("MainFT")
    FT = HDB.getTable("ElementaryFactor")
    DTs = FT.getDateTime(ifactor_name="复权收盘价", start_dt=dt.datetime(2017, 1, 1), end_dt=dt.datetime(2018, 1, 1))
    MonthLastDTs = QS.Tools.DateTime.getMonthLastDateTime(DTs)
    MainFT.addFactors(factor_table=FT, factor_names=["复权收盘价", "流通市值"], args={})
    MainFT.setDateTime(MonthLastDTs)
    MainFT.setID(FT.getID(ifactor_name="复权收盘价"))
    
    # 创建回测模型
    Model = QS.HistoryTest.HistoryTestModel()
    class DemoStrategy(QS.HistoryTest.Strategy.Strategy):
        def init(self):
            self.ModelArgs["TargetID"] = "000001.SZ"# 进行交易的目标 ID
            return ()
        def trade(self, idt, trading_record, signal):
            if self.UserData.get("Buy", True) or ((self.Accounts[0].Position!=0).sum()==0):
                self.Accounts[0].order(self.ModelArgs["TargetID"], 1)
                self.UserData["Buy"] = False
    iModule = DemoStrategy(name="Demo")
    iAccount = QS.HistoryTest.Strategy.StockAccount.TimeBarAccount(market_ft=MainFT)
    iModule.Accounts.append(iAccount)
    iModule["比较基准"]["因子表"] = MainFT
    iModule["比较基准"]["价格因子"] = "复权收盘价"
    iModule["比较基准"]["基准ID"] = "000001.SZ"
    Model.Modules.append(iModule)
    
    # 设置模型参数
    #Model.setArgs()
    
    # 运行模型
    TestDateTimes = MainFT.getDateTime()
    Model.run(test_dts=TestDateTimes)
    
    # 查看结果
    QS.Tools.QtGUI.showOutput(Model.output())