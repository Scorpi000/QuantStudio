# coding=utf-8
import datetime as dt

if __name__=='__main__':
    import QuantStudio.api as QS
    
    # 创建因子库
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    
    # 创建自定义因子表
    MainFT = QS.FactorDB.CustomFT("MainFT")
    ElementaryFT = HDB.getTable("ElementaryFactor")
    DTs = ElementaryFT.getDateTime(ifactor_name="复权收盘价", start_dt=dt.datetime(2007, 1, 1), end_dt=dt.datetime(2018, 9, 1))
    MonthLastDTs = QS.Tools.DateTime.getMonthLastDateTime(DTs)
    FactorNames = ["BP_LR"]
    MainFT.addFactors(factor_table=ElementaryFT, factor_names=["复权收盘价", "中信行业", "是否在市"], args={})
    MainFT.addFactors(factor_table=HDB.getTable("StyleValueFactor"), factor_names=FactorNames, args={})
    
    MainFT.setDateTime(MonthLastDTs)
    MainFT.setID(ElementaryFT.getID(ifactor_name="复权收盘价"))
    
    # 创建回测模型
    Model = QS.HistoryTest.HistoryTestModel()
    # --------因子测试模块--------
    # IC 测试
    iModule = QS.HistoryTest.Section.IC(factor_table=MainFT)
    iModule["测试因子"] = FactorNames
    iModule["价格因子"] = "复权收盘价"
    iModule["计算时点"] = MonthLastDTs
    iModule["筛选条件"] = "@是否在市==1"
    Model.Modules.append(iModule)
    # 分位数组合测试
    for iFactorName in FactorNames:
        iModule = QS.HistoryTest.Section.QuantilePortfolio(factor_table=MainFT)# IC 测试
        iModule["测试因子"] = iFactorName
        iModule["价格因子"] = "复权收盘价"
        iModule["调仓时点"] = MonthLastDTs
        iModule["筛选条件"] = "@是否在市==1"
        Model.Modules.append(iModule)
    
    # 运行模型
    TestDateTimes = MainFT.getDateTime()
    Model.run(test_dts=TestDateTimes)
    
    #display(Model)
    #Model.genHTMLReport("aha.html")
    QS.Tools.QtGUI.showOutput(Model.output())