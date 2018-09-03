# coding=utf-8
"""测试横截面回测"""
import os
import datetime as dt

import numpy as np
import pandas as pd


if __name__=='__main__':
    import QuantStudio.api as QS
    
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    FT = HDB.getTable("ElementaryFactor")
    DTs = FT.getDateTime(ifactor_name="复权收盘价", start_dt=dt.datetime(2010, 1, 1), end_dt=dt.datetime(2018, 1, 1))
    IDs = FT.getID(ifactor_name="复权收盘价")
    MonthLastDTs = QS.Tools.DateTime.getMonthLastDateTime(DTs)
    
    # 创建自定义因子表
    MainFT = QS.FactorDB.CustomFT("MainFT")
    MainFT.addFactors(factor_table=FT, factor_names=["复权收盘价", "流通市值", "成交量", "Wind行业"], args={})
    MainFT.setDateTime(MonthLastDTs)
    MainFT.setID(IDs)
    
    # 创建回测模型
    Model = QS.HistoryTest.HistoryTestModel()
    # --------因子测试模块--------
    # IC 测试
    iModule = QS.HistoryTest.Section.IC(factor_table=MainFT)
    iModule["测试因子"] = ["流通市值", "成交量"]
    iModule["价格因子"] = "复权收盘价"
    iModule["计算时点"] = MonthLastDTs
    Model.Modules.append(iModule)
    ## 风险调整的 IC
    #iModule = QS.HistoryTest.Section.RiskAdjustedIC(factor_table=MainFT)
    #iModule["测试因子"] = ["流通市值"]
    #iModule["价格因子"] = "复权收盘价"
    #iModule["风险因子"] = ["成交量"]
    #iModule["行业因子"] = "Wind行业"
    #iModule["计算时点"] = MonthLastDTs
    #Model.Modules.append(iModule)
    ## IC 衰减
    #iModule = QS.HistoryTest.Section.ICDecay(factor_table=MainFT)
    #iModule["测试因子"] = "流通市值"
    #iModule["价格因子"] = "复权收盘价"
    #iModule["计算时点"] = MonthLastDTs
    #Model.Modules.append(iModule)
    ## 分位数组合测试
    #iModule = QS.HistoryTest.Section.QuantilePortfolio(factor_table=MainFT)
    #iModule["测试因子"] = "流通市值"
    #iModule["价格因子"] = "复权收盘价"
    #iModule["调仓时点"] = MonthLastDTs
    #Model.Modules.append(iModule)
    ## 因子值的行业分布
    #iModule = QS.HistoryTest.Section.IndustryDistribution(factor_table=MainFT)
    #iModule["测试因子"] = ["流通市值", "成交量"]
    #iModule["行业因子"] = "Wind行业"
    #iModule["计算时点"] = MonthLastDTs
    #Model.Modules.append(iModule)
    ## 因子截面相关性
    #iModule = QS.HistoryTest.Section.SectionCorrelation(factor_table=MainFT)
    #iModule["测试因子"] = ["流通市值", "成交量"]
    #iModule["计算时点"] = MonthLastDTs
    #Model.Modules.append(iModule)
    ## 因子换手率
    #iModule = QS.HistoryTest.Section.FactorTurnover(factor_table=MainFT)
    #iModule["测试因子"] = ["流通市值", "成交量"]
    #iModule["计算时点"] = MonthLastDTs
    #Model.Modules.append(iModule)
    ## Fama-MacBeth 回归
    #iModule = QS.HistoryTest.Section.FamaMacBethRegression(factor_table=MainFT)
    #iModule["测试因子"] = ["流通市值", "成交量"]
    #iModule["价格因子"] = "复权收盘价"
    #iModule["行业因子"] = "Wind行业"
    #iModule["计算时点"] = MonthLastDTs
    #Model.Modules.append(iModule)
    
    # 运行模型
    TestDTs = MainFT.getDateTime()
    Model.run(test_dts=TestDTs)
    
    # 查看结果
    QS.Tools.QtGUI.showOutput(Model.output())
    #Model.genHTMLReport("aha.html")