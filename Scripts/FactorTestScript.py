# coding=utf-8
import os

import numpy as np
import pandas as pd

from QuantStudio import QSEnv
from QuantStudio.GUI.QtGUIFun import showOutput
from QuantStudio.FactorTest import IC, RiskAdjustedIC


if __name__=='__main__':
    QSE = QSEnv()
    
    # 创建数据源
    # GUI方式创建
    #MainDS = setDataSource(QSE)
    # 手动方式创建
    from QuantStudio.DataSource import MMAPLocalCacheDS
    from QuantStudio.FunLib.DateTimeFun import getMonthLastDateTime
    from QuantStudio.GUI.ResultDlg import PlotlyResultDlg
    MainDS = MMAPLocalCacheDS("MainDS", QSE)
    MainDS.addFactors("ElementaryFactor", ["复权收盘价"], db_name="FactorDB")
    MainDS.DateTimes = getMonthLastDateTime(MainDS.extractDateTimes("复权收盘价"))
    MainDS.IDs = MainDS.extractIDs("复权收盘价")
    
    #SecondaryDS = MMAPLocalCacheDS("SecondaryDS", QSE)
    #SecondaryDS.addFactors("ElementaryFactor", ["复权收盘价"], db_name="FactorDB")
    #SecondaryDS.DateTimes = getMonthLastDateTime(SecondaryDS.extractDates())
    #SecondaryDS.IDs = SecondaryDS.extractIDs()
    # 将数据源添加入 QS 系统
    QSE.DSs[MainDS.Name] = MainDS
    #QSE.DSs[SecondaryDS.Name] = SecondaryDS
    
    # 设置因子测试模块
    iModule = IC(QSE)
    QSE.FTM.append(iModule)
    #setArgs(QSE, iModule)# GUI方式设置参数
    
    #iModule = RiskAdjustedIC(QSE)
    #QSE.FTM.append(iModule)
    #setArgs(QSE, iModule)
    # 手动设置参数
    #iModule.SysArgs["测试因子"] = pd.DataFrame([("升序",)],index=["成交金额"],columns=["排序方向"])
    #iModule.SysArgs["价格因子"] = "复权收盘价"
    
    
    # 测试
    QSE.FTM.run(test_dts=MainDS.getDateTime())
    
    
    # 查看结果
    showOutput(QSE, QSE.FTM.output())
    
    # 生成报告
    #QSE.FTM.genExcelReport(os.getcwd()+os.sep+"因子测试报告.xlsx")
    QSE.close()