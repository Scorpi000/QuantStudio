# coding=utf-8
"""生成风险数据"""
import datetime as dt

if __name__=='__main__':
    import QuantStudio.api as QS
    
    #DTs = [dt.datetime(2005,1,31,23,59,59,999999), dt.datetime(2005,2,28,23,59,59,999999)]
    StartDT, EndDT = dt.datetime(2017,1,1,23,59,59,999999), dt.datetime(2017,9,29,23,59,59,999999)
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    DTs = QS.Tools.DateTime.getMonthLastDateTime(HDB.getTable("ElementaryFactor").getDateTime("日收益率", start_dt=StartDT, end_dt=EndDT))
    
    FT = QS.FactorDB.CustomFT("MainFT")
    FT.addFactors(factor_table=HDB.getTable("ElementaryFactor"), factor_names=["日收益率", "总市值"])
    FT.addFactors(factor_table=HDB.getTable("BarraDescriptor"), factor_names=["ESTU", "Industry"])
    FT.addFactors(factor_table=HDB.getTable("BarraFactor"), factor_names=None)
    FT.renameFactor("总市值", "市值")
    
    RiskDB = QS.RiskModel.FRDB()
    
    Model = QS.RiskModel.BarraModel("MainModel", factor_table=FT, risk_db=RiskDB, table_name="Barra风险数据", config_file=None)
    Model.setRiskESTDateTime(DTs)
    Model.run()