# coding=utf-8
import datetime as dt

if __name__=='__main__':
    from QuantStudio.Factor.HDF5DB import HDF5DB
    from QuantStudio.Risk.HDF5RDB import HDF5FRDB
    from QuantStudio.Risk.RiskModel.BarraModel import BarraModel
    from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime

    StartDT, EndDT = dt.datetime(2017, 1, 23), dt.datetime(2017, 9, 29)
    FDB = HDF5DB(args={"MainDir": "../data/HDF5"}).connect()
    DTs = getMonthLastDateTime(FDB.getTable("stock_cn_day_bar_nafilled").getDateTime("close", start_dt=StartDT, end_dt=EndDT))

    FT = QS.FactorDB.CustomFT("MainFT")
    FT.addFactors(factor_table=FDB.getTable("ElementaryFactor"), factor_names=["日收益率", "总市值"])
    FT.addFactors(factor_table=FDB.getTable("BarraDescriptor"), factor_names=["ESTU", "Industry"])
    FT.addFactors(factor_table=FDB.getTable("BarraFactor"), factor_names=None)
    FT.setDateTime(FDB.getTable("BarraDescriptor").getDateTime(ifactor_name="ESTU"))
    FT.setID(FDB.getTable("BarraDescriptor").getID(ifactor_name="ESTU"))

    RDB = HDF5FRDB(args={"MainDir": "../data/Risk"}).connect()
    
    Model = BarraModel(name="MainModel", factor_table=FT, risk_db=RDB, table_name="stock_cn_barra_risk_model", config_file=None)
    Model.setRiskESTDateTime(DTs)
    Model.run()