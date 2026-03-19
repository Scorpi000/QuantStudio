# coding=utf-8
import datetime as dt

if __name__=='__main__':
    from QuantStudio.Factor.HDF5DB import HDF5DB
    from QuantStudio.Risk.HDF5RDB import HDF5FRDB
    from QuantStudio.Risk.RiskModel.BarraModel import BarraModel
    from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime

    StartDT, EndDT = dt.datetime(2020, 9, 30), dt.datetime(2020, 12, 31)
    FDB = HDF5DB(args={"MainDir": "/mnt/d/Data/HDF5DB"}).connect()
    DTs = getMonthLastDateTime(FDB.getTable("stock_cn_day_bar_nafilled").getDateTime("close", start_dt=StartDT, end_dt=EndDT))

    FT = FDB.getTable("stock_cn_factor_barra")

    RDB = HDF5FRDB(args={"MainDir": "/mnt/d/Data/HDF5RDB"}).connect()
    
    Model = BarraModel(name="MainModel", factor_table=FT, risk_db=RDB, table_name="stock_cn_barra_risk_model", config_file=None)
    Model.setRiskESTDateTime(DTs)
    Model.run()