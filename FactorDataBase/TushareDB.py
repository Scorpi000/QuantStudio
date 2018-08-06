# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio import QSObject, QSError, QSArgs
from QuantStudio.FactorDataBase.FactorDB import FactorDB
from QuantStudio.FactorDataBase.FactorTable import FactorTable, _adjustDateTime, _genStartEndDate

class _BarTable(FactorTable):
    """Bar 线因子表"""
    def __init__(self, name, qs_env, sys_args={}):
        super().__init__(name, qs_env, sys_args)
        self._DB = None# 所属的数据源
        return
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if args is None:
            SysArgs = {"时间周期":"D", "复权方式":None, "是否指数":False, "缺失填充":True, "回溯天数":0}
            ArgInfo = {}
            ArgInfo["时间周期"] = {"type":"SingleOption", "order":0, "range":["D", "W", "M", "5", "15", "30", "60"]}
            ArgInfo["复权方式"] = {"type":"SingleOption", "order":1, "range":[None, "qfq", "hfq"]}
            ArgInfo["是否指数"] = {"type":"Bool", "order":2}
            ArgInfo["缺失填充"] = {"type":"Bool", "order":3}
            ArgInfo["回溯天数"] = {"type":"Integer", "order":4, "min":0, "max":np.inf, "single_step":1}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        return args
    @property
    def FactorNames(self):
        return np.array(("open", "low", "high", "close", "volume"), dtype=np.dtype("O"))
    def readIDData(self, iid, factor_names=None, dts=None, start_dt=None, end_dt=None, args={}):
        StartDate, EndDate = _genStartEndDate(dts, start_dt, end_dt)
        FillNa = args.get("缺失填充", self.SysArgs["缺失填充"])
        if FillNa:
            StartDate -= dt.timedelta(args.get("回溯天数", self.SysArgs["回溯天数"]))
        if factor_names is None:
            factor_names = self.FactorNames
        Code = ".".join(iid.split(".")[:-1])
        KType = args.get("时间周期", self.SysArgs["时间周期"])
        Data = self._DB.ts.get_k_data(code=Code, start=StartDate.strftime("%Y-%m-%d"), end=EndDate.strftime("%Y-%m-%d"), 
                                      ktype=KType, autype=args.get("复权方式", self.SysArgs["复权方式"]),
                                      index=args.get("是否指数", self.SysArgs["是否指数"]))
        Data.pop("code")
        Data = Data.set_index(["date"])
        if KType in ("D","M","W"):
            iTime = dt.time(23,59,59,999999)
            Data.index = [dt.datetime.combine(dt.datetime.strptime(iDate, "%Y-%m-%d").date(), iTime) for iDate in Data.index]
        else:
            Data.index = [dt.datetime.strptime(iDate, "%Y-%m-%d %H:%M") for iDate in Data.index]
        return _adjustDateTime(Data.ix[:, factor_names], dts, start_dt, end_dt, fillna=FillNa, method="pad")
     # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if ids is None:
            ids = ["000001.SH"]
            args["是否指数"] = True
        if factor_names is None:
            factor_names = self.FactorNames
        Data = {}
        for iID in ids:
            Data[iID] = self.readIDData(iID, factor_names=factor_names, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        Data = pd.Panel(Data)
        return Data.swapaxes(0, 2)

class TushareDB(FactorDB):
    """tushare 网络数据源"""
    def __init__(self, qs_env, sys_args={}):
        super().__init__(qs_env, sys_args)
        # 继承来的属性
        # self.QSEnv QS 系统环境对象
        self._Name = "TushareDB"
        self.ts = None# tushare 模块
        self._FTs = {}# 因子表对象池
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        state["ts"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self,state):
        self.__dict__.update(state)
        if self.ts:
            self.ts = None
            self.connect()
        else:
            self.ts = None
    def connect(self):
        import tushare as ts
        self.ts = ts
        return 0
    def disconnect(self):
        self.ts = None
        return 0
    def isAvailable(self):
        return (self.ts is not None)
    @property
    def TableNames(self):
        return np.array(("Bar 线数据", ), dtype=np.dtype("O"))
    @property
    def DefaultTable(self):
        return self.getTable("Bar 线数据")    
    def getTable(self, table_name):
        if table_name in self._FTs:
            return self._FTs[table_name]
        elif table_name in ("Bar 线数据", ):
            FT = _BarTable(table_name, self.QSEnv)
        FT._DB = self
        self._FTs[table_name] = FT
        return FT
    def getTradeDay(self, start_date=None, end_date=None, exchange="上海证券交易所"):
        if start_date is None:
            start_date = dt.date(1900,1,1)
        if end_date is None:
            end_date = dt.date.today()
        Dates = self.ts.trade_cal()
        Dates = Dates["calendarDate"][Dates["isOpen"]==1].map(lambda x:dt.datetime.strptime(x, "%Y-%m-%d").date()).values
        return Dates[(Dates>=start_date) & (Dates<=end_date)]

if __name__=="__main__":
    from QuantStudio import QSEnv
    QSE = QSEnv()
    TSDB = TushareDB(QSE)
    TSDB.connect()
    
    Dates = TSDB.getTradeDay(start_date=dt.date(2018,1,1))
    print(TSDB.TableNames)
    FT = TSDB.DefaultTable
    print(FT.FactorNames)
    FactorData = FT.readData(factor_names=["close"], ids=["600000.SH", "000001.SZ"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999)).iloc[0]
    IDData = FT.readData(ids=["600000.SH"], factor_names=["close", "volume"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999)).iloc[:,:,0]
    DateTimeData = FT.readData(dts=[dt.datetime(2018,4,26,23,59,59,999999)], factor_names=["close", "volume"], ids=["600000.SH", "000001.SZ"]).iloc[:,0,:]
    Data = FT.readData(factor_names=["close", "volume"], ids=["600000.SH", "000001.SZ"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999))
    
    TSDB.disconnect()
    QSE.close()