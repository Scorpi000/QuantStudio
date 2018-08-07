# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Bool, Int

from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable, _adjustDateTime, _genStartEndDate

class _BarTable(FactorTable):
    """Bar 线因子表"""
    Freq = Enum("D", "W", "M", "5", "15", "30", "60", arg_type="SingleOption", order=0, label="时间周期")
    AdjustType = Enum(None, "qfq", "hfq", arg_type="SingleOption", order=1, label="复权方式")
    isIndex = Bool(False, arg_type="Bool", order=2, label="复权方式")
    FillNa = Bool(True, arg_type="Bool", order=3, label="缺失填充")
    LookBack = Int(0, arg_type="Integer", order=4, label="回溯天数")
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = name
        return    
    @property
    def FactorNames(self):
        return ["open", "low", "high", "close", "volume"]
    def _readIDData(self, iid, factor_names=None, dts=None, args={}):
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = dt.date.today()-dt.timedelta(365), dt.date.today()
        FillNa = args.get("缺失填充", self.FillNa)
        if FillNa: StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        if factor_names is None: factor_names = self.FactorNames
        Code = ".".join(iid.split(".")[:-1])
        KType = args.get("时间周期", self.Freq)
        Data = self.FactorDB._ts.get_k_data(code=Code, start=StartDate.strftime("%Y-%m-%d"), end=EndDate.strftime("%Y-%m-%d"), 
                                            ktype=KType, autype=args.get("复权方式", self.AdjustType),
                                            index=args.get("是否指数", self.isIndex))
        Data.pop("code")
        Data = Data.set_index(["date"])
        if KType in ("D","M","W"):
            iTime = dt.time(23,59,59,999999)
            Data.index = [dt.datetime.combine(dt.datetime.strptime(iDate, "%Y-%m-%d").date(), iTime) for iDate in Data.index]
        else:
            Data.index = [dt.datetime.strptime(iDate, "%Y-%m-%d %H:%M") for iDate in Data.index]
        return _adjustDateTime(Data.ix[:, factor_names], dts, fillna=FillNa, method="pad")
     # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def __QS_readData__(self, factor_names=None, ids=None, dts=None, args={}):
        if ids is None:
            ids = ["000001.SH"]
            args["是否指数"] = True
        if factor_names is None: factor_names = self.FactorNames
        Data = pd.Panel({iID:self._readIDData(iID, factor_names=factor_names, dts=dts, args=args) for iID in ids})
        return Data.swapaxes(0, 2)

class TushareDB(FactorDB):
    """tushare 网络数据源"""
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = "TushareDB"
        self._ts = None# tushare 模块
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ts"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self,state):
        self.__dict__.update(state)
        if self._ts:
            self._ts = None
            self.connect()
        else:
            self._ts = None
    def connect(self):
        import tushare as ts
        self._ts = ts
        return 0
    def disconnect(self):
        self._ts = None
        return 0
    def isAvailable(self):
        return (self._ts is not None)
    @property
    def TableNames(self):
        return ["Bar 线数据"]
    def getTable(self, table_name):
        if table_name in ("Bar 线数据", ): FT = _BarTable(table_name)
        FT.FactorDB = self
        return FT
    def getTradeDay(self, start_date=None, end_date=None, exchange="上海证券交易所"):
        if start_date is None: start_date = dt.date(1900,1,1)
        if end_date is None: end_date = dt.date.today()
        Dates = self._ts.trade_cal()
        Dates = Dates["calendarDate"][Dates["isOpen"]==1].map(lambda x:dt.datetime.strptime(x, "%Y-%m-%d").date()).values
        return sorted(Dates[(Dates>=start_date) & (Dates<=end_date)])

if __name__=="__main__":
    TSDB = TushareDB()
    TSDB.connect()
    
    Dates = TSDB.getTradeDay(start_date=dt.date(2018,1,1))
    print(TSDB.TableNames)
    FT = TSDB.getTable("Bar 线数据")
    print(FT.FactorNames)
    FactorData = FT.readData(factor_names=["close"], ids=["600000.SH", "000001.SZ"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999)).iloc[0]
    IDData = FT.readData(ids=["600000.SH"], factor_names=["close", "volume"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999)).iloc[:,:,0]
    DateTimeData = FT.readData(dts=[dt.datetime(2018,4,26,23,59,59,999999)], factor_names=["close", "volume"], ids=["600000.SH", "000001.SZ"]).iloc[:,0,:]
    Data = FT.readData(factor_names=["close", "volume"], ids=["600000.SH", "000001.SZ"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999))
    
    TSDB.disconnect()