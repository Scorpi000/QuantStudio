# coding=utf-8
"""基于行情 API 的因子库"""
import os
import datetime as dt
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas as pd
from traits.api import Str, Password, Directory, ListStr

from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.TraderAPI.CTP.QSCTP import MdApi

class _MdApi(MdApi):
    def __init__(self):
        super().__init__()
        self._FactorDB = fdb
        self._ReqRef = 0
    def disconnect(self):
        self._ReqRef += 1
        ErrorCode = self.reqUserLogout({}, self._ReqRef)
    def py_onFrontConnected(self):
        """服务器连接"""
        # 登陆
        LoginReq = {"UserID": self._FactorDB.UserID,
                           "Password": self._FactorDB.Pwd,
                           "BrokerID": self._FactorDB.BrokerID}
        self._ReqRef += 1# 请求数必须保持唯一性
        ErrorCode = self.reqUserLogin(LoginReq, self._ReqRef)
    def py_onRspError(self, error, n, last):
        """错误"""
        print(error)
    def py_onRspUserLogin(self, data, error, n, last):
        """登陆回报"""
        # 订阅合约
        for iID in self._FactorDB.IDs:
            ErrorCode = self.subscribeMarketData(iID)
    def py_onRspUserLogout(self, data, error, n, last):
        """登出回报"""
        self.exit()
    def py_onRtnDepthMarketData(self, data):
        """行情推送"""
        ID = data.pop("InstrumentID")
        IDData = self._FactorDB._CacheData.setdefault(ID, {})
        iDate = data.pop("TradingDay")
        iTime = data.pop("UpdateTime")
        iMillisec = str(data.pop("UpdateMillisec"))
        iMillisec += "0" * (6-len(iMillisec))
        iDT = dt.datetime.strptime(iDate+" "+iTime+"."+iMillisec, "%Y%m%d %H:%M:%S.%f")
        IDData[iDT] = data

# 基于原始 tick 行情的因子表
class _TickTable(FactorTable):
    """原始 tick 行情的因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        return []
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        elif set(factor_names).isdisjoint(self.FactorNames): return super().getFactorMetaData(factor_names=factor_names, key=key)
        if key=="DataType": return self._DataType.loc[factor_names]
        with self._FactorDB._DataLock:
            MetaData = {}
            for iFactorName in factor_names:
                if iFactorName in self.FactorNames:
                    with h5py.File(self._FactorDB.MainDir+os.sep+self.Name+os.sep+iFactorName+"."+self._Suffix) as File:
                        if key is None: MetaData[iFactorName] = pd.Series(File.attrs)
                        elif key in File.attrs: MetaData[iFactorName] = File.attrs[key]
        if not MetaData: return super().getFactorMetaData(factor_names=factor_names, key=key)
        if key is None: return pd.DataFrame(MetaData).loc[:, factor_names]
        else: return pd.Series(MetaData).loc[factor_names]
    def getID(self, ifactor_name=None, idt=None, args={}):
        return sorted(self._FactorDB._MdAPI._CacheData)
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Data = pd.Panel(self._FactorDB._CacheData)
        return Data

# 基于行情 API 的因子数据库, TODO
class MdAPIDB(FactorDB):
    """MdAPIDB"""
    UserID = Str("118073", arg_type="String", label="UserID", order=0)
    Pwd = Password("shuntai11", arg_type="String", label="Password", order=1)
    BrokerID = Str("9999", arg_type="String", label="BrokerID", order=2)
    FrontAddr = Str("tcp://180.168.146.187:10010", arg_type="String", label="前置机地址", order=3)
    ConDir =  Directory(label="流文件目录", arg_type="Directory", order=4)
    IDs = ListStr(label="订阅ID", arg_type="MultiOption", order=5)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"MdAPIDBConfig.json" if config_file is None else config_file), **kwargs)
        self._MdAPI = None
        self._CacheData = {}
        # 继承来的属性
        self.Name = "MdAPIDB"
        return
    def connect(self):
        self._MdAPI = _MdApi(fdb=self)# 创建 API 对象
        self._MdAPI.createFtdcMdApi(self.ConDir)# 在 C++ 环境中创建对象, 传入参数是希望用来保存 .con 文件的地址
        self._MdAPI.registerFront(self.FrontAddr)# 注册前置机地址
        self._MdAPI.init()# 初始化, 连接前置机
        return 0
    def disconnect(self):
        self._MdAPI.disconnect()
        self._MdAPI = None
        return 0
    def isAvailable(self):
        return (self._MdAPI is not None)
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        return ["MarketData"]
    def getTable(self, table_name, args={}):
        return _TickTable(name=table_name, fdb=self, sys_args=args)