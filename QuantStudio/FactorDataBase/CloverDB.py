# coding=utf-8
"""clover 数据源"""
import os
import re
import json
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import File, List, Float, Int, Bool, Enum, Str, Range, Password

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import adjustDateTime

class _CalendarTable(FactorTable):
    """交易日历因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        self._IDs = ["SSE", "SZSE", "SHFE", "CFFEX"]
        return
    @property
    def FactorNames(self):
        return ["交易日"]
    def getID(self, ifactor_name=None, idt=None, args={}):
        return self._IDs
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if iid is None: iid = "SSE"
        if start_dt is None: start_dt = dt.date(1900,1,1)
        start_date = start_dt.strftime("%Y-%m-%d")
        if end_dt is None: end_dt = dt.date.today()
        end_date = (end_dt + dt.timedelta(1)).strftime("%Y-%m-%d")
        jpckg = self._FactorDB._jpype.JPackage('clover.epsilon.database')
        Dates = jpckg.DatabaseUtil.getTradeDateList(self._jConn, iid, start_date, end_date)
        iTime = dt.time(0)
        return [dt.datetime.combine(dt.datetime.strptime(iDate, "%Y-%m-%d").date(), iTime) for iDate in Dates]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Data = pd.DataFrame({iID:pd.Series(1.0, index=self.getDateTime(iid=iID, start_dt=dts[0], end_dt=dts[-1])) for iID in ids})
        Data = Data.loc[dts, :]
        Data[pd.isnull(Data)] = 0
        return pd.Panel({"交易日":Data})

# tms: 时间戳, 单位: 毫秒; date: 日期; update: 最新的采样区间内有多少个行情点;
# lst: 最新价; vol: 最新成交量; amt: 最新成交额; oin: 最新持仓量;
# bid: 买一价; bsz: 买一量; ask: 卖一价; asz: 卖一量; 
# mid: 中间价, bid 和ask 的均值; wmp: bid 和 ask 的以 bsz 和 asz 的加权均值
# buyVolume: 确定为买方发起成交的成交量; sellVolume: 确定为卖方发起成交的成交量
class _TickTable(FactorTable):
    """Clover Tick 因子表"""
    TmsIntv = Float(3, arg_type="Double", label="时间间隔", order=0, low=0.5, high=86400, single_step=0.5)# 以秒为单位
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._FactorNames = ["lst","vol","amt","mid","bid","bsz","ask","asz","wmp","oin",
                             "buyVolume","sellVolume","bidSizeChange","askSizeChange",
                             "crossBidVolume","crossAskVolume","vi","ofi"]
        self._PriceFactors = {"lst", "amt", "mid", "bid", "ask", "wmp"}
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        return self._FactorNames
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self._FactorNames
        MetaData = pd.DataFrame("double", index=factor_names, columns=["DataType"], dtype=np.dtype("O"))
        if key is None: return MetaData
        elif key in MetaData: return MetaData.loc[:, key]
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if iid is None: return []
        if end_dt is None: end_dt = dt.datetime.today()
        if start_dt is None: start_dt = dt.datetime.combine(end_dt.date(), dt.time(0))
        StartDate, EndDate = start_dt.date().strftime("%Y-%m-%d"), (end_dt.date() + dt.timedelta(1)).strftime("%Y-%m-%d")
        SecDef = self._FactorDB.createSecDef([iid])
        tms_intv = int(args.get("时间间隔", self.TmsIntv)*1000)
        mdreader = self._FactorDB._jpype.JPackage("clover.epsilon.util").TestUtils.createMDReader("JavaSerialPT", "mdreader.SHFE")
        JPckg_matlab = self._FactorDB._jpype.JPackage("clover.model.matlab")
        sdm_def = JPckg_matlab.SecurityDataMatrixPT.generateSDMDataDef_CN(self._FactorDB._jConn, SecDef, StartDate, EndDate, tms_intv)
        sdm = JPckg_matlab.SecurityDataMatrixPT()
        sdm.fetchData(sdm_def, mdreader, tms_intv)
        tms = np.array(sdm.tms) / 1000
        start_dt, end_dt = start_dt.timestamp(), end_dt.timestamp()
        return [dt.datetime.fromtimestamp(iTMS / 1000) for iTMS in sdm.tms if (iTMS/1000<=end_dt) and (iTMS/1000>=start_dt)]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        SecDef = self._FactorDB.createSecDef(ids)
        tms_intv = int(args.get("时间间隔", self.TmsIntv)*1000)
        StartDate = dts[0].strftime("%Y-%m-%d")
        EndDate = (dts[-1] + dt.timedelta(1)).strftime("%Y-%m-%d")
        if self._PriceFactors.intersection(set(factor_names)):
            TradeDates = self._FactorDB._jpype.JPackage("clover.epsilon.database").DatabaseUtil.getTradeDateList(self._FactorDB._jConn, "SHFE", StartDate, EndDate)
            if len(TradeDates)==0: return adjustDateTime(pd.Panel(Data, major_axis=tms, minor_axis=ids), dts=dts)
            JPckg_util = self._FactorDB._jpype.JPackage("clover.model.util")
            PriceMultiplier = np.ones(len(SecDef))
            for j, jSecDef in enumerate(SecDef):
                jSecurity = JPckg_util.ModelUtils.parseSecurity(self._FactorDB._jConn, TradeDates[-1], JPckg_util.Json.parse(jSecDef))
                PriceMultiplier[j] = jSecurity.getMinPriceIncrement()
        mdreader = self._FactorDB._jpype.JPackage("clover.epsilon.util").TestUtils.createMDReader("Kryo.IQuotePT", "mdreader.SHFE")
        JPckg_matlab = self._FactorDB._jpype.JPackage("clover.model.matlab")
        sdm_def = JPckg_matlab.SecurityDataMatrixPT.generateSDMDataDef_CN(self._FactorDB._jConn, SecDef, StartDate, EndDate, tms_intv)
        sdm = JPckg_matlab.SecurityDataMatrixPT()
        sdm.fetchData(sdm_def, mdreader, tms_intv)
        tms = np.array(sdm.tms) / 1000
        tms = [dt.datetime.fromtimestamp(iTMS) for iTMS in tms]
        Data = {}
        for iField in factor_names:
            iData = np.array(getattr(sdm, iField))
            if iField in self._PriceFactors: iData = iData * PriceMultiplier
            Data[iField] = iData
        return pd.Panel(Data, major_axis=tms, minor_axis=ids).loc[factor_names]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if not dts: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        RawData = self.__QS_prepareRawData__(factor_names, ids, dts=[start_date, end_date], args=args)
        return adjustDateTime(RawData, dts=dts)
    def readDayData(self, factor_names, ids, start_date, end_date, args={}):
        RawData = self.__QS_prepareRawData__(factor_names, ids, dts=[start_date, end_date], args=args)
        if RawData.shape[2]==0: return pd.Panel(items=factor_names, major_axis=[], minor_axis=ids)
        return RawData.loc[:, :, ids]

class _TimeBarTable(FactorTable):
    """Clover Time Bar 因子表"""
    TmsIntv = Float(3, arg_type="Double", label="时间间隔", order=0, low=0.5, high=86400, single_step=0.5)
    Depth = Int(5, arg_type="Integer", label="深度", order=1, low=1, single_step=1)
    DynamicSecID = Bool(True, arg_type="Bool", label="动态证券ID", order=2)
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = name
    @property
    def FactorNames(self):
        FactorNames = ["mid","vol","amt","bid","ask","lowbid","lowask","lowtrade",
                       "highbid","highask","hightrade","buyvol","sellvol"]
        for i in range(self.Depth):
            FactorNames.append("bsz"+str(i+1))
            FactorNames.append("asz"+str(i+1))
        return np.array(FactorNames, dtype=np.dtype("O"))
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None:
            factor_names = self.FactorNames
        MetaData = pd.DataFrame("double", index=factor_names, columns=["DataType"], dtype=np.dtype("O"))
        if key is None:
            return MetaData
        elif key in MetaData:
            return MetaData.loc[:, key]
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
     # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 时间间隔: 以秒为单位, 默认是 3 秒
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if not dts: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        TBs, AllSecurityIDs, SecurityIDs, PriceMultiplier = self._FactorDB.getTimeBar(ids, StartDate, EndDate, 
                                                                                      tms_intv=args.get("时间间隔", self.TmsIntv)*1000, 
                                                                                      depth=args.get("深度", self.Depth), 
                                                                                      dynamic_security_id=args.get("动态证券ID", self.DynamicSecID))
        Data = self._FactorDB.fetchTimeBarData(factor_names, TBs, AllSecurityIDs, SecurityIDs, PriceMultiplier)
        return adjustDateTime(Data, dts=dts)

class _FeatureTable(FactorTable):
    """特征因子表"""
    @property
    def FactorNames(self):
        return ['Symbol', 'SecurityID', 'SecurityType', 'SecurityExchange', 'MinTradeVol', 'RoundLot', 'MinPriceIncrement', 'Currency', 'TimeZone',
                'MaturityMonthYear', 'ContractMultiplier', 'UnderlyingSecurityID', 'TrdSessLstGrp']
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        MetaData = pd.DataFrame("string", index=factor_names, columns=["DataType"], dtype=np.dtype("O"))
        if key is None: return MetaData
        elif key in MetaData: return MetaData.loc[:, key]
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        Data = {}
        for i in range((EndDate-StartDate).days):
            iDT = dt.datetime.combine(StartDate + dt.timedelta(i), dt.time(0))
            Data[iDT] = pd.DataFrame(self._FactorDB.getSecurityInfo(ids, idt=iDT), index=ids).loc[:, factor_names].T
        Data = pd.Panel(Data).swapaxes(0, 1)
        Data = adjustDateTime(Data, dts, fillna=True, method="bfill")
        return Data
class CloverDB(FactorDB):
    """Clover 数据库"""
    DBName = Str("epsilon", arg_type="String", label="数据库名", order=0)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=2)
    User = Str("epsilon", arg_type="String", label="用户名", order=3)
    Pwd = Password("epsilon7777", arg_type="String", label="密码", order=4)
    JVMPath = File("", arg_type="SingleOption", label="Java虚拟机", order=5, filter=["DLL (*.dll)"])
    JavaPckg = List(File("", filter=["Java Package (*.jar)"]), arg_type="ArgList", label="Java包", order=6)
    JavaOption = List(arg_type="StrList", label="Java选项", order=7)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._jpype = None# jpype 模块
        self._jConn = None# epsilon 数据库连接
        self._Suffix2Exchange = {"SH":"SSE", "SZ":"SZSE", "CFE":"CFFEX", "INE":"INE", "CZC":"CZCE", "DCE":"DCE", "SHF":"SHFE"}
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"CloverDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "CloverDB"
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_jConn"] = state["_jpype"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        isConnected = self._jpype
        self._jpype, self._jConn = None, None
        if isConnected: self.connect()
    def connect(self):
        if self._jpype is None:
            if not (os.path.isfile(self.JVMPath)): raise __QS_Error__("Java 虚拟机的路径设置有误!")
            import jpype
            self._jpype = jpype
        if not self._jpype.isJVMStarted():
            JOption = list(self.JavaOption)
            if self.JavaPckg: JOption.append("-Djava.class.path="+";".join(self.JavaPckg))
            self._jpype.startJVM(self.JVMPath, *JOption)
            jpckg = self._jpype.JPackage("java.sql")
            ConnStr = "jdbc:mysql://"+self.IPAddr+":"+str(self.Port)+"/"+self.DBName
            self._jConn = jpckg.DriverManager.getConnection(ConnStr, self.User, self.Pwd)
        return 0
    def disconnect(self):
        self._jConn = None
        #self._jpype.shutdownJVM()
        self._jpype = None
        return 0
    def isAvailable(self):
        if self._jpype is not None: return self._jpype.isJVMStarted()
        else: return False
    @property
    def TableNames(self):
        return ["Tick 数据", "Time Bar 数据", "证券特征", "交易日历"]
    def getTable(self, table_name, args={}):
        if table_name=="Tick 数据": return _TickTable(table_name, fdb=self, sys_args=args)
        elif table_name=="Time Bar 数据": return _TimeBarTable(table_name, fdb=self, sys_args=args)
        elif table_name=="证券特征": return _FeatureTable(table_name, fdb=self, sys_args=args)
        elif table_name=="交易日历": return _CalendarTable(table_name, fdb=self, sys_args=args)
        return None
    # 创建证券定义
    def createSecDef(self, ids, idate=dt.date.today()):
        SecDef = []
        for i, iID in enumerate(ids):
            iID = iID.split(".")
            iSuffix = iID[-1]
            iCode = ".".join(iID[:-1])
            iSecDef = {"SecurityExchange": self._Suffix2Exchange[iSuffix]}
            if iCode.isnumeric():# 全是数字
                iSecDef["Symbol"] = iCode
            elif iCode.isalnum():# 字母数字组合
                iSym, iNum = re.findall("\D+", iCode)[0], re.findall("\d+", iCode)[0]
                iSecDef["Symbol"] = iSym
                if len(iNum)==4: iSecDef["MaturityMonthYear"] = ("20%s" % (iNum, ))
                elif len(iNum)==3: iSecDef["MaturityMonthYear"] = ("201%s" % (iNum, ))
                else: iSecDef["MaturitySequence"] = ("+%d" % (int(iNum)+1, ))
            else: raise __QS_Error__("ID: %s 解析失败!" % (ids[i], ))# TODO
            SecDef.append(json.dumps(iSecDef))
        return SecDef
    # 获取证券的描述信息
    def getSecurityInfo(self, ids, idt=None):
        if idt is None: idt = dt.date.today()
        idt = idt.strftime("%Y-%m-%d")
        sec_def = self.createSecDef(ids)
        SecInfo = []
        jpckg = self._jpype.JPackage('clover.model.util')
        for iSecDef in sec_def:
            try:
                iSecInfo = jpckg.ModelUtils.parseSecurity(self._jConn, idt, jpckg.Json.parse(iSecDef))
                SecInfo.append(json.loads(iSecInfo.toString()))
            except:
                SecInfo.append(None)
        return SecInfo
    # ID 转换成证券 ID
    def ID2SecurityID(self, ids, idt=None):
        SecInfo = self.getSecurityInfo(ids, idt=idt)
        return [(str(iSecInfo["SecurityID"]) if iSecInfo is not None else None) for iSecInfo in SecInfo]
    # 获取 Time Bar, 返回 (Time Bar 对象 Array, [所有的证券 ID], DataFrame(证券ID, index=[日期], columns=ids), DataFrame(价格乘数, index=[日期], columns=ids))
    def getTimeBar(self, ids, start_date, end_date, tms_intv, depth, time_period='0930~1130,1300~1500', dynamic_security_id=True):
        StartDate = start_date.strftime("%Y-%m-%d")
        EndDate = (end_date + dt.timedelta(1)).strftime("%Y-%m-%d")
        TradeDates = self._jpype.JPackage('clover.epsilon.database').DatabaseUtil.getTradeDateList(self._jConn, 'SHFE', StartDate, EndDate)
        nID = len(ids)
        SecDef = self.createSecDef(ids)
        JPckg_util = self._jpype.JPackage('clover.model.util')
        JPckg_lowfreq = self._jpype.JPackage('clover.model.beta.lowfreq')
        PriceMultiplier = np.ones((len(TradeDates), nID))
        SecurityIDs = np.zeros((len(TradeDates), nID), dtype=np.int)
        if dynamic_security_id:
            for i, iDate in enumerate(TradeDates):
                for j, jSecDef in enumerate(SecDef):
                    ijSecurity = JPckg_util.ModelUtils.parseSecurity(self._jConn, iDate, JPckg_util.Json.parse(jSecDef))
                    SecurityIDs[i, j] = ijSecurity.getSecurityID()
                    PriceMultiplier[i, j] = ijSecurity.getMinPriceIncrement()
            AllSecurityIDs = np.unique(SecurityIDs).tolist()
        else:
            for j, jSecDef in enumerate(SecDef):
                jSecurity = JPckg_util.ModelUtils.parseSecurity(self._jConn, TradeDates[-1], JPckg_util.Json.parse(jSecDef))
                SecurityIDs[:, j] = jSecurity.getSecurityID()
                PriceMultiplier[:, j] = jSecurity.getMinPriceIncrement()
            AllSecurityIDs = SecurityIDs[0, :].tolist()
        TBP = self._jpype.JClass('clover.model.beta.lowfreq.TimeBarUtil$TimeBarProperty')(tms_intv, depth, time_period)
        TBs = JPckg_lowfreq.TimeBarUtil.query(TradeDates, AllSecurityIDs, 'SSE', 'Asia/Shanghai', TBP)
        TradeDates = [dt.datetime.strptime(iDate, "%Y-%m-%d").date() for iDate in TradeDates]
        SecurityIDs = pd.DataFrame(SecurityIDs, index=TradeDates, columns=ids)
        PriceMultiplier = pd.DataFrame(PriceMultiplier, index=TradeDates, columns=ids)
        return (TBs, AllSecurityIDs, SecurityIDs, PriceMultiplier)
    # 获取 Time Bar 数据, 返回 Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def fetchTimeBarData(self, fields, tbs, all_security_ids, security_ids, price_multiplier):
        if len(tbs)==0:
            return pd.Panel(items=fields, minor_axis=security_ids.columns)
        JPckg_lowfreq = self._jpype.JPackage('clover.model.beta.lowfreq')
        TBMatrix = JPckg_lowfreq.TimeBarUtil.toMatrix(tbs)
        TBDateTimes = [dt.datetime.fromtimestamp(iTMS) for iTMS in np.array(TBMatrix.time1)/1000]
        nDT, nID = len(TBDateTimes), security_ids.shape[1]
        iPreDate = None
        IDIndex = np.zeros((nDT, nID), dtype=np.int)
        DTPriceMultiplier = np.ones((nDT, nID))
        DTSecurityIDs = np.zeros((nDT, nID), dtype=np.int)
        TradeDates = list(security_ids.index)
        for i, iDT in enumerate(TBDateTimes):
            iDate = iDT.date()
            if iDate!=iPreDate:
                iIndex = TradeDates.index(iDate)
                for j, jSecurityID in enumerate(security_ids.values[iIndex]):
                    IDIndex[i, j] = all_security_ids.index(jSecurityID)
                DTPriceMultiplier[i, :] = price_multiplier.values[iIndex, :]
                DTSecurityIDs[i, :] = security_ids.values[iIndex, :]
                iPreDate = iDate
            else:
                IDIndex[i, :] = IDIndex[i-1, :]
                DTPriceMultiplier[i, :] = DTPriceMultiplier[i-1, :]
                DTSecurityIDs[i, :] = DTSecurityIDs[i-1, :]
        Data = {}
        RowIndex = np.arange(nDT)
        for iField in fields:
            if iField[:3]=="asz":
                iData = np.array(getattr(TBMatrix, "asz"))[:,:,int(iField[3])-1]
            elif iField[:3]=="bsz":
                iData = np.array(getattr(TBMatrix, "bsz"))[:,:,int(iField[3])-1]
            elif iField[:3]=="mid":
                iData = (np.array(getattr(TBMatrix, "ask"))+np.array(getattr(TBMatrix, "bid")))/2
            else:
                iData = np.array(getattr(TBMatrix, iField))
            iData = pd.DataFrame({iID:iData[RowIndex, IDIndex[:, i]] for i, iID in enumerate(security_ids.columns)}, index=TBDateTimes).loc[:, security_ids.columns]
            if iField in ("mid", "bid", "ask", "amt", "lowbid", "lowask", "highbid", "highask", "hightrade"):
                iData = iData*DTPriceMultiplier
            Data[iField] = iData
        return pd.Panel(Data).loc[fields]

class _security_information(FactorTable):
    """特征因子表"""
    def __init__(self, name, fdb=None, sys_args={}, config_file=None, **kwargs):
        self._FactorNames = []
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def FactorNames(self):
        if not self._FactorNames:
            SQLStr = "SELECT DetailInformation FROM security_information WHERE SecurityID=10001"
            self._FactorNames = sorted(json.loads(self._FactorDB.fetchall(SQLStr)[0][0]))
        return self._FactorNames
    def getID(self, ifactor_name=None, idt=None, args={}):
        SQLStr = "SELECT DISTINCT CAST(SecurityID AS CHAR) AS SecurityID FROM security_information ORDER BY SecurityID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        SQLStr = "SELECT DISTINCT CAST(SecurityID AS CHAR) AS SecurityID, DetailInformation FROM security_information "
        SQLStr += "WHERE ("+genSQLInCondition("SecurityID", ids, is_str=False, max_num=1000)+") "
        SQLStr += "ORDER BY SecurityID"
        RawData = {iID:json.loads(iInfo) for iID, iInfo in self._FactorDB.fetchall(SQLStr)}
        RawData = pd.DataFrame(RawData).T
        RawData.index.name = "ID"
        return RawData.loc[:, factor_names].reset_index()
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data = raw_data.set_index(["ID"])
        if raw_data.index.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        Data = pd.Panel(raw_data.values.T.reshape((raw_data.shape[1], raw_data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=raw_data.index, minor_axis=dts).swapaxes(1, 2)
        return Data.loc[:, :, ids]
class _security_index(FactorTable):
    """证券索引表"""
    def __init__(self, name, fdb=None, sys_args={}, config_file=None, **kwargs):
        self._FactorNames = []
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def FactorNames(self):
        return ["SecurityType", "SecuritySubType", "CFICode", "SecurityExchange", "Symbol", "Currency", "MaturityMonthYear", "MaturityDate", "MaturityTime", "StrikePrice", "StrikeCurrency", "PutOrCall", "UnderlyingSecurityID"]
    def getID(self, ifactor_name=None, idt=None, args={}):
        SQLStr = "SELECT DISTINCT CAST(SecurityID AS CHAR) AS SecurityID FROM security_index ORDER BY SecurityID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        SQLStr = "SELECT DISTINCT CAST(SecurityID AS CHAR) AS SecurityID, "+", ".join(factor_names)+" "
        SQLStr += "FROM security_index "
        SQLStr += "WHERE ("+genSQLInCondition("SecurityID", ids, is_str=False, max_num=1000)+") "
        SQLStr += "ORDER BY SecurityID"
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]: return pd.DataFrame(columns=["ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data = raw_data.set_index(["ID"])
        if raw_data.index.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        Data = pd.Panel(raw_data.values.T.reshape((raw_data.shape[1], raw_data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=raw_data.index, minor_axis=dts).swapaxes(1, 2)
        return Data.loc[:, :, ids]

class _market_data_daily(FactorTable):
    """日行情因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DataType = None
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        #if not self._DataType:
            #SQLStr = ("SELECT COLUMN_NAME, DATA_TYPE FROM market_data_daily.COLUMNS WHERE table_schema='%s' " % self._FactorDB.DBName)
            #SQLStr += ("AND TABLE_NAME = '%s' " % self.Name)
            #SQLStr += "AND COLUMN_NAME NOT IN ('SecurityID', 'TradeDate') "
            #SQLStr += "ORDER BY COLUMN_NAME"
            #Rslt = self._FactorDB.fetchall(SQLStr)
            #self._DataType = pd.DataFrame(np.array(Rslt), columns=["因子", "DataType"]).set_index(["因子"])["DataType"]
        #return self._DataType.index.tolist()
        return ["OpeningPx", "HighPx", "LowPx", "ClosingPx", "LowLimitPx", "HighLimitPx", "TotalVolumeTraded", "OpenInterest"]
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        SQLStr = "SELECT DISTINCT CAST(SecurityID AS CHAR) AS SecurityID FROM market_data_daily "
        if idt is not None: SQLStr += "WHERE TradeDate = '"+idt.strftime("%Y-%m-%d")+"' "
        SQLStr += "ORDER BY SecurityID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        SQLStr = "SELECT DISTINCT TradeDate FROM market_data_daily "
        if iid is not None: SQLStr += "WHERE SecurityID = "+iid+" "
        else: SQLStr += "WHERE SecurityID IS NOT NULL "
        if start_dt is not None: SQLStr += "AND TradeDate >= '"+start_dt.strftime("%Y-%m-%d")+"' "
        if end_dt is not None: SQLStr += "AND TradeDate <= '"+end_dt.strftime("%Y-%m-%d")+"' "
        SQLStr += "ORDER BY TradeDate"
        return [dt.datetime.combine(x[0], dt.time(0)) for x in self._FactorDB.fetchall(SQLStr)]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        # 形成 SQL 语句, 日期, ID, 因子数据
        SQLStr = "SELECT TradeDate, CAST(SecurityID AS CHAR) AS SecurityID, " + ", ".join(factor_names)+" FROM market_data_daily "
        SQLStr += "WHERE ("+genSQLInCondition("SecurityID", ids, is_str=False, max_num=1000)+") "
        SQLStr += "AND TradeDate >= '"+dts[0].strftime("%Y-%m-%d")+"' "
        SQLStr += "AND TradeDate <= '"+dts[-1].strftime("%Y-%m-%d")+"' "
        SQLStr += "ORDER BY SecurityID, TradeDate"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["日期", "ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {iFactorName: raw_data[iFactorName].unstack().astype("float") for iFactorName in raw_data.columns}
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime.combine(iDate, dt.time(0)) for iDate in Data.major_axis]
        return Data.loc[:, dts, ids]

class EpsilonDB(FactorDB):
    """epsilon 数据库"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("epsilon", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=1521, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"EpsilonDBConfig.json" if config_file is None else config_file), **kwargs)
        self._Connection = None# 数据库链接
        self._AllTables = []# 数据库中的所有表名, 用于查询时解决大小写敏感问题
        self._Suffix2Exchange = {"SH":"SSE", "SZ":"SZSE", "CFE":"CFFEX", "INE":"INE", "CZC":"CZCE", "DCE":"DCE", "SHF":"SHFE"}
        self.Name = "EpsilonDB"
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_Connection"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._Connection:
            self.connect()
        else:
            self._Connection = None
        self._AllTables = state.get("_AllTables", [])
    def connect(self):
        if (self.Connector=='cx_Oracle') or ((self.Connector=='default') and (self.DBType=='Oracle')):
            try:
                import cx_Oracle
                self._Connection = cx_Oracle.connect(self.User, self.Pwd, cx_Oracle.makedsn(self.IPAddr, str(self.Port), self.DBName))
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='pymssql') or ((self.Connector=='default') and (self.DBType=='SQL Server')):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='mysql.connector') or ((self.Connector=='default') and (self.DBType=='MySQL')):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                if self.Connector!='default': raise e
        else:
            if self.Connector not in ('default', 'pyodbc'):
                self._Connection = None
                raise __QS_Error__("不支持该连接器(connector) : "+self.Connector)
            else:
                import pyodbc
                if self.DSN:
                    self._Connection = pyodbc.connect('DSN=%s;PWD=%s' % (self.DSN, self.Pwd))
                else:
                    self._Connection = pyodbc.connect('DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s' % (self.DBType, self.DBName, self.IPAddr, self.User, self.Pwd))
        self._AllTables = []
        self._Connection.autocommit = True
        return 0
    def disconnect(self):
        if self._Connection is not None:
            try:
                self._Connection.close()
            except Exception as e:
                raise e
            finally:
                self._Connection = None
        return 0
    def isAvailable(self):
        return (self._Connection is not None)
    def cursor(self, sql_str=None):
        if self._Connection is None: raise __QS_Error__("%s尚未连接!" % self.__doc__)
        Cursor = self._Connection.cursor()
        if sql_str is None: return Cursor
        if not self._AllTables:
            if self.DBType=="SQL Server":
                Cursor.execute("SELECT Name FROM SysObjects Where XType='U'")
                self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
            elif self.DBType=="MySQL":
                Cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='"+self.DBName+"' AND table_type='base table'")
                self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
        for iTable in self._AllTables:
            sql_str = re.sub(iTable, iTable, sql_str, flags=re.IGNORECASE)
        Cursor.execute(sql_str)
        return Cursor
    def fetchall(self, sql_str):
        Cursor = self.cursor(sql_str=sql_str)
        Data = Cursor.fetchall()
        Cursor.close()
        return Data
    @property
    def TableNames(self):
        return ["security_information", "security_index", "market_data_daily"]
    def getTable(self, table_name, args={}):
        if table_name in self.TableNames: return eval("_"+table_name+"(name='"+table_name+"', fdb=self, sys_args=args)")
        raise __QS_Error__("因子库目前尚不支持表: '%s'" % table_name)
    # 创建证券定义
    def _createSecDef(self, ids):
        SecDef = []# DataFrame(columns=["SecurityExchange", "Symbol", "MaturityMonthYear"])
        for i, iID in enumerate(ids):
            iID = iID.split(".")
            iSuffix = iID[-1]
            iCode = ".".join(iID[:-1])
            iSecDef = [self._Suffix2Exchange[iSuffix]]
            if iCode.isnumeric():# 全是数字
                iSecDef.extend([iCode, None])
            elif iCode.isalnum():# 字母数字组合
                iSym, iNum = re.findall("\D+", iCode)[0], re.findall("\d+", iCode)[0]
                iSecDef.append(iSym)
                if len(iNum)==4: iSecDef.append("20%s" % (iNum, ))
                elif len(iNum)==3: iSecDef["MaturityMonthYear"] = ("201%s" % (iNum, ))
                else: raise __QS_Error__("ID: %s 解析失败!" % (ids[i], ))# TODO
            else: raise __QS_Error__("ID: %s 解析失败!" % (ids[i], ))# TODO
            SecDef.append(iSecDef)
        return pd.DataFrame(SecDef, columns=["SecurityExchange", "Symbol", "MaturityMonthYear"])
    # ID 转换成证券 ID
    def ID2SecurityID(self, ids):
        SecDef = self._createSecDef(ids=ids)
        SecurityExchanges, Symbols = set(SecDef["SecurityExchange"][pd.notnull(SecDef["SecurityExchange"])].values), set(SecDef["Symbol"][pd.notnull(SecDef["Symbol"])].values)
        MaturityMonthYears = set(SecDef["MaturityMonthYear"][pd.notnull(SecDef["MaturityMonthYear"])].values)
        SQLStr = "SELECT CAST(SecurityID AS CHAR) AS SecurityID, SecurityExchange, Symbol, MaturityMonthYear "
        SQLStr += "FROM security_index "
        SQLStr += "WHERE ("+genSQLInCondition("SecurityExchange", SecurityExchanges, is_str=True, max_num=1000)+") "
        SQLStr += "AND ("+genSQLInCondition("Symbol", Symbols, is_str=True, max_num=1000)+") "
        if MaturityMonthYears: SQLStr += "AND ("+genSQLInCondition("MaturityMonthYear", MaturityMonthYears, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY SecurityID"
        SecurityIDs = [None] * len(ids)
        Index = np.arange(len(ids))
        for iSecurityID, iExchange, iSymbol, iMaturityMonthYear in self.fetchall(SQLStr):
            iMask = ((SecDef["SecurityExchange"]==iExchange) & (SecDef["Symbol"]==iSymbol))
            if pd.notnull(iMaturityMonthYear): iMask = (iMask & (SecDef["MaturityMonthYear"]==iMaturityMonthYear))
            iIndex = Index[iMask.values]
            if iIndex.shape[0]==0: continue
            elif iIndex.shape[0]>1: raise __QS_Error__("Security ID 解析不唯一!")
            else: SecurityIDs[iIndex] = iSecurityID
        return SecurityIDs
        
            

if __name__=="__main__":
    import time
    CDB = CloverDB()
    CDB.connect()
    
    EDB = EpsilonDB()
    EDB.connect()
    
    
    #IDs = ["510050.SH", "IH.CFE", "IF1812.CFE"]
    #print(CDB.TableNames)
    #DTs = CDB.getTable("交易日历").getDateTime(start_dt=dt.datetime(2018,1,1), end_dt=dt.datetime(2018,1,5))
    #SecDef = CDB.createSecDef(ids=IDs)
    #SecInfo = CDB.getSecurityInfo(ids=IDs)
    #SecIDs = CDB.ID2SecurityID(ids=IDs)
    #TBs = CDB.getTimeBar(IDs, start_date=dt.date(2017,1,1), end_date=dt.date(2017,1,31), tms_intv=60000, depth=5)
    #RawData = CDB.getTable("Time Bar 数据")._getRawData(IDs, ["vol","amt"], dt.date(2018,1,1), dt.date(2018,1,5), 60000, 5)
    
    #FT = CDB.getTable("Tick 数据")
    #print(FT.FactorNames)
    #Data = FT.readData(factor_names=["lst"], ids=["510050.SH"])
    
    #FT = CDB.getTable("Time Bar 数据")
    #Data = FT.readData(factor_names=["mid", "amt"], ids=IDs, args={"时间间隔":60, "动态证券ID":False})
    
    
    #IDs = ["000001.SZ", "510050.SH", "SC1810.INE", "IF1812.CFE"]
    IDs = ["SC2112.INE", "SC1901.INE"]
    
    SecurityIDs = CDB.ID2SecurityID(IDs)
    
    #FT = EDB.getTable("security_information")
    #print(FT.FactorNames)
    #IDs = FT.getID()
    #DTs = FT.getDateTime()
    #Data = FT.readData(factor_names=FT.FactorNames, ids=IDs, dts=[dt.datetime.today()])
    
    FT = EDB.getTable("market_data_daily")
    print(FT.FactorNames)
    #IDs = FT.getID()
    DTs = FT.getDateTime()
    Data = FT.readData(factor_names=FT.FactorNames, ids=SecurityIDs, dts=DTs)
    Data.minor_axis = IDs
    
    
    
    EDB.disconnect()
    CDB.disconnect()