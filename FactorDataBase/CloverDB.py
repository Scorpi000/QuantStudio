# coding=utf-8
"""clover 数据源"""
import os
import re
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import File, List, Float, Int, Bool, Enum, Str, Range, Password

from QuantStudio import __QS_Error__, __QS_LibPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable, _adjustDateTime

# tms: 时间戳, 单位: 毫秒; date: 日期; update: 最新的采样区间内有多少个行情点;
# lst: 最新价; vol: 最新成交量; amt: 最新成交额; oin: 最新持仓量;
# bid: 买一价; bsz: 买一量; ask: 卖一价; asz: 卖一量; 
# mid: 中间价, bid 和ask 的均值; wmp: bid 和 ask 的以 bsz 和 asz 的加权均值
# buyVolume: 确定为买方发起成交的成交量; sellVolume: 确定为卖方发起成交的成交量
class _TickTable(FactorTable):
    """Clover Tick 因子表"""
    TmsIntv = Float(3, arg_type="Double", label="时间间隔", order=0, low=0.5, high=86400, single_step=0.5)
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = name
        self._FactorNames = ["lst","vol","amt","mid","bid","bsz","ask","asz","wmp","oin",
                             "buyVolume","sellVolume","bidSizeChange","askSizeChange",
                             "crossBidVolume","crossAskVolume","vi","ofi"]
        self._PriceFactors = {"lst", "amt", "mid", "bid", "ask", "wmp"}
    @property
    def FactorNames(self):
        return self._FactorNames
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        MetaData = pd.DataFrame("double", index=factor_names, columns=["DataType"], dtype=np.dtype("O"))
        if key is None: return MetaData
        elif key in MetaData: return MetaData.ix[:, key]
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
     # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 时间间隔: 以秒为单位, 默认是 3 秒
    def __QS_readData__(self, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if ids is None: raise __QS_Error__("请指定提取数据的 ID 序列!")
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = dt.date.today()-dt.timedelta(365), dt.date.today()
        SecDef = self.FactorDB.createSecDef(ids)
        tms_intv = int(args.get("时间间隔", self.TmsIntv)*1000)
        StartDate = StartDate.strftime("%Y-%m-%d")
        EndDate = (EndDate + dt.timedelta(1)).strftime("%Y-%m-%d")
        if self._PriceFactors.intersection(set(factor_names)):
            TradeDates = self.FactorDB._jpype.JPackage('clover.epsilon.database').DatabaseUtil.getTradeDateList(self.FactorDB._jConn, 'SHFE', StartDate, EndDate)
            if len(TradeDates)==0:
                return _adjustDateTime(pd.Panel(Data, major_axis=tms, minor_axis=ids), dts=dts, start_dt=start_dt, end_dt=end_dt)
            JPckg_util = self.FactorDB._jpype.JPackage('clover.model.util')
            PriceMultiplier = np.ones(len(SecDef))
            for j, jSecDef in enumerate(SecDef):
                jSecurity = JPckg_util.ModelUtils.parseSecurity(self.FactorDB._jConn, TradeDates[-1], JPckg_util.Json.parse(jSecDef))
                PriceMultiplier[j] = jSecurity.getMinPriceIncrement()
        mdreader = self.FactorDB._jpype.JPackage('clover.epsilon.util').TestUtils.createMDReader('JavaSerialPT', 'mdreader.SHFE')
        JPckg_matlab = self.FactorDB._jpype.JPackage('clover.model.matlab')
        sdm_def = JPckg_matlab.SecurityDataMatrixPT.generateSDMDataDef_CN(self.FactorDB._jConn, SecDef, StartDate, EndDate, tms_intv)
        sdm = JPckg_matlab.SecurityDataMatrixPT()
        sdm.fetchData(sdm_def, mdreader, tms_intv)
        tms = np.array(sdm.tms)/1000
        tms = [dt.datetime.fromtimestamp(iTMS) for iTMS in tms]
        Data = {}
        for iField in factor_names:
            iData = np.array(getattr(sdm, iField))
            if iField in self._PriceFactors:
                iData = iData * PriceMultiplier
            Data[iField] = iData
        Data = pd.Panel(Data, major_axis=tms, minor_axis=ids).loc[factor_names]
        return _adjustDateTime(Data, dts=dts)

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
            return MetaData.ix[:, key]
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
     # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 时间间隔: 以秒为单位, 默认是 3 秒
    def __QS_readData__(self, factor_names=None, ids=None, dts=None, args={}):
        if ids is None: raise __QS_Error__("请指定提取数据的 ID 序列!")
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = dt.date.today()-dt.timedelta(1), dt.date.today()
        if factor_names is None: factor_names = self.FactorNames
        TBs, AllSecurityIDs, SecurityIDs, PriceMultiplier = self.FactorDB.getTimeBar(ids, StartDate, EndDate, 
                                                                                      tms_intv=args.get("时间间隔", self.TmsIntv)*1000, 
                                                                                      depth=args.get("深度", self.Depth), 
                                                                                      dynamic_security_id=args.get("动态证券ID", self.DynamicSecID))
        Data = self.FactorDB.fetchTimeBarData(factor_names, TBs, AllSecurityIDs, SecurityIDs, PriceMultiplier)
        return _adjustDateTime(Data, dts=dts)

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
        elif key in MetaData: return MetaData.ix[:, key]
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def __QS_readData__(self, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if ids is None: ids = ["000001.SH"]
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = dt.date.today()-dt.timedelta(365), dt.date.today()
        Data = {}
        for i in range((EndDate-StartDate).days):
            iDT = dt.datetime.combine(StartDate + dt.timedelta(i), dt.time(23, 59, 59, 999999))
            Data[iDT] = pd.DataFrame(self.FactorDB.getSecurityInfo(ids, idt=iDT), index=ids).ix[:, factor_names].T
        Data = pd.Panel(Data).swapaxes(0, 1)
        Data = _adjustDateTime(Data, dts, fillna=True, method="bfill")
        return Data

class CloverDB(FactorDB):
    """Clover 数据库"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("epsilon", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("192.168.100.2", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("epsilon", arg_type="String", label="用户名", order=4)
    Pwd = Password("epsilon7777", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    JVMPath = File("", arg_type="SingleOption", label="Java虚拟机", order=0, filter=["DLL (*.dll)"])
    JavaPckg = List(File("", filter=["Java Package (*.jar)"]), arg_type="ArgList", label="Java包", order=1)
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        # 继承来的属性
        self.Name = "CloverDB"
        self._jpype = None# jpype 模块
        self._jConn = None# epsilon 数据库连接
        self._ExchangeDict = {"上海证券交易所":"SHFE", "深圳证券交易所":"SZSE"}
        return
    def __QS_initArgs__(self):
        self.JVMPath = "C:\\Program Files\\Java\\jdk1.8.0_172\\jre\\bin\\server\\jvm.dll"
        self.JavaPckg.append("X:/java_lib/model-jar-with-dependencies.jar")
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_jpype"] = (True if self.isAvailable() else False)
        state["_jConn"] = state["_jpype"]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        isConnected = self._jpype
        self._jpype, self._jConn = None, None
        if isConnected:
            self.connect()
    def connect(self):
        if self._jpype is None:
            if not (os.path.isfile(self.JVMPath)):
                raise __QS_Error__("Java 虚拟机的设置有误!")
            import jpype
            self._jpype = jpype
        if not self._jpype.isJVMStarted():
            JOption = ["-Xmx64g", "-Xms16g"]
            if self.JavaPckg: JOption.append("-Djava.class.path="+";".join(self.JavaPckg))
            self._jpype.startJVM(self.JVMPath, *JOption)
            jpckg = self._jpype.JPackage('java.sql')
            ConnStr = "jdbc:mysql://"+self.IPAddr+":"+str(self.Port)+"/"+self.DBName
            self._jConn = jpckg.DriverManager.getConnection(ConnStr, self.User, self.Pwd)
        return 0
    def disconnect(self):
        self._jConn = None
        self._jpype.shutdownJVM()
        self._jpype = None
        return 0
    def isAvailable(self):
        if self._jpype is not None:
            return self._jpype.isJVMStarted()
        else:
            return False
    @property
    def TableNames(self):
        return ["Tick 数据", "Time Bar 数据", "证券特征"]
    def getTable(self, table_name, args={}):
        if table_name=="Tick 数据":
            FT = _TickTable(table_name, sys_args=args)
        elif table_name=="Time Bar 数据":
            FT = _TimeBarTable(table_name, sys_args=args)
        elif table_name=="证券特征":
            FT = _FeatureTable(table_name, sys_args=args)
        FT.FactorDB = self
        return FT
    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="上海证券交易所"):
        if exchange not in ("上海证券交易所", "深圳证券交易所"):
            raise __QS_Error__("不支持交易所：%s的交易日序列！" % exchange)
        exchange = self._ExchangeDict["上海证券交易所"]
        if start_date is None:
            start_date = dt.date(1900,1,1)
        start_date = start_date.strftime("%Y-%m-%d")
        if end_date is None:
            end_date = dt.date.today()
        end_date = (end_date + dt.timedelta(1)).strftime("%Y-%m-%d")
        jpckg = self._jpype.JPackage('clover.epsilon.database')
        Dates = jpckg.DatabaseUtil.getTradeDateList(self._jConn, exchange, start_date, end_date)
        Dates = np.array(Dates)
        return np.array([dt.datetime.strptime(iDate, "%Y-%m-%d").date() for iDate in Dates])
    # 创建证券定义
    def createSecDef(self, ids, idate=dt.date.today()):
        SecDef = []
        for i, iID in enumerate(ids):
            iID = iID.split(".")
            iSuffix = iID[-1]
            iCode = ".".join(iID[:-1])
            if iSuffix=="SH":
                SecDef.append('{"SecurityExchange":"SSE", "Symbol":"%s"}' % iCode)
            elif iSuffix=="SZ":
                SecDef.append('{"SecurityExchange":"SZSE", "Symbol":"%s"}' % iCode)
            elif iSuffix=="CFE":
                if iCode.isalpha():# 全是字母
                    SecDef.append('{"SecurityExchange":"CFFEX", "Symbol":"%s", "MaturitySequence":"+1"}' % (iCode[:2], ))
                elif iCode.isalnum():# 字母数字组合
                    iSym = re.findall("\D+")[0]
                    iNum = re.findall("\d+")[0]
                    if len(iNum)==4:
                        SecDef.append('{"SecurityExchange":"CFFEX", "Symbol":"%s", "MaturityMonthYear":"20%d"}' % (iSym, int(iNum)))
                    else:
                        raise __QS_Error__("ID: %s 解析失败!" % (ids[i], ))# TODO
                else:
                    raise __QS_Error__("ID: %s 解析失败!" % (ids[i], ))
        return SecDef
    # 获取证券的描述信息
    def getSecurityInfo(self, ids, idt=None):
        if idt is None:
            idt = dt.date.today()
        idt = idt.strftime("%Y-%m-%d")
        sec_def = self.createSecDef(ids)
        SecInfo = []
        jpckg = self._jpype.JPackage('clover.model.util')
        for iSecDef in sec_def:
            iSecInfo = jpckg.ModelUtils.parseSecurity(self._jConn, idt, jpckg.Json.parse(iSecDef))
            SecInfo.append(eval(iSecInfo.toString()))
        return SecInfo
    # ID 转换成证券 ID
    def ID2SecurityID(self, ids, idt=None):
        SecInfo = self.getSecurityInfo(ids, idt=idt)
        return [iSecInfo["SecurityID"] for iSecInfo in SecInfo]
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

if __name__=="__main__":
    import time
    CDB = CloverDB()
    CDB.connect()
    
    IDs = ["510050.SH", "IH.CFE"]
    #IDs = ["IH1701.CFE", "IH1702.CFE"]
    #print(CDB.TableNames)
    #Dates = CDB.getTradeDay(start_date=dt.date(2018,1,1), end_date=dt.date(2018,1,5))
    #SecDef = CDB.createSecDef(ids=IDs)
    #SecInfo = CDB.getSecurityInfo(ids=IDs)
    #SecIDs = CDB.ID2SecurityID(ids=IDs)
    #TBs = CDB.getTimeBar(IDs, start_date=dt.date(2017,1,1), end_date=dt.date(2017,1,31), tms_intv=60000, depth=5)
    #RawData = CDB.getTable("Time Bar 数据")._getRawData(IDs, ["vol","amt"], dt.date(2018,1,1), dt.date(2018,1,5), 60000, 5)
    
    FT = CDB.getTable("Tick 数据")
    print(FT.FactorNames)
    Data = FT.readData(factor_names=["lst"], ids=["510050.SH"])
    
    #FT = CDB.getTable("Time Bar 数据")
    ##print(FT.FactorNames)
    #Data = FT.readData(factor_names=["mid", "amt"], ids=IDs, args={"时间间隔":60, "动态证券ID":False})
    CDB.disconnect()