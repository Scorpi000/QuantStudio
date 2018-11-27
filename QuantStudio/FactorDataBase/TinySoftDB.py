# coding=utf-8
"""基于天软的因子库(TODO)"""
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str, Range, Directory, Password, Either, Int, Enum

from QuantStudio import __QS_Error__, __QS_LibPath__, __QS_MainPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5

class _TSTable(FactorTable):
    def getMetaData(self, key=None):
        TableInfo = self._FactorDB._TableInfo.loc[self.Name]
        if key is None: return TableInfo
        else: return TableInfo.get(key, None)
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        if key=="DataType":
            if hasattr(self, "_DataType"): return self._DataType.loc[factor_names]
            MetaData = FactorInfo["DataType"].loc[factor_names]
            for i in range(MetaData.shape[0]):
                iDataType = MetaData.iloc[i].lower()
                if (iDataType.find("real")!=-1) or (iDataType.find("int")!=-1): MetaData.iloc[i] = "double"
                else: MetaData.iloc[i] = "string"
            return MetaData
        elif key=="Description": return FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType"),
                                 "Description":self.getFactorMetaData(factor_names, key="Description")})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []

class _CalendarTable(FactorTable):
    """交易日历因子表"""
    @property
    def FactorNames(self):
        return ["交易日"]
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType": return pd.Series(["double"]*len(factor_names), index=factor_names)
        elif key=="Description": return pd.Series(["0 or nan: 非交易日; 1: 交易日"]*len(factor_names), index=factor_names)
        elif key is None:
            return pd.DataFrame({"DataType": self.getFactorMetaData(factor_names, key="DataType"),
                                 "Description": self.getFactorMetaData(factor_names, key="Description")})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 返回交易所列表
    def getID(self, ifactor_name=None, idt=None, args={}):
        return ["SSE", "SZSE"]
    # 返回交易所为 iid 的交易日列表
    # 如果 iid 为 None, 将返回表中有记录数据的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if start_dt is None: start_dt = dt.date(1900, 1, 1)
        if end_dt is None: end_dt = dt.date.today()
        CodeStr = "SetSysParam(pn_cycle(), cy_day());return MarketTradeDayQk(inttodate({StartDate}), inttodate({EndDate}));"
        CodeStr = CodeStr.format(StartDate=start_dt.strftime("%Y%m%d"), EndDate=end_dt.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        return list(map(lambda x: dt.datetime(*self.TSLPy.DecodeDate(x)), Data))
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Data = pd.DataFrame(1, index=self.getDateTime(start_dt=dts[0], end_dt=dts[-1]), columns=["SSE", "SZSE"])
        if Data.index.intersection(dts).shape[0]==0: return pd.Panel(np.nan, items=factor_names, major_axis=dts, minor_axis=ids)
        Data = Data.loc[dts, ids]
        return pd.Panel({"交易日": Data})
class _TradeTable(_TSTable):
    """tradetable"""
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if iid is None: iid = "000001.SH"
        if start_dt is None: start_dt = dt.datetime(1970, 1, 1)
        if end_dt is None: end_dt = dt.datetime.now()
        CodeStr = "return select "+"['date'] "
        CodeStr += "from tradetable datekey inttodate("+start_dt.strftime("%Y%m%d")+") "
        CodeStr += "to (inttodate("+end_dt.strftime("%Y%m%d")+")+0.9999) of '{ID}' end;"
        ErrorCode, Data, Msg = self._FactorDB.TSLPy.RemoteExecute(CodeStr.format(ID="".join(reversed(iid.split(".")))), {})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        DTs = np.array([dt.datetime(*self._FactorDB.TSLPy.DecodeDateTime(iData[b"date"])) for iData in Data], dtype="O")
        return DTs[(DTs>=start_dt) & (DTs<=end_dt)].tolist()
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        Fields = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[factor_names].tolist()
        CodeStr = "return select "+"['date'],['"+("'],['".join(Fields))+"'] "
        CodeStr += "from tradetable datekey inttodate("+dts[0].strftime("%Y%m%d")+") "
        CodeStr += "to (inttodate("+dts[-1].strftime("%Y%m%d")+")+0.9999) of '{ID}' end;"
        Data = {}
        for iID in ids:
            iCodeStr = CodeStr.format(ID="".join(reversed(iID.split("."))))
            ErrorCode, iData, Msg = self._FactorDB.TSLPy.RemoteExecute(iCodeStr, {})
            if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
            if iData: Data[iID] = pd.DataFrame(iData).set_index([b"date"])
        if not Data: return pd.Panel(Data)
        Data = pd.Panel(Data).swapaxes(0, 2)
        Data.major_axis = [dt.datetime(*self._FactorDB.TSLPy.DecodeDateTime(iDT)) for iDT in Data.major_axis]
        Data.items = [(iCol.decode("gbk") if isinstance(iCol, bytes) else iCol) for i, iCol in enumerate(Data.items)]
        Data = Data.loc[Fields]
        Data.items = factor_names
        return Data
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[2]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return raw_data.loc[:, dts, ids]
    def readDayData(self, factor_names, ids, start_date, end_date, args={}):
        RawData = self.__QS_prepareRawData__(factor_names, ids, dts=[start_date, end_date], args=args)
        if RawData.shape[2]==0: return pd.Panel(items=factor_names, major_axis=[], minor_axis=ids)
        return RawData.loc[:, :, ids]

class _MarketTable(_TSTable):
    """markettable"""
    Cycle = Either(Int(60), Enum("day", "week", "month", "quarter", "halfyear", "year"), arg_type="Integer", label="周期", order=0)
    CycleUnit = Enum("s", "d", arg_type="SingleOption", label="周期单位", order=1)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.Cycle = 60
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if iid is None: iid = "000001.SH"
        CycleStr = self._genCycleStr(args.get("周期", self.Cycle), args.get("周期单位", self.CycleUnit))
        if start_dt is None: start_dt = dt.datetime(1970, 1, 1)
        if end_dt is None: end_dt = dt.datetime.now()
        CodeStr = "SetSysParam(pn_cycle(),"+CycleStr+");"
        CodeStr += "return select "+"['date'] "
        CodeStr += "from markettable datekey inttodate("+start_dt.strftime("%Y%m%d")+") "
        CodeStr += "to (inttodate("+end_dt.strftime("%Y%m%d")+")+0.9999) of '{ID}' end;"
        ErrorCode, Data, Msg = self._FactorDB.TSLPy.RemoteExecute(CodeStr.format(ID="".join(reversed(iid.split(".")))), {})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        DTs = np.array([dt.datetime(*self._FactorDB.TSLPy.DecodeDateTime(iData[b"date"])) for iData in Data], dtype="O")
        return DTs[(DTs>=start_dt) & (DTs<=end_dt)].tolist()
    def _genCycleStr(self, cycle, cycle_unit):
        if isinstance(cycle, str): return "cy_"+cycle+"()"
        elif cycle_unit=="s": return ("cy_trailingseconds(%d)" % cycle)
        elif cycle_unit=="d": return ("cy_trailingdays(%d)" % cycle)
        else: raise __QS_Error__("不支持的和周期单位: '%s'!" % (cycle_unit, ))
    def __QS_genGroupInfo__(self, factors, operation_mode):
        CycleStrGroup = {}
        for iFactor in factors:
            iCycleStr = self._genCycleStr(iFactor.Cycle, iFactor.CycleUnit)
            if iCycleStr not in CycleStrGroup:
                CycleStrGroup[iCycleStr] = {"FactorNames":[iFactor.Name], 
                                            "RawFactorNames":{iFactor._NameInFT}, 
                                            "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                            "args":iFactor.Args.copy()}
            else:
                CycleStrGroup[iCycleStr]["FactorNames"].append(iFactor.Name)
                CycleStrGroup[iCycleStr]["RawFactorNames"].add(iFactor._NameInFT)
                CycleStrGroup[iCycleStr]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], CycleStrGroup[iCycleStr]["StartDT"])
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iCycleStr in CycleStrGroup:
            StartInd = operation_mode.DTRuler.index(CycleStrGroup[iCycleStr]["StartDT"])
            Groups.append((self, CycleStrGroup[iCycleStr]["FactorNames"], list(CycleStrGroup[iCycleStr]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], CycleStrGroup[iCycleStr]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        CycleStr = self._genCycleStr(args.get("周期", self.Cycle), args.get("周期单位", self.CycleUnit))
        Fields = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[factor_names].tolist()
        CodeStr = "SetSysParam(pn_cycle(),"+CycleStr+");"
        CodeStr += "return select "+"['date'],['"+"'],['".join(Fields)+"'] "
        CodeStr += "from markettable datekey inttodate("+dts[0].strftime("%Y%m%d")+") "
        CodeStr += "to (inttodate("+dts[-1].strftime("%Y%m%d")+")+0.9999) of '{ID}' end;"
        Data = {}
        for iID in ids:
            iCodeStr = CodeStr.format(ID="".join(reversed(iID.split("."))))
            ErrorCode, iData, Msg = self._FactorDB.TSLPy.RemoteExecute(iCodeStr, {})
            if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
            if iData: Data[iID] = pd.DataFrame(iData).set_index([b"date"])
        if not Data: return pd.Panel(Data)
        Data = pd.Panel(Data).swapaxes(0, 2)
        Data.major_axis = [dt.datetime(*self._FactorDB.TSLPy.DecodeDateTime(iDT)) for iDT in Data.major_axis]
        Data.items = [(iCol.decode("gbk") if isinstance(iCol, bytes) else iCol) for i, iCol in enumerate(Data.items)]
        Data = Data.loc[Fields]
        Data.items = factor_names
        return Data
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[2]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return raw_data.loc[:, dts, ids]
    def readDayData(self, factor_names, ids, start_date, end_date, args={}):
        RawData = self.__QS_prepareRawData__(factor_names, ids, dts=[start_date, end_date], args=args)
        if RawData.shape[2]==0: return pd.Panel(items=factor_names, major_axis=[], minor_axis=ids)
        return RawData.loc[:, :, ids]

class TinySoftDB(FactorDB):
    """TinySoft"""
    InstallDir = Directory(label="安装目录", arg_type="Directory", order=0)
    IPAddr = Str("tsl.tinysoft.com.cn", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=443, arg_type="Integer", label="端口", order=2)
    User = Str("Scorpio", arg_type="String", label="用户名", order=3)
    Pwd = Password("shuntai11", arg_type="String", label="密码", order=4)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"TinySoftDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "TinySoftDB"
        self.TSLPy = None
        self._TableInfo = None# 数据库中的表信息
        self._FactorInfo = None# 数据库中的表字段信息
        self._InfoFilePath = __QS_LibPath__+os.sep+"TinySoftDBInfo.hdf5"# 数据库信息文件路径
        self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"TinySoftDBInfo.xlsx"# 数据库信息源文件路径
        self._updateInfo()
        return
    def _updateInfo(self):
        if not os.path.isfile(self._InfoFilePath):
            print("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (self._InfoFilePath, self._InfoResourcePath))
        elif (os.path.getmtime(self._InfoResourcePath)>os.path.getmtime(self._InfoFilePath)):
            print("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % self._InfoResourcePath)
        else:
            try:
                self._TableInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/TableInfo")
                self._FactorInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/FactorInfo")
                return 0
            except:
                print("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (self._InfoFilePath, self._InfoResourcePath))
        if not os.path.isfile(self._InfoResourcePath): raise __QS_Error__("缺失数据库信息源文件: %s" % self._InfoResourcePath)
        self.importInfo(self._InfoResourcePath)
        self._TableInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/TableInfo")
        self._FactorInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/FactorInfo")
        return 0
    def __getstate__(self):
        state = self.__dict__.copy()
        state["TSLPy"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.TSLPy: self.connect()
        else: self.TSLPy = None
    def connect(self):
        if not (os.path.isdir(self.InstallDir)): raise __QS_Error__("TinySoft 的安装目录设置有误!")
        elif self.InstallDir not in sys.path: sys.path.append(self.InstallDir)
        import TSLPy3
        self.TSLPy = TSLPy3
        ErrorCode = self.TSLPy.ConnectServer(self.IPAddr, int(self.Port))
        if ErrorCode!=0:
            self.TSLPy = None
            raise __QS_Error__("TinySoft 服务器连接失败!")
        Rslt = self.TSLPy.LoginServer(self.User, self.Pwd)
        if Rslt is not None:
            ErrorCode, Msg = Rslt
            if ErrorCode!=0:
                self.TSLPy = None
                raise __QS_Error__("TinySoft 登录失败: "+Msg)
        else:
            raise __QS_Error__("TinySoft 登录失败!")
        return 0
    def disconnect(self):
        self.TSLPy.Disconnect()
        self.TSLPy = None
    def isAvailable(self):
        if self.TSLPy is not None:
            return self.TSLPy.Logined()
        else:
            return False
    @property
    def TableNames(self):
        if self._TableInfo is not None: return ["交易日历"]+self._TableInfo.index.tolist()
        else: return ["交易日历"]
    def getTable(self, table_name, args={}):
        if table_name=="交易日历": return _CalendarTable(name=table_name, fdb=self, sys_args=args)
        TableClass = self._TableInfo.loc[table_name, "TableClass"]
        return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=args)")
    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if exchange not in ("SSE", "SZSE"): raise __QS_Error__("不支持交易所: '%s' 的交易日序列!" % exchange)
        if start_date is None: start_date = dt.date(1900, 1, 1)
        if end_date is None: end_date = dt.date.today()
        CodeStr = "SetSysParam(pn_cycle(), cy_day());return MarketTradeDayQk(inttodate({StartDate}), inttodate({EndDate}));"
        CodeStr = CodeStr.format(StartDate=start_date.strftime("%Y%m%d"), EndDate=end_date.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        return list(map(lambda x: dt.date(*self.TSLPy.DecodeDate(x)), Data))
    # 获取指定日当前或历史上的全体 A 股 ID，返回在市场上出现过的所有A股, 目前仅支持提取当前的所有 A 股
    def _getAllAStock(self, date=None, is_current=True):# TODO
        if date is None: date = dt.date.today()
        CodeStr = "return getBK('深证A股;中小企业板;创业板;上证A股');"
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        IDs = []
        for iID in Data:
            iID = iID.decode("gbk")
            IDs.append(iID[2:]+"."+iID[:2])
        return IDs
    # 给定指数名称和ID，获取指定日当前或历史上的指数中的股票ID，is_current=True:获取指定日当天的ID，False:获取截止指定日历史上出现的ID, 目前仅支持提取当前的指数成份股
    def getID(self, index_id, date=None, is_current=True):# TODO
        if index_id=="全体A股": return self._getAllAStock(date=date, is_current=is_current)
        if date is None: date = dt.date.today()
        CodeStr = "return GetBKByDate('{IndexID}',IntToDate({Date}));"
        CodeStr = CodeStr.format(IndexID="".join(reversed(index_id.split("."))), Date=date.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        IDs = []
        for iID in Data:
            iID = iID.decode("gbk")
            IDs.append(iID[2:]+"."+iID[:2])
        return IDs
    # 将 Excel 文件中的表和字段信息导入信息文件
    def importInfo(self, excel_file_path):
        TableInfo = pd.read_excel(excel_file_path, "TableInfo").set_index(["TableName"])
        FactorInfo = pd.read_excel(excel_file_path, 'FactorInfo').set_index(['TableName', 'FieldName'])
        writeNestedDict2HDF5(TableInfo, self._InfoFilePath, "/TableInfo")
        writeNestedDict2HDF5(FactorInfo, self._InfoFilePath, "/FactorInfo")
        return 0