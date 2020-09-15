# coding=utf-8
"""基于天软的因子库(TODO)"""
import os
import sys
import json
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str, Range, Directory, File, Password, Either, Int, Enum, Dict

from QuantStudio import __QS_Error__, __QS_LibPath__, __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import updateInfo, SQL_InfoPublTable, SQL_MarketTable, SQL_MultiInfoPublTable

def _adjustID(self, ids):
    return pd.Series(ids, index=["".join(reversed(iID.split("."))) for iID in IDs])

class _TSTable(FactorTable):
    def getMetaData(self, key=None, args={}):
        TableInfo = self._FactorDB._TableInfo.loc[self.Name]
        if key is None: return TableInfo
        else: return TableInfo.get(key, None)
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
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
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args),
                                 "Description":self.getFactorMetaData(factor_names, key="Description", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []

class _CalendarTable(FactorTable):
    """交易日历因子表"""
    @property
    def FactorNames(self):
        return ["交易日"]
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType": return pd.Series(["double"]*len(factor_names), index=factor_names)
        elif key=="Description": return pd.Series(["0 or nan: 非交易日; 1: 交易日"]*len(factor_names), index=factor_names)
        elif key is None:
            return pd.DataFrame({"DataType": self.getFactorMetaData(factor_names, key="DataType", args=args),
                                 "Description": self.getFactorMetaData(factor_names, key="Description", args=args)})
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
        ErrorCode, Data, Msg = self._TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        return list(map(lambda x: dt.datetime(*self._TSLPy.DecodeDate(x)), Data))
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
        ErrorCode, Data, Msg = self._FactorDB._TSLPy.RemoteExecute(CodeStr.format(ID="".join(reversed(iid.split(".")))), {})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        DTs = np.array([dt.datetime(*self._FactorDB._TSLPy.DecodeDateTime(iData[b"date"])) for iData in Data], dtype="O")
        return DTs[(DTs>=start_dt) & (DTs<=end_dt)].tolist()
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        Fields = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[factor_names].tolist()
        CodeStr = "return select "+"['date'],['"+("'],['".join(Fields))+"'] "
        CodeStr += "from tradetable datekey inttodate("+dts[0].strftime("%Y%m%d")+") "
        CodeStr += "to (inttodate("+dts[-1].strftime("%Y%m%d")+")+0.9999) of '{ID}' end;"
        Data = {}
        for iID in ids:
            iCodeStr = CodeStr.format(ID="".join(reversed(iID.split("."))))
            ErrorCode, iData, Msg = self._FactorDB._TSLPy.RemoteExecute(iCodeStr, {})
            if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
            if iData: Data[iID] = pd.DataFrame(iData).set_index([b"date"])
        if not Data: return pd.Panel(Data)
        Data = pd.Panel(Data).swapaxes(0, 2)
        Data.major_axis = [dt.datetime(*self._FactorDB._TSLPy.DecodeDateTime(iDT)) for iDT in Data.major_axis]
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

class _QuoteTable(_TSTable):
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
        ErrorCode, Data, Msg = self._FactorDB._TSLPy.RemoteExecute(CodeStr.format(ID="".join(reversed(iid.split(".")))), {})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        DTs = np.array([dt.datetime(*self._FactorDB._TSLPy.DecodeDateTime(iData[b"date"])) for iData in Data], dtype="O")
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
            ErrorCode, iData, Msg = self._FactorDB._TSLPy.RemoteExecute(iCodeStr, {})
            if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
            if iData: Data[iID] = pd.DataFrame(iData).set_index([b"date"])
        if not Data: return pd.Panel(Data)
        Data = pd.Panel(Data).swapaxes(0, 2)
        Data.major_axis = [dt.datetime(*self._FactorDB._TSLPy.DecodeDateTime(iDT)) for iDT in Data.major_axis]
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

class _MarketTable(SQL_MarketTable):
    """行情因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix="", table_info=fdb._TableInfo, factor_info=fdb._FactorInfo, security_info=None, exchange_info=None, **kwargs)
        self._DTFormat = "%Y%m%d"
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DateField = "['"+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]+"']"
        SQLStr = "SELECT DISTINCT "+DateField+" "
        SQLStr += "FROM INFOTABLE "+self._DBTableName+" "
        if iid is not None:
            SQLStr += "OF ARRAY('"+_adjustID([iid]).index[0]+"') "
        else:
            raise ValueError("参数 iid 不能为空!")
        SQLStr += "WHERE "+DateField+"<>0 "
        if start_dt is not None: SQLStr += "AND "+DateField+">="+start_dt.strftime(self._DTFormat)+" "
        if end_dt is not None: SQLStr += "AND "+DateField+"<="+end_dt.strftime(self._DTFormat)+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY "+DateField+" END"
        Rslt = self._FactorDB.fetchall("RETURN exportjsonstring("+SQLStr+");")
        Rslt = pd.DataFrame(json.loads(Rslt.decode("gbk"))).iloc[:, 0]
        return Rslt.apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d")).tolist()
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        IDField = "['stockid']"
        IDMapping = _adjustID(ids)
        IDStr ="','".join(IDMapping.index)
        DateField = "['"+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]+"']"
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAX("+DateField+") AS 'MaxEndDate'"
        SubSQLStr += "FROM INFOTABLE "+self._DBTableName+" "
        SubSQLStr += "OF ARRAY('"+IDStr+"') "
        SubSQLStr += "WHERE "+DateField+"<"+end_date.strftime(self._DTFormat)+" "
        ConditionSQLStr = self._genConditionSQLStr(args=args)
        SubSQLStr += ConditionSQLStr+" "
        SubSQLStr += "GROUP BY "+IDField+" END"
        SQLStr = "SELECT [1]."+DateField+" AS '日期', "
        SQLStr += "[1]."+IDField+" AS 'ID', "
        for iField in factor_names: SQLStr += "[1].['"+self._FactorInfo.loc[iField, "DBFieldName"]+"'], "
        SQLStr += SQLStr[:-2]+" FROM INFOTABLE "+self._DBTableName+" "
        SQLStr += "OF ARRAY('"+IDStr+"') "
        SQLStr += "JOIN ("+SubSQLStr+") WITH ([1]."+IDField+", [1]."+DateField+" ON [2]."+IDField+", [2].['MaxEndDate']) END"
        TSLStr = "RETURN exportjsonstring("+SQLStr+");"
        RawData = json.loads(self._FactorDB.fetchall(TSLStr).decode("gbk"))
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID"]+factor_names)
        else:
            RawData = pd.DataFrame(RawData).loc[:, ["日期", "ID"]+factor_names]
            RawData["ID"] = IDMapping.loc[RawData["ID"].values].values
        return RawData
    def _genSQLStr(self, factor_names, ids, start_date, end_date, args={}):
        IDField = "['stockid']"
        IDMapping = _adjustID(ids)
        DateField = "['"+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]+"']"
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DateField+" AS '日期', "
        SQLStr += IDField+" AS 'ID', "
        for iField in factor_names: SQLStr += "['"+self._FactorInfo.loc[iField, "DBFieldName"]+"'], "
        SQLStr = SQLStr[:-2]+" FROM INFOTABLE "+self._DBTableName+" "
        SQLStr += "OF ARRAY('"+"','".join(IDMapping.index)+"') "
        SQLStr += "WHERE "+DateField+">="+start_date.strftime(self._DTFormat)+" "
        SQLStr += "AND "+DateField+"<="+end_date.strftime(self._DTFormat)+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID, "+DateField
        TSLStr = "RETURN exportjsonstring("+SQLStr+");"
        RawData = json.loads(self._FactorDB.fetchall(TSLStr).decode("gbk"))
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID"]+factor_names)
        else:
            RawData = pd.DataFrame(RawData).loc[:, ["日期", "ID"]+factor_names]
            RawData["ID"] = IDMapping.loc[RawData["ID"].values].values
        return RawData
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        LookBack = args.get("回溯天数", self.LookBack)
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        RawData = self._genSQLStr(factor_names, ids, start_date=StartDate, end_date=EndDate, args=args)
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["日期"]==dt.datetime.combine(StartDate,dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._genNullIDSQLStr(factor_names, list(NullIDs), StartDate, args=args)
                if NullRawData.shape[0]>0:
                    NullRawData = pd.DataFrame(NullRawData).loc[:, ["日期", "ID"]+factor_names]
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "日期"])
        return RawData


class _InfoPublTable(SQL_InfoPublTable):
    """信息发布表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    OnlyStartLookBack = Bool(False, label="只起始日回溯", arg_type="Bool", order=1)
    OnlyLookBackNontarget = Bool(False, label="只回溯非目标日", arg_type="Bool", order=2)
    OnlyLookBackDT = Bool(False, label="只回溯时点", arg_type="Bool", order=3)
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=4)
    IgnorePublDate = Bool(False, label="忽略公告日", arg_type="Bool", order=5)
    IgnoreTime = Bool(True, label="忽略时间", arg_type="Bool", order=6)
    EndDateASC = Bool(False, label="截止日期递增", arg_type="Bool", order=7)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        self._AnnDateField = self._FactorInfo["DBFieldName"][self._FactorInfo["FieldType"]=="AnnDate"]
        if self._AnnDateField.shape[0]>0: self._AnnDateField = self._AnnDateField.iloc[0]# 公告日期
        else: self._AnnDateField = None
    def getID(self, ifactor_name=None, idt=None, args={}):
        EndDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        if self._AnnDateField is None: AnnDateField = EndDateField
        else: AnnDateField = self._DBTableName+"."+self._AnnDateField
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr()+" "
        if args.get("忽略时间", self.IgnoreTime): DTFormat = "%Y-%m-%d"
        else: DTFormat = "%Y-%m-%d %H:%M:%S"
        if AnnDateField!=EndDateField:
            if idt is not None:
                SQLStr += "WHERE (CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END)='"+idt.strftime(DTFormat)+"' "
            else:
                SQLStr += "WHERE "+AnnDateField+" IS NOT NULL AND "+EndDateField+" IS NOT NULL "
        else:
            if idt is not None:
                SQLStr += "WHERE "+AnnDateField+"='"+idt.strftime(DTFormat)+"' "
            else:
                SQLStr += "WHERE "+AnnDateField+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if args.get("忽略公告日", self.IgnorePublDate) or (self._AnnDateField is None): return super().getDateTime(ifactor_name=ifactor_name, iid=iid, start_dt=start_dt, end_dt=end_dt, args=args)
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        EndDateField = self._DBTableName+"."+self._FactorDB._FactorInfo.loc[self.Name].loc[args.get("日期字段", self.DateField), "DBFieldName"]
        if self._AnnDateField is None: AnnDateField = EndDateField
        else: AnnDateField = self._DBTableName+"."+self._AnnDateField
        if AnnDateField!=EndDateField:
            if IgnoreTime:
                SQLStr = "SELECT DISTINCT STR_TO_DATE(CONCAT(DATE(CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END), ' ', TIME(0)), '%Y-%m-%d %H:%i:%s') AS DT "
            else:
                SQLStr = "SELECT DISTINCT CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END AS DT "
        else:
            if IgnoreTime:
                SQLStr = "SELECT DISTINCT STR_TO_DATE(CONCAT(DATE("+AnnDateField+"), ' ', TIME(0)), '%Y-%m-%d %H:%i:%s') AS DT "
            else:
                SQLStr = "SELECT DISTINCT "+AnnDateField+" AS DT "
        if iid is not None:
            SQLStr += self._genFromSQLStr()+" "
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+deSuffixID([iid])[0]+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else:
            SQLStr += "FROM "+self._DBTableName+" "
            SQLStr += "WHERE "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        if IgnoreTime: DTFormat = "%Y-%m-%d"
        else: DTFormat = "%Y-%m-%d %H:%M:%S"
        if AnnDateField!=EndDateField:
            if start_dt is not None:
                SQLStr += "AND ("+AnnDateField+">='"+start_dt.strftime(DTFormat)+"' "
                SQLStr += "OR "+EndDateField+">='"+start_dt.strftime(DTFormat)+"') "
            if end_dt is not None:
                SQLStr += "AND ("+AnnDateField+"<='"+end_dt.strftime(DTFormat)+"' "
                SQLStr += "AND "+EndDateField+"<='"+end_dt.strftime(DTFormat)+"') "
        else:
            if start_dt is not None:
                SQLStr += "AND "+AnnDateField+">='"+start_dt.strftime(DTFormat)+"' "
            if end_dt is not None:
                SQLStr += "AND "+AnnDateField+"<='"+end_dt.strftime(DTFormat)+"' "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY DT"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def _genNullIDSQLStr_InfoPubl(self, factor_names, ids, end_date, args={}):
        EndDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        if self._AnnDateField is None: AnnDateField = EndDateField
        else: AnnDateField = self._DBTableName+"."+self._AnnDateField
        SubSQLStr = "SELECT "+self._DBTableName+"."+self._IDField+", "
        SubSQLStr += "MAX("+EndDateField+") AS MaxEndDate "
        SubSQLStr += "FROM "+self._DBTableName+" "
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        if IgnoreTime: DTFormat = "%Y-%m-%d"
        else: DTFormat = "%Y-%m-%d %H:%M:%S"
        SubSQLStr += "WHERE ("+AnnDateField+"<'"+end_date.strftime(DTFormat)+"' "
        SubSQLStr += "AND "+EndDateField+"<'"+end_date.strftime(DTFormat)+"') "
        ConditionSQLStr = self._genConditionSQLStr(args=args)
        SubSQLStr += ConditionSQLStr+" "
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            if args.get("预筛选ID", self.PreFilterID):
                SubSQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+self._IDField, deSuffixID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SubSQLStr += "AND "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField
        if IgnoreTime:
            SQLStr = "SELECT DATE(CASE WHEN "+AnnDateField+">=t.MaxEndDate THEN "+AnnDateField+" ELSE t.MaxEndDate END) AS DT, "
        else:
            SQLStr = "SELECT CASE WHEN "+AnnDateField+">=t.MaxEndDate THEN "+AnnDateField+" ELSE t.MaxEndDate END AS DT, "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t."+self._IDField+"="+self._DBTableName+"."+self._IDField+" "
        SQLStr += "AND "+EndDateField+"=t.MaxEndDate) "
        if not ((self._MainTableName is None) or (self._MainTableName==self._DBTableName)):
            if args.get("预筛选ID", self.PreFilterID):
                SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        else:
            SQLStr += "WHERE TRUE "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += ConditionSQLStr
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if args.get("忽略公告日", self.IgnorePublDate) or (self._AnnDateField is None): return super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        if IgnoreTime: DTFormat = "%Y-%m-%d"
        else: DTFormat = "%Y-%m-%d %H:%M:%S"
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        LookBack = args.get("回溯天数", self.LookBack)
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        EndDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        AnnDateField = self._DBTableName+"."+self._AnnDateField
        SubSQLStr = "SELECT "+self._DBTableName+"."+self._IDField+", "
        if IgnoreTime:
            SubSQLStr += "DATE(CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END) AS AnnDate, "
        else:
            SubSQLStr += "CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END AS AnnDate, "
        SubSQLStr += "MAX("+EndDateField+") AS MaxEndDate "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "WHERE ("+AnnDateField+">='"+StartDate.strftime(DTFormat)+"' "
        SubSQLStr += "OR "+EndDateField+">='"+StartDate.strftime(DTFormat)+"') "
        SubSQLStr += "AND ("+AnnDateField+"<='"+EndDate.strftime(DTFormat)+"' "
        SubSQLStr += "AND "+EndDateField+"<='"+EndDate.strftime(DTFormat)+"') "
        ConditionSQLStr = self._genConditionSQLStr(args=args)
        SubSQLStr += ConditionSQLStr+" "
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            if args.get("预筛选ID", self.PreFilterID):
                SubSQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+self._IDField, deSuffixID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SubSQLStr += "AND "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        if IgnoreTime:
            SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField+", DATE(AnnDate)"
        else:
            SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField+", AnnDate"
        SQLStr = "SELECT t.AnnDate AS DT, "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t."+self._IDField+"="+self._DBTableName+"."+self._IDField+") "
        SQLStr += "AND (t.MaxEndDate="+EndDateField+") "
        if not ((self._MainTableName is None) or (self._MainTableName==self._DBTableName)):
            if args.get("预筛选ID", self.PreFilterID):
                SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        else:
            SQLStr += "WHERE TRUE "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += ConditionSQLStr+" "
        SQLStr += "ORDER BY ID, DT"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID", "MaxEndDate"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID", "MaxEndDate"]+factor_names)
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["日期"]==dt.datetime.combine(StartDate,dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr_InfoPubl(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["日期", "ID", "MaxEndDate"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "日期"])
        if RawData.shape[0]==0: return RawData.loc[:, ["日期", "ID"]+factor_names]
        if args.get("截止日期递增", self.EndDateASC):# 删除截止日期非递增的记录
            DTRank = RawData.loc[:, ["ID", "日期", "MaxEndDate"]].set_index(["ID"]).astype(np.datetime64).groupby(axis=0, level=0).rank(method="min")
            RawData = RawData[(DTRank["日期"]<=DTRank["MaxEndDate"]).values]
        RawData = RawData.loc[:, ["日期", "ID"]+factor_names]
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData


class _MultiInfoPublTable(SQL_MultiInfoPublTable):
    """多重信息发布表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    OnlyStartLookBack = Bool(False, label="只起始日回溯", arg_type="Bool", order=1)
    OnlyLookBackNontarget = Bool(False, label="只回溯非目标日", arg_type="Bool", order=2)
    OnlyLookBackDT = Bool(False, label="只回溯时点", arg_type="Bool", order=3)
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=4)
    IgnorePublDate = Bool(False, label="忽略公告日", arg_type="Bool", order=5)
    IgnoreTime = Bool(True, label="忽略时间", arg_type="Bool", order=6)
    EndDateASC = Bool(False, label="截止日期递增", arg_type="Bool", order=7)
    Operator = Either(Function(None), None, arg_type="Function", label="算子", order=8)
    OperatorDataType = Enum("object", "double", "string", arg_type="SingleOption", label="算子数据类型", order=9)
    OrderFields = List(arg_type="List", label="排序字段", order=10)# [("字段名", "ASC" 或者 "DESC")]
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "回溯天数", "只起始日回溯", "只回溯非目标日", "只回溯时点", "算子", "算子数据类型")
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if key=="DataType":
            if factor_names is None: factor_names = self.FactorNames
            if args.get("算子", self.Operator) is None:
                return pd.Series(["object"]*len(factor_names), index=factor_names)
            else:
                return pd.Series([args.get("算子数据类型", self.OperatorDataType)]*len(factor_names), index=factor_names)
        else:
            return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        OrderFields = args.get("排序字段", self.OrderFields)
        if OrderFields:
            OrderFields, Orders = np.array(OrderFields).T.tolist()
        else:
            OrderFields, Orders = [], []
        FactorNames = list(set(factor_names).union(OrderFields))
        RawData = super().__QS_prepareRawData__(FactorNames, ids, dts, args=args)
        RawData = RawData.sort_values(by=["ID", "日期"]+OrderFields, ascending=[True, True]+[(iOrder.lower()=="asc") for iOrder in Orders])
        return RawData.loc[:, ["日期", "ID"]+factor_names]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Operator = args.get("算子", self.Operator)
        if Operator is None: Operator = (lambda x: x.tolist())
        if args.get("只回溯时点", self.OnlyLookBackDT):
            DeduplicatedIndex = raw_data.index(~raw_data.index.duplicated())
            RowIdxMask = pd.Series(False, index=DeduplicatedIndex).unstack(fill_value=True).astype(bool)
            RawIDs = RowIdxMask.columns
            if RawIDs.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
            RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
            RowIdx[RowIdxMask] = np.nan
            RowIdx = adjustDataDTID(pd.Panel({"RowIdx": RowIdx}), args.get("回溯天数", self.LookBack), ["RowIdx"], RowIdx.columns.tolist(), dts, 
                                              args.get("只起始日回溯", self.OnlyStartLookBack), 
                                              args.get("只回溯非目标日", self.OnlyLookBackNontarget), 
                                              logger=self._QS_Logger).iloc[0].values
            RowIdx[pd.isnull(RowIdx)] = -1
            RowIdx = RowIdx.astype(int)
            ColIdx = np.arange(RowIdx.shape[1]).reshape((1, RowIdx.shape[1])).repeat(RowIdx.shape[0], axis=0)
            RowIdxMask = (RowIdx==-1)
            Data = {}
            for iFactorName in factor_names:
                iRawData = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
                iRawData = iRawData.values[RowIdx, ColIdx]
                iRawData[RowIdxMask] = None
                Data[iFactorName] = pd.DataFrame(iRawData, index=dts, columns=RawIDs)
            return pd.Panel(Data).loc[factor_names, :, ids]
        else:
            Data = {}
            for iFactorName in factor_names:
                Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
            Data = pd.Panel(Data).loc[factor_names, :, ids]
            return adjustDataDTID(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts, 
                                  args.get("只起始日回溯", self.OnlyStartLookBack), 
                                  args.get("只回溯非目标日", self.OnlyLookBackNontarget),
                                  logger=self._QS_Logger)


class TinySoftDB(FactorDB):
    """TinySoft"""
    InstallDir = Directory(label="安装目录", arg_type="Directory", order=0)
    IPAddr = Str("tsl.tinysoft.com.cn", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=443, arg_type="Integer", label="端口", order=2)
    User = Str("", arg_type="String", label="用户名", order=3)
    Pwd = Password("", arg_type="String", label="密码", order=4)
    DBInfoFile = File(label="库信息文件", arg_type="File", order=100)
    FTArgs = Dict(label="因子表参数", arg_type="Dict", order=101)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"TinySoftDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "TinySoftDB"
        self._TSLPy = None
        self._TableInfo = None# 数据库中的表信息
        self._FactorInfo = None# 数据库中的表字段信息
        self._InfoFilePath = __QS_LibPath__+os.sep+"TinySoftDBInfo.hdf5"# 数据库信息文件路径
        self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"TinySoftDBInfo.xlsx"# 数据库信息源文件路径
        self._TableInfo, self._FactorInfo = updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger)
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_TSLPy"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._TSLPy: self.connect()
        else: self._TSLPy = None
    def connect(self):
        if not (os.path.isdir(self.InstallDir)): raise __QS_Error__("TinySoft 的安装目录设置有误!")
        elif self.InstallDir not in sys.path: sys.path.append(self.InstallDir)
        import TSLPy3
        self._TSLPy = TSLPy3
        ErrorCode = self._TSLPy.ConnectServer(self.IPAddr, int(self.Port))
        if ErrorCode!=0:
            self._TSLPy = None
            raise __QS_Error__("TinySoft 服务器连接失败!")
        Rslt = self._TSLPy.LoginServer(self.User, self.Pwd)
        if Rslt is not None:
            ErrorCode, Msg = Rslt
            if ErrorCode!=0:
                self._TSLPy = None
                raise __QS_Error__("TinySoft 登录失败: "+Msg)
        else:
            raise __QS_Error__("TinySoft 登录失败!")
        return 0
    def disconnect(self):
        self._TSLPy.Disconnect()
        self._TSLPy = None
    def isAvailable(self):
        if self._TSLPy is not None:
            return self._TSLPy.Logined()
        else:
            return False
    def fetchall(self, tsl_str, sys_param={}):
        ErrorCode, Data, Msg = self._TSLPy.RemoteExecute(tsl_str, sys_param)
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误, Error Code: %d, 错误信息: %s" % (ErrorCode, Msg.decode("gbk")))
        return Data
    @property
    def TableNames(self):
        if self._TableInfo is not None: return self._TableInfo[pd.notnull(self._TableInfo["TableClass"])].index.tolist()
        else: return []
    def getTable(self, table_name, args={}):
        if table_name in self._TableInfo.index:
            TableClass = args.get("因子表类型", self._TableInfo.loc[table_name, "TableClass"])
            if pd.notnull(TableClass) and (TableClass!=""):
                Args = self.FTArgs.copy()
                Args.update(args)
                return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=Args, logger=self._QS_Logger)")
        Msg = ("因子库 ‘%s' 目前尚不支持因子表: '%s'" % (self.Name, table_name))
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)
    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if exchange not in ("SSE", "SZSE"): raise __QS_Error__("不支持交易所: '%s' 的交易日序列!" % exchange)
        if start_date is None: start_date = dt.date(1900, 1, 1)
        if end_date is None: end_date = dt.date.today()
        CodeStr = "SetSysParam(pn_cycle(), cy_day());return MarketTradeDayQk(inttodate({StartDate}), inttodate({EndDate}));"
        CodeStr = CodeStr.format(StartDate=start_date.strftime("%Y%m%d"), EndDate=end_date.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self._TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        return list(map(lambda x: dt.date(*self._TSLPy.DecodeDate(x)), Data))
    # 获取指定日当前或历史上的全体 A 股 ID，返回在市场上出现过的所有A股, 目前仅支持提取当前的所有 A 股
    def _getAllAStock(self, date=None, is_current=True):# TODO
        if date is None: date = dt.date.today()
        CodeStr = "return getBK('深证A股;中小企业板;创业板;上证A股');"
        ErrorCode, Data, Msg = self._TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        IDs = []
        for iID in Data:
            iID = iID.decode("gbk")
            IDs.append(iID[2:]+"."+iID[:2])
        return IDs
    # 给定指数 ID, 获取指定日当前或历史上的指数中的股票 ID, is_current=True:获取指定日当天的 ID, False:获取截止指定日历史上出现的 ID, 目前仅支持提取当前的指数成份股
    def getStockID(self, index_id, date=None, is_current=True):# TODO
        if index_id=="全体A股": return self._getAllAStock(date=date, is_current=is_current)
        if date is None: date = dt.date.today()
        CodeStr = "return GetBKByDate('{IndexID}',IntToDate({Date}));"
        CodeStr = CodeStr.format(IndexID="".join(reversed(index_id.split("."))), Date=date.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self._TSLPy.RemoteExecute(CodeStr, {})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        IDs = []
        for iID in Data:
            iID = iID.decode("gbk")
            IDs.append(iID[2:]+"."+iID[:2])
        return IDs
    # 给定期货 ID, 获取指定日当前或历史上的该期货的所有 ID, is_current=True:获取指定日当天的 ID, False:获取截止指定日历史上出现的 ID, 目前仅支持提取当前在市的 ID
    def getFutureID(self, future_code="IF", date=None, is_current=True):
        if date is None: date = dt.date.today()
        if is_current: CodeStr = "EndT:= {Date}T;return GetFuturesID('{FutureID}', EndT);"
        else: raise __QS_Error__("目前不支持提取历史 ID")
        CodeStr = CodeStr.format(FutureID="".join(future_code.split(".")), Date=date.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self._TSLPy.RemoteExecute(CodeStr, {})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        return [iID.decode("gbk") for iID in Data]