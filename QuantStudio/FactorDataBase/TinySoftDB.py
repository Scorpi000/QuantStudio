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
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import updateInfo, importInfo, SQL_Table, SQL_WideTable, SQL_FeatureTable, SQL_MappingTable
from QuantStudio.Tools.api import Panel

def _adjustID(ids):
    return pd.Series(ids, index=["".join(reversed(iID.split("."))) for iID in ids])

class _TSTable(FactorTable):
    FilterCondition = Str("", arg_type="Dict", label="筛选条件", order=100)
    TableType = Str("", arg_type="SingleOption", label="因子表类型", order=200)
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
        if Data.index.intersection(dts).shape[0]==0: return Panel(np.nan, items=factor_names, major_axis=dts, minor_axis=ids)
        Data = Data.loc[dts, ids]
        return Panel({"交易日": Data})
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
        if not Data: return Panel(Data)
        Data = Panel(Data).swapaxes(0, 2)
        Data.major_axis = [dt.datetime(*self._FactorDB._TSLPy.DecodeDateTime(iDT)) for iDT in Data.major_axis]
        Data.items = [(iCol.decode("gbk") if isinstance(iCol, bytes) else iCol) for i, iCol in enumerate(Data.items)]
        Data = Data.loc[Fields]
        Data.items = factor_names
        return Data
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[2]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return raw_data.loc[:, dts, ids]
    def readDayData(self, factor_names, ids, start_date, end_date, args={}):
        RawData = self.__QS_prepareRawData__(factor_names, ids, dts=[start_date, end_date], args=args)
        if RawData.shape[2]==0: return Panel(items=factor_names, major_axis=[], minor_axis=ids)
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
        if not Data: return Panel(Data)
        Data = Panel(Data).swapaxes(0, 2)
        Data.major_axis = [dt.datetime(*self._FactorDB._TSLPy.DecodeDateTime(iDT)) for iDT in Data.major_axis]
        Data.items = [(iCol.decode("gbk") if isinstance(iCol, bytes) else iCol) for i, iCol in enumerate(Data.items)]
        Data = Data.loc[Fields]
        Data.items = factor_names
        return Data
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[2]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return raw_data.loc[:, dts, ids]
    def readDayData(self, factor_names, ids, start_date, end_date, args={}):
        RawData = self.__QS_prepareRawData__(factor_names, ids, dts=[start_date, end_date], args=args)
        if RawData.shape[2]==0: return Panel(items=factor_names, major_axis=[], minor_axis=ids)
        return RawData.loc[:, :, ids]

class _TS_SQL_Table(SQL_Table):
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix="", table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)
        self._DBTableName = "[1]"
        self._MainTableName = self._DBTableName
        self._DTFormat = "%Y%m%d"
    def __QS_adjustID__(self, ids):
        return ["".join(reversed(iID.split("."))) for iID in ids]
    def __QS_restoreID__(self, ids):
        return ids
    def _genFromSQLStr(self, setable_join_str=[]):
        SQLStr = "FROM INFOTABLE "+str(int(self._TableInfo["DBTableName"]))+" "
        return SQLStr[:-1]
    def _genIDSQLStr(self, ids, init_keyword="AND", args={}):
        if ids is None:
            raise __QS_Error__("TinysoftDB 的因子表方法参数 ids 不能为 None")
        SQLStr = "OF ARRAY('"+"','".join(self.__QS_adjustID__(ids))+"')"
        return SQLStr
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []

class _WideTable(_TS_SQL_Table, SQL_WideTable):
    """宽因子表"""
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTField = self._FactorInfo.loc[args.get("时点字段", self.DTField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+DTField+" "
        if iid is not None:
            SQLStr += self._genFromSQLStr()+" "
            SQLStr += self._genIDSQLStr([iid], init_keyword="WHERE", args=args)+" "
            SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        else:
            raise __QS_Error__("TinysoftDB 的因子表方法 getDateTime 参数 iid 不能为 None")
        if start_dt is not None: SQLStr += "AND "+DTField+">="+start_dt.strftime(self._DTFormat)+" "
        if end_dt is not None: SQLStr += "AND "+DTField+"<="+end_dt.strftime(self._DTFormat)+" "
        SQLStr += "ORDER BY "+DTField+" END"
        Rslt = self._FactorDB.fetchall("RETURN exportjsonstring("+SQLStr+");")
        Rslt = pd.DataFrame(json.loads(Rslt.decode("gbk"))).iloc[:, 0]
        return Rslt.apply(lambda x: dt.datetime.strptime(str(x), self._DTFormat)).tolist()
    def _genNullIDSQLStr_WithPublDT(self, factor_names, ids, end_date, args={}):
        IDStr = "','".join(self.__QS_adjustID__(ids))
        EndDTField = self._FactorInfo.loc[args.get("时点字段", self.DTField), "DBFieldName"]
        AnnDTField = self._FactorInfo.loc[args.get("公告时点字段", self.PublDTField), "DBFieldName"]
        IDField = self._FactorInfo.loc[args.get("ID字段", self.IDField), "DBFieldName"]
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAXOF("+EndDTField+") AS 'MaxEndDate' "
        SubSQLStr += self._genFromSQLStr()+" "
        SubSQLStr += "OF ARRAY('"+IDStr+"') "
        SubSQLStr += "WHERE ("+AnnDTField+"<"+end_date.strftime(self._DTFormat)+" "
        SubSQLStr += "AND "+EndDTField+"<"+end_date.strftime(self._DTFormat)+") "
        SubSQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        SubSQLStr += "GROUP BY "+IDField+" END"
        SQLStr = "SELECT MAX([1]."+AnnDTField+", [2].['MaxEndDate']) AS 'QS_DT', "
        SQLStr += "[1]."+IDField+" AS 'ID', "
        SQLStr += "[2].['MaxEndDate'] AS 'MaxEndDate', "
        for iField in factor_names: SQLStr += "[1]."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
        SQLStr = SQLStr[:-2]+" "+self._genFromSQLStr()+" "
        SQLStr += "OF ARRAY('"+IDStr+"') "
        SQLStr += "JOIN ("+SubSQLStr+") WITH ([1]."+IDField+", [1]."+EndDTField+" ON [2]."+IDField+", [2].['MaxEndDate']) "
        SQLStr += self._genConditionSQLStr(use_main_table=False, init_keyword="WHERE", args=args)+" END"
        return "RETURN exportjsonstring("+SQLStr+");"
    def _prepareRawData_WithPublDT(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDMapping = _adjustID(ids)
        IDStr = "','".join(self.__QS_adjustID__(IDMapping.index))
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        LookBack = args.get("回溯天数", self.LookBack)
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        EndDTField = self._FactorInfo.loc[args.get("时点字段", self.DTField), "DBFieldName"]
        AnnDTField = self._FactorInfo.loc[args.get("公告时点字段", self.PublDTField), "DBFieldName"]
        IDField = self._FactorInfo.loc[args.get("ID字段", self.IDField), "DBFieldName"]
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAX("+AnnDTField+", "+EndDTField+") AS 'AnnDate', "
        SubSQLStr += "MAXOF("+EndDTField+") AS 'MaxEndDate' "
        SubSQLStr += self._genFromSQLStr()+" "
        SubSQLStr += "OF ARRAY('"+IDStr+"') "
        SubSQLStr += "WHERE ("+AnnDTField+">="+StartDate.strftime(self._DTFormat)+" "
        SubSQLStr += "OR "+EndDTField+">="+StartDate.strftime(self._DTFormat)+") "
        SubSQLStr += "AND ("+AnnDTField+"<="+EndDate.strftime(self._DTFormat)+" "
        SubSQLStr += "AND "+EndDTField+"<="+EndDate.strftime(self._DTFormat)+") "
        SubSQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        SubSQLStr += "GROUP BY "+IDField+", "+ "MAX("+AnnDTField+", "+EndDTField+") END"
        SQLStr = "SELECT [2].['AnnDate'] AS 'QS_DT', "
        SQLStr += IDField+" AS 'ID', "
        SQLStr += "[2].['MaxEndDate'] AS 'MaxEndDate', "
        for iField in factor_names: SQLStr += "[1]."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
        SQLStr = SQLStr[:-2]+" "+self._genFromSQLStr()+" "
        SQLStr += "OF ARRAY('"+IDStr+"') "
        SQLStr += "JOIN ("+SubSQLStr+") WITH ([1]."+IDField+", [1]."+EndDTField+" ON [2]."+IDField+", [2].['MaxEndDate']) "
        SQLStr += self._genConditionSQLStr(use_main_table=False, init_keyword="WHERE", args=args)+" "
        SQLStr += "ORDER BY [1].['ID'], [1].['QS_DT'] END"
        RawData = json.loads(self._FactorDB.fetchall("RETURN exportjsonstring("+SQLStr+");").decode("gbk"))
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID", "MaxEndDate"]+factor_names)
        else: RawData = pd.DataFrame(RawData).loc[:, ["QS_DT", "ID", "MaxEndDate"]+factor_names]
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==dt.datetime.combine(StartDate,dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = json.loads(self._FactorDB.fetchall(self._genNullIDSQLStr_WithPublDT(factor_names, list(NullIDs), StartDate, args=args)).decode("gbk"))
                if NullRawData:
                    NullRawData = pd.DataFrame(NullRawData).loc[:, ["QS_DT", "ID", "MaxEndDate"]+factor_names]
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "QS_DT"])
        if RawData.shape[0]==0: return RawData.loc[:, ["QS_DT", "ID"]+factor_names]
        RawData["ID"] = IDMapping.loc[RawData["ID"].values].values
        RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(str(int(x)), self._DTFormat))
        if args.get("截止日期递增", self.EndDateASC):# 删除截止日期非递增的记录
            DTRank = RawData.loc[:, ["ID", "QS_DT", "MaxEndDate"]].set_index(["ID"]).astype(np.datetime64).groupby(axis=0, level=0).rank(method="min")
            RawData = RawData[(DTRank["QS_DT"]<=DTRank["MaxEndDate"]).values]
        return RawData.loc[:, ["QS_DT", "ID"]+factor_names]
    def _genNullIDSQLStr_IgnorePublDT(self, factor_names, ids, end_date, args={}):
        IDStr ="','".join(self.__QS_adjustID__(ids))
        DTField = self._FactorInfo.loc[args.get("时点字段", self.DTField), "DBFieldName"]
        IDField = self._FactorInfo.loc[args.get("ID字段", self.IDField), "DBFieldName"]
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAXOF("+DTField+") AS 'MaxEndDate' "
        SubSQLStr += self._genFromSQLStr()+" "
        SubSQLStr += "OF ARRAY('"+IDStr+"') "
        SubSQLStr += "WHERE "+DTField+"<"+end_date.strftime(self._DTFormat)+" "
        SubSQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        SubSQLStr += "GROUP BY "+IDField+" END"
        SQLStr = "SELECT [1]."+DTField+" AS 'QS_DT', "
        SQLStr += "[1]."+IDField+" AS 'ID', "
        for iField in factor_names: SQLStr += "[1]."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
        SQLStr = SQLStr[:-2]+" "+self._genFromSQLStr()+" "
        SQLStr += "OF ARRAY('"+IDStr+"') "
        SQLStr += "JOIN ("+SubSQLStr+") WITH ([1]."+IDField+", [1]."+DTField+" ON [2]."+IDField+", [2].['MaxEndDate']) "
        SQLStr += self._genConditionSQLStr(use_main_table=False, init_keyword="WHERE", args=args)+" END"
        return "RETURN exportjsonstring("+SQLStr+");"
    def _prepareRawData_IgnorePublDT(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDMapping = _adjustID(ids)
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        LookBack = args.get("回溯天数", self.LookBack)
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        DTField = self._FactorInfo.loc[args.get("时点字段", self.DTField), "DBFieldName"]
        IDField = self._FactorInfo.loc[args.get("ID字段", self.IDField), "DBFieldName"]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DTField+" AS 'QS_DT', "
        SQLStr += IDField+" AS 'ID', "
        for iField in factor_names: SQLStr += self._FactorInfo.loc[iField, "DBFieldName"]+", "
        SQLStr = SQLStr[:-2]+" "+self._genFromSQLStr()+" "
        SQLStr += "OF ARRAY('"+"','".join(IDMapping.index)+"') "
        SQLStr += "WHERE "+DTField+">="+StartDate.strftime(self._DTFormat)+" "
        SQLStr += "AND "+DTField+"<="+EndDate.strftime(self._DTFormat)+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ['ID'], ['QS_DT'] END"
        RawData = json.loads(self._FactorDB.fetchall("RETURN exportjsonstring("+SQLStr+");").decode("gbk"))
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        else: RawData = pd.DataFrame(RawData).loc[:, ["QS_DT", "ID"]+factor_names]
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==dt.datetime.combine(StartDate,dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = json.loads(self._FactorDB.fetchall(self._genNullIDSQLStr_IgnorePublDT(factor_names, list(NullIDs), StartDate, args=args)).decode("gbk"))
                if NullRawData:
                    NullRawData = pd.DataFrame(NullRawData).loc[:, ["QS_DT", "ID"]+factor_names]
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "QS_DT"])
        RawData["ID"] = IDMapping.loc[RawData["ID"].values].values
        RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(str(int(x)), self._DTFormat))
        return RawData

class _FeatureTable(_TS_SQL_Table, SQL_FeatureTable):
    """特征因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if ids==[]: return pd.DataFrame(columns=["ID"]+factor_names)
        IDMapping = _adjustID(ids)
        IDField = self._FactorInfo.loc[args.get("ID字段", self.IDField), "DBFieldName"]
        # 形成SQL语句, ID, 因子数据
        SQLStr = "SELECT "+IDField+" AS 'ID', "
        for iField in factor_names: SQLStr += self._FactorInfo.loc[iField, "DBFieldName"]+", "
        SQLStr = SQLStr[:-2]+" "+self._genFromSQLStr()+" "
        SQLStr += "OF ARRAY('"+"','".join(IDMapping.index)+"') "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ['ID'] END"
        RawData = json.loads(self._FactorDB.fetchall("RETURN exportjsonstring("+SQLStr+");").decode("gbk"))
        if not RawData: return pd.DataFrame(columns=["ID"]+factor_names)
        RawData = pd.DataFrame(RawData).loc[:, ["ID"]+factor_names]
        RawData["ID"] = IDMapping.loc[RawData["ID"].values].values
        return RawData

class _MappingTable(_TS_SQL_Table, SQL_MappingTable):
    """映射因子表"""
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTField = self._FactorInfo.loc[args.get("时点字段", self.DTField), "DBFieldName"]
        SQLStr = "SELECT MINOF("+DTField+") AS 'StartDT'"# 起始日期
        if iid is not None:
            SQLStr += self._genFromSQLStr()+" "
            SQLStr += self._genIDSQLStr([iid])+" "
            SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        else:
            raise __QS_Error__("TinysoftDB 的因子表方法 getDateTime 参数 iid 不能为 None")
        StartDT = dt.datetime.strptime(str(int(json.loads(self._FactorDB.fetchall("RETURN exportjsonstring("+SQLStr+"END);").decode("gbk"))[0]["StartDT"])), self._DTFormat)
        if start_dt is not None: StartDT = max((StartDT, start_dt))
        if end_dt is None: end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=StartDT, end_dt=end_dt, timedelta=dt.timedelta(1))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IDMapping = _adjustID(ids)
        IDField = self._FactorInfo.loc[args.get("ID字段", self.IDField), "DBFieldName"]
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DTField = self._FactorInfo.loc[args.get("时点字段", self.DTField), "DBFieldName"]
        # 形成SQL语句, ID, 开始日期, 结束日期, 因子数据
        SQLStr = "SELECT "+IDField+" AS 'ID', "
        SQLStr += DTField+" AS 'QS_起始日', "
        SQLStr += self._EndDateField+" AS 'QS_结束日', "
        for iField in factor_names: SQLStr += self._FactorInfo.loc[iField, "DBFieldName"]+", "
        SQLStr = SQLStr[:-2]+" "+self._genFromSQLStr()+" "
        SQLStr += "OF ARRAY('"+"','".join(IDMapping.index)+"') "
        SQLStr += "WHERE (("+self._EndDateField+">="+StartDate.strftime(self._DTFormat)+") "
        SQLStr += "OR ("+self._EndDateField+" IS NULL) "
        SQLStr += "OR ("+self._EndDateField+"<"+DTField+")) "
        SQLStr += "AND "+DTField+"<="+EndDate.strftime(self._DTFormat)+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ['ID'], ['QS_起始日'] END"
        RawData = json.loads(self._FactorDB.fetchall("RETURN exportjsonstring("+SQLStr+");").decode("gbk"))
        if not RawData: return pd.DataFrame(columns=["ID", "QS_起始日", "QS_结束日"]+factor_names)
        RawData = pd.DataFrame(RawData).loc[:, ["ID", "QS_起始日", "QS_结束日"]+factor_names]
        RawData["ID"] = IDMapping.loc[RawData["ID"].values].values
        RawData["QS_起始日"] = RawData["QS_起始日"].apply(lambda x: dt.datetime.strptime(str(int(x)), self._DTFormat) if pd.notnull(x) and (x!=0) else None)
        RawData["QS_结束日"] = RawData["QS_结束日"].apply(lambda x: dt.datetime.strptime(str(int(x)), self._DTFormat) if pd.notnull(x) and (x!=0) else None)
        return RawData


class TinySoftDB(FactorDB):
    """TinySoft"""
    Name = Str("TinySoftDB", arg_type="String", label="名称", order=-100)
    InstallDir = Directory(label="安装目录", arg_type="Directory", order=0)
    IPAddr = Str("tsl.tinysoft.com.cn", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=443, arg_type="Integer", label="端口", order=2)
    User = Str("", arg_type="String", label="用户名", order=3)
    Pwd = Password("", arg_type="String", label="密码", order=4)
    DBInfoFile = File(label="库信息文件", arg_type="File", order=100)
    FTArgs = Dict(label="因子表参数", arg_type="Dict", order=101)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"TinySoftDBConfig.json" if config_file is None else config_file), **kwargs)
        self._TSLPy = None
        self._TableInfo = None# 数据库中的表信息
        self._FactorInfo = None# 数据库中的表字段信息
        self._InfoFilePath = __QS_LibPath__+os.sep+"TinySoftDBInfo.hdf5"# 数据库信息文件路径
        if not os.path.isfile(self.DBInfoFile):
            if self.DBInfoFile: self._QS_Logger.warning("找不到指定的库信息文件 : '%s'" % self.DBInfoFile)
            self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"TinySoftDBInfo.xlsx"# 默认数据库信息源文件路径
            self._TableInfo, self._FactorInfo = updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger)
        else:
            self._InfoResourcePath = self.DBInfoFile
            self._TableInfo, self._FactorInfo = importInfo(self._InfoFilePath, self._InfoResourcePath)# 数据库表信息, 数据库字段信息
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
            DefaultArgs = self._TableInfo.loc[table_name, "DefaultArgs"]
            if pd.isnull(DefaultArgs): DefaultArgs = {}
            else: DefaultArgs = eval(DefaultArgs)
            if pd.notnull(TableClass) and (TableClass!=""):
                Args = self.FTArgs.copy()
                Args.update(DefaultArgs)
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
        if kwargs.get("output_type", "date")=="date":
            return list(map(lambda x: dt.date(*self._TSLPy.DecodeDate(x)), Data))
        else:
            return list(map(lambda x: dt.datetime(*self._TSLPy.DecodeDate(x)), Data))
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