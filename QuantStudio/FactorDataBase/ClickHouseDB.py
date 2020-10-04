# coding=utf-8
"""基于 ClickHouse 数据库的因子库(TODO)"""
import re
import os
import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, File, Float, Bool, ListStr, on_trait_change, Either, Date

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.QSObjects import QSClickHouseObject
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.SQLDB import SQLDB, _adjustData

def _identifyDataType(db_type, dtypes):
    if np.dtype("O") in dtypes.values: return "String"
    else: return "double"

class _CH_SQL_Table(SQL_Table):
    pass

class _WideTable(SQL_WideTable):
    """ClickHouseDB 宽因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _NarrowTable(SQL_NarrowTable):
    """ClickHouseDB 窄因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _FeatureTable(SQL_FeatureTable):
    """ClickHouseDB 特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _TimeSeriesTable(SQL_TimeSeriesTable):
    """ClickHouseDB 时序因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _MappingTable(SQL_MappingTable):
    """ClickHouseDB 映射因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _WideTable_Old(FactorTable):
    """SQLDB 宽因子表"""
    TableType = Enum("宽表", arg_type="SingleOption", label="因子表类型", order=0)
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=1)
    FilterCondition = Str("", arg_type="String", label="筛选条件", order=2)
    #DTField = Enum("dt", arg_type="SingleOption", label="时点字段", order=3)
    #IDField = Enum("code", arg_type="SingleOption", label="ID字段", order=4)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DataType = fdb._TableFactorDict[name]
        self._DBDataType = fdb._TableFieldDataType[name]
        self._DBTableName = fdb.TablePrefix+fdb.InnerPrefix+name
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        Fields = ["dt"] + self._DBDataType[self._DBDataType.str.contains("date")].index.tolist()
        self.add_trait("DTField", Enum(*Fields, arg_type="SingleOption", label="时点字段", order=4))
        StrMask = (self._DBDataType.str.contains("char") | self._DBDataType.str.contains("text"))
        Fields = ["code"] + self._DBDataType[StrMask].index.tolist()
        self.add_trait("IDField", Enum(*Fields, arg_type="SingleOption", label="ID字段", order=5))
    @property
    def FactorNames(self):
        return self._DataType.index.tolist()+["dt", "code"]
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType": return self._DataType.append(pd.Series(["string", "string"], index=["dt","code"])).loc[factor_names]
        if key is None: return pd.DataFrame(self._DataType.append(pd.Series(["string", "string"], index=["dt","code"])).loc[factor_names], columns=["DataType"])
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+IDField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+IDField+" IS NOT NULL "
        if idt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+ifactor_name+" IS NOT NULL "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+IDField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+DTField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
        if iid is not None: SQLStr += "AND "+self._DBTableName+"."+IDField+"='"+iid+"' "
        if start_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+">='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+ifactor_name+" IS NOT NULL "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+DTField
        if self._FactorDB.DBType!="sqlite3": return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        else: return [dt.datetime.strptime(iRslt[0], "%Y-%m-%d %H:%M:%S.%f") for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ArgConditionGroup = {}
        ArgNames = self.ArgNames
        ArgNames.remove("回溯天数")
        ArgNames.remove("因子值类型")
        ArgNames.remove("遍历模式")
        for iFactor in factors:
            iArgConditions = (";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in ArgNames]))
            if iArgConditions not in ArgConditionGroup:
                ArgConditionGroup[iArgConditions] = {"FactorNames":[iFactor.Name], 
                                                     "RawFactorNames":{iFactor._NameInFT}, 
                                                     "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                     "args":iFactor.Args.copy()}
            else:
                ArgConditionGroup[iArgConditions]["FactorNames"].append(iFactor.Name)
                ArgConditionGroup[iArgConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ArgConditionGroup[iArgConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ArgConditionGroup[iArgConditions]["StartDT"])
                ArgConditionGroup[iArgConditions]["args"]["回溯天数"] = max(ArgConditionGroup[iArgConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iArgConditions in ArgConditionGroup:
            StartInd = operation_mode.DTRuler.index(ArgConditionGroup[iArgConditions]["StartDT"])
            Groups.append((self, ArgConditionGroup[iArgConditions]["FactorNames"], list(ArgConditionGroup[iArgConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ArgConditionGroup[iArgConditions]["args"]))
        return Groups
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        DT2Str = args.get("时间转字符串", self.DT2Str)
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAX("+DTField+") "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "WHERE "+DTField+"<'"+end_date.strftime("%Y-%m-%d:%H:%M:%S.%f")+"' "
        SubSQLStr += "AND ("+genSQLInCondition(IDField, ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SubSQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SubSQLStr += "GROUP BY "+IDField
        SQLStr = "SELECT "+DTField+", "
        SQLStr += IDField+", "
        for iField in factor_names:
            if iField=="dt": iDBDataType = "DateTime"
            elif iField=="code": iDBDataType = "String"
            else: iDBDataType = self._FactorDB._TableFieldDataType[self._Name][iField]            
            if DT2Str and (iDBDataType.lower().find("date")!=-1):
                SQLStr += "DATE_FORMAT("+iField+", '%Y-%m-%d %H:%i:%s'), "
            else:
                SQLStr += iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "WHERE ("+IDField+", "+DTField+") IN ("+SubSQLStr+") "
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        LookBack = args.get("回溯天数", self.LookBack)
        DT2Str = args.get("时间转字符串", self.DT2Str)
        if dts is not None:
            dts = sorted(dts)
            StartDate, EndDate = dts[0].date(), dts[-1].date()
            if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        else:
            StartDate = EndDate = None
        # 形成 SQL 语句, 时点, ID, 因子数据
        SQLStr = "SELECT "+self._DBTableName+"."+DTField+", "
        SQLStr += self._DBTableName+"."+IDField+", "
        for iField in factor_names:
            if iField=="datetime": iDBDataType = "datetime"
            elif iField=="code": iDBDataType = "varchar(40)"
            else: iDBDataType = self._FactorDB._TableFieldDataType[self._Name][iField]
            if DT2Str and (iDBDataType.lower().find("date")!=-1):
                SQLStr += "DATE_FORMAT("+self._DBTableName+"."+iField+", '%Y-%m-%d %H:%i:%s'), "
            else:
                SQLStr += self._DBTableName+"."+iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        if StartDate is not None:
            SQLStr += "WHERE "+self._DBTableName+"."+DTField+">='"+StartDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            SQLStr += "AND "+self._DBTableName+"."+DTField+"<='"+EndDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        else:
            SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+IDField, ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+DTField+", "+self._DBTableName+"."+IDField
        if args.get("因子值类型", self.ValueType)!="scalar":
            SQLStr += ", "+self._DBTableName+"."+factor_names[0]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData), columns=["QS_DT", "ID"]+factor_names)
        if (StartDate is not None) and np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==dt.datetime.combine(StartDate, dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "ID"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["QS_DT", "ID"])
        if self._FactorDB.DBType=="sqlite3": RawData["QS_DT"] = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in RawData.pop("QS_DT")]
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["QS_DT", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        return _adjustData(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts)

class ClickHouseDB(QSClickHouseObject, SQLDB):
    """ClickHouseDB"""
    DBType = Enum("ClickHouse", arg_type="SingleOption", label="数据库类型", order=0)
    Connector = Enum("default", "clickhouse-driver", arg_type="SingleOption", label="连接器", order=7)
    CheckWriteData = Bool(False, arg_type="Bool", label="检查写入值", order=100)
    IgnoreFields = ListStr(arg_type="List", label="忽略字段", order=101)
    InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=102)
    FTArgs = Dict(label="因子表参数", arg_type="Dict", order=103)
    DTField = Str("dt", arg_type="String", label="时点字段", order=104)
    IDField = Str("code", arg_type="String", label="ID字段", order=105)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"ClickHouseDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "ClickHouseDB"
        return
    def _genFactorInfo(self, factor_info):
        factor_info["FieldName"] = factor_info["DBFieldName"]
        factor_info["FieldType"] = "因子"
        DTMask = factor_info["DataType"].str.contains("date")
        factor_info["FieldType"][DTMask] = "Date"
        StrMask = (factor_info["DataType"].str.contains("str") | factor_info["DataType"].str.contains("uuid") | factor_info["DataType"].str.contains("ip"))
        factor_info["FieldType"][(factor_info["DBFieldName"].str.lower()==self.IDField) & StrMask] = "ID"
        factor_info["Supplementary"] = None
        factor_info["Supplementary"][DTMask & (factor_info["DBFieldName"].str.lower()==self.DTField)] = "Default"
        factor_info["Description"] = ""
        factor_info = factor_info.set_index(["TableName", "FieldName"])
        return factor_info
    def connect(self):
        QSClickHouseObject.connect(self)
        nPrefix = len(self.InnerPrefix)
        SQLStr = f"SELECT RIGHT(table, LENGTH(table)-{nPrefix}) AS TableName, table AS DBTableName, name AS DBFieldName, LOWER(type) AS DataType FROM system.columns WHERE database='{self.DBName}' "
        SQLStr += f"AND table LIKE '{self.InnerPrefix}%%' "
        if len(self.IgnoreFields)>0:
            SQLStr += "AND name NOT IN ('"+"','".join(self.IgnoreFields)+"') "
        SQLStr += "ORDER BY TableName, DBFieldName"
        self._FactorInfo = pd.read_sql_query(SQLStr, self._Connection, index_col=None)
        self._TableInfo = self._FactorInfo.loc[:, ["TableName", "DBTableName"]].copy().groupby(by=["TableName"], as_index=True).last().sort_index()
        self._TableInfo["TableClass"] = "WideTable"
        self._FactorInfo.pop("DBTableName")
        self._FactorInfo = self._genFactorInfo(self._FactorInfo)
        return 0
    def getTable(self, table_name, args={}):
        if table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不存在因子表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        Args = self.FTArgs.copy()
        Args.update(args)
        TableClass = Args.get("因子表类型", self._TableInfo.loc[table_name, "TableClass"])
        return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=Args, logger=self._QS_Logger)")
    def createTable(self, table_name, field_types):
        FieldTypes = field_types.copy()
        FieldTypes[self.DTField] = FieldTypes.pop(self.DTField, "DATETIME NOT NULL")
        FieldTypes[self.IDField] = FieldTypes.pop(self.IDField, "STRING NOT NULL")
        self.createDBTable(self.InnerPrefix+table_name, FieldTypes, primary_keys=[self.DTField, self.IDField], index_fields=[self.IDField])
        self._TableInfo = self._TableInfo.append(pd.Series([self.InnerPrefix+table_name, "WideTable"], index=["DBTableName", "TableClass"], name=table_name))
        NewFactorInfo = pd.DataFrame(FieldTypes, index=["DataType"], columns=pd.Index(sorted(FieldTypes.keys()), name="DBFieldName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = self._FactorInfo.append(self._genFactorInfo(NewFactorInfo))
        return 0
    def addFactor(self, table_name, field_types):
        if table_name not in self._TableInfo.index: return self.createTable(table_name, field_types)
        self.addField(self.InnerPrefix+table_name, field_types)
        NewFactorInfo = pd.DataFrame(field_types, index=["DataType"], columns=pd.Index(sorted(field_types.keys()), name="DBFieldName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = self._FactorInfo.append(self._genFactorInfo(NewFactorInfo)).sort_index()
        return 0
        # ----------------------------因子操作---------------------------------
    def _adjustWriteData(self, data):# TODO
        NewData = []
        DataLen = data.applymap(lambda x: len(x) if isinstance(x, list) else 1)
        DataLenMax = DataLen.max(axis=1)
        DataLenMin = DataLen.min(axis=1)
        if (DataLenMax!=DataLenMin).sum()>0:
            self._QS_Logger.warning("'%s' 在写入因子 '%s' 时出现因子值长度不一致的情况, 将填充缺失!" % (self.Name, str(data.columns.tolist())))
        for i in range(data.shape[0]):
            iDataLen = DataLen.iloc[i]
            if iDataLen>0:
                iData = data.iloc[i].apply(lambda x: [None]*(iDataLen-len(x))+x if isinstance(x, list) else [x]*iDataLen).tolist()
                NewData.extend(zip(*iData))
        NewData = pd.DataFrame(NewData, dtype="O")
        return NewData.where(pd.notnull(NewData), None).to_records(index=False).tolist()
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._TableFactorDict:
            FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            self.createTable(table_name, field_types=FieldTypes)
            SQLStr = "INSERT INTO "+self.TablePrefix+self.InnerPrefix+table_name+" (`datetime`, `code`, "
        else:
            NewFactorNames = data.items.difference(self._TableFactorDict[table_name].index).tolist()
            if NewFactorNames:
                FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(NewFactorNames)}
                self.addFactor(table_name, FieldTypes)
            AllFactorNames = self._TableFactorDict[table_name].index.tolist()
            if self.CheckWriteData:
                OldData = self.getTable(table_name, args={"因子值类型":"list", "时间转字符串":True}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
            else:
                OldData = self.getTable(table_name, args={"时间转字符串":True}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
            if if_exists=="append":
                for iFactorName in AllFactorNames:
                    if iFactorName in data:
                        data[iFactorName] = OldData[iFactorName].where(pd.notnull(OldData[iFactorName]), data[iFactorName])
                    else:
                        data[iFactorName] = OldData[iFactorName]
            elif if_exists=="update":
                for iFactorName in AllFactorNames:
                    if iFactorName in data:
                        data[iFactorName] = data[iFactorName].where(pd.notnull(data[iFactorName]), OldData[iFactorName])
                    else:
                        data[iFactorName] = OldData[iFactorName]
            SQLStr = "REPLACE INTO "+self.TablePrefix+self.InnerPrefix+table_name+" (`datetime`, `code`, "
        data.major_axis = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in data.major_axis]
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
            SQLStr += "`"+iFactorName+"`, "
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        Mask = pd.notnull(NewData).any(axis=1)
        NewData = NewData[Mask]
        # TODEBUG: 删除全部缺失的行
        #Mask = Mask[~Mask]
        #if Mask.shape[0]>0:
            #self.deleteData(table_name, dt_ids=Mask.index.tolist())
        if NewData.shape[0]==0: return 0
        if self._Connector in ("pyodbc", "sqlite3"):
            SQLStr = SQLStr[:-2] + ") VALUES (" + "?, " * (NewData.shape[1]+2)
        else:
            SQLStr = SQLStr[:-2] + ") VALUES (" + "%s, " * (NewData.shape[1]+2)
        SQLStr = SQLStr[:-2]+") "
        Cursor = self._Connection.cursor()
        if self.CheckWriteData:
            NewData = self._adjustWriteData(NewData.reset_index())
            Cursor.executemany(SQLStr, NewData)
        else:
            NewData = NewData.astype("O").where(pd.notnull(NewData), None)
            Cursor.executemany(SQLStr, NewData.reset_index().values.tolist())
        self._Connection.commit()
        Cursor.close()
        return 0