# coding=utf-8
"""基于 ClickHouse 数据库的因子库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Bool, ListStr, Dict

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.QSObjects import QSClickHouseObject
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.SQLDB import SQLDB
from QuantStudio.FactorDataBase.FDBFun import SQL_Table, SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable

def _identifyFieldType(factor_data, data_type=None):
    if (data_type is None) or (data_type=="double"):
        try:
            factor_data = factor_data.astype(float)
        except:
            FieldType = "Nullable(String)"
            factor_data = factor_data.where(pd.notnull(factor_data), None)
        else:
            FieldType = "Float64"
    else:
        FieldType = "Nullable(String)"
    return (factor_data, FieldType)

class _CH_SQL_Table(SQL_Table):
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)
        self._DTFormat = "'%Y-%m-%d %H:%M:%S'"
        return
    def __QS_identifyDataType__(self, field_data_type):
        field_data_type = field_data_type.lower()
        if (field_data_type.find("array")!=-1) or (field_data_type.find("tuple")!=-1):
            return "object"
        elif (field_data_type.find("num")!=-1) or (field_data_type.find("int")!=-1) or (field_data_type.find("decimal")!=-1) or (field_data_type.find("double")!=-1) or (field_data_type.find("float")!=-1) or (field_data_type.find("real")!=-1):
            return "double"
        elif (field_data_type.find("char")!=-1) or (field_data_type.find("text")!=-1) or (field_data_type.find("str")!=-1):
            return "string"
        else:
            return "object"

class _WideTable(_CH_SQL_Table, SQL_WideTable):
    """ClickHouseDB 宽因子表"""
    pass

class _NarrowTable(SQL_NarrowTable):
    """ClickHouseDB 窄因子表"""
    pass

class _FeatureTable(SQL_FeatureTable):
    """ClickHouseDB 特征因子表"""
    pass

class _TimeSeriesTable(SQL_TimeSeriesTable):
    """ClickHouseDB 时序因子表"""
    pass

class _MappingTable(SQL_MappingTable):
    """ClickHouseDB 映射因子表"""
    pass

class ClickHouseDB(QSClickHouseObject, SQLDB):
    """ClickHouseDB"""
    Name = Str("ClickHouseDB", arg_type="String", label="名称", order=-100)
    DBType = Enum("ClickHouse", arg_type="SingleOption", label="数据库类型", order=0)
    Connector = Enum("default", "clickhouse-driver", arg_type="SingleOption", label="连接器", order=7)
    CheckWriteData = Bool(False, arg_type="Bool", label="检查写入值", order=100)
    IgnoreFields = ListStr(arg_type="List", label="忽略字段", order=101)
    InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=102)
    FTArgs = Dict(label="因子表参数", arg_type="Dict", order=103)
    DTField = Str("qs_datetime", arg_type="String", label="时点字段", order=104)
    IDField = Str("qs_code", arg_type="String", label="ID字段", order=105)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"ClickHouseDBConfig.json" if config_file is None else config_file), **kwargs)
        return
    def _genFactorInfo(self, factor_info):
        factor_info["FieldName"] = factor_info["DBFieldName"]
        factor_info["FieldType"] = "因子"
        DataTypeStr = factor_info["DataType"].str
        DTMask = DataTypeStr.contains("date")
        factor_info["FieldType"][DTMask] = "Date"
        StrMask = (DataTypeStr.contains("str") | DataTypeStr.contains("uuid") | DataTypeStr.contains("ip"))
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
        Args = self.__QS_initFTArgs__(table_name=table_name, args=args)
        return eval("_"+Args["因子表类型"]+"(name='"+table_name+"', fdb=self, sys_args=Args, logger=self._QS_Logger)")
    def createTable(self, table_name, field_types):
        FieldTypes = field_types.copy()
        FieldTypes[self.DTField] = FieldTypes.pop(self.DTField, "DateTime")
        FieldTypes[self.IDField] = FieldTypes.pop(self.IDField, "String")
        self.createDBTable(self.InnerPrefix+table_name, FieldTypes, primary_keys=[self.IDField], index_fields=[self.IDField])
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
    def deleteData(self, table_name, ids=None, dts=None, dt_ids=None):
        if table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 deleteData 错误: 不存在因子表 '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (ids is None) and (dts is None): return self.truncateDBTable(self.InnerPrefix+table_name)
        DBTableName = self.TablePrefix+self.InnerPrefix+table_name
        SQLStr = "ALTER TABLE "+DBTableName+" DELETE "
        if dts is not None:
            DTs = [iDT.strftime("%Y-%m-%d %H:%M:%S") for iDT in dts]
            SQLStr += "WHERE "+genSQLInCondition(self.DTField, DTs, is_str=True, max_num=1000)+" "
        else:
            SQLStr += "WHERE "+self.DTField+" IS NOT NULL "
        if ids is not None:
            SQLStr += "AND "+genSQLInCondition(self.IDField, ids, is_str=True, max_num=1000)
        if dt_ids is not None:
            dt_ids = ["('"+iDTIDs[0].strftime("%Y-%m-%d %H:%M:%S")+"', '"+iDTIDs[1]+"')" for iDTIDs in dt_ids]
            SQLStr += "AND "+genSQLInCondition("("+elf.DTField+", "+self.IDField+")", dt_ids, is_str=False, max_num=1000)
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteData 删除表 '%s' 中数据时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        return 0
    def _adjustWriteData(self, data, factor_info):
        NewData = []
        DataLen = data.applymap(lambda x: len(x) if isinstance(x, list) else 1)
        DataLenMax = DataLen.max(axis=1)
        DataLenMin = DataLen.min(axis=1)
        if (DataLenMax!=DataLenMin).sum()>0:
            self._QS_Logger.warning("'%s' 在写入因子 '%s' 时出现因子值长度不一致的情况, 将填充缺失!" % (self.Name, str(data.columns.tolist())))
        for i in range(data.shape[0]):
            iDataLen = DataLenMax.iloc[i]
            if iDataLen>0:
                iData = data.iloc[i].apply(lambda x: [None]*(iDataLen-len(x))+x if isinstance(x, list) else [x]*iDataLen).tolist()
                NewData.extend(zip(*iData))
        NewData = pd.DataFrame(NewData, dtype="O", columns=data.columns)
        factor_info = factor_info.loc[data.columns[2:]]
        DataTypeStr = factor_info["DataType"].str
        NumMask = (DataTypeStr.contains("decimal") | DataTypeStr.contains("int") | DataTypeStr.contains("float") | DataTypeStr.contains("num"))
        for i, iFactorName in enumerate(factor_info.index):
            if NumMask.iloc[i]:
                NewData[iFactorName] = NewData[iFactorName].astype(float)
            else:
                NewData[iFactorName] = NewData.iloc[iFactorName].astype("O").where(pd.notnull(NewData[iFactorName]), None)
        return NewData.to_records(index=False).tolist()
    def _adjustListData(self, data, factor_info):
        factor_info = factor_info.loc[data.columns]
        DataTypeStr = factor_info["DataType"].str
        ListMask = (DataTypeStr.contains("array") | DataTypeStr.contains("tuple"))
        NumMask = (DataTypeStr.contains("decimal") | DataTypeStr.contains("int") | DataTypeStr.contains("float") | DataTypeStr.contains("num"))
        for i, iFactorName in enumerate(factor_info.index):
            if ListMask.iloc[i]:
                data[iFactorName] = data[iFactorName].apply(lambda x: [] if pd.isnull(x) else x)
            elif NumMask.iloc[i]:
                data[iFactorName] = data[iFactorName].astype(float)
            else:
                data[iFactorName] = data[iFactorName].astype("O").where(pd.notnull(data[iFactorName]), None)
        return data
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        FieldTypes = {}
        for i, iFactorName in enumerate(data.items):
            data[iFactorName], FieldTypes[iFactorName] = _identifyFieldType(data[iFactorName], data_type.get(iFactorName, None))
        if table_name not in self._TableInfo.index:
            self.createTable(table_name, field_types=FieldTypes)
        else:
            NewFactorNames = data.items.difference(self._FactorInfo.loc[table_name].index).tolist()
            if NewFactorNames:
                self.addFactor(table_name, {iFactorName: FieldTypes[iFactorName] for iFactorName in NewFactorNames})
            if if_exists=="update":
                OldFactorNames = self._FactorInfo.loc[table_name].index.difference(data.items).difference({self.IDField, self.DTField}).tolist()
                if OldFactorNames:
                    if self.CheckWriteData:
                        OldData = self.getTable(table_name, args={"多重映射": True}).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    else:
                        OldData = self.getTable(table_name, args={"多重映射": False}).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    for iFactorName in OldFactorNames: data[iFactorName] = OldData[iFactorName]
            else:
                AllFactorNames = self._FactorInfo.loc[table_name].index.difference({self.IDField, self.DTField}).tolist()
                if self.CheckWriteData:
                    OldData = self.getTable(table_name, args={"多重映射": True}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                else:
                    OldData = self.getTable(table_name, args={"多重映射": False}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                if if_exists=="append":
                    for iFactorName in AllFactorNames:
                        if iFactorName in data:
                            data[iFactorName] = OldData[iFactorName].where(pd.notnull(OldData[iFactorName]), data[iFactorName])
                        else:
                            data[iFactorName] = OldData[iFactorName]
                elif if_exists=="update_notnull":
                    for iFactorName in AllFactorNames:
                        if iFactorName in data:
                            data[iFactorName] = data[iFactorName].where(pd.notnull(data[iFactorName]), OldData[iFactorName])
                        else:
                            data[iFactorName] = OldData[iFactorName]
                else:
                    Msg = ("因子库 '%s' 调用方法 writeData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
                    self._QS_Logger.error(Msg)
                    raise __QS_Error__(Msg)
        SQLStr = f"INSERT INTO {self.TablePrefix+self.InnerPrefix+table_name} (`{self.DTField}`, `{self.IDField}`, "
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
            SQLStr += "`"+iFactorName+"`, "
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        Mask = pd.notnull(NewData).any(axis=1)
        NewData = NewData[Mask]
        if NewData.shape[0]==0: return 0
        SQLStr = SQLStr[:-2] + ") VALUES "
        self.deleteData(table_name, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
        Cursor = self.cursor()
        if self.CheckWriteData:
            NewData = self._adjustWriteData(NewData.reset_index(), self._FactorInfo.loc[table_name])
        else:
            NewData = self._adjustListData(NewData, self._FactorInfo.loc[table_name]).reset_index().values.tolist()
        Cursor.executemany(SQLStr, NewData)
        self.Connection.commit()
        Cursor.close()
        return 0