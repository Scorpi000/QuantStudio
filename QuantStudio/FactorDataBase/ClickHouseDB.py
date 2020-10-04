# coding=utf-8
"""基于 ClickHouse 数据库的因子库"""
import re
import os
import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, File, Float, Bool, ListStr, Dict, on_trait_change, Either, Date

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.QSObjects import QSClickHouseObject
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.SQLDB import SQLDB
from QuantStudio.FactorDataBase.FDBFun import SQL_Table, SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable

def _identifyDataType(db_type, dtypes):
    if np.dtype("O") in dtypes.values: return "String"
    else: return "Float64"

class _CH_SQL_Table(SQL_Table):
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)
        self._DTFormat = "'%Y-%m-%d %H:%M:%S'"
        return

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
        self._SQLFun = {"toDate": "toDate(%s)"}
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
            SQLStr += "WHERE "+elf.DTField+" IS NOT NULL "
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
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._TableInfo.index:
            FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            self.createTable(table_name, field_types=FieldTypes)
        else:
            NewFactorNames = data.items.difference(self._FactorInfo.loc[table_name].index).tolist()
            if NewFactorNames:
                FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(NewFactorNames)}
                self.addFactor(table_name, FieldTypes)
            if if_exists=="update":
                OldFactorNames = self._FactorInfo.loc[table_name].index.difference(data.items).tolist()
                if OldFactorNames:
                    if self.CheckWriteData:
                        OldData = self.getTable(table_name, args={"多重映射": True}).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    else:
                        OldData = self.getTable(table_name, args={"多重映射": False}).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    for iFactorName in OldFactorNames: data[iFactorName] = OldData[iFactorName]
            else:
                AllFactorNames = self._FactorInfo.loc[table_name].index.tolist()
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
            NewData = self._adjustWriteData(NewData.reset_index())
        else:
            NewData = NewData.astype("O").where(pd.notnull(NewData), None).reset_index().values.tolist()
        Cursor.executemany(SQLStr, NewData)
        self.Connection.commit()
        Cursor.close()
        return 0