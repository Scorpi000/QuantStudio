# coding=utf-8
"""基于 SQLite3 数据库的因子库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str

from QuantStudio.Tools.QSObjects import QSSQLite3Object
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.SQLDB import SQLDB
from QuantStudio.FactorDataBase.FDBFun import SQL_Table, SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable

class _SQLite3_SQL_Table(SQL_Table):
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)
        return
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTs = super().getDateTime(ifactor_name=ifactor_name, iid=iid, start_dt=start_dt, end_dt=end_dt, args=args)
        return [dt.datetime.strptime(iDT, "%Y-%m-%d") for iDT in DTs]
        

class _WideTable(_SQLite3_SQL_Table, SQL_WideTable):
    """ClickHouseDB 宽因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
        return RawData
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTs = SQL_WideTable.getDateTime(self, ifactor_name=ifactor_name, iid=iid, start_dt=start_dt, end_dt=end_dt, args=args)
        return [dt.datetime.strptime(iDT, "%Y-%m-%d") for iDT in DTs]

class _NarrowTable(_SQLite3_SQL_Table, SQL_NarrowTable):
    """ClickHouseDB 窄因子表"""
    pass

class _FeatureTable(_SQLite3_SQL_Table, SQL_FeatureTable):
    """ClickHouseDB 特征因子表"""
    pass

class _TimeSeriesTable(_SQLite3_SQL_Table, SQL_TimeSeriesTable):
    """ClickHouseDB 时序因子表"""
    pass

class _MappingTable(_SQLite3_SQL_Table, SQL_MappingTable):
    """ClickHouseDB 映射因子表"""
    pass


class SQLite3DB(QSSQLite3Object, SQLDB):
    """SQLite3DB"""
    Name = Str("SQLite3DB", arg_type="String", label="名称", order=-100)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"SQLite3DBConfig.json" if config_file is None else config_file), **kwargs)
        return
    def _genFactorInfo(self, factor_info):
        factor_info["FieldName"] = factor_info["DBFieldName"]
        factor_info["FieldType"] = "因子"
        factor_info["DataType"] = factor_info["DataType"].str.lower()
        DTMask = factor_info["DataType"].str.contains("text")
        factor_info["FieldType"][DTMask] = "Date"
        StrMask = (factor_info["DataType"].str.contains("char") | factor_info["DataType"].str.contains("text"))
        factor_info["FieldType"][(factor_info["DBFieldName"].str.lower()=="code") & StrMask] = "ID"
        factor_info["Supplementary"] = None
        factor_info["Supplementary"][DTMask & (factor_info["DBFieldName"].str.lower()=="datetime")] = "Default"
        factor_info["Description"] = ""
        factor_info["Nullable"] = np.where(factor_info["Nullable"].values==1, "NO", "YES")
        factor_info["FieldKey"] = np.where(factor_info["FieldKey"].values>0, "PRI", None)
        factor_info = factor_info.set_index(["TableName", "FieldName"])
        return factor_info
    def connect(self):
        QSSQLite3Object.connect(self)
        nPrefix = len(self.InnerPrefix)
        SQLStr = f"SELECT name AS DBTableName FROM sqlite_master WHERE type='table' AND name LIKE '{self.InnerPrefix}%%' ORDER BY name"
        self._TableInfo = pd.read_sql_query(SQLStr, self._Connection)
        self._TableInfo["TableName"] = self._TableInfo["DBTableName"].apply(lambda x: x[nPrefix:])
        self._TableInfo["TableClass"] = "WideTable"
        self._TableInfo = self._TableInfo.set_index(["TableName"])
        self._FactorInfo = pd.DataFrame(columns=["TableName", "DBFieldName", "DataType"])
        Cursor = self.cursor()
        for iTableName in self._TableInfo.index:
            Cursor.execute(f"PRAGMA table_info([{self.InnerPrefix+iTableName}])")
            iFactorInfo = np.array(Cursor.fetchall())
            iFactorInfo = pd.DataFrame(iFactorInfo[:, 1:6], columns=["DBFieldName", "DataType", "Nullable", "DefaultValue", "FieldKey"])
            iFactorInfo = iFactorInfo[~iFactorInfo["DBFieldName"].isin(self.IgnoreFields)]
            if iFactorInfo.shape[0]>0:
                iFactorInfo["TableName"] = iTableName
                self._FactorInfo = self._FactorInfo.append(iFactorInfo)
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
        FieldTypes[self.DTField] = FieldTypes.pop(self.DTField, "text NOT NULL")
        FieldTypes[self.IDField] = FieldTypes.pop(self.IDField, "text NOT NULL")
        self.createDBTable(self.InnerPrefix+table_name, FieldTypes, primary_keys=[self.DTField, self.IDField], index_fields=[self.IDField])
        self._TableInfo = self._TableInfo.append(pd.Series([self.InnerPrefix+table_name, "WideTable"], index=["DBTableName", "TableClass"], name=table_name))
        NewFactorInfo = pd.DataFrame(FieldTypes, index=["DataType"], columns=pd.Index(sorted(FieldTypes.keys()), name="DBFieldName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = self._FactorInfo.append(self._genFactorInfo(NewFactorInfo))
        return 0