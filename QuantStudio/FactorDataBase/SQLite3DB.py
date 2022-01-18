# coding=utf-8
"""基于 SQLite3 数据库的因子库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str

from QuantStudio.Tools.QSObjects import QSSQLite3Object
from QuantStudio import __QS_ConfigPath__
from QuantStudio.FactorDataBase.SQLDB import SQLDB
from QuantStudio.FactorDataBase.FDBFun import SQL_Table, SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable, SQL_ConstituentTable, SQL_FinancialTable

class _SQLite3_SQL_Table(SQL_Table):
    DTFmt = Str("%Y-%m-%d", arg_type="String", label="时点格式", order=300)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)
        return
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTs = super().getDateTime(ifactor_name=ifactor_name, iid=iid, start_dt=start_dt, end_dt=end_dt, args=args)
        return [dt.datetime.strptime(iDT, args.get("时点格式", self.DTFmt)) for iDT in DTs]

class _WideTable(_SQLite3_SQL_Table, SQL_WideTable):
    """SQLite3 宽因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        if args.get("回溯期数", self.PeriodLookBack) is None:
            DTFmt = args.get("时点格式", self.DTFmt)
            RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        return RawData

class _NarrowTable(_SQLite3_SQL_Table, SQL_NarrowTable):
    """SQLite3 窄因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        DTFmt = args.get("时点格式", self.DTFmt)
        RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        return RawData

class _FeatureTable(_SQLite3_SQL_Table, SQL_FeatureTable):
    """SQLite3 特征因子表"""
    def _getMaxDT(self, args={}):
        MaxDT = super()._getMaxDT(args=args)
        if MaxDT is not None:
            DTFmt = args.get("时点格式", self.DTFmt)
            return dt.datetime.strptime(MaxDT, DTFmt)
        else:
            return None
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        if args.get("时点字段", self.DTField) is not None:
            DTFmt = args.get("时点格式", self.DTFmt)
            RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        return RawData

class _TimeSeriesTable(_SQLite3_SQL_Table, SQL_TimeSeriesTable):
    """SQLite3 时序因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        DTFmt = args.get("时点格式", self.DTFmt)
        RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        return RawData

class _MappingTable(_SQLite3_SQL_Table, SQL_MappingTable):
    """SQLite3 映射因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        DTFmt = args.get("时点格式", self.DTFmt)
        RawData["QS_起始日"] = RawData["QS_起始日"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        RawData["QS_结束日"] = RawData["QS_结束日"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        return RawData

class _ConstituentTable(_SQLite3_SQL_Table, SQL_ConstituentTable):
    """SQLite3 成份因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        DTFmt = args.get("时点格式", self.DTFmt)
        RawData["InDate"] = RawData["InDate"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        RawData["OutDate"] = RawData["OutDate"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        return RawData

class _FinancialTable(_SQLite3_SQL_Table, SQL_FinancialTable):
    """SQLite3 财务因子表"""
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        RawData = super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        DTFmt = args.get("时点格式", self.DTFmt)
        RawData["AnnDate"] = RawData["AnnDate"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        RawData["ReportDate"] = RawData["ReportDate"].apply(lambda x: dt.datetime.strptime(x, DTFmt))
        return RawData

class SQLite3DB(QSSQLite3Object, SQLDB):
    """SQLite3DB"""
    Name = Str("SQLite3DB", arg_type="String", label="名称", order=-100)
    DTFmt = Str("%Y-%m-%d", label="时点格式", arg_type="String", order=12)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"SQLite3DBConfig.json" if config_file is None else config_file), **kwargs)
        return
    def _genFactorInfo(self, factor_info):
        factor_info["FieldName"] = factor_info["DBFieldName"]
        factor_info["FieldType"] = "因子"
        factor_info["DataType"] = factor_info["DataType"].str.lower()
        StrMask = factor_info["DataType"].str.contains("text")
        factor_info["FieldType"][StrMask] = "Date"
        factor_info["FieldType"][(factor_info["DBFieldName"].str.lower()==self.IDField) & StrMask] = "ID"
        factor_info["Supplementary"] = None
        factor_info["Supplementary"][StrMask & (factor_info["DBFieldName"].str.lower()==self.DTField)] = "Default"
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
    def __QS_initFTArgs__(self, table_name, args):
        Args = super().__QS_initFTArgs__(table_name=table_name, args=args)
        Args["时点格式"] = Args.get("时点格式", self.DTFmt)
        return Args
    def getTable(self, table_name, args={}):
        Args = self.__QS_initFTArgs__(table_name=table_name, args=args)
        return eval("_"+Args["因子表类型"]+"(name='"+table_name+"', fdb=self, sys_args=Args, logger=self._QS_Logger)")
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