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
from QuantStudio.FactorDataBase.FDBFun import SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable, SQL_ConstituentTable, SQL_FinancialTable

class _WideTable(SQL_WideTable):
    """SQLite3 宽因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _NarrowTable(SQL_NarrowTable):
    """SQLite3 窄因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _FeatureTable(SQL_FeatureTable):
    """SQLite3 特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _TimeSeriesTable(SQL_TimeSeriesTable):
    """SQLite3 时序因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _MappingTable(SQL_MappingTable):
    """SQLite3 映射因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _ConstituentTable(SQL_ConstituentTable):
    """SQLite3 成份因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _FinancialTable(SQL_FinancialTable):
    """SQLite3 财务因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)
    
    def _genConditionSQLStr(self, use_main_table=True, init_keyword="AND", args={}):
        SQLStr = super()._genConditionSQLStr(use_main_table=use_main_table, init_keyword=init_keyword, args=args)
        if SQLStr: init_keyword = "AND"
        ReportDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        if args.get("忽略非季末报告", self._QSArgs.IgnoreNonQuarter) or (not ((args.get("报告期", self._QSArgs.ReportDate)=="所有") and (args.get("计算方法", self._QSArgs.CalcType)=="最新") and (args.get("回溯年数", self._QSArgs.YearLookBack)==0) and (args.get("回溯期数", self._QSArgs.PeriodLookBack)==0))):
            DTFmt = args.get("时点格式", self._QSArgs.DTFmt).replace("%Y", "")
            SQLStr + " "+init_keyword+" ("+ReportDTField+f" LIKE '{DTFmt.replace('%m', '03').replace('%d','31')}' "
            SQLStr + "OR "+ReportDTField+f" LIKE '{DTFmt.replace('%m', '06').replace('%d','30')}' "
            SQLStr + "OR "+ReportDTField+f" LIKE '{DTFmt.replace('%m', '09').replace('%d','30')}' "
            SQLStr + "OR "+ReportDTField+f" LIKE '{DTFmt.replace('%m', '12').replace('%d','31')}') "
            init_keyword = "AND"
        AdjustTypeField = args.get("调整类型字段", self._QSArgs.AdjustTypeField)
        if AdjustTypeField is not None:
            iConditionVal = args.get("调整类型", self._QSArgs.AdjustType)
            if iConditionVal:
                if self.__QS_identifyDataType__(self._FactorInfo.loc[AdjustTypeField, "DataType"])!="double":
                    SQLStr += " "+init_keyword+" "+self._DBTableName+"."+self._FactorInfo.loc[AdjustTypeField, "DBFieldName"]+" IN ('"+"','".join(iConditionVal.split(","))+"') "
                else:
                    SQLStr += " "+init_keyword+" "+self._DBTableName+"."+self._FactorInfo.loc[AdjustTypeField, "DBFieldName"]+" IN ("+iConditionVal+") "
        return SQLStr

class SQLite3DB(QSSQLite3Object, SQLDB):
    """SQLite3DB"""
    class __QS_ArgClass__(QSSQLite3Object.__QS_ArgClass__, SQLDB.__QS_ArgClass__):
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
        Args["时点格式"] = Args.get("时点格式", self._QSArgs.DTFmt)
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

if __name__=="__main__":
    SDB = SQLite3DB(sys_args={"sqlite3文件": "D:/Project/Research/QSDemo/Data/SQLite3/TestData.sqlite3", 
                                                                         "内部前缀": "", 
                                                                         "时点字段": "datetime",
                                                                         "ID字段": "code",
                                                                         "时点格式": "%Y-%m-%d",
                                                                         "因子表参数": {"忽略时间": True}})
    SDB.connect()
    
    TargetTable = "test_FinancialTable1"
    SQLStr = f"SELECT * FROM {TargetTable}"
    print("库原始数据 : ")
    print(pd.read_sql_query(SQLStr, SDB.Connection))
    
    DTs = [dt.datetime(2022,1,17), dt.datetime(2022,1,18)]
    IDs = ["000001.SZ"]
    
    Args = {
        "因子表类型": "FinancialTable",
        "时点字段": "report_date",
        "公告时点字段": "ann_dt",
        "报告期": "年报",
        "计算方法": "最新",
        "回溯年数": 1,
        "回溯期数": 0,
    }
    FT = SDB.getTable(TargetTable, args=Args)
    print("参数 : ")
    print(Args)
    print("数据 : ")
    print(FT.readData(factor_names=["factor1", "factor2"], ids=IDs, dts=DTs).iloc[:, :, 0])
    