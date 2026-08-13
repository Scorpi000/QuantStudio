# coding=utf-8
"""基于 SQL 数据库的因子库"""
import os
import datetime as dt
from typing import List, Optional, Dict, Tuple, Literal, Any

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.QSObject import QSSQLObject, Panel
from QuantStudio.Factor.FactorDB import WritableFactorDB
from QuantStudio.Factor.FactorUtils import SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable, SQL_Table
from QuantStudio.Tools.SQLDBFun import genSQLInCondition


def _identifyDataType(db_type, dtypes):
    if db_type=="PostgreSQL":
        if np.dtype("O") in dtypes.values: return "TEXT"
        else: return "DOUBLE PRECISION"
    elif db_type!="sqlite3":
        if np.dtype("O") in dtypes.values: return "varchar(40)"
        else: return "double"
    else:
        if np.dtype("O") in dtypes.values: return "text"
        else: return "real"

class SQLDB(QSSQLObject, WritableFactorDB):
    """基于关系数据库的因子库"""

    class __QS_ArgClass__(QSSQLObject.__QS_ArgClass__, WritableFactorDB.__QS_ArgClass__):
        Name: str = Field(default="SQLDB", title="名称", frozen=True)
        FTArgs: dict = Field(default={}, title="因子表参数", frozen=True, exclude=True)
        InnerPrefix: str = Field(default="qs_", title="内部前缀", frozen=True)
        DTField: str = Field(default="datetime", title="时点字段", frozen=True)
        IDField: str = Field(default="code", title="ID字段", frozen=True)
        IgnoreFields: List[str] = Field(default=[], title="忽略字段", frozen=True)
        CheckWriteData: bool = Field(default=False, title="检查写入值", frozen=False)
        CheckNullable: bool = Field(default=False, title="检查缺失容许", frozen=False)
        MetaTableName: str = Field(default="qs_meta", title="元数据表名", frozen=True, description="用于存储表级和因子级额外元数据的侧表名")
    
    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        #self._TableFactorDict = {}# {表名: pd.Series(数据类型, index=[因子名])}
        #self._TableFieldDataType = {}# {表名: pd.Series(数据库数据类型, index=[因子名])}
        self._TableInfo = pd.DataFrame()# DataFrame(index=[表名], columns=["DBTableName", "TableClass"])
        self._FactorInfo = pd.DataFrame()# DataFrame(index=[(表名,因子名)], columns=["DBFieldName", "DataType", "FieldType", "Supplementary", "Description"])
        super().__init__(args=args, config_file=(__QS_ConfigPath__+os.sep+"SQLDBConfig.json" if config_file is None else config_file), **kwargs)
        return
    
    # factor_info: DataFrame(columns=["TableName", "DBFieldName", "FieldType", "Supplementary", "DataType", "Nullable", "FieldKey", "Description"])
    def _genFactorInfo(self, factor_info):
        factor_info["FieldName"] = factor_info["DBFieldName"]
        factor_info["FieldType"] = "因子"
        DTMask = factor_info["DataType"].str.contains("date|timestamp", case=False, regex=True)
        factor_info.loc[DTMask, "FieldType"] = "Date"
        StrMask = factor_info["DataType"].str.contains("char|text", case=False, regex=True)
        factor_info.loc[(factor_info["DBFieldName"].str.lower()==self._QSArgs.IDField) & StrMask, "FieldType"] = "ID"
        factor_info["Supplementary"] = None
        factor_info.loc[DTMask & (factor_info["DBFieldName"].str.lower()==self._QSArgs.DTField), "Supplementary"] = "Default"
        factor_info = factor_info.set_index(["TableName", "FieldName"])
        return factor_info
    
    def connect(self):
        super().connect()
        nPrefix = len(self._QSArgs.InnerPrefix)
        if self._QSArgs.DBType=="MySQL":
            SQLStr = f"""
            SELECT 
                RIGHT(t.TABLE_NAME, CHAR_LENGTH(t.TABLE_NAME)-{nPrefix}) AS TableName,
                t.TABLE_NAME AS DBTableName, 
                t.COLUMN_NAME AS DBFieldName, 
                LOWER(t.DATA_TYPE) AS DataType, 
                t.IS_NULLABLE AS Nullable, 
                t.COLUMN_KEY AS FieldKey, 
                t.COLUMN_COMMENT AS Description, 
                t1.TABLE_COMMENT AS TableDescription
            FROM information_schema.COLUMNS t 
            LEFT JOIN information_schema.TABLES t1 
            ON (t.TABLE_SCHEMA = t1.TABLE_SCHEMA AND t.TABLE_NAME = t1.TABLE_NAME) 
            WHERE t.TABLE_SCHEMA='{self._QSArgs.DBName}'
            AND t.TABLE_NAME LIKE '{self._QSArgs.InnerPrefix}%%'
            """
            if len(self._QSArgs.IgnoreFields)>0:
                SQLStr += "AND t.COLUMN_NAME NOT IN ('"+"','".join(self._QSArgs.IgnoreFields)+"') "
            SQLStr += "ORDER BY TableName, DBFieldName"
        elif self._QSArgs.DBType=="PostgreSQL":
            SQLStr = f"""
            SELECT 
                RIGHT(t.table_name, LENGTH(t.table_name) - {nPrefix}) AS "TableName",
                t.table_name AS "DBTableName",
                t.column_name AS "DBFieldName",
                LOWER(t.data_type) AS "DataType", 
                t.is_nullable AS "Nullable",
                CASE WHEN pk.column_name IS NOT NULL THEN 'PRI' ELSE '' END AS "FieldKey",
                col_description(c.oid, t.ordinal_position) AS "Description",
                obj_description(c.oid, 'pg_class') AS "TableDescription"
            FROM information_schema.columns t
            JOIN pg_class c ON c.relname = t.table_name
            JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.table_schema
            LEFT JOIN (
                SELECT 
                    kcu.table_schema,
                    kcu.table_name,
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name 
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk ON pk.table_schema = t.table_schema 
                AND pk.table_name = t.table_name 
                AND pk.column_name = t.column_name
            WHERE t.table_schema = 'public'
            AND t.table_name LIKE '{self._QSArgs.InnerPrefix}%'
            """
            if len(self._QSArgs.IgnoreFields)>0:
                SQLStr += "AND t.column_name NOT IN ('"+"','".join(self._QSArgs.IgnoreFields)+"') "
            SQLStr += "ORDER BY t.table_name, t.column_name"
        else:
            raise NotImplementedError("'%s' 调用方法 connect 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
        self._FactorInfo = pd.read_sql_query(SQLStr, self._Connection, index_col=None)
        self._TableInfo = self._FactorInfo.loc[:, ["TableName", "DBTableName", "TableDescription"]].copy().groupby(by=["TableName"], as_index=True).last().sort_index()
        self._TableInfo = self._TableInfo.rename(columns={"TableDescription": "Description"})
        self._TableInfo["TableClass"] = "WideTable"
        self._FactorInfo.pop("DBTableName")
        self._FactorInfo = self._genFactorInfo(self._FactorInfo)
        # 加载侧表元数据
        self._createMetaTable()
        self._loadMetaData()
        return self

    # 创建元数据侧表
    def _createMetaTable(self):
        DBMetaTableName = self._QSArgs.TablePrefix + self._QSArgs.InnerPrefix + self._QSArgs.MetaTableName
        if self._QSArgs.DBType == "MySQL":
            SQLStr = f"""
            CREATE TABLE IF NOT EXISTS {DBMetaTableName} (
                table_name VARCHAR(256) NOT NULL,
                field_name VARCHAR(256) NOT NULL DEFAULT '',
                meta_key VARCHAR(256) NOT NULL,
                meta_value TEXT,
                PRIMARY KEY (table_name, field_name, meta_key)
            )
            """
        elif self._QSArgs.DBType == "PostgreSQL":
            SQLStr = f"""
            CREATE TABLE IF NOT EXISTS {DBMetaTableName} (
                table_name VARCHAR(256) NOT NULL,
                field_name VARCHAR(256) NOT NULL DEFAULT '',
                meta_key VARCHAR(256) NOT NULL,
                meta_value TEXT,
                PRIMARY KEY (table_name, field_name, meta_key)
            )
            """
        else:
            raise NotImplementedError("'%s' 调用方法 _createMetaTable 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
        try:
            self.execute(SQLStr)
        except Exception as e:
            self._QS_Logger.warning("'%s' 创建元数据侧表失败: %s" % (self.Name, str(e)))

    # 从侧表加载元数据并合并到 _TableInfo 和 _FactorInfo
    def _loadMetaData(self):
        DBMetaTableName = self._QSArgs.TablePrefix + self._QSArgs.InnerPrefix + self._QSArgs.MetaTableName
        try:
            MetaData = pd.read_sql_query(f"SELECT * FROM {DBMetaTableName}", self._Connection)
        except Exception:
            return
        if MetaData.empty:
            return
        # 表级元数据 (field_name == '')
        TableMeta = MetaData[MetaData["field_name"] == ""].copy()
        if not TableMeta.empty:
            for _, row in TableMeta.iterrows():
                self._TableInfo.loc[row["table_name"], row["meta_key"]] = row["meta_value"]
        # 因子级元数据 (field_name != '')
        FactorMeta = MetaData[MetaData["field_name"] != ""].copy()
        if not FactorMeta.empty:
            for _, row in FactorMeta.iterrows():
                self._FactorInfo.loc[(row["table_name"], row["field_name"]), row["meta_key"]] = row["meta_value"]

    # 单条元数据 UPSERT / DELETE
    def _upsertMetaData(self, table_name:str, field_name:str, meta_key:str, meta_value):
        DBMetaTableName = self._QSArgs.TablePrefix + self._QSArgs.InnerPrefix + self._QSArgs.MetaTableName
        Cursor = self.cursor()
        try:
            if meta_value is None:
                SQLStr = f"DELETE FROM {DBMetaTableName} WHERE table_name = {self._PlaceHolder} AND field_name = {self._PlaceHolder} AND meta_key = {self._PlaceHolder}"
                Cursor.execute(SQLStr, (table_name, field_name, meta_key))
            elif self._QSArgs.DBType == "MySQL":
                SQLStr = f"REPLACE INTO {DBMetaTableName} (table_name, field_name, meta_key, meta_value) VALUES ({self._PlaceHolder}, {self._PlaceHolder}, {self._PlaceHolder}, {self._PlaceHolder})"
                Cursor.execute(SQLStr, (table_name, field_name, meta_key, str(meta_value)))
            elif self._QSArgs.DBType == "PostgreSQL":
                SQLStr = f"""INSERT INTO {DBMetaTableName} (table_name, field_name, meta_key, meta_value) VALUES ({self._PlaceHolder}, {self._PlaceHolder}, {self._PlaceHolder}, {self._PlaceHolder}) ON CONFLICT (table_name, field_name, meta_key) DO UPDATE SET meta_value = EXCLUDED.meta_value"""
                Cursor.execute(SQLStr, (table_name, field_name, meta_key, str(meta_value)))
            else:
                raise NotImplementedError("'%s' 调用方法 _upsertMetaData 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
            self.Connection.commit()
        except Exception as e:
            self._QS_Logger.warning("'%s' 写入元数据失败: %s" % (self.Name, str(e)))
        finally:
            Cursor.close()

    # 实现 WritableFactorDB 接口
    def setTableMetaData(self, table_name:str, key:Optional[str]=None, value:Any=None, meta_data:Optional[dict]=None):
        if meta_data is not None:
            meta_data = dict(meta_data)
        else:
            meta_data = {}
        if key is not None:
            meta_data[key] = value
        if not meta_data:
            return 0
        for k, v in meta_data.items():
            self._upsertMetaData(table_name, '', k, v)
            self._TableInfo.loc[table_name, k] = v
        return 0

    def setFactorMetaData(self, table_name:str, ifactor_name:str, key:Optional[str]=None, value:Any=None, meta_data:Optional[dict]=None):
        if meta_data is not None:
            meta_data = dict(meta_data)
        else:
            meta_data = {}
        if key is not None:
            meta_data[key] = value
        if not meta_data:
            return 0
        for k, v in meta_data.items():
            self._upsertMetaData(table_name, ifactor_name, k, v)
            self._FactorInfo.loc[(table_name, ifactor_name), k] = v
        return 0

    @property
    def TableNames(self) -> List[str]:
        return sorted(self._TableInfo.index)
    
    def _initFTArgs(self, table_name:str, args:dict) -> dict:
        if table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不存在因子表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        Args = self._QSArgs.FTArgs.copy()
        Args.update(args)
        # 确定时点字段和 ID 字段
        iFactorInfo = self._FactorInfo.loc[table_name]
        if "DTField" in Args:
            DTField = Args["DTField"]
        else:
            Mask = (iFactorInfo["FieldType"] == "Date")
            DTField = iFactorInfo.index[Mask & (iFactorInfo["Supplementary"]=="Default")]
            DTField = (DTField[0] if DTField.shape[0]>0 else (iFactorInfo.index[Mask][0] if Mask.any() else None))
        if "IDField" in Args:
            IDField = Args["IDField"]
        else:
            Mask = (iFactorInfo["FieldType"] == "ID")
            IDField = (iFactorInfo.index[Mask][0] if Mask.any() else None)
        # 确定因子表类型
        if "TableType" in Args:
            TableClass = Args["TableType"]
        elif ((DTField is not None) and (IDField is not None)) or ((DTField is None) and (IDField is None)):
            TableClass = self._TableInfo.loc[table_name, "TableClass"]
        elif DTField is None:
            TableClass = "FeatureTable"
        elif IDField is None:
            TableClass = "TimeSeriesTable"
        Args["TableType"] = TableClass
        # 确定多重映射参数
        PrimaryKeys = iFactorInfo[iFactorInfo["FieldKey"]=="PRI"].index
        Args.setdefault("MultiMapping", (PrimaryKeys.difference({DTField, IDField}).shape[0]>0))
        Args["Name"] = table_name
        return Args
    
    def getTable(self, table_name:str, args:dict={}) -> SQL_Table:
        Args = self._initFTArgs(table_name=table_name, args=args)
        return eval("SQL_"+Args["TableType"]+"(fdb=self, args=Args, table_info=self._TableInfo.loc[table_name], factor_info=self._FactorInfo.loc[table_name], logger=self._QS_Logger)")

    # region 表的操作
    def renameTable(self, old_table_name:str, new_table_name:str):
        if old_table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 不存在因子表 '%s'!" % (self.Name, old_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableInfo.index):
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 新因子表名 '%s' 已经存在于库中!" % (self.Name, new_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        self.renameDBTable(self._QSArgs.InnerPrefix+old_table_name, self._QSArgs.InnerPrefix+new_table_name)
        self._TableInfo = self._TableInfo.rename(index={old_table_name: new_table_name})
        self._FactorInfo = self._FactorInfo.rename(index={old_table_name: new_table_name}, level=0)
        # 级联更新侧表
        self._cascadeRenameTableMeta(old_table_name, new_table_name)

    # 级联更新侧表: 重命名表
    def _cascadeRenameTableMeta(self, old_table_name:str, new_table_name:str):
        DBMetaTableName = self._QSArgs.TablePrefix + self._QSArgs.InnerPrefix + self._QSArgs.MetaTableName
        try:
            Cursor = self.cursor()
            SQLStr = f"UPDATE {DBMetaTableName} SET table_name = {self._PlaceHolder} WHERE table_name = {self._PlaceHolder}"
            Cursor.execute(SQLStr, (new_table_name, old_table_name))
            self.Connection.commit()
            Cursor.close()
        except Exception as e:
            self._QS_Logger.warning("'%s' 级联更新元数据侧表失败: %s" % (self.Name, str(e)))

    # 创建表, field_types: {字段名: 数据库数据类型}
    def createTable(self, table_name:str, field_types:Dict[str, str]):
        FieldTypes = field_types.copy()
        if self._QSArgs.DBType=="MySQL":
            FieldTypes[self._QSArgs.DTField] = FieldTypes.pop(self._QSArgs.DTField, "DATETIME(6) NOT NULL")
            FieldTypes[self._QSArgs.IDField] = FieldTypes.pop(self._QSArgs.IDField, "VARCHAR(40) NOT NULL")
        elif self._QSArgs.DBType=="PostgreSQL":
            FieldTypes[self._QSArgs.DTField] = FieldTypes.pop(self._QSArgs.DTField, "TIMESTAMP NOT NULL")
            FieldTypes[self._QSArgs.IDField] = FieldTypes.pop(self._QSArgs.IDField, "VARCHAR(40) NOT NULL")
        else:
            raise NotImplementedError("'%s' 调用方法 createTable 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
        self.createDBTable(self._QSArgs.InnerPrefix+table_name, FieldTypes, primary_keys=[self._QSArgs.DTField, self._QSArgs.IDField], index_fields=[self._QSArgs.IDField])
        self._TableInfo = pd.concat([self._TableInfo, pd.Series([self._QSArgs.InnerPrefix+table_name, "WideTable"], index=["DBTableName", "TableClass"], name=table_name).to_frame().T])
        NewFactorInfo = pd.DataFrame(FieldTypes, index=["DataType"], columns=pd.Index(sorted(FieldTypes.keys()), name="DBFieldName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = pd.concat([self._FactorInfo, self._genFactorInfo(NewFactorInfo)])

    def deleteTable(self, table_name:str):
        if table_name not in self._TableInfo.index: return
        self.deleteDBTable(self._QSArgs.InnerPrefix+table_name)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._TableInfo = self._TableInfo.loc[TableNames]
        self._FactorInfo = self._FactorInfo.loc[TableNames]
        # 级联删除侧表
        self._cascadeDeleteTableMeta(table_name)

    # 级联删除侧表: 删除表
    def _cascadeDeleteTableMeta(self, table_name:str):
        DBMetaTableName = self._QSArgs.TablePrefix + self._QSArgs.InnerPrefix + self._QSArgs.MetaTableName
        try:
            Cursor = self.cursor()
            SQLStr = f"DELETE FROM {DBMetaTableName} WHERE table_name = {self._PlaceHolder}"
            Cursor.execute(SQLStr, (table_name,))
            self.Connection.commit()
            Cursor.close()
        except Exception as e:
            self._QS_Logger.warning("'%s' 级联删除元数据失败: %s" % (self.Name, str(e)))
    # endregion    
    
    # region 因子操作
    def addFactor(self, table_name:str, field_types:Dict[str, str]):
        """添加因子

        Args:
            table_name: 表名
            field_types: {字段名: 数据库数据类型}
        """
        if table_name not in self._TableInfo.index: return self.createTable(table_name, field_types)
        self.addField(self._QSArgs.InnerPrefix+table_name, field_types)
        NewFactorInfo = pd.DataFrame(field_types, index=["DataType"], columns=pd.Index(sorted(field_types.keys()), name="DBFieldName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = pd.concat([self._FactorInfo, self._genFactorInfo(NewFactorInfo)]).sort_index()

    def renameFactor(self, table_name:str, old_factor_name:str, new_factor_name:str):
        """重命名因子"""
        if old_factor_name not in self._FactorInfo.loc[table_name].index:
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 因子表 '%s' 中不存在因子 '%s'!" % (self.Name, table_name, old_factor_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_factor_name!=old_factor_name) and (new_factor_name in self._FactorInfo.loc[table_name].index):
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 新因子名 '%s' 已经存在于因子表 '%s' 中!" % (self.Name, new_factor_name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        self.renameField(self._QSArgs.InnerPrefix+table_name, old_factor_name, new_factor_name)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = pd.concat([self._FactorInfo.loc[TableNames], self._FactorInfo.loc[[table_name]].rename(index={old_factor_name: new_factor_name}, level=1)])
        # 级联更新侧表
        self._cascadeRenameFactorMeta(table_name, old_factor_name, new_factor_name)
    
    def deleteFactor(self, table_name:str, factor_names:List[str]):
        if (not factor_names) or (table_name not in self._TableInfo.index): return 0
        FactorIndex = self._FactorInfo.loc[table_name].index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        self.deleteField(self._QSArgs.InnerPrefix+table_name, factor_names)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        idx = pd.IndexSlice
        self._FactorInfo = pd.concat([self._FactorInfo.loc[TableNames], self._FactorInfo.loc[idx[table_name, FactorIndex], :]])
        # 级联删除侧表
        self._cascadeDeleteFactorMeta(table_name, factor_names)
    # endregion

    # 级联更新侧表: 重命名因子
    def _cascadeRenameFactorMeta(self, table_name:str, old_factor_name:str, new_factor_name:str):
        DBMetaTableName = self._QSArgs.TablePrefix + self._QSArgs.InnerPrefix + self._QSArgs.MetaTableName
        try:
            Cursor = self.cursor()
            SQLStr = f"UPDATE {DBMetaTableName} SET field_name = {self._PlaceHolder} WHERE table_name = {self._PlaceHolder} AND field_name = {self._PlaceHolder}"
            Cursor.execute(SQLStr, (new_factor_name, table_name, old_factor_name))
            self.Connection.commit()
            Cursor.close()
        except Exception as e:
            self._QS_Logger.warning("'%s' 级联更新因子元数据失败: %s" % (self.Name, str(e)))

    # 级联删除侧表: 删除因子
    def _cascadeDeleteFactorMeta(self, table_name:str, factor_names:List[str]):
        DBMetaTableName = self._QSArgs.TablePrefix + self._QSArgs.InnerPrefix + self._QSArgs.MetaTableName
        try:
            Cursor = self.cursor()
            SQLStr = f"DELETE FROM {DBMetaTableName} WHERE table_name = {self._PlaceHolder} AND field_name IN ({', '.join([self._PlaceHolder] * len(factor_names))})"
            Cursor.execute(SQLStr, [table_name] + factor_names)
            self.Connection.commit()
            Cursor.close()
        except Exception as e:
            self._QS_Logger.warning("'%s' 级联删除因子元数据失败: %s" % (self.Name, str(e)))

    # region 数据操作
    def deleteData(self, table_name:str, ids:Optional[List[str]]=None, dts:Optional[List[dt.datetime]]=None, dt_ids:Optional[List[Tuple[dt.datetime, str]]]=None):
        if table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 deleteData 错误: 不存在因子表 '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (ids is None) and (dts is None): return self.truncateDBTable(self._QSArgs.InnerPrefix+table_name)
        DBTableName = self._QSArgs.TablePrefix+self._QSArgs.InnerPrefix+table_name
        IDField = DBTableName+"."+self._QSArgs.IDField
        DTField = DBTableName+"."+self._QSArgs.DTField
        SQLStr = "DELETE FROM "+DBTableName+" "
        if dts is not None:
            DTs = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts]
            SQLStr += "WHERE ("+genSQLInCondition(DTField, DTs, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DTField+" IS NOT NULL "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(IDField, ids, is_str=True, max_num=1000)+") "
        if dt_ids is not None:
            dt_ids = ["('"+iDTIDs[0].strftime("%Y-%m-%d %H:%M:%S.%f")+"', '"+iDTIDs[1]+"')" for iDTIDs in dt_ids]
            SQLStr += "AND ("+genSQLInCondition("("+DTField+", "+IDField+")", dt_ids, is_str=False, max_num=1000)+")"
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteData 删除表 '%s' 中数据时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
    
    def _adjustWriteData(self, data, table_name):
        NewData = []
        DataLen = data.applymap(lambda x: max(1, len(x)) if isinstance(x, list) else 1)
        DataLenMax = DataLen.iloc[:, 2:].max(axis=1)
        DataLenMin = DataLen.iloc[:, 2:].min(axis=1)
        if (DataLenMax!=DataLenMin).sum()>0:
            self._QS_Logger.warning("'%s' 在写入因子 '%s' 时出现因子值长度不一致的情况, 将填充缺失!" % (self.Name, str(data.columns.tolist())))
        for i in range(data.shape[0]):
            iDataLen = DataLenMax.iloc[i]
            if iDataLen>0:
                iData = data.iloc[i].apply(lambda x: [None]*(iDataLen-len(x))+x if isinstance(x, list) else [x]*iDataLen).tolist()
                NewData.extend(zip(*iData))
        NewData = pd.DataFrame(NewData, columns=data.columns, dtype="O")
        if self._QSArgs.CheckNullable:
            NewData = self._dropWriteDataNa(NewData, table_name)
        # 将时点类型的数据转换成 str
        iDataType = self._FactorInfo["DataType"].loc[table_name]
        StrftimeFun = lambda d: d.strftime("%Y-%m-%d %H:%M:%S.%f") if isinstance(d, dt.datetime) else (d.strftime("%Y-%m-%d") if isinstance(d, dt.date) else None)
        for iFactorName in iDataType[iDataType.str.contains("date")].index:
            NewData[iFactorName] = NewData[iFactorName].apply(StrftimeFun)
        return NewData.where(pd.notnull(NewData), None).to_records(index=False).tolist()
    
    def _dropWriteDataNa(self, data, table_name):
        DropNaFields = self._FactorInfo["Nullable"].loc[table_name].loc[data.columns]
        DropNaFields = DropNaFields[DropNaFields=="NO"].index.tolist()
        if DropNaFields:
            OldRowNum = data.shape[0]
            data = data.dropna(subset=DropNaFields)
            if data.shape[0]<OldRowNum:
                self._QS_Logger.warning("因子库 %s 中的因子表 %s 中的字段 %s 不允许 NULL, 但写入数据中出现 NULL, 删除相应行后执行写入!" % (self.Name, table_name, str(DropNaFields)))
        return data
    
    def _genMySQLInsertSQL(self, table_name:str, fields:List[str], replace:bool=False, unique_fields:Optional[List[str]]=None) -> str:
        SQLStr = f"""{"REPLACE" if replace else "INSERT"} INTO {self._QSArgs.TablePrefix+self._QSArgs.InnerPrefix+table_name} (`{"`, `".join(fields)}`)
        VALUES ({", ".join([self._PlaceHolder] * len(fields))})
        """
        return SQLStr
    
    def _genPostgreSQLInsertSQL(self, table_name:str, fields:List[str], replace:bool=False, unique_fields:Optional[List[str]]=None) -> str:
        SQLStr = f"""INSERT INTO {self._QSArgs.TablePrefix+self._QSArgs.InnerPrefix+table_name} ({", ".join(fields)})
        VALUES ({", ".join([self._PlaceHolder] * len(fields))})
        """
        if replace:
            if not unique_fields:
                raise __QS_Error__("当 replace=True 时必须指定 unique_fields")
            SQLStr += f"""ON CONFLICT ({", ".join(unique_fields)}) DO UPDATE SET {", ".join(f"{iField}=EXCLUDED.{iField}" for iField in fields if iField not in unique_fields)}"""
        return SQLStr

    def writeData(self, data:Panel, table_name:str, if_exists:Literal["update", "replace", "append"]="update", data_type:Dict[str, Literal["double", "string", "object"]]={}, **kwargs):
        if table_name not in self._TableInfo.index:
            FieldTypes = {iFactorName:_identifyDataType(self._QSArgs.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            try:
                self.createTable(table_name, field_types=FieldTypes)
            except Exception as e:
                self.connect()
                if table_name not in self._TableInfo.index:
                    raise e
        else:
            NewFactorNames = data.items.difference(self._FactorInfo.loc[table_name].index).tolist()
            if NewFactorNames:
                FieldTypes = {iFactorName:_identifyDataType(self._QSArgs.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(NewFactorNames)}
                try:
                    self.addFactor(table_name, FieldTypes)
                except Exception as e:
                    self.connect()
                    if data.items.difference(self._FactorInfo.loc[table_name].index).shape[0]>0:
                        raise e
            if if_exists=="update":
                OldFactorNames = self._FactorInfo.loc[table_name].index.difference(data.items).difference({self._QSArgs.IDField, self._QSArgs.DTField}).tolist()
                if OldFactorNames:
                    if self._QSArgs.CheckWriteData:
                        OldData = self.getTable(table_name, args={"多重映射": True}).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    else:
                        OldData = self.getTable(table_name, args={"多重映射": False}).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    for iFactorName in OldFactorNames: data[iFactorName] = OldData[iFactorName]
            else:
                AllFactorNames = self._FactorInfo.loc[table_name].index.difference({self._QSArgs.IDField, self._QSArgs.DTField}).tolist()
                if self._QSArgs.CheckWriteData:
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
        DTs = data.major_axis
        # data.major_axis = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in DTs]
        data.major_axis = DTs.astype(str)
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        Mask = pd.notnull(NewData).any(axis=1)
        NewData = NewData[Mask]
        if NewData.shape[0]==0: return
        DimFields = [self._QSArgs.DTField, self._QSArgs.IDField]
        if self._QSArgs.DBType=="MySQL":
            SQLStr = self._genMySQLInsertSQL(table_name=table_name, fields=DimFields+list(data.items), replace=(table_name in self._TableInfo.index), unique_fields=DimFields)
        elif self._QSArgs.DBType=="PostgreSQL":
            SQLStr = self._genPostgreSQLInsertSQL(table_name=table_name, fields=DimFields+list(data.items), replace=(table_name in self._TableInfo.index), unique_fields=DimFields)
        else:
            raise NotImplementedError("'%s' 调用方法 writeData 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
        Cursor = self.cursor()
        if self._QSArgs.CheckWriteData:
            NewData = self._adjustWriteData(NewData.reset_index(), table_name)
            self.deleteData(table_name, ids=data.minor_axis.tolist(), dts=DTs.tolist())
            Cursor.executemany(SQLStr, NewData)
        else:
            NewData = NewData.astype("O").where(pd.notnull(NewData), None)
            if self._QSArgs.CheckNullable:
                NewData = self._dropWriteDataNa(NewData, table_name)
            # 将时点类型的数据转换成 str
            iDataType = self._FactorInfo["DataType"].loc[table_name]
            StrftimeFun = lambda d: d.strftime("%Y-%m-%d %H:%M:%S.%f") if isinstance(d, dt.datetime) else (d.strftime("%Y-%m-%d") if isinstance(d, dt.date) else None)
            for iFactorName in iDataType[iDataType.str.contains("date")].index:
                NewData[iFactorName] = NewData[iFactorName].apply(StrftimeFun)
            Cursor.executemany(SQLStr, NewData.reset_index().values.tolist())
        self.Connection.commit()
        Cursor.close()
    # endregion