# coding=utf-8
"""基于 SQL 数据库的因子库"""
import os

import numpy as np
import pandas as pd
from traits.api import on_trait_change, Str, Bool, ListStr, Dict

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.QSObjects import QSSQLObject
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB
from QuantStudio.FactorDataBase.FDBFun import SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable

def _identifyDataType(db_type, dtypes):
    if db_type!="sqlite3":
        if np.dtype("O") in dtypes.values: return "varchar(40)"
        else: return "double"
    else:
        if np.dtype("O") in dtypes.values: return "text"
        else: return "real"


class _WideTable(SQL_WideTable):
    """SQLDB 宽因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _NarrowTable(SQL_NarrowTable):
    """SQLDB 窄因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _FeatureTable(SQL_FeatureTable):
    """SQLDB 特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _TimeSeriesTable(SQL_TimeSeriesTable):
    """SQLDB 时序因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _MappingTable(SQL_MappingTable):
    """SQLDB 映射因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class SQLDB(QSSQLObject, WritableFactorDB):
    """SQLDB"""
    class __QS_ArgClass__(QSSQLObject.__QS_ArgClass__, WritableFactorDB.__QS_ArgClass__):
        Name = Str("SQLDB", arg_type="String", label="名称", order=-100)
        CheckWriteData = Bool(False, arg_type="Bool", label="检查写入值", order=100)
        IgnoreFields = ListStr(arg_type="List", label="忽略字段", order=101)
        InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=102)
        FTArgs = Dict(label="因子表参数", arg_type="Dict", order=103)
        DTField = Str("datetime", arg_type="String", label="时点字段", order=104)
        IDField = Str("code", arg_type="String", label="ID字段", order=105)
        CheckNullable = Bool(False, arg_type="Bool", label="检查缺失容许", order=106)
        @on_trait_change("DTField")
        def _on_DTField_changed(self, obj, name, old, new):
            if self._Owner._FactorInfo.shape[0]>0:
                self._Owner._FactorInfo["Supplementary"][(self._Owner._FactorInfo["FieldType"]=="Date") & (self._Owner._FactorInfo["Supplementary"]=="Default")] = None
                self._Owner._FactorInfo["Supplementary"][(self._Owner._FactorInfo["FieldType"]=="Date") & (self._Owner._FactorInfo["DBFieldName"]==new)] = "Default"
        @on_trait_change("IDField")
        def _on_IDField_changed(self, obj, name, old, new):
            if self._Owner._FactorInfo.shape[0]>0:
                self._Owner._FactorInfo["FieldType"][self._Owner._FactorInfo["FieldType"]=="ID"] = None
                self._Owner._FactorInfo["FieldType"][self._Owner._FactorInfo["DBFieldName"]==new] = "ID"
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        #self._TableFactorDict = {}# {表名: pd.Series(数据类型, index=[因子名])}
        #self._TableFieldDataType = {}# {表名: pd.Series(数据库数据类型, index=[因子名])}
        self._TableInfo = pd.DataFrame()# DataFrame(index=[表名], columns=["DBTableName", "TableClass"])
        self._FactorInfo = pd.DataFrame()# DataFrame(index=[(表名,因子名)], columns=["DBFieldName", "DataType", "FieldType", "Supplementary", "Description"])
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"SQLDBConfig.json" if config_file is None else config_file), **kwargs)
        return
    # factor_info: DataFrame(columns=["TableName", "DBFieldName", "FieldType", "Supplementary", "DataType", "Nullable", "FieldKey", "Description"])
    def _genFactorInfo(self, factor_info):
        factor_info["FieldName"] = factor_info["DBFieldName"]
        factor_info["FieldType"] = "因子"
        DTMask = factor_info["DataType"].str.contains("date")
        factor_info["FieldType"][DTMask] = "Date"
        StrMask = (factor_info["DataType"].str.contains("char") | factor_info["DataType"].str.contains("text"))
        factor_info["FieldType"][(factor_info["DBFieldName"].str.lower()==self._QSArgs.IDField) & StrMask] = "ID"
        factor_info["Supplementary"] = None
        factor_info["Supplementary"][DTMask & (factor_info["DBFieldName"].str.lower()==self._QSArgs.DTField)] = "Default"
        factor_info["Description"] = ""
        factor_info = factor_info.set_index(["TableName", "FieldName"])
        return factor_info
    def connect(self):
        super().connect()
        nPrefix = len(self._QSArgs.InnerPrefix)
        if self._QSArgs.DBType=="MySQL":
            SQLStr = f"SELECT RIGHT(t.TABLE_NAME, CHAR_LENGTH(t.TABLE_NAME)-{nPrefix}) AS TableName, t.TABLE_NAME AS DBTableName, t.COLUMN_NAME AS DBFieldName, LOWER(t.DATA_TYPE) AS DataType, t.IS_NULLABLE AS Nullable, t.COLUMN_KEY AS FieldKey, t.COLUMN_COMMENT AS Description, t1.TABLE_COMMENT AS TableDescription "
            SQLStr += f"FROM information_schema.COLUMNS t LEFT JOIN information_schema.TABLES t1 ON (t.TABLE_SCHEMA = t1.TABLE_SCHEMA AND t.TABLE_NAME = t1.TABLE_NAME) "
            SQLStr += f"WHERE t.TABLE_SCHEMA='{self._QSArgs.DBName}' "
            SQLStr += f"AND t.TABLE_NAME LIKE '{self._QSArgs.InnerPrefix}%%' "
            if len(self._QSArgs.IgnoreFields)>0:
                SQLStr += "AND t.COLUMN_NAME NOT IN ('"+"','".join(self._QSArgs.IgnoreFields)+"') "
            SQLStr += "ORDER BY TableName, DBFieldName"
        else:
            raise NotImplementedError("'%s' 调用方法 connect 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
        self._FactorInfo = pd.read_sql_query(SQLStr, self._Connection, index_col=None)
        self._TableInfo = self._FactorInfo.loc[:, ["TableName", "DBTableName", "TableDescription"]].copy().groupby(by=["TableName"], as_index=True).last().sort_index()
        self._TableInfo = self._TableInfo.rename(columns={"TableDescription": "Description"})
        self._TableInfo["TableClass"] = "WideTable"
        self._FactorInfo.pop("DBTableName")
        self._FactorInfo = self._genFactorInfo(self._FactorInfo)
        return 0
    @property
    def TableNames(self):
        return sorted(self._TableInfo.index)
    def __QS_initFTArgs__(self, table_name, args):
        if table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不存在因子表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        Args = self._QSArgs.FTArgs.copy()
        Args.update(args)
        # 确定时点字段和 ID 字段
        iFactorInfo = self._FactorInfo.loc[table_name]
        if "时点字段" in Args:
            DTField = Args["时点字段"]
        else:
            Mask = (iFactorInfo["FieldType"] == "Date")
            DTField = iFactorInfo.index[Mask & (iFactorInfo["Supplementary"]=="Default")]
            DTField = (DTField[0] if DTField.shape[0]>0 else (iFactorInfo.index[Mask][0] if Mask.any() else None))
        if "ID字段" in Args:
            IDField = Args["ID字段"]
        else:
            Mask = (iFactorInfo["FieldType"] == "ID")
            IDField = (iFactorInfo.index[Mask][0] if Mask.any() else None)
        # 确定因子表类型
        if "因子表类型" in Args:
            TableClass = Args["因子表类型"]
        elif ((DTField is not None) and (IDField is not None)) or ((DTField is None) and (IDField is None)):
            TableClass = self._TableInfo.loc[table_name, "TableClass"]
        elif DTField is None:
            TableClass = "FeatureTable"
        elif IDField is None:
            TableClass = "TimeSeriesTable"
        Args["因子表类型"] = TableClass
        # 确定多重映射参数
        PrimaryKeys = iFactorInfo[iFactorInfo["FieldKey"]=="PRI"].index
        Args.setdefault("多重映射", (PrimaryKeys.difference({DTField, IDField}).shape[0]>0))
        return Args
    def getTable(self, table_name, args={}):
        Args = self.__QS_initFTArgs__(table_name=table_name, args=args)
        return eval("_"+Args["因子表类型"]+"(name='"+table_name+"', fdb=self, sys_args=Args, logger=self._QS_Logger)")
    def renameTable(self, old_table_name, new_table_name):
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
        return 0
    # 创建表, field_types: {字段名: 数据库数据类型}
    def createTable(self, table_name, field_types):
        FieldTypes = field_types.copy()
        if self._QSArgs.DBType=="MySQL":
            FieldTypes[self._QSArgs.DTField] = FieldTypes.pop(self._QSArgs.DTField, "DATETIME(6) NOT NULL")
            FieldTypes[self._QSArgs.IDField] = FieldTypes.pop(self._QSArgs.IDField, "VARCHAR(40) NOT NULL")
        else:
            raise NotImplementedError("'%s' 调用方法 createTable 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
        self.createDBTable(self._QSArgs.InnerPrefix+table_name, FieldTypes, primary_keys=[self._QSArgs.DTField, self._QSArgs.IDField], index_fields=[self._QSArgs.IDField])
        self._TableInfo = self._TableInfo.append(pd.Series([self._QSArgs.InnerPrefix+table_name, "WideTable"], index=["DBTableName", "TableClass"], name=table_name))
        NewFactorInfo = pd.DataFrame(FieldTypes, index=["DataType"], columns=pd.Index(sorted(FieldTypes.keys()), name="DBFieldName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = self._FactorInfo.append(self._genFactorInfo(NewFactorInfo))
        return 0
    # 增加因子，field_types: {字段名: 数据库数据类型}
    def addFactor(self, table_name, field_types):
        if table_name not in self._TableInfo.index: return self.createTable(table_name, field_types)
        self.addField(self._QSArgs.InnerPrefix+table_name, field_types)
        NewFactorInfo = pd.DataFrame(field_types, index=["DataType"], columns=pd.Index(sorted(field_types.keys()), name="DBFieldName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = self._FactorInfo.append(self._genFactorInfo(NewFactorInfo)).sort_index()
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableInfo.index: return 0
        self.deleteDBTable(self._QSArgs.InnerPrefix+table_name)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._TableInfo = self._TableInfo.loc[TableNames]
        self._FactorInfo = self._FactorInfo.loc[TableNames]
        return 0
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
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
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].rename(index={old_factor_name: new_factor_name}, level=1))
        return 0
    def deleteFactor(self, table_name, factor_names):
        if (not factor_names) or (table_name not in self._TableInfo.index): return 0
        FactorIndex = self._FactorInfo.loc[table_name].index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        self.deleteField(self._QSArgs.InnerPrefix+table_name, factor_names)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].loc[FactorIndex])
        return 0
    def deleteData(self, table_name, ids=None, dts=None, dt_ids=None):
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
        return 0
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
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._TableInfo.index:
            FieldTypes = {iFactorName:_identifyDataType(self._QSArgs.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            try:
                self.createTable(table_name, field_types=FieldTypes)
            except Exception as e:
                self.connect()
                if table_name not in self._TableInfo.index:
                    raise e
            SQLStr = f"INSERT INTO {self._QSArgs.TablePrefix+self._QSArgs.InnerPrefix+table_name} (`{self._QSArgs.DTField}`, `{self._QSArgs.IDField}`, "
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
            SQLStr = f"REPLACE INTO {self._QSArgs.TablePrefix+self._QSArgs.InnerPrefix+table_name} (`{self._QSArgs.DTField}`, `{self._QSArgs.IDField}`, "
        DTs = data.major_axis
        # data.major_axis = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in DTs]
        data.major_axis = DTs.astype(str)
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
            SQLStr += "`"+iFactorName+"`, "
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        Mask = pd.notnull(NewData).any(axis=1)
        NewData = NewData[Mask]
        if NewData.shape[0]==0: return 0
        SQLStr = SQLStr[:-2] + ") VALUES (" + (self._PlaceHolder+", ") * (NewData.shape[1]+2)
        SQLStr = SQLStr[:-2]+")"
        Cursor = self.cursor()
        if self._QSArgs.CheckWriteData:
            NewData = self._adjustWriteData(NewData.reset_index(), table_name)
            self.deleteData(table_name, ids=data.minor_axis.tolist(), dts=DTs.tolist())
            Cursor.executemany(SQLStr, NewData)
        else:
            NewData = NewData.astype("O").where(pd.notnull(NewData), None)
            if self._QSArgs.CheckNullable:
                NewData = self._dropWriteDataNa(NewData, table_name)
            Cursor.executemany(SQLStr, NewData.reset_index().values.tolist())
        self.Connection.commit()
        Cursor.close()
        return 0