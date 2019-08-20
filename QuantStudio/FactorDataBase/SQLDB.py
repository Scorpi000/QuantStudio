# coding=utf-8
"""基于 SQL 数据库的因子库"""
import re
import os
import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, File, Float, Bool

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.FileFun import readJSONFile
from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable

def _identifyDataType(db_type, dtypes):
    if db_type!="sqlite3":
        if np.dtype("O") in dtypes.values: return "varchar(40)"
        else: return "double"
    else:
        if np.dtype("O") in dtypes.values: return "text"
        else: return "real"

class _FactorTable(FactorTable):
    """SQLDB 因子表"""
    FilterCondition = Str("", arg_type="Dict", label="筛选条件", order=0)
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=1)
    ValueType = Enum("scalar", "list", "scalar or list", arg_type="SingleOption", label="因子值类型", order=2)
    def __init__(self, name, fdb, data_type, sys_args={}, **kwargs):
        self._DataType = data_type
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        return self._DataType.index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType": return self._DataType.loc[factor_names]
        if key is None: return pd.DataFrame(self._DataType.loc[factor_names], columns=["DataType"])
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB._Prefix+self.Name
        SQLStr = "SELECT DISTINCT "+DBTableName+".ID "
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None:
            SQLStr += "WHERE "+DBTableName+".DateTime='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            JoinStr = "AND "
        else:
            JoinStr = "WHERE "
        if ifactor_name is not None: SQLStr += JoinStr+DBTableName+"."+ifactor_name+" IS NOT NULL "
        SQLStr += "ORDER BY "+DBTableName+".ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB._Prefix+self.Name
        SQLStr = "SELECT DISTINCT "+DBTableName+".DateTime "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+DBTableName+".ID='"+iid+"' "
            JoinStr = "AND "
        else:
            JoinStr = "WHERE "
        if start_dt is not None:
            SQLStr += JoinStr+DBTableName+".DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            JoinStr = "AND "
        if end_dt is not None:
            SQLStr += JoinStr+DBTableName+".DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY "+DBTableName+".DateTime"
        if self._FactorDB.DBType!="sqlite3": return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        else: return [dt.datetime.strptime(iRslt[0], "%Y-%m-%d %H:%M:%S.%f") for iRslt in self._FactorDB.fetchall(SQLStr)]
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB._Prefix+self.Name
        SubSQLStr = "SELECT "+DBTableName+".ID, "
        SubSQLStr += "MAX("+DBTableName+".DateTime) "
        SubSQLStr += "FROM "+DBTableName+" "
        SubSQLStr += "WHERE "+DBTableName+".DateTime<'"+end_date.strftime("%Y-%m-%d:%H:%M:%S.%f")+"' "
        SubSQLStr += "AND ("+genSQLInCondition(DBTableName+".ID", ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SubSQLStr += "AND "+FilterStr+" "
        SubSQLStr += "GROUP BY "+DBTableName+".ID"
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".ID, "
        for iField in factor_names: SQLStr += DBTableName+"."+iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE ("+DBTableName+".ID, "+DBTableName+".DateTime) IN ("+SubSQLStr+") "
        if FilterStr: SQLStr += "AND "+FilterStr+" "
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (not dts) or (not ids): return pd.DataFrame(columns=["DateTime", "ID"]+factor_names)
        dts = sorted(dts)
        LookBack = args.get("回溯天数", self.LookBack)
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB._Prefix+self.Name
        # 形成 SQL 语句, 时点, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".ID, "
        for iField in factor_names: SQLStr += DBTableName+"."+iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        if (LookBack>0) or (len(dts)==1) or (np.nanmax(np.diff(dts)).days<=10):
            SQLStr += "WHERE "+DBTableName+".DateTime>='"+StartDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            SQLStr += "AND "+DBTableName+".DateTime<='"+EndDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        else:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        if len(ids)<=1000:
            SQLStr += "AND ("+genSQLInCondition(DBTableName+".ID", ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr+" "
        SQLStr += "ORDER BY "+DBTableName+".DateTime, "+DBTableName+".ID"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["DateTime", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData), columns=["DateTime", "ID"]+factor_names)
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["DateTime"]==dt.datetime.combine(StartDate, dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["DateTime", "ID"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["DateTime", "ID"])
        if self._FactorDB.DBType=="sqlite3": RawData["DateTime"] = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in RawData.pop("DateTime")]
        return RawData
    def _calcListData(self, raw_data, factor_names, ids, dts, args={}):
        Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        Data = pd.Panel(Data).loc[factor_names, :, ids]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, :]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, :]
        if np.isinf(LookBack):
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = Data.iloc[i].fillna(method="pad")
        else:
            Limits = LookBack*24.0*3600
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["DateTime", "ID"])
        ValueType = args.get("因子值类型", self.ValueType)
        if ValueType=="list":
            return self._calcListData(raw_data, factor_names, ids, dts, args=args)
        elif ValueType=="scalar":
            if not raw_data.index.is_unique:
                FilterStr = args.get("筛选条件", self.FilterCondition)
                raise __QS_Error__("筛选条件: '%s' 无法保证唯一性!" % FilterStr)
        else:
            if not raw_data.index.is_unique:
                return self._calcListData(raw_data, factor_names, ids, dts, args=args)
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names, :, ids]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, :]
        if np.isinf(LookBack):
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = Data.iloc[i].fillna(method="pad")
        else:
            Limits = LookBack*24.0*3600
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]


class SQLDB(WritableFactorDB):
    """SQLDB"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", "sqlite3", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "sqlite3", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    SQLite3File = File(label="sqlite3文件", arg_type="File", order=10)
    CheckWriteData = Bool(False, arg_type="Bool", label="检查写入值", order=11)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Connection = None# 数据库链接
        self._Connector = None# 实际使用的数据库链接器
        self._Prefix = "QS_"
        self._TableFactorDict = {}# {表名: pd.Series(数据类型, index=[因子名])}
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"SQLDBConfig.json" if config_file is None else config_file), **kwargs)
        self._PID = None# 保存数据库连接创建时的进程号
        self.Name = "SQLDB"
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_Connection"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._Connection: self._connect()
        else: self._Connection = None
    # -------------------------------------------数据库相关---------------------------
    def _connect(self):
        self._Connection = None
        if (self.Connector=="cx_Oracle") or ((self.Connector=="default") and (self.DBType=="Oracle")):
            try:
                import cx_Oracle
                self._Connection = cx_Oracle.connect(self.User, self.Pwd, cx_Oracle.makedsn(self.IPAddr, str(self.Port), self.DBName))
            except Exception as e:
                if self.Connector!="default": raise e
            else:
                self._Connector = "cx_Oracle"
        elif (self.Connector=="pymssql") or ((self.Connector=="default") and (self.DBType=="SQL Server")):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                if self.Connector!="default": raise e
            else:
                self._Connector = "pymssql"
        elif (self.Connector=="mysql.connector") or ((self.Connector=="default") and (self.DBType=="MySQL")):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet, autocommit=True)
            except Exception as e:
                if self.Connector!="default": raise e
            else:
                self._Connector = "mysql.connector"
        elif (self.Connector=="sqlite3") or ((self.Connector=="default") and (self.DBType=="sqlite3")):
            import sqlite3
            self._Connection = sqlite3.connect(self.SQLite3File)
            self._Connector = "sqlite3"
        if self._Connection is None:
            if self.Connector not in ("default", "pyodbc"):
                self._Connection = None
                raise __QS_Error__("不支持该连接器(connector) : "+self.Connector)
            else:
                import pyodbc
                if self.DSN: self._Connection = pyodbc.connect("DSN=%s;PWD=%s" % (self.DSN, self.Pwd))
                else: self._Connection = pyodbc.connect("DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s" % (self.DBType, self.DBName, self.IPAddr+","+str(self.Port), self.User, self.Pwd))
                self._Connector = "pyodbc"
        self._PID = os.getpid()
        return 0
    def connect(self):
        self._connect()
        nPrefix = len(self._Prefix)
        if self._Connector=="sqlite3":
            SQLStr = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%s%%' ORDER BY name"
            Cursor = self.cursor(SQLStr % self._Prefix)
            AllTables = Cursor.fetchall()
            self._TableFactorDict = {}
            for iTableName in AllTables:
                iTableName = iTableName[0][nPrefix:]
                Cursor.execute("PRAGMA table_info([%s])" % self._Prefix+iTableName)
                iDataType = np.array(Cursor.fetchall())
                iDataType = pd.Series(iDataType[:, 2], index=iDataType[:, 1])
                iDataType[iDataType=="text"] = "string"
                iDataType[iDataType=="real"] = "double"
                iDataType = iDataType[(iDataType.index!="ID") & (iDataType.index!="DateTime")]
                if iDataType.shape[0]>0: self._TableFactorDict[iTableName] = iDataType
        elif self.DBType=="MySQL":
            SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE table_schema='%s' " % self.DBName)
            SQLStr += ("AND TABLE_NAME LIKE '%s%%' " % self._Prefix)
            SQLStr += "AND COLUMN_NAME NOT IN ('ID', 'DateTime') "
            SQLStr += "ORDER BY TABLE_NAME, COLUMN_NAME"
            Rslt = self.fetchall(SQLStr)
            if not Rslt: self._TableFactorDict = {}
            else:
                self._TableFactorDict = pd.DataFrame(np.array(Rslt), columns=["表", "因子", "DataType"]).set_index(["表", "因子"])["DataType"]
                Mask = (self._TableFactorDict=="varchar")
                self._TableFactorDict[Mask] = "string"
                self._TableFactorDict[~Mask] = "double"
                self._TableFactorDict = {iTable[nPrefix:]:self._TableFactorDict.loc[iTable] for iTable in self._TableFactorDict.index.levels[0]}
        return 0
    def disconnect(self):
        if self._Connection is not None:
            try:
                self._Connection.close()
            except Exception as e:
                raise e
            finally:
                self._Connection = None
        return 0
    def isAvailable(self):
        return (self._Connection is not None)
    def cursor(self, sql_str=None):
        if self._Connection is None: raise __QS_Error__("%s尚未连接!" % self.__doc__)
        if os.getpid()!=self._PID: self.connect()# 如果进程号发生变化, 重连
        Cursor = self._Connection.cursor()
        if sql_str is None: return Cursor
        Cursor.execute(sql_str)
        return Cursor
    def fetchall(self, sql_str):
        Cursor = self.cursor(sql_str=sql_str)
        Data = Cursor.fetchall()
        Cursor.close()
        return Data
    def execute(self, sql_str):
        if self._Connection is None: raise __QS_Error__("%s尚未连接!" % self.__doc__)
        if os.getpid()!=self._PID: self.connect()# 如果进程号发生变化, 重连
        Cursor = self._Connection.cursor()
        Cursor.execute(sql_str)
        self._Connection.commit()
        Cursor.close()
        return 0
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        return sorted(self._TableFactorDict)
    def getTable(self, table_name, args={}):
        if table_name not in self._TableFactorDict: raise __QS_Error__("表 '%s' 不存在!" % table_name)
        return _FactorTable(name=table_name, fdb=self, data_type=self._TableFactorDict[table_name], sys_args=args)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableFactorDict: raise __QS_Error__("表: '%s' 不存在!" % old_table_name)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableFactorDict): raise __QS_Error__("表: '"+new_table_name+"' 已存在!")
        SQLStr = "ALTER TABLE "+self.TablePrefix+self._Prefix+old_table_name+" RENAME TO "+self.TablePrefix+self._Prefix+new_table_name
        self.execute(SQLStr)
        self._TableFactorDict[new_table_name] = self._TableFactorDict.pop(old_table_name)
        return 0
    # 为某张表增加索引
    def addIndex(self, index_name, table_name, fields=["DateTime", "ID"], index_type="BTREE"):
        if index_type is not None:
            SQLStr = "CREATE INDEX "+index_name+" USING "+index_type+" ON "+self.TablePrefix+self._Prefix+table_name+"("+", ".join(fields)+")"
        else:
            SQLStr = "CREATE INDEX "+index_name+" ON "+self.TablePrefix+self._Prefix+table_name+"("+", ".join(fields)+")"
        return self.execute(SQLStr)
    # 创建表, field_types: {字段名: 数据类型}
    def createTable(self, table_name, field_types):
        if self.DBType=="MySQL":
            SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` DATETIME(6) NOT NULL, `ID` VARCHAR(40) NOT NULL, " % (self.TablePrefix+self._Prefix+table_name)
            for iField in field_types: SQLStr += "`%s` %s, " % (iField, field_types[iField])
            SQLStr += "PRIMARY KEY (`DateTime`, `ID`)) ENGINE=InnoDB DEFAULT CHARSET=utf8"
            IndexType = "BTREE"
        elif self.DBType=="sqlite3":
            SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` text NOT NULL, `ID` text NOT NULL, " % (self.TablePrefix+self._Prefix+table_name)
            for iField in field_types: SQLStr += "`%s` %s, " % (iField, field_types[iField])
            SQLStr += "PRIMARY KEY (`DateTime`, `ID`))"
            IndexType = None
        self.execute(SQLStr)
        try:
            self.addIndex(table_name+"_index", table_name, index_type=IndexType)
        except Exception as e:
            print("索引创建失败: "+str(e))
        return 0
    # 增加字段，field_types: {字段名: 数据类型}
    def addField(self, table_name, field_types):
        if table_name not in self._TableFactorDict: return self.createTable(table_name, field_types)
        SQLStr = "ALTER TABLE %s " % (self.TablePrefix+self._Prefix+table_name)
        SQLStr += "ADD COLUMN ("
        for iField in field_types: SQLStr += "%s %s," % (iField, field_types[iField])
        SQLStr = SQLStr[:-1]+")"
        self.execute(SQLStr)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableFactorDict: return 0
        SQLStr = 'DROP TABLE %s' % (self.TablePrefix+self._Prefix+table_name)
        self.execute(SQLStr)
        self._TableFactorDict.pop(table_name, None)
        return 0
    # 清空表
    def truncateTable(self, table_name):
        if table_name not in self._TableFactorDict: raise __QS_Error__("表: '%s' 不存在!" % table_name)
        SQLStr = "TRUNCATE TABLE %s" % (self.TablePrefix+self._Prefix+table_name)
        self.execute(SQLStr)
        return 0
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name not in self._TableFactorDict[table_name]: raise __QS_Error__("因子: '%s' 不存在!" % old_factor_name)
        if (new_factor_name!=old_factor_name) and (new_factor_name in self._TableFactorDict[table_name]): raise __QS_Error__("表中的因子: '%s' 已存在!" % new_factor_name)
        if self.DBType!="sqlite3":
            SQLStr = "ALTER TABLE "+self.TablePrefix+self._Prefix+table_name
            SQLStr += " CHANGE COLUMN `"+old_factor_name+"` `"+new_factor_name+"`"
            self.execute(SQLStr)
        else:
            # 将表名改为临时表
            SQLStr = "ALTER TABLE %s RENAME TO %s"
            TempTableName = genAvailableName("TempTable", self.TableNames)
            self.execute(SQLStr % (self.TablePrefix+self._Prefix+table_name, self.TablePrefix+self._Prefix+TempTableName))
            # 创建新表
            FieldTypes = OrderedDict()
            for iFactorName, iDataType in self._TableFactorDict[table_name].items():
                iDataType = ("text" if iDataType=="string" else "real")
                if iFactorName==old_factor_name: FieldTypes[new_factor_name] = iDataType
                else: FieldTypes[iFactorName] = iDataType
            self.createTable(table_name, field_types=FieldTypes)
            # 导入数据
            OldFactorNames = ", ".join(self._TableFactorDict[table_name].index)
            NewFactorNames = ", ".join(FieldTypes)
            SQLStr = "INSERT INTO %s (DateTime, ID, %s) SELECT DateTime, ID, %s FROM %s"
            Cursor = self.cursor(SQLStr % (self.TablePrefix+self._Prefix+table_name, NewFactorNames, OldFactorNames, self.TablePrefix+self._Prefix+TempTableName))
            self._Connection.commit()
            # 删除临时表
            Cursor.execute("DROP TABLE %s" % (self.TablePrefix+self._Prefix+TempTableName, ))
            self._Connection.commit()
            Cursor.close()
        self._TableFactorDict[table_name][new_factor_name] = self._TableFactorDict[table_name].pop(old_factor_name)
        return 0
    def deleteFactor(self, table_name, factor_names):
        if not factor_names: return 0
        FactorIndex = self._TableFactorDict.get(table_name, pd.Series()).index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        if self.DBType!="sqlite3":
            SQLStr = "ALTER TABLE "+self.TablePrefix+self._Prefix+table_name
            for iFactorName in factor_names: SQLStr += " DROP COLUMN `"+iFactorName+"`,"
            self.execute(SQLStr[:-1])
        else:
            # 将表名改为临时表
            SQLStr = "ALTER TABLE %s RENAME TO %s"
            TempTableName = genAvailableName("TempTable", self.TableNames)
            self.execute(SQLStr % (self.TablePrefix+self._Prefix+table_name, self.TablePrefix+self._Prefix+TempTableName))
            # 创建新表
            FieldTypes = OrderedDict()
            for iFactorName in FactorIndex:
                FieldTypes[iFactorName] = ("text" if self._TableFactorDict[table_name].loc[iFactorName]=="string" else "real")
            self.createTable(table_name, field_types=FieldTypes)
            # 导入数据
            FactorNameStr = ", ".join(FactorIndex)
            SQLStr = "INSERT INTO %s (DateTime, ID, %s) SELECT DateTime, ID, %s FROM %s"
            Cursor = self.cursor(SQLStr % (self.TablePrefix+self._Prefix+table_name, FactorNameStr, FactorNameStr, self.TablePrefix+self._Prefix+TempTableName))
            self._Connection.commit()
            # 删除临时表
            Cursor.execute("DROP TABLE %s" % (self.TablePrefix+self._Prefix+TempTableName, ))
            self._Connection.commit()
            Cursor.close()
        self._TableFactorDict[table_name] = self._TableFactorDict[table_name][FactorIndex]
        return 0
    def deleteData(self, table_name, ids=None, dts=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        if (self.DBType!="sqlite3") and (ids is None) and (dts is None):
            SQLStr = "TRUNCATE TABLE "+DBTableName
            return self.execute(SQLStr)
        SQLStr = "DELETE * FROM "+DBTableName
        if dts is not None:
            DTs = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts]
            SQLStr += "WHERE "+genSQLInCondition(DBTableName+".DateTime", DTs, is_str=True, max_num=1000)+" "
        else:
            SQLStr += "WHERE "+DBTableName+".DateTime IS NOT NULL "
        if ids is not None:
            SQLStr += "AND "+genSQLInCondition(DBTableName+".ID", ids, is_str=True, max_num=1000)
        return self.execute(SQLStr)
    def _adjustWriteData(self, data):
        NewData = []
        DataLen = data.applymap(lambda x: len(x) if isinstance(x, list) else 1).max(axis=1)
        for i in range(data.shape[0]):
            iDataLen = DataLen.iloc[i]
            iData = data.iloc[i].apply(lambda x: x * int(np.ceil(iDataLen / len(x))) if isinstance(x, list) else [x]*iDataLen).tolist()
            NewData.extend(zip(*iData))
        return NewData
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
        if table_name not in self._TableFactorDict:
            self.createTable(table_name, field_types=FieldTypes)
            self._TableFactorDict[table_name] = pd.Series({iFactorName: ("string" if FieldTypes[iFactorName].find("char")!=-1 else "double") for iFactorName in FieldTypes})
            SQLStr = "INSERT INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `ID`, "
        else:
            NewFactorNames = data.items.difference(self._TableFactorDict[table_name].index).tolist()
            if NewFactorNames:
                self.addField(table_name, {iFactorName:FieldTypes[iFactorName] for iFactorName in NewFactorNames})
                NewDataType = pd.Series({iFactorName: ("string" if FieldTypes[iFactorName].find("char")!=-1 else "double") for iFactorName in NewFactorNames})
                self._TableFactorDict[table_name] = self._TableFactorDict[table_name].append(NewDataType)
            AllFactorNames = self._TableFactorDict[table_name].index.tolist()
            if self.CheckWriteData:
                OldData = self.getTable(table_name, args={"因子值类型":"list"}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
            else:
                OldData = self.getTable(table_name).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
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
            SQLStr = "REPLACE INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `ID`, "
        if self.DBType=="sqlite3":
            data.major_axis = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in data.major_axis]
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
            SQLStr += "`"+iFactorName+"`, "
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        NewData = NewData[pd.notnull(NewData).any(axis=1)]
        if NewData.shape[0]==0: return 0
        NewData = NewData.astype("O").where(pd.notnull(NewData), None)
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
            Cursor.executemany(SQLStr, NewData.reset_index().values.tolist())
        self._Connection.commit()
        Cursor.close()
        return 0