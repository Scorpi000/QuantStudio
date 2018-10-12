# coding=utf-8
"""基于关系数据库的风险数据库"""
import os
import datetime as dt
import json

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password

from QuantStudio.Tools.DateTimeFun import cutDateTime
from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from .RiskDataBase import RiskDataBase, FactorRDB
from QuantStudio import __QS_Object__, __QS_Error__, __QS_LibPath__

class SQLRDB(RiskDataBase):
    """基于关系数据库的风险数据库"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("shuntai11", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableNames = []# [表名]
        self._Prefix = "QSR_"
        self._Connection = None
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"SQLRDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "SQLRDB"
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_Connection"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._Connection: self._connect()
        else: self._Connection = None
    def _connect(self):
        if (self.Connector=='cx_Oracle') or ((self.Connector=='default') and (self.DBType=='Oracle')):
            try:
                import cx_Oracle
                self._Connection = cx_Oracle.connect(self.User, self.Pwd, cx_Oracle.makedsn(self.IPAddr, str(self.Port), self.DBName))
                self.Connector = "cx_Oracle"
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='pymssql') or ((self.Connector=='default') and (self.DBType=='SQL Server')):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
                self.Connector = "pymssql"
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='mysql.connector') or ((self.Connector=='default') and (self.DBType=='MySQL')):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet, autocommit=True)
                self.Connector = "mysql.connector"
            except Exception as e:
                if self.Connector!='default': raise e
        else:
            if self.Connector not in ('default', 'pyodbc'):
                self._Connection = None
                raise __QS_Error__("不支持该连接器(connector) : "+self.Connector)
            else:
                import pyodbc
                if self.DSN:
                    self._Connection = pyodbc.connect('DSN=%s;PWD=%s' % (self.DSN, self.Pwd))
                else:
                    self._Connection = pyodbc.connect('DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s' % (self.DBType, self.DBName, self.IPAddr, self.User, self.Pwd))
                self.Connector = "pyodbc"
        return 0
    def connect(self):
        self._connect()
        nPrefix = len(self._Prefix)
        if self.DBType=="MySQL":
            SQLStr = ("SELECT TABLE_NAME FROM information_schema.COLUMNS WHERE table_schema='%s' " % self.DBName)
            SQLStr += ("AND TABLE_NAME LIKE '%s%%' " % self._Prefix)
            SQLStr += "ORDER BY TABLE_NAME"
            self._TableNames = [iRslt[0][nPrefix:] for iRslt in self.fetchall(SQLStr)]
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
        Cursor = self._Connection.cursor()
        Cursor.execute(sql_str)
        self._Connection.commit()
        Cursor.close()
        return 0
    @property
    def TableNames(self):
        return self._TableNames
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableNames: raise __QS_Error__("表: '%s' 不存在!" % old_table_name)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableNames): raise __QS_Error__("表: '"+new_table_name+"' 已存在!")
        SQLStr = "ALTER TABLE "+self.TablePrefix+self._Prefix+old_table_name+" RENAME TO "+self.TablePrefix+self._Prefix+new_table_name
        self.execute(SQLStr)
        self._TableNames[self._TableNames.index(old_table_name)] = new_table_name
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableNames: return 0
        SQLStr = 'DROP TABLE %s' % (self.TablePrefix+self._Prefix+table_name)
        self.execute(SQLStr)
        self._TableNames.remove(table_name)
        return 0
    def getTableDateTime(self, table_name, start_dt=None, end_dt=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "SELECT DISTINCT "+DBTableName+".DateTime "
        SQLStr += "FROM "+DBTableName+" "
        JoinStr = "WHERE "
        if start_dt is not None:
            SQLStr += JoinStr+DBTableName+".DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            JoinStr = "AND "
        if end_dt is not None:
            SQLStr += JoinStr+DBTableName+".DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY "+DBTableName+".DateTime"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def deleteDateTime(self, table_name, dts):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        if dts is None:
            SQLStr = "TRUNCATE TABLE "+DBTableName
            return self.execute(SQLStr)
        SQLStr = "DELETE * FROM "+DBTableName
        DTs = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts]
        SQLStr += "WHERE "+genSQLInCondition(DBTableName+".DateTime", DTs, is_str=True, max_num=1000)+" "
        return self.execute(SQLStr)
    def readCov(self, table_name, dts, ids=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        # 形成 SQL 语句, DateTime, IDs, Cov
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".IDs, "
        SQLStr += DBTableName+".Cov "
        SQLStr += " FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iIDs, iCov in self.fetchall(SQLStr):
            iIDs = json.loads(iIDs)
            iCov = pd.DataFrame(json.loads(iCov), index=iIDs, columns=iIDs)
            if ids is not None:
                if iCov.index.intersection(ids).shape[0]>0: iCov = iCov.loc[ids, ids]
                else: iCov = pd.DataFrame(index=ids, columns=ids)
            Data[iDT] = iCov
        if Data: return pd.Panel(Data).loc[dts]
        if ids: return pd.Panel(items=dts, major_axis=ids, minor_axis=ids)
        return pd.Panel(items=dts)
    # 为某张表增加索引
    def addIndex(self, index_name, table_name, fields=["DateTime"], index_type="BTREE"):
        SQLStr = "CREATE INDEX "+index_name+" USING "+index_type+" ON "+self.TablePrefix+self._Prefix+table_name+"("+", ".join(fields)+")"
        return self.execute(SQLStr)
    # 创建表
    def createTable(self, table_name):
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` DATETIME(6) NOT NULL, `IDs` TEXT NOT NULL, `Cov` LONGTEXT NOT NULL, " % (self.TablePrefix+self._Prefix+table_name, )
        SQLStr += "PRIMARY KEY (`DateTime`)) ENGINE=InnoDB DEFAULT CHARSET=utf8"
        self.execute(SQLStr)
        #return self.addIndex(table_name+"_index", table_name)
    def writeData(self, table_name, idt, icov):
        if table_name not in self._TableNames:
            self.createTable(table_name)
            self._TableNames.append(table_name)
            if self.Connector == "pyodbc":
                SQLStr = "INSERT INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `IDs`, `Cov`) VALUES (?, ?, ?)"
            else:
                SQLStr = "INSERT INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `IDs`, `Cov`) VALUES (%s, %s, %s)"
        else:
            if self.Connector=="pyodbc":
                SQLStr = "REPLACE INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `IDs`, `Cov`) VALUES (?, ?, ?)"
            else:
                SQLStr = "REPLACE INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `IDs`, `Cov`) VALUES (%s, %s, %s)"
        Cursor = self._Connection.cursor()
        Cursor.execute(SQLStr, (idt, json.dumps(icov.index.tolist()), json.dumps(icov.values.tolist())))
        self._Connection.commit()
        Cursor.close()
        return 0