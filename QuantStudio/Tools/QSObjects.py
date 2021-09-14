# -*- coding: utf-8 -*-
import os
import re
import mmap
import uuid
from multiprocessing import Queue, Lock
from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, File, Bool

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import genAvailableName

os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"

class QSSQLObject(__QS_Object__):
    """基于关系数据库的对象"""
    Name = Str("QSSQLObject")
    DBType = Enum("MySQL", "SQL Server", "Oracle", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pymysql", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    AdjustTableName = Bool(False, arg_type="Bool", label="调整表名", order=10)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Connection = None# 连接对象
        self._Connector = None# 实际使用的数据库链接器
        self._AllTables = []# 数据库中的所有表名, 用于查询时解决大小写敏感问题
        self._PID = None# 保存数据库连接创建时的进程号
        self._SQLFun = {}
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_Connection"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._Connection: self._connect()
        else: self._Connection = None
    @property
    def Connection(self):
        if self._Connection is not None:
            if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        return self._Connection
    def _connect(self):
        self._Connection = None
        if (self.Connector=="cx_Oracle") or ((self.Connector=="default") and (self.DBType=="Oracle")):
            try:
                import cx_Oracle
                self._Connection = cx_Oracle.connect(self.User, self.Pwd, cx_Oracle.makedsn(self.IPAddr, str(self.Port), self.DBName))
            except Exception as e:
                Msg = ("'%s' 尝试使用 cx_Oracle 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "cx_Oracle"
        elif (self.Connector=="pymssql") or ((self.Connector=="default") and (self.DBType=="SQL Server")):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                Msg = ("'%s' 尝试使用 pymssql 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "pymssql"
        elif (self.Connector=="mysql.connector") or ((self.Connector=="default") and (self.DBType=="MySQL")):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet, autocommit=True)
            except Exception as e:
                Msg = ("'%s' 尝试使用 mysql.connector 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "mysql.connector"
        elif self.Connector=="pymysql":
            try:
                import pymysql
                self._Connection = pymysql.connect(host=self.IPAddr, port=self.Port, user=self.User, password=self.Pwd, db=self.DBName, charset=self.CharSet)
            except Exception as e:
                Msg = ("'%s' 尝试使用 pymysql 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                raise e
            else:
                self._Connector = "pymysql"
        if self._Connection is None:
            if self.Connector not in ("default", "pyodbc"):
                self._Connection = None
                Msg = ("'%s' 连接数据库时错误: 不支持该连接器(connector) '%s'" % (self.Name, self.Connector))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
            elif self.DSN:
                try:
                    import pyodbc
                    self._Connection = pyodbc.connect("DSN=%s;PWD=%s" % (self.DSN, self.Pwd))
                except Exception as e:
                    Msg = ("'%s' 尝试使用 pyodbc 连接数据库 'DSN: %s' 失败: %s" % (self.Name, self.DSN, str(e)))
                    self._QS_Logger.error(Msg)
                    raise e
            else:
                try:
                    import pyodbc
                    self._Connection = pyodbc.connect("DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s" % (self.DBType, self.DBName, self.IPAddr+","+str(self.Port), self.User, self.Pwd))
                except Exception as e:
                    Msg = ("'%s' 尝试使用 pyodbc 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                    self._QS_Logger.error(Msg)
                    raise e
            self._Connector = "pyodbc"
        self._PID = os.getpid()
        return 0
    def connect(self):
        self._connect()
        if not self.AdjustTableName:
            self._AllTables = []
        else:
            self._AllTables = self.getDBTable()
        # 设置特异性参数
        if self._Connector=="pyodbc":
            self._PlaceHolder = "?"
        else:
            self._PlaceHolder = "%s"
        # 设置 SQL 相关特异性函数
        if self.DBType=="MySQL":
            self._SQLFun = {"toDate": "DATE(%s)"}
        elif self.DBType=="Oracle":
            self._SQLFun = {"toDate": "CAST(%s AS DATE)"}# TOTEST
        elif self.DBType=="SQL Server":
            self._SQLFun = {"toDate": "CAST(%s AS DATE)"}# TOTEST
        else:
            raise NotImplementedError("'%s' 调用方法 connect 时错误: 尚不支持的数据库类型" % (self.Name, self.DBType))
        return 0
    def disconnect(self):
        if self._Connection is not None:
            try:
                self._Connection.close()
            except Exception as e:
                self._QS_Logger.warning("'%s' 断开数据库错误: %s" % (self.Name, str(e)))
            finally:
                self._Connection = None
        return 0
    def isAvailable(self):
        return (self._Connection is not None)
    def cursor(self, sql_str=None):
        if self._Connection is None:
            Msg = ("'%s' 获取 cursor 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:# 连接断开后重连
            Cursor = self._Connection.cursor()
        except:
            self._connect()
            Cursor = self._Connection.cursor()
        if sql_str is None: return Cursor
        if self.AdjustTableName:
            for iTable in self._AllTables:
                sql_str = re.sub(iTable, iTable, sql_str, flags=re.IGNORECASE)
        Cursor.execute(sql_str)
        return Cursor
    def fetchall(self, sql_str):
        Cursor = self.cursor(sql_str=sql_str)
        Data = Cursor.fetchall()
        Cursor.close()
        return Data
    def execute(self, sql_str):
        if self._Connection is None:
            Msg = ("'%s' 执行 SQL 命令失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:
            Cursor = self._Connection.cursor()
        except:
            self._connect()
            Cursor = self._Connection.cursor()
        Cursor.execute(sql_str)
        self._Connection.commit()
        Cursor.close()
        return 0
    def getDBTable(self, table_format=None):
        try:
            if self.DBType=="SQL Server":
                SQLStr = "SELECT Name FROM SysObjects Where XType='U'"
                TableField = "Name"
            elif self.DBType=="MySQL":
                SQLStr = "SELECT table_name FROM information_schema.tables WHERE table_schema='"+self.DBName+"' AND table_type='base table'"
                TableField = "table_name"
            elif self.DBType=="Oracle":
                SQLStr = "SELECT table_name FROM user_tables WHERE TABLESPACE_NAME IS NOT NULL AND user='"+self.User+"'"
                TableField = "table_name"
            else:
                raise __QS_Error__("不支持的数据库类型 '%s'" % self.DBType)
            if isinstance(table_format, str) and table_format:
                SQLStr += (" WHERE %s LIKE '%s' " % (TableField, table_format))
            AllTables = self.fetchall(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 getDBTable 时错误: %s" % (self.Name, str(e)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        else:
            return [rslt[0] for rslt in AllTables]
    def renameDBTable(self, old_table_name, new_table_name):
        SQLStr = "ALTER TABLE "+self.TablePrefix+old_table_name+" RENAME TO "+self.TablePrefix+new_table_name
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 renameDBTable 将表 '%s' 重命名为 '%s' 时错误: %s" % (self.Name, old_table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 renameDBTable 将表 '%s' 重命名为 '%s'" % (self.Name, old_table_name, new_table_name))
        return 0
    # 创建表, field_types: {字段名: 数据类型}
    def createDBTable(self, table_name, field_types, primary_keys=[], index_fields=[]):
        if self.DBType=="MySQL":
            SQLStr = "CREATE TABLE IF NOT EXISTS %s (" % (self.TablePrefix+table_name)
            for iField, iDataType in field_types.items():SQLStr += "`%s` %s, " % (iField, iDataType)
            if primary_keys:
                SQLStr += "PRIMARY KEY (`"+"`,`".join(primary_keys)+"`))"
            else:
                SQLStr += ")"
            SQLStr += " ENGINE=InnoDB DEFAULT CHARSET="+self.CharSet
            IndexType = "BTREE"
        else:
            raise NotImplementedError("'%s' 调用方法 createDBTable 在数据库中创建表 '%s' 时错误: 尚不支持的数据库类型" % (self.Name, table_name, self.DBType))
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 createDBTable 在数据库中创建表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 createDBTable 在数据库中创建表 '%s'" % (self.Name, table_name))
        try:
            self.addIndex(table_name+"_index", table_name, fields=index_fields, index_type=IndexType)
        except Exception as e:
            self._QS_Logger.warning("'%s' 调用方法 createDBTable 在数据库中创建表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
        return 0
    def deleteDBTable(self, table_name):
        SQLStr = "DROP TABLE %s" % (self.TablePrefix+table_name)
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteDBTable 从数据库中删除表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 deleteDBTable 从数据库中删除表 '%s'" % (self.Name, table_name))
        return 0
    def addIndex(self, index_name, table_name, fields, index_type="BTREE"):
        if index_type is not None:
            SQLStr = "CREATE INDEX "+index_name+" USING "+index_type+" ON "+self.TablePrefix+table_name+"("+", ".join(fields)+")"
        else:
            SQLStr = "CREATE INDEX "+index_name+" ON "+self.TablePrefix+table_name+"("+", ".join(fields)+")"
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 addIndex 为表 '%s' 添加索引时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 addIndex 为表 '%s' 添加索引 '%s'" % (self.Name, table_name, index_name))
        return 0
    def getFieldDataType(self, table_format=None, ignore_fields=[]):
        try:
            if self.DBType=="MySQL":
                SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.columns WHERE table_schema='%s' " % self.DBName)
                TableField, ColField = "TABLE_NAME", "COLUMN_NAME"
            elif self.DBType=="SQL Server":
                SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.columns WHERE table_schema='%s' " % self.DBName)
                TableField, ColField = "TABLE_NAME", "COLUMN_NAME"
            elif self.DBType=="Oracle":
                SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM user_tab_columns")
                TableField, ColField = "TABLE_NAME", "COLUMN_NAME"
            else:
                raise __QS_Error__("不支持的数据库类型 '%s'" % self.DBType)
            if isinstance(table_format, str) and table_format:
                SQLStr += ("AND %s LIKE '%s' " % (TableField, table_format))
            if ignore_fields:
                SQLStr += "AND "+ColField+" NOT IN ('"+"', '".join(ignore_fields)+"') "
            SQLStr += ("ORDER BY %s, %s" % (TableField, ColField))
            Rslt = self.fetchall(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 getFieldDataType 获取字段数据类型信息时错误: %s" % (self.Name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        return pd.DataFrame(Rslt, columns=["Table", "Field", "DataType"])
    # 增加字段, field_types: {字段名: 数据类型}
    def addField(self, table_name, field_types):
        SQLStr = "ALTER TABLE %s " % (self.TablePrefix+table_name)
        SQLStr += "ADD COLUMN ("
        for iField in field_types: SQLStr += "%s %s," % (iField, field_types[iField])
        SQLStr = SQLStr[:-1]+")"
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 addField 为表 '%s' 添加字段时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 addField 为表 '%s' 添加字段 ’%s'" % (self.Name, table_name, str(list(field_types.keys()))))
        return 0
    def renameField(self, table_name, old_field_name, new_field_name):
        try:
            SQLStr = "ALTER TABLE "+self.TablePrefix+table_name
            SQLStr += " CHANGE COLUMN `"+old_field_name+"` `"+new_field_name+"`"
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 renameField 将表 '%s' 中的字段 '%s' 重命名为 '%s' 时错误: %s" % (self.Name, table_name, old_field_name, new_field_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 renameField 在将表 '%s' 中的字段 '%s' 重命名为 '%s'" % (self.Name, table_name, old_field_name, new_field_name))
        return 0
    def deleteField(self, table_name, field_names):
        if not field_names: return 0
        try:
            SQLStr = "ALTER TABLE "+self.TablePrefix+table_name
            for iField in field_names: SQLStr += " DROP COLUMN `"+iField+"`,"
            self.execute(SQLStr[:-1])
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s' 时错误: %s" % (self.Name, table_name, str(field_names), str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s'" % (self.Name, table_name, str(field_names)))
        return 0
    def truncateDBTable(self, table_name):
        SQLStr = "TRUNCATE TABLE %s" % (self.TablePrefix+table_name)
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 truncateDBTable 清空数据库中的表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        else:
            self._QS_Logger.info("'%s' 调用方法 truncateDBTable 清空数据库中的表 '%s'" % (self.Name, table_name))
        return 0

class QSSQLite3Object(QSSQLObject):
    """基于 sqlite3 模块的对象"""
    DBType = Enum("sqlite3", arg_type="SingleOption", label="数据库类型", order=0)
    Connector = Enum("sqlite3", arg_type="SingleOption", label="连接器", order=8)
    SQLite3File = File(label="sqlite3文件", arg_type="File", order=11)
    def _connect(self):
        try:
            import sqlite3
            self._Connection = sqlite3.connect(self.SQLite3File)
        except Exception as e:
            Msg = ("'%s' 尝试使用 sqlite3 连接数据库 '%s' 失败: %s" % (self.Name, self.SQLite3File, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._Connector = "sqlite3"
        self._PID = os.getpid()
        return 0
    def connect(self):
        super().connect()
        self._PlaceHolder = "?"
        return 0
    def getDBTable(self, table_format=None):
        try:
            SQLStr = "SELECT name FROM sqlite_master WHERE type='table'"
            TableField = "name"
            if isinstance(table_format, str) and table_format:
                SQLStr += (" WHERE %s LIKE '%s' " % (TableField, table_format))
            AllTables = self.fetchall(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 getDBTable 时错误: %s" % (self.Name, str(e)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        else:
            return [rslt[0] for rslt in AllTables]
    def createDBTable(self, table_name, field_types, primary_keys=[], index_fields=[]):
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (" % (self.TablePrefix+table_name)
        for iField, iDataType in field_types.items(): SQLStr += "`%s` %s, " % (iField, iDataType)
        if primary_keys:
            SQLStr += "PRIMARY KEY (`"+"`,`".join(primary_keys)+"`))"
        else:
            SQLStr += ")"
        IndexType = None
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 createDBTable 在数据库中创建表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 createDBTable 在数据库中创建表 '%s'" % (self.Name, table_name))
        return 0
    def getFieldDataType(self, table_format=None, ignore_fields=[]):
        try:
            AllTables = self.getDBTable(table_format=table_format)
            Rslt = []
            for iTable in AllTables:
                iSQLStr = "PRAGMA table_info('"+iTable+"')"
                iRslt = pd.DataFrame(self.fetchall(iSQLStr), columns=["cid","Field","DataType","notnull","dflt_value","pk"])
                iRslt["Table"] = iTable
            if Rslt:
                Rslt = pd.concat(Rslt).drop(labels=["cid", "notnull", "dflt_value", "pk"], axis=1).loc[:, ["Table", "Field", "DataType"]].values
        except Exception as e:
            Msg = ("'%s' 调用方法 getFieldDataType 获取字段数据类型信息时错误: %s" % (self.Name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        return pd.DataFrame(Rslt, columns=["Table", "Field", "DataType"])
    def renameField(self, table_name, old_field_name, new_field_name):
        try:
            # 将表名改为临时表
            SQLStr = "ALTER TABLE %s RENAME TO %s"
            TempTableName = genAvailableName("TempTable", self.getDBTable())
            self.execute(SQLStr % (self.TablePrefix+table_name, self.TablePrefix+TempTableName))
            # 创建新表
            FieldTypes = OrderedDict()
            FieldDataType = self.getFieldDataType(table_format=table_name).loc[:, ["Field", "DataType"]].set_index(["Field"]).iloc[:,0].to_dict()
            for iField, iDataType in FieldDataType.items():
                iDataType = ("text" if iDataType=="string" else "real")
                if iField==old_field_name: FieldTypes[new_field_name] = iDataType
                else: FieldTypes[iField] = iDataType
            self.createDBTable(table_name, field_types=FieldTypes)
            # 导入数据
            OldFieldNames = ", ".join(FieldDataType.keys())
            NewFieldNames = ", ".join(FieldTypes)
            SQLStr = "INSERT INTO %s (datetime, code, %s) SELECT datetime, code, %s FROM %s"
            Cursor = self.cursor(SQLStr % (self.TablePrefix+table_name, NewFieldNames, OldFieldNames, self.TablePrefix+TempTableName))
            Conn = self.Connection
            Conn.commit()
            # 删除临时表
            Cursor.execute("DROP TABLE %s" % (self.TablePrefix+TempTableName, ))
            Conn.commit()
            Cursor.close()
        except Exception as e:
            Msg = ("'%s' 调用方法 renameField 将表 '%s' 中的字段 '%s' 重命名为 '%s' 时错误: %s" % (self.Name, table_name, old_field_name, new_field_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 renameField 在将表 '%s' 中的字段 '%s' 重命名为 '%s'" % (self.Name, table_name, old_field_name, new_field_name))
        return 0
    def deleteField(self, table_name, field_names):
        if not field_names: return 0
        try:
            # 将表名改为临时表
            SQLStr = "ALTER TABLE %s RENAME TO %s"
            TempTableName = genAvailableName("TempTable", self.getDBTable())
            self.execute(SQLStr % (self.TablePrefix+table_name, self.TablePrefix+TempTableName))
            # 创建新表
            FieldTypes = OrderedDict()
            FieldDataType = self.getFieldDataType(table_format=table_name).loc[:, ["Field", "DataType"]].set_index(["Field"]).iloc[:,0].to_dict()
            FactorIndex = list(set(FieldDataType).difference(field_names))
            for iField in FactorIndex:
                FieldTypes[iField] = ("text" if FieldDataType[iField]=="string" else "real")
            self.createTable(table_name, field_types=FieldTypes)
            # 导入数据
            FactorNameStr = ", ".join(FactorIndex)
            SQLStr = "INSERT INTO %s (datetime, code, %s) SELECT datetime, code, %s FROM %s"
            Cursor = self.cursor(SQLStr % (self.TablePrefix+table_name, FactorNameStr, FactorNameStr, self.TablePrefix+TempTableName))
            Conn = self.Connection
            Conn.commit()
            # 删除临时表
            Cursor.execute("DROP TABLE %s" % (self.TablePrefix+TempTableName, ))
            Conn.commit()
            Cursor.close()
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s' 时错误: %s" % (self.Name, table_name, str(field_names), str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s'" % (self.Name, table_name, str(field_names)))
        return 0
    def truncateDBTable(self, table_name):
        SQLStr = "DELETE FROM %s" % (self.TablePrefix+table_name)
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 truncateDBTable 清空数据库中的表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        else:
            self._QS_Logger.info("'%s' 调用方法 truncateDBTable 清空数据库中的表 '%s'" % (self.Name, table_name))
        return 0

class QSClickHouseObject(QSSQLObject):
    """ClickHouseDB"""
    DBType = Enum("ClickHouse", arg_type="SingleOption", label="数据库类型", order=0)
    Connector = Enum("default", "clickhouse-driver", arg_type="SingleOption", label="连接器", order=7)
    def _connect(self):
        self._Connection = None
        if (self.Connector=="clickhouse-driver") or (self.Connector=="default"):
            try:
                import clickhouse_driver
                if self.DSN:
                    self._Connection = clickhouse_driver.connect(dsn=self.DSN, password=self.Pwd)
                else:
                    self._Connection = clickhouse_driver.connect(user=self.User, password=self.Pwd, host=self.IPAddr, port=self.Port, database=self.DBName)
            except Exception as e:
                Msg = ("'%s' 尝试使用 clickhouse-driver 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "clickhouse-driver"
        self._PID = os.getpid()
        return 0
    def connect(self):
        self._connect()
        if not self.AdjustTableName:
            self._AllTables = []
        else:
            self._AllTables = self.getDBTable()
        # 设置特异性参数
        # 设置 SQL 相关特异性函数
        self._SQLFun = {"toDate": "DATE(%s)"}
        return 0
    def renameDBTable(self, old_table_name, new_table_name):
        SQLStr = "RENAME TABLE "+self.TablePrefix+old_table_name+" TO "+self.TablePrefix+new_table_name
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 renameDBTable 将表 '%s' 重命名为 '%s' 时错误: %s" % (self.Name, old_table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 renameDBTable 将表 '%s' 重命名为 '%s'" % (self.Name, old_table_name, new_table_name))
        return 0
    def createDBTable(self, table_name, field_types, primary_keys=[], index_fields=[]):
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (" % (self.TablePrefix+table_name)
        for iField in field_types: SQLStr += "`%s` %s, " % (iField, field_types[iField])
        SQLStr = SQLStr[:-2]+")"
        SQLStr += " ENGINE=MergeTree()"
        if primary_keys:
            SQLStr += " ORDER BY (`"+"`,`".join(primary_keys)+"`)"
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 createDBTable 在数据库中创建表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 createDBTable 在数据库中创建表 '%s'" % (self.Name, table_name))
        return 0
    def getDBTable(self):
        try:
            SQLStr = "SELECT name FROM system.tables WHERE database='"+self.DBName+"'"
            AllTables = self.fetchall(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 getDBTable 时错误: %s" % (self.Name, str(e)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        else:
            return [rslt[0] for rslt in AllTables]
    def getFieldDataType(self, table_format=None, ignore_fields=[]):
        try:
            SQLStr = ("SELECT table, name, type FROM system.columns WHERE database='%s' " % self.DBName)
            TableField, ColField = "table", "name"
            if isinstance(table_format, str) and table_format:
                SQLStr += ("AND %s LIKE '%s' " % (TableField, table_format))
            if ignore_fields:
                SQLStr += "AND "+ColField+" NOT IN ('"+"', '".join(ignore_fields)+"') "
            SQLStr += ("ORDER BY %s, %s" % (TableField, ColField))
            Rslt = self.fetchall(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 getFieldDataType 获取字段数据类型信息时错误: %s" % (self.Name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        return pd.DataFrame(Rslt, columns=["Table", "Field", "DataType"])
    def addField(self, table_name, field_types):
        SQLStr = "ALTER TABLE %s " % (self.TablePrefix+table_name)
        SQLStr += "ADD COLUMN %s %s"
        try:
            for iField in field_types:
                self.execute(SQLStr % (iField, field_types[iField]))
        except Exception as e:
            Msg = ("'%s' 调用方法 addField 为表 '%s' 添加字段时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 addField 为表 '%s' 添加字段 ’%s'" % (self.Name, table_name, str(list(field_types.keys()))))
        return 0
    def renameField(self, table_name, old_field_name, new_field_name):
        try:
            SQLStr = "ALTER TABLE "+self.TablePrefix+table_name
            SQLStr += " RENAME COLUMN `"+old_field_name+"` TO `"+new_field_name+"`"
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 renameField 将表 '%s' 中的字段 '%s' 重命名为 '%s' 时错误: %s" % (self.Name, table_name, old_field_name, new_field_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 renameField 在将表 '%s' 中的字段 '%s' 重命名为 '%s'" % (self.Name, table_name, old_field_name, new_field_name))
        return 0
    def deleteField(self, table_name, field_names):
        if not field_names: return 0
        try:
                SQLStr = "ALTER TABLE "+self.TablePrefix+table_name
                for iField in field_names: SQLStr += " DROP COLUMN `"+iField+"`,"
                self.execute(SQLStr[:-1])
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s' 时错误: %s" % (self.Name, table_name, str(field_names), str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s'" % (self.Name, table_name, str(field_names)))
        return 0


class QSNeo4jObject(__QS_Object__):
    """基于 Neo4j 的对象"""
    Name = Str("QSNeo4jObject")
    DBName = Str("neo4j", arg_type="String", label="数据库名", order=0)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=7687, arg_type="Integer", label="端口", order=2)
    User = Str("neo4j", arg_type="String", label="用户名", order=3)
    Pwd = Password("", arg_type="String", label="密码", order=4)
    Connector = Enum("default", "neo4j", arg_type="SingleOption", label="连接器", order=5)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Connection = None# 连接对象
        self._Connector = None# 实际使用的数据库链接器
        self._PID = None# 保存数据库连接创建时的进程号
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_Connection"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._Connection: self._connect()
        else: self._Connection = None
    @property
    def Connection(self):
        if self._Connection is not None:
            if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        return self._Connection
    def _connect(self):
        self._Connection = None
        if (self.Connector=="neo4j") or (self.Connector=="default"):
            try:
                import neo4j
                self._Connection = neo4j.GraphDatabase.driver(f"neo4j://{self.IPAddr}:{self.Port}", auth=(self.User, self.Pwd))
            except Exception as e:
                Msg = ("'%s' 尝试使用 neo4j 连接(%s@%s:%d)数据库失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "neo4j"
        self._PID = os.getpid()
        return 0
    def connect(self):
        return self._connect()
    def disconnect(self):
        if self._Connection is not None:
            try:
                self._Connection.close()
            except Exception as e:
                self._QS_Logger.warning("'%s' 断开数据库错误: %s" % (self.Name, str(e)))
            finally:
                self._Connection = None
        return 0
    def isAvailable(self):
        return (self._Connection is not None)
    def session(self):
        if self._Connection is None:
            Msg = ("'%s' 获取 session 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:# 连接断开后重连
            Session = self._Connection.session(database=self.DBName)
        except:
            self._connect()
            Session = self._Connection.session(database=self.DBName)
        return Session
    def fetchall(self, cypher_str, parameters=None):
        with self.session() as Session:
            with Session.begin_transaction() as tx:
                return tx.run(cypher_str, parameters=parameters).values()
    def execute(self, cypher_str, parameters=None):
        if self._Connection is None:
            Msg = ("'%s' 执行 Cypher 命令失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:
            Session = self._Connection.session(database=self.DBName)
        except:
            self._connect()
            Session = self._Connection.session(database=self.DBName)
        with Session:
            with Session.begin_transaction() as tx:
                tx.run(cypher_str, parameters=parameters)
        return 0
    # 写入实体属性数据
    # data: DataFrame(index=[ID], columns=[属性])
    def writeEntityFeatureData(self, data, entity_labels, id_field="Name", if_exists="update", **kwargs):
        IDs, FactorNames = data.index.tolist(), data.columns.tolist()
        if if_exists!="update":
            LabelStr = "`:`".join(entity_labels)
            CypherStr = f"""
                MATCH (n:`{LabelStr}`)
                WHERE n.`{id_field}` IN $ids
                RETURN n.`{id_field}`, n.`{'`, n.`'.join(FactorNames)}`
            """
            OldData = self.fetchall(CypherStr, parameters={"ids": IDs})
            if not OldData: OldData = pd.DataFrame(index=IDs, columns=FactorNames)
            else: OldData = pd.DataFrame(np.array(OldData, dtype="O"), columns=["ID"]+FactorNames).set_index(["ID"]).loc[IDs]
            if if_exists=="append":
                data = OldData.where(pd.notnull(OldData), data)
            elif if_exists=="update_notnull":
                data = data.where(pd.notnull(data), OldData)
            else:
                Msg = ("因子库 '%s' 调用方法 writeFeatureData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
        LabelStr = "`:`".join(entity_labels)
        data[id_field] = data.index
        CypherStr = f"""
            UNWIND $ids AS iID
            MERGE (n:`{LabelStr}` {{`{id_field}`: iID}})
            ON CREATE SET n = $data[iID]
            ON MATCH SET n += $data[iID]
        """
        data = data.astype("O").where(pd.notnull(data), None)
        with self.session() as Session:
            with Session.begin_transaction() as tx:
                tx.run(CypherStr, parameters={"ids": IDs, "data": data.T.to_dict(orient="dict")})
        return 0
    # 写入关系属性数据
    # data: DataFrame(index=[ID], columns=[关联实体字段])
    # retain_relation: False 表示 relation_field 为 NULL 的关系将被删除, True 表示不删除
    def writeRelationFeatureData(self, data, relation_label, relation_field, direction, id_labels, id_field, opp_labels, opp_field, retain_relation=True, if_exists="update", **kwargs):
        OppFields = data.columns.tolist()
        IDNode = f"(n1:`{'`:`'.join(id_labels)}` {{`{id_field}`: p[0]}})"
        OppNode = f"(n2:`{'`:`'.join(opp_labels)}` {{`{opp_field}`: $opp_entity[i]}})"
        if direction=="->":
            Relation = f"(n1) - [r:`{relation_label}`] -> (n2)"
        elif direction=="<-":
            Relation = f"(n1) <- [r:`{relation_label}`] - (n2)"
        else:
            Msg = ("因子库 '%s' 调用方法 writeRelationFeatureData 错误: 不支持的参数值 %s : %s " % (self.Name, "direction", str(direction)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        CypherStr = f"""
            UNWIND range(0, size($opp_entity)-1) AS i
            MERGE {OppNode}
            WITH n2, i
            UNWIND $data[i] AS p
            MERGE {IDNode}
            MERGE {Relation}
        """
        if if_exists=="update_notnull":
            data = data.apply(lambda s: s.dropna().reset_index().values.tolist(), axis=0, raw=False).tolist()
            CypherStr += f" SET r.`{relation_field}` = p[1]"
        elif if_exists=="append":
            data = data.apply(lambda s: s.dropna().reset_index().values.tolist(), axis=0, raw=False).tolist()
            CypherStr += f" WHERE r.`{relation_field}` IS NULL SET r.`{relation_field}` = p[1]"
        else:
            OldIDStr = f"""
                MATCH (n1:`{'`:`'.join(id_labels)}`) - [r:`{relation_label}`] -> (n2:`{'`:`'.join(opp_labels)}`)
                WHERE n1.`{id_field}` IS NOT NULL AND r.`{relation_field}` IS NOT NULL AND n2.`{opp_field}` IN $opp_entity
                RETURN collect(DISTINCT n1.`{id_field}`)
            """
            OldIDs = self.fetchall(OldIDStr, parameters={"opp_entity": OppFields})
            if OldIDs: OldIDs = OldIDs[0][0]
            def _chg2None(s):
                s = s.loc[s.dropna().index.union(OldIDs)].astype("O")
                return s.where(pd.notnull(s), None).reset_index().values.tolist()
            data = data.apply(_chg2None, axis=0, raw=False).tolist()
            CypherStr += f" SET r.`{relation_field}` = p[1]"
        self.execute(CypherStr, parameters={"opp_entity": OppFields, "data": data})
        # 删除属性为 NULL 的关系
        if not retain_relation:
            DelStr = f"""
                MATCH (n1:`{'`:`'.join(id_labels)}`) - [r:`{relation_label}`] -> (n2:`{'`:`'.join(opp_labels)}`)
                WHERE n1.`{id_field}` IS NOT NULL AND r.`{relation_field}` IS NULL AND n2.`{opp_field}` IN $opp_entity
                DELETE r
            """
            self.execute(DelStr, parameters={"opp_entity": OppFields})
        return 0

# put 函数会阻塞, 直至对象传输完毕
class QSPipe(object):
    """进程间 Pipe, 无大小限制"""
    # cache_size: 缓存大小, 单位是 MB
    def __init__(self, cache_size=100):
        self._CacheSize = int(cache_size*2**20)
        self._PutQueue = Queue()
        self._PutLock = Lock()
        self._GetQueue = Queue()
        if os.name=="nt":
            self._TagName = str(uuid.uuid1())# 共享内存的 tag
            self._MMAPCacheData = mmap.mmap(-1, self._CacheSize, tagname=self._TagName)# 当前共享内存缓冲区
        else:
            self._TagName = None# 共享内存的 tag
            self._MMAPCacheData = mmap.mmap(-1, self._CacheSize)# 当前共享内存缓冲区
    @property
    def CacheSize(self):
        return self._CacheSize / 2**20
    def __getstate__(self):
        state = self.__dict__.copy()
        if os.name=="nt": state["_MMAPCacheData"] = None
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        if os.name=="nt": self._MMAPCacheData = mmap.mmap(-1, self._CacheSize, tagname=self._TagName)
    def put(self, obj):
        with self._PutLock:
            DataByte = pickle.dumps(obj)
            DataLen = len(DataByte)
            for i in range(int(DataLen/self._CacheSize)+1):
                iStartInd = i * self._CacheSize
                iEndInd = min((i+1)*self._CacheSize, DataLen)
                if iEndInd>iStartInd:
                    self._MMAPCacheData.seek(0)
                    self._MMAPCacheData.write(DataByte[iStartInd:iEndInd])
                    self._PutQueue.put(iEndInd-iStartInd)
                    self._GetQueue.get()
            self._PutQueue.put(0)
        return 0
    def get(self):
        DataLen = self._PutQueue.get()
        DataByte = b""
        while DataLen>0:
            self._MMAPCacheData.seek(0)
            DataByte += self._MMAPCacheData.read(DataLen)
            self._GetQueue.put(DataLen)
            DataLen = self._PutQueue.get()
        return pickle.loads(DataByte)
    def empty(self):
        return self._PutQueue.empty()

# 消息转发器
class QSMsgRouter(object):
    def __init__(self):
        self._QueueFromSender = Queue()
        self._Sender2Listener = {None: set()}# {消息发送者: {消息接收者}}, None 表示所有消息发送者
        self._Listeners = {}# {消息接收者: Queue}
    # 注册消息发送者
    def registerMsgSender(self, name):
        self._Sender2Listener[name] = set()
        return 0
    # 注册消息接受者，sender_name 为 None 表示接受所有消息
    def registerMsgListener(self, name, sender_name=None):
        if sender_name not in self._Sender2Listener:
            print(f"警告: {sender_name} 尚未注册为消息发送者!")
        if name not in self._Listeners:
            self._Listeners[name] = Queue()
        self._Sender2Listener.setdefault(sender_name, set()).add(name)
        return 0
    # 发送消息
    def sendMsg(self, msg, name, block=True, timeout=None):
        if name not in self._Sender2Listener:
            raise __QS_Error__(f"{name} 尚未注册为消息发送者!")
        for iListener in self._Sender2Listener[name].union(self._Sender2Listener[None]):
            self._Listeners[iListener].put((name, msg), block=block, timeout=timeout)
        return 0
    # 接收消息
    def recvMsg(self, name, block=True, timeout=None):
        if name not in self._Listeners:
            raise __QS_Error__(f"{name} 尚未注册为消息接收者!")
        return self._Listeners[name].get(block=block, timeout=timeout)
    