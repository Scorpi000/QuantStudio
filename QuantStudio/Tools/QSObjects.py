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
from traits.api import Enum, Str, Range, Password, File, Bool, Either, Constant

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
    SQLite3File = Either(Enum(":memory:"), File(), label="sqlite3文件", arg_type="File", order=11)
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
        self._connect()
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

def _initArray(shape, dtype):
    if dtype in (np.dtype("datetime64[ns]"), np.dtype("datetime64"), np.dtype("timedelta64[ns]"), np.dtype("timedelta64")):
        return np.full(shape=shape, fill_value=np.nan, dtype=dtype), dtype
    else:
        a = np.full(shape=shape, fill_value=np.nan, dtype=np.dtype("O"))
    try:
        a = a.astype(dtype)
    except (ValueError, TypeError):
        return a, np.dtype("O")
    else:
        return a, dtype

# pandas Panel 的 QS 实现 TODO
class _LocIndexer(object):
    def __init__(self, p):
        self._p = p
    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key, slice(None), slice(None))
        else: key += (slice(None),) * (3 - len(key))
        if len(key)>3: raise IndexError("QuantStudio.Tools.QSObjects.Panel.loc: Too many indexers")
        Items = self._p._Items[key[0]]
        MajorAxis = self._p._MajorAxis[key[1]]
        MinorAxis = self._p._MinorAxis[key[2]]
        KeepDim = (isinstance(Items, pd.Series), isinstance(MajorAxis, pd.Series), isinstance(MinorAxis, pd.Series))
        if np.all(KeepDim):# Panel
            DTypes = self._p._DTypes.loc[Items.index]
            UniDType = DTypes.dropna().unique()
            UniDType = (UniDType[0] if UniDType.shape[0]==1 else np.dtype("O"))
            DTypes = DTypes.fillna(value=UniDType)
            Items, MajorAxis, MinorAxis = (Items + 1).fillna(value=0).astype(np.int), (MajorAxis + 1).fillna(value=0).astype(np.int), (MinorAxis + 1).fillna(value=0).astype(np.int)
            TmpShape = (Items.max()+1, MajorAxis.max()+1, MinorAxis.max()+1)
            #TmpData = np.full(shape=TmpShape, fill_value=None, dtype=UniDType)
            TmpData, UniDType = _initArray(shape=TmpShape, dtype=UniDType)
            TmpData[1:, 1:, 1:] = self._p._Data[:TmpShape[0]-1, :TmpShape[1]-1, :TmpShape[2]-1]
            p = Panel(data=TmpData[Items.values][:, MajorAxis.values][:, :, MinorAxis.values].astype(UniDType), items=Items.index, major_axis=MajorAxis.index, minor_axis=MinorAxis.index)
            p._DTypes = DTypes
            return p
        elif sum(KeepDim)==2:# DataFrame
            if not KeepDim[0]:
                try:
                    Data = pd.DataFrame(self._p._Data[Items].astype(self._p._DTypes[key[0]]), index=self._p._MajorAxis.index, columns=self._p._MinorAxis.index)
                except (ValueError, TypeError):
                    Data = pd.DataFrame(self._p._Data[Items], index=self._p._MajorAxis.index, columns=self._p._MinorAxis.index)
                return Data.loc[key[1], key[2]]
            elif not KeepDim[1]:
                Data = pd.DataFrame(self._p._Data[:, MajorAxis].T, index=self._p._MinorAxis.index, columns=self._p._Items.index)
                return Data.loc[key[2], key[0]]
            else:
                Data = pd.DataFrame(self._p._Data[:, :, MinorAxis].T, index=self._p._MajorAxis.index, columns=self._p._Items.index)
                return Data.loc[key[1], key[0]]
        elif sum(KeepDim)==1:# Series
            if KeepDim[0]:
                Data = pd.Series(self._p._Data[:, MajorAxis, MinorAxis], index=self._p._Items.index)
                return Data.loc[key[0]]
            elif KeepDim[1]:
                try:
                    Data = pd.Series(self._p._Data[Items, :, MinorAxis].astype(self._p._DTypes[key[0]]), index=self._p._MajorAxis.index)
                except (ValueError, TypeError):
                    Data = pd.Series(self._p._Data[Items, :, MinorAxis], index=self._p._MajorAxis.index)
                return Data.loc[key[1]]
            else:
                try:
                    Data = pd.Series(self._p._Data[Items, MajorAxis].astype(self._p._DTypes[key[0]]), index=self._p._MinorAxis.index)
                except (ValueError, TypeError):
                    Data = pd.Series(self._p._Data[Items, MajorAxis], index=self._p._MinorAxis.index)
                return Data.loc[key[2]]
        else:# Scalar
            return self._p._Data[Items, MajorAxis, MinorAxis]
    def __setitem__(self, key, value):
        Items, MajorAxis, MinorAxis = self._p._Items.copy(), self._p._MajorAxis.copy(), self._p._MinorAxis.copy()
        DTypes = self._p._DTypes.copy()
        Data = self._p._Data
        if not isinstance(key, tuple):
            try:
                Items.loc[key] = -1
            except:
                self._p.iloc[key] = value
                return
            else:
                key = (key, slice(None), slice(None))
        else:
            key += (slice(None),) * (3 - len(key))
        if len(key)>3: raise IndexError("QuantStudio.Tools.QSObjects.Panel.loc: Too many indexers")
        # items
        Items.loc[key[0]] = -1
        Items.loc[:] = np.arange(Items.shape[0])
        Key0 = Items.loc[key[0]]
        if isinstance(key[0], slice):
            Key0 = slice(Key0.iloc[0], Key0.iloc[-1]+1, key[0].step)
        elif isinstance(Key0, pd.Series):
            Key0 = Key0.tolist()
        if Items.shape[0]>Data.shape[0]:
            DTypes = DTypes.loc[Items.index]
            if isinstance(value, pd.DataFrame):
                value = value.loc[self._p._MajorAxis.index, self._p._MinorAxis.index]
                ValueDType = value.dtypes.unique()
                DTypes = DTypes.fillna(ValueDType[0] if ValueDType.shape[0]==1 else np.dtype("O"))
            elif hasattr(value, "dtypes"):
                DTypes.loc[value.dtypes.index] = value.dtypes
            else:
                DTypes = DTypes.fillna(getattr(value, "dtype", np.dtype("O")))
            #Data = np.concatenate((Data, np.full(shape=(Items.shape[0] - Data.shape[0], Data.shape[1], Data.shape[2]), fill_value=None, dtype=Data.dtype)), axis=0)
            Data = np.concatenate((Data, _initArray(shape=(Items.shape[0] - Data.shape[0], Data.shape[1], Data.shape[2]), dtype=Data.dtype)[0]), axis=0)
        # major_axis
        MajorAxis.loc[key[1]] = -1
        MajorAxis.loc[:] = np.arange(MajorAxis.shape[0])
        Key1 = MajorAxis.loc[key[1]]
        if isinstance(key[1], slice):
            Key1 = slice(Key1.iloc[0], Key1.iloc[-1]+1, key[1].step)
        elif isinstance(Key1, pd.Series):
            Key1 = Key1.tolist()
        if MajorAxis.shape[0]>Data.shape[1]:
            #Data = np.concatenate((Data, np.full(shape=(Data.shape[0], MajorAxis.shape[0] - Data.shape[1], Data.shape[2]), fill_value=None, dtype=Data.dtype)), axis=1)
            Data = np.concatenate((Data, _initArray(shape=(Data.shape[0], MajorAxis.shape[0] - Data.shape[1], Data.shape[2]), dtype=Data.dtype)[0]), axis=1)
        # minor_axis
        MinorAxis.loc[key[2]] = -1
        MinorAxis.loc[:] = np.arange(MinorAxis.shape[0])
        Key2 = MinorAxis.loc[key[2]]
        if isinstance(key[2], slice):
            Key2 = slice(Key2.iloc[0], Key2.iloc[-1]+1, key[2].step)
        elif isinstance(Key2, pd.Series):
            Key2 = Key2.tolist()
        if MinorAxis.shape[0]>Data.shape[2]:
            #Data = np.concatenate((Data, np.full(shape=(Data.shape[0], Data.shape[1], MinorAxis.shape[0] - Data.shape[2]), fill_value=None, dtype=Data.dtype)), axis=2)
            Data = np.concatenate((Data, _initArray(shape=(Data.shape[0], Data.shape[1], MinorAxis.shape[0] - Data.shape[2]), dtype=Data.dtype)[0]), axis=2)
        # 赋值
        try:
            Data[(Key0, Key1, Key2)] = value
        except ValueError:
            Data = Data.astype(np.dtype("O"))
            Data[(Key0, Key1, Key2)] = value
        self._p._Data = Data
        self._p._Items = Items
        self._p._MajorAxis = MajorAxis
        self._p._MinorAxis = MinorAxis
        self._p._DTypes = DTypes

class _iLocIndexer(object):
    def __init__(self, p):
        self._p = p
    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key, slice(None), slice(None))
        else: key += (slice(None),) * (3 - len(key))
        if len(key)>3: raise IndexError("QuantStudio.Tools.QSObjects.Panel.iloc: Too many indexers")
        Items = self._p._Items.index[key[0]]
        MajorAxis = self._p._MajorAxis.index[key[1]]
        MinorAxis = self._p._MinorAxis.index[key[2]]
        KeepDim = (isinstance(Items, pd.Index), isinstance(MajorAxis, pd.Index), isinstance(MinorAxis, pd.Index))
        if np.all(KeepDim):# Panel
            DTypes = self._p._DTypes.loc[Items]
            UniDType = DTypes.unique()
            UniDType = (UniDType[0] if UniDType.shape[0]==1 else np.dtype("O"))
            try:
                p = Panel(data=self._p._Data[key[0]][:, key[1]][:, :, key[2]].astype(UniDType), items=Items, major_axis=MajorAxis, minor_axis=MinorAxis)
            except (ValueError, TypeError):
                p = Panel(data=self._p._Data[key[0]][:, key[1]][:, :, key[2]], items=Items, major_axis=MajorAxis, minor_axis=MinorAxis)
            p._DTypes = DTypes
            return p
        elif sum(KeepDim)==2:# DataFrame
            if not KeepDim[0]:
                try:
                    return pd.DataFrame(self._p._Data[key[0]][key[1]][:, key[2]].astype(self._p._DTypes[Items]), index=MajorAxis, columns=MinorAxis)
                except (ValueError, TypeError):
                    return pd.DataFrame(self._p._Data[key[0]][key[1]][:, key[2]], index=MajorAxis, columns=MinorAxis)
            elif not KeepDim[1]:
                return pd.DataFrame(self._p._Data[:, key[1]][key[0]][:, key[2]].T, index=MinorAxis, columns=Items)
            else:
                return pd.DataFrame(self._p._Data[:, :, key[2]][key[0]][:, key[1]].T, index=MajorAxis, columns=Items)
        elif sum(KeepDim)==1:# Series
            if KeepDim[0]:
                return pd.Series(self._p._Data[:, key[1], key[2]][key[0]], index=Items)
            elif KeepDim[1]:
                try:
                    return pd.Series(self._p._Data[key[0], :, key[2]][key[1]].astype(self._p._DTypes[Items]), index=MajorAxis)
                except (ValueError, TypeError):
                    return pd.Series(self._p._Data[key[0], :, key[2]][key[1]], index=MajorAxis)
            else:
                try:
                    return pd.Series(self._p._Data[key[0], key[1]][key[2]].astype(self._p._DTypes[Items]), index=MinorAxis)
                except (ValueError, TypeError):
                    return pd.Series(self._p._Data[key[0], key[1]][key[2]], index=MinorAxis)
        else:
            return self._p._Data[key]
    def __setitem__(self, key, value):
        self._p._Data[key] = value
        
class Panel(object):
    """Panel"""
    def __init__(self, data=None, items=None, major_axis=None, minor_axis=None):
        # _Data: array, ndim=3
        # _Items: Series(range(len(items)), index=items)
        # _MajorAxis: Series(range(len(major_axis)), index=major_axis)
        # _MinorAxis: Series(range(len(minor_axis)), index=minor_axis)
        # _DTypes: Series(dtype, index=items)
        # _UniDType: dtype
        # _Loc: _LocIndexer
        #_iLoc: -iLocIndexer
        DataShape = ((0 if items is None else len(items)), (0 if major_axis is None else len(major_axis)), (0 if minor_axis is None else len(minor_axis)))
        if data is None:
            data = np.full(shape=DataShape, fill_value=np.nan, dtype=np.float64)
        if isinstance(data, str) or (not hasattr(data, "__iter__")):
            self._Items = pd.Series(np.arange(DataShape[0]), index=items)
            self._MajorAxis = pd.Series(np.arange(DataShape[1]), index=major_axis)
            self._MinorAxis = pd.Series(np.arange(DataShape[2]), index=minor_axis)
            self._Data = np.full(shape=DataShape, fill_value=data)
            self._DTypes = pd.Series(self._Data.dtype, index=self._Items.index)
            self._UniDType = self._Data.dtype
            self._Loc = _LocIndexer(self)
            self._iLoc = _iLocIndexer(self)
            return
        elif isinstance(data, np.ndarray) and (np.ndim(data)==3):
            self._Items = pd.Series(np.arange(data.shape[0]), index=items)
            self._MajorAxis = pd.Series(np.arange(data.shape[1]), index=major_axis)
            self._MinorAxis = pd.Series(np.arange(data.shape[2]), index=minor_axis)
            self._Data = data
            self._DTypes = pd.Series(self._Data.dtype, index=self._Items.index)
            self._UniDType = self._Data.dtype
            self._Loc = _LocIndexer(self)
            self._iLoc = _iLocIndexer(self)
            return
        # data: 可迭代对象
        try:
            data = OrderedDict(data)
        except:
            pass
        if items is None:
            if isinstance(data, dict):
                self._Items = pd.Series(np.arange(len(data)), index=pd.Index(data.keys()))
            else:
                self._Items = pd.Series(np.arange(len(data)))
        else:
            self._Items = pd.Series(np.arange(len(items)), index=items)
        Data, self._DTypes = {}, {}
        MajorAxis = pd.Index([] if major_axis is None else major_axis)
        MinorAxis = pd.Index([] if minor_axis is None else minor_axis)
        for i, iItem in enumerate(self._Items.index):
            if isinstance(data, dict):
                Data[iItem] = pd.DataFrame(data.get(iItem, None), index=major_axis, columns=minor_axis)
            else:
                Data[iItem] = pd.DataFrame(data[i], index=major_axis, columns=minor_axis)
            self._DTypes[iItem] = Data[iItem].dtypes.unique()
            if self._DTypes[iItem].shape[0]==1: self._DTypes[iItem] = self._DTypes[iItem][0]
            else: self._DTypes[iItem] = np.dtype("O")
            if major_axis is None:
                MajorAxis = MajorAxis.union(Data[iItem].index)
            if minor_axis is None:
                MinorAxis = MinorAxis.union(Data[iItem].columns)
        self._DTypes = pd.Series(self._DTypes, index=self._Items.index)
        self._UniDType = self._DTypes.unique()
        if self._UniDType.shape[0]==1:
            self._UniDType = self._UniDType[0]
            if (major_axis is None) and (minor_axis is None):
                self._Data = np.r_[[Data[iItem].loc[MajorAxis, MinorAxis].values for iItem in Data]]
            elif major_axis is None:
                self._Data = np.r_[[Data[iItem].loc[MajorAxis].values for iItem in Data]]
            elif minor_axis is None:
                self._Data = np.r_[[Data[iItem].loc[:, MinorAxis].values for iItem in Data]]
            else:
                self._Data = np.r_[[Data[iItem].values for iItem in Data]]
        else:
            self._UniDType = np.dtype("O")
            if (major_axis is None) and (minor_axis is None):
                self._Data = np.r_[[Data[iItem].loc[MajorAxis, MinorAxis].values.astype("O") for iItem in Data]]
            elif major_axis is None:
                self._Data = np.r_[[Data[iItem].loc[MajorAxis].values.astype("O") for iItem in Data]]
            elif minor_axis is None:
                self._Data = np.r_[[Data[iItem].loc[:, MinorAxis].values.astype("O") for iItem in Data]]
            else:
                self._Data = np.r_[[Data[iItem].values.astype("O") for iItem in Data]]
        self._MajorAxis = pd.Series(np.arange(len(MajorAxis)), index=MajorAxis)
        self._MinorAxis = pd.Series(np.arange(len(MinorAxis)), index=MinorAxis)
        self._Loc = _LocIndexer(self)
        self._iLoc = _iLocIndexer(self)
    def __repr__(self):
        Shape = self.shape
        return f"""<class 'QuantStudio.Tools.QSObjects.Panel'>\nDimensions: {Shape[0]} (items) x {Shape[1]} (major_axis) x {Shape[2]} (minor_axis)\nItems axis: {None if Shape[0]==0 else f"{self._Items.index[0]} to {self._Items.index[-1]}"}\nMajor_axis axis: {None if Shape[1]==0 else f"{self._MajorAxis.index[0]} to {self._MajorAxis.index[-1]}"}\nMinor_axis axis: {None if Shape[2]==0 else f"{self._MinorAxis.index[0]} to {self._MinorAxis.index[-1]}"}"""
    def __hash__(self):
        raise TypeError('{0!r} objects are mutable, thus they cannot be hashed'.format(self.__class__.__name__))
    def __len__(self):
        return len(self._Items)
    def __contains__(self, key):
        return key in self._Items.index
    def __getitem__(self, key):
        return self._Loc[key]
    def __setitem__(self, key, value):
        self.loc[key] = value
    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default
    def __iter__(self):
        return iter(self._Items.index)
    def iteritems(self):
        for iIdx in self._Items.index:
            yield iIdx, self[iIdx]
    @property
    def shape(self):
        return (self._Items.shape[0], self._MajorAxis.shape[0], self._MinorAxis.shape[0])
    @property
    def values(self):
        return self._Data.copy()
    @property
    def dtypes(self):
        return self._DTypes.copy()
    @property
    def items(self):
        return self._Items.index
    @items.setter
    def items(self, items):
        if len(items)!=self._Items.shape[0]:
            raise __QS_Error__("Panel.items.setter: 设置的 items 长度不等于数据长度")
        self._Items = pd.Series(np.arange(len(items)), index=items)
    @property
    def major_axis(self):
        return self._MajorAxis.index
    @major_axis.setter
    def major_axis(self, major_axis):
        if len(major_axis)!=self._MajorAxis.shape[0]:
            raise __QS_Error__("Panel.major_axis.setter: 设置的 major_axis 长度不等于数据长度")
        self._MajorAxis = pd.Series(np.arange(len(major_axis)), index=major_axis)
    @property
    def minor_axis(self):
        return self._MinorAxis.index
    @minor_axis.setter
    def minor_axis(self, minor_axis):
        if len(minor_axis)!=self._MinorAxis.shape[0]:
            raise __QS_Error__("Panel.minor_axis.setter: 设置的 minor_axis 长度不等于数据长度")
        self._MinorAxis = pd.Series(np.arange(len(minor_axis)), index=minor_axis)
    @property
    def loc(self):
        return self._Loc
    @property
    def iloc(self):
        return self._iLoc
    def keys(self):
        return self._Items.index
    def swapaxes(self, axis1, axis2):
        Data = self._Data.swapaxes(axis1, axis2)
        Dims = [self._Items.index, self._MajorAxis.index, self._MinorAxis.index]
        Dims[axis1], Dims[axis2] = Dims[axis2], Dims[axis1]
        return Panel(data=Data, items=Dims[0], major_axis=Dims[1], minor_axis=Dims[2])
    def to_frame(self, filter_observations=True):
        Index = pd.MultiIndex.from_product([self._MajorAxis.index, self._MinorAxis.index], names=(self._MajorAxis.name, self._MinorAxis.name))
        df = pd.DataFrame(self._Data.swapaxes(1, 2).swapaxes(0, 2).reshape((self._MajorAxis.shape[0]*self._MinorAxis.shape[0], self._Items.shape[0])), index=Index, columns=self._Items.index)
        if filter_observations:
            return df.dropna(axis=0, how="all")
        else:
            return df
    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True):
        if axis==0: Index = self._Items.sort_index(axis=0, level=level, ascending=ascending, inplace=False, kind=kind, na_position=na_position, sort_remaining=sort_remaining)
        elif axis==1: Index = self._MajorAxis.sort_index(axis=0, level=level, ascending=ascending, inplace=False, kind=kind, na_position=na_position, sort_remaining=sort_remaining)
        elif axis==2: Index = self._MinorAxis.sort_index(axis=0, level=level, ascending=ascending, inplace=False, kind=kind, na_position=na_position, sort_remaining=sort_remaining)
        else: raise ValueError(f"No axis named {axis} for object type {type(self)}")
        if inplace:
            if axis==0:
                self._Data = self._Data[Index.values]
                self._DTypes = self._DTypes.loc[Index.index]
                self._Items = pd.Series(np.arange(Index.shape[0]), index=Index.index)
            elif axis==1:
                self._Data = self._Data[:, Index.values]
                self._MajorAxis = pd.Series(np.arange(Index.shape[0]), index=Index.index)
            else:
                self._Data = self._Data[:, :, Index.values]
                self._MinorAxis = pd.Series(np.arange(Index.shape[0]), index=Index.index)
        else:
            if axis==0:
                p = Panel(data=self._Data[Index.values], items=Index.index, major_axis=self._MajorAxis.index, minor_axis=self._MinorAxis.index)
                p._DTypes = self._DTypes.loc[Index.index]
            elif axis==1:
                p = Panel(data=self._Data[:, Index.values], items=self._Items.index, major_axis=Index.index, minor_axis=self._MinorAxis.index)
                p._DTypes = self._DTypes
            else:
                p = Panel(data=self._Data[:, :, Index.values], items=self._Items.index, major_axis=self._MajorAxis.index, minor_axis=Index.index)
                p._DTypes = self._DTypes
            return p
    def fillna(self, value=0, inplace=True):
        if inplace:
            self._Data[pd.isnull(self._Data)] = value
            return
        Data = self._Data.copy()
        Data[pd.isnull(Data)] = value
        p = Panel(data=Data, items=self._Items.index, major_axis=self._MajorAxis.index, minor_axis=self._MinorAxis.index)
        p._DTypes = self._DTypes
        p._UniDType = self._UniDType
        return p

if __name__=="__main__":
    np.random.seed(0)
    #p = Panel({
        #"a1": pd.DataFrame(np.random.randn(3,4), index=["b"+str(i) for i in range(1,4)], columns=["c"+str(i) for i in range(1,5)]),
        #"a2": pd.DataFrame(np.random.randn(3,4), index=["b"+str(i) for i in range(1,4)], columns=["c"+str(i) for i in range(1,5)])
    #})
    import datetime as dt
    p = Panel({"a": np.array([[dt.datetime(2021,10,22), dt.datetime(2021, 10, 23)], [None, dt.datetime(2021, 11, 23)]])})
    df = p.loc[:, [0,1,2]].iloc[0]
    print(df)
    p1 = pd.Panel({"a": np.array([[dt.datetime(2021,10,22), dt.datetime(2021, 10, 23)], [None, dt.datetime(2021, 11, 23)]])})
    df1 = p1.loc[:, [0,1,2]].iloc[0]
    print(df1)
    #p = Panel(np.random.randn(2, 3, 4), items=["b", "a"])
    #print(p)
    #print(p.loc["a", :])
    #print(p.loc[:, [0,1], 1:2])
    #print(p.loc["b", :, 2])
    #print(p.iloc[0, :])
    #print(p.iloc[:, [0,1], 1:2])
    #print(p.iloc[1, :, 2])
    #print(p.swapaxes(0, 2))
    #d = p.loc[["b", "c"], :, [3, 4]]
    #print(p.loc[["b", "c"], :, [3, 4]])
    #print(p.to_frame())
    #print(p.sort_index(axis=0))
    
    #p.iloc[0] = np.ones((3,4))
    #p.iloc[[0,1]] = np.random.randn(2,3,4)
    #p.iloc[0:, [1,2]] = np.random.randn(2,2,4)
    #p.iloc[0:, [1,2], 3] = np.random.randn(2,2)
    #p.iloc[:] = np.random.randn(2,3,4)
    #p.iloc[p.values>0] = np.nan
    
    #p.loc["a"] = np.ones((3,4))
    #p.loc[["a", "b"]] = np.random.randn(2,3,4)
    ##p.loc["a":, [1,2]] = np.random.randn(2,2,4)
    #p.loc["a":, [1,2], 3] = np.random.randn(1,2)
    #p.loc[:] = np.random.randn(2,3,4)
    #p.loc[p.values>0] = np.nan
    
    #p["a"] = np.ones((3,4))
    #p[["a", "b"]] = np.random.randn(2,3,4)
    ##p["a":, [1,2]] = np.random.randn(2,2,4)
    #p["a":, [1,2], 3] = np.random.randn(1,2)
    #p[:] = np.random.randn(2,3,4)
    #p[p.values>0] = np.nan
    
    #p = Panel({}, items=["b","a"])
    #p = Panel(items=["a", "b"], major_axis=[1,2,3], minor_axis=[4,5,6])
    #print(p)
    print("===")