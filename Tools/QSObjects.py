# -*- coding: utf-8 -*-
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password

from QuantStudio import __QS_Object__, __QS_Error__

os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"

class __QS_SQLObject__(__QS_Object__):
    """基于关系数据库的对象"""
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
        self._Connection = None
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_Connection"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._Connection: self._connect()
        else: self._Connection = None
    def connect(self):
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