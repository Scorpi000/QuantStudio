# -*- coding: utf-8 -*-
import os
import mmap
import uuid
from multiprocessing import Queue, Lock
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, File

from QuantStudio import __QS_Object__, __QS_Error__

os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"

class QSSQLObject(__QS_Object__):
    """基于关系数据库的对象"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", "sqlite3", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pymysql", "sqlite3", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    SQLite3File = File(label="sqlite3文件", arg_type="File", order=10)
    AdjustTableName = Bool(False, arg_type="Bool", label="调整表名", order=11)
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
    def _connect(self):
        self._Connection = None
        if (self.Connector=="cx_Oracle") or ((self.Connector=="default") and (self.DBType=="Oracle")):
            try:
                import cx_Oracle
                self._Connection = cx_Oracle.connect(self.User, self.Pwd, cx_Oracle.makedsn(self.IPAddr, str(self.Port), self.DBName))
            except Exception as e:
                Msg = ("尝试使用 cx_Oracle 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "cx_Oracle"
        elif (self.Connector=="pymssql") or ((self.Connector=="default") and (self.DBType=="SQL Server")):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                Msg = ("尝试使用 pymssql 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "pymssql"
        elif (self.Connector=="mysql.connector") or ((self.Connector=="default") and (self.DBType=="MySQL")):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet, autocommit=True)
            except Exception as e:
                Msg = ("尝试使用 mysql.connector 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "mysql.connector"
        elif self.Connector=="pymysql":
            try:
                import pymysql
                self._Connection = pymysql.connect(host=self.IPAddr, port=self.Port, user=self.User, password=self.Pwd, db=self.DBName, charset=self.CharSet)
            except Exception as e:
                Msg = ("尝试使用 pymysql 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                raise e
            else:
                self._Connector = "pymysql"
        elif (self.Connector=="sqlite3") or ((self.Connector=="default") and (self.DBType=="sqlite3")):
            try:
                import sqlite3
                self._Connection = sqlite3.connect(self.SQLite3File)
            except Exception as e:
                Msg = ("尝试使用 sqlite3 连接数据库 '%s' 失败: %s" % (self.SQLite3File, str(e)))
                self._QS_Logger.error(Msg)
                raise e
            else:
                self._Connector = "sqlite3"
        if self._Connection is None:
            if self.Connector not in ("default", "pyodbc"):
                self._Connection = None
                Msg = "不支持该连接器(connector): "+self.Connector
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
            elif self.DSN:
                try:
                    import pyodbc
                    self._Connection = pyodbc.connect("DSN=%s;PWD=%s" % (self.DSN, self.Pwd))
                except Exception as e:
                    Msg = ("尝试使用 pyodbc 连接数据库 'DSN: %s' 失败: %s" % (self.DSN, str(e)))
                    self._QS_Logger.error(Msg)
                    raise e
            else:
                try:
                    import pyodbc
                    self._Connection = pyodbc.connect("DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s" % (self.DBType, self.DBName, self.IPAddr+","+str(self.Port), self.User, self.Pwd))
                except Exception as e:
                    Msg = ("尝试使用 pyodbc 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                    self._QS_Logger.error(Msg)
                    raise e
            self._Connector = "pyodbc"
        self._PID = os.getpid()
        return 0
    def connect(self):
        self._connect()
        Cursor = self._Connection.cursor()
        if not self.AdjustTableName:
            self._AllTables = []
        elif self.DBType=="SQL Server":
            Cursor.execute("SELECT Name FROM SysObjects Where XType='U'")
            self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
        elif self.DBType=="MySQL":
            Cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='"+self.DBName+"' AND table_type='base table'")
            self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
        elif self.DBType=="Oracle":
            Cursor.execute("SELECT table_name FROM user_tables WHERE TABLESPACE_NAME IS NOT NULL AND user='"+self.User+"'")
            self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
        elif self.DBType=="sqlite3":
            Cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
        else:
            self._AllTables = []
        Cursor.close()
        return 0
    def disconnect(self):
        if self._Connection is not None:
            try:
                self._Connection.close()
            except Exception as e:
                self._QS_Logger.warning("数据库 ’%s' 断开错误: %s" % (self.DBName, str(e)))
            finally:
                self._Connection = None
        return 0
    def isAvailable(self):
        return (self._Connection is not None)
    def cursor(self, sql_str=None):
        if self._Connection is None:
            Msg = ("获取 cursor 失败, 数据库 '%s' 尚未连接!" % (self.DBName,))
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
            Msg = ("execute 失败, 数据库 '%s' 尚未连接!" % (self.DBName,))
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