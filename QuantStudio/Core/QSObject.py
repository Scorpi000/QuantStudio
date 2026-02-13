# -*- coding: utf-8 -*-
import os
import re
import time
from pathlib import Path
from collections import OrderedDict
from typing import Literal

import numpy as np
import pandas as pd
import fasteners
from pydantic import Field

from QuantStudio.Core import __QS_Error__, __QS_Object__


os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"

class QSSQLObject(__QS_Object__):
    """基于关系数据库的对象"""
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name: str = Field(default="QSSQLObject", frozen=True, title="名称")
        DBType: Literal["MySQL", "SQL Server", "Oracle", "PostgreSQL"] = Field(default="MySQL", frozen=True, title="数据库类型", exclude=True)
        DBName: str = Field(default="Scorpion", title="数据库名", frozen=True, exclude=True)
        IPAddr: str = Field(default="127.0.0.1", title="IP地址", frozen=True, exclude=True)
        Port: int = Field(default=3306, ge=0, le=65535, title="端口", frozen=True, exclude=True)
        User: str = Field(default="root", title="用户名", frozen=True, exclude=True)
        Pwd: str = Field(default="", title="密码", frozen=True, exclude=True)
        TablePrefix: str = Field(default="", title="表名前缀", frozen=True, exclude=True)
        CharSet: Literal["utf8", "utf8mb4", "gbk", "gb2312", "gb18030", "cp936", "big5"] = Field(default="utf8", title="字符集", frozen=True, exclude=True)
        Connector: Literal["default", "cx_Oracle", "pymssql", "mysql.connector", "pymysql", "psycopg2", "pyodbc"] = Field(default="default", title="连接器", frozen=True, exclude=True)
        ConnRetryNum: int = Field(default=3, title="连接重试次数", frozen=False, ge=1, exclude=True)
        ConnIntervalSeconds: float = Field(default=30, title="连接重试间隔", frozen=False, ge=0, exclude=True)
        DSN: str = Field(default="", title="数据源", frozen=True, exclude=True)
        AdditionalConnArgs: dict = Field(default={}, title="其他连接参数", frozen=False, exclude=True)
        AdjustTableName: bool = Field(default=False, title="调整表名", frozen=True, exclude=True)

    def __init__(self, args={}, config_file=None, **kwargs):
        self._Connection = None# 连接对象
        self._Connector = None# 实际使用的数据库链接器
        self._AllTables = []# 数据库中的所有表名, 用于查询时解决大小写敏感问题
        self._PID = None# 保存数据库连接创建时的进程号
        self._SQLFun = {}
        return super().__init__(args=args, config_file=config_file, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_Connection"] = (self._Connection is not None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._Connection: self._connect()
        else: self._Connection = None

    @property
    def Name(self):
        return self._QSArgs.Name

    @property
    def Connection(self):
        if self._Connection is not None:
            if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        return self._Connection

    def _connect(self):
        Connector, IPAddr, Port, User, Pwd, DBName, DBType, CharSet = self._QSArgs.Connector, self._QSArgs.IPAddr, self._QSArgs.Port, self._QSArgs.User, self._QSArgs.Pwd, self._QSArgs.DBName, self._QSArgs.DBType, self._QSArgs.CharSet
        self._Connection = None
        if (Connector=="cx_Oracle") or ((Connector=="default") and (DBType=="Oracle")):
            for i in range(self._QSArgs.ConnRetryNum):
                try:
                    import cx_Oracle
                    self._Connection = cx_Oracle.connect(User, Pwd, cx_Oracle.makedsn(IPAddr, str(Port), DBName), **self._QSArgs.AdditionalConnArgs)
                except Exception as e:
                    Msg = ("'%s' 第 %d 次尝试使用 cx_Oracle 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, i+1, User, IPAddr, Port, DBName, str(e)))
                    self._QS_Logger.error(Msg)
                    time.sleep(self._QSArgs.ConnIntervalSeconds)
                else:
                    self._Connector = "cx_Oracle"
                    break
            else:
                if Connector != "default": raise e
        elif (Connector=="pymssql") or ((Connector=="default") and (DBType=="SQL Server")):
            for i in range(self._QSArgs.ConnRetryNum):
                try:
                    import pymssql
                    self._Connection = pymssql.connect(server=IPAddr, port=str(Port), user=User, password=Pwd, database=DBName, charset=CharSet, **self._QSArgs.AdditionalConnArgs)
                except Exception as e:
                    Msg = ("'%s' 第 %d 次尝试使用 pymssql 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, i+1, User, IPAddr, Port, DBName, str(e)))
                    self._QS_Logger.error(Msg)
                    time.sleep(self._QSArgs.ConnIntervalSeconds)
                else:
                    self._Connector = "pymssql"
                    break
            else:
                if Connector != "default": raise e
        elif (Connector=="mysql.connector") or ((Connector=="default") and (DBType=="MySQL")):
            for i in range(self._QSArgs.ConnRetryNum):
                try:
                    import mysql.connector
                    self._Connection = mysql.connector.connect(host=IPAddr, port=str(Port), user=User, password=Pwd, database=DBName, charset=CharSet, autocommit=True, **self._QSArgs.AdditionalConnArgs)
                except Exception as e:
                    Msg = ("'%s' 第 %d 次尝试使用 mysql.connector 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, i+1, User, IPAddr, Port, DBName, str(e)))
                    self._QS_Logger.error(Msg)
                    time.sleep(self._QSArgs.ConnIntervalSeconds)
                else:
                    self._Connector = "mysql.connector"
            else:
                if Connector != "default": raise e
        elif (Connector=="psycopg2") or ((Connector=="default") and (DBType=="PostgreSQL")):
            for i in range(self._QSArgs.ConnRetryNum):
                try:
                    import psycopg2
                    self._Connection = psycopg2.connect(host=IPAddr, port=int(Port), user=User, password=Pwd, database=DBName, **self._QSArgs.AdditionalConnArgs)
                except Exception as e:
                    Msg = ("'%s' 第 %d 次尝试使用 psycopg2 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, i+1, User, IPAddr, Port, DBName, str(e)))
                    self._QS_Logger.error(Msg)
                    time.sleep(self._QSArgs.ConnIntervalSeconds)
                else:
                    self._Connector = "psycopg2"
                    break
            else:
                if Connector!="default": raise e
        elif Connector=="pymysql":
            for i in range(self._QSArgs.ConnRetryNum):
                try:
                    import pymysql
                    self._Connection = pymysql.connect(host=IPAddr, port=Port, user=User, password=Pwd, db=DBName, charset=CharSet, **self._QSArgs.AdditionalConnArgs)
                except Exception as e:
                    Msg = ("'%s' 第 %d 次尝试使用 pymysql 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, i+1, User, IPAddr, Port, DBName, str(e)))
                    self._QS_Logger.error(Msg)
                    time.sleep(self._QSArgs.ConnIntervalSeconds)
                else:
                    self._Connector = "pymysql"
                    break
            else:
                raise e
        if self._Connection is None:
            if Connector not in ("default", "pyodbc"):
                self._Connection = None
                Msg = ("'%s' 连接数据库时错误: 不支持该连接器(connector) '%s'" % (self.Name, Connector))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
            elif self._QSArgs.DSN:
                for i in range(self._QSArgs.ConnRetryNum):
                    try:
                        import pyodbc
                        self._Connection = pyodbc.connect("DSN=%s;PWD=%s" % (self._QSArgs.DSN, Pwd), **self._QSArgs.AdditionalConnArgs)
                    except Exception as e:
                        Msg = ("'%s' 第 %d 次尝试使用 pyodbc 连接数据库 'DSN: %s' 失败: %s" % (self.Name, i+1, self._QSArgs.DSN, str(e)))
                        self._QS_Logger.error(Msg)
                        time.sleep(self._QSArgs.ConnIntervalSeconds)
                    else:
                        self._Connector = "pyodbc"
                        break
                else:
                    raise e
            else:
                for i in range(self._QSArgs.ConnRetryNum):
                    try:
                        import pyodbc
                        self._Connection = pyodbc.connect("DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s" % (DBType, DBName, IPAddr+","+str(Port), User, Pwd), **self._QSArgs.AdditionalConnArgs)
                    except Exception as e:
                        Msg = ("'%s' 第 %d 次尝试使用 pyodbc 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, i+1, User, IPAddr, Port, DBName, str(e)))
                        self._QS_Logger.error(Msg)
                        time.sleep(self._QSArgs.ConnIntervalSeconds)
                    else:
                        self._Connector = "pyodbc"
                        break
                else:
                    raise e
        self._PID = os.getpid()
        return 0

    def connect(self):
        self._connect()
        if not self._QSArgs.AdjustTableName:
            self._AllTables = []
        else:
            self._AllTables = self.getDBTable()
        # 设置特异性参数
        if self._Connector=="pyodbc":
            self._PlaceHolder = "?"
        else:
            self._PlaceHolder = "%s"
        # 设置 SQL 相关特异性函数
        if self._QSArgs.DBType=="MySQL":
            self._SQLFun = {"toDate": "DATE(%s)"}
        elif self._QSArgs.DBType=="PostgreSQL":
            self._SQLFun = {"toDate": "CAST(%s AS DATE)"}
        elif self._QSArgs.DBType=="Oracle":
            self._SQLFun = {"toDate": "CAST(%s AS DATE)"}# TOTEST
        elif self._QSArgs.DBType=="SQL Server":
            self._SQLFun = {"toDate": "CAST(%s AS DATE)"}# TOTEST
        else:
            #raise NotImplementedError("'%s' 调用方法 connect 时错误: 尚不支持的数据库类型" % (self.Name, self._QSArgs.DBType))
            self._SQLFun = {}
        return self

    def disconnect(self):
        if self._Connection is not None:
            try:
                self._Connection.close()
            except Exception as e:
                self._QS_Logger.warning("'%s' 断开数据库错误: %s" % (self.Name, str(e)))
            finally:
                self._Connection = None
        return 0

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
        if self._QSArgs.AdjustTableName:
            for iTable in self._AllTables:
                sql_str = re.sub(iTable, iTable, sql_str, flags=re.IGNORECASE)
        Cursor.execute(sql_str)
        return Cursor

    def fetchall(self, sql_str, header=False):
        Cursor = self.cursor(sql_str=sql_str)
        Data = Cursor.fetchall()
        if not header:
            Cursor.close()
            return Data
        Header = [iCol[0] for iCol in Cursor.description]
        Cursor.close()
        return Data, Header

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
            if self._QSArgs.DBType=="SQL Server":
                SQLStr = "SELECT Name FROM SysObjects Where XType='U'"
                TableField = "Name"
            elif self._QSArgs.DBType=="MySQL":
                SQLStr = "SELECT table_name FROM information_schema.tables WHERE table_schema='"+self._QSArgs.DBName+"' AND table_type='base table'"
                TableField = "table_name"
            elif self._QSArgs.DBType=="Oracle":
                SQLStr = "SELECT table_name FROM user_tables WHERE TABLESPACE_NAME IS NOT NULL AND user='"+self._QSArgs.User+"'"
                TableField = "table_name"
            elif self._QSArgs.DBType=="PostgreSQL":
                SQLStr = f"SELECT table_name FROM information_schema.tables WHERE table_catalog='{self._QSArgs.DBName}' AND table_schema='public' AND table_type='BASE TABLE'"
                TableField = "table_name"
            else:
                raise __QS_Error__("不支持的数据库类型 '%s'" % self._QSArgs.DBType)
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
        SQLStr = "ALTER TABLE "+self._QSArgs.TablePrefix+old_table_name+" RENAME TO "+self._QSArgs.TablePrefix+new_table_name
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
        if self._QSArgs.DBType=="MySQL":
            SQLStr = "CREATE TABLE IF NOT EXISTS %s (" % (self._QSArgs.TablePrefix+table_name)
            for iField, iDataType in field_types.items(): SQLStr += "`%s` %s, " % (iField, iDataType)
            if primary_keys:
                SQLStr += "PRIMARY KEY (`"+"`,`".join(primary_keys)+"`))"
            else:
                SQLStr = SQLStr[:-2] + ")"
            SQLStr += " ENGINE=InnoDB DEFAULT CHARSET="+self._QSArgs.CharSet
            IndexType = "BTREE"
        else:
            raise NotImplementedError("'%s' 调用方法 createDBTable 在数据库中创建表 '%s' 时错误: 尚不支持的数据库类型" % (self.Name, table_name, self._QSArgs.DBType))
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
        SQLStr = "DROP TABLE %s" % (self._QSArgs.TablePrefix+table_name)
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
            SQLStr = "CREATE INDEX "+index_name+" USING "+index_type+" ON "+self._QSArgs.TablePrefix+table_name+"("+", ".join(fields)+")"
        else:
            SQLStr = "CREATE INDEX "+index_name+" ON "+self._QSArgs.TablePrefix+table_name+"("+", ".join(fields)+")"
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
            if self._QSArgs.DBType=="MySQL":
                SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.columns WHERE table_schema='%s' " % self._QSArgs.DBName)
                TableField, ColField = "TABLE_NAME", "COLUMN_NAME"
            elif self._QSArgs.DBType=="PostgreSQL":
                SQLStr = "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.columns WHERE table_schema = 'public' AND table_catalog = '{self._QSArgs.DBName}'"
                TableField, ColField = "TABLE_NAME", "COLUMN_NAME"
            elif self._QSArgs.DBType=="SQL Server":
                SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.columns WHERE table_schema='%s' " % self._QSArgs.DBName)
                TableField, ColField = "TABLE_NAME", "COLUMN_NAME"
            elif self._QSArgs.DBType=="Oracle":
                SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM user_tab_columns")
                TableField, ColField = "TABLE_NAME", "COLUMN_NAME"
            else:
                raise __QS_Error__("不支持的数据库类型 '%s'" % self._QSArgs.DBType)
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
        SQLStr = "ALTER TABLE %s " % (self._QSArgs.TablePrefix+table_name)
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
            SQLStr = "ALTER TABLE "+self._QSArgs.TablePrefix+table_name
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
            SQLStr = "ALTER TABLE "+self._QSArgs.TablePrefix+table_name
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
        SQLStr = "TRUNCATE TABLE %s" % (self._QSArgs.TablePrefix+table_name)
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 truncateDBTable 清空数据库中的表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        else:
            self._QS_Logger.info("'%s' 调用方法 truncateDBTable 清空数据库中的表 '%s'" % (self.Name, table_name))
        return 0


# 文件锁
class QSFileLock(object):
    def __init__(self, path_or_lock, proc_lock=None):
        if isinstance(path_or_lock, (str, Path)):
            self._FileLock = fasteners.InterProcessLock(path_or_lock)
        else:
            self._FileLock = path_or_lock
        self._ProcLock = proc_lock
    def acquire(self):
        if self._ProcLock is not None: self._ProcLock.acquire()
        return self._FileLock.acquire()
    def release(self):
        self._FileLock.release()
        if self._ProcLock is not None: self._ProcLock.release()
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


# pandas Panel 的 QS 实现
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

class _LocIndexer(object):
    def __init__(self, p):
        self._p = p
    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key, slice(None), slice(None))
        else: key += (slice(None),) * (3 - len(key))
        if len(key)>3: raise IndexError("QuantStudio.Tools.QSObjects.Panel.loc: Too many indexers")
        try:
            Items = self._p._Items.loc[key[0]]
        except KeyError:
            Items = self._p._Items.reindex(key[0])
        try:
            MajorAxis = self._p._MajorAxis.loc[key[1]]
        except KeyError:
            MajorAxis = self._p._MajorAxis.reindex(key[1])
        try:
            MinorAxis = self._p._MinorAxis.loc[key[2]]
        except KeyError:
            MinorAxis = self._p._MinorAxis.reindex(key[2])
        KeepDim = (isinstance(Items, pd.Series), isinstance(MajorAxis, pd.Series), isinstance(MinorAxis, pd.Series))
        if np.all(KeepDim):# Panel
            DTypes = self._p._DTypes.reindex(index=Items.index)
            UniDType = DTypes.dropna().unique()
            UniDType = (UniDType[0] if UniDType.shape[0]==1 else np.dtype("O"))
            DTypes = DTypes.fillna(value=UniDType)
            Items, MajorAxis, MinorAxis = (Items + 1).fillna(value=0).astype(int), (MajorAxis + 1).fillna(value=0).astype(int), (MinorAxis + 1).fillna(value=0).astype(int)
            TmpShape = (Items.max()+1 if Items.shape[0]>0 else 0, MajorAxis.max()+1 if MajorAxis.shape[0]>0 else 0, MinorAxis.max()+1 if MinorAxis.shape[0]>0 else 0)
            #TmpData = np.full(shape=TmpShape, fill_value=None, dtype=UniDType)
            TmpData, UniDType = _initArray(shape=TmpShape, dtype=UniDType)
            TmpData[1:, 1:, 1:] = self._p._Data[:max(0, TmpShape[0]-1), :max(0, TmpShape[1]-1), :max(0, TmpShape[2]-1)]
            p = Panel(data=TmpData[Items.values][:, MajorAxis.values][:, :, MinorAxis.values].astype(UniDType), items=Items.index, major_axis=MajorAxis.index, minor_axis=MinorAxis.index)
            p._DTypes = DTypes
            return p
        elif sum(KeepDim)==2:# DataFrame
            if not KeepDim[0]:
                try:
                    Data = pd.DataFrame(self._p._Data[Items].astype(self._p._DTypes[key[0]]), index=self._p._MajorAxis.index, columns=self._p._MinorAxis.index)
                except (ValueError, TypeError):
                    Data = pd.DataFrame(self._p._Data[Items], index=self._p._MajorAxis.index, columns=self._p._MinorAxis.index)
                return Data.reindex(index=MajorAxis.index, columns=MinorAxis.index)
            elif not KeepDim[1]:
                Data = pd.DataFrame(self._p._Data[:, MajorAxis].T, index=self._p._MinorAxis.index, columns=self._p._Items.index)
                return Data.reindex(index=MinorAxis.index, columns=Items.index)
            else:
                Data = pd.DataFrame(self._p._Data[:, :, MinorAxis].T, index=self._p._MajorAxis.index, columns=self._p._Items.index)
                return Data.reindex(index=MajorAxis.index, columns=Items.index)
        elif sum(KeepDim)==1:# Series
            if KeepDim[0]:
                Data = pd.Series(self._p._Data[:, MajorAxis, MinorAxis], index=self._p._Items.index)
                return Data.reindex(index=Items.index)
            elif KeepDim[1]:
                try:
                    Data = pd.Series(self._p._Data[Items, :, MinorAxis].astype(self._p._DTypes[key[0]]), index=self._p._MajorAxis.index)
                except (ValueError, TypeError):
                    Data = pd.Series(self._p._Data[Items, :, MinorAxis], index=self._p._MajorAxis.index)
                return Data.reindex(index=MajorAxis.index)
            else:
                try:
                    Data = pd.Series(self._p._Data[Items, MajorAxis].astype(self._p._DTypes[key[0]]), index=self._p._MinorAxis.index)
                except (ValueError, TypeError):
                    Data = pd.Series(self._p._Data[Items, MajorAxis], index=self._p._MinorAxis.index)
                return Data.reindex(index=MinorAxis.index)
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
            DTypes = DTypes.reindex(index=Items.index)
            if isinstance(value, pd.DataFrame):
                value = value.reindex(index=self._p._MajorAxis.index, columns=self._p._MinorAxis.index)
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
        if isinstance(Key0, (slice, list)) and (not (isinstance(Key1, (slice, list)) and isinstance(Key2, (slice, list)))):
            value = np.array(value).T
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
                self._Data = np.r_[[Data[iItem].reindex(index=MajorAxis, columns=MinorAxis).values for iItem in Data]]
            elif major_axis is None:
                self._Data = np.r_[[Data[iItem].reindex(index=MajorAxis).values for iItem in Data]]
            elif minor_axis is None:
                self._Data = np.r_[[Data[iItem].reindex(columns=MinorAxis).values for iItem in Data]]
            else:
                self._Data = np.r_[[Data[iItem].values for iItem in Data]]
        else:
            self._UniDType = np.dtype("O")
            if (major_axis is None) and (minor_axis is None):
                self._Data = np.r_[[Data[iItem].reindex(index=MajorAxis, columns=MinorAxis).values.astype("O") for iItem in Data]]
            elif major_axis is None:
                self._Data = np.r_[[Data[iItem].reindex(index=MajorAxis).values.astype("O") for iItem in Data]]
            elif minor_axis is None:
                self._Data = np.r_[[Data[iItem].reindex(columns=MinorAxis).values.astype("O") for iItem in Data]]
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
        self._DTypes.index = items
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
    # self 和 other 的 major_axis 与 minor_axis 必须一致
    def join(self, other):
        Data = np.r_[self._Data, other._Data]
        p = Panel(data=Data, items=self._Items.index.tolist()+other._Items.index.tolist(), major_axis=self._MajorAxis.index, minor_axis=self._MinorAxis.index)
        p._DTypes = self._DTypes.append(other._DTypes)
        return p


