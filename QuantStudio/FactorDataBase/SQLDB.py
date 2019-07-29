# coding=utf-8
"""基于 SQL 数据库的因子库"""
import re
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.FileFun import readJSONFile
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable

def _identifyDataType(dtypes):
    if np.dtype('O') in dtypes.values: return 'varchar(40)'
    else: return 'double'

class _FactorTable(FactorTable):
    """SQLDB 因子表"""
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
    def getID(self, ifactor_name=None, idt=None):
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
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None):
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
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (not dts) or (not ids): return pd.DataFrame(columns=["DateTime", "ID"]+factor_names)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB._Prefix+self.Name
        # 形成 SQL 语句, 时点, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".ID, "
        for iField in factor_names: SQLStr += DBTableName+"."+iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        if (len(dts)==1) or (np.nanmax(np.diff(np.array(dts))).days<=1):
            SQLStr += "WHERE "+DBTableName+".DateTime>='"+dts[0].strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            SQLStr += "AND "+DBTableName+".DateTime<='"+dts[-1].strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        else:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        if len(ids)<=1000:
            SQLStr += "AND ("+genSQLInCondition(DBTableName+".ID", ids, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+DBTableName+".DateTime, "+DBTableName+".ID"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["DateTime", "ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["DateTime", "ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["DateTime", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names]
        return Data.loc[:, dts, ids]

class SQLDB(WritableFactorDB):
    """SQLDB"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Connection = None# 数据库链接
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
        elif (self.Connector=="pymssql") or ((self.Connector=="default") and (self.DBType=="SQL Server")):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                if self.Connector!="default": raise e
        elif (self.Connector=="mysql.connector") or ((self.Connector=="default") and (self.DBType=="MySQL")):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet, autocommit=True)
            except Exception as e:
                if self.Connector!="default": raise e
        if self._Connection is None:
            if self.Connector not in ("default", "pyodbc"):
                self._Connection = None
                raise __QS_Error__("不支持该连接器(connector) : "+self.Connector)
            else:
                import pyodbc
                if self.DSN: self._Connection = pyodbc.connect("DSN=%s;PWD=%s" % (self.DSN, self.Pwd))
                else: self._Connection = pyodbc.connect("DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s" % (self.DBType, self.DBName, self.IPAddr+","+str(self.Port), self.User, self.Pwd))
                self.Connector = "pyodbc"
        self._PID = os.getpid()
        return 0
    def connect(self):
        self._connect()
        if self.DBType=="MySQL":
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
                nPrefix = len(self._Prefix)
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
        SQLStr = "CREATE INDEX "+index_name+" USING "+index_type+" ON "+self.TablePrefix+self._Prefix+table_name+"("+", ".join(fields)+")"
        return self.execute(SQLStr)
    # 创建表, field_types: {字段名: 数据类型}
    def createTable(self, table_name, field_types):
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` DATETIME(6) NOT NULL, `ID` VARCHAR(40) NOT NULL, " % (self.TablePrefix+self._Prefix+table_name)
        for iField in field_types: SQLStr += "`%s` %s, " % (iField, field_types[iField])
        SQLStr += "PRIMARY KEY (`DateTime`, `ID`)) ENGINE=InnoDB DEFAULT CHARSET=utf8"
        self.execute(SQLStr)
        try:
            self.addIndex(table_name+"_index", table_name)
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
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name not in self._TableFactorDict[table_name]: raise __QS_Error__("因子: '%s' 不存在!" % old_factor_name)
        if (new_factor_name!=old_factor_name) and (new_factor_name in self._TableFactorDict[table_name]): raise __QS_Error__("表中的因子: '%s' 已存在!" % new_factor_name)
        SQLStr = "ALTER TABLE "+self.TablePrefix+self._Prefix+table_name
        SQLStr += " CHANGE COLUMN `"+old_factor_name+"` `"+new_factor_name+"`"
        self.execute(SQLStr)
        self._TableFactorDict[table_name][new_factor_name] = self._TableFactorDict[table_name].pop(old_factor_name)
        return 0
    def deleteFactor(self, table_name, factor_names):
        if not factor_names: return 0
        SQLStr = "ALTER TABLE "+self.TablePrefix+self._Prefix+table_name
        for iFactorName in factor_names: SQLStr += " DROP COLUMN `"+iFactorName+"`,"
        self.execute(SQLStr[:-1])
        FactorIndex = list(set(self._TableFactorDict.get(table_name, pd.Series()).index).difference(set(factor_names)))
        if not FactorIndex: self._TableFactorDict.pop(table_name, None)
        else: self._TableFactorDict[table_name] = self._TableFactorDict[table_name][FactorIndex]
        return 0
    def deleteData(self, table_name, ids=None, dts=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        if (ids is None) and (dts is None):
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
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        FieldTypes = {iFactorName:_identifyDataType(data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
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
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
            SQLStr += "`"+iFactorName+"`, "
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        NewData = NewData[pd.notnull(NewData).any(axis=1)]
        if NewData.shape[0]==0: return 0
        NewData = NewData.astype("O").where(pd.notnull(NewData), None)
        if self.Connector=="pyodbc":
            SQLStr = SQLStr[:-2] + ") VALUES (" + "?, " * (NewData.shape[1]+2)
        else:
            SQLStr = SQLStr[:-2] + ") VALUES (" + "%s, " * (NewData.shape[1]+2)
        SQLStr = SQLStr[:-2]+") "
        Cursor = self._Connection.cursor()
        Cursor.executemany(SQLStr, NewData.reset_index().values.tolist())
        self._Connection.commit()
        Cursor.close()
        return 0