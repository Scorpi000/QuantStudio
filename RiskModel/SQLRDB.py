# coding=utf-8
"""基于关系数据库的风险数据库"""
import os
import datetime as dt
import pickle

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
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableNames = []# [表名]
        self._Prefix = "QSR_"
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
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='pymssql') or ((self.Connector=='default') and (self.DBType=='SQL Server')):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='mysql.connector') or ((self.Connector=='default') and (self.DBType=='MySQL')):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet, autocommit=True)
            except Exception as e:
                if self.Connector!='default': raise e
        else:
            if self.Connector not in ('default', 'pyodbc'):
                self._Connection = None
                raise __QS_Error__("不支持该连接器(connector) : "+self.Connector)
            else:
                import pyodbc
                self._Connection = pyodbc.connect('DSN=工作机测试数据库;PWD=%s' % (self.Pwd))
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
            iIDs = pickle.loads(iIDs)
            iCov = pd.DataFrame(pickle.loads(iCov), index=iIDs, columns=iIDs)
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
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` DATETIME(6) NOT NULL, `IDs` BLOB NOT NULL, `Cov` LONGBLOB NOT NULL, " % (self.TablePrefix+self._Prefix+table_name, )
        SQLStr += "PRIMARY KEY (`DateTime`)) ENGINE=InnoDB DEFAULT CHARSET=utf8"
        self.execute(SQLStr)
        #return self.addIndex(table_name+"_index", table_name)
    def writeData(self, table_name, idt, icov):
        if table_name not in self._TableNames:
            self.createTable(table_name)
            self._TableNames.append(table_name)
            SQLStr = "INSERT INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `IDs`, `Cov`) VALUES (?, ?, ?)"
        else:
            SQLStr = "REPLACE INTO "+self.TablePrefix+self._Prefix+table_name+" (`DateTime`, `IDs`, `Cov`) VALUES (?, ?, ?)"
        Cursor = self._Connection.cursor()
        Cursor.execute(SQLStr, (idt, pickle.dumps(icov.index.tolist()), pickle.dumps(icov.values.tolist())))
        self._Connection.commit()
        Cursor.close()
        return 0

class SQLFRDB(FactorRDB):
    """基于关系数据库的多因子风险数据库"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("shuntai11", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableDT = {}#{表名：[时点]}
        self._DataLock = Lock()
        self._Suffix = "h5"
        self._isAvailable = False
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"HDF5FRDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "HDF5FRDB"
    def connect(self):
        if not os.path.isdir(self.MainDir): raise __QS_Error__("不存在 HDF5FRDB 的主目录: %s!" % self.MainDir)
        AllTables = listDirFile(self.MainDir, suffix=self._Suffix)
        TableDT = {}#{表名：[时点]}
        with self._DataLock:
            for iTable in AllTables:
                with h5py.File(self.MainDir+os.sep+iTable+"."+self._Suffix, mode="r") as iFile:
                    if "SpecificRisk" in iFile:
                        iDTs = sorted(iFile["SpecificRisk"])
                        TableDT[iTable] = [dt.datetime.strptime(ijDT, "%Y-%m-%d %H:%M:%S.%f") for ijDT in iDTs]
        self._TableDT = TableDT
        self._isAvailable = True
        return 0
    def disconnect(self):
        self._isAvailable = False
        return 0
    def isAvailable(self):
        return self._isAvailable
    @property
    def TableNames(self):
        return sorted(self._TableDT)    
    def getTableMetaData(self, table_name, key=None):
        return HDF5RDB.getTableMetaData(self, table_name, key=key)
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        return HDF5RDB.setTableMetaData(self, table_name, key=key, value=value, meta_data=meta_data)
    def renameTable(self, old_table_name, new_table_name):
        return HDF5RDB.renameTable(self, old_table_name, new_table_name)
    def deleteTable(self, table_name):
        return HDF5RDB.deleteTable(self, table_name)
    def getTableFactor(self, table_name):
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                DTStr = self._TableDT[table_name][-1].strftime("%Y-%m-%d %H:%M:%S.%f")
                Group = File["FactorCov"]
                if DTStr in Group: return sorted(Group[DTStr]["Factor"][...])
                else: return []
    def getTableDateTime(self, table_name, start_dt=None, end_dt=None):
        return cutDateTime(self._TableDT[table_name], start_dt, end_dt)
    def getTableID(self, table_name, idt=None):
        if idt is None: idt = self._TableDT[table_name][-1]
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                DTStr = self._TableDT[table_name][-1].strftime("%Y-%m-%d %H:%M:%S.%f")
                Group = File["SpecificRisk"]
                if DTStr in Group: return sorted(Group[DTStr]["ID"][...])
                else: return []
    def getFactorReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        FilePath = self.MainDir+os.sep+table_name+"."+self._Suffix
        with self._DataLock:
            if not os.path.isfile(FilePath): return []
            with h5py.File(FilePath, mode="r") as File:
                if "FactorReturn" not in File: return []
                DTs = sorted(File["FactorReturn"])
        DTs = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in DTs]
        return cutDateTime(DTs, start_dt=start_dt, end_dt=end_dt)
    def getSpecificReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        FilePath = self.MainDir+os.sep+table_name+"."+self._Suffix
        with self._DataLock:
            if not os.path.isfile(FilePath): return []
            with h5py.File(FilePath, mode="r") as File:
                if "SpecificReturn" not in File: return []
                DTs = sorted(File["SpecificReturn"])
        DTs = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in DTs]
        return cutDateTime(DTs, start_dt=start_dt, end_dt=end_dt)
    def readCov(self, table_name, dts, ids=None):
        FactorCov = self.readFactorCov(table_name, dts=dts)
        FactorData = self.readFactorData(table_name, dts=dts, ids=ids)
        SpecificRisk = self.readSpecificRisk(table_name, dts=dts, ids=ids)
        Data = {}
        if ids is None:
            ids = SpecificRisk.columns
            FactorData = FactorData.loc[:, :, ids]
        for iDT in FactorCov.items:
            iFactorData = FactorData.loc[:, iDT].values
            iCov = np.dot(np.dot(iFactorData, FactorCov.loc[iDT].values), iFactorData.T) + np.diag(SpecificRisk.loc[iDT].values**2)
            Data[iDT] = pd.DataFrame(iCov, index=ids, columns=ids)
        return pd.Panel(Data).loc[dts]
    def readFactorCov(self, table_name, dts):
        Data = {}
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                Group = File["FactorCov"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    iFactors = iGroup["Factor"][...]
                    Data[iDT] = pd.DataFrame(iGroup["Data"][...], index=iFactors, columns=iFactors)
        if Data: return pd.Panel(Data).loc[dts]
        return pd.Panel(items=dts)
    def readSpecificRisk(self, table_name, dts, ids=None):
        Data = {}
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                Group = File["SpecificRisk"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    Data[iDT] = pd.Series(iGroup["Data"][...], index=iGroup["ID"][...])
        if not Data: return pd.DataFrame(index=dts, columns=([] if ids is None else ids))
        Data = pd.DataFrame(Data).T.loc[dts]
        if ids is not None:
            if Data.columns.intersection(ids).shape[0]>0: Data = Data.loc[:, ids]
            else: Data = pd.DataFrame(index=dts, columns=ids)
        return Data
    def readFactorData(self, table_name, dts, ids=None):
        Data = {}
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                Group = File["FactorData"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    Data[iDT] = pd.DataFrame(iGroup["Data"][...], index=iGroup["ID"][...], columns=iGroup["Factor"][...]).T
        if not Data: return pd.Panel(items=[], index=dts, columns=([] if ids is None else ids))
        Data = pd.Panel(Data).swapaxes(0, 1).loc[:, dts, :]
        if ids is not None:
            if Data.minor_axis.intersection(ids).shape[0]>0: Data = Data.loc[:, :, ids]
            else: Data = pd.Panel(items=Data.items, major_axis=dts, minor_axis=ids)
        return Data
    def readFactorReturn(self, table_name, dts):
        Data = {}
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                Group = File["FactorReturn"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    Data[iDT] = pd.Series(iGroup["Data"][...], index=iGroup["Factor"][...])
        if not Data: return pd.DataFrame(index=dts, columns=[])
        return pd.DataFrame(Data).T.loc[dts]
    def readSpecificReturn(self, table_name, dts, ids=None):
        Data = {}
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                Group = File["SpecificReturn"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    Data[iDT] = pd.Series(iGroup["Data"][...], index=iGroup["ID"][...])
        if not Data: return pd.DataFrame(index=dts, columns=([] if ids is None else ids))
        Data = pd.DataFrame(Data).T.loc[dts]
        if ids is not None:
            if Data.columns.intersection(ids).shape[0]>0: Data = Data.loc[ids]
            else: Data = pd.DataFrame(index=dts, columns=ids)
        return Data
    def readData(self, table_name, data_item, dts):
        Data = {}
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+"."+self._Suffix, mode="r") as File:
                if data_item not in File: return None
                Group = File[data_item]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    if "columns" in iGroup:
                        Type = "DataFrame"
                        Data[iDT] = pd.DataFrame(iGroup["Data"][...], index=iGroup["index"][...], columns=iGroup["columns"][...])
                    else:
                        Type = "Series"
                        Data[iDT] = pd.Series(iGroup["Data"][...], index=iGroup["index"][...])
        if not Data: return None
        if Type=="Series": return pd.DataFrame(Data).T.loc[dts]
        else: return pd.Panel(Data).loc[dts]
    def writeData(self, table_name, idt, factor_data=None, factor_cov=None, specific_risk=None, factor_ret=None, specific_ret=None, **kwargs):
        iDTStr = idt.strftime("%Y-%m-%d %H:%M:%S.%f")
        StrType = h5py.special_dtype(vlen=str)
        FilePath = self.MainDir+os.sep+table_name+"."+self._Suffix
        with self._DataLock:
            if not os.path.isfile(FilePath): open(FilePath, mode="a").close()# h5py 直接创建文件名包含中文的文件会报错.
            with h5py.File(FilePath) as File:
                if factor_data is not None:
                    if "FactorData" not in File: Group = File.create_group("FactorData")
                    else: Group = File["FactorData"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="Factor", shape=(factor_data.shape[1], ), dtype=StrType, data=factor_data.columns.values)
                    iGroup.create_dataset(name="ID", shape=(factor_data.shape[0], ), dtype=StrType, data=factor_data.index.values)
                    iGroup.create_dataset(name="Data", shape=factor_data.shape, dtype=np.float, data=factor_data.values)
                if factor_cov is not None:
                    if "FactorCov" not in File: Group = File.create_group("FactorCov")
                    else: Group = File["FactorCov"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="Factor", shape=(factor_cov.shape[0], ), dtype=StrType, data=factor_cov.index.values)
                    iGroup.create_dataset(name="Data", shape=factor_cov.shape, dtype=np.float, data=factor_cov.values)
                if specific_risk is not None:
                    if "SpecificRisk" not in File: Group = File.create_group("SpecificRisk")
                    else: Group = File["SpecificRisk"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="ID", shape=(specific_risk.shape[0], ), dtype=StrType, data=specific_risk.index.values)
                    iGroup.create_dataset(name="Data", shape=specific_risk.shape, dtype=np.float, data=specific_risk.values)
                if factor_ret is not None:
                    if "FactorReturn" not in File: Group = File.create_group("FactorReturn")
                    else: Group = File["FactorReturn"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="Factor", shape=(factor_ret.shape[0], ), dtype=StrType, data=factor_ret.index.values)
                    iGroup.create_dataset(name="Data", shape=factor_ret.shape, dtype=np.float, data=factor_ret.values)
                if specific_ret is not None:
                    if "SpecificReturn" not in File: Group = File.create_group("SpecificReturn")
                    else: Group = File["SpecificReturn"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="ID", shape=(specific_ret.shape[0], ), dtype=StrType, data=specific_ret.index.values)
                    iGroup.create_dataset(name="Data", shape=specific_ret.shape, dtype=np.float, data=specific_ret.values)
                for iKey, iValue in kwargs.items():
                    if iKey not in File: Group = File.create_group(iKey)
                    else: Group = File[iKey]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="index", shape=(iValue.shape[0], ), dtype=StrType, data=iValue.index.values)
                    iGroup.create_dataset(name="Data", shape=iValue.shape, dtype=np.float, data=iValue.values)
                    if isinstance(iValue, pd.DataFrame): iGroup.create_dataset(name="columns", shape=(iValue.shape[1], ), dtype=StrType, data=iValue.columns.values)
        return 0