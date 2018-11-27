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
from QuantStudio.Tools.QSObjects import QSSQLObject
from QuantStudio import __QS_Object__, __QS_Error__, __QS_LibPath__
from .RiskDataBase import RiskDataBase, FactorRDB

class SQLRDB(QSSQLObject, RiskDataBase):
    """基于关系数据库的风险数据库"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableNames = []# [表名]
        self._Prefix = "QSR_"
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"SQLRDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "SQLRDB"
    def connect(self):
        super().connect()
        nPrefix = len(self._Prefix)
        if self.DBType=="MySQL":
            SQLStr = ("SELECT DISTINCT TABLE_NAME FROM information_schema.COLUMNS WHERE table_schema='%s' " % self.DBName)
            SQLStr += ("AND TABLE_NAME LIKE '%s%%' " % self._Prefix)
            SQLStr += "ORDER BY TABLE_NAME"
            self._TableNames = [iRslt[0][nPrefix:] for iRslt in self.fetchall(SQLStr)]
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
        return [iRslt[0] for iRslt in self.fetchall(SQLStr)]
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
        try:
            self.addIndex(table_name+"_index", table_name)
        except Exception as e:
            print("索引创建失败: "+str(e))
        return 0
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

# TODO: 附加数据的读写
class SQLFRDB(QSSQLObject, FactorRDB):
    """基于关系型数据库的多因子风险数据库"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableNames = []# [表名]
        self._Prefix = "QSFR_"
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"SQLRDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "SQLRDB"
    def connect(self):
        super().connect()
        nPrefix = len(self._Prefix)
        if self.DBType=="MySQL":
            SQLStr = ("SELECT DISTINCT TABLE_NAME FROM information_schema.COLUMNS WHERE table_schema='%s' " % self.DBName)
            SQLStr += ("AND TABLE_NAME LIKE '%s%%' " % self._Prefix)
            SQLStr += "ORDER BY TABLE_NAME"
            self._TableNames = [iRslt[0][nPrefix:] for iRslt in self.fetchall(SQLStr)]
        return 0
    @property
    def TableNames(self):
        return self._TableNames
    def getTableMetaData(self, table_name, key=None):
        return SQLRDB.getTableMetaData(self, table_name, key=key)
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        return SQLRDB.setTableMetaData(self, table_name, key=key, value=value, meta_data=meta_data)
    def renameTable(self, old_table_name, new_table_name):
        return SQLRDB.renameTable(self, old_table_name, new_table_name)
    def deleteTable(self, table_name):
        return SQLRDB.deleteTable(self, table_name)
    # 为某张表增加索引
    def addIndex(self, index_name, table_name, fields=["DateTime"], index_type="BTREE"):
        SQLStr = "CREATE INDEX "+index_name+" USING "+index_type+" ON "+self.TablePrefix+self._Prefix+table_name+"("+", ".join(fields)+")"
        return self.execute(SQLStr)
    # 创建表
    def createTable(self, table_name):
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` DATETIME(6) NOT NULL, `FactorIDs` MEDIUMTEXT, `Factors` TEXT, `FactorData` LONGTEXT, `FactorCov` LONGTEXT, `FactorReturn` MEDIUMTEXT, `SpecificIDs` MEDIUMTEXT, `SpecificRisk` LONGTEXT, `SpecificReturn` LONGTEXT, " % (self.TablePrefix+self._Prefix+table_name, )
        SQLStr += "PRIMARY KEY (`DateTime`)) ENGINE=InnoDB DEFAULT CHARSET=utf8"
        self.execute(SQLStr)
        try:
            self.addIndex(table_name+"_index", table_name)
        except Exception as e:
            print("索引创建失败: "+str(e))
        return 0
    def getTableFactor(self, table_name):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "SELECT "+DBTableName+".Factors "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE Factors IS NOT NULL "
        SQLStr += "ORDER BY "+DBTableName+".DateTime DESC"
        Cursor = self.cursor(SQLStr)
        Factors = json.loads(Cursor.fetchone()[0])
        Cursor.close()
        return Factors
    def getTableDateTime(self, table_name, start_dt=None, end_dt=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "SELECT DISTINCT "+DBTableName+".DateTime "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE SpecificIDs IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+".DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+".DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY "+DBTableName+".DateTime"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr)]
    def getTableID(self, table_name, idt=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "SELECT "+DBTableName+".SpecificIDs "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE SpecificIDs IS NOT NULL "
        if idt is None:
            SQLStr += "ORDER BY "+DBTableName+".DateTime DESC"
        else:
            SQLStr += "AND "+DBTableName+".DateTime='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"'"
        Cursor = self.cursor(SQLStr)
        IDs = json.loads(Cursor.fetchone()[0])
        Cursor.close()
        return IDs
    def getFactorReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "SELECT DISTINCT "+DBTableName+".DateTime "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE FactorReturn IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+".DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+".DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY "+DBTableName+".DateTime"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr)]
    def getSpecificReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "SELECT DISTINCT "+DBTableName+".DateTime "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE SpecificReturn IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+".DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+".DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY "+DBTableName+".DateTime"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr)]
    def readCov(self, table_name, dts, ids=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        # 形成 SQL 语句, DateTime, FactorIDs, FactorCov, FactorData, SpecificIDs, SpecificRisk
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".FactorIDs, "
        SQLStr += DBTableName+".FactorCov, "
        SQLStr += DBTableName+".FactorData, "
        SQLStr += DBTableName+".SpecificIDs, "
        SQLStr += DBTableName+".SpecificRisk "
        SQLStr += " FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+".FactorCov IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iFactorIDs, iFactorCov, iFactorData, iSpecificIDs, iSpecificRisk in self.fetchall(SQLStr):
            iFactorCov = np.array(json.loads(iFactorCov))
            iSpecificRisk = pd.Series(json.loads(iSpecificRisk), index=json.loads(iSpecificIDs))
            iFactorData = pd.DataFrame(json.loads(iFactorData), index=json.loads(iFactorIDs))
            if ids is None:
                iIDs = iSpecificRisk.index
                iFactorData = iFactorData.loc[iIDs].values
                iSpecificRisk = iSpecificRisk.values
            else:
                iIDs = ids
                iFactorData = iFactorData.loc[ids].values
                iSpecificRisk = iSpecificRisk.loc[ids].values
            iCov = np.dot(np.dot(iFactorData, iFactorCov), iFactorData.T) + np.diag(iSpecificRisk**2)
            Data[iDT] = pd.DataFrame(iCov, index=iIDs, columns=iIDs)
        if Data: return pd.Panel(Data).loc[dts]
        return pd.Panel(items=dts)
    def readFactorCov(self, table_name, dts):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        # 形成 SQL 语句, DateTime, Factors, FactorCov
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".Factors, "
        SQLStr += DBTableName+".FactorCov "
        SQLStr += " FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+".FactorCov IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iFactors, iCov in self.fetchall(SQLStr):
            iFactors = json.loads(iFactors)
            Data[iDT] = pd.DataFrame(json.loads(iCov), index=iFactors, columns=iFactors)
        if Data: return pd.Panel(Data).loc[dts]
        return pd.Panel(items=dts)
    def readSpecificRisk(self, table_name, dts, ids=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        # 形成 SQL 语句, DateTime, SepecificIDs, SepecificRisk
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".SepecificIDs, "
        SQLStr += DBTableName+".SepecificRisk "
        SQLStr += " FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+".FactorCov IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iIDs, iSepecificRisk in self.fetchall(SQLStr):
            iIDs = json.loads(iIDs)
            Data[iDT] = pd.Series(json.loads(iSepecificRisk), index=iIDs)
        if not Data: return pd.DataFrame(index=dts, columns=([] if ids is None else ids))
        Data = pd.DataFrame(Data).T.loc[dts]
        if ids is not None:
            if Data.columns.intersection(ids).shape[0]>0: Data = Data.loc[:, ids]
            else: Data = pd.DataFrame(index=dts, columns=ids)
        return Data
    def readFactorData(self, table_name, dts, ids=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        # 形成 SQL 语句, DateTime, Factors, FactorIDs, FactorData
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".Factors, "
        SQLStr += DBTableName+".FactorIDs, "
        SQLStr += DBTableName+".FactorData "
        SQLStr += " FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+".FactorData IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iFactors, iIDs, iData in self.fetchall(SQLStr):
            iFactors = json.loads(iFactors)
            iIDs = json.loads(iIDs)
            Data[iDT] = pd.DataFrame(json.loads(iData), index=iIDs, columns=iFactors).T
        if not Data: return pd.Panel(items=[], major_axis=dts, minor_axis=([] if ids is None else ids))
        Data = pd.Panel(Data).swapaxes(0, 1).loc[:, dts, :]
        if ids is not None:
            if Data.minor_axis.intersection(ids).shape[0]>0: Data = Data.loc[:, :, ids]
            else: Data = pd.Panel(items=Data.items, major_axis=dts, minor_axis=ids)
        return Data
    def readFactorReturn(self, table_name, dts):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        # 形成 SQL 语句, DateTime, Factors, FactorReturn
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".Factors, "
        SQLStr += DBTableName+".FactorReturn "
        SQLStr += " FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+".FactorReturn IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iFactors, iFactorReturn in self.fetchall(SQLStr):
            iFactors = json.loads(iFactors)
            Data[iDT] = pd.Series(json.loads(iFactorReturn), index=iFactors)
        if not Data: return pd.DataFrame(index=dts)
        return pd.DataFrame(Data).T.loc[dts]
    def readSpecificReturn(self, table_name, dts, ids=None):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        # 形成 SQL 语句, DateTime, SepecificIDs, SepecificRisk
        SQLStr = "SELECT "+DBTableName+".DateTime, "
        SQLStr += DBTableName+".SepecificIDs, "
        SQLStr += DBTableName+".SepecificReturn "
        SQLStr += " FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+".SepecificReturn IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+".DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iIDs, iSepecificReturn in self.fetchall(SQLStr):
            iIDs = json.loads(iIDs)
            Data[iDT] = pd.Series(json.loads(iSepecificReturn), index=iIDs)
        if not Data: return pd.DataFrame(index=dts, columns=([] if ids is None else ids))
        Data = pd.DataFrame(Data).T.loc[dts]
        if ids is not None:
            if Data.columns.intersection(ids).shape[0]>0: Data = Data.loc[:, ids]
            else: Data = pd.DataFrame(index=dts, columns=ids)
        return Data
    def writeData(self, table_name, idt, factor_data=None, factor_cov=None, specific_risk=None, factor_ret=None, specific_ret=None, **kwargs):
        SQLStr = "`DateTime`, "
        Data = [idt]
        hasFactorIDs = hasFactors = hasSpecificIDs = False
        if factor_data is not None:
            SQLStr += "`FactorIDs`, `Factors`, `FactorData`, "
            Data.extend([json.dumps(factor_data.index.tolist()), json.dumps(factor_data.columns.tolist()), json.dumps(factor_data.values.tolist())])
            hasFactorIDs = hasFactors = True
        if factor_cov is not None:
            if hasFactors:
                SQLStr += "`FactorCov`, "
                Data.append(json.dumps(factor_cov.values.tolist()))
            else:
                SQLStr += "`Factors`, `FactorCov`, "
                Data.extend([json.dumps(factor_cov.index.tolist()), json.dumps(factor_cov.values.tolist())])
            hasFactors = True
        if specific_risk is not None:
            SQLStr += "`SpecificIDs`, `SpecificRisk`, "
            Data.extend([json.dumps(specific_risk.index.tolist()), json.dumps(specific_risk.values.tolist())])
            hasSpecificIDs = True
        if factor_ret is not None:
            if hasFactors:
                SQLStr += "`FactorReturn`, "
                Data.append(json.dumps(factor_ret.values.tolist()))
            else:
                SQLStr += "`Factors`, `FactorReturn`, "
                Data.extend([json.dumps(factor_ret.index.tolist()), json.dumps(factor_ret.values.tolist())])
            hasFactors = True
        if specific_ret is not None:
            if hasSpecificIDs:
                SQLStr += "`SpecificReturn`, "
                Data.append(json.dumps(specific_ret.values.tolist()))
            else:
                SQLStr += "`SpecificIDs`, `SpecificReturn`, "
                Data.extend([json.dumps(specific_ret.index.tolist()), json.dumps(specific_ret.values.tolist())])
            hasSpecificIDs = True
        SQLStr = SQLStr[:-2]
        if len(SQLStr.split(","))==1: return 0
        RepSym = ("?" if self.Connector=="pyodbc" else "%s")
        if table_name not in self._TableNames:
            self.createTable(table_name)
            self._TableNames.append(table_name)
            SQLStr = "INSERT INTO "+self.TablePrefix+self._Prefix+table_name+" ("+SQLStr+") VALUES ("+", ".join([RepSym]*len(Data))+")"
        else:
            SQLStr = "REPLACE INTO "+self.TablePrefix+self._Prefix+table_name+" ("+SQLStr+") VALUES ("+", ".join([RepSym]*len(Data))+")"
        Cursor = self._Connection.cursor()
        Cursor.execute(SQLStr, tuple(Data))
        self._Connection.commit()
        Cursor.close()
        return 0