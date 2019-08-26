# coding=utf-8
"""基于关系数据库的风险数据库"""
import os
import json
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.QSObjects import QSSQLObject
from QuantStudio.RiskDataBase.RiskDB import RiskDB, RiskTable, FactorRDB, FactorRT
from QuantStudio import __QS_Error__, __QS_ConfigPath__

class _RiskTable(RiskTable):
    def __init__(self, name, rdb, sys_args={}, config_file=None, **kwargs):
        super().__init__(name=name, rdb=rdb, sys_args=sys_args, config_file=config_file, **kwargs)
        self._DBTableName = self._RiskDB.TablePrefix+self._RiskDB._Prefix+self._Name
    def getDateTime(self, start_dt=None, end_dt=None):
        SQLStr = "SELECT DISTINCT DateTime "
        SQLStr += "FROM "+self._DBTableName+" "
        JoinStr = "WHERE "
        if start_dt is not None:
            SQLStr += JoinStr+"DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            JoinStr = "AND "
        if end_dt is not None:
            SQLStr += JoinStr+"DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY DateTime"
        return [iRslt[0] for iRslt in self._RiskDB.fetchall(SQLStr)]
    def getID(self, idt=None):
        SQLStr = "SELECT JSON_EXTRACT(Cov, '$.columns') AS IDs FROM "+self._DBTableName+" "
        if idt is not None: SQLStr += "WHERE DateTime='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"'"
        else: SQLStr += "WHERE DateTime=(SELECT MAX(DateTime) FROM "+self._DBTableName+")"
        IDs = self._RiskDB.fetchall(SQLStr)
        if not IDs: return []
        return json.loads(IDs[0][0])
    def __QS_readCov__(self, dts, ids=None):
        SQLStr = "SELECT DateTime, Cov "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iCov in self._RiskDB.fetchall(SQLStr):
            iCov = pd.read_json(iCov, orient="split")
            iCov.index = iCov.columns
            if ids is not None:
                if iCov.index.intersection(ids).shape[0]>0: iCov = iCov.loc[ids, ids]
                else: iCov = pd.DataFrame(index=ids, columns=ids)
            Data[iDT] = iCov
        if Data: return pd.Panel(Data).loc[dts]
        return pd.Panel(items=dts, major_axis=ids, minor_axis=ids)

class SQLRDB(QSSQLObject, RiskDB):
    """基于关系数据库的风险数据库"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableAdditionalCols = {}# {表名:[额外的字段名]}
        self._Prefix = "QSR_"
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"SQLRDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "SQLRDB"
    def connect(self):
        super().connect()
        if self.DBType=="MySQL":
            SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS WHERE table_schema='%s' " % self.DBName)
            SQLStr += ("AND TABLE_NAME LIKE '%s%%' " % self._Prefix)
            SQLStr += "AND COLUMN_NAME NOT IN ('Cov') "
            SQLStr += "ORDER BY TABLE_NAME, COLUMN_NAME"
            Rslt = self.fetchall(SQLStr)
            if not Rslt: self._TableAdditionalCols = {}
            else:
                self._TableAdditionalCols = pd.DataFrame(np.array(Rslt), columns=["表", "字段"]).set_index(["表"])["字段"]
                nPrefix = len(self._Prefix)
                self._TableAdditionalCols = {iTable[nPrefix:]:self._TableAdditionalCols.loc[[iTable]].tolist() for iTable in self._TableAdditionalCols.index.drop_duplicates()}
        return 0
    @property
    def TableNames(self):
        return sorted(self._TableAdditionalCols)
    def getTable(self, table_name, args={}):
        return _RiskTable(table_name, self)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableAdditionalCols: raise __QS_Error__("表: '%s' 不存在!" % old_table_name)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableAdditionalCols): raise __QS_Error__("表: '"+new_table_name+"' 已存在!")
        SQLStr = "ALTER TABLE "+self.TablePrefix+self._Prefix+old_table_name+" RENAME TO "+self.TablePrefix+self._Prefix+new_table_name
        self.execute(SQLStr)
        self._TableAdditionalCols[new_table_name] = self._TableAdditionalCols.pop(old_table_name)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableAdditionalCols: return 0
        SQLStr = 'DROP TABLE %s' % (self.TablePrefix+self._Prefix+table_name)
        self.execute(SQLStr)
        self._TableAdditionalCols.pop(table_name)
        return 0
    def deleteDateTime(self, table_name, dts):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        if dts is None:
            SQLStr = "TRUNCATE TABLE "+DBTableName
            return self.execute(SQLStr)
        SQLStr = "DELETE * FROM "+DBTableName
        DTs = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts]
        SQLStr += "WHERE "+genSQLInCondition(DBTableName+".DateTime", DTs, is_str=True, max_num=1000)+" "
        return self.execute(SQLStr)
    def createTable(self, table_name):# 创建表
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` DATETIME(6) NOT NULL, `Cov` JSON NOT NULL, " % (self.TablePrefix+self._Prefix+table_name, )
        SQLStr += "PRIMARY KEY (`DateTime`)) ENGINE=InnoDB DEFAULT CHARSET=utf8"
        self.execute(SQLStr)
        self._TableAdditionalCols[table_name] = self._TableAdditionalCols.get(table_name, ["DateTime"])
        try:
            self.addIndex(table_name+"_index", self._Prefix+table_name, fields=["DateTime"])
        except Exception as e:
            self._QS_Logger.warning("风险表 '%s' 索引创建失败: %s" % (table_name, str(e)))
        return 0
    def writeData(self, table_name, idt, icov):
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "INSERT INTO "+DBTableName+" (`DateTime`, `Cov`) VALUES ("+", ".join([("?" if self.Connector=="pyodbc" else "%s")]*2)+")"
        if table_name not in self._TableAdditionalCols: self.createTable(table_name)
        else: SQLStr += " ON DUPLICATE KEY UPDATE Cov=VALUES(Cov)"
        Cursor = self._Connection.cursor()
        Cursor.execute(SQLStr, (idt, icov.to_json(orient="split", index=False)))
        self._Connection.commit()
        Cursor.close()
        return 0

class _FactorRiskTable(FactorRT):
    def __init__(self, name, rdb, sys_args={}, config_file=None, **kwargs):
        super().__init__(name=name, rdb=rdb, sys_args=sys_args, config_file=config_file, **kwargs)
        self._DBTableName = self._RiskDB.TablePrefix+self._RiskDB._Prefix+self._Name
    @property
    def FactorNames(self):
        SQLStr = "SELECT JSON_EXTRACT(FactorCov, '$.columns') AS IDs FROM "+self._DBTableName+" "
        SQLStr += "WHERE DateTime=(SELECT MAX(DateTime) FROM "+self._DBTableName+" WHERE FactorCov IS NOT NULL)"
        Factors = self._RiskDB.fetchall(SQLStr)
        if not Factors: return []
        return json.loads(Factors[0][0])
    def getDateTime(self, start_dt=None, end_dt=None):
        SQLStr = "SELECT DISTINCT DateTime "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE SpecificRisk IS NOT NULL "
        if start_dt is not None: SQLStr += "AND DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY DateTime"
        return [iRslt[0] for iRslt in self._RiskDB.fetchall(SQLStr)]
    def getID(self, idt=None):
        SQLStr = "SELECT JSON_EXTRACT(SpecificRisk, '$.index') AS IDs FROM "+self._DBTableName+" "
        if idt is not None: SQLStr += "WHERE DateTime='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"'"
        else: SQLStr += "WHERE DateTime=(SELECT MAX(DateTime) FROM "+self._DBTableName+" WHERE SpecificRisk IS NOT NULL)"
        IDs = self._RiskDB.fetchall(SQLStr)
        if not IDs: return []
        return json.loads(IDs[0][0])
    def getFactorReturnDateTime(self, start_dt=None, end_dt=None):
        SQLStr = "SELECT DISTINCT DateTime "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE FactorReturn IS NOT NULL "
        if start_dt is not None: SQLStr += "AND DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY DateTime"
        return [iRslt[0] for iRslt in self._RiskDB.fetchall(SQLStr)]
    def getSpecificReturnDateTime(self, start_dt=None, end_dt=None):
        SQLStr = "SELECT DISTINCT DateTime "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE SpecificReturn IS NOT NULL "
        if start_dt is not None: SQLStr += "AND DateTime>='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND DateTime<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        SQLStr += "ORDER BY DateTime"
        return [iRslt[0] for iRslt in self._RiskDB.fetchall(SQLStr)]
    def __QS_readFactorCov__(self, dts):
        SQLStr = "SELECT DateTime, FactorCov "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE FactorCov IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iCov in self._RiskDB.fetchall(SQLStr):
            iCov = pd.read_json(iCov, orient="split")
            iCov.index = iCov.columns
            Data[iDT] = iCov
        if Data: return pd.Panel(Data).loc[dts]
        return pd.Panel(items=dts)
    def __QS_readSpecificRisk__(self, dts, ids=None):
        SQLStr = "SELECT DateTime, SpecificRisk "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE SpecificRisk IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iSepecificRisk in self._RiskDB.fetchall(SQLStr):
            Data[iDT] = pd.read_json(iSepecificRisk, orient="split", typ="series")
        if not Data: return pd.DataFrame(index=dts, columns=ids)
        Data = pd.DataFrame(Data).T.loc[dts]
        if ids is not None:
            if Data.columns.intersection(ids).shape[0]>0: Data = Data.loc[:, ids]
            else: Data = pd.DataFrame(index=dts, columns=ids)
        return Data
    def __QS_readFactorData__(self, dts, ids=None):
        SQLStr = "SELECT DateTime, FactorData "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE FactorData IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iData in self._RiskDB.fetchall(SQLStr):
            Data[iDT] = pd.read_json(iData, orient="split").T
        if not Data: return pd.Panel(items=[], major_axis=dts, minor_axis=ids)
        Data = pd.Panel(Data).swapaxes(0, 1).loc[:, dts, :]
        if ids is not None:
            if Data.minor_axis.intersection(ids).shape[0]>0: Data = Data.loc[:, :, ids]
            else: Data = pd.Panel(items=Data.items, major_axis=dts, minor_axis=ids)
        return Data
    def readFactorReturn(self, dts):
        SQLStr = "SELECT DateTime, FactorReturn "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE FactorReturn IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iFactorReturn in self._RiskDB.fetchall(SQLStr):
            Data[iDT] = pd.read_json(iFactorReturn, orient="split", typ="series")
        if not Data: return pd.DataFrame(index=dts)
        return pd.DataFrame(Data).T.loc[dts]
    def readSpecificReturn(self, dts, ids=None):
        SQLStr = "SELECT DateTime, SpecificReturn "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE SpecificReturn IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        for iDT, iSpecificReturn in self._RiskDB.fetchall(SQLStr):
            Data[iDT] = pd.read_json(iSpecificReturn, orient="split", typ="series")
        if not Data: return pd.DataFrame(index=dts, columns=([] if ids is None else ids))
        Data = pd.DataFrame(Data).T.loc[dts]
        if ids is not None:
            if Data.columns.intersection(ids).shape[0]>0: Data = Data.loc[:, ids]
            else: Data = pd.DataFrame(index=dts, columns=ids)
        return Data
    def readData(self, data_item, dts):
        SQLStr = "SELECT DateTime, "+data_item+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+data_item+" IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        Data = {}
        Type = None
        for iDT, iData in self._RiskDB.fetchall(SQLStr):
            if Type is None:
                try:
                    Data[iDT] = pd.read_json(iData, orient="split")
                    Type = "frame"
                except:
                    Type = "series"
                    Data[iDT] = pd.read_json(iData, orient="split", typ=Type)
            else:
                Data[iDT] = pd.read_json(iData, orient="split", typ=Type)
        if not Data: return None
        if Type=="series": return pd.DataFrame(Data).T.loc[dts]
        else: return pd.Panel(Data).loc[dts]
    def __QS_readCov__(self, dts, ids=None):
        Data = {}
        SQLStr = "SELECT DateTime, FactorCov, FactorData, SpecificRisk "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE FactorCov IS NOT NULL "
        SQLStr += "AND ("+genSQLInCondition("DateTime", [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts], is_str=True, max_num=1000)+") "
        for iDT, iFactorCov, iFactorData, iSpecificRisk in self._RiskDB.fetchall(SQLStr):
            iFactorCov = pd.read_json(iFactorCov, orient="split")
            iSpecificRisk = pd.read_json(iSpecificRisk, orient="split", typ="series")
            iFactorData = pd.read_json(iFactorData, orient="split")
            if ids is None:
                iIDs = iSpecificRisk.index
                iFactorData = iFactorData.loc[iIDs].values
                iSpecificRisk = iSpecificRisk.values
            else:
                iIDs = ids
                iFactorData = iFactorData.loc[iIDs].values
                iSpecificRisk = iSpecificRisk.loc[iIDs].values
            iCov = np.dot(np.dot(iFactorData, iFactorCov.values), iFactorData.T) + np.diag(iSpecificRisk**2)
            Data[iDT] = pd.DataFrame(iCov, index=iIDs, columns=iIDs)
        if Data: return pd.Panel(Data).loc[dts]
        return pd.Panel(items=dts)

class SQLFRDB(QSSQLObject, FactorRDB):
    """基于关系型数据库的多因子风险数据库"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableAdditionalCols = {}# {表名:[额外的字段名]}
        self._Prefix = "QSFR_"
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"SQLFRDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "SQLFRDB"
    def connect(self):
        super().connect()
        nPrefix = len(self._Prefix)
        if self.DBType=="MySQL":
            SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS WHERE table_schema='%s' " % self.DBName)
            SQLStr += ("AND TABLE_NAME LIKE '%s%%' " % self._Prefix)
            SQLStr += "AND COLUMN_NAME NOT IN ('FactorCov', 'FactorData', 'SpecificRisk', 'FactorReturn', 'SpecificReturn') "
            SQLStr += "ORDER BY TABLE_NAME, COLUMN_NAME"
            Rslt = self.fetchall(SQLStr)
            if not Rslt: self._TableAdditionalCols = {}
            else:
                self._TableAdditionalCols = pd.DataFrame(np.array(Rslt), columns=["表", "字段"]).set_index(["表"])["字段"]
                nPrefix = len(self._Prefix)
                self._TableAdditionalCols = {iTable[nPrefix:]:self._TableAdditionalCols.loc[[iTable]].tolist() for iTable in self._TableAdditionalCols.index.drop_duplicates()}
        return 0
    @property
    def TableNames(self):
        return sorted(self._TableAdditionalCols)
    def getTable(self, table_name, args={}):
        return _FactorRiskTable(table_name, self)
    def renameTable(self, old_table_name, new_table_name):
        return SQLRDB.renameTable(self, old_table_name, new_table_name)
    def deleteTable(self, table_name):
        return SQLRDB.deleteTable(self, table_name)
    def deleteDateTime(self, table_name, dts):
        return SQLRDB.deleteDateTime(self, table_name, dts)
    def createTable(self, table_name):# 创建表
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (`DateTime` DATETIME(6) NOT NULL, `FactorData` JSON, `FactorCov` JSON, `SpecificRisk` JSON, `FactorReturn` JSON, `SpecificReturn` JSON, " % (self.TablePrefix+self._Prefix+table_name, )
        SQLStr += "PRIMARY KEY (`DateTime`)) ENGINE=InnoDB DEFAULT CHARSET=utf8"
        self.execute(SQLStr)
        self._TableAdditionalCols[table_name] = self._TableAdditionalCols.get(table_name, ["DateTime"])
        try:
            self.addIndex(table_name+"_index", self._Prefix+table_name, fields=["DateTime"])
        except Exception as e:
            self._QS_Logger.warning("风险表 '%s' 索引创建失败: %s", % (table_name, str(e)))
        return 0
    def writeData(self, table_name, idt, factor_data=None, factor_cov=None, specific_risk=None, factor_ret=None, specific_ret=None, **kwargs):
        if table_name not in self._TableAdditionalCols: self.createTable(table_name)
        DBTableName = self.TablePrefix+self._Prefix+table_name
        SQLStr = "INSERT INTO "+DBTableName+" (`DateTime`, "
        SubSQLStr = "ON DUPLICATE KEY UPDATE "
        Data = [idt]
        if factor_data is not None:
            SQLStr += "`FactorData`, "
            Data.append(factor_data.to_json(orient="split"))
            SubSQLStr += "FactorData=VALUES(FactorData), "
        if factor_cov is not None:
            SQLStr += "`FactorCov`, "
            Data.append(factor_cov.to_json(orient="split", index=False))
            SubSQLStr += "FactorCov=VALUES(FactorCov), "
        if specific_risk is not None:
            SQLStr += "`SpecificRisk`, "
            Data.append(specific_risk.to_json(orient="split"))
            SubSQLStr += "SpecificRisk=VALUES(SpecificRisk), "
        if factor_ret is not None:
            SQLStr += "`FactorReturn`, "
            Data.append(factor_ret.to_json(orient="split"))
            SubSQLStr += "FactorReturn=VALUES(FactorReturn), "
        if specific_ret is not None:
            SQLStr += "`SpecificReturn`, "
            Data.append(specific_ret.to_json(orient="split"))
            SubSQLStr += "SpecificReturn=VALUES(SpecificReturn), "
        if kwargs:
            AlterSQLStr = "ALTER TABLE "+DBTableName+" ADD "
            for iKey, iValue in kwargs.items():
                SQLStr += "`"+iKey+"`, "
                Data.append(iValue.to_json(orient="split"))
                SubSQLStr += iKey+"=VALUES("+iKey+"), "
                if iKey not in self._TableAdditionalCols[table_name]:
                    AlterSQLStr += iKey+" JSON, "
                    self._TableAdditionalCols[table_name].append(iKey)
            if AlterSQLStr.find("JSON")!=-1: self.execute(AlterSQLStr[:-2])
        if len(Data)==1: return 0
        SQLStr, SubSQLStr = SQLStr[:-2], SubSQLStr[:-2]
        SQLStr += ") VALUES ("+", ".join([("?" if self.Connector=="pyodbc" else "%s")]*len(Data))+") "+SubSQLStr
        Cursor = self._Connection.cursor()
        Cursor.execute(SQLStr, tuple(Data))
        self._Connection.commit()
        Cursor.close()
        return 0