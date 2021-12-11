# coding=utf-8
"""基于 Neo4j 数据库的风险数据库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str

from QuantStudio.RiskDataBase.RiskDB import RiskDB, RiskTable, FactorRDB, FactorRT
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.Tools.Neo4jFun import QSNeo4jObject, writeArgs
from QuantStudio.Tools.api import Panel

class _RiskTable(RiskTable):
    IDField = Str("ID", arg_type="String", label="ID字段", order=0)
    def __init__(self, name, rdb, sys_args={}, config_file=None, **kwargs):
        if "ID字段" not in sys_args:
            sys_args["ID字段"] = rdb.IDField
        return super().__init__(name=name, rdb=rdb, sys_args=sys_args, config_file=config_file, **kwargs)
    def getDateTime(self, start_dt=None, end_dt=None):
        CypherStr = f"MATCH (s1) - [r:`风险数据` {{`风险库`: '{self._RiskDB.Name}', `风险表`: '{self._Name}'}}] -> (s2) "
        CypherStr += "WITH keys(r) AS kk UNWIND kk AS ik RETURN collect(DISTINCT ik)"
        Rslt = self._RiskDB.fetchall(CypherStr)
        if not Rslt: return []
        Rslt = Rslt[0][0]
        DTFormat = self._RiskDB._DTFormat
        if start_dt is not None: start_dt = start_dt.strftime(DTFormat)
        if end_dt is not None: end_dt = end_dt.strftime(DTFormat)
        return [dt.datetime.strptime(iDT, DTFormat) for iDT in Rslt if pd.notnull(iDT) and (iDT not in ("风险库", "风险表")) and ((start_dt is None) or (iDT>=start_dt)) and ((end_dt is None) or (iDT<=end_dt))]
    def getID(self, idt=None):
        IDField = self.IDField
        CypherStr = f"MATCH (s1) - [r:`风险数据` {{`风险库`: '{self._RiskDB.Name}', `风险表`: '{self._Name}'}}] -> (s2) "
        CypherStr += f"WHERE s1.`{IDField}` IS NOT NULL AND s2.`{IDField}` IS NOT NULL "
        if idt is not None:
            CypherStr += f"AND r.`{idt.strftime(self._RiskDB._DTFormat)}` IS NOT NULL "
        CypherStr += f"RETURN collect(DISTINCT s1.`{IDField}`)"
        Rslt = self._RiskDB.fetchall(CypherStr)
        if not Rslt: return []
        return sorted(Rslt[0][0])
    def __QS_readCov__(self, dts, ids=None):
        IDField = self.IDField
        DTFormat = self._RiskDB._DTFormat
        CypherStr = f"MATCH (s1) - [r:`风险数据` {{`风险库`: '{self._RiskDB.Name}', `风险表`: '{self._Name}'}}] -> (s2) "
        if ids is None:
            CypherStr += f"WHERE s1.`{IDField}` IS NOT NULL AND s2.`{IDField}` IS NOT NULL "
        else:
            CypherStr += f"WHERE s1.`{IDField}` IN $ids AND s2.`{IDField}` IN $ids "
        CypherStr += "UNWIND $dts AS iDT "
        CypherStr += f"RETURN iDT, s1.`{IDField}`, s2.`{IDField}`, r[iDT]"
        Data = self._RiskDB.fetchall(CypherStr, parameters={"ids": ids, "dts": [iDT.strftime(DTFormat) for iDT in dts]})
        if not Data: return Panel(items=dts, major_axis=ids, minor_axis=ids)
        Data = pd.DataFrame(np.array(Data, dtype="O"), columns=["QS_DT", "ID1", "ID2", "Value"])
        Data = Data.set_index(["ID1", "ID2", "QS_DT"]).iloc[:, 0].unstack().to_panel()
        Data.items = [dt.datetime.strptime(iDT, DTFormat) for iDT in Data.items]
        return Data.loc[dts, ids, ids]

class Neo4jRDB(QSNeo4jObject, RiskDB):
    """基于 Neo4j 的风险数据库"""
    Name = Str("Neo4jRDB", arg_type="String", label="名称", order=-100)
    IDField = Str("ID", arg_type="String", label="ID字段", order=100)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableNames = set()# {表名}
        self._DTFormat = "%Y-%m-%d"
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"Neo4jRDBConfig.json" if config_file is None else config_file), **kwargs)
    def connect(self):
        super().connect()
        CypherStr = f"MATCH () - [r:`风险数据` {{`风险库`: '{self.Name}'}}] -> () RETURN collect(DISTINCT r.`风险表`)"
        TableNames = self.fetchall(CypherStr)
        if not TableNames:
            self._TableNames = set()
        else:
            self._TableNames = set(TableNames[0][0])
        return 0
    @property
    def TableNames(self):
        return sorted(self._TableNames)
    def getTable(self, table_name, args={}):
        if table_name not in self._TableNames:
            Msg = ("风险库 '%s' 调用方法 getTable 错误: 不存在风险表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        return _RiskTable(table_name, self)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableNames:
            Msg = ("风险库 '%s' 调用方法 renameTable 错误: 不存在风险表: '%s'!" % (self.Name, old_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableNames):
            Msg = ("风险库 '%s' 调用方法 renameTable 错误: 新风险表名 '%s' 已经存在于库中!" % (self.Name, new_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        CypherStr = f"MATCH () - [r:`风险数据` {{`风险库`: '{self.Name}', `风险表`: '{old_table_name}'}}] -> () SET r.`风险表`='{new_table_name}'"
        self.execute(CypherStr)
        self._TableNames.remove(old_table_name)
        self._TableNames.add(new_table_name)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableNames: return 0
        CypherStr = f"MATCH () - [r:`风险数据` {{`风险库`: '{self.Name}', `风险表`: '{table_name}'}}] -> () DELETE r"
        self.execute(CypherStr)
        self._TableNames.remove(table_name)
        return 0
    def deleteDateTime(self, table_name, dts):
        CypherStr = f"MATCH () - [r:`风险数据` {{`风险库`: '{self.Name}', `风险表`: '{table_name}'}}] -> () CALL apoc.create.removeRelProperties(r, $dts) YIELD rel RETURN r"
        return self.execute(CypherStr, parameters={"dts": [iDT.strftime(self._DTFormat) for iDT in dts]})
    # kwargs: 可选参数
    #     id_type: [str], 比如: ['证券', 'A股']
    #     id_field: str, ID 字段
    def writeData(self, table_name, idt, icov, **kwargs):
        IDs = icov.index.tolist()
        IDType = kwargs.get("id_type", [])
        IDField = kwargs.get("id_field", self.IDField)
        if IDType: IDType = f":`{'`:`'.join(IDType)}`"
        self.deleteDateTime(table_name, [idt])
        CypherStr = f"""
            UNWIND range(0, size($ids)-1) AS i
            UNWIND range(0, size($ids)-1) AS j
            MERGE (s1{IDType} {{{IDField}: $ids[i]}})
            MERGE (s2{IDType} {{{IDField}: $ids[j]}})
            MERGE (s1) - [r:`风险数据`] -> (s2)
            SET r.`风险库` = '{self.Name}', r.`风险表` = '{table_name}', r.`{idt.strftime(self._DTFormat)}` = $data[i][j]
        """
        self.execute(CypherStr, parameters={"ids": IDs, "data": icov.values.tolist()})
        self._TableNames.add(table_name)
        return 0

class _FactorRiskTable(FactorRT):
    IDField = Str("ID", arg_type="String", label="ID字段", order=0)
    def __init__(self, name, rdb, sys_args={}, config_file=None, **kwargs):
        if "ID字段" not in sys_args:
            sys_args["ID字段"] = rdb.IDField
        self._FactorInfo = rdb._FactorInfo.loc[name]
        super().__init__(name=name, rdb=rdb, sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def FactorNames(self):
        return sorted(self._FactorInfo.index)
    def getDateTime(self, start_dt=None, end_dt=None):
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (rt) <- [:`属于风险表`] - (f1:`模型成份`:`因子`) - [r:`协方差`] -> (f2:`模型成份`:`因子`) - [:`属于风险表`] -> (rt)
            WITH keys(r) AS kk UNWIND kk AS ik RETURN collect(DISTINCT ik)
        """
        Rslt = self._RiskDB.fetchall(CypherStr)
        if not Rslt: return []
        Rslt = Rslt[0][0]
        DTFormat = self._RiskDB._DTFormat
        if start_dt is not None: start_dt = start_dt.strftime(DTFormat)
        if end_dt is not None: end_dt = end_dt.strftime(DTFormat)
        return [dt.datetime.strptime(iDT, DTFormat) for iDT in Rslt if pd.notnull(iDT) and ((start_dt is None) or (iDT>=start_dt)) and ((end_dt is None) or (iDT<=end_dt))]
    def getID(self, idt=None):
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (sv:`模型成份` {{Name: '特异性风险'}}) - [:`属于风险表`] -> (rt)
            MATCH (s) - [r:`暴露`] - (sv) 
            WHERE s.`{self.IDField}` IS NOT NULL
        """
        if idt is not None:
            DTFormat = self._RiskDB._DTFormat
            CypherStr += f"AND r.`{idt.strftime(DTFormat)}` IS NOT NULL "
        CypherStr += f"RETURN collect(DISTINCT s.`{self.IDField}`)"
        Rslt = self._RiskDB.fetchall(CypherStr)
        if not Rslt: return []
        return sorted(Rslt[0][0])
    def getFactorReturnDateTime(self, start_dt=None, end_dt=None):
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (fr:`模型成份` {{Name: '因子收益率'}}) - [:`属于风险表`] -> (rt)
            MATCH (fr) <- [r:`暴露`] - (f:`模型成份`:`因子`) - [:`属于风险表`] -> (rt)
            WITH keys(r) AS kk UNWIND kk AS ik RETURN collect(DISTINCT ik)
        """
        Rslt = self._RiskDB.fetchall(CypherStr)
        if not Rslt: return []
        Rslt = Rslt[0][0]
        DTFormat = self._RiskDB._DTFormat
        if start_dt is not None: start_dt = start_dt.strftime(DTFormat)
        if end_dt is not None: end_dt = end_dt.strftime(DTFormat)
        return [dt.datetime.strptime(iDT, DTFormat) for iDT in Rslt if pd.notnull(iDT) and ((start_dt is None) or (iDT>=start_dt)) and ((end_dt is None) or (iDT<=end_dt))]
    def getSpecificReturnDateTime(self, start_dt=None, end_dt=None):
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (sr:`模型成份` {{Name: '特异性收益率'}}) - [:`属于风险表`] -> (rt)
            MATCH (s) - [r:`暴露`] -> (sr)
            WHERE s.`{self.IDField}` IS NOT NULL
            WITH keys(r) AS kk UNWIND kk AS ik RETURN collect(DISTINCT ik)
        """
        Rslt = self._RiskDB.fetchall(CypherStr)
        if not Rslt: return []
        Rslt = Rslt[0][0]
        DTFormat = self._RiskDB._DTFormat
        if start_dt is not None: start_dt = start_dt.strftime(DTFormat)
        if end_dt is not None: end_dt = end_dt.strftime(DTFormat)
        return [dt.datetime.strptime(iDT, DTFormat) for iDT in Rslt if pd.notnull(iDT) and ((start_dt is None) or (iDT>=start_dt)) and ((end_dt is None) or (iDT<=end_dt))]
    def __QS_readFactorCov__(self, dts):
        DTFormat = self._RiskDB._DTFormat
        FactorNames = self.FactorNames
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (f1:`模型成份`:`因子`) - [:`属于风险表`] -> (rt)
            MATCH (f2:`模型成份`:`因子`) - [:`属于风险表`] -> (rt)
            MATCH (f1) - [r:`协方差`] -> (f2)
            UNWIND $dts AS iDT
            RETURN iDT, f1.`Name`, f2.`Name`, r[iDT]
        """
        Data = self._RiskDB.fetchall(CypherStr, parameters={"dts": [iDT.strftime(DTFormat) for iDT in dts]})
        if not Data: return Panel(items=dts, major_axis=FactorNames, minor_axis=FactorNames)
        Data = pd.DataFrame(np.array(Data, dtype="O"), columns=["QS_DT", "Factor1", "Factor2", "Value"])
        Data = Data.set_index(["Factor1", "Factor2", "QS_DT"]).iloc[:, 0].unstack().to_panel()
        Data.items = [dt.datetime.strptime(iDT, DTFormat) for iDT in Data.items]
        return Data.loc[dts, FactorNames, FactorNames]
    def __QS_readSpecificRisk__(self, dts, ids=None):
        DTFormat = self._RiskDB._DTFormat
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (sv:`模型成份` {{Name: '特异性风险'}}) - [:`属于风险表`] -> (rt)
            MATCH (s) - [r:`暴露`] - (sv) 
        """
        if ids is None:
            CypherStr += f"WHERE s.`{self.IDField}` IS NOT NULL "
        else:
            CypherStr += f"WHERE s.`{self.IDField}` IN $ids "
        CypherStr += f"UNWIND $dts AS iDT RETURN iDT, s.`{self.IDField}`, r[iDT]"
        Data = self._RiskDB.fetchall(CypherStr, parameters={"dts": [iDT.strftime(DTFormat) for iDT in dts], "ids": ids})
        if not Data: return pd.DataFrame(index=dts, columns=ids)
        Data = pd.DataFrame(np.array(Data, dtype="O"), columns=["QS_DT", "ID", "Value"])
        Data = Data.set_index(["QS_DT", "ID"]).iloc[:, 0].unstack()
        Data.index = [dt.datetime.strptime(iDT, DTFormat) for iDT in Data.index]
        if ids is None:
            return Data.loc[dts]
        else:
            return Data.loc[dts, ids]
    def __QS_readFactorData__(self, dts, ids=None):
        DTFormat = self._RiskDB._DTFormat
        FactorNames = self.FactorNames
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (f:`模型成份`:`因子`) - [:`属于风险表`] -> (rt)
            MATCH (s) - [r:`暴露`] -> (f) 
        """
        if ids is None:
            CypherStr += f"WHERE s.`{self.IDField}` IS NOT NULL "
        else:
            CypherStr += f"WHERE s.`{self.IDField}` IN $ids "
        CypherStr += f"UNWIND $dts AS iDT RETURN iDT, f.`Name`, s.`{self.IDField}`, r[iDT]"
        Data = self._RiskDB.fetchall(CypherStr, parameters={"dts": [iDT.strftime(DTFormat) for iDT in dts], "ids": ids})
        if not Data: return Panel(items=FactorNames, major_axis=dts, minor_axis=ids)
        Data = pd.DataFrame(np.array(Data, dtype="O"), columns=["QS_DT", "Factor", "ID", "Value"])
        Data = Data.set_index(["QS_DT", "ID", "Factor"]).iloc[:, 0].unstack().to_panel()
        Data.major_axis = [dt.datetime.strptime(iDT, DTFormat) for iDT in Data.major_axis]
        if ids is None:
            return Data.loc[FactorNames, dts]
        else:
            return Data.loc[FactorNames, dts, ids]
    def readFactorReturn(self, dts):
        DTFormat = self._RiskDB._DTFormat
        FactorNames = self.FactorNames
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (fr:`模型成份` {{`Name`: '因子收益率'}}) - [:`属于风险表`] -> (rt)
            MATCH (rt) <- [:`属于风险表`] - (f:`模型成份`:`因子`) - [r:`暴露`] -> (fr)
            UNWIND $dts AS iDT
            RETURN iDT, f.`Name`, r[iDT]
        """
        Data = self._RiskDB.fetchall(CypherStr, parameters={"dts": [iDT.strftime(DTFormat) for iDT in dts]})
        if not Data: return pd.DataFrame(index=dts, columns=FactorNames)
        Data = pd.DataFrame(np.array(Data, dtype="O"), columns=["QS_DT", "Factor", "Value"])
        Data = Data.set_index(["QS_DT", "Factor"]).iloc[:, 0].unstack()
        Data.index = [dt.datetime.strptime(iDT, DTFormat) for iDT in Data.index]
        return Data.loc[dts, FactorNames]
    def readSpecificReturn(self, dts, ids=None):
        DTFormat = self._RiskDB._DTFormat
        CypherStr = f"""
            MATCH (rt:`风险表` {{Name: '{self._Name}'}}) - [:`属于风险库`] -> {self._RiskDB._Node}
            MATCH (sr:`模型成份` {{Name: '特异性收益率'}}) - [:`属于风险表`] -> (rt)
            MATCH (s) - [r:`暴露`] - (sr) 
        """
        if ids is None:
            CypherStr += f"WHERE s.`{self.IDField}` IS NOT NULL "
        else:
            CypherStr += f"WHERE s.`{self.IDField}` IN $ids "
        CypherStr += f"UNWIND $dts AS iDT RETURN iDT, s.`{self.IDField}`, r[iDT]"
        Data = self._RiskDB.fetchall(CypherStr, parameters={"dts": [iDT.strftime(DTFormat) for iDT in dts], "ids": ids})
        if not Data: return pd.DataFrame(index=dts, columns=ids)
        Data = pd.DataFrame(np.array(Data, dtype="O"), columns=["QS_DT", "ID", "Value"])
        Data = Data.set_index(["QS_DT", "ID"]).iloc[:, 0].unstack()
        Data.index = [dt.datetime.strptime(iDT, DTFormat) for iDT in Data.index]
        if ids is None:
            return Data.loc[dts]
        else:
            return Data.loc[dts, ids]
    def __QS_readCov__(self, dts, ids=None):
        FactorCov = self.__QS_readFactorCov__(dts)
        FactorData = self.__QS_readFactorData__(dts, ids=ids)
        SpecificRisk = self.__QS_readSpecificRisk__(dts, ids=ids)
        Data = {}
        for i, iDT in enumerate(dts):
            iFactorCov = FactorCov.loc[iDT]
            iSpecificRisk = SpecificRisk.loc[iDT]
            iFactorData = FactorData.loc[:, iDT]
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
        return Panel(Data, items=dts)

class Neo4jFRDB(QSNeo4jObject, FactorRDB):
    """基于 Neo4j 的多因子风险数据库"""
    Name = Str("Neo4jFRDB", arg_type="String", label="名称", order=-100)
    IDField = Str("ID", arg_type="String", label="ID字段", order=100)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableInfo = pd.DataFrame()# DataFrame(index=[表名], columns=[Description"])
        self._FactorInfo = pd.DataFrame()# DataFrame(index=[(表名,因子名)], columns=["DataType", "Description"])
        self._DTFormat = "%Y-%m-%d"
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"Neo4jFRDBConfig.json" if config_file is None else config_file), **kwargs)
    def connect(self):
        super().connect()
        Class = str(self.__class__)
        Class = Class[Class.index("'")+1:]
        Class = Class[:Class.index("'")]
        self._Node = f"(rdb:`风险库`:`{self.__class__.__name__}` {{`Name`: '{self.Name}', `_Class`: '{Class}'}})"
        Args = self.Args
        Args.pop("用户名")
        Args.pop("密码")
        with self._Connection.session(database=self.DBName) as Session:
            with Session.begin_transaction() as tx:
                CypherStr = f"MERGE {self._Node}"
                iCypherStr, Parameters = writeArgs(Args, arg_name=None, tx=None, parent_var="rdb")
                CypherStr += " "+iCypherStr
                tx.run(CypherStr, parameters=Parameters)
                CypherStr = f"MATCH (rt:`风险表`) - [:`属于风险库`] -> {self._Node} RETURN DISTINCT rt.Name AS TableName, rt.description AS Description"
                self._TableInfo = tx.run(CypherStr).values()
                CypherStr = f"MATCH (f:`模型成份`:`因子`) - [:`属于风险表`] -> (rt:`风险表`) - [:`属于风险库`] -> {self._Node} RETURN DISTINCT f.Name AS FactorName, rt.Name AS TableName, f.DataType AS DataType, f.description AS Description"
                self._FactorInfo = tx.run(CypherStr).values()
        self._TableInfo = pd.DataFrame(self._TableInfo, columns=["TableName", "Description"]).set_index(["TableName"])
        self._FactorInfo = pd.DataFrame(self._FactorInfo, columns=["FactorName", "TableName", "DataType", "Description"]).set_index(["TableName", "FactorName"])
        return 0
    @property
    def TableNames(self):
        return sorted(self._TableInfo.index)
    def getTable(self, table_name, args={}):
        return _FactorRiskTable(table_name, self)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableInfo.index:
            Msg = ("风险库 '%s' 调用方法 renameTable 错误: 不存在风险表 '%s'!" % (self.Name, old_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableInfo.index):
            Msg = ("风险库 '%s' 调用方法 renameTable 错误: 新风险表名 '%s' 已经存在于库中!" % (self.Name, new_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        CypherStr = f"MATCH (rt:`风险表` {{`Name`: '{old_table_name}'}}) - [:`属于风险库`] -> {self._Node} SET rt.`Name` ='{new_table_name}'"
        self.execute(CypherStr)
        self._TableInfo = self._TableInfo.rename(index={old_table_name: new_table_name})
        self._FactorInfo = self._FactorInfo.rename(index={old_table_name: new_table_name}, level=0)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableInfo.index: return 0
        CypherStr = f"MATCH (f:`因子`) - [:`属于风险表`] -> (rt:`风险表` {{`Name`: '{table_name}'}}) - [:`属于风险库`] -> {self._Node} DETACH DELETE f, rt"
        self.execute(CypherStr)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._TableInfo = self._TableInfo.loc[TableNames]
        self._FactorInfo = self._FactorInfo.loc[TableNames]
        return 0
    def deleteDateTime(self, table_name, dts):
        DTs = [iDT.strftime(self._DTFormat) for iDT in dts]
        with self.session() as Session:
            with Session.begin_transaction() as tx:
                CypherStr = f"MATCH () - [r:`暴露`] -> (f:`模型成份`:`因子`) - [:`属于风险表`] -> (rt:`风险表` {{`Name`: '{table_name}'}}) - [:`属于风险库`] -> {self._Node} CALL apoc.create.removeRelProperties(r, $dts) YIELD rel RETURN r"
                tx.run(CypherStr, parameters={"dts": DTs})
                CypherStr = f"MATCH (rt:`风险表` {{`Name`: '{table_name}'}}) - [:`属于风险库`] -> {self._Node} MATCH (rt) <- [:`属于风险表`] - (f1:`模型成份`:`因子`) - [r:`协方差`] -> (f2:`模型成份`:`因子`) - [:`属于风险表`] -> (rt) CALL apoc.create.removeRelProperties(r, $dts) YIELD rel RETURN r"
                tx.run(CypherStr, parameters={"dts": DTs})
                CypherStr = f"MATCH () - [r:`暴露`] -> (sv:`模型成份` {{Name: '特异性风险'}}) - [:`属于风险表`] -> (rt:`风险表` {{`Name`: '{table_name}'}}) - [:`属于风险库`] -> {self._Node} CALL apoc.create.removeRelProperties(r, $dts) YIELD rel RETURN r"
                tx.run(CypherStr, parameters={"dts": DTs})
                CypherStr = f"MATCH () - [r:`暴露`] -> (sr:`模型成份` {{Name: '特异性收益率'}}) - [:`属于风险表`] -> (rt:`风险表` {{`Name`: '{table_name}'}}) - [:`属于风险库`] -> {self._Node} CALL apoc.create.removeRelProperties(r, $dts) YIELD rel RETURN r"
                tx.run(CypherStr, parameters={"dts": DTs})
                CypherStr = f"MATCH (rt:`风险表` {{`Name`: '{table_name}'}}) - [:`属于风险库`] -> {self._Node} MATCH (rt) <- [:`属于风险表`] - (f:`模型成份`:`因子`) - [r:`暴露`] -> (fr:`模型成份` {{Name: '因子收益率'}}) - [:`属于风险表`] -> (rt) CALL apoc.create.removeRelProperties(r, $dts) YIELD rel RETURN r"
                tx.run(CypherStr, parameters={"dts": DTs})
        return 0
    # kwargs: 可选参数
    #     id_type: [str], 比如: ['证券', 'A股']
    #     id_field: str, ID 字段
    def writeData(self, table_name, idt, factor_data=None, factor_cov=None, specific_risk=None, factor_ret=None, specific_ret=None, **kwargs):
        iDT = idt.strftime(self._DTFormat)
        IDType = kwargs.get("id_type", [])
        IDField = kwargs.get("id_field", self.IDField)
        if IDType: IDType = f":`{'`:`'.join(IDType)}`"
        if factor_data is not None:
            CypherStr = f"""
                MATCH {self._Node}
                MERGE (rt:`风险表` {{Name: '{table_name}'}}) - [:`属于风险库`] -> (rdb)
                WITH rt
                UNWIND range(0, size($factors)-1) AS j
                MERGE (f:`模型成份`:`因子` {{Name: $factors[j]}}) - [:`属于风险表`] -> (rt)
                WITH f, j
                UNWIND range(0, size($ids)-1) AS i
                MERGE (s{IDType} {{{IDField}: $ids[i]}})
                MERGE (s) - [r:`暴露`] -> (f)
                SET r.`{iDT}` = $data[i][j]
            """
            factor_data = factor_data.astype("O").where(pd.notnull(factor_data), None)
            self.execute(CypherStr, parameters={"ids": factor_data.index.tolist(), "data": factor_data.values.tolist(), "factors": factor_data.columns.tolist()})
        if factor_cov is not None:
            CypherStr = f"""
                MATCH {self._Node}
                MERGE (rt:`风险表` {{Name: '{table_name}'}}) - [:`属于风险库`] -> (rdb)
                WITH rt
                UNWIND range(0, size($factors)-1) AS i
                UNWIND range(0, size($factors)-1) AS j
                MERGE (f1:`模型成份`:`因子` {{Name: $factors[i]}}) - [:`属于风险表`] -> (rt)
                MERGE (f2:`模型成份`:`因子` {{Name: $factors[j]}}) - [:`属于风险表`] -> (rt)
                MERGE (f1) - [r:`协方差`] -> (f2)
                SET r.`{idt.strftime(self._DTFormat)}` = $data[i][j]
            """
            self.execute(CypherStr, parameters={"data": factor_cov.values.tolist(), "factors": factor_data.columns.tolist()})
        if specific_risk is not None:
            CypherStr = f"""
                MATCH {self._Node}
                MERGE (rt:`风险表` {{Name: '{table_name}'}}) - [:`属于风险库`] -> (rdb)
                MERGE (sv:`模型成份` {{Name: '特异性风险'}}) - [:`属于风险表`] -> (rt)
                WITH sv
                UNWIND range(0, size($ids)-1) AS i
                MERGE (s{IDType} {{{IDField}: $ids[i]}})
                MERGE (s) - [r:`暴露`] -> (sv)
                SET r.`{iDT}` = $data[i]
            """
            specific_risk = specific_risk.astype("O").where(pd.notnull(specific_risk), None)
            self.execute(CypherStr, parameters={"ids": specific_risk.index.tolist(), "data": specific_risk.tolist()})
        if factor_ret is not None:
            CypherStr = f"""
                MATCH {self._Node}
                MERGE (rt:`风险表` {{Name: '{table_name}'}}) - [:`属于风险库`] -> (rdb)
                MERGE (fr:`模型成份` {{Name: '因子收益率'}}) - [:`属于风险表`] -> (rt)
                WITH fr, rt
                UNWIND range(0, size($factors)-1) AS i
                MERGE (f:`模型成份`:`因子` {{Name: $factors[i]}}) - [:`属于风险表`] -> (rt)
                MERGE (f) - [r:`暴露`] -> (fr)
                SET r.`{iDT}` = $data[i]
            """
            self.execute(CypherStr, parameters={"factors": factor_ret.index.tolist(), "data": factor_ret.tolist()})
        if specific_ret is not None:
            CypherStr = f"""
                MATCH {self._Node}
                MERGE (rt:`风险表` {{Name: '{table_name}'}}) - [:`属于风险库`] -> (rdb)
                MERGE (sr:`模型成份` {{Name: '特异性收益率'}}) - [:`属于风险表`] -> (rt)
                WITH sr
                UNWIND range(0, size($ids)-1) AS i
                MERGE (s{IDType} {{{IDField}: $ids[i]}})
                MERGE (s) - [r:`暴露`] -> (sr)
                SET r.`{iDT}` = $data[i]
            """
            specific_ret = specific_ret.astype("O").where(pd.notnull(specific_ret), None)
            self.execute(CypherStr, parameters={"ids": specific_ret.index.tolist(), "data": specific_ret.tolist()})
        return 0