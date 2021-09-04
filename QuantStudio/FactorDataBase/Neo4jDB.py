# coding=utf-8
"""基于 Neo4j 数据库的因子库(TODO)"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str, Dict, Password, Range, Enum

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable

def _identifyDataType(factor_data, data_type=None):
    if (data_type is None) or (data_type=="double"):
        try:
            factor_data = factor_data.astype(float)
        except:
            data_type = "object"
        else:
            data_type = "double"
    return (factor_data, data_type)

class _FactorTable(FactorTable):
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._TableInfo = fdb._TableInfo.loc[name]
        self._FactorInfo = fdb._FactorInfo.loc[name]
        self._QS_IgnoredGroupArgs = ("遍历模式", )
        self._DTFormat = "%Y-%m-%d"
        self._DTFormat_WithTime = "%Y-%m-%d %H:%M:%S"
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in iFactor.ArgNames if iArgName not in self._QS_IgnoredGroupArgs])
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {"FactorNames":[iFactor.Name], 
                                                       "RawFactorNames":{iFactor._NameInFT}, 
                                                       "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                       "args":iFactor.Args.copy()}
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
                if "回溯天数" in ConditionGroup[iConditions]["args"]:
                    ConditionGroup[iConditions]["args"]["回溯天数"] = max(ConditionGroup[iConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    def getMetaData(self, key=None, args={}):
        if key is None:
            return self._TableInfo.copy()
        else:
            return self._TableInfo.get(key, None)
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            return self._FactorInfo["DataType"].loc[factor_names]
        elif key=="Description":
            return self._FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args),
                                 "Description":self.getFactorMetaData(factor_names, key="Description", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    @property
    def FactorNames(self):
        return self._FactorInfo[pd.notnull(self._FactorInfo["DataType"])].index.tolist()
    def getID(self, ifactor_name=None, idt=None, args={}):
        CypherStr = f"MATCH (f:`因子`) - [:`属于`] -> (ft:`因子表` {{Name: '{self.Name}'}}) - [:`属于`] -> {self._FactorDB._Node} "
        if ifactor_name is not None:
            CypherStr += f"WHERE f.Name = '{ifactor_name}' "
        CypherStr += "MATCH (s) - [r:`暴露`] -> (f) "
        if idt is not None:
            CypherStr += f"WHERE r.`{idt.strftime(self._DTFormat)}` IS NOT NULL "
        CypherStr += "RETURN s.ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(CypherStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):# TODO
        CypherStr = f"MATCH (f:`因子`) - [:`属于`] -> (ft:`因子表` {{Name: '{self.Name}'}}) - [:`属于`] -> {self._FactorDB._Node} "
        if ifactor_name is not None:
            CypherStr += f"WHERE f.Name = '{ifactor_name}' "
        CypherStr += "MATCH (s) - [r:`暴露`] -> (f) "
        if iid is not None:
            CypherStr += f"WHERE s.ID = '{iid}' "
        CypherStr += "RETURN DISTINCT keys(r)"
        Rslt = sorted(set(sum(self._FactorDB.fetchall(CypherStr)[0], [])))
        return [dt.datetime.strptime(iDT, self._DTFormat) for iDT in Rslt if pd.notnull(iDT)]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        CypherStr = f"""
            MATCH (f:`因子`) - [:`属于`] -> (ft:`因子表` {{Name: '{self.Name}'}}) - [:`属于`] -> {self._FactorDB._Node}
            WHERE f.Name IN $factors
            MATCH (s) - [r:`暴露`] -> (f)
            WHERE s.ID IN $ids
            WITH s, f, r
            UNWIND $dts AS iDT
            RETURN s.ID AS ID, iDT AS QS_DT, f.Name AS FactorName, r[iDT] AS Value
            ORDER BY ID, QS_DT, FactorName
        """
        DTs = [iDT.strftime(self._DTFormat) for iDT in dts]
        with self._FactorDB.session() as Session:
            with Session.begin_transaction() as tx:
                RawData = tx.run(CypherStr, parameters={"factors": factor_names, "ids": ids, "dts": DTs}).values()
        RawData = pd.DataFrame(RawData, columns=["ID", "QS_DT", "FactorName", "Value"])
        #RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(x, self._DTFormat) if pd.notnull(x) else pd.NaT)
        RawData["QS_DT"] = pd.to_datetime(RawData["QS_DT"])
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        if ids is None: ids = sorted(raw_data["ID"].unique())
        raw_data = raw_data.set_index(["QS_DT", "ID", "FactorName"]).iloc[:, 0]
        raw_data = raw_data.unstack()
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        DataType = DataType[~DataType.index.duplicated()]
        Data = {}
        for iFactorName in factor_names:
            if iFactorName in raw_data:
                iRawData = raw_data[iFactorName].unstack()
                if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
                Data[iFactorName] = iRawData
        if not Data: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return pd.Panel(Data).loc[factor_names]

class Neo4jDB(WritableFactorDB):
    """Neo4jDB"""
    Name = Str("Neo4jDB")
    DBName = Str("neo4j", arg_type="String", label="数据库名", order=0)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=7687, arg_type="Integer", label="端口", order=2)
    User = Str("neo4j", arg_type="String", label="用户名", order=3)
    Pwd = Password("", arg_type="String", label="密码", order=4)
    Connector = Enum("default", "neo4j", arg_type="SingleOption", label="连接器", order=5)
    InnerID = Str("Test", arg_type="String", label="内部ID", order=6)
    FTArgs = Dict(label="因子表参数", arg_type="Dict", order=101)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Connection = None# 连接对象
        self._Connector = None# 实际使用的数据库链接器
        self._PID = None# 保存数据库连接创建时的进程号
        self._TableInfo = pd.DataFrame()# DataFrame(index=[表名], columns=[Description"])
        self._FactorInfo = pd.DataFrame()# DataFrame(index=[(表名,因子名)], columns=["DataType", "Description"])
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"Neo4jDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "Neo4jDB"
        return
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
        self._connect()
        self._Node = f"(fdb:`因子库`:`Neo4jDB` {{内部ID: '{self.InnerID}'}})"
        with self._Connection.session(database=self.DBName) as Session:
            with Session.begin_transaction() as tx:
                CypherStr = f"MERGE {self._Node} ON CREATE SET fdb.Name='{self.Name}', fdb.`数据库名` = '{self.DBName}', fdb.`IP地址` = '{self.IPAddr}', fdb.`端口` = {self.Port}"
                CypherStr += f" ON MATCH SET fdb.Name='{self.Name}', fdb.`数据库名` = '{self.DBName}', fdb.`IP地址` = '{self.IPAddr}', fdb.`端口` = {self.Port}"
                tx.run(CypherStr)
                CypherStr = f"MATCH (ft:`因子表`) - [:`属于`] -> {self._Node} RETURN DISTINCT ft.Name AS TableName, ft.description AS Description"
                self._TableInfo = tx.run(CypherStr).values()
                CypherStr = f"MATCH (f:`因子`) - [:`属于`] -> (ft:`因子表`) - [:`属于`] -> {self._Node} RETURN DISTINCT f.Name AS FactorName, ft.Name AS TableName, f.DataType AS DataType, f.description AS Description"
                self._FactorInfo = tx.run(CypherStr).values()
        self._TableInfo = pd.DataFrame(self._TableInfo, columns=["TableName", "Description"]).set_index(["TableName"])
        self._FactorInfo = pd.DataFrame(self._FactorInfo, columns=["FactorName", "TableName", "DataType", "Description"]).set_index(["TableName", "FactorName"])
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
    def session(self):
        if self._Connection is None:
            Msg = ("'%s' 获取 cursor 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:# 连接断开后重连
            Session = self._Connection.session(database=self.DBName)
        except:
            self._connect()
            Session = self._Connection.session(database=self.DBName)
        return Session
    def fetchall(self, cypher_str):
        with self.session() as Session:
            with Session.begin_transaction() as tx:
                return tx.run(cypher_str).values()
    def execute(self, cypher_str):
        if self._Connection is None:
            Msg = ("'%s' 执行 SQL 命令失败: 数据库尚未连接!" % (self.Name,))
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
                tx.run(cypher_str)
        return 0
    # ----------------------------因子表操作-----------------------------
    @property
    def TableNames(self):
        return sorted(self._TableInfo.index)
    def getTable(self, table_name, args={}):
        if table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不存在因子表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        Args = self.FTArgs.copy()
        Args.update(args)
        return _FactorTable(name=table_name, fdb=self, sys_args=Args, logger=self._QS_Logger)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 不存在因子表 '%s'!" % (self.Name, old_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableInfo.index):
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 新因子表名 '%s' 已经存在于库中!" % (self.Name, new_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        CypherStr = f"MATCH (ft:`因子表` {{`Name`: '{old_table_name}'}}) - [:`属于`] -> {self._Node} SET ft.`Name` ='{new_table_name}'"
        self.execute(CypherStr)
        self._TableInfo = self._TableInfo.rename(index={old_table_name: new_table_name})
        self._FactorInfo = self._FactorInfo.rename(index={old_table_name: new_table_name}, level=0)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableInfo.index: return 0
        CypherStr = f"MATCH (f:`因子`) - [:`属于`] -> (ft:`因子表` {{`Name`: '{table_name}'}}) - [:属于] -> {self._Node} DETACH DELETE f, ft"
        self.execute(CypherStr)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._TableInfo = self._TableInfo.loc[TableNames]
        self._FactorInfo = self._FactorInfo.loc[TableNames]
        return 0
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name not in self._FactorInfo.loc[table_name].index:
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 因子表 '%s' 中不存在因子 '%s'!" % (self.Name, table_name, old_factor_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_factor_name!=old_factor_name) and (new_factor_name in self._FactorInfo.loc[table_name].index):
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 新因子名 '%s' 已经存在于因子表 '%s' 中!" % (self.Name, new_factor_name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        CypherStr = f"MATCH (:`因子` {{`Name`: '{old_factor_name}'}}) - [:`属于`] -> (ft:`因子表` {{`Name`: '{table_name}'}}) - [:`属于`] -> {self._Node} SET f.`Name` = '{new_factor_name}'"
        self.execute(CypherStr)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].rename(index={old_factor_name: new_factor_name}, level=1))
        return 0
    def deleteFactor(self, table_name, factor_names):
        if (not factor_names) or (table_name not in self._TableInfo.index): return 0
        FactorIndex = self._FactorInfo.loc[table_name].index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        CypherStr = f"MATCH (f:`因子`) - [:`属于`] -> (:`因子表` {{`Name`: '{table_name}'}}) - [:属于] -> {self._Node} "
        CypherStr += "WHERE "+genSQLInCondition("f.Name", factor_names, is_str=True)+" "
        CypherStr += "DETACH DELETE f"
        self.execute(CypherStr)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].loc[FactorIndex])
        return 0
    # ----------------------------数据操作---------------------------------
    # 附加参数: id_type: str, 比如: A 股, 公募基金
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        FactorNames, DTs, IDs = data.items.tolist(), data.major_axis.tolist(), data.minor_axis.tolist()
        DataType = data_type.copy()
        for i, iFactorName in enumerate(FactorNames):
            data[iFactorName], DataType[iFactorName] = _identifyDataType(data.iloc[i], data_type.get(iFactorName, None))
        InitCypherStr = f"""
            MATCH {self._Node}
            MERGE (ft:`因子表` {{Name: '{table_name}'}})
            MERGE (ft) - [:`属于`] -> (fdb)
            WITH ft
            UNWIND $factors AS iFactor
            MERGE (f:`因子` {{Name: iFactor, DataType: $data_type[iFactor]}})
            MERGE (f) - [:`属于`] -> (ft)
        """
        IDType = kwargs.get("id_type", "")
        if IDType: IDType = f":`{IDType}`"
        WriteCypherStr = f"""
            MATCH (f:`因子` {{Name: $ifactor}}) - [:`属于`] -> (ft:`因子表` {{Name: '{table_name}'}}) - [:`属于`] -> {self._Node}
            UNWIND range(0, size($ids)-1) AS i
            MERGE (s{IDType} {{ID: $ids[i]}})
            MERGE (s) - [r:`暴露`] -> (f)
            ON CREATE SET r = $data[i]
            ON MATCH SET r += $data[i]
        """
        if (if_exists!="update") and (table_name in self._TableInfo.index):
            OldFactorNames = sorted(self._FactorInfo.loc[table_name].index.intersection(FactorNames))
            OldData = self.getTable(table_name).readData(factor_names=OldFactorNames, ids=IDs, dts=DTs)
            if if_exists=="append":
                for iFactorName in OldFactorNames:
                    data[iFactorName] = OldData[iFactorName].where(pd.notnull(OldData[iFactorName]), data[iFactorName])
            elif if_exists=="update_notnull":
                for iFactorName in OldFactorNames:
                    data[iFactorName] = data[iFactorName].where(pd.notnull(data[iFactorName]), OldData[iFactorName])
            else:
                Msg = ("因子库 '%s' 调用方法 writeData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
        data.major_axis = data.major_axis.strftime("%Y-%m-%d")
        with self.session() as Session:
            with Session.begin_transaction() as tx:
                tx.run(InitCypherStr, parameters={"factors": FactorNames, "data_type": DataType})
                for i, iFactor in enumerate(data.items):
                    iData = data.iloc[i]
                    iData = iData.astype("O").where(pd.notnull(iData), None)
                    iData = iData.T.to_dict(orient="records")
                    #iData = iData.apply(lambda s: s.to_dict(), axis=0, raw=False).tolist()
                    tx.run(WriteCypherStr, parameters={"data": iData, "ids": IDs, "ifactor": iFactor})
        if table_name not in self._TableInfo.index:
            self._TableInfo.loc[table_name] = None
        NewFactorInfo = pd.DataFrame(DataType, index=["DataType"], columns=pd.Index(sorted(DataType.keys()), name="FactorName")).T.reset_index()
        NewFactorInfo["TableName"] = table_name
        self._FactorInfo = self._FactorInfo.append(NewFactorInfo.set_index(["TableName", "FactorName"])).sort_index()
        data.major_axis = DTs
        return 0

if __name__=="__main__":
    iDB = Neo4jDB(sys_args={
        "数据库名": "neo4j",
        "IP地址": "127.0.0.1",
        "端口": 11003,
        "用户名": "neo4j",
        "密码": "shuntai11",
        "连接器": "default",
        "内部ID": "Test"
    })
    iDB.connect()
    print(iDB)