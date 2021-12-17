# coding=utf-8
import os
import urllib
import pickle
import importlib
import concurrent.futures

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, Either, Int, Bool

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.FactorDataBase.FactorOperation import DerivativeFactor
from QuantStudio.FactorDataBase.FactorDB import CustomFT
from QuantStudio.Tools.AuxiliaryFun import distributeEqual
from QuantStudio.Tools.FileFun import genAvailableFile

class QSNeo4jObject(__QS_Object__):
    """基于 Neo4j 的对象"""
    Name = Str("QSNeo4jObject")
    DBName = Str("neo4j", arg_type="String", label="数据库名", order=0)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=7687, arg_type="Integer", label="端口", order=2)
    User = Str("neo4j", arg_type="String", label="用户名", order=3)
    Pwd = Password("", arg_type="String", label="密码", order=4)
    Connector = Enum("default", "neo4j", arg_type="SingleOption", label="连接器", order=5)
    CSVImportPath = Either(None, Str("file:///"), arg_type="String", label="CSV导入地址", order=6)
    CSVExportPath = Either(None, Str(), arg_type="String", label="CSV导出地址", order=7)
    CSVSep = Str(",", arg_type="String", label="CSV分隔符", order=8)
    ClearCSV = Bool(False, arg_type="Bool", label="清除CSV", order=9)
    PeriodicSize = Int(-1, arg_type="Integer", label="定期数量", order=10)
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
    @property
    def Connection(self):
        if self._Connection is not None:
            if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        return self._Connection
    def _getConnection(self):
        if (self.Connector=="neo4j") or (self.Connector=="default"):
            try:
                import neo4j
                Conn = neo4j.GraphDatabase.driver(f"neo4j://{self.IPAddr}:{self.Port}", auth=(self.User, self.Pwd), max_connection_lifetime=3600*8)
            except Exception as e:
                Msg = ("'%s' 尝试使用 neo4j 连接(%s@%s:%d)数据库失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                Connector = "neo4j"
        return Conn, Connector
    def _connect(self):
        self._Connection, self._Connector = self._getConnection()
        self._PID = os.getpid()
        return 0
    def connect(self):
        return self._connect()
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
            Msg = ("'%s' 获取 session 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:# 连接断开后重连
            Session = self._Connection.session(database=self.DBName)
        except:
            self._connect()
            Session = self._Connection.session(database=self.DBName)
        return Session
    def fetchall(self, cypher_str, parameters=None):
        with self.session() as Session:
            return Session.run(cypher_str, parameters=parameters).values()
    def execute(self, cypher_str, parameters=None):
        if self._Connection is None:
            Msg = ("'%s' 执行 Cypher 命令失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:
            Session = self._Connection.session(database=self.DBName)
        except:
            self._connect()
            Session = self._Connection.session(database=self.DBName)
        with Session:
            Session.run(cypher_str, parameters=parameters)
        return 0
    # 写入实体标签
    # new_labels: [新的实体标签]
    # ids: [实体 ID]
    # entity_labels: [实体标签]
    # entity_id: 实体 ID 字段
    # excluded_labels: [(需要排除的标签,)]
    # kwargs: 可选参数
    #     auto_exclude: 自动排除标签, 默认值 False
    def writeEntityLabel(self, new_labels, ids, entity_labels, entity_id, excluded_labels=(), args={}, **kwargs):
        LabelStr = "`:`".join(entity_labels)
        if kwargs.get("auto_exclude", False):
            CypherStr = f"""
                MATCH (n:`{LabelStr}`)
                WHERE n.`{entity_id}` IN $ids
                WITH DISTINCT labels(n) AS Labels
                UNWIND Labels AS iLabel
                RETURN DISTINCT iLabel
            """
            excluded_labels = set(tuple(iLabels) for iLabels in excluded_labels).union(tuple(iLabels) for iLabels in self.fetchall(CypherStr, parameters={"ids": ids}) if (iLabels[0] not in entity_labels) and (iLabels[0] not in new_labels))
        NewLabelStr = "`:`".join(new_labels)
        CypherStr = f"""
            MATCH (n:`{LabelStr}`)
            WHERE n.`{entity_id}` IN $ids
        """
        if excluded_labels:
            CypherStr += f"AND (NOT n:`{'`) AND (NOT n:`'.join(':'.join(iLabels) for iLabels in excluded_labels)}`) "
        PeriodicSize = args.get("定期数量", self.PeriodicSize)
        if PeriodicSize>0:
            CypherStr = f"""CALL apoc.periodic.iterate("{CypherStr}RETURN n", "SET n:`{NewLabelStr}`", {{batchSize: {PeriodicSize}, parallel: true, params: {{`ids`: $ids}}}})"""
        else:
            CypherStr += f"SET n:`{NewLabelStr}`"
        return self.execute(CypherStr, parameters={"ids": ids})
    # 写入实体数据
    # data: DataFrame(index=[实体ID], columns=[实体属性])
    # entity_labels: [实体标签]
    # entity_id: 实体 ID 字段
    # kwargs: 可选参数
    #     thread_num: 写入线程数量
    def _writeEntityData(self, data, entity_labels, entity_id, if_exists="update", args={}, **kwargs):
        LabelStr = "`:`".join(entity_labels)
        if if_exists in ("append", "update_notnull"):
            IDs, Fields = data.index.tolist(), data.columns.tolist()
            CypherStr = f"""
                MATCH (n:`{LabelStr}`)
                WHERE n.`{entity_id}` IN $ids
                RETURN n.`{entity_id}`, n.`{'`, n.`'.join(Fields)}`
            """
            OldData = self.fetchall(CypherStr, parameters={"ids": IDs})
            if not OldData: OldData = pd.DataFrame(index=IDs, columns=Fields)
            else: OldData = pd.DataFrame(np.array(OldData, dtype="O"), columns=["ID"]+Fields).set_index(["ID"]).loc[IDs]
            if if_exists=="append":
                data = OldData.where(pd.notnull(OldData), data)
            else:
                data = data.where(pd.notnull(data), OldData)
        #data[entity_id] = data.index
        data.insert(0, entity_id, data.index)
        data = data.astype("O").where(pd.notnull(data), None)
        CypherStr1 = "UNWIND $data AS iData"
        if if_exists in ("replace", "replace_all", "create"):
            CypherStr2 = f"CREATE (n:`{LabelStr}`) SET n = iData"
        else:
            CypherStr2 = f"""
                MERGE (n:`{LabelStr}` {{`{entity_id}`: iData['{entity_id}']}})
                ON CREATE SET n = iData
                ON MATCH SET n += iData
            """
        PeriodicSize = args.get("定期数量", self.PeriodicSize)
        if PeriodicSize>0:
            CypherStr = f"""CALL apoc.periodic.iterate("{CypherStr1} RETURN iData", "{CypherStr2}", {{`batchSize`: {PeriodicSize}, `parallel`: true, `params`: {{`data`: $data}}}})"""
        else:
            CypherStr = f"""{CypherStr1} {CypherStr2}"""
        return self.execute(CypherStr, parameters={"data": data.to_dict(orient="records")})
    def _writeEntityData_LoadCSV(self, data, entity_labels, entity_id, if_exists="update", args={}, **kwargs):
        LabelStr = "`:`".join(entity_labels)
        CSVExportPath = args.get("CSV导出地址", self.CSVExportPath)
        if data.shape[0]>0:
            if if_exists in ("append", "update_notnull"):
                IDs, Fields = data.index.tolist(), data.columns.tolist()
                CypherStr = f"""
                    MATCH (n:`{LabelStr}`)
                    WHERE n.`{entity_id}` IN $ids
                    RETURN n.`{entity_id}`, n.`{'`, n.`'.join(Fields)}`
                """
                OldData = self.fetchall(CypherStr, parameters={"ids": IDs})
                if not OldData: OldData = pd.DataFrame(index=IDs, columns=Fields)
                else: OldData = pd.DataFrame(np.array(OldData, dtype="O"), columns=["ID"]+Fields).set_index(["ID"]).loc[IDs]
                if if_exists=="append":
                    data = OldData.where(pd.notnull(OldData), data)
                else:
                    data = data.where(pd.notnull(data), OldData)
            #data[entity_id] = data.index
            data.insert(0, entity_id, data.index)
            data = data.astype("O").where(pd.notnull(data), None)
        if CSVExportPath is not None:
            if os.path.isdir(CSVExportPath):
                ExportPath = genAvailableFile("QSNeo4jEntityData", target_dir=CSVExportPath, suffix="csv")
            else:
                ExportPath = CSVExportPath
            data.to_csv(ExportPath, index=False, header=True, sep=args.get("CSV分隔符", self.CSVSep))
            _, CSVFile = os.path.split(ExportPath)
            ImportPath = urllib.parse.urljoin(args.get("CSV导入地址", self.CSVImportPath), CSVFile)
        else:
            ImportPath = args.get("CSV导入地址", self.CSVImportPath)
        PeriodicSize = args.get("定期数量", self.PeriodicSize)
        if PeriodicSize>0:
            CypherStr = f"""USING PERIODIC COMMIT {PeriodicSize} LOAD CSV WITH HEADERS FROM "{ImportPath}" AS line FIELDTERMINATOR '{args.get("CSV分隔符", self.CSVSep)}'"""
        else:
            CypherStr = f"""LOAD CSV WITH HEADERS FROM "{ImportPath}" AS line FIELDTERMINATOR '{args.get("CSV分隔符", self.CSVSep)}'"""
        if if_exists in ("replace", "replace_all", "create"):
            CypherStr += f"""CREATE (n:`{LabelStr}` {{{", ".join([f"`{iField}`: line.`{iField}`" for iField in data.columns])}}})"""
        else:
            CypherStr += f"""MERGE (n:`{LabelStr}` {{`{entity_id}`: line.`{entity_id}`}})
            ON CREATE SET n = line
            ON MATCH SET n += line"""
        self.execute(CypherStr, parameters=None)
        if (CSVExportPath is not None) and args.get("清除CSV", self.ClearCSV):
            try:
                os.remove(ExportPath)
            except Exception as e:
                self._QS_Logger.warning(f" CSV 文件({ExportPath})清除失败, 请手动删除: {str(e)}")
        return 0
    def writeEntityData(self, data, entity_labels, entity_id, if_exists="update", args={}, **kwargs):
        if if_exists=="replace":
            self.deleteEntity(entity_labels, {entity_id: data.index.tolist()})
        elif if_exists=="replace_all":
            self.deleteEntity(entity_labels, None)
        if args.get("CSV导入地址", self.CSVImportPath) is not None:
            return self._writeEntityData_LoadCSV(data, entity_labels, entity_id, if_exists, args=args, **kwargs)
        ThreadNum = kwargs.get("thread_num", 0)
        if ThreadNum==0:
            self._writeEntityData(data, entity_labels, entity_id, if_exists, args=args, **kwargs)
        else:
            Tasks, NumAlloc = [], distributeEqual(data.shape[0], min(ThreadNum, data.shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(NumAlloc)) as Executor:
                for i, iNum in enumerate(NumAlloc):
                    iStartIdx = sum(NumAlloc[:i])
                    iData = data.iloc[iStartIdx:iStartIdx+iNum]
                    Tasks.append(Executor.submit(self._writeEntityData, iData, entity_labels, entity_id, if_exists, args, **kwargs))
                for iTask in concurrent.futures.as_completed(Tasks): iTask.result()
        return 0
    # 删除实体
    # entity_labels: [实体标签]
    # entity_ids: {实体 ID 字段: [实体 ID]}
    def deleteEntity(self, entity_labels, entity_ids, args={}, **kwargs):
        EntityNode = (f"(n:`{'`:`'.join(entity_labels)}`)" if entity_labels else "(n)")
        CypherStr = f"MATCH {EntityNode} "
        if entity_ids:
            CypherStr += "WHERE "+" AND ".join(f"n.`{iField}` IN $entity_ids['{iField}']" for iField in entity_ids)+" "
        PeriodicSize = args.get("定期数量", self.PeriodicSize)
        if PeriodicSize>0:
            CypherStr += f"WITH n LIMIT {args.get('定期数量', PeriodicSize)} DETACH DELETE n RETURN COUNT(*)"
            CypherStr = f"""CALL  apoc.periodic.commit("{CypherStr}", {{`entity_ids`: $entity_ids}}) YIELD updates, executions, runtime, batches"""
        else:
            CypherStr += "DETACH DELETE n"
        return self.execute(CypherStr, parameters={"entity_ids": entity_ids})
    # 创建实体索引
    # entity_label: 实体标签
    # entity_keys: [实体字段]
    # index_name: 索引名称
    def createEntityIndex(self, entity_label, entity_keys, index_name=None, args={}, **kwargs):
        CypherStr = f"CREATE INDEX {'' if index_name is None else index_name} IF NOT EXISTS FOR (n:`{entity_label}`) ON (n.`{'`, `'.join(entity_keys)}`)"
        return self.execute(CypherStr)
    # 写入关系数据
    # data: DataFrame(index=[源ID, 目标ID], columns=[关系属性]), 如果为 data.shape[1]==0 表示只创建关系
    # relation_label: 关系标签
    # source_labels: 源标签
    # target_labels: 目标标签
    # kwargs: 可选参数
    #     source_id: 源 ID 字段, 默认取 data.index.names[0]
    #     target_id: 目标 ID 字段, 默认取 data.index.names[1]
    #     create_entity: 是否创建缺失的实体, 默认取 False
    def _writeRelationData(self, data, relation_label, source_labels, target_labels, source_id, target_id, if_exists="update", conn=None, args={}, **kwargs):
        if (data.shape[1]>0) and (if_exists in ("append", "update_notnull")):
            SourceIDs = data.index.get_level_values(0).unique().tolist()
            TargetIDs = data.index.get_level_values(1).unique().tolist()
            OldDataStr = (f"MATCH (n1:`{'`:`'.join(source_labels)}`) - [r:`{relation_label}`] -> (n2:`{'`:`'.join(target_labels)}`)" if relation_label is not None else f"MATCH (n1:`{'`:`'.join(source_labels)}`) - [r] -> (n2:`{'`:`'.join(target_labels)}`)")
            OldDataStr += f" WHERE n1.`{source_id}` IN $source_ids AND n2.`{target_id}` IN $target_ids "
            OldDataStr += f" RETURN n1.`{source_id}`, n2.`{target_id}`, r.`{'`, r.`'.join(data.columns)}`"
            OldData = self.fetchall(OldDataStr, parameters={"source_ids": SourceIDs, "target_ids": TargetIDs})
            if OldData:
                OldData = pd.DataFrame(OldData, columns=[source_id, target_id]+data.columns.tolist()).set_index([source_id, target_id]).loc[data.index]
                if if_exists=="append":
                    data = OldData.where(pd.notnull(OldData), data)
                else:
                    data = data.where(pd.notnull(data), OldData)
        SourceNode = (f"(n1:`{'`:`'.join(source_labels)}` {{`{source_id}`: d[0]}})" if source_labels else f"(n1 {{`{source_id}`: d[0]}})")
        TargetNode = (f"(n2:`{'`:`'.join(target_labels)}` {{`{target_id}`: d[1]}})" if target_labels else f"(n2 {{`{target_id}`: d[1]}})")
        Relation = (f"(n1) - [r:`{relation_label}`] -> (n2)" if relation_label is not None else "(n1) - [r] -> (n2)")
        CypherStr1 = "UNWIND $data AS d"
        Keyword = ("MERGE" if kwargs.get("create_entity", False) else "MATCH")
        if if_exists in ("replace", "replace_all", "create"):
            CypherStr2 = f"""{Keyword} {SourceNode} {Keyword} {TargetNode} CREATE {Relation}"""
        else:
            CypherStr2 = f"""{Keyword} {SourceNode} {Keyword} {TargetNode} MERGE {Relation}"""
        if data.shape[1]>0:
            CypherStr2 += f" SET "
            for i, iField in enumerate(data.columns): CypherStr2 += f" r.`{iField}` = d[{2+i}], "
            CypherStr2 = CypherStr2[:-2]
        PeriodicSize = args.get("定期数量", self.PeriodicSize)
        if PeriodicSize>0:
            CypherStr = f"""CALL apoc.periodic.iterate("{CypherStr1} RETURN d", "{CypherStr2}", {{`batchSize`: {PeriodicSize}, `parallel`: false, `params`: {{`data`: $data}}}})"""
        else:
            CypherStr = f"""{CypherStr1} {CypherStr2}"""
        data = data.astype("O").where(pd.notnull(data), None).reset_index().values.tolist()
        if conn is None:
            self.execute(CypherStr, parameters={"data": data})
        else:
            with conn.session(database=self.DBName) as Session:
                Session.run(CypherStr, parameters={"data": data})
        return 0
    def _writeRelationData_LoadCSV(self, data, relation_label, source_labels, target_labels, source_id, target_id, if_exists="update", args={}, **kwargs):
        if data.shape[0]>0:
            if (data.shape[1]>0) and (if_exists in ("append", "update_notnull")):
                SourceIDs = data.index.get_level_values(0).unique().tolist()
                TargetIDs = data.index.get_level_values(1).unique().tolist()
                OldDataStr = (f"MATCH (n1:`{'`:`'.join(source_labels)}`) - [r:`{relation_label}`] -> (n2:`{'`:`'.join(target_labels)}`)" if relation_label is not None else f"MATCH (n1:`{'`:`'.join(source_labels)}`) - [r] -> (n2:`{'`:`'.join(target_labels)}`)")
                OldDataStr += f" WHERE n1.`{source_id}` IN $source_ids AND n2.`{target_id}` IN $target_ids "
                OldDataStr += f" RETURN n1.`{source_id}`, n2.`{target_id}`, r.`{'`, r.`'.join(data.columns)}`"
                OldData = self.fetchall(OldDataStr, parameters={"source_ids": SourceIDs, "target_ids": TargetIDs})
                if OldData:
                    OldData = pd.DataFrame(OldData, columns=[source_id, target_id]+data.columns.tolist()).set_index([source_id, target_id]).loc[data.index]
                    if if_exists=="append":
                        data = OldData.where(pd.notnull(OldData), data)
                    else:
                        data = data.where(pd.notnull(data), OldData)
            Fields = data.columns.tolist()
            data = data.astype("O").where(pd.notnull(data), None).reset_index()
            data.columns = ["源ID", "目标ID"] + Fields
        CSVExportPath = args.get("CSV导出地址", self.CSVExportPath)
        if CSVExportPath is not None:
            if os.path.isdir(CSVExportPath):
                ExportPath = genAvailableFile("QSNeo4jRelationData", target_dir=CSVExportPath, suffix="csv")
            else:
                ExportPath = CSVExportPath
            data.to_csv(ExportPath, index=False, header=True)
            _, CSVFile = os.path.split(ExportPath)
            ImportPath = urllib.parse.urljoin(args.get("CSV导入地址", self.CSVImportPath), CSVFile)
        else:
            ImportPath = args.get("CSV导入地址", self.CSVImportPath)
        SourceNode = (f"(n1:`{'`:`'.join(source_labels)}` {{`{source_id}`: d.`源ID`}})" if source_labels else f"(n1 {{`{source_id}`: d.`源ID`}})")
        TargetNode = (f"(n2:`{'`:`'.join(target_labels)}` {{`{target_id}`: d.`目标ID`}})" if target_labels else f"(n2 {{`{target_id}`: d.`目标ID`}})")
        Relation = (f"(n1) - [r:`{relation_label}`] -> (n2)" if relation_label is not None else "(n1) - [r] -> (n2)")
        PeriodicSize = args.get("定期数量", self.PeriodicSize)
        if PeriodicSize>0:
            CypherStr = f"USING PERIODIC COMMIT {PeriodicSize} "
        else:
            CypherStr = ""
        Keyword = ("MERGE" if kwargs.get("create_entity", False) else "MATCH")
        if if_exists in ("replace", "replace_all", "create"):
            CypherStr += f"""LOAD CSV WITH HEADERS FROM "{ImportPath}" AS d FIELDTERMINATOR '{args.get("CSV分隔符", self.CSVSep)}'
                {Keyword} {SourceNode}
                {Keyword} {TargetNode}
                CREATE {Relation}
            """
        else:
            CypherStr += f"""LOAD CSV WITH HEADERS FROM "{ImportPath}" AS d FIELDTERMINATOR '{args.get("CSV分隔符", self.CSVSep)}'
                {Keyword} {SourceNode}
                {Keyword} {TargetNode}
                MERGE {Relation}
            """
        if data.shape[1]>2:
            CypherStr += f" SET "
            for iField in data.columns[2:]: CypherStr += f" r.`{iField}` = d.`{iField}`, "
            CypherStr = CypherStr[:-2]
        self.execute(CypherStr, parameters=None)
        if (CSVExportPath is not None) and args.get("清除CSV", self.ClearCSV):
            try:
                os.remove(ExportPath)
            except Exception as e:
                self._QS_Logger.warning(f" CSV 文件({ExportPath})清除失败, 请手动删除: {str(e)}")
        return 0
    def writeRelationData(self, data, relation_label, source_labels, target_labels, if_exists="update", args={}, **kwargs):
        SourceID = kwargs.pop("source_id", data.index.names[0])
        if pd.isnull(SourceID):
            raise __QS_Error__("QSNeo4jObject.writeRelationData: 源 ID 字段不能缺失, 请指定参数 source_id")
        else:
            SourceID = str(SourceID)
        TargetID = kwargs.pop("target_id", data.index.names[1])
        if pd.isnull(TargetID):
            raise __QS_Error__("QSNeo4jObject.writeRelationData: 目标 ID 字段不能缺失, 请指定参数 target_id")
        else:
            TargetID = str(TargetID)
        data = data.loc[data.index.dropna(how="any")]
        if data.shape[0]==0: return 0
        if if_exists=="replace":
            self.deleteRelation(relation_label, {}, source_labels, {SourceID: data.index.get_level_values(0).unique().tolist()}, target_labels, {TargetID: data.index.get_level_values(1).unique().tolist()})
        elif if_exists=="replace_all":
            self.deleteRelation(relation_label, {}, source_labels, {}, target_labels, {})
        if args.get("CSV导入地址", self.CSVImportPath) is not None:
            return self._writeRelationData_LoadCSV(data, relation_label, source_labels, target_labels, SourceID, TargetID, if_exists, args=args, **kwargs)
        ThreadNum = kwargs.get("thread_num", 0)
        if ThreadNum==0:
            self._writeRelationData(data, relation_label, source_labels, target_labels, SourceID, TargetID, if_exists, conn=None, args=args, **kwargs)
        else:
            Tasks, NumAlloc = [], distributeEqual(data.shape[0], min(ThreadNum, data.shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(NumAlloc)) as Executor:
                for i, iNum in enumerate(NumAlloc):
                    iStartIdx = sum(NumAlloc[:i])
                    iData = data.iloc[iStartIdx:iStartIdx+iNum]
                    iConn, _ = self._getConnection()
                    Tasks.append(Executor.submit(self._writeRelationData, iData, relation_label, source_labels, target_labels, SourceID, TargetID, if_exists, conn=iConn, args=args, **kwargs))
                for iTask in concurrent.futures.as_completed(Tasks): iTask.result()
        return 0
    # 删除关系
    # relation_label: 关系标签
    # relation_ids: {关系 ID 字段: [关系 ID]}
    # source_labels: 源标签
    # source_ids: {源 ID 字段: [源 ID]}
    # target_labels: 目标标签
    # target_ids: {目标 ID 字段: [目标 ID]}
    def deleteRelation(self, relation_label, relation_ids, source_labels, source_ids, target_labels, target_ids, args={}, **kwargs):
        SourceNode = (f"(n1:`{'`:`'.join(source_labels)}`)" if source_labels else "(n1)")
        TargetNode = (f"(n2:`{'`:`'.join(target_labels)}`)" if target_labels else "(n2)")
        Relation = (f"[r:`{relation_label}`]" if relation_label else "[r]")
        CypherStr = f"MATCH {SourceNode} - {Relation} -> {TargetNode} "
        Conditions = []
        if relation_ids:
            for iField in relation_ids: Conditions.append(f"r.`{iField}` IN $relation_ids['{iField}']")
        if source_ids:
            for iField in source_ids: Conditions.append(f"n1.`{iField}` IN $source_ids['{iField}']")
        if target_ids:
            for iField in target_ids: Conditions.append(f"n2.`{iField}` IN $target_ids['{iField}']")
        if Conditions:
            CypherStr += "WHERE "+" AND ".join(Conditions)+" "
        PeriodicSize = args.get("定期数量", self.PeriodicSize)
        if PeriodicSize>0:
            CypherStr += f"WITH r LIMIT {args.get('定期数量', PeriodicSize)} DELETE r RETURN COUNT(*)"
            CypherStr = f"""CALL  apoc.periodic.commit("{CypherStr}", {{`relation_ids`: $relation_ids, `source_ids`: $source_ids, `target_ids`: $target_ids}}) YIELD updates, executions, runtime, batches"""
        else:
            CypherStr += "DELETE r"
        return self.execute(CypherStr, parameters={"relation_ids": relation_ids, "source_ids": source_ids, "target_ids": target_ids})


# 写入参数集
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的参数集节点 id
def writeArgs(args, arg_name=None, parent_var=None, var=None, tx=None):
    Parameters = {}
    if parent_var is not None:
        if var is None: var = parent_var + "_a"
        if arg_name is not None:
            Relation = f"[:参数 {{Name: '{arg_name}'}}]"
        else:
            Relation = f"[:参数 ]"
        CypherStr = f"MERGE ({parent_var}) - {Relation} -> ({var}:`参数集`)"
    else:
        if var is None: var = "a"
        CypherStr = f"CREATE ({var}:`参数集`)"
    if isinstance(args, __QS_Object__):
        Parameters[var] = {"_PickledObject": pickle.dumps(args)}
        args = args.Args
    else:
        Parameters[var] = {}
    if isinstance(args, dict):
        args = args.copy()
        SubArgs = {}
        for iArg in sorted(args.keys()):
            if isinstance(args[iArg], (dict, __QS_Object__)):
                SubArgs[iArg] = args.pop(iArg)
            elif callable(args[iArg]) or isinstance(args[iArg], (list, tuple, pd.Series, pd.DataFrame, np.ndarray)):
                args[iArg] = pickle.dumps(args[iArg])
        if args:
            CypherStr += f" SET {var} += ${var}"
            Parameters[var].update(args)
        for i, iArg in enumerate(SubArgs.keys()):
            iCypherStr, iParameters = writeArgs(SubArgs[iArg], arg_name=iArg, parent_var=var, var=var+"_a"+str(i), tx=None)
            CypherStr += " " + iCypherStr
            Parameters.update(iParameters)
    else:
        raise __QS_Error__(f"不支持的因子集类型 : {type(args)}")
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({var})"
        return tx.run(CypherStr, parameters=Parameters).values()[0][0]    

# 写入因子, id_var: {id(对象): 变量名}
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的 [因子节点 id]
def writeFactor(factors, tx=None, id_var=None, write_other_fundamental_factor=True):
    if id_var is None: id_var = {}
    CypherStr, Parameters, NodeVars = "", {}, []
    for iFactor in factors:
        iFID = id(iFactor)
        iVar = f"f{iFID}"
        iClass = str(iFactor.__class__)
        iClass = iClass[iClass.index("'")+1:]
        iClass = iClass[:iClass.index("'")]
        if not isinstance(iFactor, DerivativeFactor):# 基础因子
            iFT = iFactor.FactorTable
            if iFID not in id_var:
                iNode = f"({iVar}:`因子`:`基础因子` {{Name: '{iFactor.Name}', `_Class`: '{iClass}'}})"
                if iFT is not None:# 有上层因子表
                    iFTVar = f"ft{id(iFT)}"
                    iFTStr, iFTParameters = writeFactorTable(iFT, tx=None, var=iFTVar, id_var=id_var, write_other_fundamental_factor=write_other_fundamental_factor)
                    if iFTStr: CypherStr += " "+iFTStr
                    Parameters.update(iFTParameters)
                    CypherStr += f" CREATE {iNode} - [:`产生于` {{`原始因子`: '{iFactor._NameInFT}'}}] -> ({iFTVar})"
                else:
                    CypherStr += f" CREATE {iNode}"
                iArgStr, iParameters = writeArgs(iFactor.Args, arg_name=None, parent_var=iVar, tx=None)
                if iArgStr: CypherStr += " "+iArgStr
                Parameters.update(iParameters)
                id_var[iFID] = iVar
        else:# 衍生因子
            iSubStr, iSubParameters = writeFactor(iFactor.Descriptors, tx=None, id_var=id_var, write_other_fundamental_factor=write_other_fundamental_factor)
            if iSubStr: CypherStr += " "+iSubStr
            Parameters.update(iSubParameters)
            if iFID not in id_var:
                iNode = f"({iVar}:`因子`:`衍生因子` {{Name: '{iFactor.Name}', `_Class`: '{iClass}'}})"
                CypherStr += f" CREATE {iNode}"
                iArgStr, iParameters = writeArgs(iFactor.Args, arg_name=None, parent_var=iVar, tx=None)
                if iArgStr: CypherStr += " "+iArgStr
                Parameters.update(iParameters)
                for j, jDescriptor in enumerate(iFactor.Descriptors):
                    CypherStr += f" MERGE ({iVar}) - [:`依赖` {{Order: {j}}}] -> ({id_var[id(jDescriptor)]})"
                id_var[iFID] = iVar
        NodeVars.append(iVar)
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({'), id('.join(NodeVars)})"
        return tx.run(CypherStr, parameters=Parameters).values()[0]

# 删除因子
def deleteFactor(factor_name, labels=["因子"], ft=None, fdb=None, del_descriptors=True, tx=None):
    raise NotImplemented

# 写入因子表
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的因子表节点 id
def writeFactorTable(ft, tx=None, var="ft", id_var=None, write_other_fundamental_factor=True):
    if id_var is None: id_var = {}
    CypherStr, Parameters = "", {}
    Class = str(ft.__class__)
    Class = Class[Class.index("'")+1:]
    Class = Class[:Class.index("'")]
    # 写入因子库
    FDB = ft.FactorDB
    FTID, FDBID = id(ft), id(FDB)
    FDBVar = f"fdb{FDBID}"
    if FDB is not None:# 有上层因子库, 非自定义因子表
        if FDBID not in id_var:
            FDBStr, FDBParameters = writeFactorDB(FDB, tx=None, var=FDBVar)
            CypherStr += " "+FDBStr
            Parameters.update(FDBParameters)
            id_var[FDBID] = FDBVar
        # 写入因子表
        if FTID not in id_var:
            FTNode = f"({var}:`因子表`:`库因子表` {{Name: '{ft.Name}', `_Class`: '{Class}'}})"
            CypherStr += f" MERGE {FTNode} - [:`属于因子库`] -> ({FDBVar})"
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, parent_var=var,  tx=None)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            id_var[FTID] = var
            # 写入因子
            if write_other_fundamental_factor:
                for iFactorName in ft.FactorNames:
                    CypherStr += f" MERGE (:`因子`:`基础因子` {{Name: '{iFactorName}'}}) - [:`属于因子表`] -> ({var})"
    else:# 无上层因子库, 自定义因子表
        if FTID not in id_var:
            FStr, FParameters = writeFactor([ft.getFactor(iFactorName) for iFactorName in ft.FactorNames], tx=None, id_var=id_var, write_other_fundamental_factor=write_other_fundamental_factor)
            if FStr: CypherStr += " "+FStr
            Parameters.update(FParameters)
            FTNode = f"({var}:`因子表`:`自定义因子表` {{Name: '{ft.Name}', `_Class`: '{Class}'}})"
            CypherStr += f" CREATE {FTNode} "
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, parent_var=var, tx=None)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            for iFactorName in ft.FactorNames:
                CypherStr += f" MERGE ({var}) - [:`包含因子`] -> ({id_var[id(ft.getFactor(iFactorName))]})"
            id_var[FTID] = var
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({var})"
        return tx.run(CypherStr, parameters=Parameters).values()[0][0]

# 删除因子表
def deleteFactorTable(table_name, labels=["因子表"], fdb=None, del_factors=True, del_descriptors=True, tx=None):
    raise NotImplementedError

# 写入因子库
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的因子库节点 id
def writeFactorDB(fdb, tx=None, var="fdb"):
    Class = str(fdb.__class__)
    Class = Class[Class.index("'")+1:]
    Class = Class[:Class.index("'")]
    Node = f"({var}:`因子库`:`{fdb.__class__.__name__}` {{`Name`: '{fdb.Name}', `_Class`: '{Class}'}})"
    CypherStr = "MERGE "+Node
    Args = fdb.Args
    Args.pop("用户名", None)
    Args.pop("密码", None)
    ArgStr, Parameters = writeArgs(Args, arg_name=None, parent_var=var, tx=None)
    if ArgStr: CypherStr += " " + ArgStr
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({var})"
        return tx.run(CypherStr, parameters=Parameters).values()[0][0]

# 删除因子库
def deleteFactorDB(fdb_name, tx=None):
    raise NotImplementedError

# 读取参数集
def readArgs(node, node_id=None, tx=None):
    if node_id is None:
        CypherStr = f"MATCH p={node} - [:`参数`*1..] -> (args:`参数集`) RETURN length(p) AS Level, [r IN relationships(p) | r.Name] AS Path, properties(args) AS Args ORDER BY Level"
    else:
        CypherStr = f"MATCH p=(n) - [:`参数`*1..] -> (args:`参数集`) WHERE id(n)={node_id} RETURN length(p) AS Level, [r IN relationships(p) | r.Name] AS Path, properties(args) AS Args ORDER BY Level"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    for i, iRslt in enumerate(Rslt):
        iArgs = iRslt[2]
        if "_PickledObject" in iArgs:
            iObj = iArgs.pop("_PickledObject")
            try:
                iObj = pickle.loads(iObj)
            except Exception as e:
                print(e)
            else:
                for iArgName, iVal in enumerate(iArgs.items()):
                    iObj[iArgName] = iVal
                iArgs = iObj
        for iArgName, iVal in iArgs.items():
            if isinstance(iVal, bytes):
                try:
                    iArgs[iArgName] = pickle.loads(iVal)
                except Exception as e:
                    print(e)
        if i==0:
            Args = iArgs
        else:
            iDict = Args
            for iKey in iRslt[1][1:-1]:
                iSubDict = iDict.get(iKey, {})
                if iKey not in iDict:
                    iDict[iKey] = iSubDict
                iDict = iSubDict
            iDict[iRslt[1][-1]] = iArgs
    return Args

# 读取因子库
def readFactorDB(labels=["因子库"], properties={}, node_id=None, tx=None, **kwargs):
    if node_id is None:
        Constraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in properties.items()))
        CypherStr = f"MATCH (fdb:`{'`:`'.join(labels)}` {{{Constraint}}}) RETURN fdb"
    else:
        CypherStr = f"MATCH (fdb) WHERE id(fdb)={node_id} RETURN fdb"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    FDBs = []
    for iRslt in Rslt:
        iProperties = dict(iRslt[0])
        if "_PickledObject" in iProperties:
            iFDB = iProperties.pop("_PickledObject")
            try:
                iFDB = pickle.loads(iFDB)
            except Exception as e:
                print(e)
            else:
                iArgs = readArgs(None, node_id=iRslt[0].id, tx=tx)
                for iArgName, iVal in enumerate(iArgs.items()):
                    iFDB[iArgName] = iVal
                FDBs.append(iFDB)
                continue
        try:
            Class = iProperties["_Class"].split(".")
            iModule = importlib.import_module('.'.join(Class[:-1]))
            FDBClass = getattr(iModule, Class[-1])
        except Exception as e:
            raise __QS_Error__(f"无法还原因子库({iProperties})对象: {e}")
        else:
            iArgs = readArgs(None, node_id=iRslt[0].id, tx=tx)
            iFDB = FDBClass(sys_args=iArgs, **kwargs)
        FDBs.append(iFDB)
    if len(FDBs)>1:
        return FDBs
    else:
        return FDBs[0]

# 读取因子表
def readFactorTable(labels=["因子表"], properties={}, node_id=None, tx=None, **kwargs):
    if node_id is None:
        Constraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in properties.items()))
        CypherStr = f"MATCH (ft:`{'`:`'.join(labels)}` {{{Constraint}}}) RETURN ft" 
    else:
        CypherStr = f"MATCH (ft) WHERE id(ft) = {node_id} RETURN ft"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    FTs = []
    for iRslt in Rslt:
        iProperties = dict(iRslt[0])
        iFTID, iTableName = iRslt[0].id, iProperties["Name"]
        if "_PickledObject" in iProperties:
            iFT = iProperties.pop("_PickledObject")
            try:
                iFT = pickle.loads(iFT)
            except Exception as e:
                print(e)
            else:
                iArgs = readArgs(None, node_id=iFTID, tx=tx)
                for iArgName, iVal in enumerate(iArgs.items()):
                    iFT[iArgName] = iVal
                FTs.append(iFT)
                continue
        iArgs = readArgs(None, iFTID, tx=tx)
        if "库因子表" in iRslt[0].labels:# 库因子表, 构造库
            CypherStr = f"MATCH (ft:`因子表`) - [:`属于因子库`] -> (fdb:`因子库`) WHERE id(ft)={iFTID} RETURN id(fdb)"
            iFDBID = tx.run(CypherStr).values()[0][0]
            iFDB = readFactorDB(node_id=iFDBID, tx=tx, **kwargs)
            iFT = iFDB.getTable(iTableName, args=iArgs)
        else:# 自定义因子表
            CypherStr = f"MATCH (ft:`因子表`) - [:`包含因子`] -> (f:`因子`) WHERE id(ft)={iFTID} RETURN collect(DISTINCT id(f))"
            iFactorIDs = tx.run(CypherStr).values()[0][0]
            iFactors = readFactor(node_ids=iFactorIDs, tx=tx, **kwargs)
            iFT = CustomFT(iTableName, sys_args=iArgs, **kwargs)
            iFT.addFactors(factor_list=iFactors)
        FTs.append(iFT)
    if len(FTs)>1:
        return FTs
    else:
        return FTs[0]

# 读取因子
def readFactor(labels=["因子"], properties={}, node_ids=None, tx=None, id_fdb={}, id_ft={}, id_factor={}, **kwargs):
    if node_ids is None:
        Constraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in properties.items()))
        CypherStr = f"MATCH (f:`{'`:`'.join(labels)}` {{{Constraint}}}) RETURN f" 
    else:
        CypherStr = f"MATCH (f) WHERE id(f) IN {str(node_ids)} RETURN f"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    Factors = []
    for iRslt in Rslt:
        iProperties = dict(iRslt[0])
        iFactorID, iFactorName = iRslt[0].id, iProperties["Name"]
        if iFactorID in id_factor:
            Factors.append(id_factor[iFactorID])
            continue
        if "_PickledObject" in iProperties:
            iFactor = iProperties.pop("_PickledObject")
            try:
                iFactor = pickle.loads(iFactor)
            except Exception as e:
                print(e)
            else:
                iArgs = readArgs(None, node_id=iFactorID, tx=tx)
                for iArgName, iVal in enumerate(iArgs.items()):
                    iFactor[iArgName] = iVal
                id_factor[iFactorID] = iFactor
                Factors.append(iFactor)
                continue
        if "基础因子" in iRslt[0].labels:# 基础因子, 构造表和库
            CypherStr = f"MATCH (f:`因子`) - [r:`产生于`|`属于因子表`] -> (ft:`因子表`) - [:`属于因子库`] -> (fdb:`因子库`) WHERE id(f)={iFactorID} RETURN r.`原始因子`, id(ft), ft.Name, id(fdb)"
            iRawName, iFTID, iFTName, iFDBID = tx.run(CypherStr).values()[0]
            if iFDBID in id_fdb:
                iFDB = id_fdb[iFDBID]
            else:
                id_fdb[iFDBID] = iFDB = readFactorDB(node_id=iFDBID, tx=tx, **kwargs)
            if iFTID in id_ft:
                iFT = id_ft[iFTID]
            else:
                id_ft[iFTID] = iFT = iFDB.getTable(iFTName)
            iArgs = readArgs(None, iFactorID, tx=tx)
            id_factor[iFactorID] = iFactor = iFT.getFactor(iRawName, args=iArgs, new_name=iFactorName)
        else:# 衍生因子
            CypherStr = f"MATCH (f:`因子`) - [r:`依赖`] -> (d:`因子`) WHERE id(f)={iFactorID} RETURN id(d) ORDER BY r.`Order`"
            iDescriptorIDs = [iRslt[0] for iRslt in tx.run(CypherStr).values()]
            iDescriptors = readFactor(node_ids=iDescriptorIDs, tx=tx, id_fdb=id_fdb, id_ft=id_ft, id_factor=id_factor, **kwargs)
            iArgs = readArgs(None, iFactorID, tx=tx)
            iClass = iProperties["_Class"].split(".")
            iModule = importlib.import_module('.'.join(iClass[:-1]))
            iFactorClass = getattr(iModule, iClass[-1])            
            id_factor[iFactorID] = iFactor = iFactorClass(iFactorName, iDescriptors, sys_args=iArgs, **kwargs)
        Factors.append(iFactor)
    return Factors