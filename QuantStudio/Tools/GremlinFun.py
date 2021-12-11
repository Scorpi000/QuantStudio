# coding=utf-8
import os
import pickle
import importlib
import concurrent.futures

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Password

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.FactorDataBase.FactorOperation import DerivativeFactor
from QuantStudio.FactorDataBase.FactorDB import CustomFT
from QuantStudio.Tools.AuxiliaryFun import distributeEqual

# TODO
class QSGremlinObject(__QS_Object__):
    """基于 Gremlin 的对象"""
    Name = Str("QSGremlinObject")
    URL = Str("ws://127.0.0.1:8182/gremlin", arg_type="String", label="地址", order=0)
    TraversalSource = Str("g", arg_type="String", label="TraversalSource", order=1)
    User = Str("", arg_type="String", label="用户名", order=2)
    Pwd = Password("", arg_type="String", label="密码", order=3)
    Connector = Enum("default", "gremlinpython", arg_type="SingleOption", label="连接器", order=4)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Connection = None# 连接对象
        self._Connector = None# 实际使用的数据库链接器
        self._PID = None# 保存数据库连接创建时的进程号
        self._g = None# traversal source
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
        if (self.Connector=="gremlinpython") or (self.Connector=="default"):
            try:
                from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
                from gremlin_python.process.anonymous_traversal import traversal
                Conn = DriverRemoteConnection(url=self.URL, traversal_source=self.TraversalSource, username=self.User, password=self.Pwd)
                g = traversal().withRemote(Conn)
            except Exception as e:
                Msg = ("'%s' 尝试使用 gremlinpython 连接 %s 失败: %s" % (self.Name, self.URL, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                Connector = "gremlinpython"
        return Conn, g, Connector
    def _connect(self):
        self._Connection, self._g, self._Connector = self._getConnection()
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
                self._Connection = self._g = None
        return 0
    def isAvailable(self):
        return (self._Connection is not None)
    def TraversalSource(self):
        if self._Connection is None:
            Msg = ("'%s' 获取 TraversalSource 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        return self._g
    # 写入实体数据
    # data: DataFrame(index=[实体ID], columns=[实体属性])
    # entity_labels: [实体标签]
    # entity_id: 实体 ID 字段
    # kwargs: 可选参数
    #     thread_num: 写入线程数量
    def _writeEntityData(self, data, entity_labels, entity_id, if_exists="update", **kwargs):
        IDs, Fields = data.index.tolist(), data.columns.tolist()
        LabelStr = "`:`".join(entity_labels)
        if if_exists=="replace":
            CypherStr = f"UNWIND $ids AS iID CREATE (n:`{LabelStr}`) SET n = $data[iID]"
            data[entity_id] = data.index
            data = data.astype("O").where(pd.notnull(data), None)
            return self.execute(CypherStr, parameters={"ids": IDs, "data": data.T.to_dict(orient="dict")})
        if if_exists!="update":
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
            elif if_exists=="update_notnull":
                data = data.where(pd.notnull(data), OldData)
            else:
                raise __QS_Error__(f"因子库 '{self.Name}' 调用方法 writeFeatureData 错误: 不支持的写入方式 {if_exists}'!")
        CypherStr = f"""
            UNWIND $ids AS iID
            MERGE (n:`{LabelStr}` {{`{entity_id}`: iID}})
            ON CREATE SET n = $data[iID]
            ON MATCH SET n += $data[iID]
        """
        data[entity_id] = data.index
        data = data.astype("O").where(pd.notnull(data), None)
        return self.execute(CypherStr, parameters={"ids": IDs, "data": data.T.to_dict(orient="dict")})
    def writeEntityData(self, data, entity_labels, entity_id, if_exists="update", **kwargs):
        if if_exists=="replace":
            self.deleteEntity(entity_labels, {entity_id: data.index.tolist()})
        ThreadNum = kwargs.get("thread_num", 0)
        if ThreadNum==0:
            self._writeEntityData(data, entity_labels, entity_id, if_exists, **kwargs)
        else:
            Tasks, NumAlloc = [], distributeEqual(data.shape[0], min(ThreadNum, data.shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(NumAlloc)) as Executor:
                for i, iNum in enumerate(NumAlloc):
                    iStartIdx = sum(NumAlloc[:i])
                    iData = data.iloc[iStartIdx:iStartIdx+iNum]
                    Tasks.append(Executor.submit(self._writeEntityData, iData, entity_labels, entity_id, if_exists, **kwargs))
                for iTask in concurrent.futures.as_completed(Tasks): iTask.result()
        return 0
    # 删除实体
    # entity_labels: [实体标签]
    # entity_ids: {实体 ID 字段: [实体 ID]}
    def deleteEntity(self, entity_labels, entity_ids, **kwargs):
        EntityNode = (f"(n:`{'`:`'.join(entity_labels)}`)" if entity_labels else "(n)")
        CypherStr = f"MATCH {EntityNode} "
        if entity_ids:
            CypherStr += "WHERE "+" AND ".join(f"n.`{iField}` IN $entity_ids['{iField}']" for iField in entity_ids)+" "
        CypherStr += "DETACH DELETE n"
        return self.execute(CypherStr, parameters={"entity_ids": entity_ids})
    # 创建实体索引
    # entity_label: 实体标签
    # entity_keys: [实体字段]
    # index_name: 索引名称
    def createEntityIndex(self, entity_label, entity_keys, index_name=None, **kwargs):
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
    #     thread_num: 写入线程数量
    def _writeRelationData(self, data, relation_label, source_labels, target_labels, if_exists="update", conn=None, **kwargs):
        SourceID = kwargs.get("source_id", data.index.names[0])
        if pd.isnull(SourceID):
            raise __QS_Error__("QSNeo4jObject._writeRelationData: 源 ID 字段不能缺失, 请指定参数 source_id")
        else:
            SourceID = str(SourceID)
        TargetID = kwargs.get("target_id", data.index.names[1])
        if pd.isnull(TargetID):
            raise __QS_Error__("QSNeo4jObject._writeRelationData: 目标 ID 字段不能缺失, 请指定参数 target_id")
        else:
            TargetID = str(TargetID)
        SourceNode = (f"(n1:`{'`:`'.join(source_labels)}` {{`{SourceID}`: d[0]}})" if source_labels else f"(n1 {{`{SourceID}`: d[0]}})")
        TargetNode = (f"(n2:`{'`:`'.join(target_labels)}` {{`{TargetID}`: d[1]}})" if target_labels else f"(n2 {{`{TargetID}`: d[1]}})")
        Relation = (f"(n1) - [r:`{relation_label}`] -> (n2)" if relation_label is not None else "(n1) - [r] -> (n2)")
        CypherStr = f"""
            UNWIND $data AS d
            MERGE {SourceNode}
            MERGE {TargetNode}
            MERGE {Relation}
        """
        data = data.loc[data.index.dropna(how="any")]
        if data.shape[0]==0: return 0
        if data.shape[1]>0:
            CypherStr += f" SET "
            for i, iField in enumerate(data.columns): CypherStr += f" r.`{iField}` = d[{2+i}], "
            CypherStr = CypherStr[:-2]
            if if_exists!="update":
                SourceIDs = data.index.get_level_values(0).tolist()
                TargetIDs = data.index.get_level_values(1).tolist()
                OldDataStr = (f"MATCH (n1:`{'`:`'.join(source_labels)}`) - [r:`{relation_label}`] -> (n2:`{'`:`'.join(target_labels)}`)" if relation_label is not None else f"MATCH (n1:`{'`:`'.join(source_labels)}`) - [r] -> (n2:`{'`:`'.join(target_labels)}`)")
                OldDataStr += f" WHERE n1.`{SourceID}` IN $source_ids AND n2.`{TargetID}` IN $target_ids "
                OldDataStr += f" RETURN n1.`{SourceID}`, n2.`{TargetID}`, r.`{'`, r.`'.join(data.columns)}`"
                OldData = self.fetchall(OldDataStr, parameters={"source_ids": SourceIDs, "target_ids": TargetIDs})
                if OldData:
                    OldData = pd.DataFrame(OldData, columns=[SourceID, TargetID]+data.columns.tolist()).set_index([SourceID, TargetID]).loc[data.index]
                    if if_exists=="append":
                        data = OldData.where(pd.notnull(OldData), data)
                    elif if_exists=="update_notnull":
                        data = data.where(pd.notnull(data), OldData)
                    else:
                        raise __QS_Error__(f"QSNeo4jObject._writeRelationData: 不支持的写入方式 '{if_exists}'")
        data = data.astype("O").where(pd.notnull(data), None).reset_index().values.tolist()
        if conn is None:
            self.execute(CypherStr, parameters={"data": data})
        else:
            with conn.session(database=self.DBName) as Session:
                Session.run(CypherStr, parameters={"data": data})
        return 0
    def writeRelationData(self, data, relation_label, source_labels, target_labels, if_exists="update", **kwargs):
        ThreadNum = kwargs.get("thread_num", 0)
        if ThreadNum==0:
            self._writeRelationData(data, relation_label, source_labels, target_labels, if_exists, conn=None, **kwargs)
        else:
            Tasks, NumAlloc = [], distributeEqual(data.shape[0], min(ThreadNum, data.shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(NumAlloc)) as Executor:
                for i, iNum in enumerate(NumAlloc):
                    iStartIdx = sum(NumAlloc[:i])
                    iData = data.iloc[iStartIdx:iStartIdx+iNum]
                    iConn, _ = self._getConnection()
                    Tasks.append(Executor.submit(self._writeRelationData, iData, relation_label, source_labels, target_labels, if_exists, conn=iConn, **kwargs))
                for iTask in concurrent.futures.as_completed(Tasks): iTask.result()
        return 0
    # 删除关系
    # relation_label: 关系标签
    # relation_ids: {关系 ID 字段: [关系 ID]}
    # source_labels: 源标签
    # source_ids: {源 ID 字段: [源 ID]}
    # target_labels: 目标标签
    # target_ids: {目标 ID 字段: [目标 ID]}
    def deleteRelation(self, relation_label, relation_ids, source_labels, source_ids, target_labels, target_ids, **kwargs):
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
        CypherStr += "DELETE r"
        return self.execute(CypherStr, parameters={"relation_ids": relation_ids, "source_ids": source_ids, "target_ids": target_ids})

if __name__=="__main__":
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.process.anonymous_traversal import traversal
    #from gremlin_python.structure.graph import Graph
    
    conn = DriverRemoteConnection(url="ws://127.0.0.1:8182/gremlin", traversal_source="g", username="", password="")
    g = traversal().withRemote(conn)
    #g.addV("person").property("name", "Dennis").next()
    print(g.V().values("name").toList())
    print("===")