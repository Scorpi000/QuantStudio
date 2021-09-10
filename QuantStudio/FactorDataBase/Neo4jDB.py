# coding=utf-8
"""基于 Neo4j 数据库的因子库(TODO)"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str, Dict, Password, Range, Enum, ListStr, Bool

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.Neo4jFun import writeArgs
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import _QS_calcData_WideTable, _QS_calcData_NarrowTable

def _identifyDataType(factor_data, data_type=None):
    if (data_type is None) or (data_type=="double"):
        try:
            factor_data = factor_data.astype(float)
        except:
            data_type = "object"
        else:
            data_type = "double"
    return (factor_data, data_type)

class _NarrowTable(FactorTable):
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
        CypherStr = f"MATCH (f:`因子`) - [:`属于因子表`] -> (ft:`因子表` {{Name: '{self.Name}'}}) - [:`属于因子库`] -> {self._FactorDB._Node} "
        if ifactor_name is not None:
            CypherStr += f"WHERE f.Name = '{ifactor_name}' "
        CypherStr += "MATCH (s) - [r:`暴露`] -> (f) "
        if idt is not None:
            CypherStr += f"WHERE r.`{idt.strftime(self._DTFormat)}` IS NOT NULL "
        CypherStr += "RETURN s.ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(CypherStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        CypherStr = f"MATCH (f:`因子`) - [:`属于因子表`] -> (ft:`因子表` {{Name: '{self.Name}'}}) - [:`属于因子库`] -> {self._FactorDB._Node} "
        if ifactor_name is not None:
            CypherStr += f"WHERE f.Name = '{ifactor_name}' "
        CypherStr += "MATCH (s) - [r:`暴露`] -> (f) "
        if iid is not None:
            CypherStr += f"WHERE s.ID = '{iid}' "
        CypherStr += "WITH keys(r) AS kk UNWIND kk AS ik RETURN collect(DISTINCT ik)"
        Rslt = self._FactorDB.fetchall(CypherStr)
        if not Rslt: return []
        Rslt = Rslt[0][0]
        if start_dt is not None: start_dt = start_dt.strftime(self._DTFormat)
        if end_dt is not None: end_dt = end_dt.strftime(self._DTFormat)
        return [dt.datetime.strptime(iDT, self._DTFormat) for iDT in Rslt if pd.notnull(iDT) and ((start_dt is None) or (iDT>=start_dt)) and ((end_dt is None) or (iDT<=end_dt))]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        CypherStr = f"""
            MATCH (f:`因子`) - [:`属于因子表`] -> (ft:`因子表` {{Name: '{self.Name}'}}) - [:`属于因子库`] -> {self._FactorDB._Node}
            WHERE f.Name IN $factors
            MATCH (s) - [r:`暴露`] -> (f)
            WHERE s.ID IN $ids
            WITH s, f, r
            UNWIND $dts AS iDT
            RETURN s.ID AS ID, iDT AS QS_DT, f.Name AS FactorName, r[iDT] AS Value
            ORDER BY ID, QS_DT, FactorName
        """
        DTs = [iDT.strftime(self._DTFormat) for iDT in dts]
        RawData = self._FactorDB.fetchall(CypherStr, parameters={"factors": factor_names, "ids": ids, "dts": DTs})
        RawData = pd.DataFrame(RawData, columns=["ID", "QS_DT", "FactorName", "Value"])
        #RawData["QS_DT"] = RawData["QS_DT"].apply(lambda x: dt.datetime.strptime(x, self._DTFormat) if pd.notnull(x) else pd.NaT)
        RawData["QS_DT"] = pd.to_datetime(RawData["QS_DT"])
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        #if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        #if ids is None: ids = sorted(raw_data["ID"].unique())
        #raw_data = raw_data.set_index(["QS_DT", "ID", "FactorName"]).iloc[:, 0]
        #raw_data = raw_data.unstack()
        #DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        #DataType = DataType[~DataType.index.duplicated()]
        #Data = {}
        #for iFactorName in factor_names:
            #if iFactorName in raw_data:
                #iRawData = raw_data[iFactorName].unstack()
                #if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
                #Data[iFactorName] = iRawData
        #if not Data: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        #return pd.Panel(Data).loc[factor_names]
        Args = self.Args
        Args.update(args)
        Args["因子名字段"] = "FactorName"
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        ErrorFmt = {"DuplicatedIndex":  "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        return _QS_calcData_NarrowTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)

class _EntityFeatureTable(FactorTable):
    EntityLabels = ListStr(["因子库"], arg_type="List", label="实体标签", order=0)
    IDField = Str("Name", arg_type="String", label="ID字段", order=1)
    MultiMapping = Bool(False, label="多重映射", arg_type="Bool", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        LabelStr = "`:`".join(self.EntityLabels)
        CypherStr = f"""
            MATCH (n:`{LabelStr}`)
            WITH keys(n) AS kk
            UNWIND kk AS ik
            RETURN collect(DISTINCT ik)
        """
        FactorNames = self._FactorDB.fetchall(CypherStr)
        if not FactorNames: return FactorNames
        FactorNames = sorted(FactorNames[0][0])
        if self.IDField in FactorNames: FactorNames.remove(self.IDField)
        return FactorNames
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            LabelStr = "`:`".join(args.get("实体标签", self.EntityLabels))
            CypherStr = f"MATCH (n:`{LabelStr}`) WITH n, keys(n) AS kk UNWIND kk AS ik RETURN collect(DISTINCT [ik, apoc.meta.type(n[ik])])"
            DataType = self._FactorDB.fetchall(CypherStr)
            if not DataType: return pd.Series(index=factor_names, dtype="O")
            DataType = pd.DataFrame(DataType[0][0], columns=["FactorName", "DataType"]).set_index(["FactorName"]).iloc[:, 0]
            Mapping = {"STRING": "string", "INTEGER": "double", "FLOAT": "double", "LIST": "object"}
            DataType = DataType.replace(Mapping)
            DataType[~DataType.isin(Mapping)] = "object"
            return DataType.groupby(level=0).apply(lambda s: s.iloc[0] if s.shape[0]==0 else "object")
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        LabelStr = "`:`".join(args.get("实体标签", self.EntityLabels))
        IDField = args.get("ID字段", self.IDField)
        CypherStr = f"MATCH (n:`{LabelStr}`) "
        if ifactor_name is not None:
            CypherStr += f"WHERE n.`{ifactor_name}` IS NOT NULL "
        CypherStr += f"RETURN collect(DISTINCT n.`{IDField}`)"
        IDs = self._FactorDB.fetchall(CypherStr)
        if not IDs: return IDs
        else: return sorted(IDs[0][0])
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        LabelStr = "`:`".join(args.get("实体标签", self.EntityLabels))
        IDField = args.get("ID字段", self.IDField)
        CypherStr = f"MATCH (n:`{LabelStr}`) "
        CypherStr += f"WHERE n.`{IDField}` IN $ids "
        CypherStr += f"RETURN n.`{IDField}` AS QS_ID, n.`{'`, n.`'.join(factor_names)}`"
        RawData = self._FactorDB.fetchall(CypherStr, parameters={"ids": ids})
        if not RawData: return pd.DataFrame(columns=["ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID"]+factor_names)
        RawData["QS_TargetDT"] = dt.datetime.combine(dt.date.today(), dt.time(0)) + dt.timedelta(1)
        RawData["QS_DT"] = RawData["QS_TargetDT"]
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        TargetDT = raw_data.pop("QS_TargetDT").iloc[0].to_pydatetime()
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Args = self.Args
        Args.update(args)
        ErrorFmt = {"DuplicatedIndex":  "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        Data = _QS_calcData_WideTable(raw_data, factor_names, ids, [TargetDT], DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)
        Data = Data.iloc[:, 0, :]
        return pd.Panel(Data.values.T.reshape((Data.shape[1], Data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=Data.index, minor_axis=dts).swapaxes(1, 2)

class _RelationFeatureTable(FactorTable):
    IDEntity = ListStr(["因子表"], arg_type="List", label="ID实体", order=0)
    IDField = Str("Name", arg_type="String", label="ID字段", order=1)
    OppEntity = ListStr(["因子库"], arg_type="List", label="关联实体", order=2)
    OppConstraint = Dict(arg_type="Dict", label="关联约束", order=3)
    OppField = Str("Name", arg_type="Dict", label="关联字段", order=4)
    RelationLabel = Str("属于因子库", arg_type="String", label="关系标签", order=5)
    Direction = Enum("->", "<-", arg_type="String", label="关系方向", order=6)
    MultiMapping = Bool(False, label="多重映射", arg_type="Bool", order=7)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        IDNode = f"(n1:`{'`:`'.join(self.IDEntity)}`) "
        OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in self.OppConstraint.items()))
        OppNode = f"(n2:`{'`:`'.join(self.OppEntity)}` {{{OppConstraint}}})"
        if self.Direction=="->":
            CypherStr = f"MATCH {IDNode} - [r:`{self.RelationLabel}`] -> {OppNode} "
        elif self.Direction=="<-":
            CypherStr = f"MATCH {IDNode} <- [r:`{self.RelationLabel}`] - {OppNode} "
        CypherStr += f"WHERE n1.`{self.IDField}` IS NOT NULL "
        CypherStr += "WITH keys(r) AS kk UNWIND kk AS ik RETURN collect(DISTINCT ik)"
        FactorNames = self._FactorDB.fetchall(CypherStr)
        if not FactorNames: return FactorNames
        FactorNames = sorted(FactorNames[0][0])
        if self.OppField:
            return FactorNames+["是否存在", "关联实体"]
        else:
            return FactorNames+["是否存在"]
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            IDNode = f"(n1:`{'`:`'.join(args.get('ID实体', self.IDEntity))}`) "
            IDField = args.get("ID字段", self.IDField)
            OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in args.get('关联约束', self.OppConstraint).items()))
            OppNode = f"(n2:`{'`:`'.join(args.get('关联实体', self.OppEntity))}` {{{OppConstraint}}})"
            OppField = args.get("关联字段", self.OppField)
            if self.Direction=="->":
                CypherStr = f"MATCH {IDNode} - [r:`{args.get('关系标签', self.RelationLabel)}`] -> {OppNode} "
            elif self.Direction=="<-":
                CypherStr = f"MATCH {IDNode} <- [r:`{args.get('关系标签', self.RelationLabel)}`] - {OppNode} "
            CypherStr += f"WHERE n1.`{IDField}` IS NOT NULL "
            CypherStr += "WITH r, keys(r) AS kk UNWIND kk AS ik RETURN collect(DISTINCT [ik, apoc.meta.type(r[ik])])"
            DataType = self._FactorDB.fetchall(CypherStr)
            if not DataType:
                DataType = pd.Series(index=factor_names, dtype="O")
            else:
                DataType = pd.DataFrame(DataType[0][0], columns=["FactorName", "DataType"]).set_index(["FactorName"]).iloc[:, 0]
                Mapping = {"STRING": "string", "INTEGER": "double", "FLOAT": "double", "LIST": "object"}
                DataType = DataType.replace(Mapping)
                DataType[~DataType.isin(Mapping)] = "object"
            if "是否存在" in factor_names:
                DataType["是否存在"] = "double"
            if "关联实体" in factor_names:
                CypherStr = f"MATCH {OppNode} RETURN apoc.meta.type(n2.`{OppField}`)"
                DataType["关联实体"] = Mapping.get(self._FactorDB.fetchall(CypherStr)[0][0], "object")
            return DataType.groupby(level=0).apply(lambda s: s.iloc[0] if s.shape[0]==0 else "object").loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDNode = f"(n1:`{'`:`'.join(args.get('ID实体', self.IDEntity))}`) "
        IDField = args.get("ID字段", self.IDField)
        OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in args.get('关联约束', self.OppConstraint).items()))
        OppNode = f"(n2:`{'`:`'.join(args.get('关联实体', self.OppEntity))}` {{{OppConstraint}}})"
        OppField = args.get("关联字段", self.OppField)
        if self.Direction=="->":
            CypherStr = f"MATCH {IDNode} - [r:`{args.get('关系标签', self.RelationLabel)}`] -> {OppNode} "
        elif self.Direction=="<-":
            CypherStr = f"MATCH {IDNode} <- [r:`{args.get('关系标签', self.RelationLabel)}`] - {OppNode} "
        CypherStr += f"WHERE n1.`{IDField}` IS NOT NULL "
        if ifactor_name=="关联实体":
            CypherStr += f"AND n2.`{OppField}` IS NOT NULL "
        elif (ifactor_name is not None) and (ifactor_name != "是否存在"):
            CypherStr += f"AND r.`{ifactor_name}` IS NOT NULL "
        CypherStr += f"RETURN collect(DISTINCT n1.`{IDField}`)"
        IDs = self._FactorDB.fetchall(CypherStr)
        if not IDs: return IDs
        else: return sorted(IDs[0][0])
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IDNode = f"(n1:`{'`:`'.join(args.get('ID实体', self.IDEntity))}`) "
        IDField = args.get("ID字段", self.IDField)
        OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in args.get('关联约束', self.OppConstraint).items()))
        OppNode = f"(n2:`{'`:`'.join(args.get('关联实体', self.OppEntity))}` {{{OppConstraint}}})"
        OppField = args.get("关联字段", self.OppField)
        if self.Direction=="->":
            CypherStr = f"MATCH {IDNode} - [r:`{args.get('关系标签', self.RelationLabel)}`] -> {OppNode} "
        elif self.Direction=="<-":
            CypherStr = f"MATCH {IDNode} <- [r:`{args.get('关系标签', self.RelationLabel)}`] - {OppNode} "
        CypherStr += f"WHERE n1.`{IDField}` IN $ids "
        if OppField and ("关联实体" in factor_names):
            iIdx = factor_names.index("关联实体")
            factor_names = factor_names[:iIdx]+factor_names[iIdx+1:]
            CypherStr += f"RETURN n1.`{IDField}`, n2.{OppField}, r.`{'`, r.`'.join(factor_names)}`"
            factor_names = ["关联实体"] + factor_names
        else:
            CypherStr += f"RETURN n1.`{IDField}`, r.`{'`, r.`'.join(factor_names)}`"
        RawData = self._FactorDB.fetchall(CypherStr, parameters={"ids": ids})
        if not RawData: return pd.DataFrame(columns=["ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID"]+factor_names)
        RawData["QS_TargetDT"] = dt.datetime.combine(dt.date.today(), dt.time(0)) + dt.timedelta(1)
        RawData["QS_DT"] = RawData["QS_TargetDT"]
        if "是否存在" in factor_names: RawData["是否存在"] = 1
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        TargetDT = raw_data.pop("QS_TargetDT").iloc[0].to_pydatetime()
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Args = self.Args
        Args.update(args)
        ErrorFmt = {"DuplicatedIndex":  "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        Data = _QS_calcData_WideTable(raw_data, factor_names, ids, [TargetDT], DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)
        Data = Data.iloc[:, 0, :]
        return pd.Panel(Data.values.T.reshape((Data.shape[1], Data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=Data.index, minor_axis=dts).swapaxes(1, 2)

class _RelationFeatureOppFactorTable(FactorTable):
    IDEntity = ListStr(["因子表"], arg_type="List", label="ID实体", order=0)
    IDField = Str("Name", arg_type="String", label="ID字段", order=1)
    OppEntity = ListStr(["因子库"], arg_type="List", label="关联实体", order=2)
    OppConstraint = Dict(arg_type="Dict", label="关联约束", order=3)
    OppField = Str("Name", arg_type="Dict", label="因子名字段", order=4)
    RelationLabel = Str("属于因子库", arg_type="String", label="关系标签", order=5)
    Direction = Enum("->", "<-", arg_type="String", label="关系方向", order=6)
    RelationField = Str(arg_type="String", label="因子值字段", order=7)
    MultiMapping = Bool(False, label="多重映射", arg_type="Bool", order=8)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        IDNode = f"(n1:`{'`:`'.join(self.IDEntity)}`) "
        OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in self.OppConstraint.items()))
        OppNode = f"(n2:`{'`:`'.join(self.OppEntity)}` {{{OppConstraint}}})"
        if self.Direction=="->":
            Relation = f"{IDNode} - [r:`{self.RelationLabel}`] -> {OppNode}"
        elif self.Direction=="<-":
            Relation = f"{IDNode} <- [r:`{self.RelationLabel}`] - {OppNode}"
        CypherStr = f"MATCH {Relation} "
        CypherStr += f"WHERE n1.`{self.IDField}` IS NOT NULL AND n2.`{self.OppField}` IS NOT NULL "
        CypherStr += f"RETURN collect(DISTINCT n2.`{self.OppField}`)"
        FactorNames = self._FactorDB.fetchall(CypherStr)
        if not FactorNames: return FactorNames
        return sorted(FactorNames[0][0])
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            IDNode = f"(n1:`{'`:`'.join(args.get('ID实体', self.IDEntity))}`) "
            IDField = args.get("ID字段", self.IDField)
            OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in args.get('关联约束', self.OppConstraint).items()))
            OppNode = f"(n2:`{'`:`'.join(args.get('关联实体', self.OppEntity))}` {{{OppConstraint}}})"
            Direction = args.get("关系方向", self.Direction)
            RelationField = args.get("因子值字段", self.RelationField)
            if not RelationField: return pd.Series("double", index=factor_names)
            if Direction=="->":
                Relation = f"{IDNode} - [r:`{args.get('关系标签', self.RelationLabel)}`] -> {OppNode} "
            elif Direction=="<-":
                Relation = f"{IDNode} <- [r:`{args.get('关系标签', self.RelationLabel)}`] - {OppNode} "
            else:
                Msg = ("因子库 '%s' 调用方法 getFactorMetaData 错误: 不支持的参数值 %s : %s " % (self.Name, "关系方向", str(Direction)))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
            CypherStr = f"MATCH {Relation} "
            CypherStr += f"WHERE n1.`{IDField}` IS NOT NULL AND r.`{RelationField}` IS NOT NULL "
            CypherStr += f"RETURN collect(DISTINCT apoc.meta.type(r.`{RelationField}`))"
            DataType = self._FactorDB.fetchall(CypherStr)
            if not DataType:
                DataType = pd.Series(index=factor_names, dtype="O")
            else:
                Mapping = {"STRING": "string", "INTEGER": "double", "FLOAT": "double", "LIST": "object"}
                DataType = pd.Series("object" if len(DataType[0][0])>1 else Mapping.get(DataType[0][0][0], "object"), index=factor_names)
            return DataType
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDNode = f"(n1:`{'`:`'.join(args.get('ID实体', self.IDEntity))}`) "
        IDField = args.get("ID字段", self.IDField)
        OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in args.get('关联约束', self.OppConstraint).items()))
        OppNode = f"(n2:`{'`:`'.join(args.get('关联实体', self.OppEntity))}` {{{OppConstraint}}})"
        OppField = args.get("因子名字段", self.OppField)
        RelationField = args.get("因子值字段", self.RelationField)
        if self.Direction=="->":
            CypherStr = f"MATCH {IDNode} - [r:`{args.get('关系标签', self.RelationLabel)}`] -> {OppNode} "
        elif self.Direction=="<-":
            CypherStr = f"MATCH {IDNode} <- [r:`{args.get('关系标签', self.RelationLabel)}`] - {OppNode} "
        CypherStr += f"WHERE n1.`{IDField}` IS NOT NULL "
        if ifactor_name is not None:
            CypherStr += f"AND n2.`{OppField}` = {ifactor_name} "
        if RelationField:
            CypherStr += f"AND r.`{RelationField}` IS NOT NULL "
        CypherStr += f"RETURN collect(DISTINCT n1.`{IDField}`)"
        IDs = self._FactorDB.fetchall(CypherStr)
        if not IDs: return IDs
        else: return sorted(IDs[0][0])
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IDNode = f"(n1:`{'`:`'.join(args.get('ID实体', self.IDEntity))}`) "
        IDField = args.get("ID字段", self.IDField)
        OppConstraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in args.get('关联约束', self.OppConstraint).items()))
        OppNode = f"(n2:`{'`:`'.join(args.get('关联实体', self.OppEntity))}` {{{OppConstraint}}})"
        OppField = args.get("因子名字段", self.OppField)
        RelationField = args.get("因子值字段", self.RelationField)
        if self.Direction=="->":
            CypherStr = f"MATCH {IDNode} - [r:`{args.get('关系标签', self.RelationLabel)}`] -> {OppNode} "
        elif self.Direction=="<-":
            CypherStr = f"MATCH {IDNode} <- [r:`{args.get('关系标签', self.RelationLabel)}`] - {OppNode} "
        CypherStr += f"WHERE n1.`{IDField}` IN $ids AND n2.`{OppField}` IN $factor_names "
        if RelationField:
            CypherStr += f"RETURN n1.`{IDField}`, n2.`{OppField}`, r.`{RelationField}`"
        else:
            CypherStr += f"RETURN n1.`{IDField}`, `n2.{OppField}`, 1"
        RawData = self._FactorDB.fetchall(CypherStr, parameters={"ids": ids, "factor_names": factor_names})
        if not RawData: return pd.DataFrame(columns=["ID", "FactorName", "Value"])
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "FactorName", "Value"])
        RawData["QS_DT"] = dts[-1]
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
                Data[iFactorName] = iRawData.fillna(method="pad")
        if not Data: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return pd.Panel(Data).loc[factor_names]
    

class Neo4jDB(WritableFactorDB):
    """Neo4jDB"""
    Name = Str("Neo4jDB", arg_type="String", label="名称", order=-100)
    DBName = Str("neo4j", arg_type="String", label="数据库名", order=0)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=7687, arg_type="Integer", label="端口", order=2)
    User = Str("neo4j", arg_type="String", label="用户名", order=3)
    Pwd = Password("", arg_type="String", label="密码", order=4)
    Connector = Enum("default", "neo4j", arg_type="SingleOption", label="连接器", order=5)
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
        Class = str(self.__class__)
        Class = Class[Class.index("'")+1:]
        Class = Class[:Class.index("'")]
        self._Node = f"(fdb:`因子库`:`{self.__class__.__name__}` {{`Name`: '{self.Name}', `_Class`: '{Class}'}})"
        Args = self.Args
        Args.pop("用户名")
        Args.pop("密码")
        with self._Connection.session(database=self.DBName) as Session:
            with Session.begin_transaction() as tx:
                CypherStr = f"MERGE {self._Node}"
                iCypherStr, Parameters = writeArgs(Args, arg_name=None, tx=None, parent_var="fdb")
                CypherStr += " "+iCypherStr
                tx.run(CypherStr, parameters=Parameters)
                CypherStr = f"MATCH (ft:`因子表`) - [:`属于因子库`] -> {self._Node} RETURN DISTINCT ft.Name AS TableName, ft.description AS Description"
                self._TableInfo = tx.run(CypherStr).values()
                CypherStr = f"MATCH (f:`因子`) - [:`属于因子表`] -> (ft:`因子表`) - [:`属于因子库`] -> {self._Node} RETURN DISTINCT f.Name AS FactorName, ft.Name AS TableName, f.DataType AS DataType, f.description AS Description"
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
    def fetchall(self, cypher_str, parameters=None):
        with self.session() as Session:
            with Session.begin_transaction() as tx:
                return tx.run(cypher_str, parameters=parameters).values()
    def execute(self, cypher_str, parameters=None):
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
                tx.run(cypher_str, parameters=parameters)
        return 0
    # ----------------------------因子表操作-----------------------------
    @property
    def TableNames(self):
        return sorted(self._TableInfo.index)+["实体属性", "关系属性(关系字段做因子)", "关系属性(关联实体字段做因子)"]
    def getTable(self, table_name, args={}):
        if (table_name not in self._TableInfo.index) and (table_name not in ("实体属性", "关系属性(关系字段做因子)", "关系属性(关联实体字段做因子)")):
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不存在因子表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        Args = self.FTArgs.copy()
        Args.update(args)
        if table_name=="实体属性":
            return _EntityFeatureTable(name=table_name, fdb=self, sys_args=Args, logger=self._QS_Logger)
        elif table_name=="关系属性(关系字段做因子)":
            return _RelationFeatureTable(name=table_name, fdb=self, sys_args=Args, logger=self._QS_Logger)
        elif table_name=="关系属性(关联实体字段做因子)":
            return _RelationFeatureOppFactorTable(name=table_name, fdb=self, sys_args=Args, logger=self._QS_Logger)
        else:
            return _NarrowTable(name=table_name, fdb=self, sys_args=Args, logger=self._QS_Logger)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 不存在因子表 '%s'!" % (self.Name, old_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableInfo.index):
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 新因子表名 '%s' 已经存在于库中!" % (self.Name, new_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        CypherStr = f"MATCH (ft:`因子表` {{`Name`: '{old_table_name}'}}) - [:`属于因子库`] -> {self._Node} SET ft.`Name` ='{new_table_name}'"
        self.execute(CypherStr)
        self._TableInfo = self._TableInfo.rename(index={old_table_name: new_table_name})
        self._FactorInfo = self._FactorInfo.rename(index={old_table_name: new_table_name}, level=0)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableInfo.index: return 0
        CypherStr = f"MATCH (f:`因子`) - [:`属于因子表`] -> (ft:`因子表` {{`Name`: '{table_name}'}}) - [:`属于因子库`] -> {self._Node} DETACH DELETE f, ft"
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
        CypherStr = f"MATCH (:`因子` {{`Name`: '{old_factor_name}'}}) - [:`属于因子表`] -> (ft:`因子表` {{`Name`: '{table_name}'}}) - [:`属于因子库`] -> {self._Node} SET f.`Name` = '{new_factor_name}'"
        self.execute(CypherStr)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].rename(index={old_factor_name: new_factor_name}, level=1))
        return 0
    def deleteFactor(self, table_name, factor_names):
        if (not factor_names) or (table_name not in self._TableInfo.index): return 0
        FactorIndex = self._FactorInfo.loc[table_name].index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        CypherStr = f"MATCH (f:`因子`) - [:`属于因子表`] -> (:`因子表` {{`Name`: '{table_name}'}}) - [:`属于因子库`] -> {self._Node} "
        CypherStr += "WHERE "+genSQLInCondition("f.Name", factor_names, is_str=True)+" "
        CypherStr += "DETACH DELETE f"
        self.execute(CypherStr)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].loc[FactorIndex])
        return 0
    # ----------------------------数据操作---------------------------------
    # 附加参数: id_type: str, 比如: A股, 指数, 公募基金
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        FactorNames, DTs, IDs = data.items.tolist(), data.major_axis.tolist(), data.minor_axis.tolist()
        DataType = data_type.copy()
        for i, iFactorName in enumerate(FactorNames):
            data[iFactorName], DataType[iFactorName] = _identifyDataType(data.iloc[i], data_type.get(iFactorName, None))
        InitCypherStr = f"""
            MATCH {self._Node}
            MERGE (ft:`因子表` {{Name: '{table_name}'}})
            MERGE (ft) - [:`属于因子库`] -> (fdb)
            WITH ft
            UNWIND $factors AS iFactor
            MERGE (f:`因子` {{Name: iFactor, DataType: $data_type[iFactor]}})
            MERGE (f) - [:`属于因子表`] -> (ft)
        """
        IDType = kwargs.get("id_type", "")
        if IDType: IDType = f":`{IDType}`"
        WriteCypherStr = f"""
            MATCH (f:`因子` {{Name: $ifactor}}) - [:`属于因子表`] -> (ft:`因子表` {{Name: '{table_name}'}}) - [:`属于因子库`] -> {self._Node}
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
    # 写入实体属性数据
    # data: DataFrame(index=[ID], columns=[属性])
    def writeEntityFeatureData(self, data, entity_labels, id_field="Name", if_exists="update", **kwargs):
        IDs, FactorNames = data.index.tolist(), data.columns.tolist()
        if if_exists!="update":
            OldData = self.getTable("实体属性", args={"实体标签": entity_labels, "ID字段": id_field}).readData(factor_names=FactorNames, ids=IDs, dts=[dt.datetime.combine(dt.date.today(), dt.time(0))]).iloc[:, 0]
            if if_exists=="append":
                data = OldData.where(pd.notnull(OldData), data)
            elif if_exists=="update_notnull":
                data = data.where(pd.notnull(data), OldData)
            else:
                Msg = ("因子库 '%s' 调用方法 writeFeatureData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
        LabelStr = "`:`".join(entity_labels)
        data[id_field] = data.index
        CypherStr = f"""
            UNWIND $ids AS iID
            MERGE (n:`{LabelStr}` {{`{id_field}`: iID}})
            ON CREATE SET n = $data[iID]
            ON MATCH SET n += $data[iID]
        """
        data = data.astype("O").where(pd.notnull(data), None)
        with self.session() as Session:
            with Session.begin_transaction() as tx:
                tx.run(CypherStr, parameters={"ids": IDs, "data": data.T.to_dict(orient="dict")})
        return 0
    # 写入关系属性数据
    # data: DataFrame(index=[ID], columns=[关联实体字段])
    # retain_relation: False 表示 relation_field 为 NULL 的关系将被删除, True 表示不删除
    def writeRelationFeatureData(self, data, relation_label, relation_field, direction, id_labels, id_field, opp_labels, opp_field, retain_relation=True, if_exists="update", **kwargs):
        OppFields = data.columns.tolist()
        IDNode = f"(n1:`{'`:`'.join(id_labels)}` {{`{id_field}`: p[0]}})"
        OppNode = f"(n2:`{'`:`'.join(opp_labels)}` {{`{opp_field}`: $opp_entity[i]}})"
        if direction=="->":
            Relation = f"(n1) - [r:`{relation_label}`] -> (n2)"
        elif direction=="<-":
            Relation = f"(n1) <- [r:`{relation_label}`] - (n2)"
        else:
            Msg = ("因子库 '%s' 调用方法 writeRelationFeatureData 错误: 不支持的参数值 %s : %s " % (self.Name, "direction", str(direction)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        CypherStr = f"""
            UNWIND range(0, size($opp_entity)-1) AS i
            MERGE {OppNode}
            WITH n2, i
            UNWIND $data[i] AS p
            MERGE {IDNode}
            MERGE {Relation}
        """
        if if_exists=="update_notnull":
            data = data.apply(lambda s: s.dropna().reset_index().values.tolist(), axis=0, raw=False).tolist()
            CypherStr += f" SET r.`{relation_field}` = p[1]"
        elif if_exists=="append":
            data = data.apply(lambda s: s.dropna().reset_index().values.tolist(), axis=0, raw=False).tolist()
            CypherStr += f" WHERE r.`{relation_field}` IS NULL SET r.`{relation_field}` = p[1]"
        else:
            OldIDStr = f"""
                MATCH (n1:`{'`:`'.join(id_labels)}`) - [r:`{relation_label}`] -> (n2:`{'`:`'.join(opp_labels)}`)
                WHERE n1.`{id_field}` IS NOT NULL AND r.`{relation_field}` IS NOT NULL AND n2.`{opp_field}` IN $opp_entity
                RETURN collect(DISTINCT n1.`{id_field}`)
            """
            OldIDs = self.fetchall(OldIDStr, parameters={"opp_entity": OppFields})
            if OldIDs: OldIDs = OldIDs[0][0]
            def _chg2None(s):
                s = s.loc[s.dropna().index.union(OldIDs)].astype("O")
                return s.where(pd.notnull(s), None).reset_index().values.tolist()
            data = data.apply(_chg2None, axis=0, raw=False).tolist()
            CypherStr += f" SET r.`{relation_field}` = p[1]"
        self.execute(CypherStr, parameters={"opp_entity": OppFields, "data": data})
        # 删除属性为 NULL 的关系
        if not retain_relation:
            DelStr = f"""
                MATCH (n1:`{'`:`'.join(id_labels)}`) - [r:`{relation_label}`] -> (n2:`{'`:`'.join(opp_labels)}`)
                WHERE n1.`{id_field}` IS NOT NULL AND r.`{relation_field}` IS NULL AND n2.`{opp_field}` IN $opp_entity
                DELETE r
            """
            self.execute(DelStr, parameters={"opp_entity": OppFields})
        return 0