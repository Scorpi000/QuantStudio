# coding=utf-8
"""基于 Mongo 数据库的因子库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, Float, ListStr, List

from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.api import Panel

def _identifyDataType(dtypes):
    if np.dtype('O') in dtypes.values: return 'string'
    else: return 'double'

def _adjustData(data, look_back, factor_names, ids, dts):
    if ids is not None:
        data = Panel(data).loc[factor_names, :, ids]
    else:
        data = Panel(data).loc[factor_names, :, :]
    if look_back==0:
        if dts is not None:
            return data.loc[:, dts]
        else:
            return data
    if dts is not None:
        AllDTs = data.major_axis.union(dts).sort_values()
        data = data.loc[:, AllDTs, :]
    if np.isinf(look_back):
        for i, iFactorName in enumerate(data.items): data.iloc[i].fillna(method="pad", inplace=True)
    else:
        data = dict(data)
        Limits = look_back*24.0*3600
        for iFactorName in data: data[iFactorName] = fillNaByLookback(data[iFactorName], lookback=Limits)
        data = Panel(data).loc[factor_names]
    if dts is not None:
        return data.loc[:, dts]
    else:
        return data


class _WideTable(FactorTable):
    """MongoDB 宽因子表"""
    TableType = Enum("宽表", arg_type="SingleOption", label="因子表类型", order=0)
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=1)
    FilterCondition = List([], arg_type="List", label="筛选条件", order=2)
    #DTField = Enum("datetime", arg_type="SingleOption", label="时点字段", order=3)
    #IDField = Enum("code", arg_type="SingleOption", label="ID字段", order=4)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DataType = fdb._TableFactorDict[name]
        self._Collection = fdb._DB[fdb.InnerPrefix+name]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        Fields = ["datetime"] + self._DataType.index.tolist()
        self.add_trait("DTField", Enum(*Fields, arg_type="SingleOption", label="时点字段", order=4))
        Fields = ["code"] + self._DataType.index.tolist()
        self.add_trait("IDField", Enum(*Fields, arg_type="SingleOption", label="ID字段", order=5))
    @property
    def FactorNames(self):
        return sorted(self._DataType.index.union({"datetime", "code"}).difference({self.DTField, self.IDField}))
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        elif set(factor_names).isdisjoint(self.FactorNames): return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        if key=="DataType": return self._DataType.loc[factor_names]
        MetaData = {}
        Doc = self._Collection.find_one({"code": "_TableInfo"}, {"datetime": 0, "code": 0, "_id": 0})
        for iFactorName in factor_names:
            if key is None:
                MetaData[iFactorName] = pd.Series(Doc.get(iFactorName, {}))
            else:
                MetaData[iFactorName] = Doc.get(iFactorName, {}).get(key, None)
        if not MetaData: return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        if key is None: return pd.DataFrame(MetaData).loc[:, factor_names]
        else: return pd.Series(MetaData).loc[factor_names]
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        Doc = []
        if idt is not None: Doc.append({DTField: idt})
        if ifactor_name is not None: Doc.append({ifactor_name: {"$ne": None}})
        FilterConds = args.get("筛选条件", self.FilterCondition)
        if FilterConds: Doc += FilterConds
        if Doc: Doc = {"$and": Doc}
        else: Doc = {}
        IDs = self._Collection.distinct(IDField, Doc)
        IDs = pd.Series(IDs)
        return sorted(IDs[pd.notnull(IDs) & (IDs!="_TableInfo")])
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        Doc = []
        if iid is not None: Doc.append({IDField: iid})
        if start_dt is not None: Doc.append({DTField: {"$gte": start_dt}})
        if end_dt is not None: Doc.append({DTField: {"$lte": end_dt}})
        if ifactor_name is not None: Doc.append({ifactor_name: {"$ne": None}})
        FilterConds = args.get("筛选条件", self.FilterCondition)
        if FilterConds: Doc += FilterConds
        if Doc: Doc = {"$and": Doc}
        else: Doc = {}
        DTs = self._Collection.distinct(DTField, Doc)
        DTs = pd.Series(DTs)
        return sorted(DTs[pd.notnull(DTs)])
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ArgConditionGroup = {}
        ArgNames = self.ArgNames
        ArgNames.remove("回溯天数")
        ArgNames.remove("因子值类型")
        ArgNames.remove("遍历模式")
        for iFactor in factors:
            iArgConditions = (";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in ArgNames]))
            if iArgConditions not in ArgConditionGroup:
                ArgConditionGroup[iArgConditions] = {"FactorNames":[iFactor.Name], 
                                                     "RawFactorNames":{iFactor._NameInFT}, 
                                                     "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                     "args":iFactor.Args.copy()}
            else:
                ArgConditionGroup[iArgConditions]["FactorNames"].append(iFactor.Name)
                ArgConditionGroup[iArgConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ArgConditionGroup[iArgConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ArgConditionGroup[iArgConditions]["StartDT"])
                ArgConditionGroup[iArgConditions]["args"]["回溯天数"] = max(ArgConditionGroup[iArgConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iArgConditions in ArgConditionGroup:
            StartInd = operation_mode.DTRuler.index(ArgConditionGroup[iArgConditions]["StartDT"])
            Groups.append((self, ArgConditionGroup[iArgConditions]["FactorNames"], list(ArgConditionGroup[iArgConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ArgConditionGroup[iArgConditions]["args"]))
        return Groups
    def _genNullIDRawData(self, factor_names, ids, end_date, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        Doc = [{IDField: {"$in": ids}}, {DTField: {"$lt": end_date}}]
        FilterConds = args.get("筛选条件", self.FilterCondition)
        if FilterConds: Doc += FilterConds
        RawData = self._Collection.aggregate([{"$match": {"$and": Doc}}, {"$group": {"_id": "$code", DTField: {"$max": "$"+DTField}}}])
        RawData = pd.DataFrame(RawData).rename(columns={"_id": IDField})
        FieldDoc = {DTField: 1, IDField: 1, "_id": 0}
        FieldDoc.update({iField: 1 for iField in factor_names})
        RawData = self._Collection.find({"$or": RawData.to_dict(orient="records")}, FieldDoc)
        return pd.DataFrame(RawData)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        LookBack = args.get("回溯天数", self.LookBack)
        if dts is not None:
            dts = sorted(dts)
            StartDate, EndDate = dts[0], dts[-1]
            if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        else:
            StartDate = EndDate = None
        Doc = []
        if StartDate is not None:
            Doc.append({DTField: {"$gte": StartDate, "$lte": EndDate}})
        else:
            Doc.append({DTField: {"$ne": None}})
        if ids is not None:
            Doc.append({IDField: {"$in": ids}})
        FilterConds = args.get("筛选条件", self.FilterCondition)
        if FilterConds: Doc += FilterConds
        if Doc: Doc = {"$and": Doc}
        else: Doc = {}
        FieldDoc = {DTField: 1, IDField: 1, "_id": 0}
        FieldDoc.update({iField: 1 for iField in factor_names})
        RawData = pd.DataFrame(self._Collection.find(Doc, FieldDoc))
        if RawData.shape[1]==0: RawData = pd.DataFrame(columns=[DTField, IDField]+factor_names)
        if (StartDate is not None) and np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData[DTField]==StartDate][IDField]))
            if NullIDs:
                NullRawData = self._genNullIDRawData(factor_names, list(NullIDs), StartDate, args=args)
                if NullRawData.shape[0]>0:
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
        RawData = RawData.sort_values(by=[DTField, IDField]).rename(columns={DTField: "QS_DT", IDField: "ID"})
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["QS_DT", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        return _adjustData(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts)

class MongoDB(WritableFactorDB):
    """MongoDB"""
    Name = Str("MongoDB", arg_type="String", label="名称", order=-100)
    DBType = Enum("Mongo", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=27017, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=6)
    Connector = Enum("default", "pymongo", arg_type="SingleOption", label="连接器", order=7)
    IgnoreFields = ListStr(arg_type="List", label="忽略字段", order=8)
    InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=9)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"MongoDBConfig.json" if config_file is None else config_file), **kwargs)
        self._TableFactorDict = {}# {表名: pd.Series(数据类型, index=[因子名])}
        self._TableFieldDataType = {}# {表名: pd.Series(数据库数据类型, index=[因子名])}
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_Connection"] = (True if self.isAvailable() else False)
        state["_DB"] = None
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
        if (self.Connector=="pymongo") or ((self.Connector=="default") and (self.DBType=="Mongo")):
            try:
                import pymongo
                self._Connection = pymongo.MongoClient(host=self.IPAddr, port=self.Port)
            except Exception as e:
                Msg = ("'%s' 尝试使用 pymongo 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "pymongo"
        self._PID = os.getpid()
        self._DB = self._Connection[self.DBName]
        return 0
    def connect(self):
        self._connect()
        nPrefix = len(self.InnerPrefix)
        if self.DBType=="Mongo":
            self._TableFactorDict = {}
            for iTableName in self._DB.collection_names():
                if iTableName[:nPrefix]==self.InnerPrefix:
                    iTableInfo = self._DB[iTableName].find_one({"code": "_TableInfo"}, {"datetime": 0, "code": 0, "_id": 0})
                    if iTableInfo:
                        self._TableFactorDict[iTableName[nPrefix:]] = pd.Series({iFactorName: iInfo["DataType"] for iFactorName, iInfo in iTableInfo.items() if iFactorName not in self.IgnoreFields})
        return 0
    @property
    def TableNames(self):
        return sorted(self._TableFactorDict)
    def getTable(self, table_name, args={}):
        if table_name not in self._TableFactorDict:
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不存在因子表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        TableType = args.get("因子表类型", "宽表")
        if TableType=="宽表":
            return _WideTable(name=table_name, fdb=self, sys_args=args, logger=self._QS_Logger)
        else:
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不支持的因子表类型: '%s'" % (self.Name, TableType))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableFactorDict:
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 不存在因子表 '%s'!" % (self.Name, old_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableFactorDict):
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 新因子表名 '%s' 已经存在于库中!" % (self.Name, new_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        self._DB[self.InnerPrefix+old_table_name].rename(self.InnerPrefix+new_table_name)
        self._TableFactorDict[new_table_name] = self._TableFactorDict.pop(old_table_name)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableFactorDict: return 0
        self._DB.drop_collection(self.InnerPrefix+table_name)
        self._TableFactorDict.pop(table_name, None)
        return 0
    # 创建表, field_types: {字段名: 数据类型}
    def createTable(self, table_name, field_types):
        if self.InnerPrefix+table_name not in self._DB.collection_names():
            Doc = {iField: {"DataType": iDataType} for iField, iDataType in field_types.items()}
            Doc.update({"datetime": None, "code": "_TableInfo"})
            Collection = self._DB[self.InnerPrefix+table_name]
            Collection.insert(Doc)
            # 添加索引
            if self._Connector=="pymongo":
                import pymongo
                Index1 = pymongo.IndexModel([("datetime", pymongo.ASCENDING), ("code", pymongo.ASCENDING)], name=self.InnerPrefix+"datetime_code")
                Index2 = pymongo.IndexModel([("code", pymongo.HASHED)], name=self.InnerPrefix+"code")
                try:
                    Collection.create_indexes([Index1, Index2])
                except Exception as e:
                    self._QS_Logger.warning("'%s' 调用方法 createTable 在数据库中创建表 '%s' 的索引时错误: %s" % (self.Name, table_name, str(e)))
        self._TableFactorDict[table_name] = pd.Series(field_types)
        return 0
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name not in self._TableFactorDict[table_name]:
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 因子表 '%s' 中不存在因子 '%s'!" % (self.Name, table_name, old_factor_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_factor_name!=old_factor_name) and (new_factor_name in self._TableFactorDict[table_name]):
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 新因子名 '%s' 已经存在于因子表 '%s' 中!" % (self.Name, new_factor_name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        self._DB[self.InnerPrefix+table_name].update_many({}, {"$rename": {old_factor_name: new_factor_name}})
        self._TableFactorDict[table_name][new_factor_name] = self._TableFactorDict[table_name].pop(old_factor_name)
        return 0
    def deleteFactor(self, table_name, factor_names):
        if not factor_names: return 0
        FactorIndex = self._TableFactorDict.get(table_name, pd.Series()).index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        self.deleteField(self.InnerPrefix+table_name, factor_names)
        for iFactorName in factor_names:
            self._DB[self.InnerPrefix+table_name].update_many({}, {'$unset': {iFactorName: 1}})
        self._TableFactorDict[table_name] = self._TableFactorDict[table_name][FactorIndex]
        return 0
    # 增加因子，field_types: {字段名: 数据类型}
    def addFactor(self, table_name, field_types):
        if table_name not in self._TableFactorDict: return self.createTable(table_name, field_types)
        Doc = {iField: {"DataType": iDataType} for iField, iDataType in field_types.items()}
        self._DB[self.InnerPrefix+table_name].update({"code": "_TableInfo"}, {"$set": Doc})
        self._TableFactorDict[table_name] = self._TableFactorDict[table_name].append(field_types)
        return 0
    def deleteData(self, table_name, ids=None, dts=None):
        Doc = {}
        if dts is not None:
            Doc["datetime"] = {"$in": dts}
        if ids is not None:
            Doc["code"] = {"$in": ids}
        if Doc:
            self._DB[self.InnerPrefix+table_name].delete_many(Doc)
        else:
            self._DB.drop_collection(self.InnerPrefix+table_name)
            self._TableFactorDict.pop(table_name)
        return 0
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._TableFactorDict:
            FieldTypes = {iFactorName:_identifyDataType(data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            self.createTable(table_name, field_types=FieldTypes)
        else:
            NewFactorNames = data.items.difference(self._TableFactorDict[table_name].index).tolist()
            if NewFactorNames:
                FieldTypes = {iFactorName:_identifyDataType(data.iloc[i].dtypes) for i, iFactorName in enumerate(NewFactorNames)}
                self.addFactor(table_name, FieldTypes)
            if if_exists=="update":
                OldFactorNames = self._TableFactorDict[table_name].index.difference(data.items).tolist()
                if OldFactorNames:
                    OldData = self.getTable(table_name).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    for iFactorName in OldFactorNames: data[iFactorName] = OldData[iFactorName]
            else:
                AllFactorNames = self._TableFactorDict[table_name].index.tolist()
                OldData = self.getTable(table_name).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                if if_exists=="append":
                    for iFactorName in AllFactorNames:
                        if iFactorName in data:
                            data[iFactorName] = OldData[iFactorName].where(pd.notnull(OldData[iFactorName]), data[iFactorName])
                        else:
                            data[iFactorName] = OldData[iFactorName]
                elif if_exists=="update_notnull":
                    for iFactorName in AllFactorNames:
                        if iFactorName in data:
                            data[iFactorName] = data[iFactorName].where(pd.notnull(data[iFactorName]), OldData[iFactorName])
                        else:
                            data[iFactorName] = OldData[iFactorName]
                else:
                    Msg = ("因子库 '%s' 调用方法 writeData 错误: 不支持的写入方式 '%s'!" % (self.Name, if_exists))
                    self._QS_Logger.error(Msg)
                    raise __QS_Error__(Msg)
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        Mask = pd.notnull(NewData).any(axis=1)
        NewData = NewData[Mask]
        if NewData.shape[0]==0: return 0
        self.deleteData(table_name, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
        NewData = NewData.reset_index()
        NewData.columns = ["datetime", "code"] + NewData.columns[2:].tolist()
        self._DB[self.InnerPrefix+table_name].insert_many(NewData.to_dict(orient="records"))
        return 0