# coding=utf-8
"""基于 Elasticsearch 的因子库"""
import os
import time
import datetime as dt

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers, client
from elasticsearch.exceptions import ConnectionTimeout
from traits.api import Enum, Str, Float, ListStr, List, Dict, Bool

from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.api import Panel, genAvailableName

_TypeMapping = {
    "keyword": "string",
    "text": "string",
    "float": "double",
    "double": "double",
    "integer": "double",
    "long": "double",
    "short": "double",
    "byte": "double",
    "half_float": "double",
    "date": "object",
}

def _identifyDataType(dtypes):
    if np.dtype('O') in dtypes.values: return 'keyword'
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
    """ElasticSearchDB 宽因子表"""
    TableType = Enum("WideTable", arg_type="SingleOption", label="因子表类型", order=0)
    PreFilterID = Bool(True, arg_type="Bool", label="预筛选ID", order=1)
    FilterCondition = List([], arg_type="List", label="筛选条件", order=2)
    #DTField = Enum("datetime", arg_type="SingleOption", label="时点字段", order=3)
    #IDField = Enum("code", arg_type="SingleOption", label="ID字段", order=4)
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=5)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._TableInfo = fdb._TableInfo.loc[name]
        self._FactorInfo = fdb._FactorInfo.loc[name]
        self._Connection = fdb._Connection
        self._IndexName = fdb.InnerPrefix+name
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        Fields = self._FactorInfo.index.tolist()
        self.add_trait("DTField", Enum(*Fields, arg_type="SingleOption", label="时点字段", order=3))
        self.add_trait("IDField", Enum(*Fields, arg_type="SingleOption", label="ID字段", order=4))
    @property
    def FactorNames(self):
        return sorted(self._FactorInfo.index)
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            return self._FactorInfo["DataType"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        IDKeyword = (f"{IDField}.keyword" if self._FactorInfo.loc[IDField, "Keyword"] else IDField)
        Query = {"bool": {"filter": [{"exists": {"field": IDField}}]}}
        if ifactor_name is not None:
            Query["bool"]["filter"].append({"exists": {"field": ifactor_name}})
        if idt is not None:
            DTField = args.get("时点字段", self.DTField)
            Query["bool"]["filter"].append({"term": {DTField: idt}})
        Rslt = self._Connection.search(index=self._IndexName, query=Query, _source=[IDField], collapse={"field": IDKeyword}, sort=[{IDKeyword: {"order": "asc"}}])
        return [r["_source"][IDField] for r in Rslt["hits"]["hits"]]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTField = args.get("时点字段", self.DTField)
        Query = {"bool": {"filter": [{"exists": {"field": DTField}}]}}
        if ifactor_name is not None:
            Query["bool"]["filter"].append({"exists": {"field": ifactor_name}})
        if iid is not None:
            IDField = args.get("ID字段", self.IDField)
            IDKeyword = (f"{IDField}.keyword" if self._FactorInfo.loc[IDField, "Keyword"] else IDField)
            Query["bool"]["filter"].append({"term": {IDKeyword: iid}})
        if (start_dt is not None) or (end_dt is not None):
            Range = {"range": {DTField: {}}}
            if start_dt is not None:
                Range["range"][DTField]["gte"] = start_dt
            if end_dt is not None:
                Range["range"][DTField]["lte"] = end_dt
            Query["bool"]["filter"].append(Range)
        Aggs = {"qs_dt_count": {"terms": {"field": DTField}}}
        Rslt = self._Connection.search(index=self._IndexName, query=Query, size=0, aggs=Aggs)
        return [dt.datetime.strptime(r["key_as_string"], "%Y-%m-%dT%H:%M:%S.%fZ") for r in Rslt["aggregations"]["qs_dt_count"]["buckets"]]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ArgConditionGroup = {}
        ArgNames = self.ArgNames
        ArgNames.remove("回溯天数")
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
        DTField = args.get("时点字段", self.DTField)
        IDField = args.get("ID字段", self.IDField)
        IDKeyword = (f"{IDField}.keyword" if self._FactorInfo.loc[IDField, "Keyword"] else IDField)
        Query = {"bool": {"filter": [{"exists": {"field": DTField}}]}}
        Query["bool"]["filter"].append({"terms": {IDKeyword: ids}})
        Query["bool"]["filter"].append({"range": {DTField: {"lt": end_date}}})
        FilterConds = args.get("筛选条件", self.FilterCondition)
        if FilterConds: Query["bool"]["filter"] += FilterConds
        Aggs = {"qs_code_group": {"terms": {"field": IDKeyword}, "aggs": {"qs_dt_max": {"max": {"field": DTField}}}}}
        Rslt = self._Connection.search(index=self._IndexName, query=Query, size=0, aggs=Aggs)
        Query = {"bool": {"should": []}}
        for iRslt in Rslt["aggregations"]["qs_code_group"]["buckets"]:
            Query["bool"]["should"].append({"bool": {"must": [{"term": {DTField: iRslt["qs_dt_max"]["value_as_string"]}}, {"term": {IDKeyword: iRslt["key"]}}]}})
        RawData = self._Connection.search(index=self._IndexName, query=Query, _source=[IDField, DTField]+factor_names)
        RawData = pd.DataFrame(data=(iData["_source"] for iData in RawData["hits"]["hits"]))
        if RawData.shape[1]==0: return pd.DataFrame(columns=[DTField, IDField]+factor_names)
        else: return RawData
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDField = args.get("ID字段", self.IDField)
        IDKeyword = (f"{IDField}.keyword" if self._FactorInfo.loc[IDField, "Keyword"] else IDField)
        DTField = args.get("时点字段", self.DTField)
        LookBack = args.get("回溯天数", self.LookBack)
        if dts is not None:
            dts = sorted(dts)
            StartDT, EndDT = dts[0], dts[-1]
            if not np.isinf(LookBack): StartDT -= dt.timedelta(LookBack)
        else:
            StartDT = EndDT = None
        Query = {"bool": {"filter": [{"exists": {"field": DTField}}]}}
        if args.get("预筛选ID", self.PreFilterID):
            Query["bool"]["filter"].append({"terms": {IDKeyword: ids}})
        else:
            Query["bool"]["filter"].append({"exists": {"field": IDField}})
        if (StartDT is not None) or (EndDT is not None):
            Range = {"range": {DTField: {}}}
            if StartDT is not None:
                Range["range"][DTField]["gte"] = StartDT
            if EndDT is not None:
                Range["range"][DTField]["lte"] = EndDT
            Query["bool"]["filter"].append(Range)
        FilterConds = args.get("筛选条件", self.FilterCondition)
        if FilterConds: Query["bool"]["filter"] += FilterConds
        RawData = pd.DataFrame(self._FactorDB.search(index=self._IndexName, query=Query, sort=[{IDKeyword: {"order": "asc"}}, {DTField: {"order": "asc"}}], only_source=True, _source=[DTField, IDField]+factor_names))
        #RawData = self._Connection.search(index=self._IndexName, query=Query, _source=[DTField, IDField]+factor_names, sort=[{IDKeyword: {"order": "asc"}, DTField: {"order": "asc"}}])
        #RawData = pd.DataFrame(data=(iData["_source"] for iData in RawData["hits"]["hits"]))
        if RawData.shape[1]==0: RawData = pd.DataFrame(columns=[DTField, IDField]+factor_names)
        if (StartDT is not None) and np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData[DTField]==StartDT][IDField]))
            if NullIDs:
                NullRawData = self._genNullIDRawData(factor_names, list(NullIDs), StartDT, args=args)
                if NullRawData.shape[0]>0:
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
        RawData = RawData.sort_values(by=[DTField, IDField]).rename(columns={DTField: "QS_DT", IDField: "ID"})
        RawData["QS_DT"] = RawData["QS_DT"].apply(lambda d: dt.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S"))
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

class ElasticSearchDB(WritableFactorDB):
    """ElasticSearchDB"""
    Name = Str("ElasticSearchDB", arg_type="String", label="名称", order=-100)
    ConnectArgs = Dict(arg_type="Dict", label="连接参数", order=0)
    Connector = Enum("default", "elasticsearch", arg_type="SingleOption", label="连接器", order=1)
    IgnoreFields = ListStr([], arg_type="List", label="忽略字段", order=2)
    InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=3)
    FTArgs = Dict(label="因子表参数", arg_type="Dict", order=4)
    DTField = Str("datetime", arg_type="String", label="时点字段", order=5)
    IDField = Str("code", arg_type="String", label="ID字段", order=6)
    #SQLClient = Bool(False, arg_type="Bool", label="SQL接口", order=7)
    SearchRetryNum = Float(10, label="查询重试次数", arg_type="Float", order=8)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"ElasticSearchDBConfig.json" if config_file is None else config_file), **kwargs)
        self._TableInfo = pd.DataFrame()# DataFrame(index=[表名], columns=["DBTableName", "TableClass"])
        self._FactorInfo = pd.DataFrame()# DataFrame(index=[(表名,因子名)], columns=["DataType", "FieldType", "Keyword", "Supplementary", "Description"])
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
        if self.Connector in ("default", "elasticsearch"):
            try:
                self._Connection = Elasticsearch(**self.ConnectArgs)
            except Exception as e:
                Msg = ("'%s' 尝试使用 elasticsearch 连接(%s)elasticsearch 失败: %s" % (self.Name, str(self.ConnectArgs), str(e)))
                self._QS_Logger.error(Msg)
                raise e
            else:
                self._Connector = "elasticsearch"
        self._PID = os.getpid()
        return 0
    def connect(self):
        self._connect()
        nPrefix = len(self.InnerPrefix)
        self._TableInfo = []
        TableInfo = self._Connection.indices.get_settings(index=f"{self.InnerPrefix}*")
        for iTableName in TableInfo:
            self._TableInfo.append((iTableName[nPrefix:], iTableName, "WideTable"))
        self._TableInfo = pd.DataFrame(self._TableInfo, columns=["TableName", "DBTableName", "TableClass"]).set_index(["TableName"])
        self._FactorInfo = []
        FactorInfo = self._Connection.indices.get_mapping(index=f"{self.InnerPrefix}*")
        for iTableName in FactorInfo:
            iFactorInfo = FactorInfo[iTableName]["mappings"].get("properties", {})
            if iFactorInfo:
                self._FactorInfo.extend(((iTableName[nPrefix:], iFactorName, _TypeMapping.get(iInfo["type"], "object"), iInfo["type"], "keyword" in iInfo.get("fields", {})) for iFactorName, iInfo in iFactorInfo.items() if iFactorName not in self.IgnoreFields))
        self._FactorInfo = pd.DataFrame(self._FactorInfo, columns=["TableName", "FactorName", "DataType", "FieldType", "Keyword"])
        self._FactorInfo["DBFieldName"] = self._FactorInfo["FactorName"]
        self._FactorInfo["Supplementary"] = self._FactorInfo["Description"] = None
        self._FactorInfo = self._FactorInfo.set_index(["TableName", "FactorName"])
        return 0
    def disconnect(self):
        self._Connection.close()
        self._Connection = None
        return 0
    def search_scroll(self, index, query, sort, only_source=True, flattened=True, return_size=None, size=3000, scroll="10m", **kwargs):
        if self._Connection is None:
            Msg = ("'%s' 调用 search_scroll 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        if return_size is not None: size = min(size, return_size)
        Body = {"query": query, "size": size, "sort": sort}
        Body.update(kwargs)
        if "params" in Body:
            kwargs = {"params": Body.pop("params")}
            if ("filter_path" in kwargs["params"]) and ("_scroll_id" not in kwargs["params"]["filter_path"]):
                kwargs["params"]["filter_path"] += ",_scroll_id"
        iRslt = self._Connection.search(body=Body, scroll=scroll, **kwargs)
        #iRslt = self._try_search(self.SearchRetryNum, body=Body, scroll=scroll, **kwargs)
        ScrollID = iRslt.pop("_scroll_id")
        iReturnedNum = 0
        while iRslt and iRslt["hits"]["hits"]:
            iRslt = iRslt["hits"]["hits"]
            if return_size is not None:
                iRslt = iRslt[:return_size-iReturnedNum]
            if only_source:
                for ijRslt in iRslt: yield ijRslt.get("_source", {})
            elif flattened:
                for ijRslt in iRslt:
                    ijRslt.update(ijRslt.pop("_source", {}))
                    yield ijRslt
            else:
                for ijRslt in iRslt: yield ijRslt
            if len(iRslt)<size: break
            if return_size is not None:
                iReturnedNum += len(iRslt)
                if iReturnedNum>=return_size: break
            iRslt = self._Connection.scroll(scroll_id=ScrollID, scroll=scroll, **kwargs)
            ScrollID = iRslt.pop("_scroll_id")
        self.Connection.clear_scroll(scroll_id=ScrollID)
    def _try_search(self, try_num, **kwargs):
        iRetryNum = 0
        while iRetryNum<try_num:
            try:
                return self._Connection.search(**kwargs)
            except ConnectionTimeout as e:
                SleepTime = 0.05 + (iRetryNum % 10) / 10.0
                if iRetryNum % 10==0:
                    self._QS_Logger.warning("ElasticSearchDB search failed: %s, try again %s seconds later!" % (str(e), SleepTime))
                iRetryNum += 1
                time.sleep(SleepTime)
        Msg = "ElasticSearchDB search failed after trying %d times" % (iRetryNum)
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)
    def search(self, index, query, sort, only_source=True, flattened=True, return_size=None, size=3000, keep_alive="10m", **kwargs):
        if self._Connection is None:
            Msg = ("'%s' 调用 search 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        if return_size is not None: size = min(size, return_size)
        SortFields = [tuple(iSort.keys())[0].split(".")[0] for iSort in sort]
        PIT = self._Connection.open_point_in_time(index=index, keep_alive=keep_alive)
        #iRslt = self._try_search(self.SearchRetryNum, query=query, size=size, sort=sort, pit={"keep_alive": keep_alive, "id": PIT["id"]}, **kwargs)
        iRslt = self._Connection.search(query=query, size=size, sort=sort, pit={"keep_alive": keep_alive, "id": PIT["id"]}, **kwargs)
        iReturnedNum = 0
        while iRslt and iRslt["hits"]["hits"]:
            iRslt = iRslt["hits"]["hits"]
            if return_size is not None:
                iRslt = iRslt[:return_size-iReturnedNum]
            iLastRslt = iRslt[-1]
            iSorts = [iLastRslt[iField] if iField in iLastRslt else iLastRslt["_source"][iField]  for iField in SortFields]
            if only_source:
                for ijRslt in iRslt: yield ijRslt.get("_source", {})
            elif flattened:
                for ijRslt in iRslt:
                    ijRslt.update(ijRslt.pop("_source", {}))
                    yield ijRslt
            else:
                for ijRslt in iRslt: yield ijRslt
            if len(iRslt)<size: break
            if return_size is not None:
                iReturnedNum += len(iRslt)
                if iReturnedNum>=return_size: break
            iRslt = self._Connection.search(query=query, size=size, sort=sort, pit={"keep_alive": keep_alive, "id": PIT["id"]}, search_after=iSorts, **kwargs)
            #iRslt = self._try_search(self.SearchRetryNum, query=query, size=size, sort=sort, pit={"keep_alive": keep_alive, "id": PIT["id"]}, search_after=iSorts, **kwargs)
        self._Connection.close_point_in_time(body=PIT)
    def fetchall(self, sql_str):# SQL 相关, SQL 不支持 DISTINCT
        if self._Connection is None:
            Msg = ("'%s' 调用 fetchall 失败: 数据库尚未连接!" % (self.Name,))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if os.getpid()!=self._PID: self._connect()# 如果进程号发生变化, 重连
        try:# 连接断开后重连
            Cursor = client.SqlClient(self._Connection)
        except:
            self._connect()
            Cursor = client.SqlClient(self._Connection)
        Rslt = Cursor.query(body={"query": sql_str})
        return Rslt["rows"]
    @property
    def TableNames(self):
        return sorted(self._TableInfo.index)
    def _initFTArgs(self, table_name, args):
        if table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 getTable 错误: 不存在因子表: '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        Args = self.FTArgs.copy()
        Args.update(args)
        Args.setdefault("时点字段", self.DTField)
        Args.setdefault("ID字段", self.IDField)
        return Args
    def getTable(self, table_name, args={}):
        Args = self._initFTArgs(table_name=table_name, args=args)
        TableClass = Args.get("因子表类型", self._TableInfo.loc[table_name, "TableClass"])
        #if args.get("SQL接口", self.SQLClient):
            #TableClass = globals().get("SQL_"+TableClass, None)
            #if TableClass is not None:
                #FT = TableClass(name=table_name, fdb=self, sys_args=Args, table_prefix="", table_info=self._TableInfo.loc[table_name], factor_info=self._FactorInfo.loc[table_name])
                #FT._DTFormat_WithTime = "'%Y-%m-%dT%H:%M:%S'"# 修改了私有变量, TODO
                #return FT
        #else:
        TableClass = globals().get("_"+TableClass, None)
        if TableClass is not None:
            return TableClass(name=table_name, fdb=self, sys_args=Args, logger=self._QS_Logger)
        Msg = ("因子库 '%s' 调用方法 getTable 错误: 不支持的因子表类型: '%s'" % (self.Name, TableClass))
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableInfo.index:
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 不存在因子表 '%s'!" % (self.Name, old_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableInfo.index):
            Msg = ("因子库 '%s' 调用方法 renameTable 错误: 新因子表名 '%s' 已经存在于库中!" % (self.Name, new_table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        Mappings = self._Connection.indices.get_mapping(index=self.InnerPrefix+old_table_name)[self.InnerPrefix+old_table_name]["mappings"]
        self._Connection.indices.create(index=self.InnerPrefix+new_table_name, mappings=Mappings)
        self._Connection.reindex(body={"source": {"index": self.InnerPrefix+old_table_name}, "dest": {"index": self.InnerPrefix+new_table_name}})
        self._Connection.indices.delete(index=self.InnerPrefix+old_table_name)
        self._TableInfo = self._TableInfo.rename(index={old_table_name: new_table_name})
        self._FactorInfo = self._FactorInfo.rename(index={old_table_name: new_table_name}, level=0)
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableInfo.index: return 0
        self._Connection.indices.delete(index=self.InnerPrefix+table_name)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._TableInfo = self._TableInfo.loc[TableNames]
        self._FactorInfo = self._FactorInfo.loc[TableNames]
        return 0
    # 创建表, field_types: {字段名: 数据类型(数据库内)}
    def createTable(self, table_name, field_types):
        if not self._Connection.indices.exists(index=self.InnerPrefix+table_name):
            Mappings = {"properties": {self.DTField: {"type": "date"}, self.IDField: {"type": "keyword"}}}
            Mappings["properties"].update({iFactor: {"type": iFieldType} for iFactor, iFieldType in field_types.items()})
            self._Connection.indices.create(index=self.InnerPrefix+table_name, mappings=Mappings)
        self._TableInfo = self._TableInfo.append(pd.Series([self.InnerPrefix+table_name, "WideTable"], index=["DBTableName", "TableClass"], name=table_name))
        NewFactorInfo = pd.DataFrame(((table_name, iFactor, _TypeMapping[iFieldType], iFieldType, False) for iFactor, iFieldType in field_types.items()), columns=["TableName", "FactorName", "DataType", "FieldType", "Keyword"]).set_index(["TableName", "FactorName"])
        self._FactorInfo = self._FactorInfo.append(NewFactorInfo).sort_index()
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
        #self._Connection.update_by_query(index=self.InnerPrefix+table_name, 
            #body={"script": f'ctx._source.{new_factor_name} = ctx._source.remove("{old_factor_name}")', "query": {"bool": {"must": [{"exists": {"field": old_factor_name}}]}}})
        # 备份原索引
        Mappings = self._Connection.indices.get_mapping(index=self.InnerPrefix+table_name)[self.InnerPrefix+table_name]["mappings"]
        BakTableName = genAvailableName(table_name, self._TableInfo.index)
        self._Connection.indices.create(index=self.InnerPrefix+BakTableName, mappings=Mappings)
        self._Connection.reindex(body={"source": {"index": self.InnerPrefix+table_name}, "dest": {"index": self.InnerPrefix+BakTableName}})
        # 删除原索引
        self._Connection.indices.delete(index=self.InnerPrefix+table_name)
        # 修改名称并恢复索引
        Mappings["properties"][new_factor_name] = Mappings["properties"].pop(old_factor_name)
        self._Connection.indices.create(index=self.InnerPrefix+table_name, mappings=Mappings)
        self._Connection.reindex(body={"source": {"index": self.InnerPrefix+BakTableName}, "dest": {"index": self.InnerPrefix+table_name}, "script": {"inline": f'ctx._source.{new_factor_name} = ctx._source.remove("{old_factor_name}")'}})
        # 删除临时索引
        self._Connection.indices.delete(index=self.InnerPrefix+BakTableName)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].rename(index={old_factor_name: new_factor_name}, level=1))
        return 0
    def deleteFactor(self, table_name, factor_names):
        if (not factor_names) or (table_name not in self._TableInfo.index): return 0
        FactorIndex = self._FactorInfo.loc[table_name].index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        #for iFactorName in factor_names:
            #self._Connection.update_by_query(index=self.InnerPrefix+table_name, 
                #body={"script": f'ctx._source.remove("{iFactorName}")', "query": {"bool": {"must": [{"exists": {"field": iFactorName}}]}}})
        # 备份原索引
        Mappings = self._Connection.indices.get_mapping(index=self.InnerPrefix+table_name)[self.InnerPrefix+table_name]["mappings"]
        BakTableName = genAvailableName(table_name, self._TableInfo.index)
        self._Connection.indices.create(index=self.InnerPrefix+BakTableName, mappings=Mappings)
        self._Connection.reindex(body={"source": {"index": self.InnerPrefix+table_name}, "dest": {"index": self.InnerPrefix+BakTableName}})        
        # 删除原索引
        self._Connection.indices.delete(index=self.InnerPrefix+table_name)
        # 修改名称并恢复索引
        for iFactorName in factor_names: Mappings["properties"].pop(iFactorName)
        self._Connection.indices.create(index=self.InnerPrefix+table_name, mappings=Mappings)
        Script = "\n".join(f'ctx._source.remove("{iFactorName}");' for iFactorName in factor_names)
        self._Connection.reindex(body={"source": {"index": self.InnerPrefix+BakTableName}, "dest": {"index": self.InnerPrefix+table_name}, "script": {"inline": Script}})
        # 删除临时索引
        self._Connection.indices.delete(index=self.InnerPrefix+BakTableName)
        TableNames = self._TableInfo.index.tolist()
        TableNames.remove(table_name)
        self._FactorInfo = self._FactorInfo.loc[TableNames].append(self._FactorInfo.loc[[table_name]].loc[FactorIndex])
        return 0
    # 增加因子，field_types: {字段名: 数据类型(数据库内)}
    def addFactor(self, table_name, field_types):
        if table_name not in self._TableInfo.index: return self.createTable(table_name, field_types)
        self._Connection.indices.put_mapping(body={"properties": {iFactor: {"type": iFieldType} for iFactor, iFieldType in field_types.items()}}, index=self.InnerPrefix+table_name)
        NewFactorInfo = pd.DataFrame(((table_name, iFactor, _TypeMapping[iFieldType], iFieldType, False) for iFactor, iFieldType in field_types.items()), columns=["TableName", "FactorName", "DataType", "FieldType", "Keyword"]).set_index(["TableName", "FactorName"])
        self._FactorInfo = self._FactorInfo.append(NewFactorInfo).sort_index()
        return 0
    def deleteData(self, table_name, ids=None, dts=None):
        if (ids is None) and (dts is None):
            Body = {"query": {"match_all": {}}}
        else:
            Body = {"query": {"bool": {"filter": []}}}
            if ids is not None:
                Body["query"]["bool"]["filter"].append({"terms": {f"{self.IDField}.keyword": ids}})
            if dts is not None:
                Body["query"]["bool"]["filter"].append({"terms": {self.DTField: dts}})
        self._Connection.delete_by_query(index=self.InnerPrefix+table_name, body=Body)
        return 0
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._TableInfo.index:
            FieldTypes = {iFactorName:_identifyDataType(data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            self.createTable(table_name, field_types=FieldTypes)
        else:
            NewFactorNames = data.items.difference(self._FactorInfo.loc[table_name].index).tolist()
            if NewFactorNames:
                FieldTypes = {iFactorName:_identifyDataType(data.iloc[i].dtypes) for i, iFactorName in enumerate(NewFactorNames)}
                self.addFactor(table_name, FieldTypes)
            if if_exists=="update":
                OldFactorNames = self._FactorInfo.loc[table_name].index.difference(data.items).tolist()
                if OldFactorNames:
                    OldData = self.getTable(table_name).readData(factor_names=OldFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
                    for iFactorName in OldFactorNames: data[iFactorName] = OldData[iFactorName]
            else:
                AllFactorNames = self._FactorInfo.loc[table_name].index.tolist()
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
        NewData.columns = [self.DTField, self.IDField] + NewData.columns[2:].tolist()
        helpers.bulk(client=self._Connection, actions=({"_op_type": "index", "_index": self.InnerPrefix+table_name, "_source": NewData.iloc[i].to_dict()} for i in range(NewData.shape[0])))
        return 0