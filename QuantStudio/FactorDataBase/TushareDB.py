# coding=utf-8
"""基于 tushare 的因子库(TODO)"""
import os
from collections import OrderedDict
import datetime as dt

import numpy as np
import pandas as pd
import tushare as ts
from traits.api import Enum, Int, Str, Function

from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.MathFun import CartesianProduct
from QuantStudio import __QS_Error__, __QS_LibPath__, __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import updateInfo

class _TSTable(FactorTable):
    def getMetaData(self, key=None, args={}):
        TableInfo = self._FactorDB._TableInfo.loc[self.Name]
        if key is None:
            return TableInfo
        else:
            return TableInfo.get(key, None)
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        if key=="DataType":
            if hasattr(self, "_DataType"): return self._DataType.loc[factor_names]
            MetaData = FactorInfo["DataType"].loc[factor_names]
            for i in range(MetaData.shape[0]):
                iDataType = MetaData.iloc[i].lower()
                if iDataType.find("str")!=-1: MetaData.iloc[i] = "string"
                else: MetaData.iloc[i] = "double"
            return MetaData
        elif key=="Description": return FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args),
                                 "Description":self.getFactorMetaData(factor_names, key="Description", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))


class _CalendarTable(_TSTable):
    """交易日历因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    # 返回交易所列表
    # 忽略 ifactor_name, idt
    def getID(self, ifactor_name=None, idt=None, args={}):
        return  self._FactorDB._TableInfo.loc[self.Name, "Supplementary"].split(",")
    # 返回交易所为 iid 的交易日列表
    # 如果 iid 为 None, 将返回表中有记录数据的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        DateField = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[self._DateField]
        if start_dt is None: start_dt = dt.date(1900, 1, 1)
        start_dt = start_dt.strftime("%Y%m%d")
        if end_dt is None: end_dt = dt.date.today()
        end_dt = end_dt.strftime("%Y%m%d")
        if iid is None: iid="SSE"
        Dates = self._FactorDB._ts.query(DBTableName, exchange=iid, start_date=start_dt, end_date=end_dt, fields=DateField, is_open="1")
        return [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8])) for iDate in Dates[DateField].values]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo['DBFieldName'].loc[self.Name].loc[[self._DateField, self._IDField]+factor_names]
        Fields = FieldDict.tolist()
        StartDate, EndDate = dts[0].strftime("%Y%m%d"), dts[-1].strftime("%Y%m%d")
        RawData = None
        for iID in ids:
            iData = self._FactorDB._ts.query(DBTableName, exchange=iID, start_date=StartDate, end_date=EndDate, fields=Fields)
            iData.index = [iID] * iData.shape[0]
            if RawData is None: RawData = iData
            else: RawData = RawData.append(iData)
        if RawData is None: return pd.DataFrame(columns=["ID", "日期"]+factor_names)
        RawData.index, RawData.columns = np.arange(RawData.shape[0]), ["ID", "日期"]+factor_names
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Data.major_axis]
        return Data.loc[:, dts, ids]

class _FeatureTable(_TSTable):
    """特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        ConditionField = FactorInfo[pd.notnull(FactorInfo["Supplementary"])]
        for i, iCondition in enumerate(ConditionField.index):
            self.add_trait("Condition"+str(i), Enum(*ConditionField["Supplementary"].iloc[i].split(","), arg_type="String", label=iCondition, order=i))
    def getID(self, ifactor_name=None, idt=None, args={}):
        RawData = self.__QS_prepareRawData__(factor_names=[], ids=[], dts=[], args=args)
        return sorted(RawData["ID"])
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        Fields = [self._IDField]+factor_names
        DBFields = FactorInfo["DBFieldName"].loc[Fields].tolist()
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        RawData = pd.DataFrame(columns=DBFields)
        FieldStr = ",".join(DBFields)
        ConditionField = FactorInfo[pd.notnull(FactorInfo["Supplementary"])]
        if ConditionField.shape[0]>0:
            SingleCondition, MultiCondition = {}, OrderedDict()
            for i, iCondition in enumerate(ConditionField.index):
                iConditionValue = args.get(iCondition, self[iCondition])
                if iConditionValue=="All": MultiCondition[ConditionField["DBFieldName"].iloc[i]] = ConditionField["Supplementary"].iloc[i].split(",")[1:]
                else: SingleCondition[ConditionField["DBFieldName"].iloc[i]] = iConditionValue
            if MultiCondition:
                RawData = None
                MultiCondition, MultiConditionValue = list(MultiCondition.keys()), CartesianProduct(list(MultiCondition.values()))
                for iMultiConditionValue in MultiConditionValue:
                    SingleCondition.update(dict(zip(MultiCondition, iMultiConditionValue)))
                    if RawData is None: RawData = self._FactorDB._ts.query(DBTableName, fields=FieldStr, **SingleCondition)
                    else: RawData = RawData.append(self._FactorDB._ts.query(DBTableName, fields=FieldStr, **SingleCondition))
            else: RawData = self._FactorDB._ts.query(DBTableName, fields=FieldStr, **SingleCondition)
        else:
            RawData = self._FactorDB._ts.query(DBTableName, fields=FieldStr)
        RawData = RawData.loc[:, DBFields]
        RawData.columns = ["ID"]+factor_names
        RawData["ID"] = self._FactorDB.DBID2ID(RawData["ID"])
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data = raw_data.set_index(["ID"])
        if raw_data.index.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.loc[ids]
        return pd.Panel(raw_data.values.T.reshape((raw_data.shape[1], raw_data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=ids, minor_axis=dts).swapaxes(1, 2)

class _MarketTable(_TSTable):
    """行情因子表"""
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=0)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        IDInfo = FactorInfo[FactorInfo["FieldType"]=="ID"]
        if IDInfo.shape[0]==0: self._IDField = self._IDType = None            
        else: self._IDField, self._IDType = IDInfo.index[0], IDInfo["Supplementary"].iloc[0]
        self._DateFields = FactorInfo[FactorInfo["FieldType"]=="Date"].index.tolist()
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        self.add_trait("DateField", Enum(*self._DateFields, arg_type="SingleOption", label="日期字段", order=1))
        DefaultDateField = FactorInfo[(FactorInfo["Supplementary"]=="DefaultDate") & (FactorInfo["FieldType"]=="Date")]
        if DefaultDateField.shape[0]>0: self.DateField = DefaultDateField.index[0]
        else: self.DateField = self._DateFields[0]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+2))
            self[iCondition] = str(FactorInfo.loc[iCondition, "Supplementary"])
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()+self._DateFields
    def getID(self, ifactor_name=None, idt=None, args={}):
        if self._IDField is None: return ["000000.HST"]
        TableType = self._FactorDB._TableInfo.loc[self.Name, "Supplementary"]
        if TableType=="A股": return self._FactorDB.getStockID(index_id="全体A股", is_current=False)
        elif TableType=="期货": return self._FactorDB.getFutureID(future_code=None, is_current=False)
        return []
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return self._FactorDB.getTradeDay(start_date=start_dt, end_date=end_dt, exchange="", output_type="datetime")
    def __QS_genGroupInfo__(self, factors, operation_mode):
        DateConditionGroup = {}
        for iFactor in factors:
            iDateConditions = (iFactor.DateField, ";".join([iArgName+":"+iFactor[iArgName] for iArgName in iFactor.ArgNames if iArgName!="回溯天数"]))
            if iDateConditions not in DateConditionGroup:
                DateConditionGroup[iDateConditions] = {"FactorNames":[iFactor.Name], 
                                                       "RawFactorNames":{iFactor._NameInFT}, 
                                                       "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                       "args":iFactor.Args.copy()}
            else:
                DateConditionGroup[iDateConditions]["FactorNames"].append(iFactor.Name)
                DateConditionGroup[iDateConditions]["RawFactorNames"].add(iFactor._NameInFT)
                DateConditionGroup[iDateConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], DateConditionGroup[iDateConditions]["StartDT"])
                DateConditionGroup[iDateConditions]["args"]["回溯天数"] = max(DateConditionGroup[iDateConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iDateConditions in DateConditionGroup:
            StartInd = operation_mode.DTRuler.index(DateConditionGroup[iDateConditions]["StartDT"])
            Groups.append((self, DateConditionGroup[iDateConditions]["FactorNames"], list(DateConditionGroup[iDateConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], DateConditionGroup[iDateConditions]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        Fields = [self.DateField]+factor_names
        if self._IDField is not None: Fields.insert(0, self._IDField)
        DBFields = FactorInfo["DBFieldName"].loc[Fields].tolist()
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        RawData = pd.DataFrame(columns=DBFields)
        FieldStr = ",".join(DBFields)
        Conditions = {}
        for iConditionField in self._ConditionFields:
            iDataType = FactorInfo.loc[iConditionField, "DataType"]
            if iDataType=="str":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = args.get(iConditionField, self[iConditionField])
            elif iDataType=="float":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = float(args.get(iConditionField, self[iConditionField]))
            elif iDataType=="int":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = int(args.get(iConditionField, self[iConditionField]))
        StartDate, EndDate = StartDate.strftime("%Y%m%d"), EndDate.strftime("%Y%m%d")
        if self._IDField is None:
            RawData = self._FactorDB._ts.query(DBTableName, start_date=StartDate, end_date=EndDate, fields=FieldStr, **Conditions)
            RawData.insert(0, "ID", "000000.HST")
            DBFields.insert(0, "ID")
        elif pd.isnull(self._IDType):
            for iID in self._FactorDB.ID2DBID(ids):
                Conditions[DBFields[0]] = iID
                RawData = RawData.append(self._FactorDB._ts.query(DBTableName, start_date=StartDate, end_date=EndDate, fields=FieldStr, **Conditions))
        elif self._IDType=="Non-Finite":
            RawData = self._FactorDB._ts.query(DBTableName, start_date=StartDate, end_date=EndDate, fields=FieldStr, **Conditions)
        RawData = RawData.loc[:, DBFields]
        RawData.columns = ["ID", "日期"]+factor_names
        RawData["ID"] = self._FactorDB.DBID2ID(RawData["ID"])
        return RawData.sort_values(by=["ID", "日期"])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Data.major_axis]
        if Data.minor_axis.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]

# 公告信息表, 表结构特征:
# 公告日期, 表示获得信息的时点;
# 截止日期, 表示信息有效的时点, 该字段可能没有;
# 如果存在截止日期, 以截止日期和公告日期的最大值作为数据填充的时点; 如果不存在截止日期, 以公告日期作为数据填充的时点;
# 数据填充时点和 ID 不能唯一标志一行记录, 对于每个 ID 每个数据填充时点可能存在多个数据, 将所有的数据以 list 组织, 如果算子参数不为 None, 以该算子作用在数据 list 上的结果为最终填充结果, 否则以数据 list 填充;
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class _AnnTable(_TSTable):
    """公告信息表"""
    #ANNDate = Enum(None, arg_type="SingleOption", label="公告日期", order=0)
    Operator = Function(None, arg_type="Function", label="算子", order=1)
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        IDInfo = FactorInfo[FactorInfo["FieldType"]=="ID"]
        if IDInfo.shape[0]==0: self._IDField = self._IDType = None            
        else: self._IDField, self._IDType = IDInfo.index[0], IDInfo["Supplementary"].iloc[0]
        self._AnnDateField = FactorInfo[FactorInfo["FieldType"]=="ANNDate"].index[0]# 公告日期
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+2))
            self[iCondition] = str(FactorInfo.loc[iCondition, "Supplementary"])
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()+[self._AnnDateField]
    def getID(self, ifactor_name=None, idt=None, args={}):
        if self._IDField is None: return ["000000.HST"]
        TableType = self._FactorDB._TableInfo.loc[self.Name, "Supplementary"]
        if TableType=="A股": return self._FactorDB.getStockID(index_id="全体A股", is_current=False)
        elif TableType=="期货": return self._FactorDB.getFutureID(future_code=None, is_current=False)
        return []
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return self._FactorDB.getTradeDay(start_date=start_dt, end_date=end_dt, exchange="", output_type="datetime")
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+iFactor[iArgName] for iArgName in iFactor.ArgNames if iArgName not in ("回溯天数", "算子")])
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {"FactorNames":[iFactor.Name], 
                                               "RawFactorNames":{iFactor._NameInFT}, 
                                               "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                               "args":iFactor.Args.copy()}
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
                ConditionGroup[iConditions]["args"]["回溯天数"] = max(ConditionGroup[iConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        Fields = [self._AnnDateField]+factor_names
        if self._IDField is not None: Fields.insert(0, self._IDField)
        DBFields = FactorInfo["DBFieldName"].loc[Fields].tolist()
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        RawData = pd.DataFrame(columns=DBFields)
        FieldStr = ",".join(DBFields)
        Conditions = {}
        for iConditionField in self._ConditionFields:
            iDataType = FactorInfo.loc[iConditionField, "DataType"]
            if iDataType=="str":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = args.get(iConditionField, self[iConditionField])
            elif iDataType=="float":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = float(args.get(iConditionField, self[iConditionField]))
            elif iDataType=="int":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = int(args.get(iConditionField, self[iConditionField]))
        StartDate, EndDate = StartDate.strftime("%Y%m%d"), EndDate.strftime("%Y%m%d")
        if self._IDField is None:
            RawData = self._FactorDB._ts.query(DBTableName, start_date=StartDate, end_date=EndDate, fields=FieldStr, **Conditions)
            RawData.insert(0, "ID", "000000.HST")
            DBFields.insert(0, "ID")
        elif pd.isnull(self._IDType):
            for iID in self._FactorDB.ID2DBID(ids):
                Conditions[DBFields[0]] = iID
                iData = self._FactorDB._ts.query(DBTableName, start_date=StartDate, end_date=EndDate, fields=FieldStr, **Conditions)
                while iData.shape[0]>0:
                    RawData = RawData.append(iData)
                    iEndDate = iData[DBFields[1]].min()
                    iData = self._FactorDB._ts.query(DBTableName, start_date=StartDate, end_date=iEndDate, fields=FieldStr, **Conditions)
                    iEndDateData = iData[iData[DBFields[1]]==iEndDate]
                    iRawEndDateMask = (RawData[DBFields[1]]==iEndDate)
                    if iEndDateData.shape[0]>iRawEndDateMask.sum():
                        RawData = RawData[~iRawEndDateMask]
                        RawData = RawData.append(iEndDateData)
                    iData = iData[iData[DBFields[1]]<iEndDate]
        elif self._IDType=="Non-Finite":
            RawData = self._FactorDB._ts.query(DBTableName, start_date=StartDate, end_date=EndDate, fields=FieldStr, **Conditions)
        RawData = RawData.loc[:, DBFields]
        RawData.columns = ["ID", "日期"]+factor_names
        RawData["ID"] = self._FactorDB.DBID2ID(RawData["ID"])
        return RawData.sort_values(by=["ID", "日期"])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Operator = args.get("算子", self.Operator)
        if Operator is None: Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        Data = pd.Panel(Data).loc[factor_names, :, ids]
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Data.major_axis]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]

class TushareDB(FactorDB):
    """tushare"""
    Token = Str("", label="Token", arg_type="String", order=0)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"TushareDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "TushareDB"
        self._ts = None
        self._InfoFilePath = __QS_LibPath__+os.sep+"TushareDBInfo.hdf5"# 数据库信息文件路径
        self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"TushareDBInfo.xlsx"# 数据库信息源文件路径
        self._TableInfo, self._FactorInfo = updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger)
    def connect(self):
        ts.set_token(self.Token)
        self._ts = ts.pro_api()
        return 0
    def disconnect(self):
        self._ts = None
        return 0
    def isAvailable(self):
        return (self._ts is not None)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ts"] = self.isAvailable()
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._ts:
            self._ts = None
            self.connect()
        else:
            self._ts = None
    @property
    def TableNames(self):
        if self._TableInfo is not None: return self._TableInfo.index.tolist()
        else: return []
    def getTable(self, table_name, args={}):
        TableClass = self._TableInfo.loc[table_name, "TableClass"]
        return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=args, logger=self._QS_Logger)")
    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if start_date is None: start_date = dt.date(1900, 1, 1)
        start_date = start_date.strftime("%Y%m%d")
        if end_date is None: end_date = dt.date.today()
        end_date = end_date.strftime("%Y%m%d")
        Dates = self._ts.query("trade_cal", exchange=exchange, start_date=start_date, end_date=end_date, fields="cal_date", is_open="1")
        if kwargs.get("output_type", "date")=="date": return [dt.datetime.strptime(iDate, "%Y%m%d").date() for iDate in Dates["cal_date"]]
        else: return [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Dates["cal_date"]]
    # 将 QuantStudio 的 ID 转化成数据库内部 ID
    def ID2DBID(self, ids):
        return pd.Series(ids).str.replace(".CFE", ".CFX").str.replace(".CZC", ".ZCE").tolist()
    # 将数据库内部 ID 转换成 QuantStudio 的 ID
    def DBID2ID(self, ids):
        return pd.Series(ids).str.replace(".CFX", ".CFE").str.replace(".ZCE", ".CZC").tolist()
    # 获取指定日 date 的全体 A 股 ID
    # date: 指定日, datetime.date
    # is_current: False 表示上市日在指定日之前的 A 股, True 表示上市日在指定日之前且尚未退市的 A 股
    def _getAllAStock(self, date, is_current=True):
        Data = self._ts.stock_basic(exchange="", list_status="L", fields="ts_code, list_date, delist_date")
        Data = Data.append(self._ts.stock_basic(exchange="", list_status="D", fields="ts_code, list_date, delist_date"))
        Data = Data.append(self._ts.stock_basic(exchange="", list_status="P", fields="ts_code, list_date, delist_date"))
        date = date.strftime("%Y%m%d")
        Data = Data[Data["list_date"]<=date]
        if is_current: Data = Data[pd.isnull(Data["delist_date"]) | (Data["delist_date"]>date)]
        return sorted(Data["ts_code"])
    # 获取指定日 date 指数 index_id 的成份股 ID
    # index_id: 指数 ID, 默认值 "全体A股"
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示进入指数的日期在指定日之前的成份股, True 表示进入指数的日期在指定日之前且尚未剔出指数的 A 股, index_id 不是 "全体A股" 时不支持 False
    def getStockID(self, index_id="全体A股", date=None, is_current=True):
        if date is None: date = dt.date.today()
        if index_id=="全体A股": return self._getAllAStock(date=date, is_current=is_current)
        if not is_current: raise __QS_Error__("不支持提取 '%s' 的历史成分股!" % index_id)
        return self._ts.index_weight(index_code=index_id, trade_date=date.strftime("%Y%m%d"), fields="con_code")["con_code"].tolist()
    # 给定期货代码 future_code, 获取指定日 date 的期货 ID
    # future_code: 期货代码(str)或者期货代码列表(list(str)), None 表示所有期货代码
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日在指定日之前的期货, True 表示上市日在指定日之前且尚未退市的期货
    # kwargs:
    # contract_type: 合约类型, 可选 "月合约", "连续合约", "所有", 默认值 "月合约"
    def getFutureID(self, future_code="IF", date=None, is_current=True, **kwargs):
        if date is None: date = dt.date.today()
        date = date.strftime("%Y%m%d")
        ContractType = kwargs.get("contract_type", "月合约")
        if ContractType=="月合约": fut_type = "1"
        elif ContractType=="连续合约": fut_type = "2"
        else: fut_type = ""
        Exchanges = ["CFFEX", "SHFE", "DCE", "CZCE", "INE"]
        if future_code:
            if isinstance(future_code, str):
                for iExchange in Exchanges:
                    Data = self._ts.fut_basic(exchange=iExchange, fut_type=fut_type, fields="ts_code, fut_code, list_date, delist_date")
                    Data = Data[Data["fut_code"]==future_code]
                    if Data.shape[0]>0: break
                else: raise __QS_Error__("未找到期货: '%s'!" % (future_code, ))
            else:
                Data = pd.DataFrame(columns=["ts_code", "list_date", "delist_date"])
                for iExchange in Exchanges:
                    iData = self._ts.fut_basic(exchange=iExchange, fut_type=fut_type, fields="ts_code, fut_code, list_date, delist_date")
                    iData = iData[iData["fut_code"].isin(future_code)]
                    Data = Data.append(iData)
        else:
            Data = pd.DataFrame(columns=["ts_code", "list_date", "delist_date"])
            for iExchange in Exchanges:
                Data = Data.append(self._ts.fut_basic(exchange=iExchange, fut_type=fut_type, fields="ts_code, list_date, delist_date"))
        Data = Data[(Data["list_date"]<=date) | pd.isnull(Data["list_date"])]
        if is_current: Data = Data[pd.isnull(Data["delist_date"]) | (Data["delist_date"]>date)]
        return self.DBID2ID(sorted(Data["ts_code"]))