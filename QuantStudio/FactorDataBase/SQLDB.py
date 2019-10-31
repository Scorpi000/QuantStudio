# coding=utf-8
"""基于 SQL 数据库的因子库"""
import re
import os
import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, File, Float, Bool, ListStr, on_trait_change, Either, Date

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.QSObjects import QSSQLObject
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable

def _identifyDataType(db_type, dtypes):
    if db_type!="sqlite3":
        if np.dtype("O") in dtypes.values: return "varchar(40)"
        else: return "double"
    else:
        if np.dtype("O") in dtypes.values: return "text"
        else: return "real"

def _adjustData(data, look_back, factor_names, ids, dts):
    if ids is not None:
        data = pd.Panel(data).loc[factor_names, :, ids]
    else:
        data = pd.Panel(data).loc[factor_names, :, :]
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
        data = pd.Panel(data).loc[factor_names]
    if dts is not None:
        return data.loc[:, dts]
    else:
        return data


class _WideTable(FactorTable):
    """SQLDB 宽因子表"""
    TableType = Enum("宽表", arg_type="SingleOption", label="因子表类型", order=0)
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=1)
    ValueType = Enum("scalar", "list", "scalar or list", arg_type="SingleOption", label="因子值类型", order=2)
    FilterCondition = Str("", arg_type="String", label="筛选条件", order=3)
    #DTField = Enum("datetime", arg_type="SingleOption", label="时点字段", order=4)
    #IDField = Enum("code", arg_type="SingleOption", label="ID字段", order=5)
    DT2Str = Bool(False, arg_type="Bool", label="时间转字符串", order=6)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DataType = fdb._TableFactorDict[name]
        self._DBDataType = fdb._TableFieldDataType[name]
        self._DBTableName = fdb.TablePrefix+fdb.InnerPrefix+name
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        Fields = ["datetime"] + self._DBDataType[self._DBDataType.str.contains("date")].index.tolist()
        self.add_trait("DTField", Enum(*Fields, arg_type="SingleOption", label="时点字段", order=4))
        StrMask = (self._DBDataType.str.contains("char") | self._DBDataType.str.contains("text"))
        Fields = ["code"] + self._DBDataType[StrMask].index.tolist()
        self.add_trait("IDField", Enum(*Fields, arg_type="SingleOption", label="ID字段", order=5))
    @property
    def FactorNames(self):
        return self._DataType.index.tolist()+["datetime", "code"]
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType": return self._DataType.append(pd.Series(["string", "string"], index=["datetime","code"])).loc[factor_names]
        if key is None: return pd.DataFrame(self._DataType.append(pd.Series(["string", "string"], index=["datetime","code"])).loc[factor_names], columns=["DataType"])
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+IDField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+IDField+" IS NOT NULL "
        if idt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+ifactor_name+" IS NOT NULL "
        SQLStr += "ORDER BY "+self._DBTableName+"."+IDField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+DTField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
        if iid is not None: SQLStr += "AND "+self._DBTableName+"."+IDField+"='"+iid+"' "
        if start_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+">='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+ifactor_name+" IS NOT NULL "
        SQLStr += "ORDER BY "+self._DBTableName+"."+DTField
        if self._FactorDB.DBType!="sqlite3": return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        else: return [dt.datetime.strptime(iRslt[0], "%Y-%m-%d %H:%M:%S.%f") for iRslt in self._FactorDB.fetchall(SQLStr)]
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
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        DT2Str = args.get("时间转字符串", self.DT2Str)
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAX("+DTField+") "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "WHERE "+DTField+"<'"+end_date.strftime("%Y-%m-%d:%H:%M:%S.%f")+"' "
        SubSQLStr += "AND ("+genSQLInCondition(IDField, ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SubSQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SubSQLStr += "GROUP BY "+IDField
        SQLStr = "SELECT "+DTField+", "
        SQLStr += IDField+", "
        for iField in factor_names:
            if iField=="datetime": iDBDataType = "datetime"
            elif iField=="code": iDBDataType = "varchar(40)"
            else: iDBDataType = self._FactorDB._TableFieldDataType[self._Name][iField]            
            if DT2Str and (iDBDataType.lower().find("date")!=-1):
                SQLStr += "DATE_FORMAT("+iField+", '%Y-%m-%d %H:%i:%s'), "
            else:
                SQLStr += iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "WHERE ("+IDField+", "+DTField+") IN ("+SubSQLStr+") "
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        LookBack = args.get("回溯天数", self.LookBack)
        DT2Str = args.get("时间转字符串", self.DT2Str)
        if dts is not None:
            dts = sorted(dts)
            StartDate, EndDate = dts[0].date(), dts[-1].date()
            if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        else:
            StartDate = EndDate = None
        # 形成 SQL 语句, 时点, ID, 因子数据
        SQLStr = "SELECT "+self._DBTableName+"."+DTField+", "
        SQLStr += self._DBTableName+"."+IDField+", "
        for iField in factor_names:
            if iField=="datetime": iDBDataType = "datetime"
            elif iField=="code": iDBDataType = "varchar(40)"
            else: iDBDataType = self._FactorDB._TableFieldDataType[self._Name][iField]
            if DT2Str and (iDBDataType.lower().find("date")!=-1):
                SQLStr += "DATE_FORMAT("+self._DBTableName+"."+iField+", '%Y-%m-%d %H:%i:%s'), "
            else:
                SQLStr += self._DBTableName+"."+iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        if StartDate is not None:
            SQLStr += "WHERE "+self._DBTableName+"."+DTField+">='"+StartDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            SQLStr += "AND "+self._DBTableName+"."+DTField+"<='"+EndDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        else:
            SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+IDField, ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+DTField+", "+self._DBTableName+"."+IDField
        if args.get("因子值类型", self.ValueType)!="scalar":
            SQLStr += ", "+self._DBTableName+"."+factor_names[0]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData), columns=["QS_DT", "ID"]+factor_names)
        if (StartDate is not None) and np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==dt.datetime.combine(StartDate, dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "ID"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["QS_DT", "ID"])
        if self._FactorDB.DBType=="sqlite3": RawData["QS_DT"] = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in RawData.pop("QS_DT")]
        return RawData
    def _calcListData(self, raw_data, factor_names, ids, dts, args={}):
        Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        return _adjustData(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["QS_DT", "ID"])
        ValueType = args.get("因子值类型", self.ValueType)
        if ValueType=="list":
            return self._calcListData(raw_data, factor_names, ids, dts, args=args)
        elif ValueType=="scalar":
            if not raw_data.index.is_unique:
                FilterStr = args.get("筛选条件", self.FilterCondition)
                raise __QS_Error__("筛选条件: '%s' 无法保证唯一性!" % FilterStr)
        else:
            if not raw_data.index.is_unique:
                return self._calcListData(raw_data, factor_names, ids, dts, args=args)
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        return _adjustData(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts)

class _NarrowTable(FactorTable):
    """SQLDB 窄因子表"""
    TableType = Enum("窄表", arg_type="SingleOption", label="因子表类型", order=0)
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=1)
    ValueType = Enum("scalar", "list", "scalar or list", arg_type="SingleOption", label="因子值类型", order=2)
    FilterCondition = Str("", arg_type="String", label="筛选条件", order=3)
    #DTField = Enum("datetime", arg_type="SingleOption", label="时点字段", order=4)
    #IDField = Enum("code", arg_type="SingleOption", label="ID字段", order=5)
    #FactorField = Enum("code", arg_type="SingleOption", label="因子字段", order=6)
    #FactorValueField = Enum(None, arg_type="SingleOption", label="因子值字段", order=7)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DataType = "double"
        self._DBDataType = fdb._TableFieldDataType[name]
        self._DBTableName = fdb.TablePrefix+fdb.InnerPrefix+name
        self._FactorNames = None
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        Fields = ["datetime"] + self._DBDataType[self._DBDataType.str.contains("date")].index.tolist()
        self.add_trait("DTField", Enum(*Fields, arg_type="SingleOption", label="时点字段", order=3))
        StrMask = (self._DBDataType.str.contains("char") | self._DBDataType.str.contains("text"))
        Fields = ["code"] + self._DBDataType[StrMask].index.tolist()
        self.add_trait("IDField", Enum(*Fields, arg_type="SingleOption", label="ID字段", order=4))
        self.add_trait("FactorField", Enum(*Fields, arg_type="SingleOption", label="因子字段", order=5))
        Fields = [None] + self._DBDataType.index.tolist()
        self.add_trait("FactorValueField", Enum(*Fields, arg_type="SingleOption", label="因子值字段", order=6))
    @on_trait_change("FactorField")
    def _on_FactorField_changed(self, obj, name, old, new):
        if self.FactorField is not None:
            self._FactorNames = None
    @on_trait_change("FactorValueField")
    def _on_FactorValueField_changed(self, obj, name, old, new):
        if self.FactorValueField is not None:
            self._DataType = self._FactorDB._TableFactorDict[self.Name].loc[self.FactorValueField]
        else:
            self._DataType = "double"
    @property
    def FactorNames(self):
        if self._FactorNames is None:
            SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+self.FactorField+" "
            SQLStr += "FROM "+self._DBTableName+" "
            SQLStr += "WHERE "+self._DBTableName+"."+self.FactorField+" IS NOT NULL "
            SQLStr += "ORDER BY "+self._DBTableName+"."+self.FactorField
            self._FactorNames = [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        return self._FactorNames
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType": return pd.Series(self._DataType, index=factor_names, dtype=np.dtype("O"))
        if key is None: return pd.DataFrame(self._DataType, index=factor_names, columns=["DataType"], dtype=np.dtype("O"))
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        FactorField = args.get("因子字段", self.FactorField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+IDField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+IDField+" IS NOT NULL "
        if idt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+FactorField+"='"+ifactor_name+"' "
        SQLStr += "ORDER BY "+self._DBTableName+"."+IDField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        FactorField = args.get("因子字段", self.FactorField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+DTField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
        if iid is not None: SQLStr += "AND "+self._DBTableName+"."+IDField+"='"+iid+"' "
        if start_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+">='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+FactorField+"='"+ifactor_name+"' "
        SQLStr += "ORDER BY "+self._DBTableName+"."+DTField
        if self._FactorDB.DBType!="sqlite3": return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        else: return [dt.datetime.strptime(iRslt[0], "%Y-%m-%d %H:%M:%S.%f") for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ArgConditionGroup = {}
        ArgNames = self.ArgNames
        ArgNames.remove("回溯天数")
        ArgNames.remove("因子值类型")
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
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        FactorField = args.get("因子字段", self.FactorField)
        FactorValueField = args.get("因子值字段", self.FactorValueField)
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += FactorField
        SubSQLStr += "MAX("+DTField+") "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "WHERE "+DTField+"<'"+end_date.strftime("%Y-%m-%d:%H:%M:%S.%f")+"' "
        SubSQLStr += "AND ("+genSQLInCondition(IDField, ids, is_str=True, max_num=1000)+") "
        if len(factor_names)<1000:
            SubSQLStr += "AND ("+genSQLInCondition(FactorField, factor_names, is_str=True, max_num=1000)+") "
        else:
            SubSQLStr += "AND "+FactorField+" IS NOT NULL "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SubSQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SubSQLStr += "GROUP BY "+IDField+", "+FactorField
        SQLStr = "SELECT "+DTField+", "
        SQLStr += IDField+", "
        SQLStr += FactorField+", "
        if FactorValueField is not None:
            SQLStr += FactorValueField+" "
        else:
            SQLStr += "1 "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE ("+IDField+", "+FactorField+", "+DTField+") IN ("+SubSQLStr+") "
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        FactorField = args.get("因子字段", self.FactorField)
        FactorValueField = args.get("因子值字段", self.FactorValueField)
        LookBack = args.get("回溯天数", self.LookBack)
        if dts is not None:
            dts = sorted(dts)
            StartDate, EndDate = dts[0].date(), dts[-1].date()
            if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        else:
            StartDate = EndDate = None
        # 形成 SQL 语句, 时点, ID, 因子数据
        SQLStr = "SELECT "+DTField+", "
        SQLStr += IDField+", "
        SQLStr += FactorField+", "
        if FactorValueField is not None:
            SQLStr += FactorValueField+", "
        else:
            SQLStr += "1, "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        if StartDate is not None:
            SQLStr += "WHERE "+DTField+">='"+StartDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            SQLStr += "AND "+DTField+"<='"+EndDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        else:
            SQLStr += "WHERE "+DTField+" IS NOT NULL "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(IDField, ids, is_str=True, max_num=1000)+") "
        if len(factor_names)<1000:
            SQLStr += "AND ("+genSQLInCondition(FactorField, factor_names, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "AND "+FactorField+" IS NOT NULL "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+DTField+", "+IDField+", "+FactorField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID", "QS_Factor", "QS_FactorValue"])
        RawData = pd.DataFrame(np.array(RawData), columns=["QS_DT", "ID", "QS_Factor", "QS_FactorValue"])
        if (StartDate is not None) and np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==dt.datetime.combine(StartDate, dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "ID", "QS_Factor", "QS_FactorValue"])
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["QS_DT", "ID", "QS_Factor"])
        if self._FactorDB.DBType=="sqlite3": RawData["QS_DT"] = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in RawData.pop("QS_DT")]
        return RawData
    def _calcListData(self, raw_data, factor_names, ids, dts, args={}):
        raw_data.index = raw_data.index.swaplevel(i=0, j=-1)
        Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in factor_names:
            if iFactorName in raw_data:
                Data[iFactorName] = raw_data.loc[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        if not Data: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return _adjustData(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["QS_DT", "ID", "QS_Factor"]).iloc[:, 0]
        ValueType = args.get("因子值类型", self.ValueType)
        if ValueType=="list":
            return self._calcListData(raw_data, factor_names, ids, dts, args=args)
        elif ValueType=="scalar":
            if not raw_data.index.is_unique:
                FilterStr = args.get("筛选条件", self.FilterCondition)
                raise __QS_Error__("筛选条件: '%s' 无法保证唯一性!" % FilterStr)
        else:
            if not raw_data.index.is_unique:
                return self._calcListData(raw_data, factor_names, ids, dts, args=args)
        raw_data = raw_data.unstack()
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in factor_names:
            if iFactorName in raw_data:
                iRawData = raw_data[iFactorName].unstack()
                if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
                Data[iFactorName] = iRawData
        if not Data: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        return _adjustData(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts)


class _FeatureTable(_WideTable):
    """截面宽表"""
    TableType = Enum("截面宽表", arg_type="SingleOption", label="因子表类型", order=0)
    LookBack = Float(np.inf, arg_type="Integer", label="回溯天数", order=1)
    TargetDT = Either(None, Date, arg_type="DateTime", label="目标时点", order=7)
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if ids==[]: return pd.DataFrame(columns=["ID"]+factor_names)
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        TargetDT = args.get("目标时点", self.TargetDT)
        if TargetDT is None:
            SQLStr = "SELECT MAX("+self._DBTableName+"."+DTField+") FROM "+self._DBTableName+" "
            SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
            if ids is not None: SQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+IDField, ids, is_str=True, max_num=1000)+") "
            FilterStr = args.get("筛选条件", self.FilterCondition)
            if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
            TargetDT =  self._FactorDB.fetchall(SQLStr)
            if not TargetDT: return pd.DataFrame(columns=["ID"]+factor_names)
            TargetDT = TargetDT[0][0]
        RawData = super().__QS_prepareRawData__(factor_names, ids, [TargetDT], args=args)
        RawData["QS_TargetDT"] = TargetDT
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        TargetDT = raw_data.pop("QS_TargetDT").iloc[0].to_pydatetime()
        Data = super().__QS_calcData__(raw_data, factor_names, ids, [TargetDT], args=args)
        Data = Data.iloc[:, 0, :]
        return pd.Panel(Data.values.T.reshape((Data.shape[1], Data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=Data.index, minor_axis=dts).swapaxes(1, 2)

class SQLDB(QSSQLObject, WritableFactorDB):
    """SQLDB"""
    CheckWriteData = Bool(False, arg_type="Bool", label="检查写入值", order=100)
    IgnoreFields = ListStr(arg_type="List", label="忽略字段", order=101)
    InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=102)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"SQLDBConfig.json" if config_file is None else config_file), **kwargs)
        self._TableFactorDict = {}# {表名: pd.Series(数据类型, index=[因子名])}
        self._TableFieldDataType = {}# {表名: pd.Series(        数据库数据类型, index=[因子名])}
        self.Name = "SQLDB"
        return
    def connect(self):
        super().connect()
        nPrefix = len(self.InnerPrefix)
        if self._Connector=="sqlite3":
            SQLStr = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%s%%' ORDER BY name"
            Cursor = self.cursor(SQLStr % self.InnerPrefix)
            AllTables = Cursor.fetchall()
            self._TableFactorDict = {}
            self._TableFieldDataType = {}
            IgnoreFields = ["code", "datetime"]+list(self.IgnoreFields)
            for iTableName in AllTables:
                iTableName = iTableName[0][nPrefix:]
                Cursor.execute("PRAGMA table_info([%s])" % self.InnerPrefix+iTableName)
                iDataType = np.array(Cursor.fetchall())
                iDataType = pd.Series(iDataType[:, 2], index=iDataType[:, 1])
                iDataType = iDataType[iDataType.index.difference(IgnoreFields)]
                if iDataType.shape[0]>0:
                    self._TableFieldDataType[iTableName] = iDataType.copy()
                    iDataType[iDataType=="text"] = "string"
                    iDataType[iDataType=="real"] = "double"
                    self._TableFactorDict[iTableName] = iDataType
        elif self.DBType=="MySQL":
            SQLStr = ("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE table_schema='%s' " % self.DBName)
            SQLStr += ("AND TABLE_NAME LIKE '%s%%' " % self.InnerPrefix)
            SQLStr += "AND COLUMN_NAME NOT IN ('code', 'datetime'"
            if len(self.IgnoreFields)>0:
                SQLStr += ",'"+"','".join(self.IgnoreFields)+"') "
            else:
                SQLStr += ") "
            SQLStr += "ORDER BY TABLE_NAME, COLUMN_NAME"
            Rslt = self.fetchall(SQLStr)
            if not Rslt:
                self._TableFieldDataType = {}
                self._TableFactorDict = {}
            else:
                self._TableFieldDataType = pd.DataFrame(np.array(Rslt), columns=["表", "因子", "DataType"]).set_index(["表", "因子"])["DataType"]
                self._TableFactorDict = self._TableFieldDataType.copy()
                Mask = (self._TableFactorDict.str.contains("char") | self._TableFactorDict.str.contains("date"))
                self._TableFactorDict[Mask] = "string"
                self._TableFactorDict[~Mask] = "double"
                self._TableFactorDict = {iTable[nPrefix:]:self._TableFactorDict.loc[iTable] for iTable in self._TableFactorDict.index.levels[0]}
                self._TableFieldDataType = {iTable[nPrefix:]:self._TableFieldDataType.loc[iTable] for iTable in self._TableFieldDataType.index.levels[0]}
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
        elif TableType=="窄表":
            return _NarrowTable(name=table_name, fdb=self, sys_args=args, logger=self._QS_Logger)
        elif TableType=="截面宽表":
            return _FeatureTable(name=table_name, fdb=self, sys_args=args, logger=self._QS_Logger)
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
        self.renameDBTable(self.InnerPrefix+old_table_name, self.InnerPrefix+new_table_name)
        self._TableFactorDict[new_table_name] = self._TableFactorDict.pop(old_table_name)
        self._TableFieldDataType[new_table_name] = self._TableFieldDataType.pop(old_table_name)
        return 0
    # 创建表, field_types: {字段名: 数据库数据类型}
    def createTable(self, table_name, field_types):
        FieldTypes = field_types.copy()
        if self.DBType=="MySQL":
            FieldTypes["datetime"] = field_types.pop("datetime", "DATETIME(6) NOT NULL")
            FieldTypes["code"] = field_types.pop("code", "VARCHAR(40) NOT NULL")
        elif self.DBType=="sqlite3":
            FieldTypes["datetime"] = field_types.pop("datetime", "text NOT NULL")
            FieldTypes["code"] = field_types.pop("code", "text NOT NULL")
        self.createDBTable(self.InnerPrefix+table_name, FieldTypes, primary_keys=["datetime", "code"], index_fields=["datetime", "code"])
        self._TableFactorDict[table_name] = pd.Series({iFactorName: ("string" if field_types[iFactorName].find("char")!=-1 else "double") for iFactorName in field_types})
        self._TableFieldDataType[table_name] = pd.Series(field_types)
        return 0
    # 增加因子，field_types: {字段名: 数据库数据类型}
    def addFactor(self, table_name, field_types):
        if table_name not in self._TableFactorDict: return self.createTable(table_name, field_types)
        self.addField(self.InnerPrefix+table_name, field_types)
        NewDataType = pd.Series({iFactorName: ("string" if field_types[iFactorName].find("char")!=-1 else "double") for iFactorName in field_types})
        self._TableFactorDict[table_name] = self._TableFactorDict[table_name].append(NewDataType)
        self._TableFieldDataType[table_name] = self._TableFieldDataType[table_name].append(pd.Series(field_types))
        return 0
    def deleteTable(self, table_name):
        if table_name not in self._TableFactorDict: return 0
        self.deleteDBTable(self.InnerPrefix+table_name)
        self._TableFactorDict.pop(table_name, None)
        self._TableFieldDataType.pop(table_name, None)
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
        self.renameField(self.InnerPrefix+table_name, old_factor_name, new_factor_name)
        self._TableFactorDict[table_name][new_factor_name] = self._TableFactorDict[table_name].pop(old_factor_name)
        self._TableFieldDataType[table_name][new_factor_name] = self._TableFieldDataType[table_name].pop(old_factor_name)
        return 0
    def deleteFactor(self, table_name, factor_names):
        if not factor_names: return 0
        FactorIndex = self._TableFactorDict.get(table_name, pd.Series()).index.difference(factor_names).tolist()
        if not FactorIndex: return self.deleteTable(table_name)
        self.deleteField(self.InnerPrefix+table_name, factor_names)
        self._TableFactorDict[table_name] = self._TableFactorDict[table_name][FactorIndex]
        self._TableFieldDataType[table_name] = self._TableFieldDataType[table_name][FactorIndex]
        return 0
    def deleteData(self, table_name, ids=None, dts=None):
        if table_name not in self._TableFactorDict:
            Msg = ("因子库 '%s' 调用方法 deleteData 错误: 不存在因子表 '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (ids is None) and (dts is None): return self.truncateDBTable(self.InnerPrefix+table_name)
        DBTableName = self.TablePrefix+self.InnerPrefix+table_name
        SQLStr = "DELETE FROM "+DBTableName
        if dts is not None:
            DTs = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts]
            SQLStr += "WHERE "+genSQLInCondition(DBTableName+".datetime", DTs, is_str=True, max_num=1000)+" "
        else:
            SQLStr += "WHERE "+DBTableName+".datetime IS NOT NULL "
        if ids is not None:
            SQLStr += "AND "+genSQLInCondition(DBTableName+".code", ids, is_str=True, max_num=1000)
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteData 删除表 '%s' 中数据时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        return 0
    def _adjustWriteData(self, data):
        NewData = []
        DataLen = data.applymap(lambda x: len(x) if isinstance(x, list) else 1)
        DataLenMax = DataLen.max(axis=1)
        DataLenMin = DataLen.min(axis=1)
        if (DataLenMax!=DataLenMin).sum()>0:
            self._QS_Logger.warning("'%s' 在写入因子 '%s' 时出现因子值长度不一致的情况, 将填充缺失!" % (self.Name, str(data.columns.tolist())))
        for i in range(data.shape[0]):
            iDataLen = DataLen.iloc[i]
            if iDataLen>0:
                iData = data.iloc[i].apply(lambda x: [None]*(iDataLen-len(x))+x if isinstance(x, list) else [x]*iDataLen).tolist()
                NewData.extend(zip(*iData))
        NewData = pd.DataFrame(NewData, dtype="O")
        return NewData.where(pd.notnull(NewData), None).to_records(index=False).tolist()
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._TableFactorDict:
            FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            self.createTable(table_name, field_types=FieldTypes)
            SQLStr = "INSERT INTO "+self.TablePrefix+self.InnerPrefix+table_name+" (`datetime`, `code`, "
        else:
            NewFactorNames = data.items.difference(self._TableFactorDict[table_name].index).tolist()
            if NewFactorNames:
                FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(NewFactorNames)}
                self.addFactor(table_name, FieldTypes)
            AllFactorNames = self._TableFactorDict[table_name].index.tolist()
            if self.CheckWriteData:
                OldData = self.getTable(table_name, args={"因子值类型":"list", "时间转字符串":True}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
            else:
                OldData = self.getTable(table_name, args={"时间转字符串":True}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
            if if_exists=="append":
                for iFactorName in AllFactorNames:
                    if iFactorName in data:
                        data[iFactorName] = OldData[iFactorName].where(pd.notnull(OldData[iFactorName]), data[iFactorName])
                    else:
                        data[iFactorName] = OldData[iFactorName]
            elif if_exists=="update":
                for iFactorName in AllFactorNames:
                    if iFactorName in data:
                        data[iFactorName] = data[iFactorName].where(pd.notnull(data[iFactorName]), OldData[iFactorName])
                    else:
                        data[iFactorName] = OldData[iFactorName]
            SQLStr = "REPLACE INTO "+self.TablePrefix+self.InnerPrefix+table_name+" (`datetime`, `code`, "
        data.major_axis = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in data.major_axis]
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
            SQLStr += "`"+iFactorName+"`, "
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        NewData = NewData[pd.notnull(NewData).any(axis=1)]
        if NewData.shape[0]==0: return 0
        if self._Connector in ("pyodbc", "sqlite3"):
            SQLStr = SQLStr[:-2] + ") VALUES (" + "?, " * (NewData.shape[1]+2)
        else:
            SQLStr = SQLStr[:-2] + ") VALUES (" + "%s, " * (NewData.shape[1]+2)
        SQLStr = SQLStr[:-2]+") "
        Cursor = self._Connection.cursor()
        if self.CheckWriteData:
            NewData = self._adjustWriteData(NewData.reset_index())
            Cursor.executemany(SQLStr, NewData)
        else:
            NewData = NewData.astype("O").where(pd.notnull(NewData), None)
            Cursor.executemany(SQLStr, NewData.reset_index().values.tolist())
        self._Connection.commit()
        Cursor.close()
        return 0