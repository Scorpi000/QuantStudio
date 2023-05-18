# coding=utf-8
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str, Int, Float, Callable, Either, List, ListStr, Enum, Date, on_trait_change

from QuantStudio import __QS_Error__
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries, getDateSeries
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.FactorDataBase.FactorDB import FactorTable
from QuantStudio.Tools.api import Panel

# Quant Studio 系统错误: 重复索引
class __QS_Error_DuplicatedIndex__(__QS_Error__):
    """Quant Studio 重复索引错误"""
    pass

# 将信息源文件中的表和字段信息导入信息文件
def importInfo(info_file, info_resource, out_info=False):
    TableInfo = pd.read_excel(info_resource, "TableInfo").set_index(["TableName"], engine="openpyxl")
    FactorInfo = pd.read_excel(info_resource, 'FactorInfo').set_index(['TableName', 'FieldName'], engine="openpyxl")
    if not out_info:
        try:
            from QuantStudio.Tools.DataTypeFun import writeNestedDict2HDF5
            writeNestedDict2HDF5(TableInfo, info_file, "/TableInfo")
            writeNestedDict2HDF5(FactorInfo, info_file, "/FactorInfo")
        except:
            pass
    return (TableInfo, FactorInfo)

# 更新信息文件
def updateInfo(info_file, info_resource, logger, out_info=False):
    if out_info: return importInfo(info_file, info_resource, logger, out_info=out_info)
    if not os.path.isfile(info_file):
        logger.warning("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    elif (os.path.getmtime(info_resource)>os.path.getmtime(info_file)):
        logger.warning("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % info_resource)
    else:
        try:
            from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
            return (readNestedDictFromHDF5(info_file, ref="/TableInfo"), readNestedDictFromHDF5(info_file, ref="/FactorInfo"))
        except:
            logger.warning("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    if not os.path.isfile(info_resource): raise __QS_Error__("缺失数据库信息源文件: %s" % info_resource)
    return importInfo(info_file, info_resource)

def adjustDateTime(data, dts, fillna=False, **kwargs):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.shape[0]==0:
            if isinstance(data, pd.DataFrame): data = pd.DataFrame(index=dts, columns=data.columns)
            else: data = pd.Series(index=dts)
        else:
            if fillna:
                AllDTs = data.index.union(dts)
                AllDTs = AllDTs.sort_values()
                data = data.reindex(index=AllDTs)
                data = data.fillna(**kwargs)
            data = data.reindex(index=dts)
    else:
        if data.shape[1]==0:
            data = Panel(items=data.items, major_axis=dts, minor_axis=data.minor_axis)
        else:
            FactorNames = data.items
            if fillna:
                AllDTs = data.major_axis.union(dts)
                AllDTs = AllDTs.sort_values()
                data = data.loc[:, AllDTs, :]
                data = Panel({data.items[i]:data.iloc[i].fillna(axis=0, **kwargs) for i in range(data.shape[0])}, items=data.items, major_axis=AllDTs, minor_axis=data.minor_axis)
            data = data.loc[FactorNames, dts, :]
    return data

def adjustDataDTID(data, look_back, factor_names, ids, dts, only_start_lookback=False, only_lookback_nontarget=False, only_lookback_dt=False, logger=None):
    if look_back==0:
        try:
            return data.loc[:, dts, ids]
        except KeyError:
            if logger is not None:
                logger.warning("待提取的因子 %s 数据超出了原始数据的时点或 ID 范围, 将填充缺失值!" % (str(list(data.items)), ))
            return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
    AllDTs = data.major_axis.union(dts).sort_values()
    AdjData = data.loc[:, AllDTs, ids]
    if only_start_lookback:# 只在起始时点回溯填充缺失
        AllAdjData = AdjData
        AdjData = AllAdjData.loc[:, :dts[0], :]
        TargetDTs = dts[:1]
    else:
        TargetDTs = dts
    if only_lookback_dt:
        TargetDTs = sorted(set(TargetDTs).difference(data.major_axis))
    if TargetDTs:
        Limits = look_back*24.0*3600
        if only_lookback_nontarget:# 只用非目标时间序列的数据回溯填充
            Mask = pd.Series(np.full(shape=(AdjData.shape[1], ), fill_value=False, dtype=bool), index=AdjData.major_axis)
            Mask[TargetDTs] = True
            FillMask = Mask.copy()
            FillMask[Mask.astype("int").diff()!=1] = False
            TimeDelta = pd.Series(np.r_[0, np.diff(Mask.index.values) / np.timedelta64(1, "D")], index=Mask.index)
            TimeDelta[(Mask & (~FillMask)) | (Mask.astype("int").diff()==-1)] = 0
            TimeDelta = TimeDelta.cumsum().reindex(index=TargetDTs)
            FirstDelta = TimeDelta.iloc[0]
            TimeDelta = TimeDelta.diff().fillna(value=0)
            TimeDelta.iloc[0] = FirstDelta
            NewLimits = np.minimum(TimeDelta.values*24.0*3600, Limits).reshape((TimeDelta.shape[0], 1)).repeat(AdjData.shape[2], axis=1)
            Limits = pd.DataFrame(0, index=AdjData.major_axis, columns=AdjData.minor_axis)
            Limits.loc[TargetDTs, :] = NewLimits
        if only_lookback_dt:
            Mask = pd.Series(np.full(shape=(AdjData.shape[1], ), fill_value=False, dtype=bool), index=AdjData.major_axis)
            Mask[TargetDTs] = True
            FillMask = Mask.copy()
            FillMask[Mask.astype("int").diff()!=1] = False
            FillMask = FillMask.loc[TargetDTs]
            TimeDelta = pd.Series(np.r_[0, np.diff(Mask.index.values) / np.timedelta64(1, "D")], index=Mask.index).reindex(index=TargetDTs)
            NewLimits = TimeDelta.cumsum()
            Temp = NewLimits.copy()
            Temp[~FillMask] = np.nan
            Temp = Temp.fillna(method="pad")
            TimeDelta[~FillMask] = np.nan
            NewLimits = NewLimits - Temp + TimeDelta.fillna(method="pad")
            if isinstance(Limits, pd.DataFrame):
                Limits.loc[TargetDTs, :] = np.minimum(NewLimits.values.reshape((NewLimits.shape[0], 1)).repeat(AdjData.shape[2], axis=1), Limits.loc[TargetDTs].values)
            else:
                NewLimits = np.minimum(NewLimits.values*24.0*3600, Limits).reshape((NewLimits.shape[0], 1)).repeat(AdjData.shape[2], axis=1)
                Limits = pd.DataFrame(0, index=AdjData.major_axis, columns=AdjData.minor_axis)
                Limits.loc[TargetDTs, :] = NewLimits
        MajorAxis, MinorAxis = AdjData.major_axis, AdjData.minor_axis
        AdjData = dict(AdjData)
        if np.isinf(look_back) and (not only_lookback_nontarget) and (not only_lookback_dt):
            for iFactorName in AdjData: AdjData[iFactorName] = AdjData[iFactorName].fillna(method="pad")
        else:
            for iFactorName in AdjData: AdjData[iFactorName] = fillNaByLookback(AdjData[iFactorName], lookback=Limits)
        AdjData = Panel(AdjData, items=factor_names, major_axis=MajorAxis, minor_axis=MinorAxis)
    if only_start_lookback:
        AllAdjData.loc[:, dts[0], :] = AdjData.loc[:, dts[0], :]
        return AllAdjData.loc[:, dts]
    else:
        return AdjData.loc[:, dts]


# ===================== 因子数据计算 =====================
# raw_data: DataFrame(columns=["QS_DT", "ID"]+factor_names)
def _QS_calcListData_WideTable(raw_data, factor_names, ids, dts, args={}, **kwargs):
    Operator = args.get("算子", lambda x: x.tolist())
    if Operator is None: Operator = lambda x: x.tolist()
    OperatorDataType = args.get("算子数据类型", "object")
    AdditionalFields = args.get("附加字段", [])
    if args.get("只回溯时点", False):
        DeduplicatedIndex = raw_data.index(~raw_data.index.duplicated())
        RowIdxMask = pd.Series(False, index=DeduplicatedIndex).unstack(fill_value=True).astype(bool)
        RawIDs = RowIdxMask.columns
        if RawIDs.intersection(ids).shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
        RowIdx[RowIdxMask] = np.nan
        RowIdx = adjustDataDTID(Panel({"RowIdx": RowIdx}), args.get("回溯天数", 0), ["RowIdx"], RowIdx.columns.tolist(), dts, args.get("只起始日回溯", False), args.get("只回溯非目标日", False), logger=kwargs.get("logger", None)).iloc[0].values
        RowIdx[pd.isnull(RowIdx)] = -1
        RowIdx = RowIdx.astype(int)
        ColIdx = np.arange(RowIdx.shape[1]).reshape((1, RowIdx.shape[1])).repeat(RowIdx.shape[0], axis=0)
        RowIdxMask = (RowIdx==-1)
        Data = {}
        for iFactorName in factor_names:
            if AdditionalFields:
                iRawData = raw_data.reindex(columns=[iFactorName]+AdditionalFields).groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
            else:
                iRawData = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
            iRawData = iRawData.values[RowIdx, ColIdx]
            iRawData[RowIdxMask] = None
            Data[iFactorName] = pd.DataFrame(iRawData, index=dts, columns=RawIDs)
            if OperatorDataType=="double":
                Data[iFactorName] = Data[iFactorName].astype(float)
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=RawIDs).loc[:, :, ids]
    else:
        Data = {}
        for iFactorName in factor_names:
            if AdditionalFields:
                Data[iFactorName] = raw_data.reindex(columns=[iFactorName]+AdditionalFields).groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
            else:
                Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
            if OperatorDataType=="double":
                Data[iFactorName] = Data[iFactorName].astype(float)
        Data = Panel(Data, items=factor_names)
        return adjustDataDTID(Data, args.get("回溯天数", 0), factor_names, ids, dts, args.get("只起始日回溯", False), args.get("只回溯非目标日", False), logger=kwargs.get("logger", None))

def _QS_calcData_WideTable(raw_data, factor_names, ids, dts, data_type, args={}, **kwargs):
    if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
    if ids is None: ids = sorted(raw_data["ID"].unique())
    raw_data = raw_data.set_index(["QS_DT", "ID"])
    MultiMapping = args.get("多重映射", False)
    if MultiMapping:
        return _QS_calcListData_WideTable(raw_data, factor_names, ids, dts, args=args, **kwargs)
    else:
        if not raw_data.index.is_unique:
            Msg = kwargs.get("error_fmt", {}).get("DuplicatedIndex", "{Error}")
            raise __QS_Error_DuplicatedIndex__(Msg.format(Error = ("重复的索引为 : %s" % (str(raw_data.index[raw_data.index.duplicated()].tolist()), ))))
    DataType = data_type[~data_type.index.duplicated()]
    if args.get("只回溯时点", False):
        RowIdxMask = pd.Series(False, index=raw_data.index).unstack(fill_value=True).astype(bool)
        RawIDs = RowIdxMask.columns
        if RawIDs.intersection(ids).shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
        RowIdx[RowIdxMask] = np.nan
        RowIdx = adjustDataDTID(Panel({"RowIdx": RowIdx}), args.get("回溯天数", 0), ["RowIdx"], RowIdx.columns.tolist(), dts, args.get("只起始日回溯", False), args.get("只回溯非目标日", False), only_lookback_dt=True, logger=kwargs.get("logger", None)).iloc[0].values
        RowIdx[pd.isnull(RowIdx)] = -1
        RowIdx = RowIdx.astype(int)
        ColIdx = np.arange(RowIdx.shape[1]).reshape((1, RowIdx.shape[1])).repeat(RowIdx.shape[0], axis=0)
        RowIdxMask = (RowIdx==-1)
        Data = {}
        for iFactorName in raw_data.columns.intersection(factor_names):
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double":
                try:
                    iRawData = iRawData.astype("float")
                except:
                    pass
            iRawData = iRawData.values[RowIdx, ColIdx]
            iRawData[RowIdxMask] = None
            Data[iFactorName] = pd.DataFrame(iRawData, index=dts, columns=RawIDs)
        return Panel(Data, major_axis=dts, minor_axis=RawIDs).loc[factor_names, :, ids]
    else:
        Data = {}
        for iFactorName in raw_data.columns.intersection(factor_names):
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double":
                try:
                    iRawData = iRawData.astype("float")
                except:
                    pass
            Data[iFactorName] = iRawData
        Data = Panel(Data).loc[factor_names]
        return adjustDataDTID(Data, args.get("回溯天数", 0), factor_names, ids, dts, args.get("只起始日回溯", False), args.get("只回溯非目标日", False), logger=kwargs.get("logger", None))

# raw_data: DataFrame(columns=["QS_DT", "ID", 因子名字段, 因子值字段])
def _QS_calcListData_NarrowTable(raw_data, factor_names, ids, dts, args={}, **kwargs):
    raw_data.index = raw_data.index.swaplevel(i=0, j=-1)
    Operator = args.get("算子", lambda x: x.tolist())
    if Operator is None: Operator = lambda x: x.tolist()
    Data = {}
    for iFactorName in factor_names:
        if iFactorName in raw_data:
            Data[iFactorName] = raw_data.loc[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
    if not Data: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
    Data = Panel(Data, items=factor_names).swapaxes(1, 2)
    return adjustDataDTID(Data, args.get("回溯天数", 0), factor_names, ids, dts, args.get("只起始日回溯", False), args.get("只回溯非目标日", False), args.get("只回溯时点", False), logger=kwargs.get("logger", None))

def _QS_calcData_NarrowTable(raw_data, factor_names, ids, dts, data_type, args={}, **kwargs):
    if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
    if ids is None: ids = sorted(raw_data["ID"].unique())
    FactorNameField = args.get("因子名字段", "FactorName")
    raw_data = raw_data.set_index(["QS_DT", "ID", FactorNameField]).iloc[:, 0]
    MultiMapping = args.get("多重映射", False)
    if MultiMapping:
        return _QS_calcListData_NarrowTable(raw_data, factor_names, ids, dts, args=args, **kwargs)
    else:
        if not raw_data.index.is_unique:
            Msg = kwargs.get("error_fmt", {}).get("DuplicatedIndex", "{Error}")
            raise __QS_Error_DuplicatedIndex__(Msg.format(Error = ("重复的索引为 : %s" % (str(raw_data.index[raw_data.index.duplicated()].tolist()), ))))
    raw_data = raw_data.unstack()
    DataType = data_type[~data_type.index.duplicated()]
    Data = {}
    for iFactorName in factor_names:
        if iFactorName in raw_data:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
    if not Data: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
    Data = Panel(Data, items=factor_names)
    return adjustDataDTID(Data, args.get("回溯天数", 0), factor_names, ids, dts, args.get("只起始日回溯", False), args.get("只回溯非目标日", False), args.get("只回溯时点", False), logger=kwargs.get("logger", None))


# ===================== 基于 SQL 数据库表的因子表 =====================
# table_info: Series(index=["DBTableName", "TableClass"]), 可选的 index=["MainTableName", "MainTableID", "JoinCondition", "MainTableCondition", "DefaultSuffix", "Exchange", "SecurityCategory"]
# factor_info: DataFrame(index=[], columns=["DBFieldName", "DataType", "FieldType", "Supplementary", "Description"]), 可选的 columns=["RelatedSQL"]
# security_info: DataFrame(index=[], columns=["Suffix"])
# exchange_info: DataFrame(index=[], columns=["Suffix"])
# 参数编号分配:
# 0 - 100: 因子表特定参数
# 100 - 199: 条件参数, 100: 通用筛选条件
# 200 - 299: 通用参数
class SQL_Table(FactorTable):
    """SQL 因子表"""
    class __QS_ArgClass__(FactorTable.__QS_ArgClass__):
        FilterCondition = Str("", arg_type="String", label="筛选条件", order=100)
        TableType = Str(arg_type="String", label="因子表类型", order=200, mutable=False)# 不可变
        PreFilterID = Enum(True, False, arg_type="Bool", label="预筛选ID", order=201)
        #DTField = Enum(None, arg_type="SingleOption", label="时点字段", order=202)
        #IDField = Enum(None, arg_type="SingleOption", label="ID字段", order=203)
        DTFmt = Str("", arg_type="String", label="时点格式", order=204, mutable=False)
        DateFmt = Str("", arg_type="String", label="日期格式", order=205, mutable=False)
        UseIndex = ListStr(arg_type="ListStr", label="使用索引", order=206)
        ForceIndex = ListStr(arg_type="ListStr", label="强制索引", order=207)
        IgnoreIndex = ListStr(arg_type="ListStr", label="忽略索引", order=208)
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            # 设置因子表类型
            self.TableType = self._Owner._TableInfo["TableClass"]
            # 解析 ID 字段, 至多一个 ID 字段
            IDFields = [None] + self._Owner._FactorInfo[pd.notnull(self._Owner._FactorInfo["FieldType"])].index.tolist()# ID 字段
            self.add_trait("IDField", Enum(*IDFields, arg_type="SingleOption", label="ID字段", order=203, option_range=IDFields))
            self.IDField = None
            # 解析时点字段
            Mask = self._Owner._FactorInfo["FieldType"].str.lower().str.contains("date")
            Fields = self._Owner._FactorInfo[Mask].index.tolist()# 所有的时点字段列表
            Fields.append(None)
            self.add_trait("DTField", Enum(*Fields, arg_type="SingleOption", label="时点字段", order=202, option_range=Fields))
            iFactorInfo = self._Owner._FactorInfo[Mask & (self._Owner._FactorInfo["Supplementary"]=="Default")]
            if iFactorInfo.shape[0]>0: self.DTField = iFactorInfo.index[0]
            else: self.DTField = Fields[0]
            # 解析条件字段
            self._ConditionFields = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"]=="Condition"].index.tolist()
            for i, iCondition in enumerate(self._ConditionFields):
                self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+101))
                iConditionVal = self._Owner._FactorInfo.loc[iCondition, "Supplementary"]
                if pd.isnull(iConditionVal) or (isinstance(iConditionVal, str) and (iConditionVal.lower() in ("", "nan"))):
                    self[iCondition] = ""
                else:
                    self[iCondition] = str(iConditionVal).strip()
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        self._TablePrefix = table_prefix
        self._TableInfo = table_info
        self._FactorInfo = factor_info
        self._SecurityInfo = security_info
        self._ExchangeInfo = exchange_info
        self._QS_IgnoredGroupArgs = ("遍历模式", "批量模式")
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        if not self._QSArgs.DateFmt:
            self._DTFormat = "'%Y-%m-%d'"
        else:
            self._DTFormat = f"'{self._QSArgs.DateFmt}'"
        if not self._QSArgs.DTFmt:
            self._DTFormat_WithTime = "'%Y-%m-%d %H:%M:%S'"
        else:
            self._DTFormat_WithTime = f"'{self._QSArgs.DTFmt}'"
        # 解析主表
        self._DBTableName = self._TablePrefix + str(self._TableInfo.loc["DBTableName"])
        self._MainTableName = self._TableInfo.get("MainTableName", None)
        self._IDField = self._FactorInfo[self._FactorInfo["FieldType"]=="ID"].index
        self._IDField = (self._IDField[0] if self._IDField.shape[0]>0 else None)
        if self._IDField is None:# 该表无 ID 字段, 比如 SQL_TimeSeriesTable
            self._IDFieldIsStr = True
            self._MainTableName = self._DBTableName
            self._MainTableID = None
            self._MainTableCondition = None
        elif pd.isnull(self._MainTableName):
            self._IDFieldIsStr = (self.__QS_identifyDataType__(self._FactorInfo["DataType"].loc[self._IDField])!="double")
            self._MainTableName = self._DBTableName
            self._MainTableID = self._FactorInfo["DBFieldName"].loc[self._IDField]
            self._MainTableCondition = None
        else:
            self._MainTableName = self._TablePrefix + self._MainTableName
            self._MainTableID = self._TableInfo.loc["MainTableID"]
            if self._MainTableName==self._DBTableName:
                self._IDFieldIsStr = (self.__QS_identifyDataType__(self._FactorInfo["DataType"].loc[self._IDField])!="double")
            else:
                self._IDFieldIsStr = True
            self._JoinCondition = self._TableInfo.loc["JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
            self._MainTableCondition = self._TableInfo.loc["MainTableCondition"]
            if pd.notnull(self._MainTableCondition):
                self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+str(iFactor._QSArgs[iArgName]) for iArgName in iFactor._QSArgs.ArgNames if iArgName not in self._QS_IgnoredGroupArgs])
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {
                    "FactorNames":[iFactor.Name],
                    "RawFactorNames":{iFactor._NameInFT},
                    "StartDT":operation_mode._FactorStartDT[iFactor.Name],
                    "args":iFactor.Args.to_dict()
                }
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
                if "回溯天数" in ConditionGroup[iConditions]["args"]:
                    ConditionGroup[iConditions]["args"]["回溯天数"] = max(ConditionGroup[iConditions]["args"]["回溯天数"], iFactor._QSArgs.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    def __QS_identifyDataType__(self, field_data_type):
        field_data_type = field_data_type.lower()
        if (field_data_type.find("num")!=-1) or (field_data_type.find("int")!=-1) or (field_data_type.find("decimal")!=-1) or (field_data_type.find("double")!=-1) or (field_data_type.find("float")!=-1) or (field_data_type.find("real")!=-1):
            return "double"
        elif (field_data_type.find("char")!=-1) or (field_data_type.find("text")!=-1) or (field_data_type.find("str")!=-1):
            return "string"
        else:
            return "object"
    def __QS_adjustID__(self, ids):
        return ids
    def __QS_restoreID__(self, ids):
        return ids
    # dts: Series
    def __QS_adjustDT__(self, dts, args={}):
        DTFmt = args.get("时点格式", self._QSArgs.DTFmt)
        if DTFmt:
            return dts.apply(lambda d: dt.datetime.strptime(str(d), DTFmt) if d else pd.NaT)
        return dts
    def __QS_toDate__(self, field):
        return self._FactorDB._SQLFun.get("toDate", "%s") % field
    def _genIDSQLStr(self, ids, init_keyword="AND", args={}):
        IDField = args.get("ID字段", self._QSArgs.IDField)
        if IDField is not None:
            IDFieldIsStr = (self.__QS_identifyDataType__(self._FactorInfo["DataType"].loc[IDField])!="double")
            IDField = self._DBTableName+"."+self._FactorInfo.loc[IDField, "DBFieldName"]
        else:
            if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
                IDField = self._DBTableName+"."+self._FactorInfo.loc[self._IDField, "DBFieldName"]
            else:
                IDField = self._MainTableName+"."+self._MainTableID
            IDFieldIsStr = self._IDFieldIsStr
        if ids is not None:
            ids = self.__QS_adjustID__(ids)
            if args.get("预筛选ID", self._QSArgs.PreFilterID):
                SQLStr = init_keyword + " (" + genSQLInCondition(IDField, ids, is_str=IDFieldIsStr, max_num=1000) + ")"
            elif IDFieldIsStr:
                SQLStr = f"{init_keyword} ({IDField} >= '{min(ids)}' AND {IDField} <= '{max(ids)}')"
            else:
                ids = np.array(ids).astype(int)
                SQLStr = f"{init_keyword} ({IDField} >= {np.min(ids)} AND {IDField} <= {np.max(ids)})"
        else:
            SQLStr = init_keyword + " " + IDField + " IS NOT NULL"
        # if (ids is not None) and args.get("预筛选ID", self._QSArgs.PreFilterID):
        #     SQLStr = init_keyword+" ("+genSQLInCondition(IDField, self.__QS_adjustID__(ids), is_str=IDFieldIsStr, max_num=1000)+")"
        # else:
        #     SQLStr = init_keyword+" "+IDField+" IS NOT NULL"
        return SQLStr
    def _genFromSQLStr(self, setable_join_str=[], use_main_table=True, args={}):
        SQLStr = "FROM "+self._DBTableName+" "
        UseIndex = args.get("使用索引", self._QSArgs.UseIndex)
        ForceIndex = args.get("强制索引", self._QSArgs.ForceIndex)
        if UseIndex and ForceIndex: raise __QS_Error__(f"因子表 {self.Name} 在形成 SQL 查询时错误: 不能同时赋值参数 '使用索引': {str(UseIndex)} 和 '强制索引': {str(ForceIndex)}")
        elif ForceIndex: SQLStr += f"FORCE INDEX ({', '.join(ForceIndex)}) "
        elif UseIndex: SQLStr += f"USE INDEX ({', '.join(UseIndex)}) "
        IgnoreIndex = args.get("忽略索引", self._QSArgs.IgnoreIndex)
        if IgnoreIndex: SQLStr += f"IGNORE INDEX ({', '.join(IgnoreIndex)}) "
        for iJoinStr in setable_join_str: SQLStr += iJoinStr+" "
        if use_main_table and (self._DBTableName!=self._MainTableName) and (args.get("ID字段", self._QSArgs.IDField) is None):
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
        return SQLStr[:-1]
    def _getIDField(self, args={}):
        IDField = args.get("ID字段", self._QSArgs.IDField)
        if IDField is not None:
            RawIDField = self._DBTableName+"."+self._FactorInfo.loc[IDField, "DBFieldName"]
            if self.__QS_identifyDataType__(self._FactorInfo["DataType"].loc[IDField])=="string":
                return RawIDField
            else:
                return "CAST("+RawIDField+" AS CHAR)"
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            RawIDField = self._DBTableName+"."+self._FactorInfo.loc[self._IDField, "DBFieldName"]
            if not self._IDFieldIsStr: RawIDField = "CAST("+RawIDField+" AS CHAR)"
        else:
            RawIDField = self._MainTableName+"."+self._MainTableID
        DefaultSuffix = self._TableInfo.get("DefaultSuffix", None)
        Exchange = self._TableInfo.get("Exchange", None)
        SecurityCategory = self._TableInfo.get("SecurityCategory", None)
        Suffix = "{ElseSuffix}"
        if pd.notnull(SecurityCategory):
            SecuCategoryField, SecuCategoryCodes = SecurityCategory.split(":")
            if self._MainTableName is None:
                SecuCategoryField = self._DBTableName + "." + SecuCategoryField
            else:
                SecuCategoryField = self._MainTableName + "." + SecuCategoryField
            SecuCategoryCodes = SecuCategoryCodes.split(",")
            SecurityInfo = self._SecurityInfo
            iSuffix = "CASE "+SecuCategoryField+" "
            for iCode in SecuCategoryCodes:
                iSuffix += "WHEN "+iCode+" THEN '"+SecurityInfo.loc[iCode, "Suffix"]+"' "
            iSuffix += "ELSE {ElseSuffix} END"
            Suffix = Suffix.format(ElseSuffix=iSuffix)
        if pd.notnull(Exchange):
            ExchangeField, ExchangeCodes = Exchange.split(":")
            if self._MainTableName is None:
                ExchangeField = self._DBTableName + "." + ExchangeField
            else:
                ExchangeField = self._MainTableName + "." + ExchangeField
            ExchangeCodes = ExchangeCodes.split(",")
            ExchangeInfo = self._ExchangeInfo
            iSuffix = "CASE "+ExchangeField+" "
            for iCode in ExchangeCodes:
                iSuffix += "WHEN "+iCode+" THEN '"+ExchangeInfo.loc[iCode, "Suffix"]+"' "
            iSuffix += "ELSE {ElseSuffix} END"
            Suffix = Suffix.format(ElseSuffix=iSuffix)
        Suffix = Suffix.format(ElseSuffix=("''" if pd.isnull(DefaultSuffix) else "'"+DefaultSuffix+"'"))
        if Suffix=="''": return RawIDField
        else: return "CONCAT("+RawIDField+", "+Suffix+")"
    def _adjustRawDataByRelatedField(self, raw_data, fields):
        if "RelatedSQL" not in self._FactorInfo: return raw_data
        RelatedFields = self._FactorInfo["RelatedSQL"].reindex(index=fields)
        RelatedFields = RelatedFields[pd.notnull(RelatedFields)]
        if RelatedFields.shape[0]==0: return raw_data
        for iField in RelatedFields.index:
            iOldData = raw_data.pop(iField)
            iOldDataType = self.__QS_identifyDataType__(self._FactorInfo.loc[iField[:-2], "DataType"])
            iDataType = self.__QS_identifyDataType__(self._FactorInfo.loc[iField, "DataType"])
            if iDataType=="double":
                iNewData = pd.Series(np.nan, index=raw_data.index, dtype="float")
            else:
                iNewData = pd.Series(np.full(shape=(raw_data.shape[0], ), fill_value=None, dtype="O"), index=raw_data.index, dtype="O")
            iSQLStr = self._FactorInfo.loc[iField, "RelatedSQL"]
            if iSQLStr[0]=="{":
                iMapInfo = eval(iSQLStr).items()
            else:
                iStartIdx = iSQLStr.find("{KeyCondition}")
                if iStartIdx!=-1:
                    iEndIdx = iSQLStr[iStartIdx:].find(" ")
                    if iEndIdx==-1: iEndIdx = len(iSQLStr)
                    else: iEndIdx += iStartIdx
                    iStartIdx += 14
                    KeyField = iSQLStr[iStartIdx:iEndIdx]
                    iKeys = iOldData[pd.notnull(iOldData)].unique().tolist()
                    if iKeys:
                        KeyCondition = genSQLInCondition(KeyField, iKeys, is_str=(iOldDataType!="double"))
                    else:
                        KeyCondition = KeyField+" IN (NULL)"
                    iSQLStr = iSQLStr.replace("{KeyCondition}"+KeyField, "{KeyCondition}")
                else:
                    KeyCondition = ""
                if iSQLStr.find("{Keys}")!=-1:
                    if iOldDataType!="double":
                        Keys = "'"+"', '".join([str(iKey) for iKey in iOldData[pd.notnull(iOldData)].unique()])+"'"
                    else:
                        Keys = ", ".join([str(iKey) for iKey in iOldData[pd.notnull(iOldData)].unique()])
                    if not Keys: Keys = "NULL"
                else:
                    Keys = ""
                iMapInfo = self._FactorDB.fetchall(iSQLStr.format(TablePrefix=self._TablePrefix, Keys=Keys, KeyCondition=KeyCondition))
            for jVal, jRelatedVal in iMapInfo:
                if pd.notnull(jVal):
                    if iOldDataType!="double":
                        iNewData[iOldData==str(jVal)] = jRelatedVal
                    elif isinstance(jVal, str):
                        iNewData[iOldData==float(jVal)] = jRelatedVal
                    else:
                        iNewData[iOldData==jVal] = jRelatedVal
                else:
                    iNewData[pd.isnull(iOldData)] = jRelatedVal
            raw_data[iField] = iNewData
        return raw_data
    def _genFieldSQLStr(self, factor_names):
        SQLStr = ""
        JoinStr = []
        SETables = set()
        for iField in factor_names:
            iInfo = self._FactorInfo.loc[iField, "Supplementary"]
            if isinstance(iInfo, str) and (iInfo.find("从表")!=-1):
                iInfo = iInfo.split(":")
                iSETable, iJoinField = iInfo[-2:]
                SQLStr += iSETable+"."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
                if iSETable not in SETables:
                    JoinStr.append("LEFT JOIN "+iSETable+" ON "+self._DBTableName+".ID="+iSETable+"."+iJoinField)
                    SETables.add(iSETable)
            else:
                SQLStr += self._DBTableName+"."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
        return (SQLStr[:-2], JoinStr)
    def _genConditionSQLStr(self, use_main_table=True, init_keyword="AND", args={}):
        FilterStr = args.get("筛选条件", self._QSArgs.FilterCondition)
        if FilterStr:
            SQLStr = init_keyword+" "+FilterStr.format(Table=self._DBTableName, TablePrefix=self._TablePrefix)+" "
            init_keyword = "AND"
        else: SQLStr = ""
        for iConditionField in self._QSArgs._ConditionFields:
            iConditionVal = args.get(iConditionField, self._QSArgs[iConditionField])
            if iConditionVal:
                if self.__QS_identifyDataType__(self._FactorInfo.loc[iConditionField, "DataType"])!="double":
                    SQLStr += init_keyword+" "+self._DBTableName+"."+self._FactorInfo.loc[iConditionField, "DBFieldName"]+" IN ('"+"','".join(iConditionVal.split(","))+"') "
                else:
                    SQLStr += init_keyword+" "+self._DBTableName+"."+self._FactorInfo.loc[iConditionField, "DBFieldName"]+" IN ("+iConditionVal+") "
                init_keyword = "AND"
        if use_main_table and pd.notnull(self._MainTableCondition) and (args.get("ID字段", self._QSArgs.IDField) is None): SQLStr += init_keyword+" "+self._MainTableCondition+" "
        return SQLStr[:-1]
    def getCondition(self, icondition, ids=None, dts=None, args={}):
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+self._FactorInfo.loc[icondition, "DBFieldName"]+" "
        SQLStr += self._genFromSQLStr(args=args)+" "
        SQLStr += self._genIDSQLStr(ids, init_keyword="WHERE", args=args)+" "
        if (dts is not None) and hasattr(self._QSArgs, "DTField"):
            DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
            SQLStr += "AND ("+genSQLInCondition(DTField, [iDT.strftime(self._DTFormat) for iDT in dts], is_str=False, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+self._FactorInfo.loc[icondition, "DBFieldName"]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getMetaData(self, key=None, args={}):
        if key is None:
            return self._TableInfo.copy()
        else:
            return self._TableInfo.get(key, None)
    @property
    def FactorNames(self):
        return self._FactorInfo[pd.notnull(self._FactorInfo["FieldType"])].index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            return self._FactorInfo["DataType"].loc[factor_names].apply(self.__QS_identifyDataType__)
        elif key=="Description":
            return self._FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args),
                                 "Description":self.getFactorMetaData(factor_names, key="Description", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 新方法: 读取 SQL 数据, 返回 DataFrame
    def readSQLData(self, factor_names, ids=None, start_dt=None, end_dt=None, args={}):
        args = args.copy()
        if ids is None: args["预筛选ID"] = False
        return self.__QS_prepareRawData__(factor_names, ids, [start_dt, end_dt], args=args)

# 基于 SQL 数据库表的宽因子表
# 一个字段标识 ID, 一个字段标识时点, 其余字段为因子
# 回溯期数为 None 的算法的前提是一个截止时点不能对应多个公告时点，不满足则将回溯期数设为 0
class SQL_WideTable(SQL_Table):
    """SQL 宽因子表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
        OnlyStartLookBack = Enum(False, True, label="只起始日回溯", arg_type="Bool", order=1)
        OnlyLookBackNontarget = Enum(False, True, label="只回溯非目标日", arg_type="Bool", order=2)
        OnlyLookBackDT = Enum(False, True, label="只回溯时点", arg_type="Bool", order=3)
        #PublDTField = Enum(None, label="公告时点字段", arg_type="SingleOption", order=4)
        IgnoreTime = Enum(False, True, label="忽略时间", arg_type="Bool", order=5)
        EndDateASC = Enum(False, True, label="截止日期递增", arg_type="Bool", order=6)
        OrderFields = List(arg_type="List", label="排序字段", order=7)# [("字段名", "ASC" 或者 "DESC")]
        MultiMapping = Enum(False, True, label="多重映射", arg_type="Bool", order=8)
        Operator = Either(Callable(), None, arg_type="Function", label="算子", order=9)
        OperatorDataType = Enum("object", "double", "string", arg_type="SingleOption", label="算子数据类型", order=10, option_range=["object", "double", "string"])
        AdditionalFields = ListStr(arg_type="ListStr", label="附加字段", order=11)
        PeriodLookBack = Either(None, Int(0), label="回溯期数", arg_type="Integer", order=12)
        RawLookBack = Float(0, arg_type="Integer", label="原始值回溯天数", order=13)
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            # 解析公告时点字段
            Fields = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"].str.lower().str.contains("date")].index.tolist()# 所有的时点字段列表
            Fields += [None]
            self.add_trait("PublDTField", Enum(*Fields, arg_type="SingleOption", label="公告时点字段", order=4, option_range=Fields))
            PublDTField = self._Owner._FactorInfo["DBFieldName"][self._Owner._FactorInfo["FieldType"]=="AnnDate"]
            if PublDTField.shape[0]==0: self.PublDTField = None
            else: self.PublDTField = PublDTField.index[0]
            # 解析排序字段
            Fields = self._Owner._FactorInfo[self._Owner._FactorInfo["Supplementary"]=="OrderField"].index.tolist()# 所有的排序字段列表
            self.OrderFields = [(iField, "ASC") for iField in Fields]
    
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "批量模式", "回溯天数", "只起始日回溯", "只回溯非目标日", "只回溯时点", "算子", "算子数据类型", "多重映射","原始值回溯天数")
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+str(iFactor._QSArgs[iArgName]) for iArgName in iFactor._QSArgs.ArgNames if iArgName not in self._QS_IgnoredGroupArgs])
            if iFactor._QSArgs["回溯期数"] is None:
                iConditions += ";回溯期数:None"
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {"FactorNames":[iFactor.Name], 
                                                       "RawFactorNames":{iFactor._NameInFT}, 
                                                       "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                       "args":iFactor.Args.to_dict()}
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
                ConditionGroup[iConditions]["args"]["回溯天数"] = max(ConditionGroup[iConditions]["args"]["回溯天数"], iFactor._QSArgs.LookBack)
                ConditionGroup[iConditions]["args"]["原始值回溯天数"] = max(ConditionGroup[iConditions]["args"]["原始值回溯天数"], iFactor._QSArgs.RawLookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if key=="DataType":
            if not args.get("多重映射", self._QSArgs.MultiMapping): return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
            if factor_names is None: factor_names = self.FactorNames
            if args.get("算子", self._QSArgs.Operator) is None:
                return pd.Series(["object"]*len(factor_names), index=factor_names)
            else:
                return pd.Series([args.get("算子数据类型", self._QSArgs.OperatorDataType)]*len(factor_names), index=factor_names)
        else:
            return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr(args=args)+" "
        if idt is not None: SQLStr += "WHERE "+DTField+"="+idt.strftime(self._DTFormat_WithTime)+" "
        else: SQLStr += "WHERE "+DTField+" IS NOT NULL "
        SQLStr += "AND "+IDField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID"
        return self.__QS_restoreID__([iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)])
    # 返回在给定 ID iid 的有数据记录的时间点 如果 iid 为
    # None, 将返回所有有历史数据记录的时间点 忽略
    # ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+DTField+" "
        SQLStr += self._genFromSQLStr(args=args)+" "
        SQLStr += "WHERE "+DTField+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DTField+">="+start_dt.strftime(self._DTFormat_WithTime)+" "
        if end_dt is not None: SQLStr += "AND "+DTField+"<="+end_dt.strftime(self._DTFormat_WithTime)+" "
        if iid is not None: iid = [iid]
        SQLStr += self._genIDSQLStr(iid, args=args)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY "+DTField
        Rslt = pd.DataFrame(self._FactorDB.fetchall(SQLStr), dtype="O")
        if Rslt.empty:
            return []
        else:
            return self.__QS_adjustDT__(Rslt.iloc[:, 0], args=args).tolist()
    def _genNullIDSQLStr_WithPublDT(self, factor_names, ids, end_date, args={}):
        EndDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        AnnDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("公告时点字段", self._QSArgs.PublDTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        IgnoreTime = args.get("忽略时间", self._QSArgs.IgnoreTime)
        if IgnoreTime:
            DTFormat = self._DTFormat
            AdjAnnDTField = self.__QS_toDate__(AnnDTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjAnnDTField = AnnDTField
        SubSQLStr = "SELECT "+IDField+" AS ID, "
        SubSQLStr += "MAX("+EndDTField+") AS MaxEndDate "
        SubSQLStr += self._genFromSQLStr(use_main_table=False, args=args)+" "
        SubSQLStr += "WHERE ("+AdjAnnDTField+"<"+end_date.strftime(DTFormat)+" "
        SubSQLStr += "AND "+EndDTField+"<"+end_date.strftime(DTFormat)+") "
        SubSQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            SubSQLStr += self._genIDSQLStr(ids, args=args)+" "
        SubSQLStr += "GROUP BY "+IDField
        if IgnoreTime:
            SQLStr = "SELECT "+self.__QS_toDate__("CASE WHEN "+AnnDTField+">=t.MaxEndDate THEN "+AnnDTField+" ELSE t.MaxEndDate END")+" AS DT, "
        else:
            SQLStr = "SELECT CASE WHEN "+AnnDTField+">=t.MaxEndDate THEN "+AnnDTField+" ELSE t.MaxEndDate END AS DT, "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t.ID="+IDField+" "
        SQLStr += "AND "+EndDTField+"=t.MaxEndDate) "
        SQLStr += "WHERE "+AdjAnnDTField+"<"+end_date.strftime(DTFormat)+" "
        if not ((self._MainTableName is None) or (self._MainTableName==self._DBTableName)):
            SQLStr += self._genIDSQLStr(ids, args=args)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)
        return SQLStr
    def _prepareRawData_WithPublDT(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        EndDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        AnnDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("公告时点字段", self._QSArgs.PublDTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        IgnoreTime = args.get("忽略时间", self._QSArgs.IgnoreTime)
        if IgnoreTime:
            DTFormat = self._DTFormat
            AdjAnnDTField = self.__QS_toDate__(AnnDTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjAnnDTField = AnnDTField
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        LookBack = args.get("回溯天数", self._QSArgs.LookBack)
        if (StartDT is not None) and (not np.isinf(LookBack)): StartDT -= dt.timedelta(LookBack)
        SubSQLStr = "SELECT "+IDField+" AS ID, "
        GroupAnnDTField = "CASE WHEN "+AnnDTField+">="+EndDTField+" THEN "+AnnDTField+" ELSE "+EndDTField+" END"
        if IgnoreTime: GroupAnnDTField = self.__QS_toDate__(GroupAnnDTField)
        SubSQLStr += GroupAnnDTField+" AS AnnDate, "
        SubSQLStr += "MAX("+EndDTField+") AS MaxEndDate "
        SubSQLStr += self._genFromSQLStr(use_main_table=False, args=args)+f" WHERE {'TRUE' if self.FactorDB._QSArgs.DBType!='Oracle' else '(1=1)'} "
        if StartDT is not None:
            SubSQLStr += "AND ("+AdjAnnDTField+">="+StartDT.strftime(DTFormat)+" "
            SubSQLStr += "OR "+EndDTField+">="+StartDT.strftime(DTFormat)+") "
        if EndDT is not None:
            SubSQLStr += "AND ("+AdjAnnDTField+"<="+EndDT.strftime(DTFormat)+" "
            SubSQLStr += "AND "+EndDTField+"<="+EndDT.strftime(DTFormat)+") "
        SubSQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            SubSQLStr += self._genIDSQLStr(ids, args=args)+" "
        SubSQLStr += "GROUP BY "+IDField+f", {GroupAnnDTField if self.FactorDB._QSArgs.DBType=='Oracle' else 'AnnDate'}"
        SQLStr = "SELECT t.AnnDate AS DT, "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t.ID="+IDField+" "
        SQLStr += "AND t.MaxEndDate="+EndDTField+") "
        SQLStr += "WHERE t.AnnDate>="+AdjAnnDTField+" "
        if not ((self._MainTableName is None) or (self._MainTableName==self._DBTableName)):
            SQLStr += self._genIDSQLStr(ids, args=args)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID, DT"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID", "MaxEndDate"]+factor_names)
        else:
            RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["QS_DT", "ID", "MaxEndDate"]+factor_names)
            RawData["ID"] = self.__QS_restoreID__(RawData["ID"])
            RawData["QS_DT"] = self.__QS_adjustDT__(RawData["QS_DT"], args=args)
            RawData["MaxEndDate"] = self.__QS_adjustDT__(RawData["MaxEndDate"], args=args)
        if (StartDT is not None) and np.isinf(LookBack):
            if ids is None: ids = self.getID(args=args)
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==StartDT]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr_WithPublDT(factor_names, list(NullIDs), StartDT, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "ID", "MaxEndDate"]+factor_names)
                    NullRawData["ID"] = self.__QS_restoreID__(NullRawData["ID"])
                    NullRawData["QS_DT"] = self.__QS_adjustDT__(NullRawData["QS_DT"], args=args)
                    NullRawData["MaxEndDate"] = self.__QS_adjustDT__(NullRawData["MaxEndDate"], args=args)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "QS_DT"])
        if RawData.shape[0]==0: return RawData.loc[:, ["QS_DT", "ID"]+factor_names]
        if args.get("截止日期递增", self._QSArgs.EndDateASC):# 删除截止日期非递增的记录
            #DTRank = RawData.loc[:, ["ID", "QS_DT", "MaxEndDate"]].set_index(["ID"]).astype(np.datetime64).groupby(axis=0, level=0).rank(method="min")
            #RawData = RawData[(DTRank["QS_DT"]<=DTRank["MaxEndDate"]).values]
            DTRank = RawData.loc[:, ["ID", "MaxEndDate"]].set_index(["ID"]).astype(np.datetime64).groupby(axis=0, level=0).rank(method="min")["MaxEndDate"]
            RawData = RawData[DTRank.values>=DTRank.groupby(axis=0, level=0).cummax().values]
        return self._adjustRawDataByRelatedField(RawData.loc[:, ["QS_DT", "ID"]+factor_names], factor_names)
    def _genNullIDSQLStr_IgnorePublDT(self, factor_names, ids, end_date, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        if args.get("忽略时间", self._QSArgs.IgnoreTime):
            DTFormat = self._DTFormat
            AdjDTField = self.__QS_toDate__(DTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjDTField = DTField
        IDField = args.get("ID字段", self._QSArgs.IDField)
        if IDField is None:
            IDField = self._MainTableName+"."+self._MainTableID
        else:
            IDField = self._DBTableName+"."+self._FactorInfo.loc[IDField, "DBFieldName"]
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAX("+AdjDTField+") "
        SubSQLStr += self._genFromSQLStr(args=args)+" "
        SubSQLStr += "WHERE "+AdjDTField+"<"+end_date.strftime(DTFormat)+" "
        SubSQLStr += self._genIDSQLStr(ids, args=args)+" "
        ConditionSQLStr = self._genConditionSQLStr(use_main_table=True, args=args)
        SubSQLStr += ConditionSQLStr+" "
        SubSQLStr += "GROUP BY "+IDField
        SQLStr = "SELECT "+AdjDTField+", "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += "WHERE ("+IDField+", "+AdjDTField+") IN ("+SubSQLStr+") "
        SQLStr += ConditionSQLStr
        return SQLStr
    def _prepareRawData_IgnorePublDT(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        if args.get("忽略时间", self._QSArgs.IgnoreTime):
            DTFormat = self._DTFormat
            AdjAnnDTField = self.__QS_toDate__(DTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjAnnDTField = DTField
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        LookBack = args.get("回溯天数", self._QSArgs.LookBack)
        if (StartDT is not None) and (not np.isinf(LookBack)): StartDT -= dt.timedelta(LookBack)
        # 形成 SQL 语句, 时点, ID, 因子数据
        SQLStr = "SELECT "+AdjAnnDTField+", "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += self._genIDSQLStr(ids, init_keyword="WHERE", args=args)+" "
        if StartDT is not None:
            SQLStr += "AND "+AdjAnnDTField+">="+StartDT.strftime(DTFormat)+" "
        if EndDT is not None:
            SQLStr += "AND "+AdjAnnDTField+"<="+EndDT.strftime(DTFormat)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID, "+DTField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["QS_DT", "ID"]+factor_names)
            RawData["ID"] = self.__QS_restoreID__(RawData["ID"])
            RawData["QS_DT"] = self.__QS_adjustDT__(RawData["QS_DT"], args=args)
        if (StartDT is not None) and np.isinf(LookBack):
            if ids is None: ids = self.getID(args=args)
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==StartDT]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr_IgnorePublDT(factor_names, list(NullIDs), StartDT, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "ID"]+factor_names)
                    NullRawData["ID"] = self.__QS_restoreID__(NullRawData["ID"])
                    NullRawData["QS_DT"] = self.__QS_adjustDT__(NullRawData["QS_DT"], args=args)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "QS_DT"])
        if RawData.shape[0]==0: return RawData
        return self._adjustRawDataByRelatedField(RawData, factor_names)
    def _prepareRawData_PeriodLookBack(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IgnoreTime = args.get("忽略时间", self._QSArgs.IgnoreTime)
        if IgnoreTime: DTFormat = self._DTFormat
        else: DTFormat = self._DTFormat_WithTime
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        RawLookBack = args.get("原始值回溯天数", self._QSArgs.RawLookBack)
        if (StartDT is not None) and (not np.isinf(RawLookBack)): StartDT -= dt.timedelta(RawLookBack)
        SQLStr = "SELECT "+self._getIDField(args=args)+" AS ID, "
        EndDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        SQLStr += EndDTField+" AS QS_EndDT, "
        AnnDTField = args.get("公告时点字段", self._QSArgs.PublDTField)
        if AnnDTField is not None:
            AnnDTField = self._DBTableName+"."+self._FactorInfo.loc[AnnDTField, "DBFieldName"]
            if IgnoreTime:
                SQLStr += self.__QS_toDate__("CASE WHEN "+AnnDTField+">="+EndDTField+" THEN "+AnnDTField+" ELSE "+EndDTField+" END")+" AS QS_DT, "
                AdjAnnDTField = self.__QS_toDate__(AnnDTField)
            else:
                SQLStr += "CASE WHEN "+AnnDTField+">="+EndDTField+" THEN "+AnnDTField+" ELSE "+EndDTField+" END AS QS_DT, "
                AdjAnnDTField = AnnDTField
        else:
            AnnDTField = EndDTField
            if IgnoreTime:
                AdjAnnDTField = self.__QS_toDate__(AnnDTField)
            else:
                AdjAnnDTField = AnnDTField
            SQLStr += AnnDTField+" AS QS_DT, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += self._genIDSQLStr(ids, init_keyword="WHERE", args=args)+" "
        if EndDT is not None:
            SQLStr += "AND "+EndDTField+"<="+EndDT.strftime(DTFormat)+" "
        if AnnDTField!=EndDTField:
            if EndDT is not None:
                SQLStr += "AND "+AdjAnnDTField+"<="+EndDT.strftime(DTFormat)+" "
            if (StartDT is not None) and (not np.isinf(RawLookBack)):
                SQLStr += "AND ("+AdjAnnDTField+">="+StartDT.strftime(DTFormat)+" "
                SQLStr += "OR "+EndDTField+">="+StartDT.strftime(DTFormat)+") "
        elif (StartDT is not None) and (not np.isinf(RawLookBack)):
            SQLStr += "AND "+EndDTField+">="+StartDT.strftime(DTFormat)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID, QS_DT, QS_EndDT"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "QS_EndDT", "QS_DT"]+factor_names)
        RawData["QS_DT"] = self.__QS_adjustDT__(RawData["QS_DT"], args=args).astype(np.datetime64)
        RawData["QS_EndDT"] = self.__QS_adjustDT__(RawData["QS_EndDT"], args=args).astype(np.datetime64)
        # 回溯期数
        RawData["QS_EndDTPeriod"] = RawData.loc[:, ["ID", "QS_EndDT"]].set_index(["ID"]).groupby(axis=0, level=0).rank(method="dense").values
        RawData["QS_TargetPeriod"] = RawData["QS_EndDTPeriod"] - args.get("回溯期数", self._QSArgs.PeriodLookBack)
        TargetPeriod = RawData.loc[:, ["ID","QS_DT","QS_TargetPeriod"]].groupby(by=["ID", "QS_DT"]).max().reset_index()
        if args.get("截止日期递增", self._QSArgs.EndDateASC):
            #TargetPeriod = TargetPeriod[TargetPeriod["QS_TargetPeriod"]>=TargetPeriod.groupby(["ID"])["QS_TargetPeriod"].cummax().values]
            TargetPeriod["QS_TargetPeriod"] = TargetPeriod.groupby(["ID"])["QS_TargetPeriod"].cummax().values
        RawData = pd.merge(TargetPeriod, RawData.loc[:, ["ID","QS_DT","QS_EndDTPeriod"]+factor_names],
                           left_on=["ID", "QS_TargetPeriod"], right_on=["ID", "QS_EndDTPeriod"], how="inner", suffixes=("", "_y"))
        RawData = RawData[RawData["QS_DT"]>=RawData["QS_DT_y"]]
        if RawData.shape[0]==0: return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        MaxAnnDT = RawData.loc[:, ["ID","QS_DT","QS_TargetPeriod", "QS_DT_y"]].groupby(by=["ID", "QS_DT", "QS_TargetPeriod"]).max().reset_index()
        RawData = pd.merge(MaxAnnDT, RawData.loc[:, ["ID","QS_DT","QS_TargetPeriod","QS_DT_y"]+factor_names],
                           left_on=["ID","QS_DT","QS_TargetPeriod","QS_DT_y"], right_on=["ID","QS_DT","QS_TargetPeriod","QS_DT_y"], how="inner", suffixes=("", "_y"))
        RawData["ID"] = self.__QS_restoreID__(RawData["ID"])
        return self._adjustRawDataByRelatedField(RawData.loc[:, ["QS_DT", "ID"]+factor_names], factor_names)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if args.get("多重映射", self._QSArgs.MultiMapping):
            OrderFields = args.get("排序字段", self._QSArgs.OrderFields)
            if OrderFields:
                OrderFields, Orders = np.array(OrderFields).T.tolist()
            else:
                OrderFields, Orders = [], []
        else:
            OrderFields, Orders = [], []
        FactorNames = list(set(factor_names).union(OrderFields).union(args.get("附加字段", self._QSArgs.AdditionalFields)))
        if args.get("回溯期数", self._QSArgs.PeriodLookBack) is not None:
            RawData = self._prepareRawData_PeriodLookBack(factor_names=FactorNames, ids=ids, dts=dts, args=args)
        elif args.get("公告时点字段", self._QSArgs.PublDTField) is None:
            RawData = self._prepareRawData_IgnorePublDT(factor_names=FactorNames, ids=ids, dts=dts, args=args)
        else:
            RawData = self._prepareRawData_WithPublDT(factor_names=FactorNames, ids=ids, dts=dts, args=args)
        RawData = RawData.sort_values(by=["ID", "QS_DT"]+OrderFields, ascending=[True, True]+[(iOrder.lower()=="asc") for iOrder in Orders])
        return RawData.loc[:, ["QS_DT", "ID"]+list(set(factor_names).union(args.get("附加字段", self._QSArgs.AdditionalFields)))]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Args = self.Args.to_dict()
        Args.update(args)
        ErrorFmt = {"DuplicatedIndex":  "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        return _QS_calcData_WideTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)

# 基于 SQL 数据库表的窄因子表
# 一个字段标识 ID, 一个字段标识时点, 一个字段标识因子名(不存在则固定取标识因子值字段的名称作为因子名), 一个字段标识为因子值
class SQL_NarrowTable(SQL_Table):
    """SQL 窄因子表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
        OnlyStartLookBack = Enum(False, True, label="只起始日回溯", arg_type="Bool", order=1)
        OnlyLookBackNontarget = Enum(False, True, label="只回溯非目标日", arg_type="Bool", order=2)
        OnlyLookBackDT = Enum(False, True, label="只回溯时点", arg_type="Bool", order=3)
        #FactorNameField = Enum(None, arg_type="SingleOption", label="因子名字段", order=4)
        #FactorValueField = Enum(None, arg_type="SingleOption", label="因子值字段", order=5)
        MultiMapping = Enum(True, False, label="多重映射", arg_type="Bool", order=6)
        Operator = Either(Callable(), None, arg_type="Function", label="算子", order=7)
        OperatorDataType = Enum("object", "double", "string", arg_type="SingleOption", label="算子数据类型", order=8, option_range=["object", "double", "string"])
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            FactorFields = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"]=="Factor"]
            if FactorFields.shape[0]==0: FactorFields = self._Owner._FactorInfo
            self.add_trait("FactorNameField", Enum(*FactorFields.index.tolist(), arg_type="SingleOption", label="因子名字段", order=4, option_range=FactorFields.index.tolist()))
            DefaultField = FactorFields[FactorFields["Supplementary"]=="Default"].index
            if DefaultField.shape[0]==0: self.FactorNameField = FactorFields.index[0]
            else: self.FactorNameField = DefaultField[0]
            ValueFields = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"]=="Value"]
            if ValueFields.shape[0]==0: ValueFields = self._Owner._FactorInfo
            self.add_trait("FactorValueField", Enum(*ValueFields.index.tolist(), arg_type="SingleOption", label="因子值字段", order=5, option_range=ValueFields.index.tolist()))
            DefaultField = ValueFields[ValueFields["Supplementary"]=="Default"].index
            if DefaultField.shape[0]==0: self.FactorValueField = ValueFields.index[0]
            else: self.FactorValueField = DefaultField[0]
            self._FactorNames = None# 所有的因子名列表或者对照字典

        @on_trait_change("FactorNameField")
        def _on_FactorNameField_changed(self, obj, name, old, new):
            if self.FactorNameField is not None: self._FactorNames = None
        
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "批量模式", "回溯天数", "只起始日回溯")
    def _getFactorNames(self, factor_field, check_list=False):
        if (factor_field==self._QSArgs.FactorNameField) and (self._QSArgs._FactorNames is not None): return self._QSArgs._FactorNames
        FactorField = self._DBTableName+"."+self._FactorInfo.loc[factor_field, "DBFieldName"]
        if "RelatedSQL" in self._FactorInfo: SQLStr = self._FactorInfo.loc[factor_field, "RelatedSQL"]
        else: SQLStr = None
        if pd.isnull(SQLStr) or (not SQLStr):
            if check_list: return []
            SQLStr = "SELECT DISTINCT "+FactorField+" "+self._genFromSQLStr(use_main_table=False)+" WHERE "+FactorField+" IS NOT NULL ORDER BY "+FactorField
            FactorNames = [str(iName) for iName, in self._FactorDB.fetchall(SQLStr)]
        else:
            SubSQLStr = "SELECT DISTINCT "+FactorField+" "+self._genFromSQLStr(use_main_table=False)+" WHERE "+FactorField+" IS NOT NULL"
            SQLStr = SQLStr.format(TablePrefix=self._TablePrefix, Keys=SubSQLStr)
            FactorNames = {iName:iCode for iCode, iName in self._FactorDB.fetchall(SQLStr)}
        if factor_field==self._QSArgs.FactorNameField: self._QSArgs._FactorNames = FactorNames
        return FactorNames
    @property
    def FactorNames(self):
        if not hasattr(self, "_QSArgs"): return []
        if self._QSArgs._FactorNames is None:
            self._QSArgs._FactorNames = self._getFactorNames(self._QSArgs.FactorNameField)
        if isinstance(self._QSArgs._FactorNames, dict):
            return sorted(self._QSArgs._FactorNames.keys())
        else:
            return self._QSArgs._FactorNames
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if key=="DataType":
            if factor_names is None: factor_names = self.FactorNames
            if not args.get("多重映射", self._QSArgs.MultiMapping):
                return pd.Series(self.__QS_identifyDataType__(self._FactorInfo["DataType"].loc[args.get("因子值字段", self._QSArgs.FactorValueField)]), index=factor_names)
            else:
                if args.get("算子", self._QSArgs.Operator) is None:
                    return pd.Series(["object"]*len(factor_names), index=factor_names)
                else:
                    return pd.Series([args.get("算子数据类型", self._QSArgs.OperatorDataType)]*len(factor_names), index=factor_names)
        elif key is not None:
            if factor_names is None: factor_names = self.FactorNames
            return pd.Series(index=factor_names)
        else:
            return pd.DataFrame(self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args), columns=["DataType"])
    def getID(self, ifactor_name=None, idt=None, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr(args=args)+" "
        if idt is not None: SQLStr += "WHERE "+DTField+"="+idt.strftime(self._DTFormat)+" "
        else: SQLStr += "WHERE "+DTField+" IS NOT NULL "
        SQLStr += "AND "+IDField+" IS NOT NULL "
        FactorNameField = args.get("因子名字段", self._QSArgs.FactorNameField)
        DBFactorField = self._DBTableName+"."+self._FactorInfo.loc[FactorNameField, "DBFieldName"]
        if ifactor_name is not None:
            FactorNames = self._getFactorNames(FactorNameField, check_list=True)
            if isinstance(FactorNames, dict):
                ifactor_name = FactorNames[ifactor_name]
            if self.__QS_identifyDataType__(self._FactorInfo.loc[FactorNameField, "DataType"])!="double":
                SQLStr += "AND "+DBFactorField+"='"+ifactor_name+"' "
            else:
                SQLStr += "AND "+DBFactorField+"="+str(ifactor_name)+" "
        else:
            SQLStr += "AND "+DBFactorField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID"
        return self.__QS_restoreID__([iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)])
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        SQLStr = "SELECT DISTINCT "+DTField+" "
        if iid is not None:
            SQLStr += self._genFromSQLStr(args=args)+" "
            if IDField is None:
                SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+self.__QS_adjustID__([iid])[0]+"' "
            else:
                SQLStr += "WHERE "+self._DBTableName+"."+self._FactorInfo.loc[IDField, "DBFieldName"]+"='"+self.__QS_adjustID__([iid])[0]+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else:
            IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
            SQLStr += self._genFromSQLStr(use_main_table=False, args=args)+" "
            SQLStr += "WHERE "+IDField+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DTField+">="+start_dt.strftime(self._DTFormat)+" "
        if end_dt is not None: SQLStr += "AND "+DTField+"<="+end_dt.strftime(self._DTFormat)+" "
        FactorNameField = args.get("因子名字段", self._QSArgs.FactorNameField)
        DBFactorField = self._DBTableName+"."+self._FactorInfo.loc[FactorNameField, "DBFieldName"]
        if ifactor_name is not None:
            FactorNames = self._getFactorNames(FactorNameField, check_list=True)
            if isinstance(FactorNames, dict):
                ifactor_name = FactorNames[ifactor_name]
            if self.__QS_identifyDataType__(self._FactorInfo.loc[FactorNameField, "DataType"])!="double":
                SQLStr += "AND "+DBFactorField+"='"+ifactor_name+"' "
            else:
                SQLStr += "AND "+DBFactorField+"="+str(ifactor_name)+" "
        else:
            SQLStr += "AND "+DBFactorField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        SQLStr += "ORDER BY "+DTField
        Rslt = pd.DataFrame(self._FactorDB.fetchall(SQLStr), dtype="O")
        if Rslt.empty:
            return []
        else:
            return self.__QS_adjustDT__(Rslt.iloc[:, 0], args=args).tolist()
    def _genNullIDSQLStr(self, factor_names, ids, end_dt, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        FactorNameField = args.get("因子名字段", self._QSArgs.FactorNameField)
        DBFactorField = self._DBTableName+"."+self._FactorInfo.loc[FactorNameField, "DBFieldName"]
        FactorFieldStr = (self.__QS_identifyDataType__(self._FactorInfo.loc[FactorNameField, "DataType"])!="double")
        IDField = args.get("ID字段", self._QSArgs.IDField)
        if IDField is None:
            IDField = self._MainTableName+"."+self._MainTableID
        else:
            IDField = self._DBTableName+"."+self._FactorInfo.loc[IDField, "DBFieldName"]
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAX("+DTField+") "
        SubSQLStr += self._genFromSQLStr(args=args)+" "
        SubSQLStr += "WHERE "+DTField+"<"+end_dt.strftime(self._DTFormat)+" "
        SubSQLStr += self._genIDSQLStr(ids, args=args)+" "
        ConditionSQLStr = self._genConditionSQLStr(use_main_table=True, args=args)
        SubSQLStr += ConditionSQLStr+" "
        SubSQLStr += "GROUP BY "+IDField
        SQLStr = "SELECT "+DTField+", "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += DBFactorField+", "
        SQLStr += self._DBTableName+"."+self._FactorInfo.loc[args.get("因子值字段", self._QSArgs.FactorValueField), "DBFieldName"]+" "
        SQLStr += self._genFromSQLStr(args=args)+" "
        SQLStr += "WHERE ("+IDField+", "+DTField+") IN ("+SubSQLStr+") "
        FactorNames = self._getFactorNames(FactorNameField, check_list=True)
        if isinstance(FactorNames, list):
            SQLStr += "AND ("+genSQLInCondition(DBFactorField, factor_names, is_str=FactorFieldStr, max_num=1000)+") "
        elif isinstance(FactorNames, dict):
            SQLStr += "AND ("+genSQLInCondition(DBFactorField, [FactorNames[iFactor] for iFactor in factor_names], is_str=FactorFieldStr, max_num=1000)+") "
        SQLStr += ConditionSQLStr
        return SQLStr
    def _genSQLStr(self, factor_names, ids, start_dt, end_dt, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        FactorNameField = args.get("因子名字段", self._QSArgs.FactorNameField)
        DBFactorField = self._DBTableName+"."+self._FactorInfo.loc[FactorNameField, "DBFieldName"]
        FactorFieldStr = (self.__QS_identifyDataType__(self._FactorInfo.loc[FactorNameField, "DataType"])!="double")
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DTField+", "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += DBFactorField+", "
        SQLStr += self._DBTableName+"."+self._FactorInfo.loc[args.get("因子值字段", self._QSArgs.FactorValueField), "DBFieldName"]+" "
        SQLStr += self._genFromSQLStr(args=args)+" "
        SQLStr += self._genIDSQLStr(ids, init_keyword="WHERE", args=args)+" "
        if start_dt is not None:
            SQLStr += "AND "+DTField+">="+start_dt.strftime(self._DTFormat_WithTime)+" "
        if end_dt is not None:
            SQLStr += "AND "+DTField+"<="+end_dt.strftime(self._DTFormat_WithTime)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        FactorNames = self._getFactorNames(FactorNameField, check_list=True)
        if isinstance(FactorNames, list):
            SQLStr += "AND ("+genSQLInCondition(DBFactorField, factor_names, is_str=FactorFieldStr, max_num=1000)+") "
        elif isinstance(FactorNames, dict):
            SQLStr += "AND ("+genSQLInCondition(DBFactorField, [FactorNames[iFactor] for iFactor in factor_names], is_str=FactorFieldStr, max_num=1000)+") "
        SQLStr += "ORDER BY ID, "+DTField+", "+DBFactorField
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        LookBack = args.get("回溯天数", self._QSArgs.LookBack)
        if (StartDT is not None) and (not np.isinf(LookBack)): StartDT -= dt.timedelta(LookBack)
        FactorValueField = args.get("因子值字段", self._QSArgs.FactorValueField)
        FactorNameField = args.get("因子名字段", self._QSArgs.FactorNameField)
        RawData = self._FactorDB.fetchall(self._genSQLStr(factor_names, ids, start_dt=StartDT, end_dt=EndDT, args=args))
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID", FactorNameField, FactorValueField])
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["QS_DT", "ID", FactorNameField, FactorValueField])
        RawData["ID"] = self.__QS_restoreID__(RawData["ID"])
        RawData["QS_DT"] = self.__QS_adjustDT__(RawData["QS_DT"], args=args)
        if (StartDT is not None) and np.isinf(LookBack):
            if ids is None: ids = self.getID(args=args)
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==StartDT]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr(factor_names, list(NullIDs), StartDT, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "ID", FactorNameField, FactorValueField])
                    NullRawData["ID"] = self.__QS_restoreID__(NullRawData["ID"])
                    NullRawData["QS_DT"] = self.__QS_adjustDT__(NullRawData["QS_DT"], args=args)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "QS_DT", FactorNameField])
        if RawData.shape[0]==0: return RawData
        return self._adjustRawDataByRelatedField(RawData, [FactorNameField, FactorValueField])
    def __QS_saveRawData__(self, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, **kwargs):
        return super().__QS_saveRawData__(raw_data, [], raw_data_dir, pid_ids, file_name, pid_lock, **kwargs)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Args = self.Args.to_dict()
        Args.update(args)
        ErrorFmt = {"DuplicatedIndex":  "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        return _QS_calcData_NarrowTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)

# 基于 SQL 数据库表的特征因子表
# 一个字段标识 ID, 其余字段为因子
# 如果时点字段为 None, 则忽略目标时点参数; 否则如果目标时点为 None, 则默认以时点字段的最大值作为目标时点
class SQL_FeatureTable(SQL_WideTable):
    """SQL 特征因子表"""
    class __QS_ArgClass__(SQL_WideTable.__QS_ArgClass__):
        LookBack = Float(np.inf, arg_type="Integer", label="回溯天数", order=0)
        TargetDT = Either(None, Date, arg_type="DateTime", label="目标时点", order=1)
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            self.DTField = None
    
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "批量模式", "多重映射", "算子", "算子数据类型")
    def _getMaxDT(self, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        SQLStr = "SELECT MAX("+DTField+") "
        SQLStr += self._genFromSQLStr(args=args)+" "
        SQLStr += "WHERE "+DTField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)
        MaxDT =  pd.DataFrame(self._FactorDB.fetchall(SQLStr), dtype="O")
        if MaxDT.empty: return None
        return self.__QS_adjustDT__(MaxDT.iloc[:, 0], args=args).iloc[0]
    def getID(self, ifactor_name=None, idt=None, args={}):
        DTField = args.get("时点字段", self._QSArgs.DTField)
        if pd.isnull(DTField):
            IDField = args.get("ID字段", self._QSArgs.IDField)
            IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
            SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
            SQLStr += self._genFromSQLStr(args=args)+" "
            SQLStr += "WHERE "+IDField+" IS NOT NULL "
            SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
            SQLStr += "ORDER BY ID"
            return self.__QS_restoreID__([iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)])
        TargetDT = args.get("目标时点", self._QSArgs.TargetDT)
        if TargetDT is None: TargetDT = self._getMaxDT(args=args)
        if TargetDT is None: return []
        return super().getID(ifactor_name=ifactor_name, idt=TargetDT, args=args)
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if ids==[]: return pd.DataFrame(columns=["ID"]+factor_names)
        DTField = args.get("时点字段", self._QSArgs.DTField)
        TargetDT = args.get("目标时点", self._QSArgs.TargetDT)
        if DTField is not None:
            if TargetDT is None: TargetDT = self._getMaxDT(args=args)
            if TargetDT is not None:
                RawData = super().__QS_prepareRawData__(factor_names, ids, [TargetDT], args=args)
                RawData["QS_TargetDT"] = TargetDT
                return RawData
            else:
                return pd.DataFrame(columns=["ID"]+factor_names)
        # 形成SQL语句, ID, 因子数据
        SQLStr = "SELECT "+self._getIDField(args=args)+" AS ID, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += self._genIDSQLStr(ids, init_keyword="WHERE", args=args)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID"]+factor_names)
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        RawData["QS_TargetDT"] = dt.datetime.combine(dt.date.today(), dt.time(0)) + dt.timedelta(1)
        RawData["QS_DT"] = RawData["QS_TargetDT"]
        RawData["ID"] = self.__QS_restoreID__(RawData["ID"])
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        TargetDT = raw_data.pop("QS_TargetDT").iloc[0].to_pydatetime()
        Data = super().__QS_calcData__(raw_data, factor_names, ids, [TargetDT], args=args)
        Data = Data.iloc[:, 0, :]
        return Panel(Data.values.T.reshape((Data.shape[1], Data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=Data.index, minor_axis=dts).swapaxes(1, 2)

# 基于 SQL 数据库表的时序因子表
# 无 ID 字段, 一个字段标识时点, 其余字段为因子
class SQL_TimeSeriesTable(SQL_Table):
    """SQL 时序因子表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
        OnlyStartLookBack = Enum(False, True, label="只起始日回溯", arg_type="Bool", order=1)
        OnlyLookBackNontarget = Enum(False, True, label="只回溯非目标日", arg_type="Bool", order=2)
        OnlyLookBackDT = Enum(False, True, label="只回溯时点", arg_type="Bool", order=3)
        #PublDTField = Enum(None, label="公告时点字段", arg_type="SingleOption", order=4)
        IgnoreTime = Enum(False, True, label="忽略时间", arg_type="Bool", order=5)
        EndDateASC = Enum(False, True, label="截止日期递增", arg_type="Bool", order=6)
        OrderFields = List(arg_type="List", label="排序字段", order=7)# [("字段名", "ASC" 或者 "DESC")]
        MultiMapping = Enum(False, True, label="多重映射", arg_type="Bool", order=8)
        Operator = Either(Callable(), None, arg_type="Function", label="算子", order=9)
        OperatorDataType = Enum("object", "double", "string", arg_type="SingleOption", label="算子数据类型", order=10, option_range=["object", "double", "string"])
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            # 解析公告时点字段
            Fields = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"].str.lower().str.contains("date")].index.tolist()# 所有的时点字段列表
            Fields += [None]
            self.add_trait("PublDTField", Enum(*Fields, arg_type="SingleOption", label="公告时点字段", order=4, option_range=Fields))
            PublDTField = self._Owner._FactorInfo["DBFieldName"][self._Owner._FactorInfo["FieldType"]=="AnnDate"]
            if PublDTField.shape[0]==0: self.PublDTField = None
            else: self.PublDTField = PublDTField.index[0]
            # 解析排序字段
            Fields = self._Owner._FactorInfo[self._Owner._FactorInfo["Supplementary"]=="OrderField"].index.tolist()# 所有的排序字段列表
            self.OrderFields = [(iField, "ASC") for iField in Fields]
    
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        sys_args["ID字段"] = None
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "批量模式", "回溯天数", "只起始日回溯", "只回溯非目标日", "只回溯时点", "算子", "算子数据类型", "多重映射")
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if key=="DataType":
            if not args.get("多重映射", self._QSArgs.MultiMapping): return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
            if factor_names is None: factor_names = self.FactorNames
            if args.get("算子", self._QSArgs.Operator) is None:
                return pd.Series(["object"]*len(factor_names), index=factor_names)
            else:
                return pd.Series([args.get("算子数据类型", self._QSArgs.OperatorDataType)]*len(factor_names), index=factor_names)
        else:
            return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+DTField+" "
        SQLStr += self._genFromSQLStr(use_main_table=False, args=args)+" "
        SQLStr += "WHERE "+DTField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        if start_dt is not None: SQLStr += "AND "+DTField+">="+start_dt.strftime(self._DTFormat)+" "
        if end_dt is not None: SQLStr += "AND "+DTField+"<="+end_dt.strftime(self._DTFormat)+" "
        SQLStr += "ORDER BY "+DTField
        Rslt = pd.DataFrame(self._FactorDB.fetchall(SQLStr), dtype="O")
        if Rslt.empty:
            return []
        else:
            return self.__QS_adjustDT__(Rslt.iloc[:, 0], args=args).tolist()
    def _genNullIDSQLStr_IgnorePublDT(self, factor_names, ids, end_date, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        if args.get("忽略时间", self._QSArgs.IgnoreTime):
            DTFormat = self._DTFormat
            AdjDTField = self.__QS_toDate__(DTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjDTField = DTField
        SubSQLStr = "SELECT MAX("+AdjDTField+") "
        SubSQLStr += self._genFromSQLStr(args=args)+" "
        SubSQLStr += "WHERE "+AdjDTField+"<"+end_date.strftime(DTFormat)+" "
        ConditionSQLStr = self._genConditionSQLStr(use_main_table=True, args=args)
        SubSQLStr += ConditionSQLStr+" "
        SQLStr = "SELECT "+AdjDTField+", "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += "WHERE "+AdjDTField+" = ("+SubSQLStr+") "
        SQLStr += ConditionSQLStr
        return SQLStr
    def _prepareRawData_IgnorePublDT(self, factor_names, ids, dts, args={}):
        if dts==[]: return pd.DataFrame(columns=["QS_DT"]+factor_names)
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        if args.get("忽略时间", self._QSArgs.IgnoreTime):
            DTFormat = self._DTFormat
            AdjDTField = self.__QS_toDate__(DTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjDTField = DTField
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        LookBack = args.get("回溯天数", self._QSArgs.LookBack)
        if (StartDT is not None) and (not np.isinf(LookBack)): StartDT -= dt.timedelta(LookBack)      
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+AdjDTField+", "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += f"WHERE {'TRUE' if self.FactorDB._QSArgs.DBType!='Oracle' else '(1=1)'} "
        if StartDT is not None:
            SQLStr += "AND "+AdjDTField+">="+StartDT.strftime(DTFormat)+" "
        if EndDT is not None:
            SQLStr += "AND "+AdjDTField+"<="+EndDT.strftime(DTFormat)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY "+DTField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["QS_DT"]+factor_names)
        RawData["QS_DT"] = self.__QS_adjustDT__(RawData["QS_DT"], args=args)
        if (StartDT is not None) and np.isinf(LookBack):
            NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr_IgnorePublDT(factor_names, ids, StartDT, args=args))
            if NullRawData:
                NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT"]+factor_names)
                NullRawData["QS_DT"] = self.__QS_adjustDT__(NullRawData["QS_DT"], args=args)
                RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                RawData.sort_values(by=["QS_DT"])
        if RawData.shape[0]==0: return RawData
        return self._adjustRawDataByRelatedField(RawData, factor_names)
    def _genNullIDSQLStr_WithPublDT(self, factor_names, ids, end_date, args={}):
        EndDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        AnnDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("公告时点字段", self._QSArgs.PublDTField), "DBFieldName"]
        IgnoreTime = args.get("忽略时间", self._QSArgs.IgnoreTime)
        if IgnoreTime:
            DTFormat = self._DTFormat
            AdjAnnDTField = self.__QS_toDate__(AnnDTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjAnnDTField = AnnDTField
        SubSQLStr = "SELECT MAX("+EndDTField+") AS MaxEndDate "
        SubSQLStr += self._genFromSQLStr(use_main_table=False, args=args)+" "
        SubSQLStr += "WHERE ("+AdjAnnDTField+"<"+end_date.strftime(DTFormat)+" "
        SubSQLStr += "AND "+EndDTField+"<"+end_date.strftime(DTFormat)+") "
        SubSQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        if IgnoreTime:
            SQLStr = "SELECT "+self.__QS_toDate__("CASE WHEN "+AnnDTField+">=t.MaxEndDate THEN "+AnnDTField+" ELSE t.MaxEndDate END")+" AS DT, "
        else:
            SQLStr = "SELECT CASE WHEN "+AnnDTField+">=t.MaxEndDate THEN "+AnnDTField+" ELSE t.MaxEndDate END AS DT, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON "+EndDTField+"=t.MaxEndDate "
        SQLStr += self._genConditionSQLStr(use_main_table=True, init_keyword="WHERE", args=args)
        return SQLStr
    def _prepareRawData_WithPublDT(self, factor_names, ids, dts, args={}):
        if dts==[]: return pd.DataFrame(columns=["QS_DT"]+factor_names)
        EndDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        AnnDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("公告时点字段", self._QSArgs.PublDTField), "DBFieldName"]
        IgnoreTime = args.get("忽略时间", self._QSArgs.IgnoreTime)
        if IgnoreTime:
            DTFormat = self._DTFormat
            AdjAnnDTField = self.__QS_toDate__(AnnDTField)
        else:
            DTFormat = self._DTFormat_WithTime
            AdjAnnDTField = AnnDTField
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        LookBack = args.get("回溯天数", self._QSArgs.LookBack)
        if (StartDT is not None) and (not np.isinf(LookBack)): StartDT -= dt.timedelta(LookBack)
        SubSQLStr = "SELECT "
        GroupAnnDTField = "CASE WHEN "+AnnDTField+">="+EndDTField+" THEN "+AnnDTField+" ELSE "+EndDTField+" END"
        if IgnoreTime: GroupAnnDTField = self.__QS_toDate__(GroupAnnDTField)
        SubSQLStr += GroupAnnDTField+" AS AnnDate, "
        SubSQLStr += "MAX("+EndDTField+") AS MaxEndDate "
        SubSQLStr += self._genFromSQLStr(use_main_table=False, args=args)+" "
        SubSQLStr += f"WHERE {'TRUE' if self.FactorDB._QSArgs.DBType!='Oracle' else '(1=1)'} "
        if StartDT is not None:
            SubSQLStr += "AND ("+AdjAnnDTField+">="+StartDT.strftime(DTFormat)+" "
            SubSQLStr += "OR "+EndDTField+">="+StartDT.strftime(DTFormat)+") "
        if EndDT is not None:
            SubSQLStr += "AND ("+AdjAnnDTField+"<="+EndDT.strftime(DTFormat)+" "
            SubSQLStr += "AND "+EndDTField+"<="+EndDT.strftime(DTFormat)+") "
        SubSQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        SubSQLStr += f"GROUP BY {GroupAnnDTField if self.FactorDB._QSArgs.DBType=='Oracle' else 'AnnDate'}"
        SQLStr = "SELECT t.AnnDate AS DT, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t.MaxEndDate="+EndDTField+") "
        SQLStr += self._genIDSQLStr(ids, init_keyword="WHERE", args=args)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY DT"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "MaxEndDate"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["QS_DT", "MaxEndDate"]+factor_names)
        RawData["QS_DT"] = self.__QS_adjustDT__(RawData["QS_DT"], args=args)
        RawData["MaxEndDate"] = self.__QS_adjustDT__(RawData["MaxEndDate"], args=args)
        if (StartDT is not None) and np.isinf(LookBack):
            NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr_WithPublDT(factor_names, [], StartDT, args=args))
            NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "MaxEndDate"]+factor_names)
            NullRawData["QS_DT"] = self.__QS_adjustDT__(NullRawData["QS_DT"], args=args)
            NullRawData["MaxEndDate"] = self.__QS_adjustDT__(NullRawData["MaxEndDate"], args=args)
            RawData = pd.concat([NullRawData, RawData], ignore_index=True)
            RawData.sort_values(by=["QS_DT"])
        if RawData.shape[0]==0: return RawData.loc[:, ["QS_DT"]+factor_names]
        if args.get("截止日期递增", self._QSArgs.EndDateASC):# 删除截止日期非递增的记录
            #DTRank = RawData.loc[:, ["QS_DT", "MaxEndDate"]].astype(np.datetime64).rank(method="min")
            #RawData = RawData[(DTRank["QS_DT"]<=DTRank["MaxEndDate"]).values]
            DTRank = RawData.loc[:, "MaxEndDate"].astype(np.datetime64).rank(method="min")
            RawData = RawData[DTRank.values>=DTRank.cummax().values]
        return self._adjustRawDataByRelatedField(RawData, factor_names)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if args.get("多重映射", self._QSArgs.MultiMapping):
            OrderFields = args.get("排序字段", self._QSArgs.OrderFields)
            if OrderFields:
                OrderFields, Orders = np.array(OrderFields).T.tolist()
            else:
                OrderFields, Orders = [], []
        else:
            OrderFields, Orders = [], []
        FactorNames = list(set(factor_names).union(OrderFields))
        if args.get("公告时点字段", self._QSArgs.PublDTField) is None:
            RawData = self._prepareRawData_IgnorePublDT(factor_names=FactorNames, ids=ids, dts=dts, args=args)
        else:
            RawData = self._prepareRawData_WithPublDT(factor_names=FactorNames, ids=ids, dts=dts, args=args)
        RawData = RawData.sort_values(by=["QS_DT"]+OrderFields, ascending=[True]+[(iOrder.lower()=="asc") for iOrder in Orders])
        return RawData.loc[:, ["QS_DT"]+factor_names]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["QS_DT"])
        if args.get("多重映射", self._QSArgs.MultiMapping):
            Operator = args.get("算子", self._QSArgs.Operator)
            if Operator is None: Operator = (lambda x: x.tolist())
            Data = {}
            for iFactorName in factor_names:
                Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=0).apply(Operator)
            Data = pd.DataFrame(Data).loc[:, factor_names]
        else:
            Data = raw_data.loc[:, factor_names]
            DupMask = Data.index.duplicated()
            if np.any(DupMask):
                self._QS_Logger.warning("%s 的表 %s 提取的数据中包含重复时点: %s" % (self._FactorDB.Name, self.Name, str(Data.index[DupMask])))
                Data = Data[~DupMask]
        Data = Panel(Data.values.T.reshape((Data.shape[1], Data.shape[0], 1)).repeat(len(ids), axis=2), items=Data.columns, major_axis=Data.index, minor_axis=ids)
        return adjustDataDTID(Data, args.get("回溯天数", self._QSArgs.LookBack), factor_names, ids, dts, 
                              args.get("只起始日回溯", self._QSArgs.OnlyStartLookBack), 
                              args.get("只回溯非目标日", self._QSArgs.OnlyLookBackNontarget), 
                              args.get("只回溯时点", self._QSArgs.OnlyLookBackDT), logger=self._QS_Logger)

# 基于 SQL 数据库表的映射因子表
# 一个字段标识 ID, 一个字段标识起始时点, 来自参数时点字段, 一个字段标识截止时点, 其余字段为因子
class SQL_MappingTable(SQL_Table):
    """SQL 映射因子表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        OnlyStartFilled = Enum(False, True, label="只填起始日", arg_type="Bool", order=0)
        MultiMapping = Enum(False, True, label="多重映射", arg_type="Bool", order=1)
        #EndDTField = Enum(None, arg_type="SingleOption", label="结束时点字段", order=2)
        EndDTIncluded = Enum(True, False, label="包含结束时点", arg_type="Bool", order=3)
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            # 解析结束时点字段
            Fields = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"].str.lower().str.contains("date")].index.tolist()# 所有的时点字段列表
            self.add_trait("EndDTField", Enum(*Fields, arg_type="SingleOption", label="结束时点字段", order=2, option_range=Fields))
            EndDTField = self._Owner._FactorInfo["DBFieldName"][self._Owner._FactorInfo["FieldType"]=="EndDate"]
            if EndDTField.shape[0]==0: self.EndDTField = Fields[0]
            else: self.EndDTField = EndDTField.index[0]
            EndDTIncluded = self._Owner._FactorInfo.loc[self.EndDTField, "Supplementary"]
            self.EndDTIncluded = (pd.isnull(EndDTIncluded) or (EndDTIncluded=="包含"))
    
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "批量模式", "只填起始日", "多重映射")
    # 返回给定时点 idt 有数据的所有 ID
    # 如果 idt 为 None, 将返回所有有记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        EndDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("结束时点字段", self._QSArgs.EndDTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr(args=args)+" "
        if idt is not None:
            SQLStr += "WHERE "+DTField+"<="+idt.strftime(self._DTFormat)+" "
            SQLStr += " AND (("+EndDTField+" IS NULL) "
            if args.get("包含结束时点", self._QSArgs.EndDTIncluded):
                SQLStr += "OR ("+EndDTField+">="+idt.strftime(self._DTFormat)+")) "
            else:
                SQLStr += "OR ("+EndDTField+">"+idt.strftime(self._DTFormat)+")) "
        else: SQLStr += "WHERE "+DTField+" IS NOT NULL "
        SQLStr += "AND "+IDField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID"
        return self.__QS_restoreID__([iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)])
    # 返回给定 ID iid 的起始日期距今的时点序列
    # 如果 idt 为 None, 将以表中最小的起始日期作为起点
    # 忽略 ifactor_name    
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        SQLStr = "SELECT MIN("+DTField+") "# 起始日期
        if iid is not None:
            SQLStr += self._genFromSQLStr(args=args)+" "
            SQLStr += self._genIDSQLStr([iid], init_keyword="WHERE", args=args)+" "
            SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        else:
            IDField = args.get("ID字段", self._QSArgs.IDField)
            IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
            SQLStr += self._genFromSQLStr(use_main_table=False, args=args)+" "
            SQLStr += "WHERE "+IDField+" IS NOT NULL "
            SQLStr += self._genConditionSQLStr(use_main_table=False, args=args)+" "
        StartDT = pd.DataFrame(self._FactorDB.fetchall(SQLStr))
        if StartDT.emtpy: return []
        StartDT = self.__QS_adjustDT__(StartDT.iloc[:, 0], args=args).iloc[0]
        if start_dt is not None: StartDT = max((StartDT, start_dt))
        if end_dt is None: end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=StartDT, end_dt=end_dt, timedelta=dt.timedelta(1))
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if key=="DataType":
            if factor_names is None: factor_names = self.FactorNames
            if args.get("多重映射", self._QSArgs.MultiMapping):
                return pd.Series(["object"]*len(factor_names), index=factor_names)
            else:
                return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        else:
            return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        DTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        EndDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("结束时点字段", self._QSArgs.EndDTField), "DBFieldName"]
        # 形成SQL语句, ID, 开始日期, 结束日期, 因子数据
        SQLStr = "SELECT "+self._getIDField(args=args)+" AS ID, "
        SQLStr += DTField+", "
        SQLStr += EndDTField+", "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += self._genIDSQLStr(ids, init_keyword="WHERE", args=args)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        if StartDT is not None:
            SQLStr += "AND (("+EndDTField+">="+StartDT.strftime(self._DTFormat)+") "
            SQLStr += "OR ("+EndDTField+" IS NULL) "
            SQLStr += "OR ("+EndDTField+"<"+DTField+")) "
        if EndDT is not None:
            SQLStr += "AND "+DTField+"<="+EndDT.strftime(self._DTFormat)+" "
        SQLStr += "ORDER BY ID, "+DTField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "QS_起始日", "QS_结束日"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "QS_起始日", "QS_结束日"]+factor_names)
        RawData["QS_起始日"] = self.__QS_adjustDT__(RawData["QS_起始日"], args=args)
        RawData["QS_结束日"] = self.__QS_adjustDT__(RawData["QS_结束日"], args=args)
        RawData["ID"] = self.__QS_restoreID__(RawData["ID"])
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData
    def _calcMultiMappingData(self, raw_data, factor_names, ids, dts, args={}):
        Data, nDT, nFactor = {}, len(dts), len(factor_names)
        raw_data.set_index(["ID"], inplace=True)
        raw_data["QS_结束日"] = raw_data["QS_结束日"].astype("O").where(pd.notnull(raw_data["QS_结束日"]), dts[-1]+dt.timedelta(1))
        if args.get("只填起始日", self._QSArgs.OnlyStartFilled):
            if args.get("包含结束时点", self._QSArgs.EndDTIncluded):
                raw_data["QS_结束日"] = (raw_data["QS_结束日"] + dt.timedelta(1)).astype("O")
            raw_data["QS_起始日"] = raw_data["QS_起始日"].astype("O").where(raw_data["QS_起始日"]>=dts[0], dts[0])
            for iID in raw_data.index.unique():
                #iRawData = raw_data.loc[[iID]].set_index(["QS_起始日"])
                #iData = pd.DataFrame([([],)*nFactor]*nDT, index=dts, columns=factor_names, dtype="O")
                #for jStartDate in iRawData.index.drop_duplicates():
                    #iData.iloc[iData.index.searchsorted(jStartDate)] += pd.Series(iRawData.loc[[jStartDate], factor_names].values.T.tolist(), index=factor_names)
                #Data[iID] = iData
                iRawData = raw_data.loc[[iID]]
                iStartEndDates = sorted(pd.unique(np.r_[iRawData["QS_起始日"].values, iRawData["QS_结束日"].values]))
                iTempData = pd.DataFrame([([],)*nFactor]*len(iStartEndDates), index=iStartEndDates, columns=factor_names, dtype="O")
                iRawData = iRawData.set_index(["QS_起始日", "QS_结束日"])
                for jStartDate, jEndDate in iRawData.index.drop_duplicates():
                    ijRawData = iRawData.loc[jStartDate]
                    if pd.notnull(jEndDate):
                        ijRawData = ijRawData.loc[pd.notnull(ijRawData.index), factor_names]
                        ijRawData = ijRawData.loc[[jEndDate]].values.T.tolist()
                    else:
                        ijRawData = ijRawData.loc[pd.isnull(ijRawData.index), factor_names].values.T.tolist()
                    if jEndDate<jStartDate:
                        ijOldData = iTempData.loc[jStartDate:]
                        iTempData.loc[jStartDate:] += pd.DataFrame([ijRawData] * ijOldData.shape[0], index=ijOldData.index, columns=ijOldData.columns, dtype="O")
                    else:
                        jEndDate -= dt.timedelta(1)
                        ijOldData = iTempData.loc[jStartDate:jEndDate]
                        iTempData.loc[jStartDate:jEndDate] += pd.DataFrame([ijRawData] * ijOldData.shape[0], index=ijOldData.index, columns=ijOldData.columns, dtype="O")
                iData = pd.DataFrame([(None,)*nFactor]*nDT, index=dts, columns=factor_names, dtype="O")
                for j, jDate in enumerate(iStartEndDates):
                    jIdx = iData.index.searchsorted(jDate)
                    if jIdx<iData.shape[0]:
                        iData.iloc[jIdx] = iTempData.iloc[j]
                Data[iID] = iData
            return Panel(Data, major_axis=dts, minor_axis=factor_names).swapaxes(0, 2).loc[:, :, ids]
        else:
            DeltaDT = dt.timedelta(int(not args.get("包含结束时点", self._QSArgs.EndDTIncluded)))
            for iID in raw_data.index.unique():
                iRawData = raw_data.loc[[iID]].set_index(["QS_起始日", "QS_结束日"])
                iData = pd.DataFrame([([],)*nFactor]*nDT, index=dts, columns=factor_names, dtype="O")
                for jStartDate, jEndDate in iRawData.index.drop_duplicates():
                    ijRawData = iRawData.loc[jStartDate]
                    if pd.notnull(jEndDate):
                        ijRawData = ijRawData.loc[pd.notnull(ijRawData.index), factor_names]
                        ijRawData = ijRawData.loc[[jEndDate]].values.T.tolist()
                    else:
                        ijRawData = ijRawData.loc[pd.isnull(ijRawData.index), factor_names].values.T.tolist()
                    if jEndDate<jStartDate:
                        ijOldData = iData.loc[jStartDate:]
                        iData.loc[jStartDate:] += pd.DataFrame([ijRawData] * ijOldData.shape[0], index=ijOldData.index, columns=ijOldData.columns, dtype="O")
                    else:
                        jEndDate -= DeltaDT
                        ijOldData = iData.loc[jStartDate:jEndDate]
                        iData.loc[jStartDate:jEndDate] += pd.DataFrame([ijRawData] * ijOldData.shape[0], index=ijOldData.index, columns=ijOldData.columns, dtype="O")
                Data[iID] = iData
            return Panel(Data, major_axis=dts, minor_axis=factor_names).swapaxes(0, 2).loc[:, :, ids]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        if args.get("多重映射", self._QSArgs.MultiMapping): return self._calcMultiMappingData(raw_data, factor_names, ids, dts, args=args)
        raw_data.set_index(["ID"], inplace=True)
        Data, nFactor = {}, len(factor_names)
        raw_data["QS_结束日"] = raw_data["QS_结束日"].astype("O").where(pd.notnull(raw_data["QS_结束日"]), dts[-1]+dt.timedelta(1))
        if args.get("只填起始日", self._QSArgs.OnlyStartFilled):
            if args.get("包含结束时点", self._QSArgs.EndDTIncluded):
                raw_data["QS_结束日"] = (raw_data["QS_结束日"] + dt.timedelta(1)).astype("O")
            raw_data["QS_起始日"] = raw_data["QS_起始日"].astype("O").where(raw_data["QS_起始日"]>=dts[0], dts[0])
            for iID in raw_data.index.unique():
                #iRawData = raw_data.loc[[iID]].set_index(["QS_起始日"])
                #iData = pd.DataFrame(index=dts, columns=factor_names)
                #for jStartDate in iRawData.index:
                    #iData.iloc[iData.index.searchsorted(jStartDate)] = iRawData.loc[jStartDate, factor_names]
                #Data[iID] = iData
                iRawData = raw_data.loc[[iID]]
                iStartEndDates = np.r_[iRawData["QS_起始日"].values, iRawData["QS_结束日"].values]
                iStartEndDates = sorted(pd.unique(iStartEndDates[pd.notnull(iStartEndDates)]))
                iTempData = pd.DataFrame(index=iStartEndDates, columns=factor_names)
                #iRawData = iRawData.set_index(["QS_起始日", "QS_结束日"])
                for j in range(iRawData.shape[0]):
                    ijRawData = iRawData.iloc[j]
                    jStartDate, jEndDate = ijRawData["QS_起始日"], ijRawData["QS_结束日"]
                    if jEndDate<jStartDate:
                        iTempData.loc[jStartDate:] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iTempData.loc[jStartDate:].shape[0], axis=0)
                    else:
                        jEndDate -= dt.timedelta(1)
                        iTempData.loc[jStartDate:jEndDate] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iTempData.loc[jStartDate:jEndDate].shape[0], axis=0)
                iData = pd.DataFrame(index=dts, columns=factor_names)
                for j, jDate in enumerate(iStartEndDates):
                    jIdx = iData.index.searchsorted(jDate)
                    if jIdx<iData.shape[0]:
                        iData.iloc[jIdx] = iTempData.iloc[j]
                Data[iID] = iData
            return Panel(Data, major_axis=dts, minor_axis=factor_names).swapaxes(0, 2).loc[:, :, ids]
        else:
            DeltaDT = dt.timedelta(int(not args.get("包含结束时点", self._QSArgs.EndDTIncluded)))
            for iID in raw_data.index.unique():
                iRawData = raw_data.loc[[iID]]
                iData = pd.DataFrame(index=dts, columns=factor_names)
                for j in range(iRawData.shape[0]):
                    ijRawData = iRawData.iloc[j]
                    jStartDate, jEndDate = ijRawData["QS_起始日"], ijRawData["QS_结束日"]
                    if jEndDate<jStartDate:
                        iData.loc[jStartDate:] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iData.loc[jStartDate:].shape[0], axis=0)
                    else:
                        jEndDate -= DeltaDT
                        iData.loc[jStartDate:jEndDate] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iData.loc[jStartDate:jEndDate].shape[0], axis=0)
                Data[iID] = iData
            return Panel(Data, major_axis=dts, minor_axis=factor_names).swapaxes(0, 2).loc[:, :, ids]

# 基于 SQL 数据库表的成份因子表
# 一个字段标识 ID, 一个字段标识起始时点, 一个字段标识截止时点, 其余字段为因子
class SQL_ConstituentTable(SQL_Table):
    """SQL 成份因子表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        #GroupField = Enum(None, arg_type="SingleOption", label="类别字段", order=0)
        #EndDTField = Enum(None, arg_type="SingleOption", label="结束时点字段", order=1)
        #CurSignField = Enum(None, arg_type="SingleOption", label="当前状态字段", order=2)
        EndDTIncluded = Enum(False, True, label="包含结束时点", arg_type="Bool", order=3)
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            FactorInfo = self._Owner._FactorInfo
            # 解析类别字段
            Fields = FactorInfo[pd.notnull(FactorInfo["FieldType"])].index.tolist()# 所有字段列表
            self.add_trait("GroupField", Enum(*Fields, arg_type="SingleOption", label="类别字段", order=0, option_range=Fields))
            GroupField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="Group"]
            if GroupField.shape[0]==0: self.GroupField = Fields[0]
            else: self.GroupField = GroupField.index[0]
            # 解析当前状态字段
            self.add_trait("CurSignField", Enum(*(Fields+[None]), arg_type="SingleOption", label="当前状态字段", order=2, option_range=Fields+[None]))
            CurSignField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="CurSign"]
            if CurSignField.shape[0]==0: self.CurSignField = None
            else: self.CurSignField = CurSignField.index[0]
            # 解析结束时点字段
            Fields = FactorInfo[FactorInfo["FieldType"].str.lower().str.contains("date")].index.tolist()# 所有的时点字段列表
            self.add_trait("EndDTField", Enum(*Fields, arg_type="SingleOption", label="结束时点字段", order=1, option_range=Fields))
            EndDTField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="EndDate"]
            if EndDTField.shape[0]==0: self.EndDTField = Fields[0]
            else: self.EndDTField = EndDTField.index[0]
            EndDTIncluded = FactorInfo.loc[self.EndDTField, "Supplementary"]
            self.EndDTIncluded = (pd.isnull(EndDTIncluded) or (EndDTIncluded=="包含"))
    
    def __init__(self, name, fdb, sys_args={},  table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._AllGroups = None
    @property
    def FactorNames(self):
        if self._AllGroups is None:
            GroupField = self._DBTableName+"."+self._FactorInfo.loc[self._QSArgs.GroupField, "DBFieldName"]
            SQLStr = f"SELECT DISTINCT {GroupField} {self._genFromSQLStr(use_main_table=False)} ORDER BY {GroupField}"
            self._AllGroups = [str(iRslt[0]) for iRslt in self._FactorDB.fetchall(SQLStr)]
        return self._AllGroups
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType":
            return pd.Series("double", index=factor_names)
        elif key=="Description": return pd.Series(["0 or nan: 非成分; 1: 是成分"]*len(factor_names), index=factor_names)
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args),
                                 "Description":self.getFactorMetaData(factor_names, key="Description", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 返回指数 ID 为 ifactor_name 在给定时点 idt 的所有成份股
    # 如果 idt 为 None, 将返回指数 ifactor_name 的所有历史成份股
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有 ID
    def getID(self, ifactor_name=None, idt=None, args={}, **kwargs):
        GroupField = self._DBTableName+"."+self._FactorInfo.loc[args.get("类别字段", self._QSArgs.GroupField), "DBFieldName"]
        InDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        OutDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("结束时点字段", self._QSArgs.EndDTField), "DBFieldName"]
        CurSignField = args.get("当前状态字段", self._QSArgs.CurSignField)
        EndDTIncluded = args.get("包含结束时点", self._QSArgs.EndDTIncluded)
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr(args=args)+" "
        if ifactor_name is not None: SQLStr += "WHERE "+GroupField+"='"+ifactor_name+"' "
        else: SQLStr += "WHERE "+GroupField+" IS NOT NULL "
        if idt is not None:
            SQLStr += "AND "+InDTField+"<="+idt.strftime(self._DTFormat)+" "
            if kwargs.get("is_current", True):
                if EndDTIncluded:
                    SQLStr += "AND (("+OutDTField+">='"+idt.strftime(self._DTFormat)+"') "
                else:
                    SQLStr += "AND (("+OutDTField+">'"+idt.strftime(self._DTFormat)+"') "
                if CurSignField is not None:
                    SQLStr += "OR ("+self._DBTableName+"."+self._FactorInfo.loc[CurSignField, "DBFieldName"]+"=1)) "
                else:
                    SQLStr += "OR ("+OutDTField+" IS NULL)) "
        SQLStr += "AND "+IDField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID"
        return self.__QS_restoreID__([iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)])
    # 返回指数 ID 为 ifactor_name 包含成份股 iid 的时间点序列
    # 如果 iid 为 None, 将返回指数 ifactor_name 的有记录数据的时间点序列
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有时间点
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        GroupField = self._DBTableName+"."+self._FactorInfo.loc[args.get("类别字段", self._QSArgs.GroupField), "DBFieldName"]
        InDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        OutDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("结束时点字段", self._QSArgs.EndDTField), "DBFieldName"]
        EndDTIncluded = args.get("包含结束时点", self._QSArgs.EndDTIncluded)
        if iid is not None:
            SQLStr = "SELECT "+InDTField+" AS InDT, "# 纳入日期
            SQLStr += OutDTField+" AS OutDT "# 剔除日期
            SQLStr += self._genFromSQLStr(args=args)+" "
            SQLStr += "WHERE "+InDTField+" IS NOT NULL "
            if ifactor_name is not None: SQLStr += "AND "+GroupField+"='"+ifactor_name+"' "
            IDField = args.get("ID字段", self._QSArgs.IDField)
            if IDField is None:
                SQLStr += "AND "+self._MainTableName+"."+self._MainTableID+"='"+self.__QS_adjustID__([iid])[0]+"' "
            else:
                SQLStr += "AND "+self._DBTableName+"."+self._FactorInfo.loc[IDField, "DBFieldName"]+"='"+self.__QS_adjustID__([iid])[0]+"' "
            if start_dt is not None:
                if EndDTIncluded:
                    SQLStr += "AND (("+OutDTField+">="+start_dt.strftime(self._DTFormat)+") "
                else:
                    SQLStr += "AND (("+OutDTField+">"+start_dt.strftime(self._DTFormat)+") "
                SQLStr += "OR ("+OutDTField+" IS NULL))"
            if end_dt is not None:
                SQLStr += "AND "+InDTField+"<="+end_dt.strftime(self._DTFormat)+" "
            SQLStr += self._genConditionSQLStr(args=args)+" "
            SQLStr += "ORDER BY "+InDTField
            Data = pd.DataFrame(self._FactorDB.fetchall(SQLStr), dtype="O")
            if Data.empty: return []
            Data["InDT"] = self.__QS_adjustDT__(Data["InDT"], args=args)
            Data["OutDT"] = self.__QS_adjustDT__(Data["OutDT"], args=args)
            Data["OutDT"] = Data["OutDT"].where(pd.notnull(Data["OutDT"]), dt.datetime.now())
            DateTimes = set()
            for iStartDT, iEndDT in Data.to_records(index=False).tolist():
                DateTimes = DateTimes.union(getDateTimeSeries(start_dt=iStartDT, end_dt=iEndDT, timedelta=dt.timedelta(1)))
            return sorted(DateTimes)
        SQLStr = "SELECT MIN("+InDTField+") "# 纳入日期
        SQLStr += self._genFromSQLStr(use_main_table=False, args=args)+" "
        if ifactor_name is not None: SQLStr += "WHERE "+GroupField+"='"+ifactor_name+"'"
        else: SQLStr += "WHERE "+GroupField+" IS NOT NULL"
        SQLStr += self._genConditionSQLStr(args=args)
        StartDT = pd.DataFrame(self._FactorDB.fetchall(SQLStr), dtype="O")
        if StartDT.empty: return []
        StartDT = self.__QS_adjustDT__(StartDT.iloc[:, 0], args=args).iloc[0]
        if start_dt is not None: StartDT = max((StartDT, start_dt))
        if end_dt is None: end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        GroupField = self._DBTableName+"."+self._FactorInfo.loc[args.get("类别字段", self._QSArgs.GroupField), "DBFieldName"]
        InDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        OutDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("结束时点字段", self._QSArgs.EndDTField), "DBFieldName"]
        CurSignField = args.get("当前状态字段", self._QSArgs.CurSignField)
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        # 指数中成份股 ID, 指数证券 ID, 纳入日期, 剔除日期, 最新标志
        SQLStr = "SELECT "+GroupField+" AS GroupID, "# 指数证券 ID
        SQLStr += self._getIDField(args=args)+" AS SecurityID, "# ID
        SQLStr += InDTField+" AS InDate, "# 纳入日期
        SQLStr += OutDTField+" AS OutDate, "# 剔除日期
        if CurSignField is not None: SQLStr += self._DBTableName+"."+self._FactorInfo.loc[CurSignField, "DBFieldName"]+" AS CurSign "# 最新标志
        else: SQLStr += "NULL AS CurSign "# 最新标志
        SQLStr += self._genFromSQLStr(args=args)+" "
        SQLStr += "WHERE ("+genSQLInCondition(GroupField, factor_names, is_str=(self.__QS_identifyDataType__(self._FactorInfo["DataType"].loc[args.get("类别字段", self._QSArgs.GroupField)])!="double"), max_num=1000)+") "
        SQLStr += self._genIDSQLStr(ids, args=args)+" "
        if StartDT is not None:
            SQLStr += "AND (("+OutDTField+">"+StartDT.strftime(self._DTFormat)+") "
            SQLStr += "OR ("+OutDTField+" IS NULL)) "
        if EndDT is not None:
            SQLStr += "AND "+InDTField+"<="+EndDT.strftime(self._DTFormat)+" "
            SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY GroupID, SecurityID, InDate"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["Group", "SecurityID", "InDate", "OutDate", "CurSign"])
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["Group", "SecurityID", "InDate", "OutDate", "CurSign"])
        RawData["InDate"] = self.__QS_adjustDT__(RawData["InDate"], args=args)
        RawData["OutDate"] = self.__QS_adjustDT__(RawData["OutDate"], args=args)
        RawData["Group"] = RawData["Group"].astype(str)
        RawData["SecurityID"] = self.__QS_restoreID__(RawData["SecurityID"])
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        DeltaDT = dt.timedelta(int(not args.get("包含结束时点", self._QSArgs.EndDTIncluded)))
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DateSeries = getDateSeries(StartDate, EndDate)
        Data = {}
        for iGroup in factor_names:
            iRawData = raw_data[raw_data["Group"]==iGroup].set_index(["SecurityID"])
            iData = pd.DataFrame(0, index=DateSeries, columns=pd.unique(iRawData.index))
            for jID in iData.columns:
                jIDRawData = iRawData.loc[[jID]]
                for k in range(jIDRawData.shape[0]):
                    kStartDate = jIDRawData["InDate"].iloc[k].date()
                    kEndDate = (jIDRawData["OutDate"].iloc[k].date()-DeltaDT if jIDRawData["OutDate"].iloc[k] is not None else dt.date.today())
                    iData[jID].loc[kStartDate:kEndDate] = 1
            Data[iGroup] = iData
        Data = Panel(Data, major_axis=DateSeries)
        if Data.minor_axis.intersection(ids).shape[0]==0: return Panel(0.0, items=factor_names, major_axis=dts, minor_axis=ids)
        Data = Data.loc[factor_names, :, ids]
        Data.major_axis = [dt.datetime.combine(iDate, dt.time(0)) for iDate in Data.major_axis]
        Data.fillna(value=0, inplace=True)
        return adjustDateTime(Data, dts, fillna=True, method="bfill")
    # 返回 DataFrame(columns=["Group", "ID", "InDate", "OutDate", "CurSign"])
    def readSQLData(self, factor_names, ids=None, start_dt=None, end_dt=None, args={}):
        return super().readSQLData(factor_names, ids, start_dt, end_dt, args=args).rename(columns={"SecurityID": "ID"})

def RollBackNPeriod(report_date, n_period):
    nYear, nPeriod = n_period // 4, n_period % 4
    TargetYear = report_date.year - nYear
    TargetMonth = report_date.month - nPeriod * 3
    TargetYear -= (TargetMonth<=0)
    TargetMonth += (TargetMonth<=0) * 12
    TargetDay = (31 if (TargetMonth==12) or (TargetMonth==3) else 30)
    return dt.datetime(TargetYear, TargetMonth, TargetDay)

# 基于 SQL 数据库表的财务因子表
# 一个字段标识 ID, 一个字段标识报告期字段, 表示财报的报告期, 一个字段标识公告日期字段, 表示财报公布的日期, 其余字段为因子
class SQL_FinancialTable(SQL_Table):
    """财务因子表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        ReportDate = Enum("所有", "定期报告", "年报", "中报", "一季报", "三季报", label="报告期", arg_type="SingleOption", order=0, option_range=["所有", "定期报告", "年报", "中报", "一季报", "三季报"])
        CalcType = Enum("最新", "单季度", "TTM", label="计算方法", arg_type="SingleOption", order=1, option_range=["最新", "单季度", "TTM"])
        YearLookBack = Int(0, label="回溯年数", arg_type="Integer", order=2)
        PeriodLookBack = Int(0, label="回溯期数", arg_type="Integer", order=3)
        IgnoreMissing = Enum(True, False, label="忽略缺失", arg_type="Bool", order=4)
        IgnoreNonQuarter = Enum(False, True, label="忽略非季末报告", arg_type="Bool", order=5)
        #AdjustTypeField = Enum(None, label="调整类型字段", arg_type="SingleOption", order=6)
        #AdjustType = Str("2,1", label="调整类型", arg_type="String", order=7)
        #PublDTField = Enum(None, label="公告时点字段", arg_type="SingleOption", order=8)
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            FactorInfo = self._Owner._FactorInfo
            # 解析公告时点字段
            Fields = FactorInfo[FactorInfo["FieldType"].str.lower().str.contains("date")].index.tolist()# 所有的时点字段列表
            Fields += [None]
            self.add_trait("PublDTField", Enum(*Fields, arg_type="SingleOption", label="公告时点字段", order=8, option_range=Fields))
            PublDTField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="AnnDate"]
            if PublDTField.shape[0]==0: self.PublDTField = None
            else: self.PublDTField = PublDTField.index[0]
            # 调整类型字段
            Fields = [None]+FactorInfo.index.tolist()# 所有的字段列表
            self.add_trait("AdjustTypeField", Enum(*Fields, arg_type="SingleOption", label="调整类型字段", order=6, option_range=Fields))
            AdjustTypeField = FactorInfo[FactorInfo["FieldType"]=="AdjustType"].index
            if AdjustTypeField.shape[0]==0: self.AdjustTypeField = None
            else: self.AdjustTypeField = AdjustTypeField[0]
            self.add_trait("AdjustType", Str("", label="调整类型", arg_type="String", order=7))
            if self.AdjustTypeField is not None:
                iConditionVal = FactorInfo.loc[self.AdjustTypeField, "Supplementary"]
                if pd.isnull(iConditionVal) or (isinstance(iConditionVal, str) and (iConditionVal.lower() in ("", "nan"))):
                    self.AdjustType = ""
                else:
                    self.AdjustType = str(iConditionVal).strip()
    
    def __init__(self, name, fdb, sys_args={},  table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._TempData = {}
    def _genConditionSQLStr(self, use_main_table=True, init_keyword="AND", args={}):
        SQLStr = super()._genConditionSQLStr(use_main_table=use_main_table, init_keyword=init_keyword, args=args)
        if SQLStr: init_keyword = "AND"
        ReportDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        if args.get("忽略非季末报告", self._QSArgs.IgnoreNonQuarter) or (not ((args.get("报告期", self._QSArgs.ReportDate)=="所有") and (args.get("计算方法", self._QSArgs.CalcType)=="最新") and (args.get("回溯年数", self._QSArgs.YearLookBack)==0) and (args.get("回溯期数", self._QSArgs.PeriodLookBack)==0))):
            if self._FactorDB._QSArgs.DBType=="SQL Server":
                SQLStr += " "+init_keyword+" TO_CHAR("+ReportDTField+",'MMDD') IN ('0331','0630','0930','1231')"
                init_keyword = "AND"
            elif self._FactorDB._QSArgs.DBType=="MySQL":
                SQLStr += " "+init_keyword+" DATE_FORMAT("+ReportDTField+",'%m%d') IN ('0331','0630','0930','1231')"
                init_keyword = "AND"
            elif self._FactorDB._QSArgs.DBType=="Oracle":
                SQLStr += " "+init_keyword+" TO_CHAR("+ReportDTField+",'MMdd') IN ('0331','0630','0930','1231')"
                init_keyword = "AND"
            elif self._FactorDB._QSArgs.DBType=="sqlite3":
                DTFmt = args.get("时点格式", self._QSArgs.DTFmt)
                YearStartIdx = DTFmt.find("%Y") + 1
                YearEndIdx = YearStartIdx + 4
                DTFmt = DTFmt.replace("%Y", "")
                ReportDays = "','".join((dt.datetime(2000, 3, 31).strftime(DTFmt), dt.datetime(2000, 6, 30).strftime(DTFmt), dt.datetime(2000, 9, 30).strftime(DTFmt), dt.datetime(2000, 12, 31).strftime(DTFmt)))
                SQLStr += " "+init_keyword+f" (SUBSTR({ReportDTField}, {YearStartIdx}, {YearStartIdx-1}) || SUBSTR({ReportDTField}, {YearEndIdx})) IN ('{ReportDays}')"
                init_keyword = "AND"
            #else:
                #raise __QS_Error__("SQL_FinancialTable._genConditionSQLStr 不支持的数据库类型: '%s'" % (self._FactorDB._QSArgs.DBType, ))
        AdjustTypeField = args.get("调整类型字段", self._QSArgs.AdjustTypeField)
        if AdjustTypeField is not None:
            iConditionVal = args.get("调整类型", self._QSArgs.AdjustType)
            if iConditionVal:
                if self.__QS_identifyDataType__(self._FactorInfo.loc[AdjustTypeField, "DataType"])!="double":
                    SQLStr += " "+init_keyword+" "+self._DBTableName+"."+self._FactorInfo.loc[AdjustTypeField, "DBFieldName"]+" IN ('"+"','".join(iConditionVal.split(","))+"') "
                else:
                    SQLStr += " "+init_keyword+" "+self._DBTableName+"."+self._FactorInfo.loc[AdjustTypeField, "DBFieldName"]+" IN ("+iConditionVal+") "
        return SQLStr
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = (iFactor._QSArgs.IgnoreNonQuarter or (not ((iFactor._QSArgs.ReportDate=="所有") and (iFactor._QSArgs.CalcType=="最新") and (iFactor._QSArgs.YearLookBack==0) and (iFactor._QSArgs.PeriodLookBack==0))))
            iConditions = (iConditions, iFactor._QSArgs.AdjustType, iFactor._QSArgs.PreFilterID, ";".join([iCondition+":"+str(iFactor._QSArgs[iCondition]) for i, iCondition in enumerate(self._QSArgs._ConditionFields)]+["筛选条件:"+iFactor._QSArgs["筛选条件"]]))
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {"FactorNames":[iFactor.Name], 
                                                       "RawFactorNames":{iFactor._NameInFT}, 
                                                       "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                       "args":iFactor._QSArgs.to_dict()}
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    # 返回在给定时点 idt 之前有财务报告的 ID
    # 如果 idt 为 None, 将返回所有有财务报告的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        AnnDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("公告时点字段", self._QSArgs.PublDTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr(args=args)+" "
        if idt is not None: SQLStr += "WHERE "+AnnDateField+"<='"+idt.strftime("%Y-%m-%d")+"' "
        else: SQLStr += "WHERE "+AnnDateField+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+IDField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID"
        return self.__QS_restoreID__([iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)])
    # 返回在给定 ID iid 的有财务报告的公告时点
    # 如果 iid 为 None, 将返回所有有财务报告的公告时点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        AnnDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("公告时点字段", self._QSArgs.PublDTField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+AnnDateField+" "
        SQLStr += self._genFromSQLStr(args=args)+" "
        if iid is not None: iid = [iid]
        SQLStr += self._genIDSQLStr([iid], init_keyword="WHERE", args=args)+" "
        if start_dt is not None: SQLStr += "AND "+AnnDateField+">="+start_dt.strftime(self._DTFormat)+"' "
        if end_dt is not None: SQLStr += "AND "+AnnDateField+"<="+end_dt.strftime(self._DTFormat)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY "+AnnDateField
        Rslt = pd.DataFrame(self._FactorDB.fetchall(SQLStr), dtype="O")
        if Rslt.empty:
            return []
        else:
            return self.__QS_adjustDT__(Rslt.iloc[:, 0], args=args).tolist()
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if dts is not None:
            EndDT = dts[-1]
        else:
            EndDT = None
        ReportDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        AnnDTField = self._DBTableName+"."+self._FactorInfo.loc[args.get("公告时点字段", self._QSArgs.PublDTField), "DBFieldName"]
        AdjustTypeField = args.get("调整类型字段", self._QSArgs.AdjustTypeField)
        AdjustTypes = args.get("调整类型", self._QSArgs.AdjustType)
        if pd.notnull(AdjustTypes) and (AdjustTypes!=""):
            AdjustTypes = AdjustTypes.split(",")
        else:
            AdjustTypes = []
        # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
        SQLStr = "SELECT "+self._getIDField(args=args)+" AS ID, "
        SQLStr += AnnDTField+" AS AnnDate, "
        SQLStr += ReportDTField+" AS ReportDate, "
        if (len(AdjustTypes)>0) and (AdjustTypeField is not None):
            SQLStr += "CASE "+self._DBTableName+"."+self._FactorInfo.loc[AdjustTypeField, "DBFieldName"]+" "
            for i in range(len(AdjustTypes)):
                SQLStr += "WHEN "+AdjustTypes[i].strip()+" THEN "+str(i)+" "
            SQLStr += "ELSE 0 END AS AdjustType, "
        else:
            SQLStr += "0 AS AdjustType, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr, args=args)+" "
        SQLStr += "WHERE "+ReportDTField+" IS NOT NULL "
        if EndDT is not None:
            SQLStr += "AND "+AnnDTField+"<="+EndDT.strftime(self._DTFormat)+" "
        SQLStr += self._genIDSQLStr(ids, args=args)+" "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += "ORDER BY ID, "+AnnDTField+", "
        SQLStr += ReportDTField + ", AdjustType"
        #RawData = pd.read_sql_query(SQLStr, self._FactorDB.Connection)
        #RawData.columns = ["ID", "AnnDate", "ReportDate", "AdjustType"]+factor_names
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "AnnDate", "ReportDate", "AdjustType"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "AnnDate", "ReportDate", "AdjustType"]+factor_names)
        RawData["AnnDate"] = self.__QS_adjustDT__(RawData["AnnDate"], args=args)
        RawData["ReportDate"] = self.__QS_adjustDT__(RawData["ReportDate"], args=args)
        RawData["ID"] = self.__QS_restoreID__(RawData["ID"])
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        if (self._FactorDB._QSArgs.DBType not in ("MySQL", "Oracle", "SQL Server")) and (args.get("忽略非季末报告", self._QSArgs.IgnoreNonQuarter) or (not ((args.get("报告期", self._QSArgs.ReportDate)=="所有") and (args.get("计算方法", self._QSArgs.CalcType)=="最新") and (args.get("回溯年数", self._QSArgs.YearLookBack)==0) and (args.get("回溯期数", self._QSArgs.PeriodLookBack)==0)))):
            RawData = RawData[RawData["ReportDate"].dt.strftime("%m%d").str.isin(('0331','0630','0930','1231'))]
        return RawData
    def _calcData(self, raw_data, periods, factor_name, ids, dts, calc_type, report_date, ignore_missing, args={}):
        if ignore_missing: raw_data = raw_data[pd.notnull(raw_data[factor_name])]
        # TargetReportDate: 每个 ID 每个公告日对应的最大报告期
        TargetReportDate = raw_data.loc[:, ["ID", "AnnDate", "ReportDate"]]
        if report_date=="年报": TargetReportDate.loc[raw_data["ReportPeriod"]!="12-31", "ReportDate"] = dt.datetime(1899,12,31)
        elif report_date=="中报": TargetReportDate.loc[raw_data["ReportPeriod"]!="06-30", "ReportDate"] = dt.datetime(1899,12,31)
        elif report_date=="一季报": TargetReportDate.loc[raw_data["ReportPeriod"]!="03-31", "ReportDate"] = dt.datetime(1899,12,31)
        elif report_date=="三季报": TargetReportDate.loc[raw_data["ReportPeriod"]!="09-30", "ReportDate"] = dt.datetime(1899,12,31)
        TargetReportDate = TargetReportDate.set_index(["ID"]).groupby(level=0).cummax().reset_index().groupby(by=["ID", "AnnDate"], as_index=False).max()
        MaxReportDate = TargetReportDate["ReportDate"].copy()
        Data = {}
        for i, iPeriod in enumerate(periods):
            # TargetReportDate: 每个 ID 每个公告日对应的目标报告期
            if iPeriod>0: TargetReportDate["ReportDate"] = MaxReportDate.apply(RollBackNPeriod, args=(iPeriod,))
            # iData: 每个 ID 每个公告日对应的目标报告期及其因子值
            iData = TargetReportDate.merge(raw_data, how="left", on=["ID", "ReportDate"], suffixes=("", "_y"))
            #iData = iData[(iData["AnnDate"]>=iData["AnnDate_y"]) | pd.isnull(iData["AnnDate_y"])].sort_values(by=["ID", "AnnDate", "AnnDate_y", "AdjustType"])
            iData["AnnDate_y"] = iData["AnnDate_y"].where(iData["AnnDate"]>=iData["AnnDate_y"], None)
            iData.loc[pd.isnull(iData["AnnDate_y"]), factor_name] = None
            iData = iData.sort_values(by=["ID", "AnnDate", "AnnDate_y", "AdjustType"], na_position="first")
            if (i==0) and (calc_type!="最新"):
                iData = iData.loc[:, ["ID", "AnnDate", "ReportPeriod", factor_name]].groupby(by=["ID", "AnnDate"], as_index=True).last()
                ReportPeriod = iData.loc[:, "ReportPeriod"].where(pd.notnull(iData.loc[:, "ReportPeriod"]), "None").unstack().T
                iData = iData.loc[:, factor_name].where(pd.notnull(iData.loc[:, factor_name]), np.inf).unstack().T
                iIndex = iData.index.union(dts).sort_values()
                Data[iPeriod] = iData.reindex(index=iIndex).fillna(method="pad").reindex(index=dts, columns=ids)
                ReportPeriod = ReportPeriod.reindex(index=iIndex).fillna(method="pad").reindex(index=dts, columns=ids)
                Data[iPeriod] = Data[iPeriod].where(Data[iPeriod]!=np.inf, np.nan)
                ReportPeriod = ReportPeriod.where(ReportPeriod!="None", None)
            else:
                iData = iData.loc[:, ["ID", "AnnDate", factor_name]].groupby(by=["ID", "AnnDate"], as_index=True).last()
                iData = iData.loc[:, factor_name].where(pd.notnull(iData.loc[:, factor_name]), np.inf).unstack().T
                iIndex = iData.index.union(dts).sort_values()
                Data[iPeriod] = iData.reindex(index=iIndex).fillna(method="pad").reindex(index=dts, columns=ids)
                Data[iPeriod] = Data[iPeriod].where(Data[iPeriod]!=np.inf, np.nan)
            iData = None
        if calc_type=="最新": return Data[periods[0]]
        elif calc_type=="单季度":
            Rslt = Data[periods[0]] - Data[periods[1]]
            Mask = (ReportPeriod=="03-31")
            Rslt[Mask] = Data[periods[0]][Mask]
            Rslt[pd.isnull(ReportPeriod)] = None
        elif calc_type=="TTM":
            Rslt = Data[periods[0]].copy()
            Mask = (ReportPeriod=="03-31")
            Rslt[Mask] = (Data[periods[0]] + Data[periods[1]] - Data[periods[4]])[Mask]
            Mask = (ReportPeriod=="06-30")
            Rslt[Mask] = (Data[periods[0]] + Data[periods[2]] - Data[periods[4]])[Mask]
            Mask = (ReportPeriod=="09-30")
            Rslt[Mask] = (Data[periods[0]] + Data[periods[3]] - Data[periods[4]])[Mask]
            Rslt[pd.isnull(ReportPeriod)] = None
        return Rslt
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        CalcType, YearLookBack, PeriodLookBack, ReportDate, IgnoreMissing = args.get("计算方法", self._QSArgs.CalcType), args.get("回溯年数", self._QSArgs.YearLookBack), args.get("回溯期数", self._QSArgs.PeriodLookBack), args.get("报告期", self._QSArgs.ReportDate), args.get("忽略缺失", self._QSArgs.IgnoreMissing)
        if CalcType=="最新": Periods = np.array([0], dtype=int)
        elif CalcType=="单季度": Periods = np.array([0, 1], dtype=int)
        elif CalcType=="TTM": Periods = np.array([0, 1, 2, 3, 4], dtype=int)
        Periods += YearLookBack * 4 + PeriodLookBack
        raw_data["ReportPeriod"] = raw_data["ReportDate"].astype(str).str.slice(start=5, stop=10)
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = self._calcData(raw_data.loc[:, ["ID", "AnnDate", "ReportDate", "AdjustType", "ReportPeriod", iFactorName]], Periods, iFactorName, ids, dts, CalcType, ReportDate, IgnoreMissing, args=args)
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)