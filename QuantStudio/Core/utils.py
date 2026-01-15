# coding=utf-8
import os

import numpy as np
import pandas as pd

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback


# Quant Studio 系统错误: 重复索引
class __QS_Error_DuplicatedIndex__(__QS_Error__):
    """Quant Studio 重复索引错误"""
    pass

# 将信息源文件中的表和字段信息导入信息文件
def importInfo(info_file, info_resource, out_info=False):
    TableInfo = pd.read_excel(info_resource, "TableInfo", engine="openpyxl").set_index(["TableName"])
    FactorInfo = pd.read_excel(info_resource, 'FactorInfo', engine="openpyxl").set_index(['TableName', 'FieldName'])
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
# raw_data: DataFrame(columns=["QS_DT", "QS_ID"]+factor_names)
def _QS_calcListData_WideTable(raw_data, factor_names, ids, dts, args={}, **kwargs):
    Operator = args.get("算子", lambda x: x.tolist())
    if Operator is None: Operator = lambda x: x.tolist()
    OperatorDataType = args.get("OperatorDataType", "object")
    AdditionalFields = args.get("AdditionalFields", [])
    if args.get("OnlyLookBackDT", False):
        DeduplicatedIndex = raw_data.index(~raw_data.index.duplicated())
        RowIdxMask = pd.Series(False, index=DeduplicatedIndex).unstack(fill_value=True).astype(bool)
        RawIDs = RowIdxMask.columns
        if RawIDs.intersection(ids).shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
        RowIdx[RowIdxMask] = np.nan
        RowIdx = adjustDataDTID(Panel({"RowIdx": RowIdx}), args.get("LookBack", 0), ["RowIdx"], RowIdx.columns.tolist(), dts, args.get("OnlyStartLookBack", False), args.get("OnlyLookBackNontarget", False), logger=kwargs.get("logger", None)).iloc[0].values
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
        return adjustDataDTID(Data, args.get("LookBack", 0), factor_names, ids, dts, args.get("OnlyStartLookBack", False), args.get("OnlyLookBackNontarget", False), logger=kwargs.get("logger", None))

def _QS_calcData_WideTable(raw_data, factor_names, ids, dts, data_type, args={}, **kwargs):
    if raw_data.shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
    if ids is None: ids = sorted(raw_data["QS_ID"].unique())
    raw_data = raw_data.set_index(["QS_DT", "QS_ID"])
    MultiMapping = args.get("MultiMapping", False)
    if MultiMapping:
        return _QS_calcListData_WideTable(raw_data, factor_names, ids, dts, args=args, **kwargs)
    else:
        if not raw_data.index.is_unique:
            Msg = kwargs.get("error_fmt", {}).get("DuplicatedIndex", "{Error}")
            raise __QS_Error_DuplicatedIndex__(Msg.format(Error = ("重复的索引为 : %s" % (str(raw_data.index[raw_data.index.duplicated()].tolist()), ))))
    DataType = data_type[~data_type.index.duplicated()]
    if args.get("OnlyLookBackDT", False):
        RowIdxMask = pd.Series(False, index=raw_data.index).unstack(fill_value=True).astype(bool)
        RawIDs = RowIdxMask.columns
        if RawIDs.intersection(ids).shape[0]==0: return Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
        RowIdx[RowIdxMask] = np.nan
        RowIdx = adjustDataDTID(Panel({"RowIdx": RowIdx}), args.get("LookBack", 0), ["RowIdx"], RowIdx.columns.tolist(), dts, args.get("OnlyStartLookBack", False), args.get("OnlyLookBackNontarget", False), only_lookback_dt=True, logger=kwargs.get("logger", None)).iloc[0].values
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
        return adjustDataDTID(Data, args.get("LookBack", 0), factor_names, ids, dts, args.get("OnlyStartLookBack", False), args.get("OnlyLookBackNontarget", False), logger=kwargs.get("logger", None))
