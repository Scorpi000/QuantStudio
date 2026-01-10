# coding=utf-8

import numpy as np
import pandas as pd

from QuantStudio.Tools.api import Panel


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


