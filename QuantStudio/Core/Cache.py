# -*- coding: utf-8 -*-
import os
import stat
import time
import shutil
import pickle
import tempfile
import datetime as dt
from typing import Optional, List, Literal, Dict
from multiprocessing import Lock

import numpy as np
import pandas as pd
from pydantic import Field, DirectoryPath, FilePath

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.QSObject import QSFileLock

# 缓存的数据类型: DataFrame
class Cache(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        StartMode: Literal["new", "continue"] = Field(default="new", title="启动模式", description="启动缓存的方式: new 表示清空已有数据重新构造缓存；continue 表示通过 load 恢复之前的缓存状态")

    def __init__(self, args: dict={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._isStarted = False # 缓存是否已经启动

    # 并发运行后返回需要同步的内容
    def getUpdateData(self) -> dict:
        return {}

    # 并发运行后更新同步内容
    def updateCache(self, update_data: dict):
        pass
    
    # 暂存缓存状态
    def dump(self):
        raise NotImplementedError

    # 恢复缓存状态
    def load(self):
        raise NotImplementedError
    

    # 初始化缓存
    def start(self):
        if self._isStarted: return
        if self._QSArgs.StartMode == "new":
            self.clearData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        self._isStarted = True

    # 结束缓存
    def end(self, clear=False):
        if not self._isStarted: return
        if clear:
            self.clearData()
        else:
            self.dump()
        self._isStarted = False

    # 缓存是否存在
    def checkDataExistence(self, key: str, create_if_not_exists: bool=False):
        raise NotImplementedError

    # 写入数据
    def writeData(self, key: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="append", data_type: Optional[str]=None):
        raise NotImplementedError

    # 读取数据
    def readData(self, key: str, data_type: Optional[str]=None):
        raise NotImplementedError
    
    # 清理数据
    def clearData(self, key: Optional[str] = None):
        raise NotImplementedError


# 缓存的数据类型: DataFrame(index=[dt.datetime])
class DTCache(Cache):
    class __QS_ArgClass__(Cache.__QS_ArgClass__):
        DTRuler: List[dt.datetime] = Field(title="时点标尺", description="当前运行计算时点标尺", frozen=True)
        MinDTUnit: dt.timedelta = Field(default=dt.timedelta(1), title="最小时间单位", frozen=True)

    def __init__(self, args: dict={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._CachedDTRange = {}# 已经缓存的数据时点范围, {因子 QSID: DataFrame(columns=["StartDT", "EndDT"])}

    # 并发运行后返回需要同步的内容
    def getUpdateData(self) -> dict:
        return {"_CachedDTRange": self._CachedDTRange}

    # 并发运行后更新同步内容
    def updateCache(self, update_data: dict):
        for iFactorID, iDTRange in update_data.get("_CachedDTRange", {}).items():
            if iFactorID not in self._CachedDTRange:
                self._CachedDTRange[iFactorID] = iDTRange
            else:
                for iDTRange in iDTRange.astype("O").to_records(index=False):
                    self.updateDTRange(iFactorID, iDTRange)

    # 更新缓存的时点范围
    def _mergeDTRange(self, cached_dt_range, i):
        DTRuler = self._QSArgs.DTRuler
        iStartDT, iEndDT = cached_dt_range["StartDT"].iloc[i], cached_dt_range["EndDT"].iloc[i]
        DropIdx = []
        if i > 0:
            iPreEndDT = cached_dt_range["EndDT"].iloc[i - 1]
            iPreEndIdx, iStartdIdx = np.searchsorted(DTRuler, iPreEndDT, side="left"), np.searchsorted(DTRuler, iStartDT, side="right")
            if iPreEndIdx == iStartdIdx - 2:  # 两区间连续，合并
                cached_dt_range.at[cached_dt_range.index[i], "StartDT"] = cached_dt_range["StartDT"].iloc[i - 1]
                DropIdx.append(cached_dt_range.index[i - 1])
        if i < cached_dt_range.shape[0] - 1:
            iPostStartDT = cached_dt_range["StartDT"].iloc[i + 1]
            iEndIdx, iPostStartIdx = np.searchsorted(DTRuler, iEndDT, side="left"), np.searchsorted(DTRuler, iPostStartDT, side="right")
            if iEndIdx == iPostStartIdx - 2:  # 两区间连续，合并
                cached_dt_range.at[cached_dt_range.index[i], "EndDT"] = cached_dt_range["EndDT"].iloc[i + 1]
                DropIdx.append(cached_dt_range.index[i + 1])
        if DropIdx: cached_dt_range = cached_dt_range.drop(index=DropIdx)
        return cached_dt_range

    # 给定时点范围，获取缓存中缺失的时点范围
    def getDTRange(self, key, dt_range):
        CachedDTRange = self._CachedDTRange.get(key, None)
        if CachedDTRange is None:
            return dt_range
        else:
            StartIdx = CachedDTRange[
                (CachedDTRange["StartDT"] <= dt_range[0]) & (CachedDTRange["EndDT"] >= dt_range[0])]
            EndIdx = CachedDTRange[
                (CachedDTRange["StartDT"] <= dt_range[1]) & (CachedDTRange["EndDT"] >= dt_range[1])]
            if StartIdx.empty and EndIdx.empty:  # 新区间起始结束点均在空档里
                return dt_range
            elif (not StartIdx.empty) and (not EndIdx.empty):  # 新区间起始结束点均在已有区间里
                StartIdx, EndIdx = StartIdx.index[0], EndIdx.index[0]
                if StartIdx == EndIdx:  # 已有区间完全覆盖新区间
                    return None
                else:  # 新区间跨区间
                    StartDT = StartIdx["EndDT"].iloc[0] + self._QSArgs.MinDTUnit
                    EndDT = EndIdx["StartDT"].iloc[0] - self._QSArgs.MinDTUnit
                    return (StartDT, EndDT)
            elif StartIdx.empty and (not EndIdx.empty):  # 新区间起始点在空档里, 结束点在已有区间里
                EndDT = EndIdx["StartDT"].iloc[0] - self._QSArgs.MinDTUnit
                return (dt_range[0], EndDT)
            else:  # 新区间起始点在已有区间里, 结束点在空档里
                StartDT = StartIdx["EndDT"].iloc[0] + self._QSArgs.MinDTUnit
                return (StartDT, dt_range[1])
    
    # 更新缓存中数据的时点范围
    def updateDTRange(self, key, dt_range):
        CachedDTRange = self._CachedDTRange.get(key, None)
        if CachedDTRange is None:
            self._CachedDTRange[key] = pd.DataFrame([dt_range], columns=["StartDT", "EndDT"])
            return
        StartIdx = CachedDTRange[
            (CachedDTRange["StartDT"] <= dt_range[0]) & (CachedDTRange["EndDT"] >= dt_range[0] - self._QSArgs.MinDTUnit)]
        EndIdx = CachedDTRange[
            (CachedDTRange["StartDT"] <= dt_range[1] + self._QSArgs.MinDTUnit) & (CachedDTRange["EndDT"] >= dt_range[1])]
        nRange = CachedDTRange.shape[0]
        if StartIdx.empty and EndIdx.empty:  # 新区间起始结束点均在空档里
            CachedDTRange = CachedDTRange[
                ~((CachedDTRange["StartDT"] >= dt_range[0]) & (CachedDTRange["EndDT"] <= dt_range[1]))]
            CachedDTRange.loc[nRange] = list(dt_range)
            CachedDTRange = CachedDTRange.sort_values(["StartDT"])
            i = CachedDTRange.index.tolist().index(nRange)
            self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)
        elif (not StartIdx.empty) and (not EndIdx.empty):  # 新区间起始结束点均在已有区间里
            StartIdx, EndIdx = StartIdx.index[0], EndIdx.index[0]
            if StartIdx == EndIdx:  # 已有区间完全覆盖新区间
                return
            else:  # 新区间跨区间
                StartDT, EndDT = CachedDTRange.at[StartIdx, "StartDT"], CachedDTRange.at[EndIdx, "EndDT"]
                CachedDTRange = CachedDTRange[(CachedDTRange.index < StartIdx) | (CachedDTRange.index > EndIdx)]
                CachedDTRange.loc[nRange] = [StartDT, EndDT]
                CachedDTRange = CachedDTRange.sort_values(["StartDT"])
                i = CachedDTRange.index.tolist().index(nRange)
                self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)
        elif StartIdx.empty and (not EndIdx.empty):  # 新区间起始点在空档里, 结束点在已有区间里
            EndDT = EndIdx["EndDT"].iloc[0]
            CachedDTRange = CachedDTRange[
                ~((CachedDTRange["StartDT"] >= dt_range[0]) & (CachedDTRange["EndDT"] <= EndDT))]
            CachedDTRange.loc[nRange] = [dt_range[0], EndDT]
            CachedDTRange = CachedDTRange.sort_values(["StartDT"])
            i = CachedDTRange.index.tolist().index(nRange)
            self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)
        else:  # 新区间起始点在已有区间里, 结束点在空档里
            StartDT = StartIdx["StartDT"].iloc[0]
            CachedDTRange = CachedDTRange[
                ~((CachedDTRange["StartDT"] >= StartDT) & (CachedDTRange["EndDT"] <= dt_range[1]))]
            CachedDTRange.loc[nRange] = [StartDT, dt_range[1]]
            CachedDTRange = CachedDTRange.sort_values(["StartDT"])
            i = CachedDTRange.index.tolist().index(nRange)
            self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)
    
    # 初始化缓存
    def start(self):
        if self._isStarted: return
        if self._QSArgs.StartMode == "new":
            self.clearData()
            self.clearDTData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        self._isStarted = True

    # 结束缓存
    def end(self, clear=False):
        if not self._isStarted: return
        if clear:
            self.clearData()
            self.clearDTData()
        else:
            self.dump()
        self._isStarted = False    
    
    # 时点数据缓存是否存在
    def checkDTDataExistence(self, key: str, create_if_not_exists: bool=False):
        raise NotImplementedError

    # 写入时点数据
    def writeDTData(self, key: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="append", data_type: Optional[str]=None):
        raise NotImplementedError

    # 读取时点数据
    def readDTData(self, key: str, data_type: Optional[str]=None):
        raise NotImplementedError
    
    # 清理时点数据
    def clearDTData(self, key: Optional[str] = None):
        raise NotImplementedError