# -*- coding: utf-8 -*-
import os
import stat
import time
import shutil
import tempfile
import datetime as dt
from typing import Optional, List
from multiprocessing import Lock

import numpy as np
import pandas as pd
from pydantic import Field, DirectoryPath

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Tools.QSObjects import QSFileLock


class FactorCache(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        PIDs: list[str] = Field(title="进程ID", frozen=True)
        DTRuler: List[dt.datetime] = Field(title="时点标尺", description="当前运行计算时点标尺", frozen=True)
        MinDTUnit: dt.timedelta = Field(default=dt.timedelta(1), title="最小时间单位", frozen=True)
        ClearStart: bool = Field(default=True, title="启动时清空")

    def __init__(self, args: dict={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._CachedDTRange = {}  # 已经缓存的因子数据时点范围, {因子 QSID: DataFrame(columns=["StartDT", "EndDT"])}

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

    # 获取给定因子真正需要计算的时点范围
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
                    StartDT = StartIdx["EndDT"].iloc[0] + self.MinDTUnit
                    EndDT = EndIdx["StartDT"].iloc[0] - self.MinDTUnit
                    return (StartDT, EndDT)
            elif StartIdx.empty and (not EndIdx.empty):  # 新区间起始点在空档里, 结束点在已有区间里
                EndDT = EndIdx["StartDT"].iloc[0] - self.MinDTUnit
                return (dt_range[0], EndDT)
            else:  # 新区间起始点在已有区间里, 结束点在空档里
                StartDT = StartIdx["EndDT"].iloc[0] + self.MinDTUnit
                return (StartDT, dt_range[1])

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
        raise NotImplementedError

    # 结束缓存
    def end(self, clear=True):
        raise NotImplementedError

    # 原始数据缓存是否存在
    def checkRawDataExistence(self, key, pids=None, if_not_exists="create"):
        raise NotImplementedError

    # 写入原始数据
    # raw_data: {field: DataFrame}
    def writeRawData(self, key, raw_data, id_col="ID", if_exists="append", pid_ids=None):
        raise NotImplementedError

    # 读取原始数据
    def readRawData(self, key, target_fields=None, pids=None):
        raise NotImplementedError

    # 清空原始数据
    def clearRawData(self, key=None):
        return

    # 因子缓存是否存在
    def checkFactorDataExistence(self, key, pids=None, if_not_exists="create"):
        raise NotImplementedError

    # 写入因子数据
    def writeFactorData(self, key, factor_data, pid_ids, pid=None, target_field="StdData", if_exists="append"):
        raise NotImplementedError

    # 读取因子数据
    def readFactorData(self, key, ipid, target_field="StdData", pids=None, wait=True, wait_seconds=0.1):
        raise NotImplementedError

    # 清空缓存
    def clearFactorData(self, key=None):
        if key:
            self._CachedDTRange.pop(key)
        else:
            self._CachedDTRange = {}


class HDF5Cache(FactorCache):
    class __QS_ArgClass__(FactorCache.__QS_ArgClass__):
        CacheDir: Optional[DirectoryPath] = Field(default=None, title="缓存目录", frozen=True)
        HDF5Suffix: str = Field(default=".h5", title="H5文件后缀", frozen=True)

    def __init__(self, args={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._isStarted = False
        self._CacheDir = None  # 缓存主目录
        self._RawDataDir = None  # 原始数据存放根目录
        self._FactorDataDir = None  # 因子数据存放根目录
        self._DataLock = None  # 访问该缓存的锁, 防止并发访问冲突
        self._PIDLock = {}  # 访问该缓存的锁, 防止并发访问冲突

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if (self._CacheDir is not None) and (not isinstance(self._CacheDir, str)):
            state["_CacheDir"] = self._CacheDir.name
        return state

    def start(self):
        if self._isStarted: return
        CacheDir = self._QSArgs.CacheDir
        if not os.path.isdir(CacheDir):
            if CacheDir: self._QS_Logger.warning(f"缓存目录 '{CacheDir}' 不存在, 将使用系统的临时文件夹")
            self._CacheDir = tempfile.TemporaryDirectory()
            self._RawDataDir = self._CacheDir.name + os.sep + "RawData"  # 原始数据存放根目录
            self._FactorDataDir = self._CacheDir.name + os.sep + "FactorData"  # 中间数据存放根目录
            LockFile = self._CacheDir.name + os.sep + "LockFile"
        else:
            self._CacheDir = str(CacheDir)
            self._RawDataDir = self._CacheDir + os.sep + "RawData"  # 原始数据存放根目录
            self._FactorDataDir = self._CacheDir + os.sep + "FactorData"  # 中间数据存放根目录
            LockFile = self._CacheDir + os.sep + "LockFile"
        if not os.path.isfile(LockFile):
            open(LockFile, mode="a").close()
            os.chmod(LockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
        self._DataLock = QSFileLock(LockFile, proc_lock=Lock())
        if self._QSArgs.ClearStart:
            self.clearRawData()
            self.clearFactorData()
        with self._DataLock:
            if not os.path.isdir(self._RawDataDir): os.mkdir(self._RawDataDir)
            if not os.path.isdir(self._FactorDataDir): os.mkdir(self._FactorDataDir)
            # 根据进程创建缓存子目录
            for iPID in self._QSArgs.PIDs:
                if not os.path.isdir(self._RawDataDir + os.sep + iPID): os.mkdir(self._RawDataDir + os.sep + iPID)
                if not os.path.isdir(self._FactorDataDir + os.sep + iPID): os.mkdir(self._FactorDataDir + os.sep + iPID)
                iLockFile = self._FactorDataDir + os.sep + iPID + os.sep + "LockFile"
                if not os.path.isfile(iLockFile):
                    open(iLockFile, mode="a").close()
                    os.chmod(iLockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
                self._PIDLock[iPID] = QSFileLock(iLockFile, proc_lock=Lock())
        self._isStarted = True

    def end(self, clear=True):
        if not self._isStarted: return
        if clear:
            self.clearRawData()
            self.clearFactorData()
            LockFile = (self._CacheDir if isinstance(self._CacheDir, str) else self._CacheDir.name) + os.sep + "LockFile"
            try:
                if os.path.isfile(LockFile): os.remove(LockFile)
            except Exception as e:
                self._QS_Logger.error(f"锁文件: {LockFile} 清理失败: {e}")
        self._isStarted = False

    def checkRawDataExistence(self, key, pids=None, if_not_exists="create"):
        if pids is None: pids = self._QSArgs.PIDs
        IfExist = False
        with self._DataLock:
            for iPID in pids:
                iFilePath = self._RawDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix
                if os.path.isfile(iFilePath):
                    IfExist = True
                    if if_not_exists != "create":
                        break
                elif if_not_exists == "create":
                    with pd.HDFStore(iFilePath) as iFile:
                        pass
        return IfExist

    def writeRawData(self, key, raw_data, pid_ids, id_col="QS_ID", if_exists="append"):
        if raw_data is None: return 0
        for iField, iRawData in raw_data.items():
            if isinstance(iRawData, pd.DataFrame) and (id_col in iRawData):  # 如果原始数据有 ID 列，按照 ID 列划分后存入子进程的原始文件中
                iRawData = iRawData.set_index([id_col])
                AllIDs = set(iRawData.index)
                for jPID, jIDs in pid_ids.items():
                    jInterIDs = sorted(AllIDs.intersection(jIDs))
                    ijRawData = iRawData.loc[jInterIDs]
                    with self._DataLock:
                        with pd.HDFStore(self._RawDataDir + os.sep + jPID + os.sep + key + self._QSArgs.HDF5Suffix, mode="a") as jFile:
                            if (iField in jFile) and (if_exists == "append"):
                                jFile[iField] = pd.concat([jFile[iField], ijRawData.reset_index()], ignore_index=True)
                            else:
                                jFile[iField] = ijRawData.reset_index()
            else:  # 如果原始数据没有 ID 列，则将所有数据分别存入子进程的原始文件中
                for jPID, jIDs in pid_ids.items():
                    with self._DataLock:
                        with pd.HDFStore(self._RawDataDir + os.sep + jPID + os.sep + key + self._QSArgs.HDF5Suffix, mode="a") as jFile:
                            if (iField in jFile) and (if_exists == "append"):
                                jFile[iField] = pd.concat([jFile[iField], iRawData], ignore_index=True)
                            else:
                                jFile[iField] = iRawData

    def readRawData(self, key, target_fields=None, pids=None):
        if pids is None: pids = self._QSArgs.PIDs
        RawData = {}
        for iPID in pids:
            iRawDataFilePath = self._RawDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix
            with self._PIDLock[iPID]:
                if os.path.isfile(iRawDataFilePath):
                    with pd.HDFStore(iRawDataFilePath, mode="r") as iFile:
                        iFields = (sorted(jField[1:] for jField in iFile) if target_fields is None else target_fields)
                        for jField in iFields:
                            if jField in iFile:
                                RawData.setdefault(jField, []).append(iFile[jField])
        RawData = {iKey: pd.concat(iVal, ignore_index=True) for iKey, iVal in RawData.items()}
        return RawData

    def clearRawData(self, key=None):
        with self._DataLock:
            if key is None:
                try:
                    if os.path.isdir(self._RawDataDir): shutil.rmtree(self._RawDataDir)
                except Exception as e:
                    self._QS_Logger.error(f"原始数据缓存目录: {self._RawDataDir} 清理失败: {e}")
            else:
                for iPID in self._QSArgs.PIDs:
                    iRawDataFilePath = self._RawDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix
                    try:
                        if os.path.isfile(iRawDataFilePath): os.remove(iRawDataFilePath)
                    except Exception as e:
                        self._QS_Logger.error(f"原始数据缓存文件: {iRawDataFilePath} 清理失败: {e}")
        return super().clearRawData(key=key)

    def checkFactorDataExistence(self, key, pids=None):
        if pids is None: pids = self._QSArgs.PIDs
        IfExist = False
        with self._DataLock:
            for iPID in pids:
                iFilePath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix
                with pd.HDFStore(iFilePath) as iFile:
                    IfExist = ("StdData" in iFile) or IfExist
        return IfExist

    def writeFactorData(self, key, factor_data, pid_ids, pid=None, target_field="StdData", if_exists="append"):
        PIDs = (self._QSArgs.PIDs if pid is None else [pid])
        for iPID in PIDs:
            with self._PIDLock[iPID]:
                if pid_ids is not None:
                    iIDs = pid_ids.get(iPID)
                    with pd.HDFStore(
                            self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix) as CacheFile:
                        if (target_field in CacheFile) and (if_exists == "append"):
                            iOldData = CacheFile[target_field]
                            iFactorData = factor_data.reindex(columns=iIDs)
                            iFactorData = pd.concat([iOldData, iFactorData], ignore_index=False)
                            iFactorData = iFactorData[~iFactorData.index.duplicated()]
                            CacheFile[target_field] = iFactorData.sort_index()
                        else:
                            CacheFile[target_field] = factor_data.reindex(columns=iIDs)
                else:
                    with pd.HDFStore(
                            self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix) as CacheFile:
                        if (target_field in CacheFile) and (if_exists == "append"):
                            CacheFile[target_field] = pd.concat([CacheFile[target_field], factor_data], ignore_index=False).sort_index()
                        else:
                            CacheFile[target_field] = factor_data

    def readFactorData(self, key, ipid, target_field="StdData", pids=None, wait=True, wait_seconds=0.1):
        if isinstance(pids, str):
            FilePath = self._FactorDataDir + os.sep + pids + os.sep + key + self._QSArgs.HDF5Suffix
            if not os.path.isfile(FilePath):
                return None
            with self._PIDLock[pids]:
                with pd.HDFStore(FilePath, mode="r") as CacheFile:
                    FactorData = CacheFile[target_field]
            return FactorData
        iFilePath = self._FactorDataDir + os.sep + ipid + os.sep + key + self._QSArgs.HDF5Suffix
        with self._PIDLock[ipid]:
            with pd.HDFStore(iFilePath, mode="r") as CacheFile:
                DTNum = CacheFile.get_storer(target_field).shape
                if DTNum is None:
                    DTNum = CacheFile[target_field].shape[0]
                else:
                    DTNum = DTNum[0]
        if pids is None:
            pids = {ipid}
        else:
            pids = set(pids)
        StdData = []
        MTime = {}
        while len(pids) > 0:
            iPID = pids.pop()
            iFilePath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix
            if not os.path.isfile(iFilePath):  # 该进程的数据没有准备好
                if wait:
                    pids.add(iPID)
                    if wait_seconds > 0: time.sleep(wait_seconds)
                continue
            elif wait:
                iMTime = os.path.getmtime(iFilePath)
                if (iPID not in MTime) or (iMTime > MTime[iPID]):
                    MTime[iPID] = iMTime
                    with self._PIDLock[iPID]:
                        with pd.HDFStore(iFilePath, mode="r") as CacheFile:
                            iDTNum = CacheFile.get_storer(target_field).shape
                            if iDTNum is None:
                                iDTNum = CacheFile[target_field].shape[0]
                            else:
                                iDTNum = iDTNum[0]
                    if iDTNum != DTNum:
                        pids.add(iPID)
                        if wait_seconds > 0: time.sleep(wait_seconds)
                        continue
                else:
                    pids.add(iPID)
                    if wait_seconds > 0: time.sleep(wait_seconds)
                    continue
            iStdData = self.readFactorData(key, ipid, target_field=target_field, pids=iPID)
            if iStdData is not None: StdData.append(iStdData)
        if StdData:
            return pd.concat(StdData, axis=1, join='outer', ignore_index=False)
        else:
            return None

    def clearFactorData(self, key=None):
        with self._DataLock:
            if key is None:
                try:
                    if os.path.isdir(self._FactorDataDir): shutil.rmtree(self._FactorDataDir)
                except Exception as e:
                    self._QS_Logger.error(f"因子数据缓存目录: {self._FactorDataDir} 清理失败: {e}")
            else:
                for iPID in self._QSArgs.PIDs:
                    iFilePath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.HDF5Suffix
                    try:
                        if os.path.isfile(iFilePath): os.remove(iFilePath)
                    except Exception as e:
                        self._QS_Logger.error(f"因子数据缓存文件: {iFilePath} 清理失败: {e}")
        return super().clearFactorData(key=key)
