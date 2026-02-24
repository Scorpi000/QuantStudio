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

class Cache(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        PIDs: list[str] = Field(title="进程ID", frozen=True)
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
    
    def clear(self):
        raise NotImplementedError

    # 初始化缓存
    def start(self):
        if self._isStarted: return
        if self._QSArgs.StartMode == "new":
            self.clear()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        self._isStarted = True

    # 结束缓存
    def end(self, clear=False):
        if not self._isStarted: return
        if clear:
            self.clear()
        else:
            self.dump()
        self._isStarted = False

    # 缓存是否存在
    def checkExistence(self, key, pids=None, if_not_exists="create"):
        raise NotImplementedError

    # 写入数据
    def write(self, key:str, data:pd.DataFrame, pid_ids:Dict[str, List[str]], pid=None, target_field="StdData", if_exists="append", data_type=None):
        raise NotImplementedError

    # 读取数据
    def read(self, key, ipid, target_field="StdData", pids=None, wait=True, wait_seconds=0.1, data_type=None):
        raise NotImplementedError


class FileCache(Cache):
    class __QS_ArgClass__(Cache.__QS_ArgClass__):
        CacheDir: Optional[DirectoryPath] = Field(default=None, title="缓存目录", frozen=True)
        StateFile: FilePath = Field(default="state.pkl", title="状态文件", frozen=True, description="用于存储缓存的状态")
        Suffix: str = Field(default="", title="后缀", frozen=True)

    def __init__(self, args={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._CacheDir = None  # 缓存主目录
        self._DataDir = None  # 数据存放根目录
        self._DataLock = None  # 访问该缓存的锁, 防止并发访问冲突
        self._PIDLock = {}  # 访问该缓存的锁, 防止并发访问冲突

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if (self._CacheDir is not None) and (not isinstance(self._CacheDir, str)):
            state["_CacheDir"] = self._CacheDir.name
        return state
    
    def createPath(self, path: str):
        raise NotImplementedError
    
    def getPathMTime(self, path: str):
        raise NotImplementedError
    
    def writeDataFramePickle(self, path: str, target_field: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="replace", ignore_index=True, file_suffix=".pkl"):
        if not os.path.isdir(path): os.makedirs(path, exist_ok=True)
        FilePath = os.path.join(path, target_field + file_suffix)
        if (if_exists=="append") and os.path.isfile(FilePath):
            OldData = pd.read_pickle(FilePath)
            Data = pd.concat([OldData, data], ignore_index=ignore_index)
            if not ignore_index: Data = Data[~Data.index.duplicated()]
            Data.sort_index().to_pickle(FilePath)
        else:
            data.to_pickle(FilePath)

    def readDataFramePickle(self, path: str, target_fields: Optional[list[str]]=None, file_suffix=".pkl"):
        if not os.path.isdir(path): return {}
        if not target_fields: target_fields = [iFile[:-len(file_suffix)] for iFile in os.listdir(path)]
        RawData = {}
        for iField in target_fields:
            iPath = os.path.join(path, iField + file_suffix)
            if os.path.isfile(iPath):
                RawData[iField] = pd.read_pickle(iPath)
        return RawData

    def writeDataFrame(self, path: str, target_field: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="replace", ignore_index=True, data_type=None):
        raise NotImplementedError

    def readDataFrame(self, path: str, target_fields: Optional[list[str]]=None, data_type=None):
        raise NotImplementedError
    
    # 暂存缓存状态
    def dump(self):
        State = {
            "_CachedDTRange": self._CachedDTRange
        }
        if os.path.isfile(self._QSArgs.StateFile):
            StateFilePath = self._QSArgs.StateFile
        else:
            StateFilePath = os.path.join(self._QSArgs.CacheDir, self._QSArgs.StateFile)
        with open(StateFilePath, mode="wb") as StateFile:
            pickle.dump(State, StateFile)

    # 恢复缓存状态
    def load(self):
        if os.path.isfile(self._QSArgs.StateFile):
            StateFilePath = self._QSArgs.StateFile
        else:
            StateFilePath = os.path.join(self._QSArgs.CacheDir, self._QSArgs.StateFile)
        with open(StateFilePath, mode="rb") as StateFile:
            State = pickle.load(StateFile)
        self.__dict__.update(State)

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
        if self._QSArgs.StartMode == "new":
            self.clearRawData()
            self.clearFactorData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
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

    def end(self, clear=False):
        if not self._isStarted: return
        if clear:
            self.clearRawData()
            self.clearFactorData()
            LockFile = (self._CacheDir if isinstance(self._CacheDir, str) else self._CacheDir.name) + os.sep + "LockFile"
            try:
                if os.path.isfile(LockFile): os.remove(LockFile)
            except Exception as e:
                self._QS_Logger.error(f"锁文件: {LockFile} 清理失败: {e}")
        else:
            self.dump()
        self._isStarted = False

    def checkExistence(self, key, pids=None):
        if pids is None: pids = self._QSArgs.PIDs
        IfExist = False
        with self._DataLock:
            for iPID in pids:
                iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                IfExist = os.path.exists(iPath) or IfExist
        return IfExist

    def write(self, key, factor_data, pid_ids, pid=None, target_field="StdData", if_exists="append", data_type=None):
        PIDs = (self._QSArgs.PIDs if pid is None else [pid])
        for iPID in PIDs:
            with self._PIDLock[iPID]:
                if pid_ids is not None:
                    iIDs = pid_ids.get(iPID)
                    iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                    self.writeDataFrame(path=iPath, target_field=target_field, data=factor_data.reindex(columns=iIDs), if_exists=if_exists, ignore_index=False, data_type=data_type)
                else:
                    iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                    self.writeDataFrame(path=iPath, target_field=target_field, data=factor_data, if_exists=if_exists, ignore_index=False, data_type=data_type)

    def read(self, key, ipid, target_field="StdData", pids=None, wait=True, wait_seconds=0.1, data_type=None):
        if isinstance(pids, str):
            Path = self._FactorDataDir + os.sep + pids + os.sep + key + self._QSArgs.Suffix
            if not os.path.exists(Path):
                return None
            with self._PIDLock[pids]:
                return self.readDataFrame(path=Path, target_fields=[target_field], data_type=data_type).get(target_field, None)
        iPath = self._FactorDataDir + os.sep + ipid + os.sep + key + self._QSArgs.Suffix
        with self._PIDLock[ipid]:
            DTNum = self.readDataFrame(path=iPath, target_fields=[target_field], data_type=data_type).get(target_field, pd.DataFrame()).shape[0]
        if pids is None:
            pids = {ipid}
        else:
            pids = set(pids)
        StdData = []
        MTime = {}
        while len(pids) > 0:
            iPID = pids.pop()
            iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
            if not os.path.exists(iPath):  # 该进程的数据没有准备好
                if wait:
                    pids.add(iPID)
                    if wait_seconds > 0: time.sleep(wait_seconds)
                continue
            elif wait:
                iMTime = self.getPathMTime(iPath)
                if (iPID not in MTime) or (iMTime > MTime[iPID]):
                    MTime[iPID] = iMTime
                    iDTNum = self.readDataFrame(path=iPath, target_fields=[target_field], data_type=data_type).get(target_field, pd.DataFrame()).shape[0]
                    if iDTNum != DTNum:
                        pids.add(iPID)
                        if wait_seconds > 0: time.sleep(wait_seconds)
                        continue
                else:
                    pids.add(iPID)
                    if wait_seconds > 0: time.sleep(wait_seconds)
                    continue
            iStdData = self.readFactorData(key, ipid, target_field=target_field, pids=iPID, data_type=data_type)
            if iStdData is not None: StdData.append(iStdData)
        if StdData:
            return pd.concat(StdData, axis=1, join='outer', ignore_index=False)
        else:
            return None

    def clear(self, key=None):
        with self._DataLock:
            if key is None:
                try:
                    if os.path.isdir(self._FactorDataDir): shutil.rmtree(self._FactorDataDir)
                except Exception as e:
                    self._QS_Logger.error(f"因子数据缓存目录: {self._FactorDataDir} 清理失败: {e}")
            else:
                for iPID in self._QSArgs.PIDs:
                    iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                    try:
                        if os.path.isfile(iPath): os.remove(iPath)
                        elif os.path.isdir(iPath): shutil.rmtree(iPath)
                    except Exception as e:
                        self._QS_Logger.error(f"因子数据缓存: {iPath} 清理失败: {e}")
        return super().clearFactorData(key=key)
