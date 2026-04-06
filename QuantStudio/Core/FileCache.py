# -*- coding: utf-8 -*-
import os
import stat
import json
import shutil
import pickle
import tempfile
import datetime as dt
from typing import Optional, Literal

import numpy as np
import pandas as pd
from pyarrow import ArrowInvalid
from pydantic import Field, DirectoryPath, FilePath

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.QSObject import QSFileLock
from QuantStudio.Core.Cache import DTCache

class FileDTCache(DTCache):
    """基于文件的时序数据缓存"""

    class __QS_ArgClass__(DTCache.__QS_ArgClass__):
        CacheDir: Optional[DirectoryPath] = Field(default=None, title="缓存目录", frozen=True)
        StateFile: FilePath = Field(default="state.pkl", title="状态文件", frozen=True, description="用于存储缓存的状态")
        Suffix: str = Field(default="", title="后缀", frozen=True)

    def __init__(self, proc_lock=None, args={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._CacheDir = None# 缓存主目录
        self._DataDir = None# 通用数据存放根目录
        self._DTDataDir = None# 时点数据存放根目录
        self._DataLock = None# 访问该缓存的锁, 防止并发访问冲突
        self._ProcLock = proc_lock

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if hasattr(self, "_CacheDirObj"):
            state["_CacheDirObj"] = self._CacheDirObj.name
        return state
    
    def writeDataFramePickle(self, path: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="replace", ignore_index:bool=True):
        Dir = os.path.split(path)[0]
        if not os.path.isdir(Dir): os.makedirs(Dir, exist_ok=True)
        if (if_exists=="append") and os.path.isfile(path):
            OldData = pd.read_pickle(path)
            Data = pd.concat([OldData, data], ignore_index=ignore_index)
            if not ignore_index: Data = Data[~Data.index.duplicated()]
            Data.sort_index().to_pickle(path)
        else:
            data.to_pickle(path)

    def readDataFramePickle(self, path: str) -> None | pd.DataFrame:
        if not os.path.isfile(path): return None
        return pd.read_pickle(path)

    def writeDataFrame(self, path: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="replace", ignore_index: bool=True, data_type: Optional[Literal["double", "string", "object"]]=None):
        raise NotImplementedError

    def readDataFrame(self, path: str, data_type: Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        raise NotImplementedError
    
    def writeMeta(self, path: str, meta:dict):
        Dir = os.path.split(path)[0]
        if not os.path.isdir(Dir): os.makedirs(Dir, exist_ok=True)
        with open(path, mode="w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def dump(self):
        State = {"_CachedDTRange": self._CachedDTRange}
        if os.path.isfile(self._QSArgs.StateFile):
            StateFilePath = self._QSArgs.StateFile
        else:
            StateFilePath = os.path.join(self._CacheDir, self._QSArgs.StateFile)
        with open(StateFilePath, mode="wb") as StateFile:
            pickle.dump(State, StateFile)

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
            self._CacheDirObj = tempfile.TemporaryDirectory()
            self._CacheDir = self._CacheDirObj.name
        else:
            self._CacheDir = str(CacheDir)
        self._DataDir = self._CacheDir + os.sep + "Data"# 通用数据存放根目录
        self._DTDataDir = self._CacheDir + os.sep + "DTData"# 时点数据存放根目录
        LockFile = self._CacheDir + os.sep + "LockFile"
        if not os.path.isfile(LockFile):
            open(LockFile, mode="a").close()
            os.chmod(LockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
        self._DataLock = QSFileLock(LockFile, proc_lock=self._ProcLock)
        if self._QSArgs.StartMode == "new":
            self.clearData()
            self.clearDTData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        with self._DataLock:
            if not os.path.isdir(self._DataDir): os.mkdir(self._DataDir)
            if not os.path.isdir(self._DTDataDir): os.mkdir(self._DTDataDir)
        self._isStarted = True

    def checkDataExistence(self, key: str, create_if_not_exists: bool=False) -> bool:
        with self._DataLock:
            Path = self._DataDir + os.sep + key + self._QSArgs.Suffix
            ifExist = os.path.exists(Path)
            if (not ifExist) and create_if_not_exists:
                with open(Path, mode="w") as File:
                    pass
        return ifExist

    def writeData(self, key: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="append", data_type: Optional[Literal["double", "string", "object"]]=None, meta:dict={}):
        Path = self._DataDir + os.sep + key + self._QSArgs.Suffix
        with self._DataLock:
            self.writeDataFrame(path=Path, data=data, if_exists=if_exists, ignore_index=False, data_type=data_type)
            if meta: self.writeMeta(path=self._DataDir + os.sep + key + "_meta.json", meta=meta)

    def readData(self, key: str, data_type: Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        Path = self._DataDir + os.sep + key + self._QSArgs.Suffix
        with self._DataLock:
            return self.readDataFrame(path=Path, data_type=data_type)

    def clearData(self, key: Optional[str]=None):
        with self._DataLock:
            if key is None:
                try:
                    if os.path.isdir(self._DataDir): shutil.rmtree(self._DataDir)
                except Exception as e:
                    self._QS_Logger.error(f"通用数据缓存目录: {self._DataDir} 清理失败: {e}")
            else:
                iPath = self._DataDir + os.sep + key + self._QSArgs.Suffix
                try:
                    if os.path.isfile(iPath): os.remove(iPath)
                    elif os.path.isdir(iPath): shutil.rmtree(iPath)
                except Exception as e:
                    self._QS_Logger.error(f"通用数据缓存: {iPath} 清理失败: {e}")
    
    def checkDTDataExistence(self, key: str, create_if_not_exists: bool=False) -> bool:
        with self._DataLock:
            Path = self._DTDataDir + os.sep + key + self._QSArgs.Suffix
            ifExist = os.path.exists(Path)
            if (not ifExist) and create_if_not_exists:
                with open(Path, mode="w") as File:
                    pass
        return ifExist

    def writeDTData(self, key: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="append", data_type: Optional[Literal["double", "string", "object"]]=None, meta:dict={}):
        Path = self._DTDataDir + os.sep + key + self._QSArgs.Suffix
        with self._DataLock:
            self.writeDataFrame(path=Path, data=data, if_exists=if_exists, ignore_index=False, data_type=data_type)
            if meta: self.writeMeta(path=self._DTDataDir + os.sep + key + "_meta.json", meta=meta)

    def readDTData(self, key: str, data_type: Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        Path = self._DTDataDir + os.sep + key + self._QSArgs.Suffix
        with self._DataLock:
            return self.readDataFrame(path=Path, data_type=data_type)
    
    def clearDTData(self, key: Optional[str] = None):
        with self._DataLock:
            if key is None:
                try:
                    if os.path.isdir(self._DTDataDir): shutil.rmtree(self._DTDataDir)
                except Exception as e:
                    self._QS_Logger.error(f"时点数据缓存目录: {self._DTDataDir} 清理失败: {e}")
            else:
                iPath = self._DTDataDir + os.sep + key + self._QSArgs.Suffix
                try:
                    if os.path.isfile(iPath): os.remove(iPath)
                    elif os.path.isdir(iPath): shutil.rmtree(iPath)
                except Exception as e:
                    self._QS_Logger.error(f"时点数据缓存: {iPath} 清理失败: {e}")


class FeatherDTCache(FileDTCache):
    """基于 Feather 格式文件的时序数据缓存"""
    
    class __QS_ArgClass__(FileDTCache.__QS_ArgClass__):
        Suffix: str = Field(default=".feather", title="后缀", frozen=True)
    
    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(args=args, config_file=(__QS_ConfigPath__ + os.sep + "FeatherDTCacheConfig.json" if config_file is None else config_file), **kwargs)
    
    def writeDataFrame(self, path: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="replace", ignore_index: bool=True, data_type: Optional[Literal["double", "string", "object"]]=None):
        if data_type=="object": return self.writeDataFramePickle(path=path[:len(path)-len(self._QSArgs.Suffix)]+".pkl", data=data, if_exists=if_exists, ignore_index=ignore_index)
        Dir = os.path.split(path)[0]
        if not os.path.isdir(Dir): os.makedirs(Dir, exist_ok=True)
        if (if_exists=="append") and os.path.isfile(path):
            OldData = pd.read_feather(path)
            Data = pd.concat([OldData, data], ignore_index=ignore_index)
            if not ignore_index: Data = Data[~Data.index.duplicated()]
            Data.sort_index().to_feather(path, compression="uncompressed")
        else:
            data.to_feather(path, compression="uncompressed")

    def readDataFrame(self, path: str, data_type: Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        if data_type=="object": return self.readDataFramePickle(path=path[:len(path)-len(self._QSArgs.Suffix)]+".pkl")
        if not os.path.isfile(path): return None
        try:
            return pd.read_feather(path)
        except ArrowInvalid:
            return None



if __name__ == "__main__":
    DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(100)]
    
    Cache = FeatherDTCache(args={"Suffix": ".feather", "CacheDir": r"C:\Users\hst\Project\Data\FactorCache", "DTRuler": DTRuler,})
    Cache.start()
    Cache.writeData(key="aha", data=pd.DataFrame(np.random.rand(5, 3)))
    Data = Cache.readData(key="aha")
    print(Data)
    
    Cache.writeDTData(key="aha", data=pd.DataFrame(np.random.rand(5, 3), index=DTRuler[:5]))
    Data = Cache.readDTData(key="aha")
    print(Data)
    
    print("===")