# -*- coding: utf-8 -*-
import os
import stat
import time
import shutil
import pickle
import tempfile
import datetime as dt
from typing import Optional, List, Literal
from multiprocessing import Lock

import numpy as np
import pandas as pd
from pydantic import Field, DirectoryPath, FilePath

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.QSObject import QSFileLock
from QuantStudio.Core.Cache import DTCache
from QuantStudio.Core.FileCache import FileDTCache, FeatherDTCache


class FactorCache(DTCache):
    class __QS_ArgClass__(DTCache.__QS_ArgClass__):
        PIDs: list[str] = Field(default=["0"], title="进程ID", frozen=True)

    def start(self):
        if self._isStarted: return
        if self._QSArgs.StartMode == "new":
            self.clearData()
            self.clearDTData()
            self.clearRawData()
            self.clearFactorData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        self._isStarted = True

    def end(self, clear=False):
        if not self._isStarted: return
        if clear:
            self.clearData()
            self.clearDTData()
            self.clearRawData()
            self.clearFactorData()
        else:
            self.dump()
        self._isStarted = False

    # 原始数据缓存是否存在
    def checkRawDataExistence(self, key, pids=None, if_not_exists="create"):
        raise NotImplementedError

    # 写入原始数据
    # raw_data: {field: DataFrame}
    def writeRawData(self, key, raw_data, id_col="ID", if_exists="append", pid_ids=None, meta:dict={}):
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
    def writeFactorData(self, key, factor_data, pid_ids, pid=None, target_field="StdData", if_exists="append", data_type=None, meta:dict={}):
        raise NotImplementedError

    # 读取因子数据
    def readFactorData(self, key, ipid, target_field="StdData", pids=None, wait=True, wait_seconds=0.1, data_type=None):
        raise NotImplementedError

    # 清空缓存
    def clearFactorData(self, key=None):
        if key:
            self._CachedDTRange.pop(key)
        else:
            self._CachedDTRange = {}


class FileFactorCache(FileDTCache, FactorCache):
    class __QS_ArgClass__(FileDTCache.__QS_ArgClass__, FactorCache.__QS_ArgClass__):
        pass
    
    def __init__(self, args={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._RawDataDir = None  # 原始数据存放根目录
        self._FactorDataDir = None  # 因子数据存放根目录
        self._PIDLock = {}  # 访问该缓存的锁, 防止并发访问冲突
    
    def createPath(self, path: str):
        os.makedirs(path, exist_ok=True)
    
    def getPathMTime(self, path: str):
        return max(os.path.getmtime(os.path.join(path, ifile)) for ifile in ["."]+os.listdir(path))

    def start(self):
        if self._isStarted: return
        CacheDir = self._QSArgs.CacheDir
        if (not CacheDir) or (not os.path.isdir(CacheDir)):
            if CacheDir: self._QS_Logger.warning(f"缓存目录 '{CacheDir}' 不存在, 将使用系统的临时文件夹")
            self._CacheDirObj = tempfile.TemporaryDirectory()
            self._CacheDir = self._CacheDirObj.name
        else:
            self._CacheDir = str(CacheDir)
        self._DataDir = self._CacheDir + os.sep + "Data"# 通用数据存放根目录
        self._DTDataDir = self._CacheDir + os.sep + "DTData"# 时点数据存放根目录
        self._RawDataDir = self._CacheDir + os.sep + "RawData"  # 原始数据存放根目录
        self._FactorDataDir = self._CacheDir + os.sep + "FactorData"  # 因子数据存放根目录
        LockFile = self._CacheDir + os.sep + "LockFile"
        if not os.path.isfile(LockFile):
            open(LockFile, mode="a").close()
            os.chmod(LockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
        self._DataLock = QSFileLock(LockFile, proc_lock=Lock())
        if self._QSArgs.StartMode == "new":
            self.clearData()
            self.clearDTData()
            self.clearRawData()
            self.clearFactorData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        with self._DataLock:
            if not os.path.isdir(self._DataDir): os.mkdir(self._DataDir)
            if not os.path.isdir(self._DTDataDir): os.mkdir(self._DTDataDir)
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

    def checkRawDataExistence(self, key, pids=None, if_not_exists="create"):
        if pids is None: pids = self._QSArgs.PIDs
        IfExist = False
        with self._DataLock:
            for iPID in pids:
                iPath = self._RawDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                if os.path.exists(iPath):
                    IfExist = True
                    if if_not_exists != "create":
                        break
                elif if_not_exists == "create":
                    self.createPath(iPath)
        return IfExist

    def writeRawData(self, key, raw_data, pid_ids, id_col="QS_ID", if_exists="append", meta:dict={}):
        if raw_data is None: return 0
        for iField, iRawData in raw_data.items():
            if isinstance(iRawData, pd.DataFrame) and (id_col in iRawData):  # 如果原始数据有 ID 列，按照 ID 列划分后存入子进程的原始文件中
                iRawData = iRawData.set_index([id_col])
                AllIDs = set(iRawData.index)
                for jPID, jIDs in pid_ids.items():
                    jInterIDs = sorted(AllIDs.intersection(jIDs))
                    ijRawData = iRawData.loc[jInterIDs]
                    jPath = self._RawDataDir + os.sep + jPID + os.sep + key + os.sep + iField + self._QSArgs.Suffix
                    with self._DataLock:
                        self.writeDataFrame(path=jPath, data=ijRawData.reset_index(), if_exists=if_exists, ignore_index=True)
                        if meta: self.writeMeta(path=self._RawDataDir + os.sep + jPID + os.sep + key + os.sep + "meta.json", meta=meta)
            else:  # 如果原始数据没有 ID 列，则将所有数据分别存入子进程的原始文件中
                for jPID, jIDs in pid_ids.items():
                    jPath = self._RawDataDir + os.sep + jPID + os.sep + key + os.sep + "RawData" + self._QSArgs.Suffix
                    with self._DataLock:
                        self.writeDataFrame(path=jPath, data=iRawData, if_exists=if_exists, ignore_index=True)
                        if meta: self.writeMeta(path=self._RawDataDir + os.sep + jPID + os.sep + key + os.sep + "meta.json", meta=meta)

    def readRawData(self, key, target_fields=None, pids=None):
        if pids is None: pids = self._QSArgs.PIDs
        RawData = {}
        for iPID in pids:
            iRawDataPath = self._RawDataDir + os.sep + iPID + os.sep + key
            with self._PIDLock[iPID]:
                if not os.path.isdir(iRawDataPath): continue
                if target_fields is None: target_fields = [iFile[:-len(self._QSArgs.Suffix)] for iFile in os.listdir(iRawDataPath)]
                for jField in target_fields:
                    jPath = os.path.join(iRawDataPath, jField + self._QSArgs.Suffix)
                    if os.path.isfile(jPath):
                        jVal = self.readDataFrame(path=jPath)
                        RawData.setdefault(jField, []).append(jVal)
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
                    iRawDataPath = self._RawDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                    try:
                        if os.path.isfile(iRawDataPath): os.remove(iRawDataPath)
                        elif os.path.isdir(iRawDataPath): shutil.rmtree(iRawDataPath)
                    except Exception as e:
                        self._QS_Logger.error(f"原始数据缓存: {iRawDataPath} 清理失败: {e}")
        return super().clearRawData(key=key)

    def checkFactorDataExistence(self, key, pids=None):
        if pids is None: pids = self._QSArgs.PIDs
        IfExist = False
        with self._DataLock:
            for iPID in pids:
                iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                IfExist = os.path.exists(iPath) or IfExist
        return IfExist

    def writeFactorData(self, key, factor_data, pid_ids, pid=None, target_field="StdData", if_exists="append", data_type=None, meta:dict={}):
        PIDs = (self._QSArgs.PIDs if pid is None else [pid])
        for iPID in PIDs:
            with self._PIDLock[iPID]:
                iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + os.sep + target_field + self._QSArgs.Suffix
                if pid_ids is not None:
                    iIDs = pid_ids.get(iPID)
                    self.writeDataFrame(path=iPath, data=factor_data.reindex(columns=iIDs), if_exists=if_exists, ignore_index=False, data_type=data_type)
                else:
                    self.writeDataFrame(path=iPath, data=factor_data, if_exists=if_exists, ignore_index=False, data_type=data_type)
                if meta: self.writeMeta(path=self._FactorDataDir + os.sep + iPID + os.sep + key + os.sep + "meta.json", meta=meta)

    def readFactorData(self, key, ipid, target_field="StdData", pids=None, wait=True, wait_seconds=0.1, data_type=None):
        if isinstance(pids, str):
            Path = self._FactorDataDir + os.sep + pids + os.sep + key
            if not os.path.exists(Path):
                return None
            with self._PIDLock[pids]:
                return self.readDataFrame(path=os.path.join(Path, target_field+self._QSArgs.Suffix), data_type=data_type)
        iPath = self._FactorDataDir + os.sep + ipid + os.sep + key
        with self._PIDLock[ipid]:
            DTNum = self.readDataFrame(path=os.path.join(iPath, target_field + self._QSArgs.Suffix), data_type=data_type)
            if DTNum is None: DTNum = 0
            else: DTNum = DTNum.shape[0]
        if pids is None:
            pids = {ipid}
        else:
            pids = set(pids)
        StdData = []
        MTime = {}
        while len(pids) > 0:
            iPID = pids.pop()
            iPath = self._FactorDataDir + os.sep + iPID + os.sep + key
            if not os.path.exists(iPath):  # 该进程的数据没有准备好
                if wait:
                    pids.add(iPID)
                    if wait_seconds > 0: time.sleep(wait_seconds)
                continue
            elif wait:
                iMTime = self.getPathMTime(iPath)
                if (iPID not in MTime) or (iMTime > MTime[iPID]):
                    MTime[iPID] = iMTime
                    iDTNum = self.readDataFrame(path=os.path.join(iPath, target_field + self._QSArgs.Suffix), data_type=data_type)
                    if iDTNum is None: iDTNum = 0
                    else: iDTNum = iDTNum.shape[0]
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

    def clearFactorData(self, key=None):
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


class FeatherFactorCache(FileFactorCache, FeatherDTCache):
    class __QS_ArgClass__(FileFactorCache.__QS_ArgClass__, FeatherDTCache.__QS_ArgClass__):
        Suffix: str = Field(default=".feather", title="后缀", frozen=True)
    
    def __init__(self, args={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=(__QS_ConfigPath__ + os.sep + "FeatherFactorCacheConfig.json" if config_file is None else config_file), **kwargs)

if __name__ == "__main__":
    IDs = [str(i).zfill(6) for i in range(3)]
    DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(100)]
    
    Cache = FeatherFactorCache(args={"Suffix": ".feather", "CacheDir": r"C:\Users\hst\Project\Data\FactorCache", "DTRuler": DTRuler})
    Cache.start()
    
    Cache.writeFactorData(key="aha", factor_data=pd.DataFrame(np.random.rand(5, 3), index=DTRuler[:5], columns=IDs), pid_ids=None, data_type="object")
    Data = Cache.readFactorData(key="aha", ipid="0", data_type="object")
    print(Data)
    
    print("===")