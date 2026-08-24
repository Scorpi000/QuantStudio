# -*- coding: utf-8 -*-
import os
import stat
import time
import shutil
import tempfile
import datetime as dt
from typing import Optional, List, Literal, Dict

import numpy as np
import pandas as pd
from filelock import FileLock
from pydantic import Field

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.Cache import DTCache
from QuantStudio.Core.FileCache import FileDTCache, FeatherDTCache


class FactorCache(DTCache):
    """因子数据缓存"""

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
            if not self.load():
                # 状态恢复失败，降级为 new 模式
                self.clearData()
                self.clearDTData()
                self.clearRawData()
                self.clearFactorData()
        self._isStarted = True

    def end(self, clear:bool=False):
        if not self._isStarted: return
        if clear:
            self.clearData()
            self.clearDTData()
            self.clearRawData()
            self.clearFactorData()
        else:
            self.dump()
        self._isStarted = False

    def checkRawDataExistence(self, key:str, pids:Optional[List[str]]=None, create_if_not_exists: bool=True) -> bool:
        """检查原始数据缓存是否存在

        Args:
            key: 数据项的键
            pids: 进程 ID 列表, None 表示所有进程
            create_if_not_exists: 如果不存在是否要创建该数据项
        
        Returns:
            是否存在该项数据
        """
        raise NotImplementedError

    def writeRawData(self, key:str, raw_data:Dict[str, pd.DataFrame], id_col:str="ID", if_exists:Literal["append", "replace"]="append", pid_ids:Optional[Dict[str, List[str]]]=None, meta:dict={}):
        """写入原始数据
        
        Args:
            key: 数据项的键
            raw_data: 待写入的原始数据, {field: DataFrame}
            id_col: ID 所在的列名, 如果 ID 列存在则按照 ID 切分后存入各个进程空间, 否则则将所有数据分别都存入各个进程空间
            if_exists: 如果该数据已经存在的更新方式, append 表示只添加新增的数据, replace 表示替换已有数据
            pid_ids: 进程及其分配的 ID 序列, {进程ID: [ID]}
            meta: 数据项的元信息
        """
        raise NotImplementedError

    def readRawData(self, key:str, target_fields:Optional[List[str]]=None, pids:Optional[List[str]]=None) -> Dict[str, pd.DataFrame]:
        """读取原始数据
        
        Args:
            key: 数据项的键
            target_fields: 待读取的字段列表, None 表示所有的字段
            pids: 待读取的进程 ID 列表
        
        Returns:
            数据项的值, {field: DataFrame}
        """
        raise NotImplementedError

    def clearRawData(self, key:Optional[str]=None):
        """清空原始数据

        Args:
            key: 数据项的键, None 表示清空所有缓存
        """
        pass

    def checkFactorDataExistence(self, key:str, pids:Optional[List[str]]=None) -> bool:
        """检查因子数据缓存是否存在

        Args:
            key: 数据项的键
            pids: 进程 ID 列表, None 表示所有进程
        
        Returns:
            是否存在该项数据
        """
        raise NotImplementedError

    def writeFactorData(self, key:str, factor_data:pd.DataFrame, pid_ids:Dict[str, List[str]], pid:Optional[str]=None, target_field:str="StdData", if_exists:Literal["append", "replace"]="append", data_type:Optional[Literal["double", "string", "object"]]=None, meta:dict={}):
        """写入因子数据
        
        Args:
            key: 数据项的键
            factor_data: 待写入的因子数据
            pid_ids: 进程及其分配的 ID 序列, {进程ID: [ID]}
            pid: 待写入的进程, None 表示所有进程
            target_field: 数据项所属的字段
            if_exists: 如果该数据已经存在的更新方式, append 表示只添加新增的数据, replace 表示替换已有数据
            data_type: 写入数据的数据类型, None 表示让系统自己判断
            meta: 数据项的元信息
        """
        raise NotImplementedError

    def readFactorData(self, key:str, ipid:str, target_field:str="StdData", pids:Optional[List[str]]=None, wait:bool=True, wait_seconds:float=0.1, data_type:Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        """读取因子数据
        
        Args:
            key: 数据项的键
            ipid: 当前进程 ID
            target_field: 数据项所属的字段
            pids: 待读取的进程 ID 序列, None 表示只读取当前进程的数据
            wait: 如果某个进程的数据尚未就绪, 是否等待
            wait_seconds: 如果某个进程的数据尚未就绪, 反复检查的间隔秒数
            data_type: 数据的数据类型, None 表示让系统自己判断
        
        Returns:
            数据项的值, None 表示该项不存在
        """
        raise NotImplementedError

    def clearFactorData(self, key:Optional[str]=None):
        """清空因子缓存
        
        Args:
            key: 数据项的键, None 表示清空所有缓存
        """
        if key:
            self._CachedDTRange.pop(key)
        else:
            self._CachedDTRange = {}


class FileFactorCache(FileDTCache, FactorCache):
    """基于文件的因子缓存"""

    class __QS_ArgClass__(FileDTCache.__QS_ArgClass__, FactorCache.__QS_ArgClass__):
        pass
    
    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._RawDataDir = None  # 原始数据存放根目录
        self._FactorDataDir = None  # 因子数据存放根目录
    
    def createPath(self, path: str):
        os.makedirs(path, exist_ok=True)
    
    def getPathMTime(self, path: str):
        return max(os.path.getmtime(os.path.join(path, ifile)) for ifile in ["."]+os.listdir(path))

    def start(self):
        if self._isStarted: return
        CacheDir = self._QSArgs.CacheDir
        if (not CacheDir) or (not os.path.isdir(CacheDir)):
            self._QS_Logger.warning(f"缓存目录 {CacheDir} 没有指定或者不存在, 将使用系统的临时文件夹")
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
        if self._QSArgs.StartMode == "new":
            self.clearData()
            self.clearDTData()
            self.clearRawData()
            self.clearFactorData()
        elif self._QSArgs.StartMode == "continue":
            if not self.load():
                # 状态恢复失败，降级为 new 模式
                self.clearData()
                self.clearDTData()
                self.clearRawData()
                self.clearFactorData()
        if not os.path.isdir(self._DataDir): os.mkdir(self._DataDir)
        if not os.path.isdir(self._DTDataDir): os.mkdir(self._DTDataDir)
        if not os.path.isdir(self._RawDataDir): os.mkdir(self._RawDataDir)
        if not os.path.isdir(self._FactorDataDir): os.mkdir(self._FactorDataDir)
        # 根据进程创建缓存子目录
        for iPID in self._QSArgs.PIDs:
            if not os.path.isdir(self._RawDataDir + os.sep + iPID): os.mkdir(self._RawDataDir + os.sep + iPID)
            if not os.path.isdir(self._FactorDataDir + os.sep + iPID): os.mkdir(self._FactorDataDir + os.sep + iPID)
        self._isStarted = True

    def checkRawDataExistence(self, key, pids=None, create_if_not_exists: bool=True) -> bool:
        if pids is None: pids = self._QSArgs.PIDs
        IfExist = False
        with FileLock(self._RawDataDir + os.sep + key + ".lock") as DataLock:
            for iPID in pids:
                iPath = self._RawDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                if os.path.exists(iPath):
                    IfExist = True
                    if not create_if_not_exists:
                        break
                elif create_if_not_exists:
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
                    with FileLock(self._RawDataDir + os.sep + jPID + os.sep + key + ".lock") as DataLock:
                        self.writeDataFrame(path=jPath, data=ijRawData.reset_index(), if_exists=if_exists, ignore_index=True)
                        if meta: self.writeMeta(path=self._RawDataDir + os.sep + jPID + os.sep + key + os.sep + "meta.json", meta=meta)
            else:  # 如果原始数据没有 ID 列，则将所有数据分别存入子进程的原始文件中
                for jPID, jIDs in pid_ids.items():
                    jPath = self._RawDataDir + os.sep + jPID + os.sep + key + os.sep + "RawData" + self._QSArgs.Suffix
                    with FileLock(self._RawDataDir + os.sep + jPID + os.sep + key + ".lock") as DataLock:
                        self.writeDataFrame(path=jPath, data=iRawData, if_exists=if_exists, ignore_index=True)
                        if meta: self.writeMeta(path=self._RawDataDir + os.sep + jPID + os.sep + key + os.sep + "meta.json", meta=meta)

    def readRawData(self, key, target_fields=None, pids=None):
        if pids is None: pids = self._QSArgs.PIDs
        RawData = {}
        for iPID in pids:
            iRawDataPath = self._RawDataDir + os.sep + iPID + os.sep + key
            with FileLock(self._RawDataDir + os.sep + iPID + os.sep + key + ".lock") as DataLock:
                if not os.path.isdir(iRawDataPath): continue
                if target_fields is None: target_fields = [iFile[:-len(self._QSArgs.Suffix)] for iFile in os.listdir(iRawDataPath)]
                for jField in target_fields:
                    jPath = os.path.join(iRawDataPath, jField + self._QSArgs.Suffix)
                    if os.path.isfile(jPath):
                        jVal = self.readDataFrame(path=jPath)
                        RawData.setdefault(jField, []).append(jVal)
        RawData = {iKey: pd.concat(iVal, ignore_index=True) for iKey, iVal in RawData.items()}
        return RawData

    def clearRawData(self, key:Optional[str]=None):
        if key is None:
            with FileLock(self._CacheDir + os.sep + "_Cache.lock") as DataLock:
                try:
                    if os.path.isdir(self._RawDataDir): shutil.rmtree(self._RawDataDir)
                except Exception as e:
                    self._QS_Logger.error(f"原始数据缓存目录: {self._RawDataDir} 清理失败: {e}")
        else:
            for iPID in self._QSArgs.PIDs:
                iRawDataPath = self._RawDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                with FileLock(self._RawDataDir + os.sep + iPID + os.sep + key + ".lock") as DataLock:
                    try:
                        if os.path.isfile(iRawDataPath): os.remove(iRawDataPath)
                        elif os.path.isdir(iRawDataPath): shutil.rmtree(iRawDataPath)
                    except Exception as e:
                        self._QS_Logger.error(f"原始数据缓存: {iRawDataPath} 清理失败: {e}")
        return super().clearRawData(key=key)

    def checkFactorDataExistence(self, key:str, pids:Optional[List[str]]=None) -> bool:
        if pids is None: pids = self._QSArgs.PIDs
        IfExist = False
        with FileLock(self._FactorDataDir + os.sep + key + ".lock") as DataLock:
            for iPID in pids:
                iPath = self._FactorDataDir + os.sep + iPID + os.sep + key
                IfExist = os.path.exists(iPath) or IfExist
        return IfExist

    def writeFactorData(self, key:str, factor_data:pd.DataFrame, pid_ids:Dict[str, List[str]], pid:Optional[str]=None, target_field:str="StdData", if_exists:Literal["append", "replace"]="append", data_type:Optional[Literal["double", "string", "object"]]=None, meta:dict={}):
        PIDs = (self._QSArgs.PIDs if pid is None else [pid])
        for iPID in PIDs:
            with FileLock(self._FactorDataDir + os.sep + iPID + os.sep + key + ".lock") as DataLock:
                iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + os.sep + target_field + self._QSArgs.Suffix
                if pid_ids is not None:
                    iIDs = pid_ids.get(iPID)
                    self.writeDataFrame(path=iPath, data=factor_data.reindex(columns=iIDs), if_exists=if_exists, ignore_index=False, data_type=data_type)
                else:
                    self.writeDataFrame(path=iPath, data=factor_data, if_exists=if_exists, ignore_index=False, data_type=data_type)
                if meta: self.writeMeta(path=self._FactorDataDir + os.sep + iPID + os.sep + key + os.sep + "meta.json", meta=meta)

    def readFactorData(self, key:str, ipid:str, target_field:str="StdData", pids:Optional[List[str]]=None, wait:bool=True, wait_seconds:float=0.1, data_type:Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        if isinstance(pids, str):
            Path = self._FactorDataDir + os.sep + pids + os.sep + key
            if not os.path.exists(Path):
                return None
            with FileLock(self._FactorDataDir + os.sep + pids + os.sep + key + ".lock") as DataLock:
                return self.readDataFrame(path=os.path.join(Path, target_field+self._QSArgs.Suffix), data_type=data_type)
        iPath = self._FactorDataDir + os.sep + ipid + os.sep + key
        with FileLock(self._FactorDataDir + os.sep + ipid + os.sep + key + ".lock") as DataLock:
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
            if not os.path.exists(iPath):# 该进程的数据没有准备好
                if wait:
                    pids.add(iPID)
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                continue
            elif wait:
                DataLock = FileLock(self._FactorDataDir + os.sep + iPID + os.sep + key + ".lock")
                DataLock.acquire()
                iMTime = self.getPathMTime(iPath)
                if (iPID not in MTime) or (iMTime > MTime[iPID]):
                    MTime[iPID] = iMTime
                    iDTNum = self.readDataFrame(path=os.path.join(iPath, target_field + self._QSArgs.Suffix), data_type=data_type)
                    if iDTNum is None: iDTNum = 0
                    else: iDTNum = iDTNum.shape[0]
                    DataLock.release()
                    if iDTNum < DTNum:
                        pids.add(iPID)
                        if wait_seconds > 0: time.sleep(wait_seconds)
                        continue
                else:
                    DataLock.release()
                    pids.add(iPID)
                    if wait_seconds > 0: time.sleep(wait_seconds)
                    continue
            iStdData = self.readFactorData(key, ipid, target_field=target_field, pids=iPID, data_type=data_type)
            if iStdData is not None: StdData.append(iStdData)
        if StdData:
            return pd.concat(StdData, axis=1, join='outer', ignore_index=False)
        else:
            return None

    def clearFactorData(self, key:Optional[str]=None):
        if key is None:
            with FileLock(self._CacheDir + os.sep + "_Cache.lock") as DataLock:
                try:
                    if os.path.isdir(self._FactorDataDir): shutil.rmtree(self._FactorDataDir)
                except Exception as e:
                    self._QS_Logger.error(f"因子数据缓存目录: {self._FactorDataDir} 清理失败: {e}")
        else:
            for iPID in self._QSArgs.PIDs:
                iPath = self._FactorDataDir + os.sep + iPID + os.sep + key + self._QSArgs.Suffix
                with FileLock(self._FactorDataDir + os.sep + iPID + os.sep + key + ".lock") as DataLock:
                    try:
                        if os.path.isfile(iPath): os.remove(iPath)
                        elif os.path.isdir(iPath): shutil.rmtree(iPath)
                    except Exception as e:
                        self._QS_Logger.error(f"因子数据缓存: {iPath} 清理失败: {e}")
        return super().clearFactorData(key=key)


class FeatherFactorCache(FileFactorCache, FeatherDTCache):
    """基于 Feather 格式文件的因子缓存"""

    class __QS_ArgClass__(FileFactorCache.__QS_ArgClass__, FeatherDTCache.__QS_ArgClass__):
        Suffix: str = Field(default=".feather", title="后缀", frozen=True)
    
    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
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