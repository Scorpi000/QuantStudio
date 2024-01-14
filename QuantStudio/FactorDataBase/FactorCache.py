# coding=utf-8
"""因子缓存"""
import os
import stat
import shutil
import tempfile
from multiprocessing import Lock

import numpy as np
import pandas as pd
from traits.api import Directory, Str, Dict

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.Tools.QSObjects import QSFileLock

class FactorCache(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        CacheDir = Directory(arg_type="Directory", label="缓存文件夹", order=0)
        PIDIDs = Dict(arg_type="Dict", label="进程ID", order=1)# {PID: [ID]}
        HDF5Suffix = Str(".h5", arg_type="String", label="H5文件后缀", order=2)

    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        self._CacheDir = None# 缓存主目录
        self._RawDataDir = None# 原始数据存放根目录
        self._CacheDataDir = None# 因子数据存放根目录
        self._PIDLock = {}# 访问该缓存的锁, 防止并发访问冲突
        self._init()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if (self._CacheDir is not None) and (not isinstance(self._CacheDir, str)):
            state["_CacheDir"] = self._CacheDir.name
        return state

    def _init(self):
        CacheDir = self._QSArgs.CacheDir
        if not os.path.isdir(CacheDir):
            if CacheDir: self._QS_Logger.warning(f"缓存文件夹 '{CacheDir}' 不存在, 将使用系统的临时文件夹")
            self._CacheDir = tempfile.TemporaryDirectory()
            self._RawDataDir = self._CacheDir.name + os.sep + "RawData"  # 原始数据存放根目录
            self._CacheDataDir = self._CacheDir.name + os.sep + "CacheData"  # 中间数据存放根目录
            LockFile = self._CacheDir.name + os.sep + "LockFile"
        else:
            self._CacheDir = CacheDir
            self._RawDataDir = self._CacheDir + os.sep + "RawData"  # 原始数据存放根目录
            self._CacheDataDir = self._CacheDir + os.sep + "CacheData"  # 中间数据存放根目录
            LockFile = self._CacheDir + os.sep + "LockFile"
        if not os.path.isfile(LockFile):
            open(LockFile, mode="a").close()
            os.chmod(LockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
        self._DataLock = QSFileLock(LockFile, proc_lock=Lock())
        with self._DataLock:
            if not os.path.isdir(self._RawDataDir): os.mkdir(self._RawDataDir)
            if not os.path.isdir(self._CacheDataDir): os.mkdir(self._CacheDataDir)
            # 根据进程创建缓存子目录
            for iPID in self._QSArgs.PIDIDs.keys():
                if not os.path.isdir(self._RawDataDir + os.sep + iPID): os.mkdir(self._RawDataDir + os.sep + iPID)
                if not os.path.isdir(self._CacheDataDir + os.sep + iPID): os.mkdir(self._CacheDataDir + os.sep + iPID)
                iLockFile = self._CacheDataDir + os.sep + iPID + os.sep + "LockFile"
                if not os.path.isfile(iLockFile):
                    open(iLockFile, mode="a").close()
                    os.chmod(iLockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
                self._PIDLock[iPID] = QSFileLock(iLockFile, proc_lock=Lock())

    # 创建原始数据缓存文件, 返回原始数据缓存是否已经存在
    def createRawDataCache(self, file_name):
        IfExist = False
        with self._DataLock:
            for iPID, iIDs in self._QSArgs.PIDIDs.items():
                iFilePath = self._RawDataDir + os.sep + iPID + os.sep + file_name + self._QSArgs.HDF5Suffix
                if os.path.isfile(iFilePath):
                    IfExist = True
                    break
                else:
                    with pd.HDFStore(iFilePath) as ANN_ReportFile:
                        pass
        return IfExist

    # 获取原始数据缓存文件地址
    def getRawDataCachePath(self, file_name, pid):
        return self._RawDataDir + os.sep + pid + os.sep + file_name + self._QSArgs.HDF5Suffix

    # 写入原始数据
    def writeRawData(self, file_name, raw_data, target_fields, additional_data={}):
        if raw_data is None: return 0
        if isinstance(raw_data, pd.DataFrame) and ("ID" in raw_data):# 如果原始数据有 ID 列，按照 ID 列划分后存入子进程的原始文件中
            raw_data = raw_data.set_index(["ID"])
            CommonCols = raw_data.columns.difference(target_fields).tolist()
            AllIDs = set(raw_data.index)
            for iPID, iIDs in self._QSArgs.PIDIDs.items():
                iInterIDs = sorted(AllIDs.intersection(iIDs))
                iData = raw_data.loc[iInterIDs]
                with self._PIDLock[iPID]:
                    with pd.HDFStore(self._RawDataDir+os.sep+iPID+os.sep+file_name+self._QSArgs.HDF5Suffix, mode="a") as iFile:
                        if target_fields:
                            for jFactorName in target_fields: iFile[jFactorName] = iData[CommonCols+[jFactorName]].reset_index()
                        else:
                            iFile["RawData"] = iData[CommonCols].reset_index()
                        iFile["_QS_IDs"] = pd.Series(iIDs)
                        if iPID in additional_data: iFile["_QS_AdditionalData"] = additional_data[iPID]
        else:# 如果原始数据没有 ID 列，则将所有数据分别存入子进程的原始文件中
            for iPID, iIDs in self._QSArgs.PIDIDs.items():
                with self._PIDLock[iPID]:
                    with pd.HDFStore(self._RawDataDir+os.sep+iPID+os.sep+file_name+self._QSArgs.HDF5Suffix, mode="a") as iFile:
                        iFile["RawData"] = raw_data
                        iFile["_QS_IDs"] = pd.Series(iIDs)
                        if iPID in additional_data: iFile["_QS_AdditionalData"] = additional_data[iPID]

    # 读取原始数据
    def readRawData(self, file_name, pid, target_field):
        RawDataFilePath = self._QSArgs._RawDataDir + os.sep + pid + os.sep + file_name + self._QSArgs.HDF5Suffix
        with self._PIDLock[pid]:
            if os.path.isfile(RawDataFilePath):
                with pd.HDFStore(RawDataFilePath, mode="r") as File:
                    PrepareIDs = File["_QS_IDs"].to_list()
                    if target_field in File: RawData = File[target_field]
                    elif "RawData" in File: RawData = File["RawData"]
                    else: RawData = None
                    if (RawData is not None) and ("_QS_AdditionalData" in File):
                        RawData._QS_AdditionalData = File["_QS_AdditionalData"]
            else:
                PrepareIDs, RawData = None, None
        return RawData, PrepareIDs

    # 写入因子数据
    def writeFactorData(self, file_name, factor_data, pid=None, pid_ids=None):
        if pid:
            if pid_ids is not None: iIDs = pid_ids[pid]
            else: iIDs = self._QSArgs.PIDIDs[pid]
            with self._PIDLock[pid]:
                with pd.HDFStore(self._CacheDataDir + os.sep + pid + os.sep + file_name + self._QSArgs.HDF5Suffix) as CacheFile:
                    if "StdData" in CacheFile:
                        CacheFile["StdData"] = pd.concat([CacheFile["StdData"], factor_data.reindex(columns=iIDs)]).sort_index()
                    else:
                        CacheFile["StdData"] = factor_data
                    CacheFile["_QS_IDs"] = pd.Series(iIDs)
        else:
            if pid_ids is None: pid_ids = self._QSArgs.PIDIDs
            for iPID, iIDs in pid_ids.items():
                with self._PIDLock[iPID]:
                    with pd.HDFStore(self._CacheDataDir + os.sep + iPID + os.sep + file_name + self._QSArgs.HDF5Suffix) as CacheFile:
                        if "StdData" in CacheFile:
                            CacheFile["StdData"] = pd.concat([CacheFile["StdData"], factor_data.reindex(columns=iIDs)]).sort_index()
                        else:
                            CacheFile["StdData"] = factor_data.reindex(columns=iIDs)
                        CacheFile["_QS_IDs"] = pd.Series(iIDs)

    # 读取因子数据
    def readFactorData(self, file_name, pids=None):
        if isinstance(pids, str):
            FilePath = self._CacheDataDir + os.sep + pids + os.sep + file_name + self._QSArgs.HDF5Suffix
            if not os.path.isfile(FilePath):
                return None
            with self._PIDLock[pids]:
                with pd.HDFStore(FilePath, mode="r") as CacheFile:
                    FactorData = CacheFile["StdData"]
            return FactorData
        if pids is None:
            pids = set(self._QSArgs.PIDIDs)
        else:
            pids = set(pids)
        StdData = []
        while len(pids)>0:
            iPID = pids.pop()
            iFilePath = self._CacheDataDir + os.sep + pids + os.sep + file_name + self._QSArgs.HDF5Suffix
            if not os.path.isfile(iFilePath):# 该进程的数据没有准备好
                pids.add(iPID)
                continue
            iStdData = self.readFactorData(file_name, pids=iPID)
            if iStdData is not None: StdData.append(iStdData)
        StdData = pd.concat(StdData, axis=1, join='inner', ignore_index=False)
        return StdData

    # 清空缓存
    def clear(self):
        with self._DataLock:
            try:
                shutil.rmtree(self._CacheDataDir)
                shutil.rmtree(self._RawDataDir)
            except Exception as e:
                self._QS_Logger.warning(f"缓存文件: ({self._RawDataDir}, {self._CacheDataDir}) 清理失败: {e}")
            else:
                os.mkdir(self._RawDataDir)
                os.mkdir(self._CacheDataDir)