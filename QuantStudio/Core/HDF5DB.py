# -*- coding: utf-8 -*-
"""基于 HDF5 文件的因子库"""
import os
import stat
import shutil
import pickle
import time
import datetime as dt
from multiprocessing import Lock
from typing import Optional, Literal

import numpy as np
import pandas as pd
import fasteners
import h5py
from pydantic import Field, DirectoryPath

from QuantStudio.Core import __QS_Error__, __QS_ConfigPath__
from QuantStudio.Core.FactorDB import WritableFactorDB
from QuantStudio.Core.Factor import CompoundFactor, Factor
from QuantStudio.Core.utils import adjustDataDTID
from QuantStudio.Core.QSObject import QSFileLock, Panel
from QuantStudio.Tools.FileFun import listDirFile
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5


def _identifyDataType(factor_data, data_type=None):
    if (data_type is None) or (data_type == "double"):
        try:
            factor_data = factor_data.astype(float)
        except:
            data_type = "object"
        else:
            data_type = "double"
    return (factor_data, data_type)


def _adjustData(data, data_type, order="C"):
    if data_type == "string":
        if h5py.version.version < "3.0.0":
            return data.where(pd.notnull(data), None).values
        else:
            return data.where(pd.notnull(data), "").values
    elif data_type == "double":
        return data.astype("float").values
    elif data_type == "object":
        if order == "C":
            return np.ascontiguousarray(data.applymap(lambda x: np.frombuffer(pickle.dumps(x), dtype=np.uint8)).values)
        elif order == "F":
            return np.asfortranarray(data.applymap(lambda x: np.frombuffer(pickle.dumps(x), dtype=np.uint8)).values)
        else:
            raise __QS_Error__("不支持的参数 order 值: %s" % order)
    else:
        raise __QS_Error__("不支持的数据类型: %s" % data_type)


class HDF5Factor(Factor):
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        CompoundFactor: str = Field(title="所属复合因子", frozen=True)
        LookBack: int | Literal[np.inf] = Field(default=0, title="回溯天数", frozen=True)
        OnlyStartLookBack: bool = Field(default=False, title="只起始日回溯", frozen=True)
        OnlyLookBackNontarget: bool = Field(default=False, title="只回溯非目标日", frozen=True)
        OnlyLookBackDT: bool = Field(default=False, title="只回溯时点", frozen=True)
        TargetDT: Optional[dt.datetime] = Field(default=None, title="目标时点", frozen=True)
    
    def __init__(self, fdb, args={}, **kwargs):
        self._FactorDB = fdb
        self._Suffix = fdb._Suffix  # 文件后缀名
        return super().__init__(descriptors=[], args=args, **kwargs)
    
    def new(self, args={}):
        args = self._QSArgs.model_dump() | args
        return self.__class__(fdb=self._FactorDB, args=args, logger=self._QS_Logger)
    
    def getMetaData(self, key=None):
        with self._FactorDB._getLock(self._QSArgs.CompoundFactor) as DataLock:
            with self._FactorDB._openHDF5File(self._FactorDB._QSArgs.MainDir / self._QSArgs.CompoundFactor / (self._QSArgs.Name + "." + self._Suffix), mode="r") as File:
                if key is None:
                    return dict(File.attrs)
                elif key in File.attrs:
                    return File.attrs[key]
                else:
                    return None

    def getID(self, idt=None, **kwargs):
        with self._FactorDB._getLock(self._QSArgs.CompoundFactor) as DataLock:
            with self._FactorDB._openHDF5File( self._FactorDB._QSArgs.MainDir / self._QSArgs.CompoundFactor / f"{self._QSArgs.Name}.{self._Suffix}", mode="r") as ijFile:
                if h5py.version.version >= "3.0.0":
                    IDs = ijFile["ID"].asstr(encoding="utf-8")[...]
                else:
                    IDs = ijFile["ID"][...]
        IDs = sorted(IDs)
        if idt is not None:
            Data = self.readData(ids=IDs, dts=[idt]).iloc[0]
            return Data[pd.notnull(Data)].index.tolist()
        else:
            return IDs
    
    def getDateTime(self, iid=None, start_dt=None, end_dt=None, **kwargs):
        with self._FactorDB._getLock(self._QSArgs.CompoundFactor) as DataLock:
            with self._FactorDB._openHDF5File(self._FactorDB._QSArgs.MainDir / self._QSArgs.CompoundFactor / (self._QSArgs.Name + "." + self._Suffix), mode="r") as ijFile:
                Timestamps = ijFile["DateTime"][...]
        if start_dt is not None:
            if isinstance(start_dt, pd.Timestamp) and (pd.__version__ >= "0.20.0"):
                start_dt = start_dt.to_pydatetime().timestamp()
            else:
                start_dt = start_dt.timestamp()
            Timestamps = Timestamps[Timestamps >= start_dt]
        if end_dt is not None:
            if isinstance(end_dt, pd.Timestamp) and (pd.__version__ >= "0.20.0"):
                end_dt = end_dt.to_pydatetime().timestamp()
            else:
                end_dt = end_dt.timestamp()
            Timestamps = Timestamps[Timestamps <= end_dt]
        DTs = sorted(dt.datetime.fromtimestamp(iTimestamp) for iTimestamp in Timestamps)
        if iid is not None:
            Data = self.readData(ids=[iid], dts=DTs).iloc[:, 0]
            return Data[pd.notnull(Data)].index.tolist()
        else:
            return DTs    

    def _readData(self, ids, dts):
        FilePath = self._FactorDB._QSArgs.MainDir / self._QSArgs.CompoundFactor / (self._QSArgs.Name + "." + self._Suffix)
        if not os.path.isfile(FilePath): raise __QS_Error__("因子库 '%s' 的复合因子 '%s' 中不存在因子 '%s'!" % (self._FactorDB.Name, self._QSArgs._CompoundFactor, self._QSArgs.Name))
        with self._FactorDB._getLock(self._QSArgs.CompoundFactor) as DataLock:
            with self._FactorDB._openHDF5File(FilePath, mode="r") as DataFile:
                DataType = DataFile.attrs["DataType"]
                DateTimes = DataFile["DateTime"][...]
                if h5py.version.version >= "3.0.0":
                    IDs = DataFile["ID"].asstr(encoding="utf-8")[...]
                else:
                    IDs = DataFile["ID"][...]
                if dts is None:
                    if ids is None:
                        if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                            Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[...], index=DateTimes, columns=IDs).sort_index(axis=1)
                        else:
                            Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs).sort_index(axis=1)
                    elif set(ids).isdisjoint(IDs):
                        Rslt = pd.DataFrame(index=DateTimes, columns=ids)
                    else:
                        if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                            Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[...], index=DateTimes, columns=IDs).reindex(columns=ids)
                        else:
                            Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs).reindex(columns=ids)
                    Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
                elif (ids is not None) and set(ids).isdisjoint(IDs):
                    Rslt = pd.DataFrame(index=dts, columns=ids)
                else:
                    if dts and isinstance(dts[0], pd.Timestamp) and (pd.__version__ >= "0.20.0"):
                        dts = [idt.to_pydatetime().timestamp() for idt in dts]
                    else:
                        dts = [idt.timestamp() for idt in dts]
                    DateTimes = pd.Series(np.arange(0, DateTimes.shape[0]), index=DateTimes, dtype=int)
                    DateTimes = DateTimes[DateTimes.index.intersection(dts)]
                    nDT = DateTimes.shape[0]
                    if nDT == 0:
                        if ids is None:
                            Rslt = pd.DataFrame(index=dts, columns=IDs).sort_index(axis=1)
                        else:
                            Rslt = pd.DataFrame(index=dts, columns=ids)
                    elif nDT < 1000:
                        DateTimes = DateTimes.sort_values()
                        Mask = DateTimes.tolist()
                        DateTimes = DateTimes.index.values
                        if ids is None:
                            if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                                Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[Mask, :], index=DateTimes, columns=IDs).reindex(index=dts).sort_index(axis=1)
                            else:
                                Rslt = pd.DataFrame(DataFile["Data"][Mask, :], index=DateTimes, columns=IDs).reindex(index=dts).sort_index(axis=1)
                        else:
                            IDRuler = pd.Series(np.arange(0, IDs.shape[0]), index=IDs)
                            IDRuler = IDRuler.reindex(index=ids)
                            StartInd, EndInd = int(IDRuler.min()), int(IDRuler.max())
                            if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                                Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[Mask, StartInd:EndInd + 1], index=DateTimes, columns=IDs[StartInd:EndInd + 1]).reindex(index=dts, columns=ids)
                            else:
                                Rslt = pd.DataFrame(DataFile["Data"][Mask, StartInd:EndInd + 1], index=DateTimes, columns=IDs[StartInd:EndInd + 1]).reindex(index=dts, columns=ids)
                    else:
                        if (h5py.version.version >= "3.0.0") and (DataType == "string"):
                            Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[...], index=DataFile["DateTime"][...], columns=IDs).reindex(index=dts)
                        else:
                            Rslt = pd.DataFrame(DataFile["Data"][...], index=DataFile["DateTime"][...], columns=IDs).reindex(index=dts)
                        if ids is not None:
                            Rslt = Rslt.reindex(columns=ids)
                        else:
                            Rslt.sort_index(axis=1, inplace=True)
                    Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
        if DataType == "string":
            Rslt = Rslt.where(pd.notnull(Rslt), None)
            Rslt = Rslt.where(Rslt != "", None)
        elif DataType == "object":
            Rslt = Rslt.applymap(
                lambda x: pickle.loads(bytes(x)) if isinstance(x, np.ndarray) and (x.shape[0] > 0) else None)
        return Rslt.sort_index(axis=0)

    def readData(self, ids, dts, **kwargs):
        if self._QSArgs.TargetDT:
            Data = self.new(args={"TargetDT": None}).readData(ids=ids, dts=[self._QSArgs.TargetDT])
            if dts is None: dts = self.getDateTime()
            if ids is None: ids = self.getID()
            return pd.DataFrame(Data.values.repeat(repeats=len(dts), axis=0), index=dts, columns=ids)
        LookBack = self._QSArgs.LookBack
        if LookBack == 0: return self._readData(ids, dts)
        if np.isinf(LookBack):
            RawData = self._readData(ids, None)
        else:
            if dts is not None:
                StartDT = dts[0] - dt.timedelta(LookBack)
                iDTs = self.getDateTime(start_dt=StartDT, end_dt=dts[-1])
            else:
                iDTs = None
            RawData = self._readData(ids, iDTs)
        if not self._QSArgs.OnlyLookBackDT:
            RawData = Panel({self._QSArgs.Name: RawData})
            return adjustDataDTID(RawData, LookBack, [self._QSArgs.Name], ids, dts, self._QSArgs.OnlyStartLookBack, self._QSArgs.OnlyLookBackNontarget, logger=self._QS_Logger).iloc[0]
        RawData = RawData.dropna(axis=0, how="all").dropna(axis=1, how="all")
        RowIdxMask = pd.isnull(RawData)
        if RowIdxMask.shape[1] == 0: return pd.DataFrame(index=dts, columns=ids)
        RawIDs = RowIdxMask.columns
        RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
        RowIdx[RowIdxMask] = np.nan
        RowIdx = adjustDataDTID(Panel({"RowIdx": RowIdx}), LookBack, ["RowIdx"], RawIDs.tolist(), dts, self._QSArgs.OnlyStartLookBack, self._QSArgs.OnlyLookBackNontarget, logger=self._QS_Logger).iloc[0].values
        RowIdx[pd.isnull(RowIdx)] = -1
        RowIdx = RowIdx.astype(int)
        ColIdx = np.arange(RowIdx.shape[1]).reshape((1, RowIdx.shape[1])).repeat(RowIdx.shape[0], axis=0)
        RowIdxMask = (RowIdx == -1)
        RawData = RawData.values[RowIdx, ColIdx]
        RawData[RowIdxMask] = None
        return pd.DataFrame(RawData, index=dts, columns=RawIDs).reindex(columns=ids)
    

class HDF5CompoundFactor(CompoundFactor):
    """HDF5DB 复合因子"""
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        LookBack: int | Literal[np.inf] = Field(default=0, title="回溯天数", frozen=True)
        OnlyStartLookBack: bool = Field(default=False, title="只起始日回溯", frozen=True)
        OnlyLookBackNontarget: bool = Field(default=False, title="只回溯非目标日", frozen=True)
        OnlyLookBackDT: bool = Field(default=False, title="只回溯时点", frozen=True)
        TargetDT: Optional[dt.datetime] = Field(default=None, title="目标时点", frozen=True)

    def __init__(self, fdb, args={}, **kwargs):
        self._FactorDB = fdb
        CompoundFactorName = args["Name"]
        FactorList = [HDF5Factor(fdb=fdb, args=args | {"CompoundFactor": CompoundFactorName, "Name": iFactorName}, **kwargs) for iFactorName in sorted(listDirFile(self._FactorDB._QSArgs.MainDir / CompoundFactorName, suffix=fdb._Suffix))]
        return super().__init__(descriptors=FactorList, args=args, **kwargs)

    def getMetaData(self, key=None):
        with self._FactorDB._getLock(self._QSArgs.Name) as DataLock:
            if not os.path.isfile(self._FactorDB._QSArgs.MainDir / self._QSArgs.Name / "_CompoundFactorInfo.h5"):
                return (pd.Series() if key is None else None)
            if key is None:
                return pd.Series(readNestedDictFromHDF5(self._FactorDB._QSArgs.MainDir / self._QSArgs.Name / "_CompoundFactorInfo.h5", "/"))
            else:
                return readNestedDictFromHDF5(self._FactorDB._QSArgs.MainDir / self.Name / "_CompoundFactorInfo.h5", f"/{key}")


# 基于 HDF5 文件的因子数据库
# 每一个复合因子是一个文件夹, 每个因子是一个 HDF5 文件
# 每个 HDF5 文件有三个 Dataset: DateTime, ID, Data;
# 复合因子的元数据存储在复合因子文件夹下特殊文件: _CompoundFactorInfo.h5 中
# 因子的元数据存储在 HDF5 文件的 attrs 中
class HDF5DB(WritableFactorDB):
    """HDF5DB"""

    class __QS_ArgClass__(WritableFactorDB.__QS_ArgClass__):
        Name: str = Field(default="HDF5DB", frozen=True, title="名称")
        MainDir: DirectoryPath = Field(title="主目录", frozen=True)
        LockDir: Optional[DirectoryPath] = Field(default=None, title="锁目录", frozen=True)
        FileOpenRetryNum: float = Field(default=np.inf, title="文件打开重试次数")
        ProcessLock: bool = Field(default=True, title="进程锁")

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        self._LockFile = None  # 文件锁的目标文件
        self._DataLock = None  # 访问该因子库资源的文件锁, 防止并发访问冲突
        self._TableLock = None  # 访问该因子表资源的临时文件锁, 防止并发访问冲突
        self._ProcLock = None  # 访问该因子库资源的进程锁, 防止并发访问冲突
        self._Suffix = "hdf5"  # 文件的后缀名
        return super().__init__(args=args, config_file=(__QS_ConfigPath__ + os.sep + "HDF5DBConfig.json" if config_file is None else config_file), **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_DataLock"] = (True if self._DataLock is not None else False)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._DataLock:
            self._DataLock = fasteners.InterProcessLock(self._LockFile)
        else:
            self._DataLock = None

    def connect(self):
        if not os.path.isdir(self._QSArgs.MainDir):
            raise __QS_Error__("HDF5DB.connect: 不存在主目录 '%s'!" % self._QSArgs.MainDir)
        if not self._QSArgs.LockDir:
            self._LockDir = self._QSArgs.MainDir
        elif not os.path.isdir(self._QSArgs.LockDir):
            raise __QS_Error__("HDF5DB.connect: 不存在锁目录 '%s'!" % self._QSArgs.LockDir)
        else:
            self._LockDir = self._QSArgs.LockDir
        self._LockFile = self._LockDir / "LockFile"
        if not self._LockFile.is_file():
            open(self._LockFile, mode="a").close()
            os.chmod(self._LockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
        self._DataLock = fasteners.InterProcessLock(self._LockFile)
        if self._QSArgs.ProcessLock: self._ProcLock = Lock()
        self._isAvailable = True
        return self

    def disconnect(self):
        self._LockFile = None
        self._DataLock = None
        self._ProcLock = None

    def _getLock(self, compound_factor_name=None):
        if compound_factor_name is None:
            return QSFileLock(self._DataLock, proc_lock=self._ProcLock)
        TablePath = self._QSArgs.MainDir / compound_factor_name
        if not os.path.isdir(TablePath):
            Msg = ("因子库 '%s' 调用 _getLock 时错误, 不存在因子: '%s'" % (self.Name, compound_factor_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        LockFile = self._LockDir / compound_factor_name / "LockFile"
        if not os.path.isfile(LockFile):
            with QSFileLock(self._DataLock, proc_lock=self._ProcLock) as FileLock:
                if not os.path.isdir(self._LockDir / compound_factor_name):
                    os.mkdir(self._LockDir / compound_factor_name)
                if not os.path.isfile(LockFile):
                    open(LockFile, mode="a").close()
                    os.chmod(LockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
        return QSFileLock(LockFile, self._ProcLock)

    def _openHDF5File(self, filename, *args, **kwargs):
        i = 0
        while i < self._QSArgs.FileOpenRetryNum:
            try:
                f = h5py.File(filename, *args, **kwargs)
            except OSError as e:
                i += 1
                SleepTime = 0.05 + (i % 100) / 100.0
                if i % 100 == 0:
                    self._QS_Logger.warning(
                        "Can't open hdf5 file: '%s'\n %s \n try again %s seconds later!" % (filename, str(e),
                                                                                            SleepTime))
                time.sleep(SleepTime)
            else:
                return f
        Msg = "无法打开 hdf5 文件: '%s' 已经尝试 %d 次" % (filename, i)
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)

    # -------------------------------表的操作---------------------------------
    @property
    def FactorNames(self):
        return sorted(iDir.name for iDir in self._QSArgs.MainDir.iterdir() if os.path.isdir(iDir))

    def getFactor(self, factor_name, args={}):
        if not os.path.isdir(self._QSArgs.MainDir / factor_name):
            raise __QS_Error__("HDF5DB.getFactor: 因子 '%s' 不存在!" % factor_name)
        return HDF5CompoundFactor(fdb=self, args={"Name": factor_name}, logger=self._QS_Logger)

    def renameFactor(self, old_factor_name, new_factor_name, compound_factor_name=None):
        if old_factor_name == new_factor_name: return 0
        if not compound_factor_name:
            OldPath = self._QSArgs.MainDir / old_factor_name
            NewPath = self._QSArgs.MainDir / new_factor_name
        else:
            OldPath = self._QSArgs.MainDir / compound_factor_name / f"{old_factor_name}.{self._Suffix}"
            NewPath = self._QSArgs.MainDir / compound_factor_name / f"{new_factor_name}.{self._Suffix}"
        with self._DataLock:
            if not OldPath.exists(): raise __QS_Error__(f"HDF5DB.renameFactor: 因子: '{compound_factor_name}.{old_factor_name}' 不存在!")
            if NewPath.exists(): raise __QS_Error__(f"HDF5DB.renameFactor: 因子 '{compound_factor_name}.{new_factor_name}' 已存在!")
            os.rename(OldPath, NewPath)
        return 0

    def deleteFactor(self, factor_name, compound_factor_name=None):
        if not compound_factor_name:
            FactorPath = self._QSArgs.MainDir / factor_name
            with self._DataLock:
                if os.path.isdir(FactorPath):
                    shutil.rmtree(FactorPath, ignore_errors=True)
        else:
            FactorPath = self._QSArgs.MainDir / compound_factor_name
            FactorNames = listDirFile(FactorPath, suffix=self._Suffix)
            with self._DataLock:
                if FactorNames == [factor_name]:
                    shutil.rmtree(FactorPath, ignore_errors=True)
                else:
                    FactorPath = FactorPath / f"{factor_name}.{self._Suffix}"
                    if os.path.isfile(FactorPath):
                        os.remove(FactorPath)
        return 0

    def setFactorMetaData(self, factor_name, compound_factor_name=None, key=None, value=None, meta_data=None):
        if not compound_factor_name:
            if meta_data is not None:
                meta_data = dict(meta_data)
            else:
                meta_data = {}
            if key is not None:
                meta_data[key] = value
            with self._DataLock:
                writeNestedDict2HDF5(meta_data, self._QSArgs.MainDir / factor_name / "_CompoundFactorInfo.h5", "/")
        else:
            with self._getLock(compound_factor_name=compound_factor_name) as DataLock:
                with self._openHDF5File(self._QSArgs.MainDir / compound_factor_name / f"{factor_name}.{self._Suffix}", mode="a") as File:
                    if key is not None:
                        if key in File.attrs:
                            del File.attrs[key]
                        if (isinstance(value, np.ndarray)) and (value.dtype == np.dtype("O")):
                            File.attrs.create(key, data=value, dtype=h5py.special_dtype(vlen=str))
                        elif value is not None:
                            File.attrs[key] = value
            if meta_data is not None:
                for iKey in meta_data:
                    self.setFactorMetaData(factor_name=factor_name, compound_factor_name=compound_factor_name, key=iKey, value=meta_data[iKey], meta_data=None)

    def _updateFactorData(self, factor_data, compound_factor_name, factor_name, data_type):
        FilePath = self._QSArgs.MainDir / compound_factor_name / f"{factor_name}.{self._Suffix}"
        with self._getLock(compound_factor_name=compound_factor_name) as DataLock:
            with self._openHDF5File(FilePath, mode="a") as DataFile:
                OldDataType = DataFile.attrs["DataType"]
                if data_type is None: data_type = OldDataType
                factor_data, data_type = _identifyDataType(factor_data, data_type)
                if OldDataType != data_type:
                    raise __QS_Error__("HDF5DB.writeFactorData: 复合因子 '%s' 中因子 '%s' 的新数据无法转换成已有数据的数据类型 '%s'!" % (compound_factor_name, factor_name, OldDataType))
                nOldDT, OldDateTimes = DataFile["DateTime"].shape[0], DataFile["DateTime"][...]
                NewDateTimes = factor_data.index.difference(OldDateTimes).values
                if h5py.version.version < "3.0.0":
                    OldIDs = DataFile["ID"][...]
                else:
                    OldIDs = DataFile["ID"].asstr(encoding="utf-8")[...]
                NewIDs = factor_data.columns.difference(OldIDs).values
                DataFile["DateTime"].resize((nOldDT + NewDateTimes.shape[0],))
                DataFile["DateTime"][nOldDT:] = NewDateTimes
                DataFile["ID"].resize((OldIDs.shape[0] + NewIDs.shape[0],))
                DataFile["ID"][OldIDs.shape[0]:] = NewIDs
                DataFile["Data"].resize((DataFile["DateTime"].shape[0], DataFile["ID"].shape[0]))
                if NewDateTimes.shape[0] > 0:
                    DataFile["Data"][nOldDT:, :] = _adjustData(factor_data.reindex(index=NewDateTimes, columns=np.r_[OldIDs, NewIDs]), data_type)
                CrossedDateTimes = factor_data.index.intersection(OldDateTimes).values
                if CrossedDateTimes.shape[0] == 0:
                    DataFile.flush()
                    return 0
                if len(CrossedDateTimes) == len(OldDateTimes):
                    if NewIDs.shape[0] > 0:
                        DataFile["Data"][:nOldDT, OldIDs.shape[0]:] = _adjustData(factor_data.reindex(index=OldDateTimes, columns=NewIDs), data_type)
                    CrossedIDs = factor_data.columns.intersection(OldIDs)
                    if CrossedIDs.shape[0] > 0:
                        OldIDs = OldIDs.tolist()
                        CrossedIDPos = [OldIDs.index(iID) for iID in CrossedIDs]
                        CrossedIDs = CrossedIDs[np.argsort(CrossedIDPos)]
                        CrossedIDPos.sort()
                        DataFile["Data"][:nOldDT, CrossedIDPos] = _adjustData(factor_data.reindex(index=OldDateTimes, columns=CrossedIDs), data_type)
                    DataFile.flush()
                    return 0
                Sorter = np.argsort(OldDateTimes)
                CrossedDateTimePos = Sorter[np.searchsorted(OldDateTimes, CrossedDateTimes, sorter=Sorter)]
                CrossedDateTimes = CrossedDateTimes[np.argsort(CrossedDateTimePos)]
                CrossedDateTimePos.sort()
                if NewIDs.shape[0] > 0:
                    DataFile["Data"][CrossedDateTimePos, OldIDs.shape[0]:] = _adjustData(factor_data.reindex(index=CrossedDateTimes, columns=NewIDs), data_type)
                CrossedIDs = factor_data.columns.intersection(OldIDs).values
                if CrossedIDs.shape[0] > 0:
                    Sorter = np.argsort(OldIDs)
                    CrossedIDPos = Sorter[np.searchsorted(OldIDs, CrossedIDs, sorter=Sorter)]
                    CrossedIDs = CrossedIDs[np.argsort(CrossedIDPos)]
                    CrossedIDPos.sort()
                    NewData = _adjustData(factor_data.reindex(index=CrossedDateTimes, columns=CrossedIDs), data_type,order="F")
                    CrossedIDSep = np.arange(CrossedIDPos.shape[0])[np.r_[True, np.diff(CrossedIDPos) > 1]]
                    for i, iSep in enumerate(CrossedIDSep):
                        if i < CrossedIDSep.shape[0] - 1:
                            iCrossedStartIdx, iCrossedEndIdx = iSep, CrossedIDSep[i + 1]
                            iStartIdx, iEndIdx = CrossedIDPos[iSep], CrossedIDPos[CrossedIDSep[i + 1] - 1] + 1
                        else:
                            iCrossedStartIdx, iCrossedEndIdx = iSep, CrossedIDPos.shape[0]
                            iStartIdx, iEndIdx = CrossedIDPos[iSep], CrossedIDPos[-1] + 1
                        if data_type == "object":
                            DataFile["Data"][CrossedDateTimePos, iStartIdx:iEndIdx] = np.ascontiguousarray(NewData[:, iCrossedStartIdx:iCrossedEndIdx])
                        else:
                            DataFile["Data"][CrossedDateTimePos, iStartIdx:iEndIdx] = NewData[:, iCrossedStartIdx:iCrossedEndIdx]
                DataFile.flush()
        return 0

    def writeFactorData(self, factor_data, compound_factor_name, factor_name, if_exists="update", data_type=None, **kwargs):
        DTs = factor_data.index
        if pd.__version__ >= "0.20.0":
            factor_data.index = [idt.to_pydatetime().timestamp() for idt in factor_data.index]
        else:
            factor_data.index = [idt.timestamp() for idt in factor_data.index]
        TablePath = self._QSArgs.MainDir / compound_factor_name
        FilePath = TablePath / f"{factor_name}.{self._Suffix}"
        if not os.path.isdir(TablePath):
            with self._DataLock:
                if not os.path.isdir(TablePath): os.mkdir(TablePath)
        with self._getLock(compound_factor_name=compound_factor_name) as DataLock:
            if not os.path.isfile(FilePath):
                factor_data, data_type = _identifyDataType(factor_data, data_type)
                NewData = _adjustData(factor_data, data_type)
                open(FilePath, mode="a").close()  # h5py 直接创建文件名包含中文的文件会报错.
                # StrDataType = h5py.special_dtype(vlen=str)
                StrDataType = h5py.string_dtype(encoding="utf-8")
                with self._openHDF5File(FilePath, mode="a") as DataFile:
                    DataFile.attrs["DataType"] = data_type
                    DataFile.create_dataset("ID", shape=(factor_data.shape[1],), maxshape=(None,), dtype=StrDataType, data=factor_data.columns)
                    DataFile.create_dataset("DateTime", shape=(factor_data.shape[0],), maxshape=(None,), data=factor_data.index)
                    if data_type == "double":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=float, fillvalue=np.nan, data=NewData)
                    elif data_type == "string":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=StrDataType, fillvalue=None, data=NewData)
                    elif data_type == "object":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=h5py.vlen_dtype(np.uint8), data=NewData)
                    DataFile.flush()
                factor_data.index = DTs
                return 0
        if if_exists == "update":
            self._updateFactorData(factor_data, compound_factor_name, factor_name, data_type)
        else:
            OldData = self.getFactor(compound_factor_name).readFactorData(ifactor_name=factor_name, ids=factor_data.columns.tolist(), dts=DTs.tolist())
            OldData.index = factor_data.index
            if if_exists == "append":
                factor_data = OldData.where(pd.notnull(OldData), factor_data)
            elif if_exists == "update_notnull":
                factor_data = factor_data.where(pd.notnull(factor_data), OldData)
            else:
                Msg = ("因子库 '%s' 调用方法 writeFactorData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
            self._updateFactorData(factor_data, compound_factor_name, factor_name, data_type)
        factor_data.index = DTs
        return 0

    def writeData(self, data, compound_factor_name=None, if_exists="update", data_type={}, **kwargs):
        for i, iFactor in enumerate(data.items):
            self.writeFactorData(data.iloc[i], compound_factor_name, iFactor, if_exists=if_exists, data_type=data_type.get(iFactor, None), **kwargs)
        return 0


if __name__ == "__main__":
    HDB = HDF5DB(args={"MainDir": r"C:\Users\hst\Project\Data\HDF5DB"}).connect()
    print(HDB.Args)
    print(HDB.FactorNames)

    df = pd.DataFrame([(None, "aha"), ("中文", "aaa")], index=[dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 2)], columns=["000001.SZ", "000002.SZ"], dtype="O")
    HDB.writeFactorData(df, compound_factor_name="test_cfactor", factor_name="factor1", data_type="string")

    CF = HDB.getFactor("test_cfactor")
    # CF = HDB["test_cfactor"]
    print(CF.Args)

    print(CF.FactorNames)

    IDs = CF.getID()
    print(IDs)
    DTs = CF.getDateTime()
    print(DTs)

    Data = CF.readData(ids=None, dts=None, factor_names=CF.FactorNames)
    print(Data)
    F = CF.getFactor("factor1")
    print(F)
    Data = F.readData(ids=None, dts=None)
    print(Data)

    Data = F[[dt.datetime(2022, 1, 1)]]
    print(Data)
    Data = F[:, "000001.SZ"]
    print(Data)
    Data = F[dt.datetime(2022, 1, 1), "000002.SZ"]
    print(Data)

    HDB.deleteFactor(factor_name="factor1", compound_factor_name="test_cfactor")

    print("===")