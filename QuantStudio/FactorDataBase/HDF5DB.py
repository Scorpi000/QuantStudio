# coding=utf-8
"""基于 HDF5 文件的因子库"""
import os
import stat
import shutil
import pickle
import time
import datetime as dt
from multiprocessing import Lock

import numpy as np
import pandas as pd
import fasteners
import h5py
from traits.api import Directory, Enum, Float, Str, Date, Either, on_trait_change

from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.Tools.api import Panel
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import adjustDataDTID
from QuantStudio.Tools.FileFun import listDirDir, listDirFile
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5
from QuantStudio.Tools.QSObjects import QSFileLock


def _identifyDataType(factor_data, data_type=None):
    if (data_type is None) or (data_type=="double"):
        try:
            factor_data = factor_data.astype(float)
        except:
            data_type = "object"
        else:
            data_type = "double"
    return (factor_data, data_type)

def _adjustData(data, data_type, order="C"):
    if data_type=="string":
        if h5py.version.version<"3.0.0":
            return data.where(pd.notnull(data), None).values
        else:
            return data.where(pd.notnull(data), "").values
    elif data_type=="double": return data.astype("float").values
    elif data_type=="object":
        if order=="C":
            return np.ascontiguousarray(data.applymap(lambda x: np.frombuffer(pickle.dumps(x), dtype=np.uint8)).values)
        elif order=="F":
            return np.asfortranarray(data.applymap(lambda x: np.frombuffer(pickle.dumps(x), dtype=np.uint8)).values)
        else:
            raise __QS_Error__("不支持的参数 order 值: %s" % order)
    else: raise __QS_Error__("不支持的数据类型: %s" % data_type)

class _FactorTable(FactorTable):
    """HDF5DB 因子表"""
    class __QS_ArgClass__(FactorTable.__QS_ArgClass__):
        LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
        OnlyStartLookBack = Enum(False, True, label="只起始日回溯", arg_type="Bool", order=1)
        OnlyLookBackNontarget = Enum(False, True, label="只回溯非目标日", arg_type="Bool", order=2)
        OnlyLookBackDT = Enum(False, True, label="只回溯时点", arg_type="Bool", order=3)
        TargetDT = Either(None, Date, arg_type="DateTime", label="目标时点", order=4)
    
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._Suffix = fdb._Suffix# 文件后缀名
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        return sorted(listDirFile(self._FactorDB._QSArgs.MainDir+os.sep+self.Name, suffix=self._Suffix))
    def getMetaData(self, key=None, args={}):
        with self._FactorDB._getLock(self._Name) as DataLock:
            if not os.path.isfile(self._FactorDB._QSArgs.MainDir+os.sep+self.Name+os.sep+"_TableInfo.h5"):
                return (pd.Series() if key is None else None)
            if key is None:
                return pd.Series(readNestedDictFromHDF5(self._FactorDB._QSArgs.MainDir+os.sep+self.Name+os.sep+"_TableInfo.h5", "/"))
            else:
                return readNestedDictFromHDF5(self._FactorDB._QSArgs.MainDir+os.sep+self.Name+os.sep+"_TableInfo.h5", f"/{key}")
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        AllFactorNames = self.FactorNames
        if factor_names is None: factor_names = AllFactorNames
        elif set(factor_names).isdisjoint(AllFactorNames): return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        with self._FactorDB._getLock(self._Name) as DataLock:
            MetaData = {}
            for iFactorName in factor_names:
                if iFactorName in AllFactorNames:
                    with self._FactorDB._openHDF5File(self._FactorDB._QSArgs.MainDir+os.sep+self.Name+os.sep+iFactorName+"."+self._Suffix, mode="r") as File:
                        if key is None: MetaData[iFactorName] = pd.Series(dict(File.attrs))
                        elif key in File.attrs: MetaData[iFactorName] = File.attrs[key]
        if not MetaData: return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        if key is None: return pd.DataFrame(MetaData).T.reindex(index=factor_names)
        else: return pd.Series(MetaData).reindex(index=factor_names)
    def getID(self, ifactor_name=None, idt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        with self._FactorDB._getLock(self._Name) as DataLock:
            with self._FactorDB._openHDF5File(self._FactorDB._QSArgs.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix, mode="r") as ijFile:
                if h5py.version.version>="3.0.0": 
                    IDs = ijFile["ID"].asstr(encoding="utf-8")[...]
                else:
                    IDs = ijFile["ID"][...]
        IDs = sorted(IDs)
        if idt is not None:
            Data = self.readFactorData(ifactor_name, ids=IDs, dts=[idt]).iloc[0]
            return Data[pd.notnull(Data)].index.tolist()
        else:
            return IDs
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        with self._FactorDB._getLock(self._Name) as DataLock:
            with self._FactorDB._openHDF5File(self._FactorDB._QSArgs.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix, mode="r") as ijFile:
                Timestamps = ijFile["DateTime"][...]
        if start_dt is not None:
            if isinstance(start_dt, pd.Timestamp) and (pd.__version__>="0.20.0"): start_dt = start_dt.to_pydatetime().timestamp()
            else: start_dt = start_dt.timestamp()
            Timestamps = Timestamps[Timestamps>=start_dt]
        if end_dt is not None:
            if isinstance(end_dt, pd.Timestamp) and (pd.__version__>="0.20.0"): end_dt = end_dt.to_pydatetime().timestamp()
            else: end_dt = end_dt.timestamp()
            Timestamps = Timestamps[Timestamps<=end_dt]
        DTs = sorted(dt.datetime.fromtimestamp(iTimestamp) for iTimestamp in Timestamps)
        if iid is not None:
            Data = self.readFactorData(ifactor_name, ids=[iid], dts=DTs).iloc[:, 0]
            return Data[pd.notnull(Data)].index.tolist()
        else:
            return DTs            
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Data = {iFactor: self.readFactorData(ifactor_name=iFactor, ids=ids, dts=dts, args=args) for iFactor in factor_names}
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)
    def _readFactorData(self, ifactor_name, ids, dts, args={}):
        FilePath = self._FactorDB._QSArgs.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix
        if not os.path.isfile(FilePath): raise __QS_Error__("因子库 '%s' 的因子表 '%s' 中不存在因子 '%s'!" % (self._FactorDB.Name, self.Name, ifactor_name))
        with self._FactorDB._getLock(self._Name) as DataLock:
            with self._FactorDB._openHDF5File(FilePath, mode="r") as DataFile:
                DataType = DataFile.attrs["DataType"]
                DateTimes = DataFile["DateTime"][...]
                if h5py.version.version>="3.0.0":
                    IDs = DataFile["ID"].asstr(encoding="utf-8")[...]
                else:
                    IDs = DataFile["ID"][...]
                if dts is None:
                    if ids is None:
                        if (h5py.version.version>="3.0.0") and (DataType=="string"):
                            Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[...], index=DateTimes, columns=IDs).sort_index(axis=1)
                        else:
                            Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs).sort_index(axis=1)
                    elif set(ids).isdisjoint(IDs):
                        Rslt = pd.DataFrame(index=DateTimes, columns=ids)
                    else:
                        if (h5py.version.version>="3.0.0") and (DataType=="string"):
                            Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[...], index=DateTimes, columns=IDs).reindex(columns=ids)
                        else:
                            Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs).reindex(columns=ids)
                    Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
                elif (ids is not None) and set(ids).isdisjoint(IDs):
                    Rslt = pd.DataFrame(index=dts, columns=ids)
                else:
                    if dts and isinstance(dts[0], pd.Timestamp) and (pd.__version__>="0.20.0"): dts = [idt.to_pydatetime().timestamp() for idt in dts]
                    else: dts = [idt.timestamp() for idt in dts]
                    DateTimes = pd.Series(np.arange(0, DateTimes.shape[0]), index=DateTimes, dtype=int)
                    DateTimes = DateTimes[DateTimes.index.intersection(dts)]
                    nDT = DateTimes.shape[0]
                    if nDT==0:
                        if ids is None: Rslt = pd.DataFrame(index=dts, columns=IDs).sort_index(axis=1)
                        else: Rslt = pd.DataFrame(index=dts, columns=ids)
                    elif nDT<1000:
                        DateTimes = DateTimes.sort_values()
                        Mask = DateTimes.tolist()
                        DateTimes = DateTimes.index.values
                        if ids is None:
                            if (h5py.version.version>="3.0.0") and (DataType=="string"):
                                Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[Mask, :], index=DateTimes, columns=IDs).reindex(index=dts).sort_index(axis=1)
                            else:
                                Rslt = pd.DataFrame(DataFile["Data"][Mask, :], index=DateTimes, columns=IDs).reindex(index=dts).sort_index(axis=1)
                        else:
                            IDRuler = pd.Series(np.arange(0,IDs.shape[0]), index=IDs)
                            IDRuler = IDRuler.reindex(index=ids)
                            StartInd, EndInd = int(IDRuler.min()), int(IDRuler.max())
                            if (h5py.version.version>="3.0.0") and (DataType=="string"):
                                Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[Mask, StartInd:EndInd+1], index=DateTimes, columns=IDs[StartInd:EndInd+1]).reindex(index=dts, columns=ids)
                            else:
                                Rslt = pd.DataFrame(DataFile["Data"][Mask, StartInd:EndInd+1], index=DateTimes, columns=IDs[StartInd:EndInd+1]).reindex(index=dts, columns=ids)
                    else:
                        if (h5py.version.version>="3.0.0") and (DataType=="string"):
                            Rslt = pd.DataFrame(DataFile["Data"].asstr(encoding="utf-8")[...], index=DataFile["DateTime"][...], columns=IDs).reindex(index=dts)
                        else:
                            Rslt = pd.DataFrame(DataFile["Data"][...], index=DataFile["DateTime"][...], columns=IDs).reindex(index=dts)
                        if ids is not None: Rslt = Rslt.reindex(columns=ids)
                        else: Rslt.sort_index(axis=1, inplace=True)
                    Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
        if DataType=="string":
            Rslt = Rslt.where(pd.notnull(Rslt), None)
            Rslt = Rslt.where(Rslt!="", None)
        elif DataType=="object":
            Rslt = Rslt.applymap(lambda x: pickle.loads(bytes(x)) if isinstance(x, np.ndarray) and (x.shape[0]>0) else None)
        return Rslt.sort_index(axis=0)
    def readFactorData(self, ifactor_name, ids, dts, args={}):
        TargetDT = args.get("目标时点", self._QSArgs.TargetDT)
        if TargetDT:
            Args = args.copy()
            Args["目标时点"] = None
            Data = self.readFactorData(ifactor_name=ifactor_name, ids=ids, dts=[TargetDT], args=Args)
            if dts is None: dts = self.getDateTime(ifactor_name=ifactor_name)
            if ids is None: ids = self.getID(ifactor_name=ifactor_name)
            return pd.DataFrame(Data.values.repeat(repeats=len(dts), axis=0), index=dts, columns=ids)
        LookBack = args.get("回溯天数", self._QSArgs.LookBack)
        if LookBack==0: return self._readFactorData(ifactor_name, ids, dts, args=args)
        if np.isinf(LookBack):
            RawData = self._readFactorData(ifactor_name, ids, None, args=args)
        else:
            if dts is not None:
                StartDT = dts[0] - dt.timedelta(LookBack)
                iDTs = self.getDateTime(ifactor_name=ifactor_name, start_dt=StartDT, end_dt=dts[-1], args=args)
            else:
                iDTs = None
            RawData = self._readFactorData(ifactor_name, ids, iDTs, args=args)
        if not args.get("只回溯时点", self._QSArgs.OnlyLookBackDT):
            RawData = Panel({ifactor_name: RawData})
            return adjustDataDTID(RawData, LookBack, [ifactor_name], ids, dts, args.get("只起始日回溯", self._QSArgs.OnlyStartLookBack), args.get("只回溯非目标日", self._QSArgs.OnlyLookBackNontarget), logger=self._QS_Logger).iloc[0]
        RawData = RawData.dropna(axis=0, how="all").dropna(axis=1, how="all")
        RowIdxMask = pd.isnull(RawData)
        if RowIdxMask.shape[1]==0: return pd.DataFrame(index=dts, columns=ids)
        RawIDs = RowIdxMask.columns
        RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
        RowIdx[RowIdxMask] = np.nan
        RowIdx = adjustDataDTID(Panel({"RowIdx": RowIdx}), LookBack, ["RowIdx"], RawIDs.tolist(), dts, args.get("只起始日回溯", self._QSArgs.OnlyStartLookBack), args.get("只回溯非目标日", self._QSArgs.OnlyLookBackNontarget), logger=self._QS_Logger).iloc[0].values
        RowIdx[pd.isnull(RowIdx)] = -1
        RowIdx = RowIdx.astype(int)
        ColIdx = np.arange(RowIdx.shape[1]).reshape((1, RowIdx.shape[1])).repeat(RowIdx.shape[0], axis=0)
        RowIdxMask = (RowIdx==-1)
        RawData = RawData.values[RowIdx, ColIdx]
        RawData[RowIdxMask] = None
        return pd.DataFrame(RawData, index=dts, columns=RawIDs).reindex(columns=ids)

# 基于 HDF5 文件的因子数据库
# 每一张表是一个文件夹, 每个因子是一个 HDF5 文件
# 每个 HDF5 文件有三个 Dataset: DateTime, ID, Data;
# 表的元数据存储在表文件夹下特殊文件: _TableInfo.h5 中
# 因子的元数据存储在 HDF5 文件的 attrs 中
class HDF5DB(WritableFactorDB):
    """HDF5DB"""
    class __QS_ArgClass__(WritableFactorDB.__QS_ArgClass__):
        Name = Str("HDF5DB", arg_type="String", label="名称", order=-100)
        MainDir = Directory(label="主目录", arg_type="Directory", order=0)
        LockDir = Directory(label="锁目录", arg_type="Directory", order=1)
        FileOpenRetryNum = Float(np.inf, label="文件打开重试次数", arg_type="Float", order=2)
        ProcessLock = Enum(True, False, label="进程锁", arg_type="Bool", order=3)

        @on_trait_change("MainDir")
        def _on_MainDir_changed(self, obj, name, old, new):
            if self._Owner.isAvailable():
                self._Owner.connect()
        
        @on_trait_change("LockDir")
        def _on_LockDir_changed(self, obj, name, old, new):
            if self._Owner.isAvailable():
                self._Owner.connect()
        @on_trait_change("ProcessLock")
        def _on_ProcessLock_changed(self, obj, name, old, new):
            if self._Owner.isAvailable():
                self._Owner.connect()
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._LockFile = None# 文件锁的目标文件
        self._DataLock = None# 访问该因子库资源的文件锁, 防止并发访问冲突
        self._TableLock = None# 访问该因子表资源的临时文件锁, 防止并发访问冲突
        self._ProcLock = None# 访问该因子库资源的进程锁, 防止并发访问冲突
        self._isAvailable = False
        self._Suffix = "hdf5"# 文件的后缀名
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"HDF5DBConfig.json" if config_file is None else config_file), **kwargs)
        return
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
        self._LockFile = self._LockDir+os.sep+"LockFile"
        if not os.path.isfile(self._LockFile):
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
        self._isAvailable = False
    def isAvailable(self):
        return self._isAvailable
    def _getLock(self, table_name=None):
        if table_name is None:
            return QSFileLock(self._DataLock, proc_lock=self._ProcLock)
        TablePath = self._QSArgs.MainDir + os.sep + table_name
        if not os.path.isdir(TablePath):
            Msg = ("因子库 '%s' 调用 _getLock 时错误, 不存在因子表: '%s'" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        LockFile = self._LockDir + os.sep + table_name + os.sep + "LockFile"
        if not os.path.isfile(LockFile):
            with QSFileLock(self._DataLock, proc_lock=self._ProcLock) as FileLock:
                if not os.path.isdir(self._LockDir + os.sep + table_name):
                    os.mkdir(self._LockDir + os.sep + table_name)
                if not os.path.isfile(LockFile):
                    open(LockFile, mode="a").close()
                    os.chmod(LockFile, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
        return QSFileLock(LockFile, self._ProcLock)
    def _openHDF5File(self, filename, *args, **kwargs):
        i = 0
        while i<self._QSArgs.FileOpenRetryNum:
            try:
                f = h5py.File(filename, *args, **kwargs)
            except OSError as e:
                i += 1
                SleepTime = 0.05 + (i % 100) / 100.0
                if i % 100 == 0:
                    self._QS_Logger.warning("Can't open hdf5 file: '%s'\n %s \n try again %s seconds later!" % (filename, str(e), SleepTime))
                time.sleep(SleepTime)
            else:
                return f
        Msg = "Can't open hdf5 file: '%s' after trying %d times" % (filename, i)
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        return sorted(listDirDir(self._QSArgs.MainDir))
    def getTable(self, table_name, args={}):
        if not os.path.isdir(self._QSArgs.MainDir+os.sep+table_name): raise __QS_Error__("HDF5DB.getTable: 表 '%s' 不存在!" % table_name)
        return _FactorTable(name=table_name, fdb=self, sys_args=args, logger=self._QS_Logger)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name==new_table_name: return 0
        OldPath = self._QSArgs.MainDir+os.sep+old_table_name
        NewPath = self._QSArgs.MainDir+os.sep+new_table_name
        with self._DataLock:
            if not os.path.isdir(OldPath): raise __QS_Error__("HDF5DB.renameTable: 表: '%s' 不存在!" % old_table_name)
            if os.path.isdir(NewPath): raise __QS_Error__("HDF5DB.renameTable: 表 '"+new_table_name+"' 已存在!")
            os.rename(OldPath, NewPath)
        return 0
    def deleteTable(self, table_name):
        TablePath = self._QSArgs.MainDir+os.sep+table_name
        with self._DataLock:
            if os.path.isdir(TablePath):
                shutil.rmtree(TablePath, ignore_errors=True)
        return 0
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        if meta_data is not None:
            meta_data = dict(meta_data)
        else:
            meta_data = {}
        if key is not None:
            meta_data[key] = value
        with self._DataLock:
            writeNestedDict2HDF5(meta_data, self._QSArgs.MainDir+os.sep+table_name+os.sep+"_TableInfo.h5", "/")
        return 0
    def setFactorDef(self, table_name, def_file, if_exists="update"):
        if not os.path.isfile(def_file):
            Msg = f"因子文件: '{def_file}' 不存在"
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if if_exists=="append":
            MetaData = self.getTable(table_name).getMetaData(key="_QS_FactorDef")
            if MetaData.empty:
                MetaData = []
            else:
                MetaData = MetaData["_QS_FactorDef"]
        else:
            MetaData = []
        with open(def_file) as File:
            MetaData.append(File.read())
        self.setTableMetaData(table_name, key="_QS_FactorDef", value=MetaData)
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name==new_factor_name: return 0
        OldPath = self._QSArgs.MainDir+os.sep+table_name+os.sep+old_factor_name+"."+self._Suffix
        NewPath = self._QSArgs.MainDir+os.sep+table_name+os.sep+new_factor_name+"."+self._Suffix
        with self._DataLock:
            if not os.path.isfile(OldPath): raise __QS_Error__("HDF5DB.renameFactor: 表 '%s' 中不存在因子 '%s'!" % (table_name, old_factor_name))
            if os.path.isfile(NewPath): raise __QS_Error__("HDF5DB.renameFactor: 表 '%s' 中的因子 '%s' 已存在!" % (table_name, new_factor_name))
            os.rename(OldPath, NewPath)
        return 0
    def deleteFactor(self, table_name, factor_names):
        TablePath = self._QSArgs.MainDir+os.sep+table_name
        FactorNames = set(listDirFile(TablePath, suffix=self._Suffix))
        with self._DataLock:
            if FactorNames.issubset(set(factor_names)):
                shutil.rmtree(TablePath, ignore_errors=True)
            else:
                for iFactor in factor_names:
                    iFilePath = TablePath+os.sep+iFactor+"."+self._Suffix
                    if os.path.isfile(iFilePath):
                        os.remove(iFilePath)
        return 0
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        with self._getLock(table_name=table_name) as DataLock:
            with self._openHDF5File(self._QSArgs.MainDir+os.sep+table_name+os.sep+ifactor_name+"."+self._Suffix, mode="a") as File:
                if key is not None:
                    if key in File.attrs:
                        del File.attrs[key]
                    if (isinstance(value, np.ndarray)) and (value.dtype==np.dtype("O")):
                        File.attrs.create(key, data=value, dtype=h5py.special_dtype(vlen=str))
                    elif value is not None:
                        File.attrs[key] = value
        if meta_data is not None:
            for iKey in meta_data:
                self.setFactorMetaData(table_name, ifactor_name=ifactor_name, key=iKey, value=meta_data[iKey], meta_data=None)
        return 0
    def _updateFactorData(self, factor_data, table_name, ifactor_name, data_type):
        FilePath = self._QSArgs.MainDir + os.sep + table_name + os.sep + ifactor_name + "." + self._Suffix
        with self._getLock(table_name=table_name) as DataLock:
            with self._openHDF5File(FilePath, mode="a") as DataFile:
                OldDataType = DataFile.attrs["DataType"]
                if data_type is None: data_type = OldDataType
                factor_data, data_type = _identifyDataType(factor_data, data_type)
                if OldDataType != data_type:
                    raise __QS_Error__("HDF5DB.writeFactorData: 表 '%s' 中因子 '%s' 的新数据无法转换成已有数据的数据类型 '%s'!" % (table_name, ifactor_name, OldDataType))
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
                    NewData = _adjustData(factor_data.reindex(index=CrossedDateTimes, columns=CrossedIDs), data_type, order="F")
                    CrossedIDSep = np.arange(CrossedIDPos.shape[0])[np.r_[True, np.diff(CrossedIDPos) > 1]]
                    for i, iSep in enumerate(CrossedIDSep):
                        if i < CrossedIDSep.shape[0]-1:
                            iCrossedStartIdx, iCrossedEndIdx = iSep, CrossedIDSep[i + 1]
                            iStartIdx, iEndIdx = CrossedIDPos[iSep], CrossedIDPos[CrossedIDSep[i + 1] - 1] + 1
                        else:
                            iCrossedStartIdx, iCrossedEndIdx = iSep, CrossedIDPos.shape[0]
                            iStartIdx, iEndIdx = CrossedIDPos[iSep], CrossedIDPos[-1] + 1
                        if data_type=="object":
                            DataFile["Data"][CrossedDateTimePos, iStartIdx:iEndIdx] = np.ascontiguousarray(NewData[:, iCrossedStartIdx:iCrossedEndIdx])
                        else:
                            DataFile["Data"][CrossedDateTimePos, iStartIdx:iEndIdx] = NewData[:, iCrossedStartIdx:iCrossedEndIdx]
                DataFile.flush()
        return 0
    def writeFactorData(self, factor_data, table_name, ifactor_name, if_exists="update", data_type=None, **kwargs):
        DTs = factor_data.index
        if pd.__version__>="0.20.0": factor_data.index = [idt.to_pydatetime().timestamp() for idt in factor_data.index]
        else: factor_data.index = [idt.timestamp() for idt in factor_data.index]
        TablePath = self._QSArgs.MainDir+os.sep+table_name
        FilePath = TablePath+os.sep+ifactor_name+"."+self._Suffix
        if not os.path.isdir(TablePath):
            with self._DataLock:
                if not os.path.isdir(TablePath): os.mkdir(TablePath)
        with self._getLock(table_name=table_name) as DataLock:
            if not os.path.isfile(FilePath):
                factor_data, data_type = _identifyDataType(factor_data, data_type)
                NewData = _adjustData(factor_data, data_type)
                open(FilePath, mode="a").close()# h5py 直接创建文件名包含中文的文件会报错.
                #StrDataType = h5py.special_dtype(vlen=str)
                StrDataType = h5py.string_dtype(encoding="utf-8")
                with self._openHDF5File(FilePath, mode="a") as DataFile:
                    DataFile.attrs["DataType"] = data_type
                    DataFile.create_dataset("ID", shape=(factor_data.shape[1],), maxshape=(None,), dtype=StrDataType, data=factor_data.columns)
                    DataFile.create_dataset("DateTime", shape=(factor_data.shape[0],), maxshape=(None,), data=factor_data.index)
                    if data_type=="double":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=float, fillvalue=np.nan, data=NewData)
                    elif data_type=="string":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=StrDataType, fillvalue=None, data=NewData)
                    elif data_type=="object":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=h5py.vlen_dtype(np.uint8), data=NewData)
                    DataFile.flush()
                factor_data.index = DTs
                return 0
        if if_exists=="update":
            self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
        else:
            OldData = self.getTable(table_name).readFactorData(ifactor_name=ifactor_name, ids=factor_data.columns.tolist(), dts=DTs.tolist())
            OldData.index = factor_data.index
            if if_exists=="append":
                factor_data = OldData.where(pd.notnull(OldData), factor_data)
            elif if_exists=="update_notnull":
                factor_data = factor_data.where(pd.notnull(factor_data), OldData)
            else:
                Msg = ("因子库 '%s' 调用方法 writeData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
            self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
        factor_data.index = DTs
        return 0
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        for i, iFactor in enumerate(data.items):
            self.writeFactorData(data.iloc[i], table_name, iFactor, if_exists=if_exists, data_type=data_type.get(iFactor, None), **kwargs)
        return 0
    def optimizeData(self, table_name, factor_names):
        for iFactorName in factor_names:
            iFilePath = self._QSArgs.MainDir+os.sep+table_name+os.sep+iFactorName+"."+self._Suffix
            with self._DataLock:
                with self._openHDF5File(iFilePath, mode="a") as DataFile:
                    DTs = DataFile["DateTime"][...]
                    if np.any(np.diff(DTs)<0):
                        iData = pd.DataFrame(DataFile["Data"][...], index=DTs).sort_index()
                        DataFile["Data"][:, :] = iData.values
                        DataFile["DateTime"][:] = iData.index.values
                        self._QS_Logger.info("因子 '%s' : ’%s' 数据存储完成优化!" % (table_name, iFactorName))
                    else:
                        self._QS_Logger.info("因子 '%s' : ’%s' 数据存储不需要优化!" % (table_name, iFactorName))
        return 0
    def fixData(self, table_name, factor_names):
        for iFactorName in factor_names:
            iFilePath = self._QSArgs.MainDir+os.sep+table_name+os.sep+iFactorName+"."+self._Suffix
            FixMask = np.full(shape=(4, ), fill_value=True, dtype=bool)
            with self._DataLock:
                with self._openHDF5File(iFilePath, mode="a") as DataFile:
                    # 修复 ID 长度和数据长度不符
                    if DataFile["ID"].shape[0]>DataFile["Data"].shape[1]:
                        DataFile["ID"].resize((DataFile["Data"].shape[1], ))
                    elif DataFile["ID"].shape[0]<DataFile["Data"].shape[1]:
                        DataFile["Data"].resize((DataFile["Data"].shape[0], DataFile["ID"].shape[0]))
                    else:
                        FixMask[0] = False
                    # 修复 DT 长度和数据长度不符
                    if DataFile["DateTime"].shape[0]>DataFile["Data"].shape[0]:
                        DataFile["DateTime"].resize((DataFile["Data"].shape[0], ))
                    elif DataFile["DateTime"].shape[0]<DataFile["Data"].shape[0]:
                        DataFile["Data"].resize((DataFile["DateTime"].shape[0], DataFile["Data"].shape[1]))
                    else:
                        FixMask[1] = False
                    # 修复 ID 重复值
                    if h5py.version.version<"3.0.0":
                        IDs = pd.Series(np.arange(DataFile["ID"].shape[0]), index=DataFile["ID"][...])
                    else:
                        IDs = pd.Series(np.arange(DataFile["ID"].shape[0]), index=DataFile["ID"].asstr(encoding="utf-8")[...])
                    DuplicatedMask = IDs.index.duplicated()
                    if np.any(DuplicatedMask):
                        iData = DataFile["Data"][...]
                        for jID in set(IDs.index[DuplicatedMask]):
                            jIdx = IDs[jID].tolist()
                            iData[:, jIdx[0]] = pd.DataFrame(iData[:, jIdx].T).fillna(method="bfill").values[0, :]# TODO: h5py.version.version>=3.0.0 and data_type=string
                        nID = DuplicatedMask.shape[0] - np.sum(DuplicatedMask)
                        DataFile["ID"].resize((nID, ))
                        DataFile["ID"][:] = IDs.index.values[~DuplicatedMask]
                        DataFile["Data"].resize((DataFile["Data"].shape[0], nID))
                        DataFile["Data"][:, :] = iData[:, ~DuplicatedMask]
                    else:
                        FixMask[2] = False
                    # 修复 DT 重复值
                    DTs = pd.Series(np.arange(DataFile["DateTime"].shape[0]), index=DataFile["DateTime"][...])
                    DuplicatedMask = DTs.index.duplicated()
                    if np.any(DuplicatedMask):
                        iData = DataFile["Data"][...]
                        for jDT in set(DTs.index[DuplicatedMask]):
                            jIdx = DTs[jDT].tolist()
                            iData[jIdx[0], :] = pd.DataFrame(iData[jIdx, :]).fillna(method="bfill").values[0, :]# TODO: h5py.version.version>=3.0.0 and data_type=string
                        nDT = DuplicatedMask.shape[0] - np.sum(DuplicatedMask)
                        DataFile["DateTime"].resize((nDT, ))
                        DataFile["DateTime"][:] = DTs.index.values[~DuplicatedMask]
                        DataFile["Data"].resize((nDT, DataFile["Data"].shape[1]))
                        DataFile["Data"][:, :] = iData[~DuplicatedMask, :]
                    else:
                        FixMask[3] = False
                    if np.any(FixMask):
                        self._QS_Logger.info("因子 '%s' : '%s' 数据修复完成!" % (table_name, iFactorName))
        return 0


if __name__=="__main__":
    HDB = HDF5DB(sys_args={"主目录": r"D:\Project\DemoData\HDF5"}).connect()
    print(HDB.Args)
    #print(HDB.TableNames)
    
    #FT = HDB.getTable("stock_cn_quote_adj_no_nafilled")
    #print(FT.FactorNames)
    
    #IDs = FT.getID()
    #DTs = FT.getDateTime()
    
    #Data = FT.readData(factor_names=FT.FactorNames, ids=IDs[:5], dts=DTs[-7:])
    #print(Data)
    
    #print(Data.iloc[0])
    #print(Data.loc[:, :, IDs[0]])
    
    df = pd.DataFrame([(None, "aha"), ("中文", "aaa")], index=[dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 2)], columns=["000001.SZ", "000002.SZ"], dtype="O")
    HDB.writeFactorData(df, "test_table", "factor1", data_type="string")
    
    # FT = HDB.getTable("test_table")
    FT = HDB["test_table"]
    # Data = FT.readData(FT.FactorNames, ids=None, dts=None)
    F = FT["factor1"]
    print(F)
    Data = FT[FT.FactorNames]
    print(Data)
    Data = FT[FT.FactorNames, [dt.datetime(2022, 1, 1)]]
    print(Data)
    Data = FT[FT.FactorNames, dt.datetime(2022, 1, 1)]
    print(Data)
    
    Data = F[[dt.datetime(2022, 1, 1)]]
    print(Data)
    Data = F[:, "000001.SZ"]
    print(Data)
    Data = F[dt.datetime(2022, 1, 1), "000002.SZ"]
    print(Data)

    HDB.deleteTable("test_table")
    
    ## 数据转移
    #SDB = HDF5DB(sys_args={"主目录": r"D:\Data\HDF5Data_Old"})
    #SDB.connect()
    #Tables = ["stock_cn_quote_adj_no_nafilled"]
    #for iTable in Tables:
        #iFT = SDB.getTable(iTable)
        #for jFactor in iFT.FactorNames:
            #ijData = iFT.readFactorData(jFactor, None, None)
            ##ijData.columns = [iID.decode("utf-8") for iID in ijData.columns]
            #HDB.writeFactorData(ijData, iTable, jFactor)
    
    print("===")