# coding=utf-8
"""基于 HDF5 文件的因子库"""
import os
import shutil
import pickle
import datetime as dt

import numpy as np
import pandas as pd
import fasteners
import h5py
from traits.api import Directory

from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.Tools.FileFun import listDirDir, listDirFile, readJSONFile
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5

def _identifyDataType(factor_data, data_type=None):
    if (data_type is None) or (data_type=="double"):
        try:
            factor_data = factor_data.astype(float)
        except:
            data_type = "object"
        else:
            data_type = "double"
    if data_type=="string":
        factor_data = factor_data.where(pd.notnull(factor_data), None)
    elif data_type=="object":
        factor_data = factor_data.applymap(lambda x: np.fromiter(pickle.dumps(x), dtype=np.uint8))
    return (factor_data, data_type)

class _FactorTable(FactorTable):
    """HDF5DB 因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._Suffix = fdb._Suffix# 文件后缀名
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        return sorted(listDirFile(self._FactorDB.MainDir+os.sep+self.Name, suffix=self._Suffix))
    def getMetaData(self, key=None):
        with self._FactorDB._DataLock:
            return readNestedDictFromHDF5(self._FactorDB.MainDir+os.sep+self.Name+os.sep+"_TableInfo.h5", "/"+("" if key is None else key))
    def getFactorMetaData(self, factor_names=None, key=None):
        AllFactorNames = self.FactorNames
        if factor_names is None: factor_names = AllFactorNames
        elif set(factor_names).isdisjoint(AllFactorNames): return super().getFactorMetaData(factor_names=factor_names, key=key)
        with self._FactorDB._DataLock:
            MetaData = {}
            for iFactorName in factor_names:
                if iFactorName in AllFactorNames:
                    with h5py.File(self._FactorDB.MainDir+os.sep+self.Name+os.sep+iFactorName+"."+self._Suffix) as File:
                        if key is None: MetaData[iFactorName] = pd.Series(File.attrs)
                        elif key in File.attrs: MetaData[iFactorName] = File.attrs[key]
        if not MetaData: return super().getFactorMetaData(factor_names=factor_names, key=key)
        if key is None: return pd.DataFrame(MetaData).loc[:, factor_names]
        else: return pd.Series(MetaData).loc[factor_names]
    def getID(self, ifactor_name=None, idt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        with self._FactorDB._DataLock:
            with h5py.File(self._FactorDB.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix) as ijFile:
                return sorted(ijFile["ID"][...])
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        with self._FactorDB._DataLock:
            with h5py.File(self._FactorDB.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix) as ijFile:
                Timestamps = ijFile["DateTime"][...]
        if start_dt is not None:
            if isinstance(start_dt, pd.Timestamp) and (pd.__version__>="0.20.0"): start_dt = start_dt.to_pydatetime().timestamp()
            else: start_dt = start_dt.timestamp()
            Timestamps = Timestamps[Timestamps>=start_dt]
        if end_dt is not None:
            if isinstance(end_dt, pd.Timestamp) and (pd.__version__>="0.20.0"): end_dt = end_dt.to_pydatetime().timestamp()
            else: end_dt = end_dt.timestamp()
            Timestamps = Timestamps[Timestamps<=end_dt]
        return sorted(dt.datetime.fromtimestamp(iTimestamp) for iTimestamp in Timestamps)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Data = {iFactor: self.readFactorData(ifactor_name=iFactor, ids=ids, dts=dts, args=args) for iFactor in factor_names}
        return pd.Panel(Data).loc[factor_names]
    def readFactorData(self, ifactor_name, ids, dts, args={}):
        FilePath = self._FactorDB.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix
        if not os.path.isfile(FilePath): raise __QS_Error__("因子库 '%s' 的因子表 '%s' 中不存在因子 '%s'!" % (self._FactorDB.Name, self.Name, ifactor_name))
        with self._FactorDB._DataLock:
            with h5py.File(FilePath, mode="r") as DataFile:
                DataType = DataFile.attrs["DataType"]
                DateTimes = DataFile["DateTime"][...]
                IDs = DataFile["ID"][...]
                if dts is None:
                    if ids is None:
                        Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs).sort_index(axis=1)
                    elif set(ids).isdisjoint(IDs):
                        Rslt = pd.DataFrame(index=DateTimes, columns=ids)
                    else:
                        Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs).loc[:, ids]
                    Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
                elif (ids is not None) and set(ids).isdisjoint(IDs):
                    Rslt = pd.DataFrame(index=dts, columns=ids)
                else:
                    if dts and isinstance(dts[0], pd.Timestamp) and (pd.__version__>="0.20.0"): dts = [idt.to_pydatetime().timestamp() for idt in dts]
                    else: dts = [idt.timestamp() for idt in dts]
                    DateTimes = pd.Series(np.arange(0, DateTimes.shape[0]), index=DateTimes, dtype=np.int)
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
                            Rslt = pd.DataFrame(DataFile["Data"][Mask, :], index=DateTimes, columns=IDs).loc[dts].sort_index(axis=1)
                        else:
                            IDRuler = pd.Series(np.arange(0,IDs.shape[0]), index=IDs)
                            IDRuler = IDRuler.loc[ids]
                            StartInd, EndInd = int(IDRuler.min()), int(IDRuler.max())
                            Rslt = pd.DataFrame(DataFile["Data"][Mask, StartInd:EndInd+1], index=DateTimes, columns=IDs[StartInd:EndInd+1]).loc[dts, ids]
                    else:
                        Rslt = pd.DataFrame(DataFile["Data"][...], index=DataFile["DateTime"][...], columns=IDs).loc[dts]
                        if ids is not None: Rslt = Rslt.loc[:, ids]
                        else: Rslt.sort_index(axis=1, inplace=True)
                    Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
        if DataType=="string":
            Rslt = Rslt.where(pd.notnull(Rslt), None)
            Rslt = Rslt.where(Rslt!="", None)
        elif DataType=="object":
            Rslt = Rslt.applymap(lambda x: pickle.loads(bytes(x)) if isinstance(x, np.ndarray) and (x.shape[0]>0) else None)
        return Rslt.sort_index(axis=0)

# 基于 HDF5 文件的因子数据库
# 每一张表是一个文件夹, 每个因子是一个 HDF5 文件
# 每个 HDF5 文件有三个 Dataset: DateTime, ID, Data;
# 表的元数据存储在表文件夹下特殊文件: _TableInfo.h5 中
# 因子的元数据存储在 HDF5 文件的 attrs 中
class HDF5DB(WritableFactorDB):
    """HDF5DB"""
    MainDir = Directory(label="主目录", arg_type="Directory", order=0)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._LockFile = None# 文件锁的目标文件
        self._DataLock = None# 访问该因子库资源的锁, 防止并发访问冲突
        self._isAvailable = False
        self._Suffix = "hdf5"# 文件的后缀名
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"HDF5DBConfig.json" if config_file is None else config_file), **kwargs)
        # 继承来的属性
        self.Name = "HDF5DB"
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_DataLock"] = (True if self._DataLock is not None else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._DataLock:
            self._DataLock = fasteners.InterProcessLock(self._LockFile)
        else:
            self._DataLock = None
    def connect(self):
        if not os.path.isdir(self.MainDir): raise __QS_Error__("HDF5DB.connect: 不存在主目录 '%s'!" % self.MainDir)
        if not os.path.isfile(self.MainDir+os.sep+"LockFile"):
            open(self.MainDir+os.sep+"LockFile", mode="a").close()
        self._LockFile = self.MainDir+os.sep+"LockFile"
        self._DataLock = fasteners.InterProcessLock(self._LockFile)
        self._isAvailable = True
        return 0
    def disconnect(self):
        self._LockFile = None
        self._DataLock = None
        self._isAvailable = False
    def isAvailable(self):
        return self._isAvailable
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        return sorted(listDirDir(self.MainDir))
    def getTable(self, table_name, args={}):
        if not os.path.isdir(self.MainDir+os.sep+table_name): raise __QS_Error__("HDF5DB.getTable: 表 '%s' 不存在!" % table_name)
        return _FactorTable(name=table_name, fdb=self, sys_args=args, logger=self._QS_Logger)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name==new_table_name: return 0
        OldPath = self.MainDir+os.sep+old_table_name
        NewPath = self.MainDir+os.sep+new_table_name
        with self._DataLock:
            if not os.path.isdir(OldPath): raise __QS_Error__("HDF5DB.renameTable: 表: '%s' 不存在!" % old_table_name)
            if os.path.isdir(NewPath): raise __QS_Error__("HDF5DB.renameTable: 表 '"+new_table_name+"' 已存在!")
            os.rename(OldPath, NewPath)
        return 0
    def deleteTable(self, table_name):
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            if os.path.isdir(TablePath):
                shutil.rmtree(TablePath, ignore_errors=True)
        return 0
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+os.sep+"_TableInfo.h5") as File:
                if key is not None:
                    if key in File:
                        del File[key]
                    if (isinstance(value, np.ndarray)) and (value.dtype==np.dtype("O")):
                        File.create_dataset(key, data=value, dtype=h5py.special_dtype(vlen=str))
                    elif value is not None:
                        File.create_dataset(key, data=value)
        if meta_data is not None:
            for iKey in meta_data:
                self.setTableMetaData(table_name, key=iKey, value=meta_data[iKey], meta_data=None)
        return 0
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name==new_factor_name: return 0
        OldPath = self.MainDir+os.sep+table_name+os.sep+old_factor_name+"."+self._Suffix
        NewPath = self.MainDir+os.sep+table_name+os.sep+new_factor_name+"."+self._Suffix
        with self._DataLock:
            if not os.path.isfile(OldPath): raise __QS_Error__("HDF5DB.renameFactor: 表 ’%s' 中不存在因子 '%s'!" % (table_name, old_factor_name))
            if os.path.isfile(NewPath): raise __QS_Error__("HDF5DB.renameFactor: 表 ’%s' 中的因子 '%s' 已存在!" % (table_name, new_factor_name))
            os.rename(OldPath, NewPath)
        return 0
    def deleteFactor(self, table_name, factor_names):
        TablePath = self.MainDir+os.sep+table_name
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
        with self._DataLock:
            with h5py.File(self.MainDir+os.sep+table_name+os.sep+ifactor_name+"."+self._Suffix) as File:
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
        FilePath = self.MainDir+os.sep+table_name+os.sep+ifactor_name+"."+self._Suffix
        with self._DataLock:
            with h5py.File(FilePath) as DataFile:
                OldDataType = DataFile.attrs["DataType"]
                if data_type is None: data_type = OldDataType
                factor_data, data_type = _identifyDataType(factor_data, data_type)
                if OldDataType!=data_type: raise __QS_Error__("HDF5DB.writeFactorData: 新数据无法转换成已有数据的数据类型 '%s'!" % OldDataType)
                nOldDT, OldDateTimes = DataFile["DateTime"].shape[0], DataFile["DateTime"][...].tolist()
                NewDateTimes = factor_data.index.difference(OldDateTimes).values
                OldIDs = DataFile["ID"][...]
                NewIDs = factor_data.columns.difference(OldIDs).values
                DataFile["DateTime"].resize((nOldDT+NewDateTimes.shape[0], ))
                DataFile["DateTime"][nOldDT:] = NewDateTimes
                DataFile["ID"].resize((OldIDs.shape[0]+NewIDs.shape[0], ))
                DataFile["ID"][OldIDs.shape[0]:] = NewIDs
                DataFile["Data"].resize((DataFile["DateTime"].shape[0], DataFile["ID"].shape[0]))
                if NewDateTimes.shape[0]>0:
                    NewData = factor_data.loc[NewDateTimes, np.r_[OldIDs, NewIDs]]
                    if data_type=="string": NewData = NewData.where(pd.notnull(NewData), None)
                    elif data_type=="double": NewData = NewData.astype("float")
                    elif data_type=="object": NewData = NewData.applymap(lambda x: x if isinstance(x, np.ndarray) else np.fromiter(pickle.dumps(x), dtype=np.uint8))
                    DataFile["Data"][nOldDT:, :] = NewData.values
                CrossedDateTimes = factor_data.index.intersection(OldDateTimes)
                if CrossedDateTimes.shape[0]==0: return 0
                if len(CrossedDateTimes)==len(OldDateTimes):
                    if NewIDs.shape[0]>0:
                        NewData = factor_data.loc[OldDateTimes, NewIDs]
                        if data_type=="string": NewData = NewData.where(pd.notnull(NewData), None)
                        elif data_type=="double": NewData = NewData.astype("float")
                        elif data_type=="object": NewData = NewData.applymap(lambda x: x if isinstance(x, np.ndarray) else np.fromiter(pickle.dumps(x), dtype=np.uint8))
                        DataFile["Data"][:nOldDT, OldIDs.shape[0]:] = NewData.values
                    CrossedIDs = factor_data.columns.intersection(OldIDs)
                    if CrossedIDs.shape[0]>0:
                        OldIDs = OldIDs.tolist()
                        CrossedIDPos = [OldIDs.index(iID) for iID in CrossedIDs]
                        CrossedIDs = CrossedIDs[np.argsort(CrossedIDPos)]
                        CrossedIDPos.sort()
                        DataFile["Data"][:nOldDT, CrossedIDPos] = factor_data.loc[OldDateTimes, CrossedIDs].values
                    return 0
                CrossedDateTimePos = [OldDateTimes.index(iDT) for iDT in CrossedDateTimes]
                CrossedDateTimes = CrossedDateTimes[np.argsort(CrossedDateTimePos)]
                CrossedDateTimePos.sort()
                if NewIDs.shape[0]>0:
                    NewData = factor_data.loc[CrossedDateTimes, NewIDs]
                    if data_type=="string": NewData = NewData.where(pd.notnull(NewData), None)
                    elif data_type=="double": NewData = NewData.astype("float")
                    elif data_type=="object": NewData = NewData.applymap(lambda x: x if isinstance(x, np.ndarray) else np.fromiter(pickle.dumps(x), dtype=np.uint8))
                    DataFile["Data"][CrossedDateTimePos, OldIDs.shape[0]:] = NewData.values
                CrossedIDs = factor_data.columns.intersection(OldIDs)
                if CrossedIDs.shape[0]>0:
                    NewData = factor_data.loc[CrossedDateTimes, CrossedIDs].values
                    OldIDs = OldIDs.tolist()
                    if data_type=="object":
                        for i, iID in enumerate(CrossedIDs):
                            iPos = OldIDs.index(iID)
                            DataFile["Data"][CrossedDateTimePos, iPos:iPos+1] = NewData[:, i:i+1]
                    else:
                        for i, iID in enumerate(CrossedIDs):
                            iPos = OldIDs.index(iID)
                            DataFile["Data"][CrossedDateTimePos, iPos] = NewData[:, i]
        return 0
    def writeFactorData(self, factor_data, table_name, ifactor_name, if_exists="update", data_type=None):
        DTs = factor_data.index
        if pd.__version__>="0.20.0": factor_data.index = [idt.to_pydatetime().timestamp() for idt in factor_data.index]
        else: factor_data.index = [idt.timestamp() for idt in factor_data.index]
        TablePath = self.MainDir+os.sep+table_name
        FilePath = TablePath+os.sep+ifactor_name+"."+self._Suffix
        with self._DataLock:
            if not os.path.isdir(TablePath): os.mkdir(TablePath)
            if not os.path.isfile(FilePath):
                factor_data, data_type = _identifyDataType(factor_data, data_type)
                open(FilePath, mode="a").close()# h5py 直接创建文件名包含中文的文件会报错.
                #StrDataType = h5py.special_dtype(vlen=str)
                StrDataType = h5py.string_dtype(encoding="utf-8")
                with h5py.File(FilePath, mode="r+") as DataFile:
                    DataFile.attrs["DataType"] = data_type
                    DataFile.create_dataset("ID", shape=(factor_data.shape[1],), maxshape=(None,), dtype=StrDataType, data=factor_data.columns)
                    DataFile.create_dataset("DateTime", shape=(factor_data.shape[0],), maxshape=(None,), data=factor_data.index)
                    if data_type=="double":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=np.float, fillvalue=np.nan, data=factor_data.values)
                    elif data_type=="string":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=StrDataType, fillvalue=None, data=factor_data.values)
                    elif data_type=="object":
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=h5py.vlen_dtype(np.uint8), data=factor_data.values)
                factor_data.index = DTs
                return 0
        if if_exists=="update":
            self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
        elif if_exists=="append":
            OldData = self.getTable(table_name).readFactorData(ifactor_name=ifactor_name, ids=factor_data.columns.tolist(), dts=DTs.tolist())
            OldData.index = factor_data.index
            factor_data = OldData.where(pd.notnull(OldData), factor_data)
            self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
        factor_data.index = DTs
        return 0
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        for i, iFactor in enumerate(data.items):
            self.writeFactorData(data.iloc[i], table_name, iFactor, if_exists=if_exists, data_type=data_type.get(iFactor, None))
        return 0