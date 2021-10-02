# coding=utf-8
"""基于 zarr 模块的因子库"""
import os
import shutil
import datetime as dt

import numpy as np
import pandas as pd
import fasteners
import numcodecs
import zarr
from traits.api import Directory, Str

from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.Tools.FileFun import listDirDir
from QuantStudio.Tools.api import Panel

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
        pass
    return (factor_data, data_type)

class _FactorTable(FactorTable):
    """ZarrDB 因子表"""
    def getMetaData(self, key=None, args={}):
        iZTable = zarr.open(self._FactorDB.MainDir+os.sep+self.Name, mode="r")
        with self._FactorDB._DataLock:
            if key is not None:
                if key not in iZTable.attrs: return None
                MetaData = iZTable.attrs[key]
                if isinstance(MetaData, dict):
                    Type = MetaData.get("_Type")
                    if Type=="Array": return np.array(MetaData["List"])
                    elif Type=="Series": return pd.read_json(MetaData["Json"], typ="series")
                    elif Type=="DataFrame": return pd.read_json(MetaData["Json"], typ="frame")
                    else: return MetaData
                else: return MetaData
        MetaData = {}
        for iKey in iZTable.attrs:
            MetaData[iKey] = self.getMetaData(key=iKey, args=args)
        return MetaData
    @property
    def FactorNames(self):
        with self._FactorDB._DataLock:
            ZTable = zarr.open(self._FactorDB.MainDir+os.sep+self.Name, mode="r")
            DataType = ZTable.attrs.get("DataType", {})
        return sorted(DataType)
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        AllFactorNames = self.FactorNames
        if factor_names is None: factor_names = AllFactorNames
        elif set(factor_names).isdisjoint(AllFactorNames): return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        with self._FactorDB._DataLock:
            MetaData = {}
            ZTable = zarr.open(self._FactorDB.MainDir+os.sep+self.Name, mode="r")
            for iFactorName in factor_names:
                if iFactorName in AllFactorNames:
                    iZFactor = ZTable[iFactorName]
                    if key is None: MetaData[iFactorName] = pd.Series(iZFactor.attrs)
                    elif key in iZFactor.attrs: MetaData[iFactorName] = iZFactor.attrs[key]
        if not MetaData: return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        if key is None: return pd.DataFrame(MetaData).loc[:, factor_names]
        else: return pd.Series(MetaData).loc[factor_names]
    def getID(self, ifactor_name=None, idt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        with self._FactorDB._DataLock:
            ZTable = zarr.open(self._FactorDB.MainDir+os.sep+self.Name, mode="r")
            ZFactor = ZTable[ifactor_name]
            return sorted(ZFactor["ID"][:])
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        with self._FactorDB._DataLock:
            ZTable = zarr.open(self._FactorDB.MainDir+os.sep+self.Name, mode="r")
            ZFactor = ZTable[ifactor_name]
            Timestamps = ZFactor["DateTime"][:]
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
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)
    def readFactorData(self, ifactor_name, ids, dts, args={}):
        with self._FactorDB._DataLock:
            ZTable = zarr.open(self._FactorDB.MainDir+os.sep+self.Name, mode="r")
            if ifactor_name not in ZTable: raise __QS_Error__("因子库 '%s' 的因子表 '%s' 中不存在因子 '%s'!" % (self._FactorDB.Name, self.Name, ifactor_name))
            ZFactor = ZTable[ifactor_name]
            DataType = ZFactor.attrs["DataType"]
            DateTimes = ZFactor["DateTime"][:]
            IDs = ZFactor["ID"][:]
            if ids is not None:
                IDIndices = pd.Series(np.arange(0, IDs.shape[0]), index=IDs)
                IDIndices = IDIndices[IDIndices.index.intersection(ids)].astype('int')
            else:
                IDIndices = slice(None)
            if dts is not None:
                DTIndices = pd.Series(np.arange(0, DateTimes.shape[0]), index=DateTimes)
                DTIndices = DTIndices[DTIndices.index.intersection(dts)].astype('int')
            else:
                DTIndices = slice(None)
            Rslt = pd.DataFrame(ZFactor["Data"].get_orthogonal_selection((DTIndices, IDIndices)), index=DateTimes[DTIndices], columns=IDs[IDIndices])
        if ids is not None:
            if Rslt.shape[1]>0: Rslt = Rslt.loc[:, ids]
            else: Rslt = pd.DataFrame(index=Rslt.index, columns=ids)
        else:
            Rslt = Rslt.sort_index(axis=1)
        if dts is not None:
            if Rslt.shape[0]>0: Rslt = Rslt.loc[dts, :]
            else: Rslt = pd.DataFrame(index=dts, columns=Rslt.columns)
        else:
            Rslt = Rslt.sort_index(axis=0)
        if DataType=="string":
            Rslt = Rslt.where(pd.notnull(Rslt), None)
            Rslt = Rslt.where(Rslt!="", None)
        return Rslt

# 基于 zarr 模块的因子数据库
# 每一张表是一个 zarr 根 group, 每个因子是一个子 group
# 每个因子 group 有三个 Dataset: DateTime, ID, Data;
# 表的元数据存储在根 group 的 attrs 中
# 因子的元数据存储在因子 group 的 attrs 中
class ZarrDB(WritableFactorDB):
    """ZarrDB"""
    Name = Str("ZarrDB", arg_type="String", label="名称", order=-100)
    MainDir = Directory(label="主目录", arg_type="Directory", order=0)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._LockFile = None# 文件锁的目标文件
        self._DataLock = None# 访问该因子库资源的锁, 防止并发访问冲突
        self._isAvailable = False
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"ZarrDBConfig.json" if config_file is None else config_file), **kwargs)
        return
    def connect(self):
        if not os.path.isdir(self.MainDir): raise __QS_Error__("ZarrDB.connect: 不存在主目录 '%s'!" % self.MainDir)
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
        if not os.path.isdir(self.MainDir+os.sep+table_name): raise __QS_Error__("ZarrDB.getTable: 表 '%s' 不存在!" % table_name)
        return _FactorTable(name=table_name, fdb=self, sys_args=args, logger=self._QS_Logger)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name==new_table_name: return 0
        OldPath = self.MainDir+os.sep+old_table_name
        NewPath = self.MainDir+os.sep+new_table_name
        with self._DataLock:
            if not os.path.isdir(OldPath): raise __QS_Error__("ZarrDB.renameTable: 表: '%s' 不存在!" % old_table_name)
            if os.path.isdir(NewPath): raise __QS_Error__("ZarrDB.renameTable: 表 '"+new_table_name+"' 已存在!")
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
            iZTable = zarr.open(self.MainDir+os.sep+table_name, mode="a")
            if key is not None:
                if key in iZTable.attrs:
                    del iZTable.attrs[key]
                if isinstance(value, np.ndarray):
                    iZTable.attrs[key] = {"_Type":"Array", "List":value.tolist()}
                elif isinstance(value, pd.Series):
                    iZTable.attrs[key] = {"_Type":"Series", "Json":value.to_json(index=True)}
                elif isinstance(value, pd.DataFrame):
                    iZTable.attrs[key] = {"_Type":"DataFrame", "Json":value.to_json(index=True)}
                elif value is not None:
                    iZTable.attrs[key] = value
        if meta_data is not None:
            for iKey in meta_data:
                self.setTableMetaData(table_name, key=iKey, value=meta_data[iKey], meta_data=None)
        return 0
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name==new_factor_name: return 0
        with self._DataLock:
            iZTable = zarr.open(self.MainDir+os.sep+table_name, mode="a")
            if old_factor_name not in iZTable: raise __QS_Error__("ZarrDB.renameFactor: 表 ’%s' 中不存在因子 '%s'!" % (table_name, old_factor_name))
            if new_factor_name in iZTable: raise __QS_Error__("ZarrDB.renameFactor: 表 ’%s' 中的因子 '%s' 已存在!" % (table_name, new_factor_name))
            iZTable[new_factor_name] = iZTable.pop(old_factor_name)
            DataType = iZTable.attrs.get("DataType", {})
            DataType[new_factor_name] = DataType.pop(old_factor_name)
            iZTable.attrs["DataType"] = DataType
        return 0
    def deleteFactor(self, table_name, factor_names):
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            iZTable = zarr.open(TablePath, mode="a")
            DataType = iZTable.attrs.get("DataType", {})
            if set(DataType).issubset(set(factor_names)):
                shutil.rmtree(TablePath, ignore_errors=True)
            else:
                for iFactor in factor_names:
                    if iFactor in iZTable: del iZTable[iFactor]
                    DataType.pop(iFactor, None)
                iZTable.attrs["DataType"] = DataType
        return 0
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        with self._DataLock:
            iZTable = zarr.open(self.MainDir+os.sep+table_name, mode="a")
            iZFactor = iZTable[ifactor_name]
            if key is not None:
                if key in iZFactor.attrs:
                    del iZFactor.attrs[key]
                if isinstance(value, np.ndarray):
                    iZFactor.attrs[key] = {"_Type":"Array", "List":value.tolist()}
                elif isinstance(value, pd.Series):
                    iZFactor.attrs[key] = {"_Type":"Series", "Json":value.to_json(index=True)}
                elif isinstance(value, pd.DataFrame):
                    iZFactor.attrs[key] = {"_Type":"DataFrame", "Json":value.to_json(index=True)}
                elif value is not None:
                    iZFactor.attrs[key] = value
        if meta_data is not None:
            for iKey in meta_data:
                self.setFactorMetaData(table_name, ifactor_name=ifactor_name, key=iKey, value=meta_data[iKey], meta_data=None)
        return 0
    def _updateFactorData(self, factor_data, table_name, ifactor_name, data_type):
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            ZTable = zarr.open(TablePath, mode="a")
            ZFactor = ZTable[ifactor_name]
            OldDateTimes, OldIDs = ZFactor["DateTime"][:], ZFactor["ID"][:]
            NewDateTimes = factor_data.index.difference(OldDateTimes).values
            NewIDs = factor_data.columns.difference(OldIDs).values
            ZFactor["DateTime"].append(NewDateTimes, axis=0)
            ZFactor["ID"].append(NewIDs, axis=0)
            ZFactor["Data"].resize((ZFactor["DateTime"].shape[0], ZFactor["ID"].shape[0]))
            IDIndices = pd.Series(np.arange(ZFactor["ID"].shape[0]), index=np.r_[OldIDs, NewIDs]).loc[factor_data.columns].tolist()
            DTIndices = pd.Series(np.arange(ZFactor["DateTime"].shape[0]), index=np.r_[OldDateTimes, NewDateTimes]).loc[factor_data.index].tolist()
            if data_type!="double": factor_data = factor_data.where(pd.notnull(factor_data), None)
            else: factor_data = factor_data.astype("float")
            ZFactor["Data"].set_orthogonal_selection((DTIndices, IDIndices), factor_data.values)
        return 0
    def writeFactorData(self, factor_data, table_name, ifactor_name, if_exists="update", data_type=None, **kwargs):
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            ZTable = zarr.open(TablePath, mode="a")
            if ifactor_name not in ZTable:
                factor_data, data_type = _identifyDataType(factor_data, data_type)
                ZFactor = ZTable.create_group(ifactor_name, overwrite=True)
                ZFactor.create_dataset("ID", shape=(factor_data.shape[1], ), data=factor_data.columns.values, dtype=object, object_codec=numcodecs.VLenUTF8(), overwrite=True)
                ZFactor.create_dataset("DateTime", shape=(factor_data.shape[0], ), data=factor_data.index.values, dtype="M8[ns]", overwrite=True)
                if data_type=="double":
                    ZFactor.create_dataset("Data", shape=factor_data.shape, data=factor_data.values, dtype="f8", fill_value=np.nan, overwrite=True)
                elif data_type=="string":
                    ZFactor.create_dataset("Data", shape=factor_data.shape, data=factor_data.values, dtype=object, object_codec=numcodecs.VLenUTF8(), overwrite=True)
                elif data_type=="object":
                    ZFactor.create_dataset("Data", shape=factor_data.shape, data=factor_data.values, dtype=object, object_codec=numcodecs.Pickle(), overwrite=True)
                ZFactor.attrs["DataType"] = data_type
                DataType = ZTable.attrs.get("DataType", {})
                DataType[ifactor_name] = data_type
                ZTable.attrs["DataType"] = DataType
                return 0
        if if_exists=="update":
            self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
        else:
            OldData = self.getTable(table_name).readFactorData(ifactor_name=ifactor_name, ids=factor_data.columns.tolist(), dts=factor_data.index.tolist())
            if if_exists=="append":
                factor_data = OldData.where(pd.notnull(OldData), factor_data)
            elif if_exists=="update_notnull":
                factor_data = factor_data.where(pd.notnull(factor_data), OldData)
            else:
                Msg = ("因子库 '%s' 调用方法 writeData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
                self._QS_Logger.error(Msg)
                raise __QS_Error__(Msg)
            self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
        return 0
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        for i, iFactor in enumerate(data.items):
            self.writeFactorData(data.iloc[i], table_name, iFactor, if_exists=if_exists, data_type=data_type.get(iFactor, None), **kwargs)
        return 0