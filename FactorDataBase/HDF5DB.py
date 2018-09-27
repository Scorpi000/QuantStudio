# coding=utf-8
"""基于 HDF5 文件的因子库"""
import os
import datetime as dt
from multiprocessing import Lock

import numpy as np
import pandas as pd
import h5py
from traits.api import Directory, File, on_trait_change

from QuantStudio import __QS_Error__, __QS_LibPath__
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.Tools.FileFun import listDirDir, listDirFile, readJSONFile
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5

def _identifyDataType(dtypes):
    if np.dtype('O') in dtypes.values: return 'string'
    else: return 'double'

class _FactorTable(FactorTable):
    """HDF5DB 因子表"""
    def __init__(self, name, fdb, data_type, sys_args={}, **kwargs):
        self._Suffix = "hdf5"# 文件后缀名
        self._DataType = data_type
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def getMetaData(self, key=None):
        with self._FactorDB._DataLock:
            return readNestedDictFromHDF5(self._FactorDB.MainDir+os.sep+self.Name+os.sep+"_TableInfo.h5", "/"+("" if key is None else key))
    @property
    def FactorNames(self):
        return self._DataType.index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            return self._DataType.ix[factor_names]
        with self._FactorDB._DataLock:
            MetaData = {}
            for iFactorName in factor_names:
                with h5py.File(self._FactorDB.MainDir+os.sep+self.Name+os.sep+iFactorName+"."+self._Suffix) as File:
                    if key is None:
                        MetaData[iFactorName] = pd.Series(File.attrs)
                    elif key in File.attrs:
                        MetaData[iFactorName] = File.attrs[key]
        if key is None:
            return pd.DataFrame(MetaData)
        else:
            return pd.Series(MetaData)
    def getID(self, ifactor_name=None, idt=None):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        with self._FactorDB._DataLock:
            with h5py.File(self._FactorDB.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix) as ijFile:
                return sorted(ijFile["ID"][...])
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None):
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
        if ifactor_name not in self.FactorNames: raise __QS_Error__("因子: '%s' 不存在!" % ifactor_name)
        with self._FactorDB._DataLock:
            with h5py.File(self._FactorDB.MainDir+os.sep+self.Name+os.sep+ifactor_name+"."+self._Suffix) as DataFile:
                DataType = DataFile.attrs["DataType"]
                DateTimes = DataFile["DateTime"][...]
                IDs = DataFile["ID"][...]
                if dts is None:
                    if ids is None:
                        Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs)
                    else:
                        Rslt = pd.DataFrame(DataFile["Data"][...], index=DateTimes, columns=IDs).ix[:, ids]
                else:
                    if dts and isinstance(dts[0], pd.Timestamp) and (pd.__version__>="0.20.0"): dts = [idt.to_pydatetime().timestamp() for idt in dts]
                    else: dts = [idt.timestamp() for idt in dts]
                    DateTimes = pd.Series(np.arange(0, DateTimes.shape[0]), index=DateTimes)
                    DateTimes = DateTimes.ix[dts]
                    DateTimes = DateTimes[pd.notnull(DateTimes)].astype('int')
                    nDT = DateTimes.shape[0]
                    DateTimes = DateTimes.sort_values()
                    Mask = DateTimes.tolist()
                    DateTimes = DateTimes.index.values
                    if nDT==0:
                        if ids is None:
                            Rslt = pd.DataFrame([], index=[], columns=IDs).ix[dts]
                        else:
                            Rslt = pd.DataFrame([], index=[], columns=ids).ix[dts]
                    elif nDT<1000:
                        if ids is None:
                            Rslt = pd.DataFrame(DataFile["Data"][Mask, :], index=DateTimes, columns=IDs).ix[dts]
                        else:
                            IDRuler = pd.Series(np.arange(0,IDs.shape[0]), index=IDs)
                            IDRuler = IDRuler.loc[ids]
                            IDRuler = IDRuler[pd.notnull(IDRuler)].astype('int')
                            StartInd = IDRuler.min()
                            EndInd = IDRuler.max()
                            Rslt = pd.DataFrame(DataFile["Data"][Mask, StartInd:EndInd+1], index=DateTimes, columns=IDs[StartInd:EndInd+1]).ix[dts, ids]
                    else:
                        Rslt = pd.DataFrame(DataFile["Data"][...], index=DataFile["DateTime"][...], columns=IDs).ix[dts]
                        if ids is not None: Rslt = Rslt.ix[:, ids]
        if DataType!="double":
            Rslt = Rslt.where(pd.notnull(Rslt), None)
            Rslt = Rslt.where(Rslt!="", None)
        Rslt.sort_index(axis=0, inplace=True)
        Rslt.sort_index(axis=1, inplace=True)
        Rslt.index = [dt.datetime.fromtimestamp(itms) for itms in Rslt.index]
        return Rslt

# 基于 HDF5 文件的因子数据库
# 每一张表是一个文件夹, 每个因子是一个 HDF5 文件
# 每个 HDF5 文件有三个 Dataset: DateTime, ID, Data;
# 表的元数据存储在表文件夹下特殊文件: _TableInfo.h5 中
# 因子的元数据存储在 HDF5 文件的 attrs 中
class HDF5DB(WritableFactorDB):
    """HDF5DB"""
    MainDir = Directory(label="主目录", arg_type="Directory", order=0)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._TableFactorDict = {}# {表名: pd.Series(数据类型, index=[因子名])}
        self._DataLock = Lock()# 访问该因子库资源的锁, 防止并发访问冲突
        self._isAvailable = False
        self._Suffix = "hdf5"# 文件的后缀名
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"HDF5DBConfig.json" if config_file is None else config_file), **kwargs)
        # 继承来的属性
        self.Name = "HDF5DB"
        return
    def connect(self):
        if not os.path.isdir(self.MainDir): raise __QS_Error__("不存在主目录: %s!" % self.MainDir)
        AllTables = listDirDir(self.MainDir)
        _TableFactorDict = {}
        with self._DataLock:
            for iTable in AllTables:
                iTablePath = self.MainDir+os.sep+iTable
                iFactors = set(listDirFile(iTablePath, suffix=self._Suffix))
                if not iFactors:
                    continue
                iDataType = readNestedDictFromHDF5(iTablePath+os.sep+"_TableInfo.h5", "/DataType")
                if (iDataType is None) or (iFactors!=set(iDataType.index)):
                    iDataType = {}
                    for ijFactor in iFactors:
                        with h5py.File(iTablePath+os.sep+ijFactor+"."+self._Suffix) as ijDataFile:
                            iDataType[ijFactor] = ijDataFile.attrs["DataType"]
                    iDataType = pd.Series(iDataType)
                    writeNestedDict2HDF5(iDataType, iTablePath+os.sep+"_TableInfo.h5", "/DataType")
                _TableFactorDict[iTable] = iDataType
        self._TableFactorDict = _TableFactorDict
        self._isAvailable = True
        return 0
    def disconnect(self):
        self._TableFactorDict = {}
        self._isAvailable = False
    def isAvailable(self):
        return self._isAvailable
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        return list(self._TableFactorDict)
    def getTable(self, table_name, args={}):
        if table_name not in self._TableFactorDict: raise __QS_Error__("表 '%s' 不存在!" % table_name)
        return _FactorTable(name=table_name, fdb=self, data_type=self._TableFactorDict[table_name], sys_args=args)
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name not in self._TableFactorDict: raise __QS_Error__("表: '%s' 不存在!" % old_table_name)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableFactorDict): raise __QS_Error__("表: '"+new_table_name+"' 已存在!")
        OldPath = self.MainDir+os.sep+old_table_name
        NewPath = self.MainDir+os.sep+new_table_name
        if OldPath==NewPath: pass
        elif os.path.isdir(NewPath): raise __QS_Error__("目录: '%s' 被占用!" % NewPath)
        with self._DataLock:
            os.rename(OldPath, NewPath)
        self._TableFactorDict[new_table_name] = self._TableFactorDict.pop(old_table_name)
        return 0
    def deleteTable(self, table_name):
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            if os.path.isdir(TablePath):
                shutil.rmtree(TablePath, ignore_errors=True)
        self._TableFactorDict.pop(table_name, None)
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
        if old_factor_name not in self._TableFactorDict[table_name]: raise __QS_Error__("因子: '%s' 不存在!" % old_factor_name)
        if (new_factor_name!=old_factor_name) and (new_factor_name in self._TableFactorDict[table_name]): raise __QS_Error__("表中的因子: '%s' 已存在!" % new_factor_name)
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            os.rename(TablePath+os.sep+old_factor_name+"."+self._Suffix, TablePath+os.sep+new_factor_name+"."+self._Suffix)
        self._TableFactorDict[table_name][new_factor_name] = self._TableFactorDict[table_name].pop(old_factor_name)
        return 0
    def deleteFactor(self, table_name, factor_names):
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            if set(self.getFactorName(table_name)).issubset(set(factor_names)):
                shutil.rmtree(TablePath, ignore_errors=True)
            else:
                for iFactor in factor_names:
                    iFilePath = TablePath+os.sep+iFactor+"."+self._Suffix
                    if os.path.isfile(iFilePath):
                        os.remove(iFilePath)
        FactorIndex = list(set(self._TableFactorDict.get(table_name, pd.Series()).index).difference(set(factor_names)))
        if not FactorIndex:
            self._TableFactorDict.pop(table_name, None)
        else:
            self._TableFactorDict[table_name] = self._TableFactorDict[table_name][FactorIndex]
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
        with h5py.File(FilePath) as DataFile:
            nOldDT, OldDateTimes = DataFile["DateTime"].shape[0], DataFile["DateTime"][...].tolist()
            OldDTSet = set(OldDateTimes)
            NewDateTimes = factor_data.index.difference(OldDTSet).values
            OldIDs = DataFile["ID"][...]
            OldIDSet = set(OldIDs)
            NewIDs = factor_data.columns.difference(OldIDSet).values
            DataFile["DateTime"].resize((nOldDT+NewDateTimes.shape[0], ))
            DataFile["DateTime"][nOldDT:] = NewDateTimes
            DataFile["ID"].resize((OldIDs.shape[0]+NewIDs.shape[0], ))
            DataFile["ID"][OldIDs.shape[0]:] = NewIDs
            DataFile["Data"].resize((DataFile["DateTime"].shape[0], DataFile["ID"].shape[0]))
            if NewDateTimes.shape[0]>0:
                NewData = factor_data.ix[NewDateTimes, np.r_[OldIDs, NewIDs]]
                if data_type!="double": NewData = NewData.where(pd.notnull(NewData), None)
                DataFile["Data"][nOldDT:,:] = NewData.values
            CrossedDateTimes = factor_data.index.intersection(OldDTSet)
            if CrossedDateTimes.shape[0]>0:
                CrossedDateTimePos = [OldDateTimes.index(iDT) for iDT in CrossedDateTimes]
                CrossedDateTimes = CrossedDateTimes[np.argsort(CrossedDateTimePos)]
                CrossedDateTimePos.sort()
                if NewIDs.shape[0]>0:
                    NewData = factor_data.ix[CrossedDateTimes, NewIDs]
                    if data_type!="double": NewData = NewData.where(pd.notnull(NewData), None)
                    DataFile["Data"][CrossedDateTimePos, OldIDs.shape[0]:] = NewData.values
                CrossedIDs = factor_data.columns.intersection(OldIDSet)
                NewData = factor_data.ix[CrossedDateTimes, CrossedIDs].values
                OldIDs = OldIDs.tolist()
                for i, iID in enumerate(CrossedIDs):
                    iPos = OldIDs.index(iID)
                    DataFile["Data"][CrossedDateTimePos, iPos] = NewData[:, i]
        return 0
    def writeFactorData(self, factor_data, table_name, ifactor_name, if_exists="update", data_type=None):
        if data_type is None: data_type = _identifyDataType(factor_data.dtypes)
        if data_type=='double':
            try:
                factor_data = factor_data.astype('float')
                data_type = 'double'
            except:
                factor_data = factor_data.where(pd.notnull(factor_data), None)
                data_type = 'string'
        else:
            factor_data = factor_data.where(pd.notnull(factor_data), None)
        if if_exists=="append": DTs = factor_data.index.tolist()
        if pd.__version__>="0.20.0": factor_data.index = [idt.to_pydatetime().timestamp() for idt in factor_data.index]
        else: factor_data.index = [idt.timestamp() for idt in factor_data.index]
        TablePath = self.MainDir+os.sep+table_name
        FilePath = TablePath+os.sep+ifactor_name+"."+self._Suffix
        with self._DataLock:
            if not os.path.isdir(TablePath): os.mkdir(TablePath)
            if ifactor_name not in self._TableFactorDict.get(table_name, {}):
                self._TableFactorDict[table_name] = self._TableFactorDict.get(table_name, pd.Series()).append(pd.Series(data_type, index=[ifactor_name]))
            if not os.path.isfile(FilePath):
                open(FilePath, mode="a").close()
                StrDataType = h5py.special_dtype(vlen=str)
                with h5py.File(FilePath) as DataFile:
                    DataFile.attrs["DataType"] = data_type
                    DataFile.create_dataset("ID", shape=(factor_data.shape[1],), maxshape=(None,), dtype=StrDataType, 
                                            data=factor_data.columns)
                    DataFile.create_dataset("DateTime", shape=(factor_data.shape[0],), maxshape=(None,),
                                            data=factor_data.index)
                    if data_type=='double':
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=np.float, 
                                                fillvalue=np.nan, data=factor_data.values)
                    else:
                        DataFile.create_dataset("Data", shape=factor_data.shape, maxshape=(None, None), dtype=StrDataType,
                                                fillvalue=None, data=factor_data.values)
                return 0
            elif if_exists=="update": return self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
        if if_exists=="append":
            OldData = self.getTable(table_name).readFactorData(ifactor_name=ifactor_name, ids=factor_data.columns.tolist(), dts=DTs)
            OldData.index = factor_data.index
            factor_data = OldData.where(pd.notnull(OldData), factor_data)
            self._updateFactorData(factor_data, table_name, ifactor_name, data_type)
    def writeData(self, data, table_name, if_exists="append", data_type={}, **kwargs):
        for i, iFactor in enumerate(data.items):
            iDataType = data_type.get(iFactor, _identifyDataType(data.iloc[i].dtypes))
            self.writeFactorData(data.iloc[i], table_name, iFactor, if_exists=if_exists, data_type=iDataType)
        return 0
    