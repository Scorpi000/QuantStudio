# coding=utf-8
"""基于 HDF5 文件的风险数据库"""
import os
import datetime as dt
from multiprocessing import Lock
from typing import Optional, Self, List, Any, Union

import numpy as np
import pandas as pd
import h5py
from pydantic import Field, DirectoryPath

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Risk.RiskDB import RiskDB, FactorRDB
from QuantStudio.Risk.RiskTable import RiskTable, FactorRT
from QuantStudio.Tools.FileFun import listDirFile
from QuantStudio.Tools.DateTimeFun import cutDateTime


class HDF5RiskTable(RiskTable):
    """基于 HDF5 文件的风险表"""

    def getMetaData(self, key:Optional[str]=None) -> Union[Any, pd.Series]:
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                if key is None: return pd.Series(dict(File.attrs))
                elif key in File.attrs: return File.attrs[key]
                else: return None
    
    def getDateTime(self, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        return cutDateTime(self._RiskDB._TableDT[self._QSArgs.Name], start_dt, end_dt)
    
    def readCov(self, dts:List[dt.datetime], ids:Optional[str]=None) -> Panel:
        Data = {}
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                CovGroup = File["Cov"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in CovGroup: continue
                    iGroup = CovGroup[iDTStr]
                    if h5py.version.version < "3.0.0":
                        iIDs = iGroup["ID"][...]
                    else:
                        iIDs = iGroup["ID"].asstr(encoding="utf-8")[...]
                    iCov = pd.DataFrame(iGroup["Data"][...], index=iIDs, columns=iIDs)
                    if ids is not None: iCov = iCov.reindex(index=ids, columns=ids)
                    Data[iDT] = iCov
        if Data: return Panel(Data, items=dts, major_axis=ids, minor_axis=ids)
        return Panel(items=dts, major_axis=ids, minor_axis=ids)


class HDF5RDB(RiskDB):
    """基于 HDF5 文件的风险数据库"""

    class __QS_ArgClass__(RiskDB.__QS_ArgClass__):
        Name: str = Field(default="HDF5RDB", title="名称", frozen=True)
        MainDir: DirectoryPath = Field(title="主目录", frozen=True, description="存放数据的主目录")
    
    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        self._TableDT = {}# {表名：[时点]}
        self._DataLock = Lock()
        self._Suffix = "hdf5"
        return super().__init__(args=args, config_file=(__QS_ConfigPath__+os.sep+"HDF5RDBConfig.json" if config_file is None else config_file), **kwargs)
    
    def connect(self) -> Self:
        if not os.path.isdir(self._QSArgs.MainDir): raise __QS_Error__("不存在 HDF5RDB 的主目录: %s!" % self._QSArgs.MainDir)
        AllTables = listDirFile(str(self._QSArgs.MainDir), suffix=self._Suffix)
        TableDT = {}#{表名：[时点]}
        with self._DataLock:
            for iTable in AllTables:
                with h5py.File(self._QSArgs.MainDir / (iTable+"."+self._Suffix), mode="r") as iFile:
                    if "Cov" in iFile:
                        iDTs = sorted(iFile["Cov"])
                        TableDT[iTable] = [dt.datetime.strptime(ijDT, "%Y-%m-%d %H:%M:%S.%f") for ijDT in iDTs]
        self._TableDT = TableDT
        return self
    
    def disconnect(self) -> int:
        self._TableDT = {}
        return 0

    @property
    def TableNames(self) -> List[str]:
        return sorted(self._TableDT)
    
    def getTable(self, table_name:str, args:dict={}) -> HDF5RiskTable:
        return HDF5RiskTable(self, args=args | {"Name": table_name})
    
    def setTableMetaData(self, table_name:str, key:Optional[str]=None, value:Any=None, meta_data:Optional[dict]=None):
        with self._DataLock:
            with h5py.File(self._QSArgs.MainDir / (table_name+"."+self._Suffix), mode="a") as File:
                if meta_data is None: meta_data = {}
                if key is not None: meta_data[key] = value
                for iKey, iValue in meta_data.items():
                    if iKey in File.attrs:
                        del File.attrs[iKey]
                    if (isinstance(iValue, np.ndarray)) and (iValue.dtype==np.dtype("O")):
                        File.attrs.create(iKey, data=iValue, dtype=h5py.special_dtype(vlen=str))
                    elif iValue is not None:
                        File.attrs[iKey] = iValue
    
    def renameTable(self, old_table_name:str, new_table_name:str):
        if old_table_name not in self._TableDT: raise __QS_Error__("表: '%s' 不存在!" % old_table_name)
        if (new_table_name!=old_table_name) and (new_table_name in self._TableDT): raise __QS_Error__("表: '%s' 已存在!" % new_table_name)
        with self._DataLock:
            os.rename(self._QSArgs.MainDir / (old_table_name+"."+self._Suffix), self._QSArgs.MainDir / (new_table_name+"."+self._Suffix))
        self._TableDT[new_table_name] = self._TableDT.pop(old_table_name)
    
    def deleteTable(self, table_name:str):
        with self._DataLock:
            iFilePath = self._QSArgs.MainDir / (table_name+"."+self._Suffix)
            if os.path.isfile(iFilePath): os.remove(iFilePath)
        self._TableDT.pop(table_name, None)
    
    def deleteDateTime(self, table_name:str, dts:List[dt.datetime]):
        with self._DataLock:
            with h5py.File(self._QSArgs.MainDir / (table_name+"."+self._Suffix), mode="a") as File:
                CovGroup = File["Cov"]
                for iDT in dts:
                    if iDT not in self._TableDT[table_name]: continue
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr in CovGroup: del CovGroup[iDTStr]
        self._TableDT[table_name] = sorted(set(self._TableDT[table_name]).difference(dts))
        if not self._TableDT[table_name]: self.deleteTable(table_name)
    
    def writeData(self, table_name:str, idt:dt.datetime, icov:pd.DataFrame, **kwargs):
        FilePath = self._QSArgs.MainDir / (table_name+"."+self._Suffix)
        with self._DataLock:
            if not os.path.isfile(FilePath): open(FilePath, mode="a").close()# h5py 直接创建文件名包含中文的文件会报错.
            with h5py.File(FilePath, mode="a") as File:
                iDTStr = idt.strftime("%Y-%m-%d %H:%M:%S.%f")
                if "Cov" not in File: CovGroup = File.create_group("Cov")
                else: CovGroup = File["Cov"]
                if iDTStr in CovGroup: del CovGroup[iDTStr]
                iGroup = CovGroup.create_group(iDTStr)
                StrDataType = h5py.string_dtype(encoding="utf-8")
                iGroup.create_dataset("ID", shape=(icov.shape[0], ), dtype=StrDataType, data=icov.index.values)
                iGroup.create_dataset("Data", shape=icov.shape, dtype=float, data=icov.values)
        if table_name not in self._TableDT: self._TableDT[table_name] = []
        if idt not in self._TableDT[table_name]:
            self._TableDT[table_name].append(idt)
            self._TableDT[table_name].sort()


class HDF5FactorRiskTable(FactorRT):
    """基于 HDF5 文件的多因子风险表"""

    def __init__(self, rdb: "HDF5FRDB", args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(rdb=rdb, args=args, config_file=config_file, **kwargs)
        DTs = self._RiskDB._TableDT.get(self._QSArgs.Name, [])
        if not DTs: self._FactorNames = []
        else:
            DTStr = DTs[-1].strftime("%Y-%m-%d %H:%M:%S.%f")
            with self._RiskDB._DataLock:
                with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                    if "FactorCov" in File:
                        Group = File["FactorCov"]
                        if DTStr in Group:
                            if h5py.version.version<"3.0.0":
                                self._FactorNames = sorted(Group[DTStr]["Factor"][...])
                            else:
                                self._FactorNames = sorted(Group[DTStr]["Factor"].asstr(encoding="utf-8")[...])
                        else: self._FactorNames = []
                    else: self._FactorNames = []
    
    def getMetaData(self, key:Optional[str]=None) -> Union[pd.Series, Any]:
        return HDF5RiskTable.getMetaData(self, key=key)
    
    @property
    def FactorNames(self) -> List[str]:
        return self._FactorNames
    
    def getDateTime(self, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        return cutDateTime(self._RiskDB._TableDT[self._QSArgs.Name], start_dt, end_dt)
    
    def getID(self, idt:Optional[dt.datetime]=None) -> List[str]:
        if idt is None: idt = self._RiskDB._TableDT[self._QSArgs.Name][-1]
        DTStr = idt.strftime("%Y-%m-%d %H:%M:%S.%f")
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                Group = File["SpecificRisk"]
                if DTStr in Group:
                    if h5py.version.version >= "3.0.0":
                        return sorted(Group[DTStr]["ID"].asstr(encoding="utf-8")[...])
                    else:
                        return sorted(Group[DTStr]["ID"][...])
                else: return []
    
    def getFactorReturnDateTime(self, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        FilePath = self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix)
        with self._RiskDB._DataLock:
            if not os.path.isfile(FilePath): return []
            with h5py.File(FilePath, mode="r") as File:
                if "FactorReturn" not in File: return []
                DTs = sorted(File["FactorReturn"])
        DTs = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in DTs]
        return cutDateTime(DTs, start_dt=start_dt, end_dt=end_dt)
    
    def getSpecificReturnDateTime(self, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        FilePath = self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix)
        with self._RiskDB._DataLock:
            if not os.path.isfile(FilePath): return []
            with h5py.File(FilePath, mode="r") as File:
                if "SpecificReturn" not in File: return []
                DTs = sorted(File["SpecificReturn"])
        DTs = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in DTs]
        return cutDateTime(DTs, start_dt=start_dt, end_dt=end_dt)
    
    def readFactorCov(self, dts:List[dt.datetime]) -> Panel:
        Data = {}
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                Group = File["FactorCov"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    iFactors = (iGroup["Factor"][...] if h5py.version.version<"3.0.0" else iGroup["Factor"].asstr(encoding="utf-8")[...])
                    Data[iDT] = pd.DataFrame(iGroup["Data"][...], index=iFactors, columns=iFactors)
        if Data: return Panel(Data, items=dts)
        return Panel(items=dts)
    
    def readSpecificRisk(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> pd.DataFrame:
        Data = {}
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                Group = File["SpecificRisk"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    iIDs = (iGroup["ID"][...] if h5py.version.version<"3.0.0" else iGroup["ID"].asstr(encoding="utf-8")[...])
                    Data[iDT] = pd.Series(iGroup["Data"][...], index=iIDs)
        if not Data: return pd.DataFrame(index=dts, columns=([] if ids is None else ids))
        Data = pd.DataFrame(Data).T.reindex(index=dts)
        if ids is not None: Data = Data.reindex(columns=ids)
        return Data
    
    def readFactorData(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> Panel:
        Data = {}
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                Group = File["FactorData"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    iIDs = (iGroup["ID"][...] if h5py.version.version<"3.0.0" else iGroup["ID"].asstr(encoding="utf-8")[...])
                    iFactors = (iGroup["Factor"][...] if h5py.version.version<"3.0.0" else iGroup["Factor"].asstr(encoding="utf-8")[...])
                    Data[iDT] = pd.DataFrame(iGroup["Data"][...], index=iIDs, columns=iFactors).T
        if not Data: return Panel(major_axis=dts, minor_axis=ids)
        Data = Panel(Data).swapaxes(0, 1).loc[:, dts, :]
        if ids is not None:
            if Data.minor_axis.intersection(ids).shape[0]>0: Data = Data.loc[:, :, ids]
            else: Data = Panel(items=Data.items, major_axis=dts, minor_axis=ids)
        return Data
    
    def readFactorReturn(self, dts:List[dt.datetime]) -> pd.DataFrame:
        Data = {}
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                Group = File["FactorReturn"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    iFactors = (iGroup["Factor"][...] if h5py.version.version<"3.0.0" else iGroup["Factor"].asstr(encoding="utf-8")[...])
                    Data[iDT] = pd.Series(iGroup["Data"][...], index=iFactors)
        if not Data: return pd.DataFrame(index=dts, columns=[])
        return pd.DataFrame(Data).T.reindex(index=dts)
    
    def readSpecificReturn(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> pd.DataFrame:
        Data = {}
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                Group = File["SpecificReturn"]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    iIDs = (iGroup["ID"][...] if h5py.version.version<"3.0.0" else iGroup["ID"].asstr(encoding="utf-8")[...])
                    Data[iDT] = pd.Series(iGroup["Data"][...], index=iIDs)
        if not Data: return pd.DataFrame(index=dts, columns=([] if ids is None else ids))
        Data = pd.DataFrame(Data).T.reindex(index=dts)
        if ids is not None: Data = Data.reindex(columns=ids)
        return Data
    
    def readData(self, data_item:str, dts:List[dt.datetime]) -> Union[pd.DataFrame, Panel]:
        Data = {}
        with self._RiskDB._DataLock:
            with h5py.File(self._RiskDB._QSArgs.MainDir / (self._QSArgs.Name+"."+self._RiskDB._Suffix), mode="r") as File:
                if data_item not in File: return None
                Group = File[data_item]
                for iDT in dts:
                    iDTStr = iDT.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if iDTStr not in Group: continue
                    iGroup = Group[iDTStr]
                    if "columns" in iGroup:
                        Type = "DataFrame"
                        iIndex = (iGroup["index"][...] if h5py.version.version<"3.0.0" else iGroup["index"].asstr(encoding="utf-8")[...])
                        iColumns = (iGroup["columns"][...] if h5py.version.version<"3.0.0" else iGroup["columns"].asstr(encoding="utf-8")[...])
                        Data[iDT] = pd.DataFrame(iGroup["Data"][...], index=iIndex, columns=iColumns)
                    else:
                        Type = "Series"
                        iIndex = (iGroup["index"][...] if h5py.version.version<"3.0.0" else iGroup["index"].asstr(encoding="utf-8")[...])
                        Data[iDT] = pd.Series(iGroup["Data"][...], index=iIndex)
        if not Data: return None
        if Type=="Series": return pd.DataFrame(Data).T.reindex(index=dts)
        else: return Panel(Data, items=dts)


class HDF5FRDB(FactorRDB):
    """基于 HDF5 文件的多因子风险数据库"""
    
    class __QS_ArgClass__(FactorRDB.__QS_ArgClass__):
        Name: str = Field(default="HDF5FRDB", title="名称", frozen=True)
        MainDir: DirectoryPath = Field(title="主目录", frozen=True, description="存放数据的主目录")

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        self._TableDT = {}#{表名：[时点]}
        self._DataLock = Lock()
        self._Suffix = "h5"
        super().__init__(args=args, config_file=(__QS_ConfigPath__+os.sep+"HDF5FRDBConfig.json" if config_file is None else config_file), **kwargs)
    
    def connect(self) -> Self:
        if not os.path.isdir(self._QSArgs.MainDir): raise __QS_Error__("不存在 HDF5FRDB 的主目录: %s!" % self._QSArgs.MainDir)
        AllTables = listDirFile(str(self._QSArgs.MainDir), suffix=self._Suffix)
        TableDT = {}
        with self._DataLock:
            for iTable in AllTables:
                with h5py.File(self._QSArgs.MainDir / (iTable+"."+self._Suffix), mode="r") as iFile:
                    if ("SpecificRisk" in iFile) or ("FactorReturn" in iFile) or ("FactorCov" in iFile) or ("SpecificReturn" in iFile) or ("FactorData" in iFile):
                        iDTs = (sorted(iFile["SpecificRisk"]) if "SpecificRisk" in iFile else [])
                        TableDT[iTable] = [dt.datetime.strptime(ijDT, "%Y-%m-%d %H:%M:%S.%f") for ijDT in iDTs]
        self._TableDT = TableDT
        return self
    
    def disconnect(self) -> int:
        self._TableDT = {}
        return 0
    
    @property
    def TableNames(self) -> List[str]:
        return sorted(self._TableDT)
    
    def getTable(self, table_name:str, args:dict={}) -> HDF5FactorRiskTable:
        return HDF5FactorRiskTable(self, args=args | {"Name": table_name})
    
    def setTableMetaData(self, table_name:str, key:Optional[str]=None, value:Any=None, meta_data:Optional[dict]=None):
        return HDF5RDB.setTableMetaData(self, table_name, key=key, value=value, meta_data=meta_data)
    
    def renameTable(self, old_table_name:str, new_table_name:str):
        return HDF5RDB.renameTable(self, old_table_name, new_table_name)
    
    def deleteTable(self, table_name:str):
        return HDF5RDB.deleteTable(self, table_name)
    
    def writeData(self, table_name:str, idt:dt.datetime, factor_data:Optional[pd.DataFrame]=None, factor_cov:Optional[pd.DataFrame]=None, specific_risk:Optional[pd.Series]=None, factor_ret:Optional[pd.Series]=None, specific_ret:Optional[pd.Series]=None, **kwargs):
        iDTStr = idt.strftime("%Y-%m-%d %H:%M:%S.%f")
        StrType = h5py.string_dtype(encoding="utf-8")
        FilePath = self._QSArgs.MainDir / (table_name+"."+self._Suffix)
        with self._DataLock:
            if not os.path.isfile(FilePath): open(FilePath, mode="a").close()# h5py 直接创建文件名包含中文的文件会报错.
            with h5py.File(FilePath, mode="a") as File:
                if factor_data is not None:
                    if "FactorData" not in File: Group = File.create_group("FactorData")
                    else: Group = File["FactorData"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="Factor", shape=(factor_data.shape[1], ), dtype=StrType, data=factor_data.columns.values)
                    iGroup.create_dataset(name="ID", shape=(factor_data.shape[0], ), dtype=StrType, data=factor_data.index.values)
                    iGroup.create_dataset(name="Data", shape=factor_data.shape, dtype=float, data=factor_data.values)
                if factor_cov is not None:
                    if "FactorCov" not in File: Group = File.create_group("FactorCov")
                    else: Group = File["FactorCov"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="Factor", shape=(factor_cov.shape[0], ), dtype=StrType, data=factor_cov.index.values)
                    iGroup.create_dataset(name="Data", shape=factor_cov.shape, dtype=float, data=factor_cov.values)
                if specific_risk is not None:
                    if "SpecificRisk" not in File: Group = File.create_group("SpecificRisk")
                    else: Group = File["SpecificRisk"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="ID", shape=(specific_risk.shape[0], ), dtype=StrType, data=specific_risk.index.values)
                    iGroup.create_dataset(name="Data", shape=specific_risk.shape, dtype=float, data=specific_risk.values)
                if factor_ret is not None:
                    if "FactorReturn" not in File: Group = File.create_group("FactorReturn")
                    else: Group = File["FactorReturn"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="Factor", shape=(factor_ret.shape[0], ), dtype=StrType, data=factor_ret.index.values)
                    iGroup.create_dataset(name="Data", shape=factor_ret.shape, dtype=float, data=factor_ret.values)
                if specific_ret is not None:
                    if "SpecificReturn" not in File: Group = File.create_group("SpecificReturn")
                    else: Group = File["SpecificReturn"]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="ID", shape=(specific_ret.shape[0], ), dtype=StrType, data=specific_ret.index.values)
                    iGroup.create_dataset(name="Data", shape=specific_ret.shape, dtype=float, data=specific_ret.values)
                for iKey, iValue in kwargs.items():
                    if iKey not in File: Group = File.create_group(iKey)
                    else: Group = File[iKey]
                    if iDTStr in Group: del Group[iDTStr]
                    iGroup = Group.create_group(iDTStr)
                    iGroup.create_dataset(name="index", shape=(iValue.shape[0], ), dtype=StrType, data=iValue.index.values)
                    iGroup.create_dataset(name="Data", shape=iValue.shape, dtype=float, data=iValue.values)
                    if isinstance(iValue, pd.DataFrame): iGroup.create_dataset(name="columns", shape=(iValue.shape[1], ), dtype=StrType, data=iValue.columns.values)
        if table_name not in self._TableDT: self._TableDT[table_name] = []
        if idt not in self._TableDT[table_name]:
            self._TableDT[table_name].append(idt)
            self._TableDT[table_name].sort()
