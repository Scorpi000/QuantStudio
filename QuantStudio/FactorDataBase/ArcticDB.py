# -*- coding: utf-8 -*-
"""基于 Arctic 数据库的因子数据库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
import arctic
from traits.api import Password, Str, Range

from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio import __QS_Error__, __QS_ConfigPath__

class _FactorTable(FactorTable):
    """ArcticDB 因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._Lib = fdb._Arctic[name]
        self._DataType = self._Lib.read(symbol="_FactorInfo", columns=["FactorName", "DataType"]).set_index(["FactorName"])["DataType"]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def getMetaData(self, key=None):
        TableInfo = self._Lib.read_metadata("_FactorInfo")
        if TableInfo is None: TableInfo = {}
        if key is not None: return TableInfo.get(key, None)
        else: return pd.Series(TableInfo)
    @property
    def FactorNames(self):
        return self._DataType.index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None):
        if key=="DataType": return self._DataType
        elif key is not None: MetaData = self._Lib.read(symbol="_FactorInfo", columns=["FactorName", key]).set_index(["FactorName"])[key]
        else: MetaData = self._Lib.read(symbol="_FactorInfo").set_index(["FactorName"])
        if factor_names is None: return MetaData
        return MetaData.loc[factor_names]
    def getID(self, ifactor_name=None, idt=None):
        IDs = self._Lib.list_symbols()
        IDs.remove("_FactorInfo")
        return sorted(IDs)
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None):
        if iid is None: iid = self.getID()[0]
        DTs = self._Lib.read(symbol=iid, columns=["0"]).index
        if start_dt is not None: DTs = DTs[DTs>=start_dt]
        if end_dt is not None: DTs = DTs[DTs<=end_dt]
        return DTs.tolist()
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Data = {}
        for iID in ids:
            if self._Lib.has_symbol(iID):
                iMetaData = self._Lib.read_metadata(symbol=iID)
                iCols = pd.Series(iMetaData["Cols"], index=iMetaData["FactorNames"])
                iCols = iCols.loc[iCols.index.intersection(factor_names)]
                if iCols.shape[0]>0:
                    Data[iID] = self._Lib.read(symbol=iID, columns=iCols.values.tolist(), chunk_range=pd.DatetimeIndex(dts), filter_data=True)
                    Data[iID].columns = iCols.index
        if Data: return pd.Panel(Data).swapaxes(0, 2).loc[factor_names, :, ids]
        else: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
            

# 基于 Arctic 数据库的因子数据库
# 使用 CHUNKSTORE
# 每张表是一个 library, 每个 ID 是一个 Symbol
# 每个 Symbol 存储一个 DataFrame: index 是时间点, columns 是因子
# 因子描述信息存储在特殊的 Symbol: _FactorInfo 中
# 因子表的描述信息存储在 Symbol: _FactorInfo 的 metadata 中
class ArcticDB(WritableFactorDB):
    """ArcticDB"""
    DBName = Str("arctic", arg_type="String", label="数据库名", order=0)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=27017, arg_type="Integer", label="端口", order=2)
    User = Str("", arg_type="String", label="用户名", order=3)
    Pwd = Password("", arg_type="String", label="密码", order=4)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._Arctic = None# Arctic 对象
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"ArcticDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "ArcticDB"
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_Arctic"] = self.isAvailable()
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._Arctic: self.connect()
        else: self._Arctic = None
    def connect(self):
        self._Arctic = arctic.Arctic(self.IPAddr)
        return 0
    def disconnect(self):
        self._Arctic = None
        return 1
    def isAvailable(self):
        return (self._Arctic is not None)
    @property
    def TableNames(self):
        return sorted(self._Arctic.list_libraries())
    def getTable(self, table_name, args={}):
        if table_name not in self._Arctic.list_libraries(): raise __QS_Error__("表 '%s' 不存在!" % table_name)
        return _FactorTable(name=table_name, fdb=self, sys_args=args)
    def renameTable(self, old_table_name, new_table_name):
        self._Arctic.rename_library(old_table_name, new_table_name)
        return 0
    def deleteTable(self, table_name):
        self._Arctic.delete_library(table_name)
        return 0
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        Lib = self._Arctic[table_name]
        TableInfo = Lib.read_metadata("_FactorInfo")
        if TableInfo is None: TableInfo = {}
        if meta_data is not None: TableInfo.update(dict(meta_data))
        if key is not None: TableInfo[key] = value
        Lib.write_metadata("_FactorInfo", TableInfo)
        return 0
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if table_name not in self._Arctic.list_libraries(): raise __QS_Error__("表: '%s' 不存在!" % table_name)
        Lib = self._Arctic[table_name]
        FactorInfo = Lib.read(symbol="_FactorInfo").set_index(["FactorName"])
        if old_factor_name not in FactorInfo.index: raise __QS_Error__("因子: '%s' 不存在!" % old_factor_name)
        if new_factor_name in FactorInfo.index: raise __QS_Error__("因子: '%s' 已经存在!" % new_factor_name)
        FactorNames = FactorInfo.index.tolist()
        FactorNames[FactorNames.index(old_factor_name)] = new_factor_name
        FactorInfo.index = FactorNames
        FactorInfo.index.name = "FactorName"
        Lib.write("_FactorInfo", FactorInfo.reset_index(), chunker=arctic.chunkstore.passthrough_chunker.PassthroughChunker())
        IDs = Lib.list_symbols()
        IDs.remove("_FactorInfo")
        for iID in IDs:
            iMetaData = Lib.read_metadata(iID)
            if old_factor_name in iMetaData["FactorNames"]:
                iMetaData["FactorNames"][iMetaData["FactorNames"].index(old_factor_name)] = new_factor_name
                Lib.write_metadata(iID, iMetaData)
        return 0
    def deleteFactor(self, table_name, factor_names):
        if table_name not in self._Arctic.list_libraries(): return 0
        Lib = self._Arctic[table_name]
        FactorInfo = Lib.read(symbol="_FactorInfo").set_index(["FactorName"])
        FactorInfo = FactorInfo.loc[FactorInfo.index.difference(factor_names)]
        if FactorInfo.shape[0]==0: return self.deleteTable(table_name)
        IDs = Lib.list_symbols()
        IDs.remove("_FactorInfo")
        for iID in IDs:
            iMetaData = Lib.read_metadata(iID)
            iFactorIndex = pd.Series(iMetaData["Cols"], index=iMetaData["FactorNames"])
            iFactorIndex = iFactorIndex[iFactorIndex.index.difference(factor_names)]
            if iFactorIndex.shape[0]==0:
                Lib.delete(iID)
                continue
            iFactorNames = iFactorIndex.values.tolist()
            iData = Lib.read(symbol=iID, columns=iFactorNames)
            iCols = [str(i) for i in range(iFactorIndex.shape[0])]
            iData.columns = iCols
            iMetaData["FactorNames"], iMetaData["Cols"] = iFactorNames, iCols
            Lib.write(iID, iData, metadata=iMetaData)
        Lib.write("_FactorInfo", FactorInfo.reset_index(), chunker=arctic.chunkstore.passthrough_chunker.PassthroughChunker())
        return 0
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        if (key is None) and (meta_data is None): return 0
        Lib = self._Arctic[table_name]
        FactorInfo = Lib.read(symbol="_FactorInfo").set_index(["FactorName"])
        if key is not None: FactorInfo.loc[ifactor_name, key] = value
        if meta_data is not None:
            for iKey in meta_data: FactorInfo.loc[ifactor_name, iKey] = meta_data[iKey]
        Lib.write("_FactorInfo", FactorInfo.reset_index(), chunker=arctic.chunkstore.passthrough_chunker.PassthroughChunker())
        return 0
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if data.shape[0]==0: return 0
        if table_name not in self._Arctic.list_libraries(): return self._writeNewData(data, table_name, data_type=data_type)
        Lib = self._Arctic[table_name]
        DataCols = [str(i) for i in range(data.shape[0])]
        #DTRange = pd.date_range(data.major_axis[0], data.major_axis[-1], freq=Freq)
        DTRange = data.major_axis
        OverWrite = (if_exists=="update")
        for i, iID in enumerate(data.minor_axis):
            iData = data.iloc[:, :, i]
            if not Lib.has_symbol(iID):
                iMetaData = {"FactorNames":iData.columns.tolist(), "Cols":DataCols}
                iData.index.name, iData.columns = "date", DataCols
                Lib.write(iID, iData, metadata=iMetaData)
                continue
            iMetaData = Lib.read_metadata(symbol=iID)
            iOldFactorNames, iCols = iMetaData["FactorNames"], iMetaData["Cols"]
            iNewFactorNames = iData.columns.difference(iOldFactorNames).tolist()
            #iCrossFactorNames = iOldFactorNames.intersection(iData.columns).tolist()
            iOldData = Lib.read(symbol=iID, chunk_range=DTRange, filter_data=True)
            if iOldData.shape[0]>0:
                iOldData.columns = iOldFactorNames
                iOldData = iOldData.loc[iOldData.index.union(iData.index), iOldFactorNames+iNewFactorNames]
                iOldData.update(iData, overwrite=OverWrite)
            else:
                iOldData = iData.loc[:, iOldFactorNames+iNewFactorNames]
            if iNewFactorNames:
                iCols += [str(i) for i in range(iOldData.shape[1], iOldData.shape[1]+len(iNewFactorNames))]
                #iOldData = pd.merge(iOldData, iData.loc[:, iNewFactorNames], how="outer", left_index=True, right_index=True)
            #if iCrossFactorNames:
                #iOldData = iOldData.loc[iOldData.index.union(iData.index), :]
                #iOldData.update(iData, overwrite=OverWrite)
                #if if_exists=="update": iOldData.loc[iData.index, iCrossFactorNames] = iData.loc[:, iCrossFactorNames]
                #else: iOldData.loc[iData.index, iCrossFactorNames] = iOldData.loc[iData.index, iCrossFactorNames].where(pd.notnull(iOldData.loc[iData.index, iCrossFactorNames]), iData.loc[:, iCrossFactorNames])
            iOldData.index.name, iOldData.columns ="date", iCols
            iMetaData["FactorNames"], iMetaData["Cols"] = iOldFactorNames+iNewFactorNames, iCols
            Lib.update(iID, iOldData, metadata=iMetaData, chunk_range=DTRange)
        FactorInfo = Lib.read(symbol="_FactorInfo").set_index("FactorName")
        NewFactorNames = data.items.difference(FactorInfo.index).tolist()
        FactorInfo = FactorInfo.loc[FactorInfo.index.tolist()+NewFactorNames, :]
        for iFactorName in NewFactorNames:
            if iFactorName in data_type: FactorInfo.loc[iFactorName, "DataType"] = data_type[iFactorName]
            elif np.dtype('O') in data.loc[iFactorName].dtypes: FactorInfo.loc[iFactorName, "DataType"] = "string"
            else: FactorInfo.loc[iFactorName, "DataType"] = "double"
        Lib.write("_FactorInfo", FactorInfo.reset_index(), chunker=arctic.chunkstore.passthrough_chunker.PassthroughChunker())
        return 0
    def _writeNewData(self, data, table_name, data_type):
        FactorNames = data.items.tolist()
        DataType = pd.Series("double", index=data.items)
        for i, iFactorName in enumerate(DataType.index):
            if iFactorName in data_type: DataType.iloc[i] = data_type[iFactorName]
            elif np.dtype('O') in data.iloc[i].dtypes: DataType.iloc[i] = "string"
        DataCols = [str(i) for i in range(data.shape[0])]
        data.items = DataCols
        self._Arctic.initialize_library(table_name, lib_type=arctic.CHUNK_STORE)
        Lib = self._Arctic[table_name]
        for i, iID in enumerate(data.minor_axis):
            iData = data.iloc[:, :, i]
            iMetaData = {"FactorNames":FactorNames, "Cols":DataCols}
            iData.index.name = "date"
            Lib.write(iID, iData, metadata=iMetaData)
        DataType = DataType.reset_index()
        DataType.columns = ["FactorName", "DataType"]
        Lib.write("_FactorInfo", DataType, chunker=arctic.chunkstore.passthrough_chunker.PassthroughChunker())
        data.items = FactorNames
        return 0