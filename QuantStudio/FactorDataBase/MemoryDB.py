# coding=utf-8
"""内存因子库"""
import os
import datetime as dt
from multiprocessing import Lock

import numpy as np
import pandas as pd
from traits.api import Str, Float, Enum, Either, Date

from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.Tools.api import Panel
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import adjustDataDTID


class _FactorTable(FactorTable):
    """内存因子表"""
    class __QS_ArgClass__(FactorTable.__QS_ArgClass__):
        LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
        OnlyStartLookBack = Enum(False, True, label="只起始日回溯", arg_type="Bool", order=1)
        OnlyLookBackNontarget = Enum(False, True, label="只回溯非目标日", arg_type="Bool", order=2)
        OnlyLookBackDT = Enum(False, True, label="只回溯时点", arg_type="Bool", order=3)
        TargetDT = Either(None, Date, arg_type="DateTime", label="目标时点", order=4)
    
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._Data = fdb._Data[name]
        self._TableMeta = fdb._TableMeta.get(name, {})
        self._FactorMeta = fdb._FactorMeta.get(name, {})
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    
    @property
    def FactorNames(self):
        return sorted(self._Data.keys())
    
    def getMetaData(self, key=None, args={}):
        if not self._TableMeta:
            return (pd.Series() if key is None else None)
        if key is None:
            return pd.Series(self._TableMeta)
        else:
            return self._TableMeta.get(key, None)
    
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        AllFactorNames = self.FactorNames
        if factor_names is None: factor_names = AllFactorNames
        elif set(factor_names).isdisjoint(AllFactorNames): return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        MetaData = {}
        for iFactorName in factor_names:
            if iFactorName in self._FactorMeta:
                if key is None:
                    MetaData[iFactorName] = self._FactorMeta[iFactorName]
                else:
                    MetaData[iFactorName] = self._FactorMeta[iFactorName].get(key, None)
        if not MetaData: return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
        if key is None: return pd.DataFrame(MetaData).T.reindex(index=factor_names)
        else: return pd.Series(MetaData).reindex(index=factor_names)
    
    def getID(self, ifactor_name=None, idt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        if idt is None:
            return sorted(self._Data[ifactor_name].columns)
        elif idt not in self._Data[ifactor_name].index:
            return []
        else:
            return sorted(self._Data[ifactor_name].loc[idt].dropna().index)
    
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if ifactor_name is None: ifactor_name = self.FactorNames[0]
        if iid is None:
            DTs = self._Data[ifactor_name].index
        elif iid not in self._Data[ifactor_name].columns:
            return []
        else:
            DTs = self._Data[ifactor_name].loc[:, iid].dropna().index
        if start_dt: DTs = DTs[DTs>=start_dt]
        if end_dt: DTs = DTs[DTs<=end_dt]
        return sorted(DTs)
    
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Data = {iFactor: self.readFactorData(ifactor_name=iFactor, ids=ids, dts=dts, args=args) for iFactor in factor_names}
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)
    
    def _readFactorData(self, ifactor_name, ids, dts, args={}):
        return self._Data[ifactor_name].reindex(index=dts, columns=ids)
    
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

# 内存因子数据库, {表: {因子: DataFrame}}
class MemoryDB(WritableFactorDB):
    """MemoryDB"""
    class __QS_ArgClass__(WritableFactorDB.__QS_ArgClass__):
        Name = Str("MemoryDB", arg_type="String", label="名称", order=-100)
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._isAvailable = False
        self._Data = {}# 所有的因子数据
        self._TableMeta = {}# 表元信息
        self._FactorMeta = {}# 因子元信息
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"MemoryDBConfig.json" if config_file is None else config_file), **kwargs)
        return
    
    def connect(self):
        self._isAvailable = True
        self._Data = {}
        return self
    
    def disconnect(self):
        self._isAvailable = False
        # 清空所有数据
        self._Data = {}# 所有的因子数据
        self._TableMeta = {}# 表元信息
        self._FactorMeta = {}# 因子元信息
    
    def isAvailable(self):
        return self._isAvailable
    
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        return sorted(self._Data)
    
    def getTable(self, table_name, args={}):
        if table_name not in self._Data: raise __QS_Error__("MemoryDB.getTable: 表 '%s' 不存在!" % table_name)
        return _FactorTable(name=table_name, fdb=self, sys_args=args, logger=self._QS_Logger)
    
    def renameTable(self, old_table_name, new_table_name):
        if old_table_name==new_table_name: return 0
        if old_table_name not in self._Data: raise __QS_Error__("MemoryDB.renameTable: 表: '%s' 不存在!" % old_table_name)
        if new_table_name in self._Data: raise __QS_Error__("MemoryDB.renameTable: 表 '"+new_table_name+"' 已存在!")
        self._Data[new_table_name] = self._Data.pop(old_table_name)
        if old_table_name in self._TableMeta:
            self._TableMeta[new_table_name] = self._TableMeta.pop(old_table_name)
        if old_table_name in self._FactorMeta:
            self._FactorMeta[new_table_name] = self._FactorMeta.pop(old_table_name)
        return 0
    
    def deleteTable(self, table_name):
        self._Data.pop(table_name, None)
        self._TableMeta.pop(table_name, None)
        self._FactorMeta.pop(table_name, None)
        return 0
    
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        if meta_data is not None:
            meta_data = dict(meta_data)
        else:
            meta_data = {}
        if key is not None:
            meta_data[key] = value
        if table_name not in self._Data:
            raise __QS_Error__("MemoryDB.getTable: 表 '%s' 不存在!" % table_name)
        if table_name not in self._TableMeta:
            self._TableMeta[table_name] = meta_data
        else:
            self._TableMeta[table_name].update(meta_data)
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
        if table_name not in self._Data: raise __QS_Error__("MemoryDB.renameFactor: 表 '%s' 不存在!" % (table_name, ))
        if old_factor_name not in self._Data[table_name]: raise __QS_Error__("MemoryDB.renameFactor: 表 '%s' 中不存在因子 '%s'!" % (table_name, old_factor_name))
        if new_factor_name not in self._Data[table_name]: raise __QS_Error__("MemoryDB.renameFactor: 表 '%s' 中的因子 '%s' 已存在!" % (table_name, new_factor_name))
        self._Data[table_name][new_factor_name] = self._Data[table_name].pop(old_factor_name)
        if (table_name in self._FactorMeta) and (old_factor_name in self._FactorMeta[table_name]):
            self._FactorMeta[table_name][new_factor_name] = self._FactorMeta[table_name].pop(old_factor_name)
        return 0
    
    def deleteFactor(self, table_name, factor_names):
        if table_name not in self._Data: return 0
        FactorNames = set(self._Data[table_name])
        if FactorNames.issubset(set(factor_names)):
            return self.deleteTable(table_name)
        for iFactorName in factor_names:
            self._Data[table_name].pop(iFactorName)
            if table_name in self._FactorMeta:
                self._FactorMeta[table_name].pop(iFactorName)
        return 0
    
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        iMeta = self._FactorMeta.setdefault(table_name, {}).setdefault(ifactor_name, {})
        if key is not None:
            iMeta[key] = value
        if meta_data is not None:
            for iKey in meta_data:
                iMeta[iKey] = meta_data[iKey]
        return 0
    
    def writeFactorData(self, factor_data, table_name, ifactor_name, if_exists="update", data_type=None, **kwargs):
        if table_name not in self._Data:
            self._Data[table_name] = {ifactor_name: factor_data}
            return 0
        if ifactor_name not in self._Data[table_name]:
            self._Data[table_name][ifactor_name] = factor_data
            return 0
        iOldData = self._Data[table_name][ifactor_name]
        iAllDTs = sorted(factor_data.index.union(iOldData.index))
        iAllIDs = sorted(factor_data.columns.union(iOldData.columns))
        iOldDTs = sorted(iOldData.index.difference(factor_data.index))
        iOldIDs = sorted(iOldData.columns.difference(factor_data.columns))
        iOldData, factor_data = iOldData.reindex(index=iAllDTs, columns=iAllIDs), factor_data.reindex(index=iAllDTs, columns=iAllIDs)
        if if_exists=="update":
            if iOldIDs: factor_data.loc[:, iOldIDs] = iOldData.loc[:, iOldIDs]
            if iOldDTs: factor_data.loc[iOldDTs, :] = iOldData.loc[iOldDTs, :]
        elif if_exists=="append":
            factor_data = iOldData.where(pd.notnull(iOldData), factor_data)
        elif if_exists=="update_notnull":
            factor_data = factor_data.where(pd.notnull(factor_data), iOldData)
        else:
            Msg = ("因子库 '%s' 调用方法 writeData 错误: 不支持的写入方式 '%s'!" % (self.Name, str(if_exists)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        self._Data[table_name][ifactor_name] = factor_data
        return 0
    
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._Data:
            self._Data[table_name] = {iFactorName: data.loc[iFactorName] for iFactorName in data.items}
            return 0
        for iFactor in data.items:
            self.writeFactorData(data.loc[iFactor], table_name, iFactor, if_exists=if_exists, data_type=data_type.get(iFactor, None), **kwargs)
        return 0

if __name__=="__main__":
    MDB = MemoryDB().connect()
    print(MDB.Args)
    
    df = pd.DataFrame(
        [(None, "aha"), ("中文", "aaa")], 
        index=[dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 2)], 
        columns=["000001.SZ", "000002.SZ"], dtype="O")
    MDB.writeFactorData(df, "test_table", "factor1", data_type="string")
    
    FT = MDB["test_table"]
    Data = FT.readData(factor_names=["factor1"], ids=["000001.SZ", "000003.SZ"], dts=[dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 2), dt.datetime(2022, 1, 3)])
    print(Data.iloc[0])
    
    print("===")