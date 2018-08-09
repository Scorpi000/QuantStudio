# coding=utf-8
import os
import uuid
import mmap
import pickle
import gc
from multiprocessing import Process, Queue

import pandas as pd
import numpy as np
from traits.api import Int, Event, Enum, Any

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable, Factor

# 自定义因子库
class CustomDB(FactorDB):
    """自定义因子库"""
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = name
        self._FTs = {}
    # -------------------------------表的操作---------------------------------
    # 表名, 返回: array([表名])
    @property
    def TableNames(self):
        return sorted(self._FTs.keys())
    # 返回因子表对象
    def getTable(self, table_name, args={}):
        return self._FTs[table_name]
    # 添加因子表
    def addFactorTable(self, ft):
        if ft.Name in self._FTs: raise __QS_Error__("因子表中有重名!")
        self._FTs[ft.Name] = ft
    # 删除因子表
    def popFactorTable(self, table_name):
        if table_name not in self._FTs: return None
        return self._FTs.pop(table_name)

# 自定义因子表
class CustomFT(FactorTable):
    """自定义因子表"""
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = name# 因子表名称
        self._DateTimes = []# 数据源可提取的最长时点序列，[datetime.datetime]
        self._IDs = []# 数据源可提取的最长ID序列，['600000.SH']
        self._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))# 数据源中因子的来源信息
        self._TableArgDict = {}# 数据源中的表和参数信息, {FTID : (FT, [args]), None : ([Factor,] [args])}
        self._IDFilterStr = None# ID 过滤条件字符串, "@收益率>0", 给定日期, 数据源的 getID 将返回过滤后的 ID 序列
        self._CompiledIDFilter = {}# 编译过的过滤条件字符串以及对应的因子列表, {条件字符串: (编译后的条件字符串,[因子])}
        self._isStarted = False# 数据源是否启动
        return
    @property
    def FactorNames(self):
        return self._FactorDict.index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            return self._FactorDict["DataType"].ix[factor_names]
        MetaData = {}
        for iFactorName in factor_names:
            iFTID = self._FactorDict.loc[iFactorName, "FTID"]
            iArgIndex = self._FactorDict.loc[iFactorName, "ArgIndex"]
            if iFTID==id(None):
                iFactor = self._TableArgDict[iFTID][0][iArgIndex]
                MetaData[iFactorName] = iFactor.getMetaData(key=key)
            else:
                iFT = self._TableArgDict[iFTID][0]
                iNameInFT = self._FactorDict["NameInFT"].loc[iFactorName]
                MetaData[iFactorName] = FT.getFactorMetaData(factor_names=[iNameInFT], key=key).ix[iNameInFT]
        if key is None:
            return pd.DataFrame(MetaData)
        else:
            return pd.Series(MetaData)
    def getFactor(self, ifactor_name, args={}):
        iFTID = self._FactorDict.loc[ifactor_name, "FTID"]
        iArgIndex = self._FactorDict.loc[ifactor_name, "ArgIndex"]
        if iFTID==id(None):
            return self._TableArgDict[iFTID][0][iArgIndex]
        else:
            iFT = self._TableArgDict[iFTID][0]
            iNameInFT = self._FactorDict["NameInFT"].loc[ifactor_name]
            return iFT.getFactor(ifactor_name=iNameInFT, args=args)
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DateTimes = self._DateTimes
        if start_dt is not None:
            DateTimes = DateTimes[DateTimes>=start_dt]
        if end_dt is not None:
            DateTimes = DateTimes[DateTimes<=end_dt]
        return DateTimes
    def getID(self, ifactor_name=None, idt=None, args={}):
        return self._IDs
    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None: ids = self.getID(idt=idt, args=args)
        OldIDFilterStr = self.setIDFilter(id_filter_str)
        if self._IDFilterStr is None:
            self._IDFilterStr = OldIDFilterStr
            return pd.Series(True, index=ids)
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        self._IDFilterStr = OldIDFilterStr
        return eval(CompiledFilterStr)
    def getFilteredID(self, idt, id_filter_str=None, args={}):
        if not id_filter_str: return self.getID(idt=idt, args=args)
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        self._IDFilterStr = OldIDFilterStr
        return eval("temp["+CompiledIDFilterStr+"].index.tolist()")
    def __QS_readData__(self, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if dts is None: dts = self._DateTimes
        if ids is None: ids = self._IDs
        Data = {}
        TableArgFactor = self._FactorDict.loc[factor_names].groupby(by=["FTID", "ArgIndex"]).groups
        for iFTID, iArgIndex in TableArgFactor:
            if iFTID==id(None):
                iFactorList, iArgList = self._TableArgDict[iFTID]
                iFactor = iFactorList[iArgIndex]
                iArgs = iArgList[iArgIndex]
                Data[iFactor] = iFactor.readData(ids=ids, dts=dts, args=iArgs)
            else:
                iFT, iArgList = self._TableArgDict[iFTID]
                iArgs = iArgList[iArgIndex]
                iFactorNames = TableArgFactor[(iFTID, iArgIndex)]
                iNameInFT = self._FactorDict["NameInFT"].loc[iFactorNames].values.tolist()
                iData = iFT.readData(factor_names=iNameInFT, ids=ids, dts=dts, args=iArgs)
                iData.items = iFactorNames
                Data.update(dict(iData))
        return pd.Panel(Data).loc[factor_names, :, :]
    # ---------------新的接口------------------
    # 添加因子, factor_list: 因子对象列表
    def addFactors(self, factor_list=[], factor_table=None, factor_names=None, args={}):
        for iFactor in factor_list:
            if iFactor.Name in self._FactorDict.index:
                raise QSError("因子: '%s' 有重名!" % iFactor.Name)
            iFT = iFactor.FactorTable
            iFTID = id(iFT)
            iDataType = iFactor.getMetaData(key="DataType")
            if iFT is None:
                iFactorList, iArgList = self._TableArgDict.get(iFTID, ([], []))
                self._FactorDict.loc[iFactor.Name] = (iFTID, len(iArgList), None, iDataType)
                iFactorList.append(iFactor)
                iArgList.append(args)
                self._TableArgDict[iFTID] = (iFactorList, iArgList)
            else:
                iFT, iArgList = self._TableArgDict.get(iFTID, (iFT, []))
                iArgIndex = (len(iArgList) if args not in iArgList else iArgList.index(args))
                self._FactorDict.loc[iFactor.Name] = (iFTID, iArgIndex, iFactor._NameInFT, iDataType)
                iArgList.append(args)
                self._TableArgDict[iFTID] = (iFT, iArgList)
        if factor_table is None:
            return 0
        if factor_names is None:
            factor_names = factor_table.FactorNames
        iFTID = id(factor_table)
        factor_table, iArgList = self._TableArgDict.get(iFTID, (factor_table, []))
        if args in iArgList:
            iArgIndex = iArgList.index(args)
        else:
            iArgIndex = len(iArgList)
            iArgList.append(args)
        DataTypes = factor_table.getFactorMetaData(factor_names, key="DataType")
        for iFactorName in factor_names:
            if iFactorName not in factor_table.FactorNames: raise __QS_Error__("指定的因子: '%s' 不存在!" % iFactorName)
            if iFactorName in self._FactorDict.index: raise __QS_Error__("因子: '%s' 有重名!" % iFactorName)
            iDataType = DataTypes[iFactorName]
            self._FactorDict.loc[iFactorName] = (iFTID, iArgIndex, iFactorName, iDataType)
        self._TableArgDict[iFTID] = (factor_table, iArgList)
        self._FactorDict["ArgIndex"] = self._FactorDict["ArgIndex"].astype(np.int64)
        self._FactorDict["FTID"] = self._FactorDict["FTID"].astype(np.int64)
        return 0
    # 删除因子, factor_names = None 表示删除所有因子
    def deleteFactors(self, factor_names=None):
        if factor_names is None:
            factor_names = self.FactorNames
        for iFactorName in factor_names:
            if iFactorName not in self._FactorDict.index:
                continue
            iFTID = self._FactorDict.loc[iFactorName, "FTID"]
            iArgIndex = self._FactorDict.loc[iFactorName, "ArgIndex"]
            if iFTID==id(None):
                iFactorList, iArgList = self._TableArgDict[iFTID]
                iFactorList.pop(iArgIndex)
                iArgList.pop(iArgIndex)
            else:
                iFT, iArgList = self._TableArgDict[iFTID]
                iArgList.pop(iArgIndex)
            if not iArgList:
                self._TableArgDict.pop(iFTID)
        self._FactorDict = self._FactorDict.loc[sorted(set(self._FactorDict.index).difference(set(factor_names)))]
        return 0
    # 重命名因子
    def renameFactor(self, factor_name, new_factor_name):
        if factor_name not in self._FactorDict.index:
            raise QSError("因子: '%s' 不存在!" % factor_name)
        if (new_factor_name!=factor_name) and (new_factor_name in self._FactorDict.index):
            raise QSError("因子: '%s' 有重名!" % new_factor_name)
        FactorNames = list(self._FactorDict.index)
        FactorNames[FactorNames.index(factor_name)] = new_factor_name
        self._FactorDict.index = FactorNames
        return 0
    # 设置时间点序列
    def setDateTime(self, dts):
        self._DateTimes = sorted(dts)
    # 设置 ID 序列
    def setID(self, ids):
        self._IDs = sorted(ids)
    # ID 过滤条件
    @property
    def IDFilterStr(self):
        return self._IDFilterStr
    # 设置 ID 过滤条件, id_filter_str, '@收益率$>0'
    def setIDFilter(self, id_filter_str):
        OldIDFilterStr = self._IDFilterStr
        if id_filter_str is None:
            self._IDFilterStr = None
            return OldIDFilterStr
        elif (not isinstance(id_filter_str, str)) or (id_filter_str==""):
            raise QSError("条件字符串必须为非空字符串或者 None!")
        CompiledIDFilter = self._CompiledIDFilter.get(id_filter_str, None)
        if CompiledIDFilter is not None:# 该条件已经编译过
            self._IDFilterStr = id_filter_str
            return OldIDFilterStr
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None:
            raise QSError("条件字符串有误!")
        self._IDFilterStr = id_filter_str
        self._CompiledIDFilter[id_filter_str] = (CompiledIDFilterStr, IDFilterFactors)
        return OldIDFilterStr

if __name__=='__main__':
    import time
    import datetime as dt
    
    from QuantStudio.FactorDataBase.HDF5DB import HDF5DB
    
    # 创建因子数据库
    MainDB = HDF5DB()
    MainDB.connect()
    FT = MainDB.getTable("ElementaryFactor")
    # 创建自定义的因子表
    MainFT = CustomFT("MainFT")
    MainFT.addFactors(factor_table=FT, factor_names=["复权收盘价"], args={})
    MainFT.setDateTime(FT.getDateTime(ifactor_name="复权收盘价", start_dt=dt.datetime(2014,1,1), end_dt=dt.datetime(2018,1,1)))
    MainFT.setID(["000001.SZ", "600000.SH"])
    MainFT.ErgodicMode.CacheMode = "ID"
    StartT = time.process_time()
    MainFT.start()
    for iDateTime in MainFT.getDateTime():
        MainFT.move(iDateTime)
        iData = MainFT.readData(dts=[iDateTime]).iloc[:, 0, :]
        print(iDateTime)
    MainFT.end()
    print(time.process_time()-StartT)
    pass