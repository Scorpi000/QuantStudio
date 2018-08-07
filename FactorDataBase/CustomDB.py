# coding=utf-8
import os
import uuid
import mmap
import pickle
import gc
from multiprocessing import Process, Queue

import pandas as pd
import numpy as np
from traits.api import Int, Event

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable

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
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if dts is None:
            dts = self._DateTimes
        if ids is None:
            ids = self._IDs
        Data = {}
        TableArgFactor = self._FactorDict.loc[factor_names].groupby(by=["FTID", "ArgIndex"]).groups
        for iFTID, iArgIndex in TableArgFactor:
            if iFTID==id(None):
                iFactorList, iArgList = self._TableArgDict[iFTID]
                iFactor = iFactorList[iArgIndex]
                iArgs = iArgList[iArgIndex]
                Data[iFactor] = iFactor.readFactorData(ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=iArgs)
            else:
                iFT, iArgList = self._TableArgDict[iFTID]
                iArgs = iArgList[iArgIndex]
                iFactorNames = TableArgFactor[(iFTID, iArgIndex)]
                iNameInFT = self._FactorDict["NameInFT"].loc[iFactorNames].values.tolist()
                iData = iFT.readData(factor_names=iNameInFT, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=iArgs)
                iData.items = iFactorNames
                Data.update(dict(iData))
        return pd.Panel(Data).loc[factor_names, :, :]
    # -----------遍历模式操作----------
    def start(self, dts=None, dates=None, times=None):
        self._isStarted = True
        return 0
    def move(self, idt, *args, **kwargs):
        return 0
    def end(self):
        self._isStarted = False
        return 0
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
            if iFactorName not in factor_table:
                raise QSError("指定的因子: '%s' 不存在!" % iFactorName)
            if iFactorName in self._FactorDict.index:
                raise QSError("因子: '%s' 有重名!" % iFactorName)
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

# 基于 mmap 的缓冲因子表, 如果开启遍历模式, 那么限制缓冲的时间点长度, 缓冲区里是部分数据, 如果未开启, 则调用基类提取数据的方法. 适合遍历数据读取的操作, 内存消耗小, 首次提取时间不长
def _prepareMMAPCacheData(arg):
    CacheData = {}
    CacheDates = np.array([])
    while True:
        Task = arg["Queue2SubProcess"].get()
        if Task is None:
            break
        if Task[0] is None:# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(pd.Panel(CacheData))
            DataLen = len(CacheDataByte)
            Msg = None
            if os.name=='nt':
                MMAPCacheData = mmap.mmap(-1, DataLen, tagname=arg["TagName"])# 当前MMAP缓存区
            else:
                Msg = MMAPCacheData = mmap.mmap(-1, DataLen)
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            arg["Queue2MainProcess"].put((DataLen, Msg))
            Msg = None
            gc.collect()
        else:# 准备缓冲区
            MMAPCacheData = None
            CurInd = Task[0]+arg["ForwardPeriod"]+1
            if CurInd<arg["DateNum"]:# 未到结尾处, 需要再准备缓存数据
                OldCacheDates = CacheDates
                CacheDates = arg["DateTimes"][max((0,CurInd-arg["BackwardPeriod"])):min((arg["DateNum"],CurInd+arg["ForwardPeriod"]+1))]
                NewCacheDates = np.array(list(set(CacheDates).difference(set(OldCacheDates))))
                NewCacheDates.sort()
                TableArgFactor = arg["FactorDict"].groupby(by=["FTID", "ArgIndex"]).groups
                for iFTID, iArgIndex in TableArgFactor:
                    if iFTID==id(None):
                        if NewCacheDates.shape[0]>0:
                            iFactorName = TableArgFactor[(iFTID, iArgIndex)][0]
                            iFactorList, iArgList = arg["TableArgDict"][iFTID]
                            iFactor = iFactorList[iArgIndex]
                            iArgs = iArgList[iArgIndex]
                            iData = iFactor.readData(ids=arg["IDs"], dts=NewCacheDates, args=iArgs)
                            if iFactorName not in CacheData:
                                CacheData[iFactorName] = pd.DataFrame(index=CacheDates, columns=arg["IDs"])
                            else:
                                CacheData[iFactorName] = CacheData[iFactorName].ix[CacheDates,:]
                            CacheData[iFactorName].ix[NewCacheDates, :] = iData
                    else:
                        iFT, iArgList = arg["TableArgDict"][iFTID]
                        iArgs = iArgList[iArgIndex]
                        iFactorNames = TableArgFactor[(iFTID, iArgIndex)]
                        if NewCacheDates.shape[0]>0:
                            iNameInFT = arg["FactorDict"]["NameInFT"].loc[iFactorNames].values.tolist()
                            iData = iFT.readData(factor_names=iNameInFT, ids=arg["IDs"], dts=NewCacheDates, args=iArgs)
                            iData.items = iFactorNames
                            iData = dict(iData)
                            for jFactorName in iData:
                                if jFactorName not in CacheData:
                                    CacheData[jFactorName] = pd.DataFrame(index=CacheDates, columns=arg["IDs"])
                                else:
                                    CacheData[jFactorName] = CacheData[jFactorName].ix[CacheDates,:]
                                CacheData[jFactorName].ix[NewCacheDates, :] = iData[jFactorName]
                        else:
                            for jFactorName in iData:
                                if jFactorName not in CacheData:
                                    CacheData[jFactorName] = pd.DataFrame(index=CacheDates, columns=arg["IDs"])
                                else:
                                    CacheData[jFactorName] = CacheData[jFactorName].ix[CacheDates,:]
                iData = None
    return 0
class CacheFT(CustomFT):
    ForwardPeriod = Int(600, arg_type="Integer", label="向前缓冲时点数", order=0)
    BackwardPeriod = Int(1, arg_type="Integer", label="向后缓冲时点数", order=1)
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(name=name, sys_args=sys_args)
        # 遍历模式变量
        self._isErgodicState = False
        self._CurInd = -1# 当前日期在self.Dates中的位置, 以此作为缓冲数据的依据
        self._DateNum = None# 日期数
        self._CacheDates = []# 缓冲的日期序列
        self._CacheData = None# 当前缓冲区
        self._Queue2SubProcess = None# 主进程向数据准备子进程发送消息的管道
        self._Queue2MainProcess = None# 数据准备子进程向主进程发送消息的管道
        return
    def start(self, dts=None, dates=None, times=None):
        if self._isStarted: return 0
        super().start(dts=dts, dates=dates, times=times)
        self._isErgodicState = True
        self._CurInd = -1
        self._DateNum = self._DateTimes.shape[0]
        self._CacheDates = np.array([])
        self._CacheData = None
        self._Queue2SubProcess = Queue()
        self._Queue2MainProcess = Queue()
        arg = {}
        arg['Queue2SubProcess'] = self._Queue2SubProcess
        arg['Queue2MainProcess'] = self._Queue2MainProcess
        arg["FactorDict"] = self._FactorDict
        arg["TableArgDict"] = self._TableArgDict
        arg["DateTimes"] = self._DateTimes
        arg["IDs"] = self._IDs
        arg['DateNum'] = self._DateNum
        arg['ForwardPeriod'] = self.ForwardPeriod
        arg["BackwardPeriod"] = self.BackwardPeriod
        # 准备缓冲区
        if os.name=="nt": arg["TagName"] = self._TagName = str(uuid.uuid1())
        self._CacheDataProcess = Process(target=_prepareMMAPCacheData, args=(arg,), daemon=True)
        self._CacheDataProcess.start()
        return 0
    def move(self, idt, *args, **kwargs):
        PreInd = self._CurInd
        self._CurInd = PreInd+1+self._DateTimes[PreInd+1:].index(idt)
        if (self._CurInd>-1) and ((self._CacheDates.shape[0]==0) or (self._DateTimes[self._CurInd]>self._CacheDates[-1])):# 需要读入缓冲区的数据
            self._Queue2SubProcess.put((None,None))
            DataLen,Msg = self._Queue2MainProcess.get()
            if os.name=="nt":
                MMAPCacheData = mmap.mmap(-1, DataLen, tagname=self._TagName)# 当前共享内存缓冲区
            else:
                MMAPCacheData, Msg = Msg, None
            if self._CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self._Queue2SubProcess.put((self._CurInd,None))
                self._CacheDates = self._DateTimes[max((0,self._CurInd-self.BackwardPeriod)):min((self._DateNum, self._CurInd+self.ForwardPeriod+1))]
            else:# 出现了跳跃
                LastCacheInd = (self._DateTimes.index(self._CacheDates[-1]) if self._CacheDates!=[] else self._CurInd-1)
                self._Queue2SubProcess.put((LastCacheInd+1,None))
                self._CacheDates = self._DateTimes[max((0,LastCacheInd+1-self.BackwardPeriod)):min((self._DateNum, LastCacheInd+1+self.ForwardPeriod+1))]
            MMAPCacheData.seek(0)
            self._CacheData = pickle.loads(MMAPCacheData.read(DataLen))
        return 0
    def end(self):
        if not self._isStarted: return 0
        self._CacheData = None
        self._FactorReadNum = None
        if self._isErgodicState:
            self._Queue2SubProcess.put(None)
        self._isErgodicState = False
        return super().end()
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if not self._isErgodicState:# 非遍历模式, 或者未开启缓存, 或者无缓存机制
            return super().readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        else:# 检查是否超出缓存区
            if self._CacheDates.shape[0]>0:
                BeyondCache = False
                StartDT = (self._DateTimes[0] if dts is None else dts[0])
                EndDT = (self._DateTimes[-1] if dts is None else dts[-1])
                BeyondCache = BeyondCache or ((max((start_dt, StartDT)) if start_dt is not None else StartDT)<self._CacheDates[0])
                BeyondCache = BeyondCache or ((min((end_dt, EndDT)) if end_dt is not None else EndDT)>self._CacheDates[-1])
            else:
                BeyondCache = True
            if BeyondCache:
                return super().readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        if (self._CacheData is None) or self._CacheData.empty:
            self._CacheData = super().readData(factor_names=self.FactorNames, ids=self._IDs, dts=self._CacheDates, args=args)
        Data = self._CacheData.loc[factor_names]
        if ids is None:
            ids = self._IDs
        if dts is not None:
            Data = Data.ix[:, dts, ids]
        else:
            Data = Data.ix[:, :, ids]
        if start_dt is not None:
            Data = Data.ix[:, start_dt:, :]
        if end_dt is not None:
            Data = Data.ix[:, :end_dt, :]
        return Data


# 基于 mmap 的因子缓冲因子表, 如果开启遍历模式, 那么限制缓冲的因子个数和时间点长度, 缓冲区里是因子的部分数据, 如果未开启, 则调用基类提取数据的方法. 适合遍历数据且按因子进行读取的操作, 内存消耗小, 首次提取时间不长
def _prepareMMAPFactorCacheData(arg):
    CacheData = {}
    CacheDates = np.array([])
    #print("启动缓冲"))
    while True:
        Task = arg["Queue2SubProcess"].get()
        if Task is None:
            break
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            Msg = None
            if os.name=='nt':
                MMAPCacheData = mmap.mmap(-1, DataLen, tagname=arg["TagName"])# 当前MMAP缓存区
            else:
                Msg = MMAPCacheData = mmap.mmap(-1, DataLen)
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            arg["Queue2MainProcess"].put((DataLen,Msg))
            Msg = None
            gc.collect()
            #print("装入数据:"+str(DataLen))# debug
        elif Task[0] is None:# 调整缓存区数据的任务
            NewFactors, PopFactors = Task[1]
            for iFactorName in PopFactors:
                CacheData.pop(PopFactor)
            if NewFactors:
                TableArgFactor = arg["FactorDict"].loc[NewFactors].groupby(by=["FTID", "ArgIndex"]).groups
                for iFTID, iArgIndex in TableArgFactor:
                    if iFTID==id(None):
                        iFactorName = TableArgFactor[(iFTID, iArgIndex)][0]
                        iFactorList, iArgList = arg["TableArgDict"][iFTID]
                        iFactor = iFactorList[iArgIndex]
                        iArgs = iArgList[iArgIndex]
                        if CacheDates.shape[0]>0:
                            CacheData[iFactorName] = iFactor.readData(ids=arg["IDs"], dts=CacheDates, args=iArgs)
                        else:
                            CacheData[iFactorName] = pd.DataFrame(columns=arg["IDs"])
                    else:
                        iFT, iArgList = arg["TableArgDict"][iFTID]
                        iArgs = iArgList[iArgIndex]
                        iFactorNames = TableArgFactor[(iFTID, iArgIndex)]
                        iNameInFT = arg["FactorDict"]["NameInFT"].loc[iFactorNames].values.tolist()
                        if CacheDates.shape[0]>0:
                            iData = iFT.readData(factor_names=iNameInFT, ids=arg["IDs"], dts=CacheDates, args=iArgs)
                            iData.items = iFactorNames
                            CacheData.update(dict(iData))
                        else:
                            CacheData.update({iFactorName:pd.DataFrame(columns=arg["IDs"]) for iFactorName in iFactorNames})
                iData = None
                #print("调整因子:"+str([PopFactor,NewFactor]))# debug
        else:# 准备缓冲区
            MMAPCacheData = None
            CurInd = Task[0]+arg["ForwardPeriod"]+1
            if CurInd<arg["DateNum"]:# 未到结尾处, 需要再准备缓存数据
                OldCacheDates = CacheDates
                CacheDates = arg["DateTimes"][max((0,CurInd-arg["BackwardPeriod"])):min((arg["DateNum"], CurInd+arg["ForwardPeriod"]+1))]
                NewCacheDates = np.array(list(set(CacheDates).difference(set(OldCacheDates))))
                NewCacheDates.sort()
                NewFactors = list(CacheData.keys())
                if NewFactors:
                    TableArgFactor = arg["FactorDict"].loc[NewFactors].groupby(by=["FTID", "ArgIndex"]).groups
                    for iFTID, iArgIndex in TableArgFactor:
                        if iFTID==id(None):
                            if NewCacheDates.shape[0]>0:
                                iFactorName = TableArgFactor[(iFTID, iArgIndex)][0]
                                CacheData[iFactorName] = CacheData[iFactorName].ix[CacheDates,:]
                                iFactorList, iArgList = arg["TableArgDict"][iFTID]
                                iFactor = iFactorList[iArgIndex]
                                iArgs = iArgList[iArgIndex]
                                iData = iFactor.readData(ids=arg["IDs"], dts=NewCacheDates, args=iArgs)
                                CacheData[iFactorName].ix[NewCacheDates, :] = iData
                        else:
                            iFT, iArgList = arg["TableArgDict"][iFTID]
                            iArgs = iArgList[iArgIndex]
                            iFactorNames = TableArgFactor[(iFTID, iArgIndex)]
                            if NewCacheDates.shape[0]>0:
                                iNameInFT = arg["FactorDict"]["NameInFT"].loc[iFactorNames].values.tolist()
                                iData = iFT.readData(factor_names=iNameInFT, ids=arg["IDs"], dts=NewCacheDates, args=iArgs)
                                iData.items = iFactorNames
                                iData = dict(iData)
                                for jFactorName in iData:
                                    CacheData[jFactorName] = CacheData[jFactorName].ix[CacheDates,:]
                                    CacheData[jFactorName].ix[NewCacheDates, :] = iData[jFactorName]
                            else:
                                for jFactorName in iData:
                                    CacheData[jFactorName] = CacheData[jFactorName].ix[CacheDates,:]
                    iData = None
                #print("准备因子:"+str(list(CacheData.keys())))# debug
    return 0
class FactorCacheFT(CustomFT):
    """MMAP Factor 缓冲因子表"""
    ForwardPeriod = Int(600, arg_type="Integer", label="向前缓冲时点数", order=0)
    BackwardPeriod = Int(1, arg_type="Integer", label="向后缓冲时点数", order=1)
    FactorCacheNum = Int(60, arg_type="Integer", label="最大缓冲因子数", order=2)
    def __init__(self, name, sys_args={}):
        super().__init__(name=name, sys_args=sys_args)
        # 遍历模式变量
        self._isErgodicState = False
        self._CurInd = -1# 当前日期在self.Dates中的位置, 以此作为缓冲数据的依据
        self._DateNum = None# 日期数
        self._CacheDates = []# 缓冲的日期序列
        self._CacheData = {}# 当前缓冲区
        self._CacheFactorNum = 0# 当前缓存因子个数, 小于等于 self.FactorCacheNum
        self._FactorReadNum = None# 因子读取次数, pd.Series(读取次数,index=self.FactorNames)
        self._Queue2SubProcess = None# 主进程向数据准备子进程发送消息的管道
        self._Queue2MainProcess = None# 数据准备子进程向主进程发送消息的管道
        return
    def start(self, dts=None, dates=None, times=None):
        if self._isStarted: return 0
        super().start(dts=dts, dates=dates, times=times)
        self._FactorReadNum = pd.Series(np.zeros(len(self._FactorDict)),index=self.FactorNames)
        self._isErgodicState = True
        self._CurInd = -1
        self._DateNum = self._DateTimes.shape[0]
        self._CacheDates = np.array([])
        self._CacheData = {}
        self._CacheFactorNum = 0
        self._Queue2SubProcess = Queue()
        self._Queue2MainProcess = Queue()
        arg = {}
        arg['Queue2SubProcess'] = self._Queue2SubProcess
        arg['Queue2MainProcess'] = self._Queue2MainProcess
        arg["FactorDict"] = self._FactorDict
        arg["TableArgDict"] = self._TableArgDict
        arg["DateTimes"] = self._DateTimes
        arg["IDs"] = self._IDs
        arg['DateNum'] = self._DateNum
        arg['BackwardPeriod'] = self.BackwardPeriod
        arg['ForwardPeriod'] = self.ForwardPeriod
        # 准备缓冲区
        if os.name=="nt": arg["TagName"] = self._TagName = str(uuid.uuid1())
        self._CacheDataProcess = Process(target=_prepareMMAPFactorCacheData, args=(arg,), daemon=True)
        self._CacheDataProcess.start()
        return 0
    def move(self, idt, *args, **kwargs):
        PreInd = self._CurInd
        self._CurInd = PreInd+1+self._DateTimes[PreInd+1:].index(idt)
        if (self._CurInd>-1) and ((self._CacheDates.shape[0]==0) or (self._DateTimes[self._CurInd]>self._CacheDates[-1])):# 需要读入缓冲区的数据
            self._Queue2SubProcess.put((None, None))
            DataLen,Msg = self._Queue2MainProcess.get()
            if os.name=="nt":
                MMAPCacheData = mmap.mmap(-1, DataLen, tagname=self._TagName)# 当前共享内存缓冲区
            else:
                MMAPCacheData, Msg = Msg, None
            if self._CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self._Queue2SubProcess.put((self._CurInd,None))
                self._CacheDates = self._DateTimes[max((0,self._CurInd-self.BackwardPeriod)):min((self._DateNum,self._CurInd+self.ForwardPeriod+1))]
            else:# 出现了跳跃
                LastCacheInd = (self._DateTimes.index(self._CacheDates[-1]) if self._CacheDates else self._CurInd-1)
                self._Queue2SubProcess.put((LastCacheInd+1,None))
                self._CacheDates = self._DateTimes[max((0,LastCacheInd+1-self.BackwardPeriod)):min((self._DateNum,LastCacheInd+1+self.ForwardPeriod+1))]
            MMAPCacheData.seek(0)
            self._CacheData = pickle.loads(MMAPCacheData.read(DataLen))
        return 0
    def end(self):
        if not self._isStarted: return 0
        self._CacheData = {}
        self._CacheFactorNum = 0
        self._FactorReadNum = None
        if self._isErgodicState:
            self._Queue2SubProcess.put(None)
        self._isErgodicState = False
        return super().end()
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if self._FactorReadNum is not None:
            self._FactorReadNum[factor_names] += 1
        if (not self._isErgodicState) or (self.FactorCacheNum==0):# 非遍历模式, 或者未开启缓存, 或者无缓存机制
            #print("重新提数据!")# debug
            return super().readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        else:# 检查是否超出缓存区
            if self._CacheDates.shape[0]>0:
                BeyondCache = False
                StartDT = (self._DateTimes[0] if dts is None else dts[0])
                EndDT = (self._DateTimes[-1] if dts is None else dts[-1])
                BeyondCache = BeyondCache or ((max((start_dt, StartDT)) if start_dt is not None else StartDT)<self._CacheDates[0])
                BeyondCache = BeyondCache or ((min((end_dt, EndDT)) if end_dt is not None else EndDT)>self._CacheDates[-1])
            else:
                BeyondCache = True
            if BeyondCache:
                return super().readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        Data = {}
        DataFactorNames = []
        CacheFactorNames = []
        PopFactorNames = []
        for iFactorName in factor_names:
            iFactorData = self._CacheData.get(iFactorName)
            if iFactorData is None:# 尚未进入缓存
                if self._CacheFactorNum<self.FactorCacheNum:# 当前缓存因子数小于最大缓存因子数，那么将该因子数据读入缓存
                    self._CacheFactorNum += 1
                    CacheFactorNames.append(iFactorName)
                else:# 当前缓存因子数等于最大缓存因子数，那么将检查最小读取次数的因子
                    CacheFactorReadNum = self._FactorReadNum[self._CacheData.keys()]
                    MinReadNumInd = CacheFactorReadNum.argmin()
                    if CacheFactorReadNum.loc[MinReadNumInd]<self._FactorReadNum[ifactor_name]:# 当前读取的因子的读取次数超过了缓存因子读取次数的最小值，缓存该因子数据
                        CacheFactorNames.append(iFactorName)
                        PopFactor = MinReadNumInd
                        self._CacheData.pop(PopFactor)
                        PopFactorNames.append(PopFactor)
                    else:
                        DataFactorNames.append(iFactorName)
            else:
                Data[iFactorName] = iFactorData
        if CacheFactorNames:
            iData = dict(super().readData(factor_names=CacheFactorNames, ids=self._IDs, dts=self._CacheDates, args=args))
            Data.update(iData)
            self._CacheData.update(iData)
        self._Queue2SubProcess.put((None, (CacheFactorNames, PopFactorNames)))
        Data = pd.Panel(Data)
        if Data.shape[0]>0:
            if ids is None:
                ids = self._IDs
            if dts is not None:
                Data = Data.ix[:, dts, ids]
            else:
                Data = Data.ix[:, :, ids]
            if start_dt is not None:
                Data = Data.ix[:, start_dt:, :]
            if end_dt is not None:
                Data = Data.ix[:, :end_dt, :]
        if not DataFactorNames:
            return Data
        return super().readData(factor_names=DataFactorNames, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args).join(Data)

# 基于 mmap 的 ID 缓冲的因子表, 如果开启遍历模式, 那么限制缓冲的 ID 个数和时间点长度, 缓冲区里是 ID 的部分数据, 如果未开启, 则调用基类提取数据的方法. 适合遍历数据且按 ID 进行读取的操作, 内存消耗小, 首次提取时间不长
def _prepareMMAPIDCacheData(arg):
    CacheData = {}
    CacheDates = np.array([])
    #print("启动缓冲"))
    while True:
        Task = arg["Queue2SubProcess"].get()
        if Task is None:
            break
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            Msg = None
            if os.name=="nt":
                MMAPCacheData = mmap.mmap(-1,DataLen,tagname=arg["TagName"])# 当前MMAP缓存区
            else:
                Msg = MMAPCacheData = mmap.mmap(-1,DataLen)
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            arg["Queue2MainProcess"].put((DataLen,Msg))
            Msg = None
            gc.collect()
            #print("装入数据:"+str(DataLen))# debug
        elif Task[0] is None:# 调整缓存区数据的任务
            NewID,PopID = Task[1]
            if PopID is not None:# 用新 ID 数据替换旧 ID
                CacheData.pop(PopID)
            Data = {}
            TableArgFactor = arg["FactorDict"].groupby(by=["FTID", "ArgIndex"]).groups
            for iFTID, iArgIndex in TableArgFactor:
                if iFTID==id(None):
                    iFactorName = TableArgFactor[(iFTID, iArgIndex)][0]
                    iFactorList, iArgList = arg["TableArgDict"][iFTID]
                    iFactor = iFactorList[iArgIndex]
                    iArgs = iArgList[iArgIndex]
                    if CacheDates.shape[0]>0:
                        Data[iFactorName] = iFactor.readData(ids=[NewID], dts=CacheDates, args=iArgs).iloc[:, 0]
                    else:
                        Data[iFactorName] = pd.Series([])
                else:
                    iFT, iArgList = arg["TableArgDict"][iFTID]
                    iArgs = iArgList[iArgIndex]
                    iFactorNames = TableArgFactor[(iFTID, iArgIndex)]
                    iNameInFT = arg["FactorDict"]["NameInFT"].loc[iFactorNames].values.tolist()
                    if CacheDates.shape[0]>0:
                        iData = iFT.readData(factor_names=iNameInFT, ids=[NewID], dts=CacheDates, args=iArgs).iloc[:, :, 0]
                        iData.columns = iFactorNames
                    else:
                        iData = pd.DataFrame(columns=iFactorNames)
                    Data.update(dict(iData))
            CacheData[NewID] = pd.DataFrame(Data).ix[:, arg["FactorDict"].index]
            Data = None
            #print("调整 ID:"+str([PopID,NewID]))# debug
        else:# 准备缓冲区
            MMAPCacheData = None
            CurInd = Task[0]+arg["ForwardPeriod"]+1
            if CurInd<arg["DateNum"]:# 未到结尾处, 需要再准备缓存数据
                OldCacheDates = CacheDates
                CacheDates = arg["DateTimes"][max((0,CurInd-arg["BackwardPeriod"])):min((arg["DateNum"],CurInd+arg["ForwardPeriod"]+1))]
                NewCacheDates = np.array(list(set(CacheDates).difference(set(OldCacheDates))))
                NewCacheDates.sort()
                TableArgFactor = arg["FactorDict"].groupby(by=["FTID", "ArgIndex"]).groups
                IDs = list(CacheData.keys())
                if IDs:
                    Data = {}
                    for iFTID, iArgIndex in TableArgFactor:
                        if iFTID==id(None):
                            if NewCacheDates.shape[0]>0:
                                iFactorName = TableArgFactor[(iFTID, iArgIndex)][0]
                                iFactorList, iArgList = arg["TableArgDict"][iFTID]
                                iFactor = iFactorList[iArgIndex]
                                iArgs = iArgList[iArgIndex]
                                iData = iFactor.readData(ids=IDs, dts=NewCacheDates, args=iArgs)
                            else:
                                iData = pd.DataFrame(columns=IDs)
                            Data[iFactorName] = iData
                        else:
                            iFT, iArgList = arg["TableArgDict"][iFTID]
                            iArgs = iArgList[iArgIndex]
                            iFactorNames = TableArgFactor[(iFTID, iArgIndex)]
                            iNameInFT = arg["FactorDict"]["NameInFT"].loc[iFactorNames].values.tolist()
                            if NewCacheDates.shape[0]>0:
                                iData = iFT.readData(factor_names=iNameInFT, ids=IDs, dts=NewCacheDates, args=iArgs)
                                iData.items = iFactorNames
                                Data.update(dict(iData))
                            else:
                                Data.update({iFactorName:pd.DataFrame(columns=IDs) for iFactorName in iFactorNames})
                    iData = None
                    Data = dict(pd.Panel(Data).loc[arg["FactorDict"].index].swapaxes(0,2))
                    for iID in CacheData:
                        CacheData[iID] = CacheData[iID].ix[CacheDates,:]
                        CacheData[iID].ix[NewCacheDates, :] = Data[iID]
                    Data = None
                    #print("准备 ID:"+str(list(CacheData.keys())))# debug
    return 0
class IDCacheFT(CustomFT):
    """MMAP ID 缓冲因子表"""
    ForwardPeriod = Int(600, arg_type="Integer", label="向前缓冲时点数", order=0)
    BackwardPeriod = Int(1, arg_type="Integer", label="向后缓冲时点数", order=1)
    IDCacheNum = Int(60, arg_type="Integer", label="最大缓冲ID数", order=2)
    def __init__(self, name, sys_args={}):
        super().__init__(name=name, sys_args=sys_args)
        # 遍历模式变量
        self._isErgodicState = False
        self._CurInd = -1# 当前日期在self.Dates中的位置, 以此作为缓冲数据的依据
        self._DateNum = None# 日期数
        self._CacheDates = []# 缓冲的日期序列
        self._CacheData = {}# 当前缓冲区
        self._CacheIDNum = 0# 当前缓存 ID 个数, 小于等于 self.IDCacheNum
        self._IDReadNum = None# ID 读取次数, pd.Series(读取次数, index=self.IDs)
        self._Queue2SubProcess = None# 主进程向数据准备子进程发送消息的管道
        self._Queue2MainProcess = None# 数据准备子进程向主进程发送消息的管道
        return
    def start(self, dts=None, dates=None, times=None):
        if self._isStarted: return 0
        super().start(dts=dts, dates=dates, times=times)
        self._IDReadNum = pd.Series(np.zeros(len(self._IDs)),index=self._IDs)
        self._isErgodicState = True
        self._CurInd = -1
        self._DateNum = self._DateTimes.shape[0]
        self._CacheDates = np.array([])
        self._CacheData = {}
        self._CacheIDNum = 0
        self._Queue2SubProcess = Queue()
        self._Queue2MainProcess = Queue()
        arg = {}
        arg['Queue2SubProcess'] = self._Queue2SubProcess
        arg['Queue2MainProcess'] = self._Queue2MainProcess
        arg["FactorDict"] = self._FactorDict
        arg["TableArgDict"] = self._TableArgDict
        arg["DateTimes"] = self._DateTimes
        arg['DateNum'] = self._DateNum
        arg['ForwardPeriod'] = self.ForwardPeriod
        arg['BackwardPeriod'] = self.BackwardPeriod
        # 准备缓冲区
        if os.name=="nt": arg["TagName"] = self._TagName = str(uuid.uuid1())
        self._CacheDataProcess = Process(target=_prepareMMAPIDCacheData,args=(arg,),daemon=True)
        self._CacheDataProcess.start()
        return 0
    def move(self, idt, *args, **kwargs):
        PreInd = self._CurInd
        self._CurInd = PreInd+1+self._DateTimes[PreInd+1:].index(idt)
        if (self._CurInd>-1) and ((self._CacheDates.shape[0]==0) or (self._DateTimes[self._CurInd]>self._CacheDates[-1])):# 需要读入缓冲区的数据
            self._Queue2SubProcess.put((None,None))
            DataLen,Msg = self._Queue2MainProcess.get()
            if os.name=="nt":
                MMAPCacheData = mmap.mmap(-1, DataLen, tagname=self._TagName)# 当前共享内存缓冲区
            else:
                MMAPCacheData, Msg = Msg, None
            if self._CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self._Queue2SubProcess.put((self._CurInd,None))
                self._CacheDates = self._DateTimes[max((0,self._CurInd-self.BackwardPeriod)):min((self._DateNum, self._CurInd+self.ForwardPeriod+1))]
            else:# 出现了跳跃
                LastCacheInd = (self._DateTimes.index(self._CacheDates[-1]) if self._CacheDates else self._CurInd-1)
                self._Queue2SubProcess.put((LastCacheInd+1,None))
                self._CacheDates = self._DateTimes[max((0,LastCacheInd+1-self.BackwardPeriod)):min((self._DateNum, LastCacheInd+1+self.ForwardPeriod+1))]
            MMAPCacheData.seek(0)
            self._CacheData = pickle.loads(MMAPCacheData.read(DataLen))
        return 0
    def end(self):
        if not self._isStarted: return 0
        self._CacheData = {}
        self._CacheFactorNum = 0
        self._IDReadNum = None
        if self._isErgodicState:
            self._Queue2SubProcess.put(None)
        self._isErgodicState = False
        return super().end()
    def readIDData(self, iid, factor_names=None, dts=None, start_dt=None, end_dt=None, args={}):
        if self._IDReadNum is not None:
            self._IDReadNum[iid] += 1
        if (not self._isErgodicState) or (self.IDCacheNum==0):# 非遍历模式, 或者未开启缓存, 或者无缓存机制
            #print("重新提数据!")# debug
            return super().readData(factor_names=factor_names, ids=[iid], dts=dts, start_dt=start_dt, end_dt=end_dt, args=args).iloc[:, :, 0]
        else:# 检查是否超出缓存区
            if self._CacheDates.shape[0]>0:
                BeyondCache = False
                StartDT = (self._DateTimes[0] if dts is None else dts[0])
                EndDT = (self._DateTimes[-1] if dts is None else dts[-1])
                BeyondCache = BeyondCache or ((max((start_dt, StartDT)) if start_dt is not None else StartDT)<self._CacheDates[0])
                BeyondCache = BeyondCache or ((min((end_dt, EndDT)) if end_dt is not None else EndDT)>self._CacheDates[-1])
            else:
                BeyondCache = True
            if BeyondCache:
                return super().readData(factor_names=factor_names, ids=[iid], dts=dts, start_dt=start_dt, end_dt=end_dt, args=args).iloc[:, :, 0]
        IDData = self._CacheData.get(iid)
        if IDData is None:# 尚未进入缓存
            if self._CacheIDNum<self.IDCacheNum:# 当前缓存 ID 数小于最大缓存 ID 数，那么将该 ID 数据读入缓存
                self._CacheIDNum += 1
                IDData = super().readData(factor_names=None, ids=[iid], dts=self._CacheDates, args=args).iloc[:, :, 0]
                self._CacheData[iid] = IDData
                self._Queue2SubProcess.put((None,(iid,None)))
            else:# 当前缓存 ID 数等于最大缓存 ID 数，那么将检查最小读取次数的 ID
                CacheIDReadNum = self._IDReadNum[self._CacheData.keys()]
                MinReadNumInd = CacheIDReadNum.argmin()
                if CacheIDReadNum.loc[MinReadNumInd]<self._IDReadNum[iid]:# 当前读取的 ID 的读取次数超过了缓存 ID 读取次数的最小值，缓存该 ID 数据
                    IDData = super().readData(factor_names=None, ids=[iid], dts=self._CacheDates, args=args).iloc[:, :, 0]
                    PopID = MinReadNumInd
                    self._CacheData.pop(PopID)
                    self._CacheData[iid] = IDData
                    self._Queue2SubProcess.put((None,(iid, PopID)))
                else:# 当前读取的 ID 的读取次数没有超过缓存 ID 读取次数的最小值, 放弃缓存该 ID 数据
                    return super().readData(factor_names=factor_names, ids=[iid], dts=dts, start_dt=start_dt, end_dt=end_dt, args=args).iloc[:, :, 0]
        if factor_names is not None:
            IDData = IDData.ix[:, factor_names]
        if dts is not None:
            IDData = IDData.ix[dts, :]
        if start_dt is not None:
            IDData = IDData.ix[start_dt:]
        if end_dt is not None:
            IDData = IDData.ix[:end_dt]
        return IDData
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if ids is None:
            ids = self.getID()
        Data = {}
        for iID in ids:
            Data[iID] = self.readIDData(iID, factor_names=factor_names, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        Data = pd.Panel(Data)
        return Data.swapaxes(0, 2)


if __name__=='__main__':
    import time
    import datetime as dt
    
    from QuantStudio.FactorDataBase.HDF5DB import HDF5DB
    
    # 创建因子数据库
    MainDB = HDF5DB()
    MainDB.connect()
    FT = MainDB.getTable("ElementaryFactor")
    # 创建自定义的因子表
    MainFT = FactorCacheFT("MainFT")
    #MainFT = IDCacheFT("MainFT")
    #MainFT = CacheFT("MainFT")
    MainFT.addFactors(factor_table=FT, factor_names=["复权收盘价"], args={})
    MainFT.setDateTime(FT.getDateTime(ifactor_name="复权收盘价", start_dt=dt.datetime(2014,1,1), end_dt=dt.datetime(2018,1,1)))
    MainFT.setID(["000001.SZ","600000.SH"])
    #Data = MainFT.readData()
    StartT = time.clock()
    MainFT.start()
    for iDateTime in MainFT.getDateTime():
        MainFT.move(iDateTime)
        iData = MainFT.readData(dts=[iDateTime]).iloc[:, 0, :]
        print(iDateTime)
    MainFT.end()
    print(time.clock()-StartT)
    pass