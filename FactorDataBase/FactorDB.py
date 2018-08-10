# coding=utf-8
import time
import os
import uuid
import mmap
import pickle
import gc
import shelve
import datetime as dt
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
from tqdm import tqdm
from progressbar import ProgressBar
from traits.api import Instance, Str, File, List, Int, Bool, Directory, Enum, ListStr

from QuantStudio import __QS_Object__, __QS_Error__, __QS_CachePath__, __QS_CacheLock__
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio.Tools.AuxiliaryFun import genAvailableName, partitionList, startMultiProcess
from QuantStudio.Tools.FileFun import listDirDir

def _adjustDateTime(data, dts=None, fillna=False, **kwargs):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if dts is not None:
            if fillna:
                AllDTs = data.index.union(set(dts))
                AllDTs = AllDTs.sort_values()
                data = data.ix[AllDTs]
                data = data.fillna(**kwargs)
            data = data.ix[dts]
    else:
        if dts is not None:
            if fillna:
                AllDTs = data.major_axis.union(set(dts))
                AllDTs = AllDTs.sort_values()
                data = data.ix[:, AllDTs, :]
                data = data.fillna(axis=1, **kwargs)
            data = data.ix[:, dts, :]
    return data

# 因子库, 只读, 接口类
# 数据库由若干张因子表组成
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorDB(__QS_Object__):
    """因子库"""
    Name = Str("因子库")
    # ------------------------------数据源操作---------------------------------
    # 链接到数据库
    def connect(self):
        return 0
    # 断开到数据库的链接
    def disconnect(self):
        return 0
    # 检查数据库是否可用
    def isAvailable(self):
        return True
    # -------------------------------表的操作---------------------------------
    # 表名, 返回: array([表名])
    @property
    def TableNames(self):
        return []
    # 返回因子表对象
    def getTable(self, table_name, args={}):
        return None

# 支持写入的因子库, 接口类
class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""
    # -------------------------------表的操作---------------------------------
    # 重命名表. 必须具体化
    def renameTable(self, old_table_name, new_table_name):
        return 0
    # 删除表. 必须具体化
    def deleteTable(self, table_name):
        return 0
    # 设置表的元数据. 必须具体化
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        return 0
    # --------------------------------因子操作-----------------------------------
    # 对一张表的因子进行重命名. 必须具体化
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        return 0
    # 删除一张表中的某些因子. 必须具体化
    def deleteFactor(self, table_name, factor_names):
        return 0
    # 设置因子的元数据. 必须具体化
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        return 0
    # 写入数据, if_exists: append, update, replace, skip. 必须具体化
    def writeData(self, data, table_name, if_exists='append', **kwargs):
        return 0
    # -------------------------------数据变换------------------------------------
    # 复制因子, 并不删除原来的因子
    def copyFactor(self, target_table, table_name, factor_names=None, if_exists='append', args={}):
        FT = self.getTable(table_name)
        if factor_names is None:
            factor_names = FT.FactorNames
        Data = FT.readData(factor_names=factor_names, args=args)
        return self.writeData(Data, target_table, if_exists=if_exists)
    # 时间平移, 沿着时间轴将所有数据纵向移动 lag 期, lag>0 向前移动, lag<0 向后移动, 空出来的地方填 nan
    def offsetDateTime(self, lag, table_name, factor_names=None, args={}):
        if lag==0:
            return 0
        FT = self.getTable(table_name)
        Data = FT.readData(factor_names=factor_names, args=args)
        if lag>0:
            Data.iloc[:,lag:,:] = Data.iloc[:,:-lag,:].values
            Data.iloc[:,:lag,:] = None
        elif lag<0:
            Data.iloc[:,:lag,:] = Data.iloc[:,-lag:,:].values
            Data.iloc[:,:lag,:] = None
        self.writeData(Data, table_name, if_exists='replace')
        return 0
    # 数据变换, 对原来的时间和ID序列通过某种变换函数得到新的时间序列和ID序列, 调整数据
    def changeData(self, table_name, factor_names=None, ids=None, dts=None, args={}):
        if dts is None:
            return 0
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        self.writeData(Data, table_name, if_exists='replace')
        return 0
    # 填充缺失值
    def fillNA(self, filled_value, table_name, factor_names=None, ids=None, dts=None, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        Data.fillna(filled_value, inplace=True)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 替换数据
    def replaceData(self, old_value, new_value, table_name, factor_names=None, ids=None, dts=None, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        Data = Data.where(Data!=old_value, new_value)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 压缩数据
    def compressData(self, table_name=None, factor_names=None):
        return 0

# 因子表的遍历模式参数对象
class _ErgodicMode(__QS_Object__):
    """遍历模式"""
    ForwardPeriod = Int(600, arg_type="Integer", label="向前缓冲时点数", order=0)
    BackwardPeriod = Int(1, arg_type="Integer", label="向后缓冲时点数", order=1)
    CacheMode = Enum("因子", "ID", arg_type="SingleOption", label="缓冲模式", order=2)
    MaxFactorCacheNum = Int(60, arg_type="Integer", label="最大缓冲因子数", order=3)
    MaxIDCacheNum = Int(10000, arg_type="Integer", label="最大缓冲ID数", order=4)
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self._isStarted = False
    def __getstate__(self):
        state = self.__dict__.copy()
        if "_CacheDataProcess" in state: state["_CacheDataProcess"] = None
        return state
# 基于 mmap 的缓冲数据, 如果开启遍历模式, 那么限制缓冲的因子个数, ID 个数, 时间点长度, 缓冲区里是因子的部分数据
def _prepareMMAPFactorCacheData(ft):
    CacheData, CacheDTs, Msg, MMAPCacheData, DTNum = {}, [], None, None, len(ft.ErgodicMode._DateTimes)
    while True:
        Task = ft.ErgodicMode._Queue2SubProcess.get()# 获取任务
        if Task is None: break# 结束进程
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            if os.name=='nt': MMAPCacheData = mmap.mmap(-1, DataLen, tagname=ft.ErgodicMode._TagName)
            else: Msg = MMAPCacheData = mmap.mmap(-1, DataLen)
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            ft.ErgodicMode._Queue2MainProcess.put((DataLen, Msg))
            gc.collect()
        elif Task[0] is None:# 调整缓存区数据
            NewFactors, PopFactors = Task[1]
            for iFactorName in PopFactors: CacheData.pop(iFactorName)
            if NewFactors: CacheData.update(dict(ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=NewFactors, ids=ft.ErgodicMode._IDs, dts=CacheDTs), factor_names=NewFactors, ids=ft.ErgodicMode._IDs, dts=CacheDTs)))
        else:# 准备缓冲区
            Msg = MMAPCacheData = None# 这句话必须保留...诡异
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            if CurInd < DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[max((0, CurInd-ft.ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+ft.ErgodicMode.ForwardPeriod+1))].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    CacheFactorNames = list(CacheData.keys())
                    NewCacheData = ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=CacheFactorNames, ids=ft.ErgodicMode._IDs, dts=NewCacheDTs), factor_names=CacheFactorNames, ids=ft.ErgodicMode._IDs, dts=NewCacheDTs)
                    for iFactorName in CacheData:
                        CacheData[iFactorName] = CacheData[iFactorName].ix[CacheDTs, :]
                        CacheData[iFactorName].ix[NewCacheDTs, :] = NewCacheData[iFactorName]
                    NewCacheData = None
    return 0
# 基于 mmap 的 ID 缓冲的因子表, 如果开启遍历模式, 那么限制缓冲的 ID 个数和时间点长度, 缓冲区里是 ID 的部分数据
def _prepareMMAPIDCacheData(ft):
    CacheData, CacheDTs, Msg, MMAPCacheData, DTNum = {}, [], None, None, len(ft.ErgodicMode._DateTimes)
    while True:
        Task = ft.ErgodicMode._Queue2SubProcess.get()# 获取任务
        if Task is None: break# 结束进程
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            if os.name=='nt': MMAPCacheData = mmap.mmap(-1, DataLen, tagname=ft.ErgodicMode._TagName)
            else: Msg = MMAPCacheData = mmap.mmap(-1, DataLen)
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            ft.ErgodicMode._Queue2MainProcess.put((DataLen, Msg))
            gc.collect()
        elif Task[0] is None:# 调整缓存区数据
            NewID, PopID = Task[1]
            if PopID: CacheData.pop(PopID)# 用新 ID 数据替换旧 ID
            if NewID: CacheData[NewID] = ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=ft.FactorNames, ids=[NewID], dts=CacheDTs), factor_names=ft.FactorNames, ids=[NewID], dts=CacheDTs).iloc[:, :, 0]
        else:# 准备缓冲区
            Msg = MMAPCacheData = None# 这句话必须保留...诡异
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            if CurInd<DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[max((0, CurInd-ft.ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+ft.ErgodicMode.ForwardPeriod+1))].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    CacheIDs = list(CacheData.keys())
                    NewCacheData = ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=ft.FactorNames, ids=CacheIDs, dts=NewCacheDTs), factor_names=ft.FactorNames, ids=CacheIDs, dts=NewCacheDTs)
                    for iID in CacheData:
                        CacheData[iID] = CacheData[iID].ix[CacheDTs, :]
                        CacheData[iID].ix[NewCacheDTs, :] = NewCacheData.loc[:, :, iID]
                    NewCacheData = None
    return 0
# 因子表的运算模式参数对象
class _OperationMode(__QS_Object__):
    """运算模式"""
    DateTimes = List(dt.datetime, arg_type="DateTimeList", label="运算时点", order=0)
    IDs = List(str, arg_type="IDList", label="运算ID", order=1)
    FactorNames = ListStr(arg_type="MultiOption", label="运算因子", order=2, option_range=())
    SubProcessNum = Int(0, arg_type="Integer", label="子进程数", order=3)
    DTRuler = List(dt.datetime, arg_type="DateTimeList", label="时点标尺", order=4)
    def __init__(self, ft, sys_args={}, **kwargs):
        self._FT = ft
        self._isStarted = False
        super().__init__(sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        self.add_trait("FactorNames", ListStr(arg_type="MultiOption", label="运算因子", order=2, option_range=tuple(self._FT.FactorNames)))
# 因子表准备子进程
def _prepare(args):
    if "Sub2MainQueue" not in args:# 运行模式为串行
        nTask = len(args['GroupInfo'])
        for i in tqdm(range(nTask)):
            iFT, iFactorNames, iDTs, iArgs = args['GroupInfo'][i]
            iRawData = iFT.__QS_prepareRawData__(iFactorNames, args['FT'].OperationMode.IDs, iDTs, iArgs)
            args['FT']._QS_saveRawData(iRawData)
    else:# 运行模式为并行
        for iFT, iFactorNames, iDTs, iArgs in args['GroupInfo']:
            iRawData = iFT.__QS_prepareRawData__(iFactorNames, args['FT'].OperationMode.IDs, iDTs, iArgs)
            args['FT']._QS_saveRawData(iRawData)
            args['Sub2MainQueue'].put((args["PID"], 1, None))
    return 0
# 因子表运算子进程
def _calculate(args):
    FT = args["FT"]
    if FT.ExternArgs['运行模式']=='串行':# 运行模式为串行
        nTask = len(FT.Factors)
        with ProgressBar(max_value=nTask) as ProgBar:
            for i,iFactor in enumerate(FT.Factors):
                iData = iFactor.getData(FT.Dates,pids=[qs_env.PID],extern_args=FT.ExternArgs,to_save=True)
                iFactorDB,iTableName,iUpdateMode = FT.SaveInfo[iFactor.FactorName]
                getattr(qs_env,iFactorDB).writeFactorData(iData,FT.TempSaveInfo[iFactorDB],iFactor.FactorName,if_exists='merge',data_type=iFactor.FactorDataType)
                iData = None
                ProgBar.update(i+1)
    else:
        for i,iFactor in enumerate(FT.Factors):
            iData = iFactor.getData(FT.Dates,pids=[qs_env.PID],extern_args=FT.ExternArgs,to_save=True)
            iFactorDB,iTableName,iUpdateMode = FT.SaveInfo[iFactor.FactorName]
            iData = getattr(qs_env,iFactorDB).writeFactorData(iData,FT.TempSaveInfo[iFactorDB],iFactor.FactorName,if_exists='merge',data_type=iFactor.FactorDataType)
            iData = None
            args['Sub2MainQueue'].put((qs_env.PID,1,None))
        qs_env.closeResource()
    return 0
# 因子表, 接口类
# 因子表可看做一个独立的数据集或命名空间, 可看做 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])
# 因子表的数据有三个维度: 时间点, ID, 因子
# 时间点数据类型是 datetime.datetime, ID 和因子名称的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorTable(__QS_Object__):
    Name = Str("因子表")
    FactorDB = Instance(FactorDB)
    ErgodicMode = Instance(_ErgodicMode, arg_type="ArgObject", label="遍历模式", order=0)
    OperationMode = Instance(_OperationMode, arg_type="ArgObject", label="运算模式", order=1)
    def __init__(self, sys_args={}, **kwargs):
        return super().__init__(sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        self.ErgodicMode = _ErgodicMode()
        self.OperationMode = _OperationMode(ft=self)
    # -------------------------------表的信息---------------------------------
    # 获取表的元数据
    def getMetaData(self, key=None):
        if key is None: return {}
        return None
    # -------------------------------维度信息-----------------------------------
    # 返回所有因子名
    @property
    def FactorNames(self):
        return []
    # 获取因子对象
    def getFactor(self, ifactor_name, args={}):
        iFactor = Factor(ifactor_name)
        iFactor.FactorTable = self
        for iArgName in self.ArgNames:
            iTraitName, iTrait = self.getTrait(iArgName)
            iFactor.add_trait(iTraitName, iTrait)
            iFactor[iArgName] = args.get(iArgName, self[iArgName])
        return iFactor
    # 获取因子的元数据
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None:
            factor_names = self.FactorNames
        if key is None:
            return pd.DataFrame(index=factor_names, dtype=np.dtype("O"))
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 获取 ID 序列
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []
    # 获取 ID 的 Mask, 返回: Series(True or False, index=[ID])
    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None: ids = self.getID(idt=idt, args=args)
        if not id_filter_str: return pd.Series(True, index=ids)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval(CompiledIDFilterStr)
    # 获取过滤后的ID
    def getFilteredID(self, idt, id_filter_str=None, args={}):
        if not id_filter_str: return self.getID(idt=idt, args=args)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval("temp["+CompiledIDFilterStr+"].index.tolist()")
    # 获取时间点序列
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    # -------------------------------读取数据---------------------------------
    # 计算数据的接口, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def __QS_calcData__(self, raw_data, factor_names=None, ids=None, dts=None, args={}):
        return None
    # 准备原始数据的接口
    def __QS_prepareRawData__(self, factor_names=None, ids=None, dts=None, args={}):
        return None
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, factor_names=None, ids=None, dts=None, args={}):
        if self.ErgodicMode._isStarted: return self._readData_ErgodicMode(factor_names=factor_names, ids=ids, dts=dts, args=args)
        return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args), factor_names=factor_names, ids=ids, dts=dts, args=args)
    # ------------------------------------遍历模式------------------------------------
    def _readData_FactorCacheMode(self, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        self.ErgodicMode._FactorReadNum[factor_names] += 1
        if (self.ErgodicMode.MaxFactorCacheNum<=0) or (not self.ErgodicMode._CacheDTs) or ((self.ErgodicMode._DateTimes[0] if dts is None else dts[0]) < self.ErgodicMode._CacheDTs[0]) or ((self.ErgodicMode._DateTimes[-1] if dts is None else dts[-1]) >self.ErgodicMode._CacheDTs[-1]):
            return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args), factor_names=factor_names, ids=ids, dts=dts, args=args)
        Data = {}
        DataFactorNames = []
        CacheFactorNames = []
        PopFactorNames = []
        for iFactorName in factor_names:
            iFactorData = self.ErgodicMode._CacheData.get(iFactorName)
            if iFactorData is None:# 尚未进入缓存
                if self.ErgodicMode._CacheFactorNum<self.ErgodicMode.MaxFactorCacheNum:# 当前缓存因子数小于最大缓存因子数，那么将该因子数据读入缓存
                    self.ErgodicMode._CacheFactorNum += 1
                    CacheFactorNames.append(iFactorName)
                else:# 当前缓存因子数等于最大缓存因子数，那么将检查最小读取次数的因子
                    CacheFactorReadNum = self.ErgodicMode._FactorReadNum[self.ErgodicMode._CacheData.keys()]
                    MinReadNumInd = CacheFactorReadNum.argmin()
                    if CacheFactorReadNum.loc[MinReadNumInd]<self.ErgodicMode._FactorReadNum[ifactor_name]:# 当前读取的因子的读取次数超过了缓存因子读取次数的最小值，缓存该因子数据
                        CacheFactorNames.append(iFactorName)
                        PopFactor = MinReadNumInd
                        self.ErgodicMode._CacheData.pop(PopFactor)
                        PopFactorNames.append(PopFactor)
                    else:
                        DataFactorNames.append(iFactorName)
            else:
                Data[iFactorName] = iFactorData
        if CacheFactorNames:
            iData = dict(self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=CacheFactorNames, ids=self.ErgodicMode._IDs, dts=self.ErgodicMode._CacheDTs, args=args), factor_names=CacheFactorNames, ids=self.ErgodicMode._IDs, dts=self.ErgodicMode._CacheDTs, args=args))
            Data.update(iData)
            self.ErgodicMode._CacheData.update(iData)
        self.ErgodicMode._Queue2SubProcess.put((None, (CacheFactorNames, PopFactorNames)))
        Data = pd.Panel(Data)
        if Data.shape[0]>0:
            if ids is None: ids = self.ErgodicMode._IDs
            if dts is not None: Data = Data.ix[:, dts, ids]
            else: Data = Data.ix[:, :, ids]
        if not DataFactorNames: return Data
        return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=DataFactorNames, ids=ids, dts=dts, args=args), factor_names=DataFactorNames, ids=ids, dts=dts, args=args).join(Data)
    def _readIDData(self, iid, factor_names=None, dts=None, args={}):
        self.ErgodicMode._IDReadNum[iid] = self.ErgodicMode._IDReadNum.get(iid, 0) + 1
        if (self.ErgodicMode.MaxIDCacheNum<=0) or (not self.ErgodicMode._CacheDTs) or ((self.ErgodicMode._DateTimes[0] if dts is None else dts[0]) < self.ErgodicMode._CacheDTs[0]) or ((self.ErgodicMode._DateTimes[-1] if dts is None else dts[-1]) >self.ErgodicMode._CacheDTs[-1]):
            return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=[iid], dts=dts, args=args), factor_names=factor_names, ids=[iid], dts=dts, args=args).iloc[:, :, 0]
        IDData = self.ErgodicMode._CacheData.get(iid)
        if IDData is None:# 尚未进入缓存
            if self.ErgodicMode._CacheIDNum<self.ErgodicMode.MaxIDCacheNum:# 当前缓存 ID 数小于最大缓存 ID 数，那么将该 ID 数据读入缓存
                self.ErgodicMode._CacheIDNum += 1
                IDData = self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=None, ids=[iid], dts=self.ErgodicMode._CacheDTs, args=args), factor_names=None, ids=[iid], dts=self.ErgodicMode._CacheDTs, args=args).iloc[:, :, 0]
                self.ErgodicMode._CacheData[iid] = IDData
                self.ErgodicMode._Queue2SubProcess.put((None, (iid, None)))
            else:# 当前缓存 ID 数等于最大缓存 ID 数，那么将检查最小读取次数的 ID
                CacheIDReadNum = self.ErgodicMode._IDReadNum[self.ErgodicMode._CacheData.keys()]
                MinReadNumInd = CacheIDReadNum.argmin()
                if CacheIDReadNum.loc[MinReadNumInd]<self.ErgodicMode._IDReadNum[iid]:# 当前读取的 ID 的读取次数超过了缓存 ID 读取次数的最小值，缓存该 ID 数据
                    IDData = self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=None, ids=[iid], dts=self.ErgodicMode._CacheDTs, args=args), factor_names=None, ids=[iid], dts=self.ErgodicMode._CacheDTs, args=args).iloc[:, :, 0]
                    PopID = MinReadNumInd
                    self.ErgodicMode._CacheData.pop(PopID)
                    self.ErgodicMode._CacheData[iid] = IDData
                    self.ErgodicMode._Queue2SubProcess.put((None,(iid, PopID)))
                else:# 当前读取的 ID 的读取次数没有超过缓存 ID 读取次数的最小值, 放弃缓存该 ID 数据
                    return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=[iid], dts=dts, args=args), factor_names=factor_names, ids=[iid], dts=dts, args=args).iloc[:, :, 0]
        if factor_names is not None: IDData = IDData.ix[:, factor_names]
        if dts is not None: IDData = IDData.ix[dts, :]
        return IDData
    def _readData_ErgodicMode(self, factor_names=None, ids=None, dts=None, args={}):
        if self.ErgodicMode.CacheMode=="因子": return self._readData_FactorCacheMode(factor_names=factor_names, ids=ids, dts=dts, args=args)
        if ids is None: ids = self._IDs
        return pd.Panel({iID: self._readIDData(iID, factor_names=factor_names, dts=dts, args=args) for iID in ids}).swapaxes(0, 2)
    # 启动遍历模式, dts: 遍历的时间点序列或者迭代器
    def start(self, dts=None, ids=None, **kwargs):
        if self.ErgodicMode._isStarted: return 0
        self.ErgodicMode._DateTimes = np.array(self.getDateTime(), dtype="O")
        self.ErgodicMode._IDs = (self.getID() if ids is None else ids)
        self.ErgodicMode._CurInd = -1# 当前时点在 dts 中的位置, 以此作为缓冲数据的依据
        self.ErgodicMode._DTNum = self.ErgodicMode._DateTimes.shape[0]# 时点数
        self.ErgodicMode._CacheDTs = []# 缓冲的时点序列
        self.ErgodicMode._CacheData = {}# 当前缓冲区
        self.ErgodicMode._CacheFactorNum = 0# 当前缓存因子个数, 小于等于 self.MaxFactorCacheNum
        self.ErgodicMode._CacheIDNum = 0# 当前缓存ID个数, 小于等于 self.MaxIDCacheNum
        self.ErgodicMode._FactorReadNum = pd.Series(0, index=self.FactorNames)# 因子读取次数, pd.Series(读取次数, index=self.FactorNames)
        self.ErgodicMode._IDReadNum = pd.Series()# ID读取次数, pd.Series(读取次数, index=self.FactorNames)
        self.ErgodicMode._Queue2SubProcess = Queue()# 主进程向数据准备子进程发送消息的管道
        self.ErgodicMode._Queue2MainProcess = Queue()# 数据准备子进程向主进程发送消息的管道
        self.ErgodicMode._TagName = (str(uuid.uuid1()) if os.name=="nt" else None)# 共享内存的 tag
        if self.ErgodicMode.CacheMode=="因子": self.ErgodicMode._CacheDataProcess = Process(target=_prepareMMAPFactorCacheData, args=(self, ), daemon=True)
        else: self.ErgodicMode._CacheDataProcess = Process(target=_prepareMMAPIDCacheData, args=(self, ), daemon=True)
        self.ErgodicMode._CacheDataProcess.start()
        self.ErgodicMode._isStarted = True
        return 0
    # 时间点向前移动, idt: 时间点, datetime.dateime
    def move(self, idt, *args, **kwargs):
        PreInd = self.ErgodicMode._CurInd
        self.ErgodicMode._CurInd = PreInd + np.sum(self.ErgodicMode._DateTimes[PreInd+1:]<=idt)
        if (self.ErgodicMode._CurInd>-1) and ((not self.ErgodicMode._CacheDTs) or (self.ErgodicMode._DateTimes[self.ErgodicMode._CurInd]>self.ErgodicMode._CacheDTs[-1])):# 需要读入缓冲区的数据
            self.ErgodicMode._Queue2SubProcess.put((None, None))
            DataLen, Msg = self.ErgodicMode._Queue2MainProcess.get()
            if os.name=="nt": MMAPCacheData = mmap.mmap(-1, DataLen, tagname=self.ErgodicMode._TagName)# 当前共享内存缓冲区
            else: MMAPCacheData, Msg = Msg, None
            if self.ErgodicMode._CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self.ErgodicMode._Queue2SubProcess.put((self.ErgodicMode._CurInd, None))
                self.ErgodicMode._CacheDTs = self.ErgodicMode._DateTimes[max((0, self.ErgodicMode._CurInd-self.ErgodicMode.BackwardPeriod)):min((self.ErgodicMode._DTNum, self.ErgodicMode._CurInd+self.ErgodicMode.ForwardPeriod+1))].tolist()
            else:# 出现了跳跃
                LastCacheInd = (self.ErgodicMode._DateTimes.searchsorted(self.ErgodicMode._CacheDTs[-1]) if self.ErgodicMode._CacheDTs else self.ErgodicMode._CurInd-1)
                self.ErgodicMode._Queue2SubProcess.put((LastCacheInd+1, None))
                self.ErgodicMode._CacheDTs = self.ErgodicMode._DateTimes[max((0, LastCacheInd+1-self.ErgodicMode.BackwardPeriod)):min((self.ErgodicMode._DTNum, LastCacheInd+1+self.ErgodicMode.ForwardPeriod+1))].tolist()
            MMAPCacheData.seek(0)
            self.ErgodicMode._CacheData = pickle.loads(MMAPCacheData.read(DataLen))
    # 结束遍历模式
    def end(self):
        if not self.ErgodicMode._isStarted: return 0
        self.ErgodicMode._CacheData, self.ErgodicMode._FactorReadNum, self.ErgodicMode._IDReadNum = None, None, None
        self.ErgodicMode._Queue2SubProcess.put(None)
        self.ErgodicMode._CacheDataProcess = None
        self.ErgodicMode._isStarted = False
        return 0
    # ------------------------------------运算模式------------------------------------
    # 获取因子表准备原始数据的分组信息, [(因子表对象, [因子名], [时点], {参数})] 
    def __QS_genGroupInfo__(self, ft, factors):
        StartDT = dt.datetime.now()
        FactorNames = []
        for iFactor in factors:
            FactorNames.append(iFactor.Name)
            StartDT = min((StartDT, ft.OperationMode._FactorStartDT[iFactor.Name]))
        EndDT = ft.OperationMode.DateTimes[-1]
        StartInd, EndInd = ft.OperationMode.DTRuler.index(StartDT), ft.OperationMode.DTRuler.index(EndDT)
        return [(self, FactorNames, ft.OperationMode.DTRuler[StartDT:EndDT], {})]
    def _genFactorDict(self, factors, factor_dict={}):
        for iFactor in factors:
            if (not isinstance(iFactor.Name, str)) or (iFactor.Name=="") or (iFactor is not factor_dict.get(iFactor.Name, iFactor)):# 该因子命名错误或者未命名, 或者有因子重名
                iFactor.Name = genAvailableName("TempFactor", factor_dict)
            factor_dict[iFactor.Name] = iFactor
            factor_dict.update(self._genFactorDict(getattr(iFactor, "Descriptors", []), factor_dict))
        return factor_dict
    def _initOperation(self):
        # 检查因子的合法性
        if not self.OperationMode.FactorNames: raise __QS_Error__("运算因子不能为空!")
        self.OperationMode._Factors = []# 因子列表
        self.OperationMode._FactorDict = {}# 因子字典, {因子名:因子}, 包括所有的因子, 即衍生因子所依赖的描述子也在内
        for iFactorName in self.OperationMode.FactorNames:
            self.OperationMode._Factors.append(self.getFactor(iFactorName))
            self.OperationMode._FactorDict[iFactorName] = iFactor
        self.OperationMode._FactorDict = self._genFactorDict(self.OperationMode._Factors, self.OperationMode._FactorDict)
        # 检查时点, ID 序列的合法性
        if not self.OperationMode.DateTimes: raise __QS_Error__("运算时点序列不能为空!")
        if not self.OperationMode.IDs: raise __QS_Error__("运算 ID 序列不能为空!")
        # 检查时点标尺是否合适
        DTs = pd.Series(np.arange(0, len(self.OperationMode.DTRuler)), index=self.OperationMode.DTRuler).ix[self.OperationMode.DateTimes]
        if pd.isnull(DTs).sum()>0: raise __QS_Error__("运算时点序列超出了时点标尺!")
        elif (DTs.diff().iloc[1:]!=1).sum()>0: raise __QS_Error__("运算时点序列的频率与时点标尺不一致!")
        # 生成原始数据和缓存数据存储目录
        with __QS_CacheLock__:
            FTPath = __QS_CachePath__ + os.sep + genAvailableName("FT", listDirDir(__QS_CachePath__))
            os.mkdir(FTPath)
            self.OperationMode._RawDataDir = FTPath+os.sep+'RawData'# 原始数据存放根目录
            self.OperationMode._CacheDataDir = FTPath+os.sep+'CacheData'# 中间数据存放根目录
            os.mkdir(self.OperationMode._RawDataDir)
            os.mkdir(self.OperationMode._CacheDataDir)
        if self.OperationMode.SubProcessNum==0:# 串行模式
            self.OperationMode._PIDs = ["0"]
            self.OperationMode._PID_IDs = {"0":self.OperationMode.IDs}
            self.OperationMode._RawDataDirLock = None
        else:
            self.OperationMode._PIDs = []
            self.OperationMode._PID_IDs = {}
            nPrcs = min((self.OperationMode.SubProcessNum, len(self.OperationMode.IDs)))
            SubIDs = partitionList(self.OperationMode.IDs, nPrcs)
            for i in range(nPrcs):
                iPID = "0-"+str(i)
                self.OperationMode._PIDs.append(iPID)
                self.OperationMode._PID_IDs[iPID] = SubIDs[i]
                os.mkdir(self.OperationMode._RawDataDir+os.sep+iPID)
                os.mkdir(self.OperationMode._CacheDataDir+os.sep+iPID)
            self.OperationMode._RawDataDirLock = Lock()# 原始数据目录锁, 用于读写可能会引起冲突的文件
        # 创建用于多进程的 Event 数据
        self.OperationMode._Event = {}# {因子名: (Sub2MainQueue, Event)}
        # 给每个因子设置运算模式参数对象
        for iFactor in self.OperationMode._FactorDict.values(): iFactor.OperationMode = self.OperationMode
        # 生成所有因子的起始时点信息
        self.OperationMode._FactorStartDT = {}# {因子名: 起始时点}
        for iFactor in self.Factors: iFactor._QS_updateStartDT(self.OperationMode.DateTimes[0], self.OperationMode._FactorStartDT)
    def _QS_saveRawData(self, raw_data, file_name, key_fields={}, extern_args={}):
        if 'ID' in raw_data:# 如果原始数据有ID列，按照ID列划分后存入子进程的原始文件中
            raw_data = raw_data.set_index(['ID'])
            AllIDs = set(raw_data.index)
            for iPID in self.OperationMode._PID_IDs:
                with shelve.open(self.OperationMode._RawDataDir+os.sep+iPID+os.sep+file_name) as iFile:
                    iIDs = list(set(extern_args["PID_ID"][iPID]).intersection(AllIDs))
                    iIDs.sort()
                    iData = raw_data.loc[iIDs]
                    for jKey in key_fields:
                        iFile[jKey] = iData[key_fields[jKey]]
        else:# 如果原始数据没有ID列，则将所有数据分别存入子进程的原始文件中
            for iPID in extern_args['PID_ID']:
                with shelve.open(extern_args["RawDataDir"]+os.sep+iPID+os.sep+file_name) as iFile:
                    for jKey in key_fields:
                        iFile[jKey] = raw_data[key_fields[jKey]]
        return 0
    def _adjustSavePosition(self, factor_db):# TODO
        for iFactorDB in self.QSEnv.SysArgs['FactorDBList']:
            getattr(self.QSEnv,iFactorDB).connect()
        for iFactorName in self.SaveInfo:
            iFactorDB,iTableName,iUpdateMode = self.SaveInfo[iFactorName]
            iTempTableName = self.TempSaveInfo[iFactorDB]
            iFactorDB = getattr(self.QSEnv,iFactorDB)
            if iFactorName not in iFactorDB.getFactorName(iTableName):# 目标表或者因子不存在
                iFactorDB.moveFactor({iTempTableName:[iFactorName]},iTableName)
            elif iUpdateMode=='覆盖':
                iFactorDB.deleteFactor(iTableName,[iFactorName])
                iFactorDB.moveFactor({iTempTableName:[iFactorName]},iTableName)
            elif iUpdateMode=='新值为准':
                iFactorDB.appendFactor(iTableName,iTempTableName,iFactorName,iFactorName,if_overlapping=2)
            else:# 旧值为准
                iFactorDB.appendFactor(iTableName,iTempTableName,iFactorName,iFactorName,if_overlapping=1)
        for iFactorDB,iTempTableName in self.TempSaveInfo.items():
            getattr(self.QSEnv,iFactorDB).deleteTable(iTempTableName)
    def prepare(self):
        self._initOperation()
        print("原始数据准备中...")
        StartT = time.process_time()
        # 分组准备数据
        FTs, FT_Factors = {}, {}# {id(因子表) : 因子表}, {id(因子表) : [因子]}
        for iFactor in self.OperationMode._FactorDict.values():
            if iFactor.FactorTable is not None:
                iFTID = id(iFactor.FactorTable)
                iFactorList = FT_Factors.setdefault(iFTID, [])
                iFactorList.append(iFactor)
                FTs[iFTID] = iFactor.FactorTable
        GroupInfo = []
        for iFTID in FTs: GroupInfo.extend(FTs[iFTID].__QS_genGroupInfo__(self, FT_Factors[iFTID]))
        args = {"GroupInfo":GroupInfo, "FT":self}
        if self.OperationMode.SubProcessNum==0:
            Error = _prepare(args)
        else:
            nPrcs = min((self.OperationMode.SubProcessNum, len(args["GroupInfo"])))
            Procs,Main2SubQueue,Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_prepare,
                                                                  arg=args, partition_arg="GroupInfo",
                                                                  n_partition_head=0, n_partition_tail=0,
                                                                  main2sub_queue="None", sub2main_queue="Single")
            for i in tqdm(range(len(DTDictPartitionList))):
                iPID, Error, iMsg = Sub2MainQueue.get()
                if Error!=1:
                    for iPID, iProc in Procs.items():
                        if iProc.is_alive(): iProc.terminate()
                    raise __QS_Error__(iMsg)
            for iPrcs in Procs.values(): iPrcs.join()
        print("原始数据准备完成, 耗时 : %.2f" % (time.process_time()-StartT))
    def calculate(self, factor_db, table_name=None, if_exists="append", **kwargs):
        print("因子数据计算中...")
        StartT = time.process_time()
        if self.OperationMode.SubProcessNum==0:
            _calculate({"FT": self})
        else:
            nPrcs = len(self.OperationMode._PIDs)
            nTask = len(self.OperationMode._Factors) * nPrcs
            EventState = {iFactorName:0 for iFactorName in self.OperationMode._Event}
            Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_calculate, arg={"FT":self},
                                                                    main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            with ProgressBar(max_value=nTask) as ProgBar:
                while True:
                    nEvent = len(EventState)
                    if nEvent>0:
                        FactorNames = tuple(EventState.keys())
                        for iFactorName in FactorNames:
                            iQueue = self.OperationMode._Event[iFactorName][0]
                            while not iQueue.empty():
                                jInc = iQueue.get()
                                EventState[iFactorName] += jInc
                            if EventState[iFactorName]>=nPrcs:
                                self.OperationMode._Event[iFactorName][1].set()
                                EventState.pop(iFactorName)
                    while ((not Sub2MainQueue.empty()) or (nEvent==0)) and (iProg<nTask):
                        iPID,iErrorCode,iMsg = Sub2MainQueue.get()
                        if iErrorCode==-1:
                            for iProc in Procs:
                                if iProc.is_alive(): iProc.terminate()
                            raise __QS_Error__('进程 '+iPID+' :运行失败:'+str(iMsg))
                        else:
                            iProg += 1
                            ProgBar.update(iProg)
                    if iProg>=nTask: break
            for iPID,iPrcs in Procs.items(): iPrcs.join()
        print("因子数据计算完成, 耗时 : %.2f" % (time.process_time()-StartT))
        print("调整存储位置中...")
        StartT = time.process_time()
        self._adjustSavePosition()
        print("调整存储位置完成, 耗时 : %.2f" % (time.process_time()-StartT))
        return 0
# 因子
# 因子可看做一个 DataFrame(index=[时间点], columns=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class Factor(__QS_Object__):
    Name = Str("因子")
    FactorTable = Instance(FactorTable)
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self._NameInFT = name# 因子在所属的因子表中的名字
        self.Name = name
        return
    # 获取因子的元数据
    def getMetaData(self, key=None):
        if key is None: return {}
        return None
    # 获取 ID 序列
    def getID(self, idt=None, args={}):
        return []
    # 获取时间点序列
    def getDateTime(self, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    # --------------------------------数据读取---------------------------------
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, ids=None, dts=None, args={}):
        if self.FactorTable is None: return None
        Args = self.Args
        Args.update(args)
        return self.FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, args=Args).loc[self._NameInFT]
    # 获取数据的开始时点, start_dt:新起始时点, dt_dict: 当前所有因子的时点信息: {因子名 : 开始时点}
    def _QS_updateStartDT(self, start_dt, dt_dict):
        OldStartDT = dt_dict.get(self.Name, start_dt)
        dt_dict[self.Name] = (start_dt if start_dt<OldStartDT else OldStartDT)    