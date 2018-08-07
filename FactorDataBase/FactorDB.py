# coding=utf-8
import os
import uuid
import mmap
import pickle
import gc
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
from traits.api import Instance, Str, File, List, Int, Bool, Directory, Enum

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.Tools.IDFun import testIDFilterStr

def _genStartEndDate(dts=None, start_dt=None, end_dt=None):
    if dts is not None:
        StartDate = dts[0].date()
        EndDate = dts[-1].date()
        if start_dt is not None:
            StartDate = max((StartDate, start_dt.date()))
        if end_dt is not None:
            EndDate = min((EndDate, end_dt.date()))
    else:
        StartDate = (dt.date.today() if start_dt is None else start_dt.date())
        EndDate = (dt.date.today() if end_dt is None else end_dt.date())
    return (StartDate, EndDate)
def _adjustDateTime(data, dts=None, start_dt=None, end_dt=None, fillna=False, **kwargs):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if dts is not None:
            if fillna:
                AllDTs = data.index.union(set(dts))
                AllDTs = AllDTs.sort_values()
                data = data.ix[AllDTs]
                data = data.fillna(**kwargs)
            data = data.ix[dts]
        if start_dt is not None:
            data = data.loc[data.index>=start_dt]
        if end_dt is not None:
            data = data.loc[data.index<=end_dt]
    else:
        if dts is not None:
            if fillna:
                AllDTs = data.major_axis.union(set(dts))
                AllDTs = AllDTs.sort_values()
                data = data.ix[:, AllDTs, :]
                data = data.fillna(axis=1, **kwargs)
            data = data.ix[:, dts, :]
        if start_dt is not None:
            data = data.loc[:,data.major_axis>=start_dt]
        if end_dt is not None:
            data = data.loc[:,data.major_axis<=end_dt]
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
    def changeData(self, table_name, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if dts is None:
            return 0
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        self.writeData(Data, table_name, if_exists='replace')
        return 0
    # 填充缺失值
    def fillNA(self, filled_value, table_name, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        Data.fillna(filled_value, inplace=True)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 替换数据
    def replaceData(self, old_value, new_value, table_name, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        Data = Data.where(Data!=old_value, new_value)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 压缩数据
    def compressData(self, table_name=None, factor_names=None):
        return 0

# 基于 mmap 的缓冲数据, 如果开启遍历模式, 那么限制缓冲的因子个数, ID 个数, 时间点长度, 缓冲区里是因子的部分数据
def _prepareMMAPFactorCacheData(ft):
    CacheData, CacheDTs, Msg, MMAPCacheData, DTNum = {}, [], None, None, len(ft.ErgodicMode._DateTimes)
    while True:
        Task = ft.ErgodicMode._Queue2SubProcess.get()# 获取任务
        if Task is None: break# 结束进程
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            if os.name=='nt': MMAPCacheData = mmap.mmap(-1, DataLen, tagname=ft.ErgodicMode._TagName)# 当前MMAP缓存区
            else: Msg = MMAPCacheData = mmap.mmap(-1, DataLen)
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            ft.ErgodicMode._Queue2MainProcess.put((DataLen, Msg))
            Msg = None
            gc.collect()
        elif Task[0] is None:# 调整缓存区数据
            NewFactors, PopFactors = Task[1]
            for iFactorName in PopFactors: CacheData.pop(PopFactor)
            if NewFactors: CacheData.update(dict(ft.readData(factor_names=NewFactors, ids=ft.ErgodicMode._IDs, dts=CacheDTs)))
        else:# 准备缓冲区
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            if CurInd < DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[max((0, CurInd-ft.ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+ft.ErgodicMode.ForwardPeriod+1))]
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    NewCacheData = ft.readData(factor_names=list(CacheData.keys()), ids=ft.ErgodicMode._IDs, dts=NewCacheDTs)
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
            if os.name=='nt': MMAPCacheData = mmap.mmap(-1, DataLen, tagname=ft.ErgodicMode._TagName)# 当前MMAP缓存区
            else: Msg = MMAPCacheData = mmap.mmap(-1, DataLen)
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            ft.ErgodicMode._Queue2MainProcess.put((DataLen, Msg))
            Msg = None
            gc.collect()
        elif Task[0] is None:# 调整缓存区数据
            NewID, PopID = Task[1]
            if PopID: CacheData.pop(PopID)# 用新 ID 数据替换旧 ID
            if NewID: CacheData[NewID] = ft.readData(factor_names=ft.ErgodicMode._FactorNames, ids=[NewID], dts=CacheDTs).iloc[:, :, 0]
        else:# 准备缓冲区
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            if CurInd<DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[max((0, CurInd-ft.ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+ft.ErgodicMode.ForwardPeriod+1))]
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    NewCacheData = ft.readData(factor_names=ft.ErgodicMode._FactorNames, ids=list(CacheData.keys()), dts=NewCacheDTs)
                    for iID in CacheData:
                        CacheData[iID] = CacheData[iID].ix[CacheDTs, :]
                        CacheData[iID].ix[NewCacheDTs, :] = NewCacheData.loc[:, :, iID]
                    NewCacheData = None
    return 0
class _ErgodicMode(__QS_Object__):
    """遍历模式"""
    ForwardPeriod = Int(600, arg_type="Integer", label="向前缓冲时点数", order=0)
    BackwardPeriod = Int(1, arg_type="Integer", label="向后缓冲时点数", order=1)
    CacheMode = Enum("因子", "ID", arg_type="SingleOption", label="缓冲模式", order=2)
    MaxFactorCacheNum = Int(60, arg_type="Integer", label="最大缓冲因子数", order=3)
    MaxIDCacheNum = Int(10000, arg_type="Integer", label="最大缓冲ID数", order=4)
    def __init__(self, ft, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self._isStarted = False
        self._FT = ft
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_CacheDataProcess"] = None
        return state
    def start(self, dts):
        if self._isStarted: return 0
        self._DateTimes = dts
        self._CurInd = -1# 当前时点在 dts 中的位置, 以此作为缓冲数据的依据
        self._DTNum = len(dts)# 时点数
        self._CacheDTs = []# 缓冲的时点序列
        self._CacheData = {}# 当前缓冲区
        self._CacheFactorNum = 0# 当前缓存因子个数, 小于等于 self.MaxFactorCacheNum
        self._CacheIDNum = 0# 当前缓存ID个数, 小于等于 self.MaxIDCacheNum
        self._FactorReadNum = pd.Series(0, index=self._FT.FactorNames)# 因子读取次数, pd.Series(读取次数, index=self._FT.FactorNames)
        self._IDReadNum = pd.Series()# ID读取次数, pd.Series(读取次数, index=self._FT.FactorNames)
        self._Queue2SubProcess = Queue()# 主进程向数据准备子进程发送消息的管道
        self._Queue2MainProcess = Queue()# 数据准备子进程向主进程发送消息的管道
        self._TagName = (str(uuid.uuid1()) if os.name=="nt" else None)# 共享内存的 tag
        if self.CacheMode=="因子": self._CacheDataProcess = Process(target=_prepareMMAPCacheData, args=(self._FT, ), daemon=True)
        else: self._CacheDataProcess = Process(target=_prepareMMAPIDCacheData, args=(self._FT, ), daemon=True)
        self._CacheDataProcess.start()
        self._isStarted = True
        return 0
    def move(self, idt, *args, **kwargs):
        PreInd = self._CurInd
        self._CurInd = PreInd + 1 + self._DateTimes[PreInd+1:].index(idt)
        if (self._CurInd>-1) and ((self._CacheDTs.shape[0]==0) or (self._DateTimes[self._CurInd]>self._CacheDTs[-1])):# 需要读入缓冲区的数据
            self._Queue2SubProcess.put((None, None))
            DataLen, Msg = self._Queue2MainProcess.get()
            if os.name=="nt": MMAPCacheData = mmap.mmap(-1, DataLen, tagname=self._TagName)# 当前共享内存缓冲区
            else: MMAPCacheData, Msg = Msg, None
            if self._CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self._Queue2SubProcess.put((self._CurInd, None))
                self._CacheDTs = self._DateTimes[max((0, self._CurInd-self.BackwardPeriod)):min((self._DTNum, self._CurInd+self.ForwardPeriod+1))]
            else:# 出现了跳跃
                LastCacheInd = (self._DateTimes.index(self._CacheDTs[-1]) if self._CacheDTs else self._CurInd-1)
                self._Queue2SubProcess.put((LastCacheInd+1, None))
                self._CacheDTs = self._DateTimes[max((0, LastCacheInd+1-self.BackwardPeriod)):min((self._DTNum, LastCacheInd+1+self.ForwardPeriod+1))]
            MMAPCacheData.seek(0)
            self._CacheData = pickle.loads(MMAPCacheData.read(DataLen))
        return 0
    def end(self):
        if not self._isStarted: return 0
        self._CacheData, self._FactorReadNum, self._IDReadNum = None, None, None
        self._Queue2SubProcess.put(None)
        self._CacheDataProcess = None
        self._isStarted = False
        return 0

class _OperationMode(__QS_Object__):
    """运算模式"""
    Factors = List(arg_type="MultiOption", label="运算因子", order=0)
    IDs = List(arg_type="IDList", label="运算ID", order=1)
    DateTimes = List(arg_type="DateTimeList", label="运算时点", order=2)
    DTRuler = List(arg_type="DateTimeList", label="时点标尺", order=3)
    isStarted = Bool(False)
    CacheDir = Directory()

# 因子表, 接口类
# 因子表可看做一个独立的数据集或命名空间, 可看做 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])
# 因子表的数据有三个维度: 时间点, ID, 因子
# 时间点数据类型是 datetime.datetime, ID 和因子名称的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorTable(__QS_Object__):
    Name = Str("因子表")
    FactorDB = Instance(FactorDB)
    ErgodicMode = Instance(_ErgodicMode)
    OperationMode = Instance(_OperationMode)
    def __QS_initArgs__(self):
        self.ErgodicMode = _ErgodicMode()
        self.OperationMode = _OperationMode()
    # -------------------------------表的操作---------------------------------
    # 获取表的元数据
    def getMetaData(self, key=None):
        if key is None: return {}
        return None
    # --------------------------------三个维度的操作-----------------------------------
    # 返回所有因子名
    @property
    def FactorNames(self):
        return []
    # 获取因子对象
    def getFactor(self, ifactor_name, args={}):
        Args = dict(self.SysArgs)
        Args.update(args)
        iFactor = Factor(ifactor_name, self.QSEnv, Args)
        iFactor.FactorTable = self
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
    # --------------------------------数据操作---------------------------------
    def _readIDData(self, iid, factor_names=None, dts=None, start_dt=None, end_dt=None, args={}):
        self.ErgodicMode_IDReadNum[iid] = self.ErgodicMode_IDReadNum.get(iid, 0) + 1
        if (not self._isErgodicState) or (self.IDCacheNum==0):# 非遍历模式, 或者未开启缓存, 或者无缓存机制
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
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        return None
    # 导出数据, CSV 格式
    def toCSV(self, dir_path, axis="Factor", factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        Data = self.readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        if axis=="Factor":
            for i, iIndex in enumerate(Data.items):
                iData = Data.iloc[i]
                iData.to_csv(dir_path+os.sep+iIndex+".csv", encoding="utf-8")
        elif axis=="DateTime":
            for i, iIndex in enumerate(Data.major_axis):
                iData = Data.iloc[:, i]
                iData.to_csv(dir_path+os.sep+iIndex.strftime("%Y%m%dT%H%M%S.%f")+".csv", encoding="utf-8")
        elif axis=="ID":
            for i, iIndex in enumerate(Data.minor_axis):
                iData = Data.iloc[:, :, i]
                iData.to_csv(dir_path+os.sep+iIndex+".csv", encoding="utf-8")
        return 0
    # ------------------------------------遍历模式操作------------------------------------
    # 启动遍历模式, dts: 遍历的时间点序列或者迭代器, dates: 遍历的日期序列, times: 遍历的时间序列
    def start(self, dts=None, **kwargs):
        return 0
    # 时间点向前移动, idt: 时间点, datetime.dateime
    def move(self, idt, *args, **kwargs):
        return 0
    # 结束遍历模式
    def end(self):
        return 0
    # -----------------------------------运算模式操作--------------------------------------
    # 初始化运算模式, dts: 待计算的时间点序列, ids: 待计算的 ID 序列
    def prepare(self, factor_names, ids, dts):
        return 0
    # 结束运算模式
    def quit(self):
        return 0
        


# 因子, 接口类
# 因子可看做一个 DataFrame(index=[时间点], columns=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class Factor(__QS_Object__):
    Name = Str("因子")
    FactorTable = Instance(FactorTable)
    CacheFilePath = File()
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self._NameInFT = name# 因子在所属的因子表中的名字
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
    # --------------------------------数据操作---------------------------------
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if self.FactorTable is None: return None
        return self.FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args).loc[self._NameInFT]
    # ------------------------------------遍历模式操作------------------------------------
    # 启动遍历模式, dts: 遍历的时间点序列或者迭代器, dates: 遍历的日期序列, times: 遍历的时间序列
    def start(self, dts=None, dates=None, times=None):
        return 0
    # 时间点向前移动, idt: 时间点, datetime.dateime
    def move(self, idt, *args, **kwargs):
        return 0
    # 结束遍历模式
    def end(self):
        return 0
    # -----------------------------------运算模式操作--------------------------------------
    # 初始化运算模式, dts: 待计算的时间点序列, ids: 待计算的 ID 序列
    def prepare(self, factor_names, ids, dts):
        return 0
    # 结束运算模式
    def quit(self):
        return 0