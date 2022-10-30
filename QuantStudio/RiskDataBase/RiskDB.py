# coding=utf-8
from operator import index
import os
import uuid
import mmap
import pickle
import gc
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str, Int, List, Instance
from multiprocessing import Process, Queue

from QuantStudio.Tools.DateTimeFun import cutDateTime
from QuantStudio.RiskModel.RiskModelFun import decomposeCov2Corr
from QuantStudio import __QS_Object__, __QS_Error__, QSArgs
from QuantStudio.Tools.api import Panel

# 风险数据库基类, 必须存储的数据有:
# 风险矩阵: Cov, Panel(items=[时点], major_axis=[ID], minor_axis=[ID])
class RiskDB(__QS_Object__):
    """风险数据库"""
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name = Str("风险数据库", arg_type="String", label="名称", order=-100)
    @property
    def Name(self):
        return self._QSArgs.Name
    # 链接数据库
    def connect(self):
        return 0
    # 断开风险数据库
    def disconnect(self):
        return 0
    # 检查数据库是否可用
    def isAvailable(self):
        return False
    # -------------------------------表的管理---------------------------------
    # 获取数据库中的表名
    @property
    def TableNames(self):
        return []
    # 返回风险表对象
    def getTable(self, table_name, args={}):
        return None
    # 设置表的元数据
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        return 0
    # 重命名表
    def renameTable(self, old_table_name, new_table_name):
        return 0
    # 删除表
    def deleteTable(self, table_name):
        return 0
    # 删除一张表中的某些时点
    def deleteDateTime(self, table_name, dts):
        return 0
    # ------------------------数据存储--------------------------------------
    # 存储数据
    def writeData(self, table_name, idt, icov, **kwargs):
        return 0

# 风险表的遍历模式参数对象
class _ErgodicMode(QSArgs):
    """遍历模式"""
    ForwardPeriod = Int(12, arg_type="Integer", label="向前缓冲时点数", order=0)
    BackwardPeriod = Int(0, arg_type="Integer", label="向后缓冲时点数", order=1)
    CacheSize = Int(300, arg_type="Integer", label="缓冲区大小", order=2)# 以 MB 为单位
    ErgodicDTs = List(arg_type="DateTimeList", label="遍历时点", order=3)
    ErgodicIDs = List(arg_type="IDList", label="遍历ID", order=4)
    def __init__(self, owner=None, sys_args={}, config_file=None, **kwargs):
        super().__init__(owner=owner, sys_args=sys_args, config_file=config_file, **kwargs)
        self._isStarted = False
        self._CurDT = None
        self._CacheData = {}
    def __getstate__(self):
        state = self.__dict__.copy()
        if "_CacheDataProcess" in state: state["_CacheDataProcess"] = None
        return state
    def start(self, dts, **kwargs):
        if self._isStarted: return 0
        self._DateTimes = np.array((self._Owner.getDateTime() if not self.ErgodicDTs else self.ErgodicDTs), dtype="O")
        if self._DateTimes.shape[0]==0: raise __QS_Error__("风险表: '%s' 的默认时间序列为空, 请设置参数 '遍历模式-遍历时点' !" % self._Owner.Name)
        self._IDs = (self._Owner.getID() if not self.ErgodicIDs else list(self.ErgodicIDs))
        if not self._IDs: raise __QS_Error__("风险表: '%s' 的默认 ID 序列为空, 请设置参数 '遍历模式-遍历ID' !" % self._Owner.Name)
        self._CurInd = -1
        self._DTNum = self._DateTimes.shape[0]# 时点数
        self._CacheDTs = []
        self._CacheData = {}
        self._Queue2SubProcess = Queue()
        self._Queue2MainProcess = Queue()
        if self.CacheSize>0:
            if os.name=="nt":
                self._TagName = str(uuid.uuid1())# 共享内存的 tag
                self._MMAPCacheData = None
            else:
                self._TagName = None# 共享内存的 tag
                self._MMAPCacheData = mmap.mmap(-1, int(self.CacheSize*2**20))# 当前共享内存缓冲区
            self._CacheDataProcess = Process(target=_prepareRTMMAPCacheData, args=(self._Owner, self._MMAPCacheData), daemon=True)
            self._CacheDataProcess.start()
            if os.name=="nt": self._MMAPCacheData = mmap.mmap(-1, int(self.CacheSize*2**20), tagname=self._TagName)# 当前共享内存缓冲区
        self._isStarted = True
        return 0
    def move(self, idt, *args, **kwargs):
        if idt==self._CurDT: return 0
        self._CurDT = idt
        PreInd = self._CurInd
        self._CurInd = PreInd + np.sum(self._DateTimes[PreInd+1:]<=idt)
        if (self.CacheSize>0) and (self._CurInd>-1) and ((not self._CacheDTs) or (self._DateTimes[self._CurInd]>self._CacheDTs[-1])):# 需要读入缓冲区的数据
            self._Queue2SubProcess.put((None,None))
            DataLen = self._Queue2MainProcess.get()
            CacheData = b""
            while DataLen>0:
                self._MMAPCacheData.seek(0)
                CacheData += self._MMAPCacheData.read(DataLen)
                self._Queue2SubProcess.put(DataLen)
                DataLen = self._Queue2MainProcess.get()
            self._CacheData = pickle.loads(CacheData)
            if self._CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self._Queue2SubProcess.put((self._CurInd, None))
                self._CacheDTs = self._DateTimes[max((0, self._CurInd-self.BackwardPeriod)):min((self._DTNum, self._CurInd+self.ForwardPeriod+1))].tolist()
            else:# 出现了跳跃
                LastCacheInd = (self._DateTimes.searchsorted(self._CacheDTs[-1]) if self._CacheDTs else self._CurInd-1)
                self._Queue2SubProcess.put((LastCacheInd+1, None))
                self._CacheDTs = self._DateTimes[max((0, LastCacheInd+1-self.BackwardPeriod)):min((self._DTNum, LastCacheInd+1+self.ForwardPeriod+1))].tolist()
        return 0
    def end(self):
        if not self._isStarted: return 0
        self._CacheData = None
        self._Queue2SubProcess.put(None)
        self._Queue2SubProcess = self._Queue2MainProcess = self._CacheDataProcess = None
        self._isStarted = False
        self._CurDT = None
        self._MMAPCacheData = None
        return 0

def _prepareRTMMAPCacheData(rt, mmap_cache):
    CacheData, CacheDTs, MMAPCacheData, DTNum = {}, [], mmap_cache, len(rt.ErgodicMode._DateTimes)
    CacheSize = int(rt.ErgodicMode.CacheSize*2**20)
    if os.name=='nt': MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=rt.ErgodicMode._TagName)
    while True:
        Task = rt.ErgodicMode._Queue2SubProcess.get()# 获取任务
        if Task is None: break# 结束进程
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓存区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            for i in range(int(DataLen/CacheSize)+1):
                iStartInd = i*CacheSize
                iEndInd = max((i+1)*CacheSize, DataLen)
                if iEndInd>iStartInd:
                    MMAPCacheData.seek(0)
                    MMAPCacheData.write(CacheDataByte[iStartInd:iEndInd])
                    rt.ErgodicMode._Queue2MainProcess.put(iEndInd-iStartInd)
                    rt.ErgodicMode._Queue2SubProcess.get()
            rt.ErgodicMode._Queue2MainProcess.put(0)
            del CacheDataByte
            gc.collect()
        else:# 准备缓冲区
            CurInd = Task[0] + rt.ErgodicMode.ForwardPeriod + 1
            if CurInd < DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = CacheDTs
                CacheDTs = rt.ErgodicMode._DateTimes[max((0, CurInd-rt.ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+rt.ErgodicMode.ForwardPeriod+1))]
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                DropDTs = set(OldCacheDTs).difference(CacheDTs)
                for iDT in DropDTs: CacheData.pop(iDT)
                if NewCacheDTs:
                    Cov = rt.__QS_readCov__(dts=NewCacheDTs)
                    for iDT in NewCacheDTs: CacheData[iDT] = {"Cov": Cov[iDT]}
                    Cov = None
    return 0

# 风险表基类
class RiskTable(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        ErgodicMode = Instance(_ErgodicMode, arg_type="ArgObject", label="遍历模式", order=-1)
        def __QS_initArgs__(self):
            self.ErgodicMode = _ErgodicMode()
    
    def __init__(self, name, rdb, sys_args={}, config_file=None, **kwargs):
        self._Name = name
        self._RiskDB = rdb# 风险表所属的风险库
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def Name(self):
        return self._Name
    @property
    def RiskDB(self):
        return self._FactorDB
    # 获取表的元数据
    def getMetaData(self, key=None, args={}):
        if key is None: return {}
        return None
    # -------------------------------维度信息-----------------------------------
    # 获取时点序列
    def getDateTime(self, start_dt=None, end_dt=None):
        return []
    # 获取 ID 序列
    def getID(self, idt=None):
        if idt is None: idt = self.getDateTime()[-1]
        Cov = self.readCov(dts=[idt]).iloc[0]
        return Cov.index.tolist()
    # ------------------------数据读取--------------------------------------
    def __QS_readCov__(self, dts, ids=None):
        return Panel(items=dts, major_axis=ids, minor_axis=ids)
    # 读取协方差矩阵, Panel(items=[时点], major_axis=[ID], minor_axis=[ID])
    def readCov(self, dts, ids=None):
        NonCachedDTs, Cov = [], {}
        for iDT in dts:
            iCacheData = self._QSArgs.ErgodicMode._CacheData.get(iDT)
            if iCacheData is None:
                NonCachedDTs.append(iDT)
                continue
            if ids is not None: Cov[iDT] = iCacheData["Cov"].reindex(index=ids, columns=ids)
            else: Cov[iDT] = iCacheData["Cov"]
        if NonCachedDTs: Cov.update(dict(self.__QS_readCov__(dts=NonCachedDTs, ids=ids)))
        if not Cov: Cov = Panel(items=dts, major_axis=ids, minor_axis=ids)
        else:
            Cov = Panel(Cov).loc[dts]
            if ids is not None: Cov = Cov.loc[:, ids, ids]
        return Cov
    # 读取相关系数矩阵, Panel(items=[时点], major_axis=[ID], minor_axis=[ID])
    def readCorr(self, dts, ids=None):
        Cov = self.readCov(dts=dts, ids=ids)
        Corr = {}
        for iDT in Cov.items:
            iCov = Cov.loc[iDT]
            iCorr, _ = decomposeCov2Corr(iCov.values)
            Corr[iDT] = pd.DataFrame(iCorr, index=iCov.index, columns=iCov.columns)
        return Panel(Corr, items=Cov.items, major_axis=Cov.major_axis, minor_axis=Cov.minor_axis)
    # -----------------------遍历模式---------------------------------------
    def start(self, dts, **kwargs):
        return self._QSArgs.ErgodicMode.start(dts, **kwargs)
    def move(self, idt, *args, **kwargs):
        return self._QSArgs.ErgodicMode.move(idt, *args, **kwargs)
    def end(self):
        return self._QSArgs.ErgodicMode.end()

# 多因子风险数据库基类, 即风险矩阵可以分解成 V=X*F*X'+D 的模型, 其中 D 是对角矩阵, 必须存储的数据有:
# 因子风险矩阵: FactorCov(F), Panel(items=[时点], major_axis=[因子], minor_axis=[因子])
# 特异性风险: SpecificRisk(D), DataFrame(index=[时点], columns=[ID])
# 因子截面数据: FactorData(X), Panel(items=[因子], major_axis=[时点], minor_axis=[ID])
# 因子收益率: FactorReturn, DataFrame(index=[时点], columns=[因子])
# 特异性收益率: SpecificReturn, DataFrame(index=[时点], columns=[ID])
# 可选存储的数据有:
# 回归统计量: Statistics, {"tValue":Series(data=统计量, index=[因子]),"FValue":double,"rSquared":double,"rSquared_Adj":double}
class FactorRDB(RiskDB):
    """多因子风险数据库"""
    # 存储数据
    def writeData(self, table_name, idt, factor_data=None, factor_cov=None, specific_risk=None, factor_ret=None, specific_ret=None, **kwargs):
        return 0

def _prepareFRTMMAPCacheData(rt, mmap_cache):
    CacheData, CacheDTs, MMAPCacheData, DTNum = {}, [], mmap_cache, len(rt.ErgodicMode._DateTimes)
    CacheSize = int(rt.ErgodicMode.CacheSize*2**20)
    if os.name=='nt': MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=rt.ErgodicMode._TagName)
    while True:
        Task = rt.ErgodicMode._Queue2SubProcess.get()# 获取任务
        if Task is None: break# 结束进程
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓存区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            for i in range(int(DataLen/CacheSize)+1):
                iStartInd = i*CacheSize
                iEndInd = max((i+1)*CacheSize, DataLen)
                if iEndInd>iStartInd:
                    MMAPCacheData.seek(0)
                    MMAPCacheData.write(CacheDataByte[iStartInd:iEndInd])
                    rt.ErgodicMode._Queue2MainProcess.put(iEndInd-iStartInd)
                    rt.ErgodicMode._Queue2SubProcess.get()
            rt.ErgodicMode._Queue2MainProcess.put(0)
            del CacheDataByte
            gc.collect()
        else:# 准备缓冲区
            CurInd = Task[0] + rt.ErgodicMode.ForwardPeriod + 1
            if CurInd < DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = CacheDTs
                CacheDTs = rt.ErgodicMode._DateTimes[max((0, CurInd-rt.ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+rt.ErgodicMode.ForwardPeriod+1))]
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                DropDTs = set(OldCacheDTs).difference(CacheDTs)
                for iDT in DropDTs: CacheData.pop(iDT)
                if NewCacheDTs:
                    FactorCov = rt.__QS_readFactorCov__(dts=NewCacheDTs)
                    SpecificRisk = rt.__QS_readSpecificRisk__(dts=NewCacheDTs)
                    FactorData = rt.__QS_readFactorData__(dts=NewCacheDTs)
                    for iDT in NewCacheDTs:
                        CacheData[iDT] = {"FactorCov": FactorCov.loc[iDT],
                                          "SpecificRisk": SpecificRisk.loc[iDT],
                                          "FactorData": FactorData.loc[:, iDT]}
                    FactorCov = SpecificRisk = FactorData = None
    return 0
# 风险表的遍历模式参数对象
class _FactorErgodicMode(_ErgodicMode):
    """遍历模式"""
    def start(self, dts, **kwargs):
        if self._isStarted: return 0
        self._DateTimes = np.array((self._Owner.getDateTime() if not self.ErgodicDTs else self.ErgodicDTs), dtype="O")
        if self._DateTimes.shape[0]==0: raise __QS_Error__("风险表: '%s' 的默认时间序列为空, 请设置参数 '遍历模式-遍历时点' !" % self._Name)
        self._IDs = (self._Owner.getID() if not self.ErgodicIDs else list(self.ErgodicIDs))
        if not self._IDs: raise __QS_Error__("风险表: '%s' 的默认 ID 序列为空, 请设置参数 '遍历模式-遍历ID' !" % self._Name)
        self._CurInd = -1
        self._DTNum = self._DateTimes.shape[0]# 时点数
        self._CacheDTs = []
        self._CacheData = {}
        self._Queue2SubProcess = Queue()
        self._Queue2MainProcess = Queue()
        if self.CacheSize>0:
            if os.name=="nt":
                self._TagName = str(uuid.uuid1())# 共享内存的 tag
                self._MMAPCacheData = None
            else:
                self._TagName = None# 共享内存的 tag
                self._MMAPCacheData = mmap.mmap(-1, int(self.CacheSize*2**20))# 当前共享内存缓冲区
            self._CacheDataProcess = Process(target=_prepareFRTMMAPCacheData, args=(self, self._MMAPCacheData), daemon=True)
            self._CacheDataProcess.start()
            if os.name=="nt": self._MMAPCacheData = mmap.mmap(-1, int(self.CacheSize*2**20), tagname=self._TagName)# 当前共享内存缓冲区
        self._isStarted = True
        return 0

# 多因子风险表基类
class FactorRT(RiskTable):
    class __QS_ArgClass__(RiskTable.__QS_ArgClass__):
        ErgodicMode = Instance(_FactorErgodicMode, arg_type="ArgObject", label="遍历模式", order=-1)
        def __QS_initArgs__(self):
            self.ErgodicMode = _FactorErgodicMode()
    # -------------------------------维度信息-----------------------------------
    @property
    def FactorNames(self):
        return []
    # 获取时点序列
    def getDateTime(self, start_dt=None, end_dt=None):
        return []
    # 获取 ID 序列
    def getID(self, idt=None):
        if idt is None: idt = self.getDateTime()[-1]
        SpecificRisk = self.readSpecificRisk(idt=idt)
        return SpecificRisk.index.tolist()
    # ------------------------数据读取--------------------------------------
    # 获取因子收益的时点
    def getFactorReturnDateTime(self, start_dt=None, end_dt=None):
        FactorReturn = self.readFactorReturn()
        if FactorReturn is not None: return cutDateTime(FactorReturn.index, start_dt=start_dt, end_dt=end_dt)
        return []
    # 获取特异性收益的时点
    def getSpecificReturnDateTime(self, start_dt=None, end_dt=None):
        SpecificReturn = self.readSpecificReturn()
        if SpecificReturn is not None: return cutDateTime(SpecificReturn.index, start_dt=start_dt, end_dt=end_dt)
        return []
    def __QS_readCov__(self, dts, ids=None):
        FactorCov = self.__QS_readFactorCov__(dts=dts)
        FactorData = self.__QS_readFactorData__(dts=dts, ids=ids)
        SpecificRisk = self.__QS_readSpecificRisk__(dts=dts, ids=ids)
        Data = {}
        for iDT in FactorCov:
            if ids is None:
                iIDs = SpecificRisk.loc[iDT].index
                iFactorData = FactorData.loc[:, iDT].reindex(index=iIDs)
            else:
                iIDs = ids
                iFactorData = FactorData.loc[:, iDT]
            iCov = np.dot(np.dot(iFactorData.values, FactorCov[iDT].values), iFactorData.values.T) + np.diag(SpecificRisk.loc[iDT].values**2)
            Data[iDT] = pd.DataFrame(iCov, index=iIDs, columns=iIDs)
        return Panel(Data, items=dts)
    def readCov(self, dts, ids=None):
        Data = {}
        CachedDTs = sorted(set(dts).intersection(self._QSArgs.ErgodicMode._CacheData))
        if CachedDTs:
            FactorCov = self.readFactorCov(dts=CachedDTs)
            FactorData = self.readFactorData(dts=CachedDTs, ids=ids)
            SpecificRisk = self.readSpecificRisk(dts=CachedDTs, ids=ids)
            for iDT in FactorCov:
                if ids is None:
                    iIDs = SpecificRisk.loc[iDT].index
                    iFactorData = FactorData.loc[:, iDT].reindex(index=iIDs)
                else:
                    iIDs = ids
                    iFactorData = FactorData.loc[:, iDT]
                iCov = np.dot(np.dot(iFactorData.values, FactorCov[iDT].values), iFactorData.values.T) + np.diag(SpecificRisk.loc[iDT].values**2)
                Data[iDT] = pd.DataFrame(iCov, index=iIDs, columns=iIDs)
        NewDTs = sorted(set(dts).difference(self._QSArgs.ErgodicMode._CacheData))
        if NewDTs: Data.update(dict(self.__QS_readCov__(dts=NewDTs, ids=ids)))
        return Panel(Data, items=dts)
    # 读取因子风险矩阵
    def __QS_readFactorCov__(self, dts):
        return Panel(items=dts)
    def readFactorCov(self, dts):
        NonCachedDTs, Cov = [], {}
        for iDT in dts:
            iCacheData = self._QSArgs.ErgodicMode._CacheData.get(iDT)
            if iCacheData is None:
                NonCachedDTs.append(iDT)
                continue
            Cov[iDT] = iCacheData["FactorCov"]
        if NonCachedDTs: Cov.update(dict(self.__QS_readFactorCov__(dts=NonCachedDTs)))
        if not Cov: return Panel(items=dts)
        else: return Panel(Cov, items=dts)
    # 读取特异性风险
    def __QS_readSpecificRisk__(self, dts, ids=None):
        return pd.DataFrame(index=dts, columns=ids)
    def readSpecificRisk(self, dts, ids=None):
        NonCachedDTs, Data = [], {}
        for iDT in dts:
            iCacheData = self._QSArgs.ErgodicMode._CacheData.get(iDT)
            if iCacheData is None:
                NonCachedDTs.append(iDT)
                continue
            if ids is not None: Data[iDT] = iCacheData["SpecificRisk"].reindex(index=ids)
            else: Data[iDT] = iCacheData["SpecificRisk"]
        if NonCachedDTs: Data.update(dict(self.__QS_readSpecificRisk__(dts=NonCachedDTs, ids=ids).T))
        if not Data: return pd.DataFrame(index=dts, columns=ids)
        else:
            Data = pd.DataFrame(Data).T.reindex(index=dts)
            if ids is not None: Data = Data.reindex(columns=ids)
            return Data
    # 读取截面数据
    def __QS_readFactorData__(self, dts, ids=None):
        return Panel(major_axis=dts, minor_axis=ids)
    def readFactorData(self, dts, ids=None):
        NonCachedDTs, Data = [], {}
        for iDT in dts:
            iCacheData = self._QSArgs.ErgodicMode._CacheData.get(iDT)
            if iCacheData is None:
                NonCachedDTs.append(iDT)
                continue
            if ids is not None: Data[iDT] = iCacheData["FactorData"].reindex(index=ids).T
            else: Data[iDT] = iCacheData["FactorData"].T
        if NonCachedDTs: Data.update(dict(self.__QS_readFactorData__(dts=NonCachedDTs, ids=ids).swapaxes(0, 1)))
        if not Data: return Panel(major_axis=dts, minor_axis=ids)
        else:
            Data = Panel(Data, items=dts).swapaxes(0, 1)
            if ids is not None: Data = Data.loc[:, :, ids]
            return Data
    # 读取因子收益率
    def readFactorReturn(self, dts):
        return pd.DataFrame(index=dts)
    # 读取残余收益率
    def readSpecificReturn(self, dts, ids=None):
        return pd.DataFrame(index=dts, columns=ids)