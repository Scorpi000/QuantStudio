# -*- coding: utf-8 -*-
"""因子运算"""
import os
import shelve
from multiprocessing import Queue, Event, Lock

import pandas as pd
import numpy as np
from traits.api import Function, Dict, Enum, List, Int

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.Tools.AuxiliaryFun import partitionList

def _DefaultOperator(f, idt, iid, x, args):
    return np.nan

class DerivativeFactor(Factor):
    Operator = Function(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
    ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
    DataType = Enum("double", "string", arg_type="SingleOption", label="数据类型", order=2)
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        self.Descriptors = descriptors
        self.UserData = {}
        return super().__init__(name=name, ft=None, sys_args=sys_args, config_file=None, **kwargs)
    def getMetaData(self, key=None):
        if key is None: return pd.Series({"DataType":self.DataType})
        elif key=="DataType": return self.DataType
        return None
    def start(self, dts, ids=None, **kwargs):
        for iDescriptor in self.Descriptors: iDescriptor.start(dts=dts, ids=ids, **kwargs)
        return 0
    def end(self):
        for iDescriptor in self.Descriptors: iDescriptor.end()
        return 0


# 单点运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算时点为多时点，则该值为[时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [单个描述子值 or array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么 x 元素为单个描述子值, 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nID, )), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID,))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么 x 元素为 array(shape=(nDT, )), 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
class PointOperation(DerivativeFactor):
    """单点运算"""
    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=4)
    def readData(self, ids, dts, args={}):
        StdData = self._calcData(ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts).values for iDescriptor in self.Descriptors])
        return pd.DataFrame(StdData, index=dts, columns=ids)
    def _QS_updateStartDT(self, start_dt, dt_dict):
        super()._QS_updateStartDT(start_dt, dt_dict)
        for i, iDescriptor in enumerate(self.Descriptors): iDescriptor._QS_updateStartDT(dt_dict[self.Name], dt_dict)
    def _calcData(self, ids, dts, descriptor_data):
        if (self.DTMode=='多时点') and (self.IDMode=='多ID'):
            StdData = self.Operator(self, dts, ids, descriptor_data, self.ModelArgs)
        else:
            if self.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
            else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
            if (self.DTMode=='单时点') and (self.IDMode=='单ID'):
                for i, iDT in enumerate(dts):
                    for j, jID in enumerate(ids):
                        StdData[i, j] = self.Operator(self, iDT, jID, [iData[i, j] for iData in descriptor_data], self.ModelArgs)
            elif (self.DTMode=='多时点') and (self.IDMode=='单ID'):
                for j, jID in enumerate(ids):
                    StdData[:, j] = self.Operator(self, dts, jID, [iData[:, j] for iData in descriptor_data], self.ModelArgs)
            elif (self.DTMode=='单时点') and (self.IDMode=='多ID'):
                for i, iDT in enumerate(dts):
                    StdData[i, :] = self.Operator(self, iDT, ids, [iData[i, :] for iData in descriptor_data], self.ModelArgs)
        return StdData
    def __QS_prepareCacheData__(self):
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = list(self._OperationMode.DTRuler[StartInd:EndInd+1])
        IDs = list(self._OperationMode._PID_IDs[PID])
        StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=[PID]).values for iDescriptor in self.Descriptors])
        StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        with self._OperationMode._PID_Lock[PID]:
            with shelve.open(self._OperationMode._CacheDataDir+os.sep+PID+os.sep+self.Name) as CacheFile:
                CacheFile["StdData"] = StdData
        self._isCacheDataOK = True
        return StdData

# 时间序列运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数, )), 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数+nDT, )), 返回 array(shape=(nDate,))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数+nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
class TimeOperation(DerivativeFactor):
    """时间序列运算"""
    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=4)
    LookBack = List(arg_type="ArgList", label="回溯期数", order=5)# 描述子向前回溯的时点数(不包括当前时点)
    LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=6)
    iLookBack = Int(0, arg_type="Integer", label="自身回溯期数", order=7)
    iLookBackMode = Enum("滚动窗口", "扩张窗口", arg_type="SingleOption", label="自身回溯模式", order=8)
    def __QS_initArgs__(self):
        self.LookBack = [0]*len(self.Descriptors)
        self.LookBackMode = ["滚动窗口"]*len(self.Descriptors)
    def _QS_updateStartDT(self, start_dt, dt_dict):
        super()._QS_updateStartDT(start_dt, dt_dict)
        if len(self.Descriptors)>len(self.LookBack): raise  __QS_Error__("时间序列运算因子 : '%s' 的参数'回溯期数'序列长度小于描述子个数!" % self.Name)
        StartDT = dt_dict[self.Name]
        DTRuler = self._OperationMode.DTRuler
        StartInd = DTRuler.index(StartDT)
        for i, iDescriptor in enumerate(self.Descriptors):
            iStartInd = StartInd - self.LookBack[i]
            if iStartInd<0:
                #raise __QS_Error__("时点标尺长度不足, 请将起始点前移!")
                print("注意: 对于因子 '%s' 的描述子 '%s', 时点标尺长度不足!" % (self.Name, iDescriptor.Name))
            iStartDT = DTRuler[max(0, iStartInd)]
            iDescriptor._QS_updateStartDT(iStartDT, dt_dict)
    def readData(self, ids, dts, args={}):
        StdData = self._calcData(ids=ids, dts=dts, descriptor_data=[np.r_[np.full((self.LookBack[i], len(ids)), np.nan), iDescriptor.readData(ids=ids, dts=dts).values] for i, iDescriptor in enumerate(self.Descriptors)])
        return pd.DataFrame(StdData, index=dts, columns=ids)
    def _calcData(self, ids, dts, descriptor_data):
        if self.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
        for i in range(len(self.Descriptors)):
            iLookBack = self.LookBack[i]
            if self.LookBackMode[i]=="滚动窗口":
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        if self.iLookBack!=0:
            if self.iLookBackMode=="扩张窗口":
                StartIndAndLen.insert(0, (-1, np.inf))
                MaxLen = np.inf
            else:
                StartIndAndLen.insert(0, (-1, self.iLookBack))
                MaxLen = max(MaxLen, self.iLookBack+1)
            MaxLookBack = max(MaxLookBack, self.iLookBack)
            descriptor_data.insert(0, StdData)
        if self._OperationMode is None: DTRuler = [None] * MaxLookBack + dts
        else:
            StartInd = self._OperationMode.DTRuler.index(dts[0])
            if StartInd>=MaxLookBack: DTRuler = self._OperationMode.DTRuler[StartInd-MaxLookBack:]
            else: DTRuler = [None]*(MaxLookBack-StartInd) + self._OperationMode.DTRuler
        if (self.DTMode=='单时点') and (self.IDMode=='单ID'):
            for i, iDT in enumerate(dts):
                iDTs = DTRuler[max(0, MaxLookBack+i-MaxLen):i+1+MaxLookBack]
                for j, jID in enumerate(ids):
                    x = []
                    for k, kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i, j])
                    StdData[i, j] = self.Operator(self, iDTs, jID, x, self.ModelArgs)
        elif (self.DTMode=='单时点') and (self.IDMode=='多ID'):
            for i, iDT in enumerate(dts):
                iDTs = DTRuler[max(0, MaxLookBack+i-MaxLen):i+1+MaxLookBack]
                x = []
                for k,kDescriptorData in enumerate(descriptor_data):
                    kStartInd, kLen = StartIndAndLen[k]
                    x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                StdData[i, :] = self.Operator(self, iDTs, ids, x, self.ModelArgs)
        elif (self.DTMode=='多时点') and (self.IDMode=='单ID'):
            for j, jID in enumerate(ids):
                StdData[:, j] = self.Operator(self, DTRuler, jID, [kDescriptorData[:, j] for kDescriptorData in descriptor_data], self.ModelArgs)
        else:
            StdData = self.Operator(self, DTRuler, ids, descriptor_data, self.ModelArgs)
        return StdData
    def __QS_prepareCacheData__(self):
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = list(self._OperationMode.DTRuler[StartInd:EndInd+1])
        IDs = list(self._OperationMode._PID_IDs[PID])
        DescriptorData = []
        for i, iDescriptor in enumerate(self.Descriptors):
            iStartInd = StartInd - self.LookBack[i]
            iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
            iDescriptorData = iDescriptor._QS_getData(iDTs, pids=[PID]).values
            if iStartInd<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
            DescriptorData.append(iDescriptorData)
        StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData)
        StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        with self._OperationMode._PID_Lock[PID]:
            with shelve.open(self._OperationMode._CacheDataDir+os.sep+PID+os.sep+self.Name) as CacheFile:
                CacheFile["StdData"] = StdData
        self._isCacheDataOK = True
        return StdData

# 截面运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(nID, )), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class SectionOperation(DerivativeFactor):
    """截面运算"""
    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=4)
    def readData(self, ids, dts, args={}):
        StdData = self._calcData(ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts).values for iDescriptor in self.Descriptors])
        return pd.DataFrame(StdData, index=dts, columns=ids)
    def _QS_updateStartDT(self, start_dt, dt_dict):
        OldStartDT = dt_dict.get(self.Name, None)
        if (OldStartDT is None) or (start_dt<OldStartDT):
            dt_dict[self.Name] = start_dt
            StartInd, EndInd = self._OperationMode.DTRuler.index(dt_dict[self.Name]), self._OperationMode.DTRuler.index(self._OperationMode.DateTimes[-1])
            DTs = self._OperationMode.DTRuler[StartInd:EndInd+1]
            DTPartition = partitionList(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID:DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
            for i, iDescriptor in enumerate(self.Descriptors): iDescriptor._QS_updateStartDT(start_dt, dt_dict)
        if (self._OperationMode.SubProcessNum>0) and (self.Name not in self._OperationMode._Event):
            self._OperationMode._Event[self.Name] = (Queue(), Event())
    def _calcData(self, ids, dts, descriptor_data):
        if self.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        if self.OutputMode=='全截面':
            if self.DTMode=='单时点':
                for i, iDT in enumerate(dts):
                    StdData[i, :] = self.Operator(self, iDT, ids, [kDescriptorData[i] for kDescriptorData in descriptor_data], self.ModelArgs)
            else:
                StdData = self.Operator(self, dts, ids, descriptor_data, self.ModelArgs)
        else:
            if self.DTMode=='单时点':
                for i, iDT in enumerate(dts):
                    x = [kDescriptorData[i] for kDescriptorData in descriptor_data]
                    for j, jID in enumerate(ids):
                        StdData[i, j] = self.Operator(self, iDT, jID, x, self.ModelArgs)
            else:
                for j, jID in enumerate(ids):
                    StdData[:, j] = self.Operator(self, dts, jID, descriptor_data, self.ModelArgs)
        return StdData
    def __QS_prepareCacheData__(self):
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        if len(DTs)==0:# 该进程未分配到计算任务
            if self._OperationMode.SubProcessNum>0:
                Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
                Sub2MainQueue.put(1)
                PIDEvent.wait()
            self._isCacheDataOK = True
            return None
        IDs = list(self._OperationMode.IDs)
        StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=None).values for iDescriptor in self.Descriptors])
        StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        for iPID, iIDs in self._OperationMode._PID_IDs.items():
            with self._OperationMode._PID_Lock[iPID]:
                with shelve.open(self._OperationMode._CacheDataDir+os.sep+iPID+os.sep+self.Name) as CacheFile:
                    if "StdData" in CacheFile:
                        CacheFile["StdData"] = pd.concat([CacheFile["StdData"], StdData.loc[:,iIDs]]).sort_index()
                    else:
                        CacheFile["StdData"] = StdData.loc[:, iIDs]
        StdData = None# 释放数据
        if self._OperationMode.SubProcessNum>0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData

# 面板运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多日期，则该值为 [回溯期数]+[时点]
# iid: 当前待计算的 ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(回溯期数, nID)), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(回溯期数+nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class PanelOperation(DerivativeFactor):
    """面板运算"""
    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=4)
    LookBack = List(arg_type="ArgList", label="回溯期数", order=5)# 描述子向前回溯的时点数(不包括当前时点)
    LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=6)
    iLookBack = Int(0, arg_type="Integer", label="自身回溯期数", order=7)
    iLookBackMode = Enum("滚动窗口", "扩张窗口", arg_type="SingleOption", label="自身回溯模式", order=8)
    def __QS_initArgs__(self):
        self.LookBack = [0]*len(self.Descriptors)
        self.LookBackMode = ["滚动窗口"]*len(self.Descriptors)
    def _QS_updateStartDT(self, start_dt, dt_dict):
        if len(self.Descriptors)>len(self.LookBack): raise  __QS_Error__("时间序列运算因子 : '%s' 的参数'回溯期数'序列长度小于描述子个数!" % self.Name)
        OldStartDT = dt_dict.get(self.Name, None)
        if (OldStartDT is None) or (start_dt<OldStartDT):
            dt_dict[self.Name] = start_dt
            StartInd, EndInd = self._OperationMode.DTRuler.index(dt_dict[self.Name]), self._OperationMode.DTRuler.index(self._OperationMode.DateTimes[-1])
            DTs = self._OperationMode.DTRuler[StartInd:EndInd+1]
            DTPartition = partitionList(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID:DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
            StartDT = dt_dict[self.Name]
            DTRuler = self._OperationMode.DTRuler
            StartInd = DTRuler.index(StartDT)
            for i, iDescriptor in enumerate(self.Descriptors):
                iStartInd = StartInd - self.LookBack[i]
                if iStartInd<0:
                    print("注意: 对于因子 '%s' 的描述子 '%s', 时点标尺长度不足!" % (self.Name, iDescriptor.Name))
                iStartDT = DTRuler[max(0, iStartInd)]
                iDescriptor._QS_updateStartDT(iStartDT, dt_dict)
        if (self._OperationMode.SubProcessNum>0) and (self.Name not in self._OperationMode._Event):
            self._OperationMode._Event[self.Name] = (Queue(), Event())
    def readData(self, ids, dts, args={}):
        StdData = self._calcData(ids=ids, dts=dts, descriptor_data=[np.r_[np.full((self.LookBack[i], len(ids)), np.nan), iDescriptor.readData(ids=ids, dts=dts).values] for i, iDescriptor in enumerate(self.Descriptors)])
        return pd.DataFrame(StdData, index=dts, columns=ids)
    def _calcData(self, ids, dts, descriptor_data):
        if self.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
        for i, iDescriptor in enumerate(self.Descriptors):
            iLookBack = self.LookBack[i]
            if self.LookBackMode[i]=="滚动窗口":
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        if self.iLookBack!=0:
            if self.iLookBackMode=="扩张窗口":# 自身为扩张窗口模式
                StartIndAndLen.insert(0, (-1, np.inf))
                MaxLen = np.inf
            else:# 自身为滚动窗口模式
                StartIndAndLen.insert(0, (-1, self.iLookBack))
                MaxLen = max(MaxLen, self.iLookBack+1)
            descriptor_data.insert(0, StdData)
            MaxLookBack = max(MaxLookBack, self.iLookBack)
        if self._OperationMode is None: DTRuler = [None] * MaxLookBack + dts
        else:
            StartInd = self._OperationMode.DTRuler.index(dts[0])
            if StartInd>=MaxLookBack: DTRuler = self._OperationMode.DTRuler[StartInd-MaxLookBack:]
            else: DTRuler = [None]*(MaxLookBack-StartInd) + self._OperationMode.DTRuler
        if self.OutputMode=='全截面':
            if self.DTMode=='单时点':
                for i, iDT in enumerate(dts):
                    iDTs = DTRuler[max(0, MaxLookBack+i-MaxLen):i+1+MaxLookBack]
                    x = []
                    for k, kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                    StdData[i, :] = self.Operator(self, iDTs, ids, x, self.ModelArgs)
            else:
                StdData = self.Operator(self, DTRuler, ids, descriptor_data, self.ModelArgs)
        else:
            if self.DTMode=='单时点':
                for i, iDT in enumerate(dts):
                    iDTs = DTRuler[max(0, MaxLookBack+i-MaxLen):i+1+MaxLookBack]
                    x = []
                    for k, kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                    for j, jID in enumerate(ids):
                        StdData[i, j] = self.Operator(self, iDTs, jID, x, self.ModelArgs)
            else:
                for j, jID in enumerate(ids):
                    StdData[:, j] = self.Operator(self, DTRuler, jID, descriptor_data, self.ModelArgs)
        return StdData
    def __QS_prepareCacheData__(self):
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        if len(DTs)==0:# 该进程未分配到计算任务
            if self._OperationMode.SubProcessNum>0:
                Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
                Sub2MainQueue.put(1)
                PIDEvent.wait()
            self._isCacheDataOK = True
            return None
        IDs = list(self._OperationMode.IDs)
        DescriptorData, StartInd = [], self._OperationMode.DTRuler.index(DTs[0])
        for i, iDescriptor in enumerate(self.Descriptors):
            iStartInd = StartInd - self.LookBack[i]
            iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
            iDescriptorData = iDescriptor._QS_getData(iDTs, pids=None).values
            if iStartInd<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
            DescriptorData.append(iDescriptorData)
        StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData)
        DescriptorData, iDescriptorData, StdData = None, None, pd.DataFrame(StdData, index=DTs, columns=IDs)
        for iPID, iIDs in self._OperationMode._PID_IDs.items():
            with self._OperationMode._PID_Lock[iPID]:
                with shelve.open(self._OperationMode._CacheDataDir+os.sep+iPID+os.sep+self.Name) as CacheFile:
                    if "StdData" in CacheFile:
                        CacheFile["StdData"] = pd.concat([CacheFile["StdData"], StdData.loc[:,iIDs]]).sort_index()
                    else:
                        CacheFile["StdData"] = StdData.loc[:, iIDs]
        StdData = None# 释放数据
        if self._OperationMode.SubProcessNum>0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData

# 截面聚合运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的 ID 截面, [ID]
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
class SectionAggregation(DerivativeFactor):
    """截面聚合运算"""
    GroupFactor = Enum(None, arg_type="SingleOption", label="类别因子", order=3)
    IDMap = Dict(arg_type="ArgDict", label="代码对照", order=4)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.ix[self.Name]
        self.add_trait("GroupFactor", Enum(*([None]+[i for i in range(len(self.Descriptors))]), arg_type="SingleOption", label="类别因子", order=2))
    def readData(self, ids, dts, args={}):
        StdData = self._calcData(ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts).values for iDescriptor in self.Descriptors])
        return pd.DataFrame(StdData, index=dts, columns=ids)
    def _QS_updateStartDT(self, start_dt, dt_dict):
        OldStartDT = dt_dict.get(self.Name, None)
        if (OldStartDT is None) or (start_dt<OldStartDT):
            dt_dict[self.Name] = start_dt
            StartInd, EndInd = self._OperationMode.DTRuler.index(dt_dict[self.Name]), self._OperationMode.DTRuler.index(self._OperationMode.DateTimes[-1])
            DTs = self._OperationMode.DTRuler[StartInd:EndInd+1]
            DTPartition = partitionList(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID:DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
            for i, iDescriptor in enumerate(self.Descriptors): iDescriptor._QS_updateStartDT(start_dt, dt_dict)
        if (self._OperationMode.SubProcessNum>0) and (self.Name not in self._OperationMode._Event): self._OperationMode._Event[self.Name] = (Queue(), Event())
    def _calcData(self, ids, dts, descriptor_data):
        if self.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        if self.GroupFactor is None:
            for i, iDT in enumerate(dts):
                x = [kDescriptorData[i] for kDescriptorData in descriptor_data]
                StdData[i] = self.Operator(self, iDT, ids, x, self.ModelArgs)
        else:
            GroupData = descriptor_data[self.GroupFactor]
            for i, iDT in enumerate(dts):
                for j, jID in enumerate(ids):
                    jMappedID = self.IDMap.get(jID, jID)
                    if pd.isnull(jMappedID): ijGroupMask = pd.isnull(GroupData[i])
                    else: ijGroupMask = (GroupData[i]==jMappedID)
                    x = [kDescriptorData[i][ijGroupMask] for kDescriptorData in descriptor_data]
                    StdData[i, j] = self.Operator(self, iDT, jID, x, self.ModelArgs)
        return StdData
    def __QS_prepareCacheData__(self):
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = list(self._OperationMode.DTRuler[StartInd:EndInd+1])
        IDs = list(self._OperationMode._PID_IDs[PID])
        DescriptorData = []
        for i, iDescriptor in enumerate(self.Descriptors):
            iStartInd = StartInd - self.LookBack[i]
            iDTs = list(self._OperationMode.DTRuler[iStartInd:StartInd]) + DTs
            DescriptorData.append(iDescriptor._QS_getData(iDTs, pids=[PID]).values)
        StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData)
        StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        with self._OperationMode._PID_Lock[PID]:
            with shelve.open(self._OperationMode._CacheDataDir+os.sep+PID+os.sep+self.Name) as CacheFile:
                CacheFile["StdData"] = StdData
        self._isCacheDataOK = True
        return StdData