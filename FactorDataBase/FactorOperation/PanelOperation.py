# coding=utf-8
"""面板运算"""
import os
import shelve
from multiprocessing import Event,Lock,Queue
import time

import pandas as pd
import numpy as np

from . import Factor
from ..FunLib.AuxiliaryFun import partitionList

# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的日期, 如果运算日期为多日期，则该值为[回溯日期]+[日期]
# iid: 当前待计算的ID, 如果输出形式为全截面, 则该值为[ID], 该序列在并发时也是全体截面ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算日期参数为单日期, 那么x元素为array(shape=(回溯日期数,nID)), 如果输出形式为全截面返回np.array(shape=(nID,)), 否则返回单个值
# 如果运算日期参数为多日期, 那么x元素为array(shape=(回溯日期数+nDate,nID)), 如果输出形式为全截面返回np.array(shape=(nDate,nID)), 否则返回np.array(shape=(nDate,))

def DefaultOperator(f,idt,iid,x,args):
    if f.SysArgs['输出形式']=='单ID':
        return f.DefaultNA
    else:
        return np.zeros((len(iid),))+f.DefaultNA

# 面板运算
class PanelOperate(Factor.DerivativeFactor):
    """因子的面板运算"""
    def __init__(self,factor_name='',descriptors=[],sys_args={},data_type="double",default_na=np.nan):
        nDesriptor = len(descriptors)
        sys_args['算子'] = sys_args.get("算子",DefaultOperator)
        sys_args['参数'] = sys_args.get("参数",{})
        sys_args['回溯期数'] = sys_args.get("回溯期数",[0]*nDesriptor)# 描述子向前回溯的日期数(不包括当前日期)
        sys_args['运算日期'] = sys_args.get("运算日期",'单日期')
        sys_args['输出形式'] = sys_args.get("输出形式","全截面")# 算子输出的形式，可选:单ID, 全截面
        # 扩张窗口模式的参数
        sys_args['起始日'] = sys_args.get("起始日",[None]*nDesriptor)# 扩张窗口的起点, None表示滚动窗口
        # 自身迭代模式的参数
        sys_args['自身回溯期数'] = sys_args.get("自身回溯期数",0)# 自身向前回溯的日期数(不包括当前日期), 0表示没有自身迭代, None表示自身为扩张窗口模式
        sys_args['初值因子'] = sys_args.get("初值因子",None)# {"因子库名":...,"表名":...,"因子名":...}, None表示当期运算日期序列首日为迭代起点
        Factor.DerivativeFactor.__init__(self,factor_name,descriptors,sys_args,data_type,default_na)
        return
    # 更新日期信息
    def updateDateDict(self,idts,date_dict,extern_args={}):
        iError, date_dict = Factor.DerivativeFactor.updateDateDict(self,idts,date_dict,extern_args)
        if iError!=1:
            return (iError,date_dict)
        if len(self.Descriptors)>len(self.SysArgs["回溯期数"]):
            self.QSEnv.SysArgs["LastErrorMsg"] = "面板运算因子的参数'回溯期数'序列长度小于描述子个数!"
            return (-1,date_dict)
        if len(self.Descriptors)>len(self.SysArgs["起始日"]):
            self.QSEnv.SysArgs["LastErrorMsg"] = "面板运算因子的参数'起始日'序列长度小于描述子个数!"
            return (-1,date_dict)
        SelfDates = date_dict[self.FactorName]
        DateRuler = extern_args["DateRuler"]
        StartInd = DateRuler.index(SelfDates[0])
        DateRuler = pd.Series(DateRuler)
        for i,iDescriptor in enumerate(self.Descriptors):
            iStartDate = self.SysArgs['起始日'][i]
            if iStartDate is None:# 该描述子为滚动窗口模式
                iStartInd = StartInd-self.SysArgs["回溯期数"][i]
            else:# 该描述子为扩张窗口模式
                if iStartDate>SelfDates[0]:
                    self.QSEnv.SysArgs["LastErrorMsg"] = "面板运算因子的参数'起始日'的第"+str(i)+"个值大于运行日期序列首日!"
                    return (-1,date_dict)
                iStartInd = (DateRuler<iStartDate).sum()-1
                if iStartInd<0:
                    self.QSEnv.SysArgs["LastErrorMsg"] = "面板运算因子的参数'起始日'的第"+str(i)+"个值不在日期标尺范围内!"
                    return (-1,date_dict)
                iStartInd = iStartInd-self.SysArgs["回溯期数"][i]
            if iStartInd<0:
                self.QSEnv.SysArgs["LastErrorMsg"] = "日期标尺长度不足, 请将起始日前移!"
                return (-1,date_dict)
            iDates = DateRuler.iloc[iStartInd:StartInd].tolist()+SelfDates
            iError, date_dict = iDescriptor.updateDateDict(iDates,date_dict,extern_args)
            if iError!=1:
                return (iError,date_dict)
        return (1,date_dict)
    # 初始化
    def initInfo(self,extern_args={}):
        Error = Factor.DerivativeFactor.initInfo(self,extern_args)
        # 分配计算任务
        if extern_args['运行模式']!="串行":
            if self.FactorName not in extern_args['Event']:
                extern_args['Event'][self.FactorName] = (Queue(),Event())
            if self.PID_Lock is None:
                self.PID_Lock = {iPID:Lock() for iPID in extern_args['PID']}
            for iDescriptor in self.Descriptors:
                if iDescriptor.PID_Lock is None:
                    iDescriptor.PID_Lock = {iPID:Lock() for iPID in extern_args['PID']}
        else:
            self.PID_Lock = None
        nPrcs = len(extern_args['PID'])
        SelfDates = extern_args["DateDict"][self.FactorName]
        if self.SysArgs["自身回溯期数"]==0:
            DatePartition = partitionList(SelfDates,nPrcs)
            self.PID_Dates = {iPID:DatePartition[i] for i,iPID in enumerate(extern_args["PID"])}
        else:
            self.PID_Dates = {iPID:[] for i,iPID in enumerate(extern_args["PID"])}
            self.PID_Dates[extern_args["PID"][0]] = SelfDates
        # 提取初值日期序列
        if (self.SysArgs["自身回溯期数"]!=0) and (self.SysArgs['初值因子'] is not None):
            FactorDB = getattr(self.QSEnv,self.SysArgs['初值因子']['因子库名'],None)
            if (FactorDB is not None) and (self.SysArgs['初值因子']['因子名'] in FactorDB.getFactorName(self.SysArgs['初值因子']['表名'])):
                InitDates = FactorDB.getDate(self.SysArgs['初值因子']['表名'],self.SysArgs['初值因子']['因子名'])
                if self.SysArgs['自身回溯期数'] is None:
                    self.TempData["InitDates"] = InitDates
                else:
                    self.TempData["InitDates"] = InitDates[max((len(InitDates)-SysArgs['自身回溯期数'],0)):]
        return Error
    # 准备因子数据
    def _prepareData(self,extern_args={}):
        self.ExternArgs = extern_args
        PID = self.QSEnv.PID
        SelfDates = self.PID_Dates[PID]
        nDate = len(SelfDates)
        if nDate==0:# 该进程未分配到计算任务
            if extern_args['运行模式']!='串行':
                Sub2MainQueue,PIDEvent = extern_args['Event'][self.FactorName]
                Sub2MainQueue.put(1)
                PIDEvent.wait()
            self.isDataOK = True
            return None
        IDs = []
        for iPID in extern_args['PID']:
            IDs += extern_args['PID_ID'][iPID]
        nID = len(IDs)
        DateRuler = extern_args["DateRuler"]
        StartInd = DateRuler.index(SelfDates[0])
        DateRuler = pd.Series(DateRuler)
        # 提取描述子数据
        DescriptorData = []
        DescriptorStartIndAndLen = []
        for i,iDescriptor in enumerate(self.Descriptors):
            iStartDate = self.SysArgs['起始日'][i]
            if iStartDate is None:# 该描述子为滚动窗口模式
                iStartInd = StartInd-self.SysArgs["回溯期数"][i]
                DescriptorStartIndAndLen.append((self.SysArgs["回溯期数"][i],self.SysArgs["回溯期数"][i]+1))
            else:# 该描述子为扩张窗口模式
                iStartInd = (DateRuler<iStartDate).sum()-self.SysArgs["回溯期数"][i]
                DescriptorStartIndAndLen.append((StartInd-iStartInd,np.inf))
            iDates = DateRuler.iloc[iStartInd:StartInd].tolist()+SelfDates
            DescriptorData.append(iDescriptor.getData(iDates,pids=None,extern_args=extern_args).values)
        # 初始化
        if self.FactorDataType=='double':
            StdData = np.zeros((nDate,nID),dtype='float')+self.DefaultNA
        else:
            StdData = np.empty((nDate,nID),dtype='O')
        # 提取初值数据
        StdStartInd = 0
        if self.SysArgs['自身回溯期数']!=0:
            if "InitDates" not in self.TempData:# 无初值
                if self.SysArgs['自身回溯期数'] is None:# 自身为扩张窗口模式
                    DescriptorStartIndAndLen.insert(0,(-1,np.inf))
                else:# 自身为滚动窗口模式
                    DescriptorStartIndAndLen.insert(0,(-1,self.SysArgs['自身回溯期数']))
            else:# 提取初值数据
                FactorDB = getattr(self.QSEnv,self.SysArgs['初值因子']['因子库名'],None)
                InitData = FactorDB.readFactorData(self.SysArgs['初值因子']['表名'],self.SysArgs['初值因子']['因子名'],dates=self.TempData["InitDates"],ids=IDs)
                StdData = np.append(InitData.values,StdData,axis=0)
                StdStartInd = InitData.shape[0]
                if self.SysArgs['自身回溯期数'] is None:# 自身为扩张窗口模式
                    DescriptorStartIndAndLen.insert(0,(InitData.shape[0]-1,np.inf))
                else:# 自身为滚动窗口模式
                    DescriptorStartIndAndLen.insert(0,(InitData.shape[0]-1,self.SysArgs['自身回溯期数']))
            DescriptorData.insert(0,StdData)
        # 分情况计算因子数据
        if self.SysArgs['输出形式']=='全截面':
            if self.SysArgs['运算日期']=='单日期':
                for i,iDate in enumerate(SelfDates):
                    x = []
                    for k,kDescriptorData in enumerate(DescriptorData):
                        kStartInd,kLen = DescriptorStartIndAndLen[k]
                        x.append(kDescriptorData[max((0,kStartInd+1+i-kLen)):kStartInd+1+i])
                    StdData[StdStartInd+i,:] = self.SysArgs['算子'](self,iDate,IDs,x,self.SysArgs['参数'])
            else:
                StdData[StdStartInd:,:] = self.SysArgs['算子'](self,SelfDates,IDs,DescriptorData,self.SysArgs['参数'])
        else:
            if self.SysArgs['运算日期']=='单日期':
                for i,iDate in enumerate(SelfDates):
                    x = []
                    for k,kDescriptorData in enumerate(DescriptorData):
                        kStartInd,kLen = DescriptorStartIndAndLen[k]
                        x.append(kDescriptorData[max((0,kStartInd+1+i-kLen)):kStartInd+1+i])
                    for j,jID in enumerate(IDs):
                        StdData[StdStartInd+i,j] = self.SysArgs['算子'](self,iDate,jID,x,self.SysArgs['参数'])
            else:
                for j,jID in enumerate(IDs):
                    StdData[StdStartInd:,j] = self.SysArgs['算子'](self,SelfDates,jID,DescriptorData,self.SysArgs['参数'])
        DescriptorData,x = None,None# 释放数据
        StdData = pd.DataFrame(StdData[StdStartInd:,:],index=SelfDates,columns=IDs)
        if self.FactorDataType=='string':
            StdData = StdData.where(pd.notnull(StdData),self.DefaultNA)
        for iPID,iIDs in extern_args['PID_ID'].items():
            iPIDData = StdData.loc[:,iIDs]
            if self.PID_Lock is not None:
                self.PID_Lock[iPID].acquire()
            with shelve.open(extern_args["CacheDataDir"]+os.sep+iPID+os.sep+self.FactorName) as CacheFile:
                if "StdData" in CacheFile:
                    iStdData = CacheFile["StdData"]
                    iPIDData = pd.concat([iStdData,iPIDData]).sort_index()
                CacheFile["StdData"] = iPIDData
            if self.PID_Lock is not None:
                self.PID_Lock[iPID].release()
        iPIDData,iStdData,StdData = None,None,None# 释放数据
        if extern_args['运行模式']!='串行':
            Sub2MainQueue,PIDEvent = extern_args['Event'][self.FactorName]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self.isDataOK = True
        return None

# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的日期, 如果运算日期为多日期，则该值为[日期]
# iid: 当前待计算的类别的成份ID
# x: 描述子当期属于该类别的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算日期参数为单日期, 那么x元素为array(shape=(回溯日期数,nID)), nID是属于类别iid的数量
# 如果运算日期参数为多日期, 那么x元素为array(shape=(回溯日期数+nDate,nID)), nID是属于类别iid的数量

# 面板聚合运算
class PanelAggregate(PanelOperate):
    """因子的面板聚合运算"""
    def __init__(self,factor_name='',descriptors=[],sys_args={},data_type="double",default_na=np.nan):
        nDesriptor = len(descriptors)
        sys_args['算子'] = sys_args.get("算子",DefaultOperator)
        sys_args['参数'] = sys_args.get("参数",{})
        sys_args['回溯期数'] = sys_args.get("回溯期数",[0]*nDesriptor)# 向前回溯的日期数(不包括当前日期)
        sys_args['起始日'] = sys_args.get("起始日",[None]*nDesriptor)
        sys_args['输出形式'] = sys_args.get("输出形式","全截面")# 算子输出的形式，可选:单ID, 全截面
        sys_args['运算日期'] = sys_args.get("运算日期",'单日期')
        sys_args['分类因子'] = sys_args.get("分类因子",None)# 聚合所依据的分类因子在描述子序列中的位置, 如果为None表示聚合成宏观因子, 否则分类聚合
        sys_args['聚合输出'] = sys_args.get("聚合输出",False)
        sys_args['聚合存储'] = sys_args.get("聚合存储",True)
        sys_args['代码对照'] = sys_args.get('代码对照',None)
        sys_args['忽略缺失'] = sys_args.get("忽略缺失",True)# 是否忽略分类因子中的缺失值
        sys_args['忽略类别'] = sys_args.get("忽略类别",[])# 忽略掉的类别
        PanelOperate.__init__(self,factor_name,descriptors,sys_args,data_type,default_na)
        return
    # 准备因子数据
    def _prepareData(self,extern_args={}):
        self.ExternArgs = extern_args
        PID = self.QSEnv.PID
        SelfDates = self.PID_Dates[PID]
        nDate = len(SelfDates)
        IDs = []
        for iPID in extern_args['PID']:
            IDs += extern_args['PID_ID'][iPID]
        nID = len(IDs)
        if SelfDates!=[]:
            DateRuler = extern_args["DateRuler"]
            StartInd = DateRuler.index(SelfDates[0])
            DescriptorData = []
            for i,iDescriptor in enumerate(self.Descriptors):
                iDates = DateRuler[StartInd-self.SysArgs["回溯期数"][i]:StartInd]+SelfDates
                DescriptorData.append(iDescriptor.getData(iDates,pids=None,extern_args=extern_args).values)
        else:
            DescriptorData = []
            for i,iDescriptor in enumerate(self.Descriptors):
                DescriptorData.append(iDescriptor.getData([],pids=None,extern_args=extern_args).values)
        if self.SysArgs['分类因子'] is not None:
            ClassData = DescriptorData[self.SysArgs['分类因子']]
            AllClasses = pd.unique(ClassData.reshape((ClassData.shape[0]*ClassData.shape[1],)))
            if self.SysArgs['忽略缺失']:
                AllClasses = AllClasses[pd.notnull(AllClasses)]
            AllClasses = list(set(AllClasses).difference(set(self.SysArgs['忽略类别'])))
            AllClasses.sort()
            nClass = len(AllClasses)
        else:
            AllClasses = ["000000.HST"]
            nClass = 1
        if self.FactorDataType=='double':
            StdData = np.zeros((nDate,nID),dtype='float')+self.DefaultNA
            StdAggrData = np.zeros((nDate,nClass),dtype='float')+self.DefaultNA
        else:
            StdData = np.empty((nDate,nID),dtype='O')
            StdAggrData = np.empty((nDate,nClass),dtype='O')
        if self.SysArgs['分类因子'] is not None:
            IDs = np.array(IDs)
            for i,iDate in enumerate(SelfDates):
                for j,jClass in enumerate(AllClasses):
                    if pd.isnull(jClass):
                        ijMask = pd.isnull(ClassData[i,:])
                    else:
                        ijMask = (ClassData[i,:]==jClass)
                    x = []
                    for k,kDescriptorData in enumerate(DescriptorData):
                        if kDescriptorData.ndim==1:
                            x.append(kDescriptorData[i:i+self.SysArgs["回溯期数"][k]+1])
                        else:
                            x.append(kDescriptorData[i:i+self.SysArgs["回溯期数"][k]+1,ijMask])
                    ijValue = self.SysArgs['算子'](self,iDate,list(IDs[ijMask]),x,self.SysArgs['参数'])
                    StdData[i,ijMask] = ijValue
                    StdAggrData[i,j] = ijValue
        else:
            for i,iDate in enumerate(SelfDates):
                x = []
                for k,kDescriptorData in enumerate(DescriptorData):
                    x.append(kDescriptorData[i:i+self.SysArgs["回溯期数"][k]+1])
                iValue = self.SysArgs['算子'](self,iDate,IDs,x,self.SysArgs['参数'])
                StdData[i,:] = iValue
                StdAggrData[i,0] = iValue
        DescriptorData,x = None,None# 释放数据
        StdData = pd.DataFrame(StdData,index=SelfDates,columns=IDs)
        if self.SysArgs['代码对照'] is not None:# 代码对照替换
            AllClasses = [self.SysArgs['代码对照'].get(jClass,jClass) for jClass in AllClasses]
        StdAggrData = pd.DataFrame(StdAggrData,index=SelfDates,columns=AllClasses)
        if self.FactorDataType=='string':
            StdData = StdData.where(pd.notnull(StdData),self.DefaultNA)
            StdAggrData = StdAggrData.where(pd.notnull(StdAggrData),self.DefaultNA)
        for iPID,iIDs in extern_args['PID_ID'].items():
            if self.PID_Lock is not None:
                self.PID_Lock[iPID].acquire()
            with shelve.open(extern_args["CacheDataDir"]+os.sep+iPID+os.sep+self.FactorName) as CacheFile:
                if "StdData" in CacheFile:
                    CacheFile["StdData"] = pd.concat([CacheFile["StdData"],StdData.loc[:,iIDs]]).sort_index()
                    CacheFile["StdAggrData"] = pd.concat([CacheFile["StdAggrData"],StdAggrData]).sort_index()
                else:
                    CacheFile["StdAggrData"] = StdAggrData
                    CacheFile["StdData"] = StdData.loc[:,iIDs]
            if self.PID_Lock is not None:
                self.PID_Lock[iPID].release()
        StdAggrData,StdData = None,None# 释放数据
        if extern_args['运行模式']!='串行':
            Sub2MainQueue,PIDEvent = extern_args['Event'][self.FactorName]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self.isDataOK = True
        return None
    # 获取因子数据, pid=None表示取所有进程的数据, to_save: 获取数据是否为了存入因子数据库, 默认是否
    def getData(self,date_seq,pids=None,extern_args={},to_save=False):
        if not self.isDataOK:# 若没有准备好数据, 准备数据
            self._prepareData(extern_args)
        FilePath = extern_args["CacheDataDir"]+os.sep+self.QSEnv.PID+os.sep+self.FactorName
        if self.PID_Lock is not None:
            self.PID_Lock[self.QSEnv.PID].acquire()
        with shelve.open(FilePath,'r') as CacheFile:
            if (to_save and self.SysArgs['聚合存储']) or ((not to_save) and self.SysArgs['聚合输出']):
                StdData = CacheFile["StdAggrData"].loc[date_seq]
            else:
                StdData = CacheFile["StdData"].loc[date_seq]
        if self.PID_Lock is not None:
            self.PID_Lock[self.QSEnv.PID].release()
        if to_save or (not self.SysArgs['聚合输出']):
            return StdData
        if self.SysArgs["分类因子"] is None:
            return StdData.iloc[:,0]
        if self.FactorDataType=="double":
            DataType = np.dtype([(iID, np.float) for iID in StdData.columns])
        else:
            DataType = np.dtype([(iID, np.object) for iID in StdData.columns])
        StdData = StdData.values
        NewStdData = np.empty((StdData.shape[0],),dtype="O")
        for i in range(StdData.shape[0]):
            NewStdData[i] = tuple(StdData[i,:])
        return pd.Series(NewStdData.astype(DataType),index=date_seq)