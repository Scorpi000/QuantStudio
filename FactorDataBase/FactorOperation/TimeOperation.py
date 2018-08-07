# coding=utf-8
"""时间序列运算"""
import os
import shelve

import numpy as np
import pandas as pd

from . import Factor

# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的日期, 如果运算日期为多日期，则该值为[日期]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为[ID]
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算日期参数为单日期, 运算ID参数为单ID, 那么x元素为array(shape=(回溯日期数,)), 返回单个元素
# 如果运算日期参数为单日期, 运算ID参数为多ID, 那么x元素为array(shape=(回溯日期数,nID)), 注意并发时ID并不是全截面, 返回np.array(shape=(nID,))
# 如果运算日期参数为多日期, 运算ID参数为单ID, 那么x元素为array(shape=(回溯日期数+nDate,)), 返回np.array(shape=(nDate,))
# 如果运算日期参数为多日期, 运算ID参数为多ID, 那么x元素为array(shape=(回溯日期数+nDate,nID)), 注意并发时ID并不是全截面, 返回np.array(shape=(nDate,nID))

def DefaultOperator(f,idt,iid,x,args):
    return f.DefaultNA

# 时间序列运算
class TimeOperate(Factor.DerivativeFactor):
    """因子的时间序列运算"""
    def __init__(self,factor_name='',descriptors=[],sys_args={},data_type="double",default_na=np.nan):
        nDesriptor = len(descriptors)
        sys_args['算子'] = sys_args.get("算子",DefaultOperator)
        sys_args['参数'] = sys_args.get("参数",{})
        sys_args['回溯期数'] = sys_args.get("回溯期数",[0]*nDesriptor)# 描述子向前回溯的日期数(不包括当前日期)
        sys_args['运算日期'] = sys_args.get("运算日期","单日期")
        sys_args['运算ID'] = sys_args.get("运算ID",'单ID')
        # 扩张窗口模式的参数
        sys_args['起始日'] = sys_args.get("起始日",[None]*nDesriptor)# 扩张窗口的起点, None表示滚动窗口
        # 自身迭代模式的参数
        sys_args['自身回溯期数'] = sys_args.get("自身回溯期数",0)# 自身向前回溯的日期数(不包括当前日期), 0表示没有自身迭代, None表示自身为扩张窗口模式
        sys_args['初值因子'] = sys_args.get("初值因子",None)# {"因子库名":...,"表名":...,"因子名":...}, None或者无法定位该因子时表示以运算日期序列首日为迭代起点
        Factor.DerivativeFactor.__init__(self,factor_name,descriptors,sys_args,data_type,default_na)
        return
    # 更新日期信息
    def updateDateDict(self,idts,date_dict,extern_args={}):
        iError, date_dict = Factor.DerivativeFactor.updateDateDict(self,idts,date_dict,extern_args)
        if iError!=1:
            return (iError,date_dict)
        if len(self.Descriptors)>len(self.SysArgs["回溯期数"]):
            self.QSEnv.SysArgs["LastErrorMsg"] = "时间序列运算因子的参数'回溯期数'序列长度小于描述子个数!"
            return (-1,date_dict)
        if len(self.Descriptors)>len(self.SysArgs["起始日"]):
            self.QSEnv.SysArgs["LastErrorMsg"] = "时间序列运算因子的参数'起始日'序列长度小于描述子个数!"
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
                    self.QSEnv.SysArgs["LastErrorMsg"] = "时间序列运算因子的参数'起始日'的第"+str(i)+"个值大于运行日期序列首日!"
                    return (-1,date_dict)
                iStartInd = (DateRuler<iStartDate).sum()-1
                if iStartInd<0:
                    self.QSEnv.SysArgs["LastErrorMsg"] = "时间序列运算因子的参数'起始日'的第"+str(i)+"个值不在日期标尺范围内!"
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
        SelfDates = extern_args["DateDict"][self.FactorName]
        nDate = len(SelfDates)
        IDs = extern_args['PID_ID'][PID]
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
            DescriptorData.append(iDescriptor.getData(iDates,pids=[PID],extern_args=extern_args).values)
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
        if (self.SysArgs['运算日期']=='单日期') and (self.SysArgs['运算ID']=='单ID'):
            for i,iDate in enumerate(SelfDates):
                for j,jID in enumerate(IDs):
                    x = []
                    for k,kDescriptorData in enumerate(DescriptorData):
                        kStartInd,kLen = DescriptorStartIndAndLen[k]
                        if kDescriptorData.ndim==1:
                            x.append(kDescriptorData[max((0,kStartInd+1+i-kLen)):kStartInd+1+i])
                        else:
                            x.append(kDescriptorData[max((0,kStartInd+1+i-kLen)):kStartInd+1+i,j])
                    StdData[StdStartInd+i,j] = self.SysArgs['算子'](self,iDate,jID,x,self.SysArgs['参数'])
        elif (self.SysArgs['运算日期']=='单日期') and (self.SysArgs['运算ID']=='多ID'):
            for i,iDate in enumerate(SelfDates):
                x = []
                for k,kDescriptorData in enumerate(DescriptorData):
                    kStartInd,kLen = DescriptorStartIndAndLen[k]
                    x.append(kDescriptorData[max((0,kStartInd+1+i-kLen)):kStartInd+1+i])
                StdData[StdStartInd+i,:] = self.SysArgs['算子'](self,iDate,IDs,x,self.SysArgs['参数'])
        elif (self.SysArgs['运算日期']=='多日期') and (self.SysArgs['运算ID']=='单ID'):
            for j,jID in enumerate(IDs):
                x = []
                for k,kDescriptorData in enumerate(DescriptorData):
                    if kDescriptorData.ndim==1:
                        x.append(kDescriptorData)
                    else:
                        x.append(kDescriptorData[:,j])
                StdData[StdStartInd:,j] = self.SysArgs['算子'](self,SelfDates,jID,x,self.SysArgs['参数'])
        else:
            StdData[StdStartInd:,:] = self.SysArgs['算子'](self,SelfDates,IDs,DescriptorData,self.SysArgs['参数'])
        x,DescriptorData = None,None# 释放数据
        StdData = pd.DataFrame(StdData[StdStartInd:,:],index=SelfDates,columns=IDs)
        if self.FactorDataType=='string':
            StdData = StdData.where(pd.notnull(StdData),self.DefaultNA)
        if self.PID_Lock is not None:
            self.PID_Lock[PID].acquire()
        with shelve.open(extern_args["CacheDataDir"]+os.sep+PID+os.sep+self.FactorName) as CacheFile:
            CacheFile["StdData"] = StdData
        self.isDataOK = True
        if self.PID_Lock is not None:
            self.PID_Lock[PID].release()
        return StdData