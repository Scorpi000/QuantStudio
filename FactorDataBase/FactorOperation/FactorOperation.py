# coding=utf-8
"""单点运算"""
import os
import shelve

import pandas as pd
import numpy as np

from . import Factor

# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的日期, 如果运算日期为多日期，则该值为[日期]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为[ID]
# x: 描述子当期的数据, [单个描述子值 or array]
# args: 参数, {参数名:参数值}
# 如果运算日期参数为单日期, 运算ID参数为单ID, 那么x元素为单个描述子值, 返回单个元素
# 如果运算日期参数为单日期, 运算ID参数为多ID, 那么x元素为array(shape=(nID,)), 注意并发时ID并不是全截面, 返回np.array(shape=(nID,))
# 如果运算日期参数为多日期, 运算ID参数为单ID, 那么x元素为array(shape=(nDate,)), 返回np.array(shape=(nID,))
# 如果运算日期参数为多日期, 运算ID参数为多ID, 那么x元素为array(shape=(nDate,nID)), 注意并发时ID并不是全截面, 返回np.array(shape=(nDate,nID))

def DefaultOperator(f, idt, iid, x, args):
    return f.DefaultNA

# 单点运算
class Operate(Factor.DerivativeFactor):
    """因子运算"""
    def __init__(self,factor_name='',descriptors=[],sys_args={},data_type="double",default_na=np.nan):
        sys_args['算子'] = sys_args.get("算子",DefaultOperator)
        sys_args['参数'] = sys_args.get("参数",{})
        sys_args['运算日期'] = sys_args.get("运算日期",'单日期')
        sys_args['运算ID'] = sys_args.get("运算ID",'单ID')
        Factor.DerivativeFactor.__init__(self,factor_name,descriptors,sys_args,data_type,default_na)
        return
    # 更新日期信息
    def updateDateDict(self,idts,date_dict,extern_args={}):
        iError, date_dict = Factor.DerivativeFactor.updateDateDict(self,idts,date_dict,extern_args)
        if iError!=1:
            return (iError,date_dict)
        SelfDates = date_dict[self.FactorName]
        for i,iDescriptor in enumerate(self.Descriptors):
            iError, date_dict = iDescriptor.updateDateDict(SelfDates,date_dict,extern_args)
            if iError!=1:
                return (iError,date_dict)
        return (1, date_dict)
    # private, 准备因子数据
    def _prepareData(self,extern_args={}):
        self.ExternArgs = extern_args
        SelfDates = extern_args["DateDict"][self.FactorName]
        PID = self.QSEnv.PID
        IDs = extern_args['PID_ID'][PID]
        nDate = len(SelfDates)
        nID = len(IDs)
        DescriptorData = []
        for iDescriptor in self.Descriptors:
            DescriptorData.append(iDescriptor.getData(SelfDates,pids=[PID],extern_args=extern_args).values)
        if self.FactorDataType=='double':
            StdData = np.zeros((nDate,nID),dtype='float')+self.DefaultNA
        else:
            StdData = np.empty((nDate,nID),dtype='O')
        x = [None,]*len(DescriptorData)
        if (self.SysArgs['运算日期']=='单日期') and (self.SysArgs['运算ID']=='单ID'):
            for j,jDate in enumerate(SelfDates):
                for i,iID in enumerate(IDs):
                    for k,kDescriptorData in enumerate(DescriptorData):
                        if kDescriptorData.ndim==1:
                            x[k] = kDescriptorData[j]
                        else:
                            x[k] = kDescriptorData[j,i]
                    StdData[j,i] = self.SysArgs['算子'](self,jDate,iID,x,self.SysArgs['参数'])
        elif (self.SysArgs['运算日期']=='多日期') and (self.SysArgs['运算ID']=='单ID'):
            for i,iID in enumerate(IDs):
                for k,kDescriptorData in enumerate(DescriptorData):
                    if kDescriptorData.ndim==1:
                        x[k] = kDescriptorData
                    else:
                        x[k] = kDescriptorData[:,i]
                StdData[:,i] = self.SysArgs['算子'](self,SelfDates,iID,x,self.SysArgs['参数'])
        elif (self.SysArgs['运算日期']=='单日期') and (self.SysArgs['运算ID']=='多ID'):
            for j,jDate in enumerate(SelfDates):
                for k,kDescriptorData in enumerate(DescriptorData):
                    x[k] = kDescriptorData[j]
                StdData[j,:] = self.SysArgs['算子'](self,jDate,IDs,x,self.SysArgs['参数'])
        else:
            StdData = self.SysArgs['算子'](self,SelfDates,IDs,DescriptorData,self.SysArgs['参数'])
        x,DescriptorData = None,None# 释放数据
        StdData = pd.DataFrame(StdData,index=SelfDates,columns=IDs)
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