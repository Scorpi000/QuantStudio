# coding=utf-8
from collections import OrderedDict
import uuid
import mmap
import pickle
from multiprocessing import Process,Queue
import gc

import numpy as np
import pandas as pd

from .RiskModelFun import dropRiskMatrixNA

# 风险数据源基类
class RiskDataSource(object):
    ClassName = 'RiskDataSource'
    def __init__(self,name,risk_db=None,qs_env={}):
        self.DSType = '风险数据源'
        self.Name = name
        self.RiskDB = risk_db# 风险数据库
        self.TableName = None# 风险数据所在的表（目录）
        self.Dates = []# 数据源可提取的最长日期序列，['20090101']
        self.QSEnv = qs_env
        self._SysArgs,_ = self.genSysArgInfo()
        return
    # 准备数据生成数据源
    def prepareData(self,table_name,start_date=None,end_date=None,dates=None):
        self.TableName = table_name
        self.Dates = self.RiskDB.getTableDate(table_name,start_date,end_date)
        if dates is not None:
            self.Dates = list(set(self.Dates).intersection(set(dates)))
            self.Dates.sort()
        return 0
    # 生成系统参数信息集以及初始值
    def genSysArgInfo(self, args=None, arg_changed=None):
        return ({},{})
    def getSysArgs(self):
        return self._SysArgs
    def setSysArgs(self, args={}):
        self._SysArgs.update(args)
    SysArgs = property(getSysArgs,setSysArgs)
    # 恢复数据源到初始状态
    def start(self):
        return 0
    # 日期向前移动,idt:当前日期,'20090101'
    def MoveOn(self,idt):
        return 0
    # 获取ID,idt:'20090101',idt为某个日期，返回对应该日期的个股风险不缺失的ID序列
    def getID(self,idt):
        iCov = self.getDateCov(idt,drop_na=True)
        return list(iCov.index)
    # 给定单个日期，提取个股的协方差矩阵
    def getDateCov(self,idt,ids=None,drop_na=True):
        CovMatrix = self.RiskDB.loadCov(self.TableName,dates=idt,ids=ids,drop_na=drop_na)
        return CovMatrix
    # 结束数据源，生成特有的结果集
    def endDS(self):
        return {}
    # 保存自身信息
    def saveInfo(self,container):
        container["DSType"] = self.DSType
        container["Name"] = self.Name
        container["Dates"] = self.Dates
        container["SysArgs"] = self.SysArgs
        container['TableName'] = self.TableName
        container['RiskDB'] = self.RiskDB.__doc__
        return container
    # 恢复信息
    def loadInfo(self,container):
        self.DSType = container["DSType"]
        self.Name = container['Name']
        self.Dates = container["Dates"]
        self.SysArgs = container['SysArgs']
        self.TableName = container['TableName']
        self.RiskDB = getattr(self.QSEnv,container['RiskDB'])
        _,self.SysArgInfos = self.genSysArgInfo(self.SysArgs)
        return 0

# 基于mmap的带并行局部缓冲的风险数据源, 如果开启遍历模式, 那么限制缓冲的日期长度, 缓冲区里是部分日期序列数据, 如果未开启, 则调用 RiskDataSource 提取数据的方法. 适合遍历数据, 内存消耗小, 首次提取时间不长
def prepareRDSMMAPCacheData(arg):
    CacheData = {}
    CacheDates = []
    #print("启动缓冲: "+str(arg['CacheSize']))
    while True:
        Task = arg["Queue2SubProcess"].get()
        if Task is None:
            break
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            Msg = None
            if arg["OSType"]=='Windows':
                MMAPCacheData = mmap.mmap(-1,DataLen,tagname=arg["TagName"])# 当前MMAP缓存区
            else:
                MMAPCacheData = mmap.mmap(-1,DataLen)
                Msg = MMAPCacheData
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            arg["Queue2MainProcess"].put((DataLen,Msg))
            Msg = None
            gc.collect()
            #print("装入数据:"+str(DataLen))# debug
        else:# 准备缓冲区
            MMAPCacheData = None
            CurInd = Task[0]+arg["SysArgs"]['向前缓冲日期数']+1
            if CurInd<arg["DateNum"]:# 未到结尾处, 需要再准备缓存数据
                OldCacheDates = CacheDates
                CacheDates = arg["Dates"][max((0,CurInd-arg["SysArgs"]['向后缓冲日期数'])):min((arg["DateNum"],CurInd+arg["SysArgs"]['向前缓冲日期数']+1))]
                NewCacheDates = list(set(CacheDates).difference(set(OldCacheDates)))
                NewCacheDates.sort()
                DropDates = list(set(OldCacheDates).difference(set(CacheDates)))
                for iDate in DropDates:
                    CacheData.pop(iDate)
                Cov = arg["RiskDB"].loadCov(arg['TableName'],dates=NewCacheDates)
                for iDate in NewCacheDates:
                    CacheData[iDate] = {}
                    CacheData[iDate]["Cov"] = Cov[iDate]
                #print("准备因子:"+str(list(CacheData.keys())))# debug
    return 0

class ParaMMAPCacheRDS(RiskDataSource):
    ClassName = 'ParaMMAPCacheRDS'
    def __init__(self,name,risk_db=None,qs_env={}):
        RiskDataSource.__init__(self,name,risk_db,qs_env)
        self.DSType = '并行缓冲风险数据源(MMAP)'
        # 遍历模式变量
        self.CurInd = -1# 当前日期在self.Dates中的位置, 以此作为缓冲数据的依据
        self.DateNum = None# 日期数
        self.CacheDates = []# 缓冲的日期序列
        self.CacheData = {}# 当前缓冲区,{"Cov":DataFrame(因子协方差,index=[因子],columns=[因子])}
        self.Queue2SubProcess = None# 主进程向数据准备子进程发送消息的管道
        self.Queue2MainProcess = None# 数据准备子进程向主进程发送消息的管道
        self.CacheFun = prepareRDSMMAPCacheData
        return
    # 生成系统参数信息集以及初始值
    def genSysArgInfo(self, args=None, arg_changed=None):
        if args is None:
            args = {'向前缓冲日期数':12,'向后缓冲日期数':0}
        ArgInfo = {}
        ArgInfo['向前缓冲日期数'] = {'type':'Integer', "order":0, 'min':0, "max":9999}
        ArgInfo['向后缓冲日期数'] = {'type':'Integer', "order":0, 'min':0, "max":9999}
        return (args,ArgInfo)
    # 恢复数据源到初始状态
    def start(self):
        self.CurInd = -1
        self.DateNum = len(self.Dates)
        self.CacheDates = []
        self.CacheData = {}
        self.CacheFactorNum = 0
        self.Queue2SubProcess = Queue()
        self.Queue2MainProcess = Queue()
        arg = {}
        arg['Queue2SubProcess'] = self.Queue2SubProcess
        arg['Queue2MainProcess'] = self.Queue2MainProcess
        arg['DSName'] = self.Name
        arg['RiskDB'] = self.RiskDB
        arg["TableName"] = self.TableName
        arg["Dates"] = self.Dates
        arg['DateNum'] = self.DateNum
        arg['SysArgs'] = self.SysArgs
        arg['PID'] = self.QSEnv.PID
        arg["OSType"] = self.QSEnv.SysArgs["OSType"]
        # 准备缓冲区
        if self.QSEnv.SysArgs["OSType"]=="Windows":
            self.TagName = str(uuid.uuid1())
            arg["TagName"] = self.TagName
        self.CacheDataProcess = Process(target=self.CacheFun,args=(arg,),daemon=True)
        self.CacheDataProcess.start()
        self.TempDates = pd.Series(self.Dates)
        return 0
    # 日期向前移动,idt:当前日期,'20090101'
    def MoveOn(self,idt):
        PreInd = self.CurInd
        self.CurInd = (self.TempDates<=idt).sum()-1
        #self.CurInd = self.CurInd+self.Dates[self.CurInd+1:].index(idt)+1
        if (self.CurInd>-1) and ((self.CacheDates==[]) or (self.Dates[self.CurInd]>self.CacheDates[-1])):# 需要读入缓冲区的数据
            self.Queue2SubProcess.put((None,None))
            DataLen,Msg = self.Queue2MainProcess.get()
            if self.QSEnv.SysArgs["OSType"]=="Windows":
                MMAPCacheData = mmap.mmap(-1,DataLen,tagname=self.TagName)# 当前共享内存缓冲区
            else:
                MMAPCacheData = Msg
                Msg = None
            if self.CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self.Queue2SubProcess.put((self.CurInd,None))
                self.CacheDates = self.Dates[max((0,self.CurInd-self.SysArgs['向后缓冲日期数'])):min((self.DateNum,self.CurInd+self.SysArgs['向前缓冲日期数']+1))]
            else:# 出现了跳跃
                LastCacheInd = (self.Dates.index(self.CacheDates[-1]) if self.CacheDates!=[] else self.CurInd-1)
                self.Queue2SubProcess.put((LastCacheInd+1,None))
                self.CacheDates = self.Dates[max((0,LastCacheInd+1-self.SysArgs['向后缓冲日期数'])):min((self.DateNum,LastCacheInd+1+self.SysArgs['向前缓冲日期数']+1))]
            MMAPCacheData.seek(0)
            self.CacheData = pickle.loads(MMAPCacheData.read(DataLen))
        return 0
    # 给定单个日期，提取风险矩阵
    def getDateCov(self,idt,ids=None,drop_na=True):
        CovMatrix = self.CacheData.get(idt)
        if CovMatrix is None:# 非遍历模式或者缓冲区无数据
            CovMatrix = self.RiskDB.loadCov(self.TableName,dates=idt)
        else:
            CovMatrix = CovMatrix.get("Cov")
        if CovMatrix is None:
            return None
        if ids is not None:
            CovMatrix = CovMatrix.ix[ids,ids]
        if drop_na:
            return dropRiskMatrixNA(CovMatrix)
        return CovMatrix
    # 结束数据源，生成特有的结果集
    def endDS(self):
        self.CacheData = {}
        self.Queue2SubProcess.put(None)
        return {}

# 多因子风险数据源基类,主要元素如下:
# 因子风险矩阵: FactorCov, DataFrame(data=协方差,index=因子,columns=因子)
# 特异性风险: SpecificRisk, Series(data=方差,index=ID)
# 因子截面数据: FactorData, DataFrame(data=因子数据,index=ID,columns=因子)
# 因子收益率: FactorReturn, Series(data=收益率,index=因子)
# 特异性收益率: SpecificReturn, Series(data=收益率,index=ID)
class FactorRDS(RiskDataSource):
    ClassName = 'FactorRDS'
    def __init__(self,name,factor_risk_db=None,qs_env={}):
        RiskDataSource.__init__(self,name,factor_risk_db,qs_env)
        self.DSType = "因子风险数据源"
        self.FactorNames = []# 数据源中所有的因子名，['LNCAP']
        return
    # 准备数据生成数据源
    def prepareData(self,table_name,start_date=None,end_date=None,dates=None):
        RiskDataSource.prepareData(self,table_name,start_date=start_date,end_date=end_date,dates=dates)
        self.FactorNames = self.RiskDB.getTableFactor(table_name)
        return 0
    # 获取ID,idt:'20090101',idt为某个日期，返回对应该日期的个股风险不缺失的ID序列
    def getID(self,idt):
        iSpecificRisk = self.getDateSpecificRisk(idt)
        return list(iSpecificRisk[pd.notnull(iSpecificRisk)].index)
    # 给定单个日期，提取因子风险矩阵
    def getDateFactorCov(self,idt,factor_names=None):
        Data = self.RiskDB.loadFactorCov(self.TableName,dates=idt)
        if factor_names is not None:
            return Data.ix[factor_names,factor_names]
        else:
            return Data
    # 给定单个日期，提取个股的特别风险
    def getDateSpecificRisk(self,idt,ids=None):
        return self.RiskDB.loadSpecificRisk(self.TableName,dates=idt,ids=ids)
    # 给定单个日期，提取因子截面数据
    def getDateFactorData(self,idt,factor_names=None,ids=None):
        Data = self.RiskDB.loadFactorData(self.TableName,dates=idt,ids=ids)
        if factor_names is not None:
            return Data.ix[:,factor_names]
        else:
            return Data
    # 给定单个日期，提取个股的协方差矩阵
    def getDateCov(self,idt,ids=None,drop_na=True):
        FactorCov = self.getDateFactorCov(idt)
        SpecificRisk = self.getDateSpecificRisk(idt,ids=ids)
        if (FactorCov is None) or (SpecificRisk is None):
            return None
        if ids is None:
            ids = list(SpecificRisk.index)
        FactorExpose = self.getDateFactorData(idt,factor_names=list(FactorCov.index),ids=ids)
        CovMatrix = np.dot(np.dot(FactorExpose.values,FactorCov.values),FactorExpose.values.T)+np.diag(SpecificRisk.values**2)
        if ids is not None:
            CovMatrix = pd.DataFrame(CovMatrix,index=ids,columns=ids)
        else:
            CovMatrix = pd.DataFrame(CovMatrix,index=SpecificRisk.index,columns=SpecificRisk.index)
        if drop_na:
            return dropRiskMatrixNA(CovMatrix)
        return CovMatrix
    # 给定单个日期，提取因子收益率
    def getDateFactorReturn(self,idt,factor_names=None):
        Data = self.RiskDB.loadFactorReturn(self.TableName,dates=idt)
        if factor_names is not None:
            return Data.ix[factor_names]
        else:
            return Data
    # 给定单个日期，提取残余收益率
    def getDateSpecificReturn(self,idt,ids=None):
        return self.RiskDB.loadSpecificReturn(self.TableName,dates=idt,ids=ids)
    # 保存自身信息
    def saveInfo(self,container):
        container = RiskDataSource.saveInfo(self,container)
        container['FactorNames'] = self.FactorNames
        return container
    # 恢复信息
    def loadInfo(self,container):
        Error = RiskDataSource.loadInfo(self,container)
        self.FactorNames = container['FactorNames']
        return Error

# 基于mmap的带并行局部缓冲的因子风险数据源, 如果开启遍历模式, 那么限制缓冲的日期长度, 缓冲区里是部分日期序列数据, 如果未开启, 则调用 FactorRDS 提取数据的方法. 适合遍历数据, 内存消耗小, 首次提取时间不长
def prepareFRDSMMAPCacheData(arg):
    CacheData = {}
    CacheDates = []
    #print("启动缓冲: "+str(arg['CacheSize']))
    while True:
        Task = arg["Queue2SubProcess"].get()
        if Task is None:
            break
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            Msg = None
            if arg["OSType"]=='Windows':
                MMAPCacheData = mmap.mmap(-1,DataLen,tagname=arg["TagName"])# 当前MMAP缓存区
            else:
                MMAPCacheData = mmap.mmap(-1,DataLen)
                Msg = MMAPCacheData
            MMAPCacheData.seek(0)
            MMAPCacheData.write(CacheDataByte)
            CacheDataByte = None
            arg["Queue2MainProcess"].put((DataLen,Msg))
            Msg = None
            gc.collect()
            #print("装入数据:"+str(DataLen))# debug
        else:# 准备缓冲区
            MMAPCacheData = None
            CurInd = Task[0]+arg["SysArgs"]['向前缓冲日期数']+1
            if CurInd<arg["DateNum"]:# 未到结尾处, 需要再准备缓存数据
                OldCacheDates = CacheDates
                CacheDates = arg["Dates"][max((0,CurInd-arg["SysArgs"]['向后缓冲日期数'])):min((arg["DateNum"],CurInd+arg["SysArgs"]['向前缓冲日期数']+1))]
                NewCacheDates = list(set(CacheDates).difference(set(OldCacheDates)))
                NewCacheDates.sort()
                DropDates = list(set(OldCacheDates).difference(set(CacheDates)))
                for iDate in DropDates:
                    CacheData.pop(iDate)
                FactorCov = arg["RiskDB"].loadFactorCov(arg['TableName'],dates=NewCacheDates)
                SpecificRisk = arg['RiskDB'].loadSpecificRisk(arg['TableName'],dates=NewCacheDates)
                FactorData = arg['RiskDB'].loadFactorData(arg['TableName'],dates=NewCacheDates)
                for iDate in NewCacheDates:
                    CacheData[iDate] = {}
                    CacheData[iDate]["FactorCov"] = FactorCov[iDate]
                    CacheData[iDate]["SpecificRisk"] = SpecificRisk.loc[iDate]
                    CacheData[iDate]["FactorData"] = FactorData[iDate]
                #print("准备因子:"+str(list(CacheData.keys())))# debug
    return 0
class ParaMMAPCacheFRDS(FactorRDS,ParaMMAPCacheRDS):
    ClassName = 'ParaMMAPCacheFRDS'
    def __init__(self,name,factor_risk_db=None,qs_env={}):
        ParaMMAPCacheRDS.__init__(self,name,factor_risk_db,qs_env)
        self.DSType = '并行缓冲因子风险数据源(MMAP)'
        self.FactorNames = []# 数据源中所有的因子名，['LNCAP']
        self.CacheFun = prepareFRDSMMAPCacheData
        return
    # 生成系统参数信息集以及初始值
    def genSysArgInfo(self, args=None, arg_changed=None):
        if args is None:
            args = {'向前缓冲日期数':12,'向后缓冲日期数':0}
        ArgInfo = {}
        ArgInfo['向前缓冲日期数'] = {'type':'Integer', "order":0, 'min':0, "max":9999}
        ArgInfo['向后缓冲日期数'] = {'type':'Integer', "order":0, 'min':0, "max":9999}
        return (args,ArgInfo)
    # 恢复数据源到初始状态
    def start(self):
        return ParaMMAPCacheRDS.start(self)
    # 日期向前移动,idt:当前日期,'20090101'
    def MoveOn(self,idt):
        return ParaMMAPCacheRDS.MoveOn(self,idt)
    # 给定单个日期，提取因子风险矩阵
    def getDateFactorCov(self,idt,factor_names=None):
        Data = self.CacheData.get(idt)
        if Data is None:# 非遍历模式或者缓冲区无数据
            Data = self.RiskDB.loadFactorCov(self.TableName,dates=idt)
        else:
            Data = Data.get("FactorCov")
        if Data is None:
            return None
        if factor_names is not None:
            return Data.ix[factor_names,factor_names]
        else:
            return Data
    # 给定单个日期，提取个股的特别风险
    def getDateSpecificRisk(self,idt,ids=None):
        Data = self.CacheData.get(idt)
        if Data is None:# 非遍历模式或者缓冲区无数据
            Data = self.RiskDB.loadSpecificRisk(self.TableName,dates=idt,ids=ids)
        else:
            Data = Data.get("SpecificRisk")
        if Data is None:
            return None
        if ids is not None:
            return Data.ix[ids]
        else:
            return Data
    # 给定单个日期，提取因子截面数据
    def getDateFactorData(self,idt,factor_names=None,ids=None):
        Data = self.CacheData.get(idt)
        if Data is None:# 非遍历模式或者缓冲区无数据
            Data = self.RiskDB.loadFactorData(self.TableName,dates=idt,ids=ids)
        else:
            Data = Data.get("FactorData")
        if Data is None:
            return None
        if ids is not None:
            Data = Data.ix[ids]
        if factor_names is not None:
            Data = Data.ix[:,factor_names]
        return Data
    # 结束数据源，生成特有的结果集
    def endDS(self):
        return ParaMMAPCacheRDS.endDS(self)

RDSClasses = {}
RDSClasses["FRDB"] = OrderedDict()
RDSClasses["FRDB"]["并行缓冲因子风险数据源(MMAP)"] = ParaMMAPCacheFRDS
RDSClasses["FRDB"]["因子风险数据源"] = FactorRDS
RDSClasses["RDB"] = OrderedDict()
RDSClasses["RDB"]["并行缓冲风险数据源(MMAP)"] = ParaMMAPCacheRDS
RDSClasses["RDB"]["风险数据源"] = RiskDataSource