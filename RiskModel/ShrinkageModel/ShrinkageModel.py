# coding=utf-8
"""基于 Shrinkage 的风险模型(未完待续)"""
import time

import pandas as pd
import numpy as np
from progressbar import ProgressBar

import DataSource
import RiskModelFun

# 估计 pi
def estimate_pi(ret,sample_cov):
    nID = sample_cov.shape[0]
    T = ret.shape[0]
    AvgRet = np.nanmean(ret,axis=0)
    PiMatrix = np.zeros(sample_cov.shape)
    SumNum = np.zeros(sample_cov.shape)
    for t in range(T):
        iRet = np.reshape(ret[t]-AvgRet,(1,nID))
        iTemp = (np.dot(iRet.T,iRet)-sample_cov)**2
        iMask = np.isnan(iTemp)
        iTemp[iMask] = 0.0
        PiMatrix += iTemp
        SumNum += (~iMask)
    SumNum[SumNum==0] = np.nan
    return PiMatrix/SumNum
# 估计 rho
def estimate_rho(ret,sample_cov,pi_matrix,avg_corr):
    nID = sample_cov.shape[0]
    T = ret.shape[0]
    AvgRet = np.nanmean(ret,axis=0)
    ThetaMatrix = np.zeros(sample_cov.shape)
    SumNum = np.zeros(sample_cov.shape)
    for t in range(T):
        iRet = np.reshape(ret[t]-AvgRet,(1,nID))
        iTemp = (np.dot(iRet.T,iRet)-sample_cov)*(iRet[0]**2-np.diag(sample_cov))
        iMask = np.isnan(iTemp)
        iTemp[iMask] = 0.0
        ThetaMatrix += iTemp
        SumNum += (~iMask)
    SumNum[SumNum==0] = np.nan
    ThetaMatrix = ThetaMatrix/SumNum
    s = np.reshape(np.diag(sample_cov)**0.5,(1,nID))
    s = np.dot(s.T,1/s)
    iTemp = (ThetaMatrix.T/s+ThetaMatrix*s)/2*avg_corr
    return np.nansum(np.diag(pi_matrix)) + np.nansum(iTemp)-np.nansum(np.diag(iTemp))
# 估计 gamma
def estimate_gamma(sample_cov,shrinkage_target):
    return np.nansum((shrinkage_target-sample_cov)**2)
# 估计协方差矩阵
def CovarianceGeneration(args,qs_env):
    ReturnDates = pd.Series(args["DS"].getDateTime())
    args["DS"].start()
    iReturn = None
    iReturnDates = []
    SampleFilterFactors = []
    SampleFilterStr = args["CovESTArgs"]["有效样本条件"]
    FactorNames = args["DS"].FactorNames.copy()
    FactorNames.sort(key=len,reverse=True)
    for iFactor in FactorNames:
        if SampleFilterStr.find('@'+iFactor)!=-1:
            SampleFilterFactors.append(iFactor)
            SampleFilterStr = SampleFilterStr.replace('@'+iFactor, 'iData[\''+iFactor+'\']')
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        ProgBar = ProgressBar(max_value=len(args["RiskESTDates"]))
        ProgBar.start()
    else:
        ProgBar = None
    for i,iDate in enumerate(args["RiskESTDates"]):
        iInd = (ReturnDates<=iDate).sum()-1
        if iInd<args["CovESTArgs"]["样本长度"]-1:# 样本不足, 跳过
            if ProgBar is not None:
                ProgBar.update(i+1)
            else:
                args['Sub2MainQueue'].put((qs_env.PID,1,None))
            continue
        args["DS"].MoveOn(iDate)
        iIDs = args["DS"].getID(idt=iDate,is_filtered=True)
        iLastDates = iReturnDates
        iReturnDates = list(ReturnDates.iloc[iInd-args["CovESTArgs"]["样本长度"]+1:iInd+1])
        iNewDates = list(set(iReturnDates).difference(set(iLastDates)))
        iNewDates.sort()
        if iReturn is not None:
            iReturn = pd.concat([iReturn,args["DS"].getFactorData(ifactor_name=args["ModelArgs"]["收益率因子"],dates=iNewDates)]).loc[iReturnDates,:]
            for jFactor in SampleFilterFactors:
                iData[jFactor] = pd.concat([iData[jFactor],args["DS"].getFactorData(ifactor_name=jFactor,dates=iNewDates)]).loc[iReturnDates,:]
        else:
            iReturn = args["DS"].getFactorData(ifactor_name=args["ModelArgs"]["收益率因子"],dates=iNewDates).loc[iReturnDates,:]
            iData = {}
            for jFactor in SampleFilterFactors:
                iData[jFactor] = args["DS"].getFactorData(ifactor_name=jFactor,dates=iNewDates).loc[iReturnDates,:]
        iMask = eval(SampleFilterStr)
        iReturn[~iMask] = np.nan
        iReturnArray = iReturn.loc[:,iIDs].values
        iSampleCov = RiskModelFun.estimateSampleCovMatrix_EWMA(iReturnArray, forcast_num=1, half_life=args["CovESTArgs"]["半衰期"])
        iAvgCorr = RiskModelFun.calcAvgCorr(iSampleCov)
        iShrinkageTarget = (np.ones(iSampleCov.shape)-np.eye(iSampleCov.shape[0]))*iAvgCorr+np.eye(iSampleCov.shape[0])
        iVol = np.diag(iSampleCov)**0.5
        iShrinkageTarget = (iShrinkageTarget*iVol).T*iVol
        iPiMatrix = estimate_pi(iReturnArray,iSampleCov)
        pi = np.nansum(iPiMatrix)
        rho = estimate_rho(iReturnArray,iSampleCov,iPiMatrix,iAvgCorr)
        gamma = estimate_gamma(iSampleCov,iShrinkageTarget)
        kappa = (pi-rho)/gamma
        T = iReturnArray.shape[0]
        delta = max((0,min((kappa/T,1))))
        iCov = (delta*iShrinkageTarget+(1-delta)*iSampleCov)*args["CovESTArgs"]["预测期数"]
        iCov = pd.DataFrame(iCov,index=iIDs,columns=iIDs)
        args["RiskDB"].saveData(args["TargetTable"],iDate,cov=iCov)
        if ProgBar is not None:
            ProgBar.update(i+1)
        else:
            args['Sub2MainQueue'].put((qs_env.PID,1,None))
    if ProgBar is not None:
        ProgBar.finish()
    args["DS"].endDS()
    if args["ModelArgs"]['运行模式']!='串行':
        qs_env.closeResource()
    return 0


class ShrinkageModel(object):
    """基于 Shrinkage 的风险模型"""
    def __init__(self,name,config_file=None,qs_env=None):
        self.ModelType = "基于 Shrinkage 的风险模型"
        # 需要预先指定的属性
        self.Name = name
        self.QSEnv = qs_env
        self.Config = self.QSEnv.loadConfigFile(config_file)
        self.RiskESTDates = []# 估计风险的日期序列
        # 模型的其他属性
        self.RiskDB = None# 风险数据库
        self.TargetTable = None# 风险数据存储的目标表
        self.DS = None# 提供因子数据的数据源
        return
    # 设置计算风险估计的日期序列
    def setRiskESTDate(self,dates):
        self.RiskESTDates = dates
        self.RiskESTDates.sort()
        return 0
    # 初始化
    def initInfo(self):
        if self.RiskESTDates==[]:
            self.QSEnv.SysArgs['LastErrorMsg'] = "没有设置计算风险数据的日期序列!"
            return 0
        # 获得风险数据库
        self.RiskDB = getattr(self.QSEnv,self.Config.SaveArgs["风险数据库"])
        self.TargetTable = self.Config.SaveArgs["风险数据表"]
        # 创建数据源
        FactorDB = getattr(self.QSEnv,self.Config.ModelArgs['因子数据库'])
        self.DS = DataSource.DataSource("MainDS",FactorDB,self.QSEnv)
        self.DS.prepareData(self.Config.DSTableFactor)
        self.DS.setIDFilter(self.Config.ModelArgs.get("ID过滤条件",None))
        self.DS.SysArgs.update(getattr(self.Config,"DSSysArgs",{}))
        if self.RiskESTDates==[]:
            self.QSEnv.SysArgs['LastErrorMsg'] = "可以计算风险数据的日期序列为空!"
            return 0
        return 0
    # 生成协方差矩阵
    def _genCovariance(self):
        Args = {"RiskDB":self.RiskDB,
                "DS":self.DS,
                "RiskESTDates":self.RiskESTDates,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs,
                "CovESTArgs":self.Config.CovESTArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            CovarianceGeneration(Args,self.QSEnv)
        else:
            nTask = len(self.RiskESTDates)
            nPrcs = min((nTask,self.Config.ModelArgs["子进程数"]))
            ProgBar = ProgressBar(max_value=nTask)
            Procs,Main2SubQueue,Sub2MainQueue = self.QSEnv.startMultiProcess(n_prc=nPrcs, target_fun=CovarianceGeneration,
                                                                             arg=Args, partition_arg="RiskESTDates",
                                                                             main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            ProgBar.start()
            while (iProg<nTask):
                iPID,iErrorCode,iMsg = Sub2MainQueue.get()
                if iErrorCode==-1:
                    for iProc in Procs:
                        if iProc.is_alive():
                            iProc.terminate()
                    print('进程 '+iPID+' :运行失败:'+str(iMsg))
                    break
                else:
                    iProg += 1
                    ProgBar.update(iProg)
            ProgBar.finish()
            for iPID,iPrcs in Procs.items():
                iPrcs.join()
        return 0
    # 生成数据
    def genData(self):
        print("风险数据计算中...")
        StartT = time.clock()
        self._genCovariance()
        print("风险数据计算完成, 耗时 : %.2f" % (time.clock()-StartT))
        self.RiskDB.connect()
        return 0