# coding=utf-8
"""基于 Shrinkage 的风险模型(TODO)"""
import time
import os
import imp

import pandas as pd
import numpy as np
from progressbar import ProgressBar

from . import RiskModelFun
from QuantStudio import __QS_LibPath__, __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import startMultiProcess

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
def _CovarianceGeneration(args):
    ReturnDTs = pd.Series(args["FT"].getDateTime())
    args["FT"].start()
    iReturn = None
    iReturnDTs = []
    SampleFilterFactors = []
    SampleFilterStr = args["CovESTArgs"]["有效样本条件"]
    FactorNames = args["FT"].FactorNames.copy()
    FactorNames.sort(key=len,reverse=True)
    for iFactor in FactorNames:
        if SampleFilterStr.find('@'+iFactor)!=-1:
            SampleFilterFactors.append(iFactor)
            SampleFilterStr = SampleFilterStr.replace('@'+iFactor, 'iData[\''+iFactor+'\']')
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        ProgBar = ProgressBar(max_value=len(args["RiskESTDTs"]))
        ProgBar.start()
    else:
        ProgBar = None
    for i, iDT in enumerate(args["RiskESTDTs"]):
        iInd = (ReturnDTs<=iDT).sum()-1
        if iInd<args["CovESTArgs"]["样本长度"]-1:# 样本不足, 跳过
            if ProgBar is not None: ProgBar.update(i+1)
            else: args['Sub2MainQueue'].put((args["PID"], 1, None))
            continue
        args["FT"].move(iDT)
        iIDs = args["FT"].getID(idt=iDT,is_filtered=True)
        iLastDTs = iReturnDTs
        iReturnDTs = list(ReturnDTs.iloc[iInd-args["CovESTArgs"]["样本长度"]+1:iInd+1])
        iNewDTs = sorted(set(iReturnDTs).difference(set(iLastDTs)))
        if iReturn is not None:
            iReturn = pd.concat([iReturn,args["FT"].readData(factor_names=[args["ModelArgs"]["收益率因子"]], dts=iNewDTs)]).iloc[0].loc[iReturnDTs,:]
            for jFactor in SampleFilterFactors:
                iData[jFactor] = pd.concat([iData[jFactor],args["FT"].readData(factor_names=[jFactor], dts=iNewDTs)]).iloc[0].loc[iReturnDTs,:]
        else:
            iReturn = args["FT"].readData(factor_names=[args["ModelArgs"]["收益率因子"]], dts=iNewDTs).iloc[0].loc[iReturnDTs,:]
            iData = {}
            for jFactor in SampleFilterFactors:
                iData[jFactor] = args["FT"].readData(factor_names=[jFactor], dts=iNewDTs).iloc[0].loc[iReturnDTs,:]
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
        args["RiskDB"].writeData(args["TargetTable"], iDT, cov=iCov)
        if ProgBar is not None: ProgBar.update(i+1)
        else: args['Sub2MainQueue'].put((args["PID"], 1, None))
    if ProgBar is not None: ProgBar.finish()
    args["FT"].end()
    return 0


class ShrinkageModel(object):
    """基于 Shrinkage 的风险模型"""
    def __init__(self, name, factor_table, risk_db, table_name, config_file=None):
        self.ModelType = "基于 Shrinkage 的风险模型"
        self.Name = name
        self.RiskESTDTs = []# 估计风险的时点序列
        self.RiskDB = risk_db# 风险数据库
        self.TargetTable = table_name# 风险数据存储的目标表
        self.FT = factor_table# 提供因子数据的因子表
        if config_file is None: config_file = __QS_LibPath__+os.sep+"ShrinkageModelConfig.py"
        self.Config = imp.load_module(config_file, *imp.find_module(config_file, [os.path.split(config_file)[0]]))
        return
    # 设置计算风险估计的时点序列
    def setRiskESTDateTime(self, dts):
        self.RiskESTDTs = sorted(dts)
        return 0
    # 初始化
    def _initInfo(self):
        if self.RiskESTDTs==[]: raise __QS_Error__("没有设置计算风险数据的时点序列!")
        FactorNames = set(self.Config.ModelArgs["所有因子"])
        if not set(self.FT.FactorNames).issuperset(FactorNames): raise __QS_Error__("因子表必须包含如下因子: %s" % FactorNames)
        return 0
    # 生成协方差矩阵
    def _genCovariance(self):
        Args = {"RiskDB":self.RiskDB,
                "FT":self.FT,
                "RiskESTDTs":self.RiskESTDTs,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs,
                "CovESTArgs":self.Config.CovESTArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            _CovarianceGeneration(Args)
        else:
            nTask = len(self.RiskESTDTs)
            nPrcs = min(nTask, self.Config.ModelArgs["子进程数"])
            ProgBar = ProgressBar(max_value=nTask)
            Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_CovarianceGeneration,
                                                                    arg=Args, partition_arg=["RiskESTDTs"],
                                                                    main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            ProgBar.start()
            while (iProg<nTask):
                iPID,iErrorCode,iMsg = Sub2MainQueue.get()
                if iErrorCode==-1:
                    for iProc in Procs:
                        if iProc.is_alive(): iProc.terminate()
                    print('进程 '+iPID+' :运行失败:'+str(iMsg))
                    break
                else:
                    iProg += 1
                    ProgBar.update(iProg)
            ProgBar.finish()
            for iPID,iPrcs in Procs.items(): iPrcs.join()
        return 0
    # 生成数据
    def run(self):
        TotalStartT = time.clock()
        print("==========基于 Shrinkage 的风险模型==========", "1. 初始化", sep="\n", end="\n")
        self._initInfo()
        print(('耗时 : %.2f' % (time.clock()-TotalStartT, )), "2. 估计协方差矩阵", sep="\n", end="\n")
        StartT = time.clock()
        self._genCovariance()
        print("耗时 : %.2f" % (time.clock()-StartT, ), ("总耗时 : %.2f" % (time.clock()-TotalStartT, )), "="*28, sep="\n", end="\n")
        return 0