# coding=utf-8
import os
import imp
import time
import logging

import pandas as pd
import numpy as np
from progressbar import ProgressBar

from . import RiskModelFun
from QuantStudio import __QS_LibPath__, __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import startMultiProcess

# 截面回归生成因子收益率和特异性收益率
def _FactorAndSpecificReturnGeneration(args):
    FT = args["FT"]
    FT.start(dts=args["RegressDTs"])
    DSDTs = FT.getDateTime()
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        nTask = len(args["RegressDTs"])
        with ProgressBar(max_value=nTask) as ProgBar:
            IDs = FT.getID(ifactor_name=args["ModelArgs"]['ESTU因子'])
            for i, iDT in enumerate(args["RegressDTs"]):
                FT.move(iDT)
                iInd = DSDTs.index(iDT)
                if iInd==0: continue
                iPreDT = DSDTs[iInd-1]
                iESTU = FT.readData(dts=[iPreDT], ids=IDs, factor_names=[args["ModelArgs"]['ESTU因子']]).iloc[0,0,:]
                iRet = FT.readData(dts=[iDT], ids=IDs, factor_names=[args["ModelArgs"]['收益率因子']]).iloc[0,0,:]
                iCap = FT.readData(dts=[iPreDT], ids=IDs, factor_names=[args["ModelArgs"]['市值因子']]).iloc[0,0,:]
                iIndustry = FT.readData(dts=[iPreDT], ids=IDs, factor_names=[args["ModelArgs"]['行业因子']]).iloc[0,0,:]
                iFactorData = FT.readData(dts=[iPreDT], ids=IDs, factor_names=args["ModelArgs"]['风格因子']).iloc[:,0,:]
                iFactorReturn, iSpecificReturn, iFactorData, iStatistics = RiskModelFun.estimateFactorAndSpecificReturn_EUE3(iRet, iFactorData, iIndustry, iCap, iESTU, iCap, args["ModelArgs"]['所有行业'])
                args["RiskDB"].writeData(args["TargetTable"], iDT, factor_ret=iFactorReturn, specific_ret=iSpecificReturn, Statistics=pd.Series(iStatistics).sort_index())
                args["RiskDB"].writeData(args["TargetTable"], iPreDT, factor_data=iFactorData, Cap=iCap)
                ProgBar.update(i+1)
    else:
        IDs = FT.getID(ifactor_name=args["ModelArgs"]['ESTU因子'])
        for i, iDT in enumerate(args["RegressDTs"]):
            FT.move(iDT)
            iInd = DSDTs.index(iDT)
            if iInd==0: continue
            iPreDT = DSDTs[iInd-1]
            iESTU = FT.readData(dts=[iPreDT], ids=IDs, factor_names=[args["ModelArgs"]['ESTU因子']]).iloc[0,0,:]
            iRet = FT.readData(dts=[iDT], ids=IDs, factor_names=[args["ModelArgs"]['收益率因子']]).iloc[0,0,:]
            iCap = FT.readData(dts=[iPreDT], ids=IDs, factor_names=[args["ModelArgs"]['市值因子']]).iloc[0,0,:]
            iIndustry = FT.readData(dts=[iPreDT], ids=IDs, factor_names=[args["ModelArgs"]['行业因子']]).iloc[0,0,:]
            iFactorData = FT.readData(dts=[iPreDT], ids=IDs, factor_names=args["ModelArgs"]['风格因子']).iloc[:,0,:]
            iFactorReturn, iSpecificReturn, iFactorData, iStatistics = RiskModelFun.estimateFactorAndSpecificReturn_EUE3(iRet, iFactorData, iIndustry, iCap, iESTU, iCap, args["ModelArgs"]['所有行业'])
            args["RiskDB"].writeData(args["TargetTable"], iDT, factor_ret=iFactorReturn, specific_ret=iSpecificReturn, Statistics=pd.Series(iStatistics).sort_index())
            args["RiskDB"].writeData(args["TargetTable"], iPreDT, factor_data=iFactorData, Cap=iCap)
            args['Sub2MainQueue'].put((args["PID"], 1, None))
    if args["RegressDTs"]!=[]:
        iESTU = FT.readData(dts=[iDT], ids=IDs, factor_names=[args["ModelArgs"]['ESTU因子']]).iloc[0,0,:]
        iCap = FT.readData(dts=[iDT], ids=IDs, factor_names=[args["ModelArgs"]['市值因子']]).iloc[0,0,:]
        iIndustry = FT.readData(dts=[iDT], ids=IDs, factor_names=[args["ModelArgs"]['行业因子']]).iloc[0,0,:]
        iFactorData = FT.readData(dts=[iDT], ids=IDs, factor_names=args["ModelArgs"]['风格因子']).iloc[:,0,:]
        iFactorReturn, iSpecificReturn, iFactorData, iStatistics = RiskModelFun.estimateFactorAndSpecificReturn_EUE3(iRet, iFactorData, iIndustry, iCap, iESTU, iCap, args["ModelArgs"]['所有行业'])
        args["RiskDB"].writeData(args["TargetTable"], iDT, factor_data=iFactorData, Cap=iCap)
    FT.end()
    return 0

# 估计因子协方差矩阵
def _FactorCovarianceGeneration(args):
    RT = args["RiskDB"].getTable(args["TargetTable"])
    FactorReturnDTs = pd.Series(RT.getFactorReturnDateTime())
    iFactorReturn = None
    iFactorReturnDTs = []
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        ProgBar = ProgressBar(max_value=len(args["RiskESTDTs"]))
        ProgBar.start()
    else:
        ProgBar = None
    for i, iDT in enumerate(args["RiskESTDTs"]):
        iInd = (FactorReturnDTs<=iDT).sum()-1
        if iInd<args["FactorCovESTArgs"]["样本长度"]-1:# 样本不足, 跳过
            if ProgBar is not None: ProgBar.update(i+1)
            else: args['Sub2MainQueue'].put((args["PID"], 1, None))
            continue
        iLastDTs = iFactorReturnDTs
        iFactorReturnDTs = list(FactorReturnDTs.iloc[iInd-args["FactorCovESTArgs"]["样本长度"]+1:iInd+1])
        iNewDTs = sorted(set(iFactorReturnDTs).difference(set(iLastDTs)))
        if iFactorReturn is not None:
            iFactorReturn = pd.concat([iFactorReturn, RT.readFactorReturn(dts=iNewDTs)]).loc[iFactorReturnDTs, :]
        else:
            iFactorReturn = RT.readFactorReturn(dts=iNewDTs).loc[iFactorReturnDTs, :]
        iFactorCov = RiskModelFun.estimateFactorCov_CHE2(iFactorReturn, forcast_num=args["FactorCovESTArgs"]["预测期数"],
                                                         auto_corr_num=args["FactorCovESTArgs"]["自相关滞后期"],
                                                         half_life_corr=args["FactorCovESTArgs"]["相关系数半衰期"],
                                                         half_life_vol=args["FactorCovESTArgs"]["波动率半衰期"])
        if args["ModelArgs"]["EigenfactorRiskAdjustment"]:
            iFactorCov = RiskModelFun.EigenfactorRiskAdjustment(iFactorCov,
                                                                monte_carlo_num=args["EigenfactorRiskAdjustmentArgs"]["MonteCarlo次数"],
                                                                date_num=args["EigenfactorRiskAdjustmentArgs"]["模拟时点长度"],
                                                                ignore_num=args["EigenfactorRiskAdjustmentArgs"]["拟合忽略样本数"],
                                                                a=args["EigenfactorRiskAdjustmentArgs"]["a"],
                                                                forcast_num=args["FactorCovESTArgs"]["预测期数"],
                                                                auto_corr_num=args["FactorCovESTArgs"]["自相关滞后期"],
                                                                half_life_corr=args["FactorCovESTArgs"]["相关系数半衰期"],
                                                                half_life_vol=args["FactorCovESTArgs"]["波动率半衰期"])
        args["RiskDB"].writeData(args["TargetTable"], iDT, factor_cov=iFactorCov)
        if ProgBar is not None: ProgBar.update(i+1)
        else: args['Sub2MainQueue'].put((args["PID"], 1, None))
    if ProgBar is not None: ProgBar.finish()
    return 0

# 估计特异性风险
def _SpecificRiskGeneration(args):
    RT = args["RiskDB"].getTable(args["TargetTable"])
    SpecificReturnDTs = pd.Series(RT.getSpecificReturnDateTime())
    iSpecificReturnDTs = []
    iSpecificReturn = None
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        ProgBar = ProgressBar(max_value=len(args["RiskESTDTs"]))
        ProgBar.start()
    else:
        ProgBar = None
    for i, iDT in enumerate(args["RiskESTDTs"]):
        iInd = (SpecificReturnDTs<=iDT).sum()-1
        if iInd<args["SpecificRiskESTArgs"]["样本长度"]-1:# 样本不足, 跳过
            if ProgBar is not None: ProgBar.update(i+1)
            else: args['Sub2MainQueue'].put((args["PID"], 1, None))
            continue
        iLastDTs = iSpecificReturnDTs
        iSpecificReturnDTs = list(SpecificReturnDTs.iloc[iInd-args["SpecificRiskESTArgs"]["样本长度"]+1:iInd+1])
        iNewDTs = sorted(set(iSpecificReturnDTs).difference(iLastDTs))
        if iSpecificReturn is None:
            iSpecificReturn = RT.readSpecificReturn(dts=iNewDTs).loc[iSpecificReturnDTs, :]
        else:
            iSpecificReturn = pd.concat([iSpecificReturn, RT.readSpecificReturn(dts=iNewDTs)]).loc[iSpecificReturnDTs, :]
        iFactorData = RT.readFactorData(dts=[iDT]).iloc[:, 0, :]
        iCap = RT.readData("Cap", dts=[iDT]).iloc[0]
        iFactorData = iFactorData.loc[iSpecificReturn.columns, args["SpecificRiskESTArgs"]["结构化模型回归风格因子"]+args["ModelArgs"]["所有行业"]]
        iCap = iCap.loc[iSpecificReturn.columns]
        iSpecificRisk = RiskModelFun.estimateSpecificRisk_EUE3(specific_ret=iSpecificReturn, factor_data=iFactorData, cap=iCap,
                                                               forcast_num=args["SpecificRiskESTArgs"]["预测期数"],
                                                               auto_corr_num=args["SpecificRiskESTArgs"]["自相关滞后期"],
                                                               half_life=args["FactorCovESTArgs"]["波动率半衰期"])
        args["RiskDB"].writeData(args["TargetTable"], iDT, specific_risk=iSpecificRisk)
        if ProgBar is not None: ProgBar.update(i+1)
        else: args['Sub2MainQueue'].put((args["PID"], 1, None))
    if ProgBar is not None: ProgBar.finish()
    return 0


class BarraModel(object):
    """Barra 风险模型"""
    def __init__(self, name, factor_table, risk_db, table_name, config_file=None, **kwargs):
        if "logger" in kwargs: self._QS_Logger = kwargs.pop("logger")
        else: self._QS_Logger = logging.getLogger(__name__)
        self.ModelType = "多因子风险模型"
        self.Name = name
        if config_file is None: config_file = __QS_LibPath__+os.sep+"BarraModelConfig.py"
        ModulePath, ConfigModule = os.path.split(config_file)
        ConfigModule = ".".join(ConfigModule.split(".")[:-1])
        self.Config = imp.load_module(config_file, *imp.find_module(ConfigModule, [ModulePath]))
        self.RiskESTDTs = []# 估计风险的时点序列
        self.RegressDTs = None# 进行截面回归的时点序列
        self.RiskDB = risk_db# 风险数据库
        self.TargetTable = table_name# 风险数据存储的目标表
        self.FT = factor_table# 提供因子数据的因子表
        return
    # 设置计算风险估计的时点序列
    def setRiskESTDateTime(self, dts):
        self.RiskESTDTs = sorted(dts)
        return 0
    # 生成回归时点序列
    def _genRegressDateTime(self):
        DSDTs = pd.Series(self.FT.getDateTime())
        RiskESTStartInd = max(((DSDTs<=self.RiskESTDTs[0]).sum()-1,0))
        if self.Config.FactorCovESTArgs["样本长度"]==-1:
            StartInd = 0
        else:
            StartInd = max(RiskESTStartInd-self.Config.FactorCovESTArgs["样本长度"]+1, 0)
        if self.Config.SpecificRiskESTArgs['样本长度']==-1:
            StartInd = 0
        else:
            StartInd = min(max(RiskESTStartInd-self.Config.SpecificRiskESTArgs["样本长度"]+1, 0), StartInd)
        self.RegressDTs = DSDTs.iloc[StartInd:]
        self.RegressDTs = list(self.RegressDTs[self.RegressDTs<=self.RiskESTDTs[-1]])
        if self.TargetTable in self.RiskDB.TableNames:
            OldDTs = self.RiskDB.getTable(self.TargetTable).getFactorReturnDateTime()
            self.RegressDTs = sorted(set(self.RegressDTs).difference(OldDTs))
        return 0
    # 调整风险数据的计算时点序列
    def _adjustRiskESTDateTime(self):
        AllReturnDTs = self.RegressDTs
        if self.TargetTable in self.RiskDB.TableNames:
            AllReturnDTs = sorted(set(self.RiskDB.getTable(self.TargetTable).getFactorReturnDateTime()).union(AllReturnDTs))
        RequiredLen = max(self.Config.FactorCovESTArgs["样本长度"], self.Config.SpecificRiskESTArgs["样本长度"])
        for i, iDT in enumerate(self.RiskESTDTs):
            iInd = AllReturnDTs.index(iDT)
            if iInd>=RequiredLen-1: break
        if iInd>=RequiredLen-1:
            self.RiskESTDTs = self.RiskESTDTs[i:]
        else:
            self.RiskESTDTs = []
        return 0
    # 初始化
    def _initInfo(self):
        if self.RiskESTDTs==[]: raise __QS_Error__("没有设置计算风险数据的时点序列!")
        FactorNames = set(self.Config.ModelArgs["所有因子"])
        if not set(self.FT.FactorNames).issuperset(FactorNames): raise __QS_Error__("因子表必须包含如下因子: %s" % FactorNames)
        self._genRegressDateTime()
        self._adjustRiskESTDateTime()
        if self.RiskESTDTs==[]: raise __QS_Error__("可以计算风险数据的时点序列为空!")
        return 0
    # 生成因子收益率以及特异性收益率
    def _genFactorAndSpecificReturn(self):
        Args = {"FT":self.FT,
                "RiskDB":self.RiskDB,
                "RegressDTs":self.RegressDTs,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            _FactorAndSpecificReturnGeneration(Args)
        else:
            nTask = len(self.RegressDTs)
            nPrcs = min(nTask, self.Config.ModelArgs["子进程数"])
            Procs,Main2SubQueue,Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_FactorAndSpecificReturnGeneration,
                                                                  arg=Args, partition_arg=["RegressDTs"], n_partition_tail=1,
                                                                  main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            with ProgressBar(max_value=nTask) as ProgBar:
                while (iProg<nTask):
                    iPID, iErrorCode, iMsg = Sub2MainQueue.get()
                    if iErrorCode==-1:
                        for iProc in Procs:
                            if iProc.is_alive(): iProc.terminate()
                        self._QS_Logger.error("进程: "+iPID+" 运行失败: "+str(iMsg))
                        break
                    else:
                        iProg += 1
                        ProgBar.update(iProg)
            for iPID,iPrcs in Procs.items(): iPrcs.join()
        return 0
    # 生成因子协方差矩阵
    def _genFactorCovariance(self):
        Args = {"RiskDB":self.RiskDB,
                "RiskESTDTs":self.RiskESTDTs,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs,
                "FactorCovESTArgs":self.Config.FactorCovESTArgs,
                "EigenfactorRiskAdjustmentArgs":self.Config.EigenfactorRiskAdjustmentArgs,
                "FactorVolatilityRegimeAdjustmentArgs":self.Config.FactorVolatilityRegimeAdjustmentArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            _FactorCovarianceGeneration(Args)
        else:
            nTask = len(self.RiskESTDTs)
            nPrcs = min((nTask,self.Config.ModelArgs["子进程数"]))
            ProgBar = ProgressBar(max_value=nTask)
            Procs,Main2SubQueue,Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_FactorCovarianceGeneration,
                                                                  arg=Args, partition_arg=["RiskESTDTs"],
                                                                  main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            ProgBar.start()
            while (iProg<nTask):
                iPID,iErrorCode,iMsg = Sub2MainQueue.get()
                if iErrorCode==-1:
                    for iProc in Procs:
                        if iProc.is_alive(): iProc.terminate()
                    self._QS_Logger.error("进程 "+iPID+" 运行失败: "+str(iMsg))
                    break
                else:
                    iProg += 1
                    ProgBar.update(iProg)
            ProgBar.finish()
            for iPID,iPrcs in Procs.items():
                iPrcs.join()
        #if self.Config.ModelArgs["FactorVolatilityRegimeAdjustment"]:# TODO
            #print("Factor Volatility Regime Adjustment进行中...")
            #FactorReturnDates = pd.Series(self.RiskDB.getFactorReturnDateTime(self.TargetTable))
            #iFactorReturn = None
            #iFactorReturnDates = []
            #FactorVolatility = []# 用于FactorVolatilityRegimeAdjustment
            #iSampleDates = []# 用于FactorVolatilityRegimeAdjustment
            #for iDate in self.RiskESTDTs:
                #iLastDates = iSampleDates
                #iSampleDates = self.getTableDate(self.TargetTable,end_date=iDate)
                #if self.Config.FactorVolatilityRegimeAdjustmentArgs["样本长度"]>0:
                    #iSampleDates = iSampleDates[max((0,len(iSampleDates)-1-self.Config.FactorVolatilityRegimeAdjustmentArgs["样本长度"])):]
                #iNewSampleDates = list(set(iSampleDates).difference(set(iLastDates)))
                #iNewSampleDates.sort()
                #FactorVolatility = self.RiskDB.readFactorReturn(self.TargetTable,dates=None)
                #iFactorVolatility = pd.Series(np.diag(iFactorCov)**0.5)
            #print("Factor Volatility Regime Adjustment完成")
        return 0
    # 生成特异性风险
    def _genSpecificRisk(self):
        Args = {"RiskDB":self.RiskDB,
                "RiskESTDTs":self.RiskESTDTs,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs,
                "FactorCovESTArgs":self.Config.FactorCovESTArgs,
                "SpecificRiskESTArgs":self.Config.SpecificRiskESTArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            _SpecificRiskGeneration(Args)
        else:
            nTask = len(self.RiskESTDTs)
            nPrcs = min((nTask,self.Config.ModelArgs["子进程数"]))
            ProgBar = ProgressBar(max_value=nTask)
            Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_SpecificRiskGeneration,
                                                                    arg=Args, partition_arg=["RiskESTDTs"],
                                                                    main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            ProgBar.start()
            while (iProg<nTask):
                iPID,iErrorCode,iMsg = Sub2MainQueue.get()
                if iErrorCode==-1:
                    for iProc in Procs:
                        if iProc.is_alive(): iProc.terminate()
                    self._QS_Logger.error("进程 "+iPID+" 运行失败: "+str(iMsg))
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
        print("==========Barra 风险模型==========", "1. 初始化", sep="\n")
        self._initInfo()
        print(('耗时 : %.2f' % (time.clock()-TotalStartT, )), "2. 截面回归", sep="\n")
        StartT = time.clock()
        self._genFactorAndSpecificReturn()
        print("耗时 : %.2f" % (time.clock()-StartT, ), "3. 估计因子协方差矩阵", sep="\n")
        StartT = time.clock()
        self._genFactorCovariance()
        print("耗时 : %.2f" % (time.clock()-StartT, ), "4. 估计特异性风险", sep="\n")
        StartT = time.clock()
        self._genSpecificRisk()
        print("耗时 : %.2f" % (time.clock()-StartT, ), ("总耗时 : %.2f" % (time.clock()-TotalStartT, )), "="*28, sep="\n")
        return 0