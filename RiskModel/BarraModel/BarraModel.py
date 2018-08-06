# coding=utf-8
import time

import pandas as pd
import numpy as np
from progressbar import ProgressBar

import DataSource
import RiskModelFun

# 截面回归生成因子收益率和特异性收益率
def FactorAndSpecificReturnGeneration(args,qs_env):
    DS = args["DS"]
    DS.start()
    DSDates = DS.getDateTime()
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        nTask = len(args["RegressDates"])
        with ProgressBar(max_value=nTask) as ProgBar:
            for i,iDate in enumerate(args["RegressDates"]):
                DS.MoveOn(iDate)
                iInd = DSDates.index(iDate)
                if iInd==0:
                    continue
                iPreDate = DSDates[iInd-1]
                iESTU = DS.getDateTimeData(iPreDate,factor_names=[args["ModelArgs"]['ESTU因子']])[args["ModelArgs"]['ESTU因子']]
                iRet = DS.getDateTimeData(iDate,factor_names=[args["ModelArgs"]['收益率因子']])[args["ModelArgs"]['收益率因子']]
                iCap = DS.getDateTimeData(iPreDate,factor_names=[args["ModelArgs"]['市值因子']])[args["ModelArgs"]['市值因子']]
                iIndustry = DS.getDateTimeData(iPreDate,factor_names=[args["ModelArgs"]['行业因子']])[args["ModelArgs"]['行业因子']]
                iFactorData = DS.getDateTimeData(iPreDate,factor_names=args["ModelArgs"]['风格因子'])
                iFactorReturn,iSpecificReturn,iFactorData,iStatistics = RiskModelFun.estimateFactorAndSpecificReturn_EUE3(iRet, iFactorData, iIndustry, iCap, iESTU, iCap, args["ModelArgs"]['所有行业'])
                args["RiskDB"].saveData(args["TargetTable"],iDate,factor_ret=iFactorReturn,specific_ret=iSpecificReturn,file_value={"Statistics":iStatistics})
                args["RiskDB"].saveData(args["TargetTable"],iPreDate,factor_data=iFactorData,file_value={"Cap":iCap})
                ProgBar.update(i+1)
    else:
        for i,iDate in enumerate(args["RegressDates"]):
            DS.MoveOn(iDate)
            iInd = DSDates.index(iDate)
            if iInd==0:
                continue
            iPreDate = DSDates[iInd-1]
            iESTU = DS.getDateTimeData(iPreDate,factor_names=[args["ModelArgs"]['ESTU因子']])[args["ModelArgs"]['ESTU因子']]
            iRet = DS.getDateTimeData(iDate,factor_names=[args["ModelArgs"]['收益率因子']])[args["ModelArgs"]['收益率因子']]
            iCap = DS.getDateTimeData(iPreDate,factor_names=[args["ModelArgs"]['市值因子']])[args["ModelArgs"]['市值因子']]
            iIndustry = DS.getDateTimeData(iPreDate,factor_names=[args["ModelArgs"]['行业因子']])[args["ModelArgs"]['行业因子']]
            iFactorData = DS.getDateTimeData(iPreDate,factor_names=args["ModelArgs"]['风格因子'])
            iFactorReturn,iSpecificReturn,iFactorData,iStatistics = RiskModelFun.estimateFactorAndSpecificReturn_EUE3(iRet, iFactorData, iIndustry, iCap, iESTU, iCap, args["ModelArgs"]['所有行业'])
            args["RiskDB"].saveData(args["TargetTable"],iDate,factor_ret=iFactorReturn,specific_ret=iSpecificReturn,file_value={"Statistics":iStatistics})
            args["RiskDB"].saveData(args["TargetTable"],iPreDate,factor_data=iFactorData,file_value={"Cap":iCap})
            args['Sub2MainQueue'].put((qs_env.PID,1,None))
        qs_env.closeResource()
    if args["RegressDates"]!=[]:
        iESTU = DS.getDateTimeData(iDate,factor_names=[args["ModelArgs"]['ESTU因子']])[args["ModelArgs"]['ESTU因子']]
        iCap = DS.getDateTimeData(iDate,factor_names=[args["ModelArgs"]['市值因子']])[args["ModelArgs"]['市值因子']]
        iIndustry = DS.getDateTimeData(iDate,factor_names=[args["ModelArgs"]['行业因子']])[args["ModelArgs"]['行业因子']]
        iFactorData = DS.getDateTimeData(iDate,factor_names=args["ModelArgs"]['风格因子'])
        iFactorReturn,iSpecificReturn,iFactorData,iStatistics = RiskModelFun.estimateFactorAndSpecificReturn_EUE3(iRet, iFactorData, iIndustry, iCap, iESTU, iCap, args["ModelArgs"]['所有行业'])
        args["RiskDB"].saveData(args["TargetTable"],iDate,factor_data=iFactorData,file_value={"Cap":iCap})
    DS.endDS()
    return 0

# 估计因子协方差矩阵
def FactorCovarianceGeneration(args,qs_env):
    FactorReturnDates = pd.Series(args["RiskDB"].getFactorReturnDate(args["TargetTable"]))
    iFactorReturn = None
    iFactorReturnDates = []
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        ProgBar = ProgressBar(max_value=len(args["RiskESTDates"]))
        ProgBar.start()
    else:
        ProgBar = None
    for i,iDate in enumerate(args["RiskESTDates"]):
        iInd = (FactorReturnDates<=iDate).sum()-1
        if iInd<args["FactorCovESTArgs"]["样本长度"]-1:# 样本不足, 跳过
            if ProgBar is not None:
                ProgBar.update(i+1)
            else:
                args['Sub2MainQueue'].put((qs_env.PID,1,None))
            continue
        iLastDates = iFactorReturnDates
        iFactorReturnDates = list(FactorReturnDates.iloc[iInd-args["FactorCovESTArgs"]["样本长度"]+1:iInd+1])
        iNewDates = list(set(iFactorReturnDates).difference(set(iLastDates)))
        iNewDates.sort()
        if iFactorReturn is not None:
            iFactorReturn = pd.concat([iFactorReturn,args["RiskDB"].loadFactorReturn(args["TargetTable"],dates=iNewDates)]).loc[iFactorReturnDates,:]
        else:
            iFactorReturn = args["RiskDB"].loadFactorReturn(args["TargetTable"],dates=iNewDates).loc[iFactorReturnDates,:]
        iFactorCov = RiskModelFun.estimateFactorCov_CHE2(iFactorReturn,forcast_num=args["FactorCovESTArgs"]["预测期数"],
                                                         auto_corr_num=args["FactorCovESTArgs"]["自相关滞后期"],
                                                         half_life_corr=args["FactorCovESTArgs"]["相关系数半衰期"],
                                                         half_life_vol=args["FactorCovESTArgs"]["波动率半衰期"])
        if args["ModelArgs"]["EigenfactorRiskAdjustment"]:
            iFactorCov = RiskModelFun.EigenfactorRiskAdjustment(iFactorCov,
                                                                monte_carlo_num=args["EigenfactorRiskAdjustmentArgs"]["MonteCarlo次数"],
                                                                date_num=args["EigenfactorRiskAdjustmentArgs"]["模拟日期长度"],
                                                                ignore_num=args["EigenfactorRiskAdjustmentArgs"]["拟合忽略样本数"],
                                                                a=args["EigenfactorRiskAdjustmentArgs"]["a"],
                                                                forcast_num=args["FactorCovESTArgs"]["预测期数"],
                                                                auto_corr_num=args["FactorCovESTArgs"]["自相关滞后期"],
                                                                half_life_corr=args["FactorCovESTArgs"]["相关系数半衰期"],
                                                                half_life_vol=args["FactorCovESTArgs"]["波动率半衰期"])
        args["RiskDB"].saveData(args["TargetTable"],iDate,factor_cov=iFactorCov)
        if ProgBar is not None:
            ProgBar.update(i+1)
        else:
            args['Sub2MainQueue'].put((qs_env.PID,1,None))
    if ProgBar is not None:
        ProgBar.finish()
    if args["ModelArgs"]['运行模式']!='串行':
        qs_env.closeResource()
    return 0

# 估计特异性风险
def SpecificRiskGeneration(args,qs_env):
    SpecificReturnDates = pd.Series(args["RiskDB"].getSpecificReturnDate(args["TargetTable"]))
    iSpecificReturnDates = []
    iSpecificReturn = None
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        ProgBar = ProgressBar(max_value=len(args["RiskESTDates"]))
        ProgBar.start()
    else:
        ProgBar = None
    for i,iDate in enumerate(args["RiskESTDates"]):
        iInd = (SpecificReturnDates<=iDate).sum()-1
        if iInd<args["SpecificRiskESTArgs"]["样本长度"]-1:# 样本不足, 跳过
            if ProgBar is not None:
                ProgBar.update(i+1)
            else:
                args['Sub2MainQueue'].put((qs_env.PID,1,None))
            continue
        iCap = args["RiskDB"].loadData(args["TargetTable"],"Cap",dates=iDate)
        if iCap is None:# debug
            print("\n"+iDate)
        iLastDates = iSpecificReturnDates
        iSpecificReturnDates = list(SpecificReturnDates.iloc[iInd-args["SpecificRiskESTArgs"]["样本长度"]+1:iInd+1])
        iNewDates = list(set(iSpecificReturnDates).difference(set(iLastDates)))
        iNewDates.sort()
        if iSpecificReturn is None:
            iSpecificReturn = args["RiskDB"].loadSpecificReturn(args["TargetTable"],dates=iNewDates).loc[iSpecificReturnDates,:]
        else:
            iSpecificReturn = pd.concat([iSpecificReturn,args["RiskDB"].loadSpecificReturn(args["TargetTable"],dates=iNewDates)]).loc[iSpecificReturnDates,:]
        iFactorData = args["RiskDB"].loadFactorData(args["TargetTable"],dates=iDate)
        if iFactorData is None:# debug
            print("\n"+iDate)
        iFactorData = iFactorData.loc[:,args["SpecificRiskESTArgs"]["结构化模型回归风格因子"]+args["ModelArgs"]["所有行业"]]
        iSpecificRisk = RiskModelFun.estimateSpecificRisk_EUE3(specific_ret=iSpecificReturn,factor_data=iFactorData,cap=iCap,
                                                               forcast_num=args["SpecificRiskESTArgs"]["预测期数"],
                                                               auto_corr_num=args["SpecificRiskESTArgs"]["自相关滞后期"],
                                                               half_life=args["FactorCovESTArgs"]["波动率半衰期"])
        args["RiskDB"].saveData(args["TargetTable"],iDate,specific_risk=iSpecificRisk)
        if ProgBar is not None:
            ProgBar.update(i+1)
        else:
            args['Sub2MainQueue'].put((qs_env.PID,1,None))
    if ProgBar is not None:
        ProgBar.finish()
    if args["ModelArgs"]['运行模式']!='串行':
        qs_env.closeResource()
    return 0

class BarraModel(object):
    """Barra 风险模型"""
    def __init__(self,name,config_file=None,qs_env=None):
        self.ModelType = "多因子风险模型"
        # 需要预先指定的属性
        self.Name = name
        self.QSEnv = qs_env
        self.Config = self.QSEnv.loadConfigFile(config_file)
        self.RiskESTDates = []# 估计风险的日期序列
        # 模型的其他属性
        self.RegressDates = None# 进行截面回归的日期序列
        self.RiskDB = None# 风险数据库
        self.TargetTable = None# 风险数据存储的目标表
        self.DS = None# 提供因子数据的数据源
        
        return
    # 设置计算风险估计的日期序列
    def setRiskESTDate(self,dates):
        self.RiskESTDates = dates
        self.RiskESTDates.sort()
        return 0
    # 生成回归日期序列
    def _genRegressDate(self):
        DSDates = pd.Series(self.DS.getDateTime())
        RiskESTStartInd = max(((DSDates<=self.RiskESTDates[0]).sum()-1,0))
        if self.Config.FactorCovESTArgs["样本长度"]==-1:
            StartInd = 0
        else:
            StartInd = max((RiskESTStartInd-self.Config.FactorCovESTArgs["样本长度"]+1,0))
        if self.Config.SpecificRiskESTArgs['样本长度']==-1:
            StartInd = 0
        else:
            StartInd = min((max((RiskESTStartInd-self.Config.SpecificRiskESTArgs["样本长度"]+1,0)),StartInd))
        self.RegressDates = DSDates.iloc[StartInd:]
        self.RegressDates = list(self.RegressDates[self.RegressDates<=self.RiskESTDates[-1]])
        if self.RiskDB.checkTableExistence(self.TargetTable):# 目标风险数据表已经存在
            OldDates = self.RiskDB.getFactorReturnDate(self.TargetTable)
            self.RegressDates = list(set(self.RegressDates).difference(set(OldDates)))
            self.RegressDates.sort()
        return 0
    # 调整风险数据的计算日期序列
    def _adjustRiskESTDate(self):
        AllReturnDates = self.RegressDates
        if self.RiskDB.checkTableExistence(self.TargetTable):# 目标风险数据表已经存在
            AllReturnDates = list(set(self.RiskDB.getFactorReturnDate(self.TargetTable)).union(set(AllReturnDates)))
            AllReturnDates.sort()
        RequiredLen = max((self.Config.FactorCovESTArgs["样本长度"],self.Config.SpecificRiskESTArgs["样本长度"]))
        for i,iDate in enumerate(self.RiskESTDates):
            iInd = AllReturnDates.index(iDate)
            if iInd>=RequiredLen-1:
                break
        if iInd>=RequiredLen-1:
            self.RiskESTDates = self.RiskESTDates[i:]
        else:
            self.RiskESTDates = []
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
        StartT = time.clock()
        FactorDB = getattr(self.QSEnv,self.Config.ModelArgs['因子数据库'])
        self.DS = DataSource.ParaMMAPCacheDSWithDateCons("MainDS",FactorDB,self.QSEnv)
        self.DS.prepareData(self.Config.DSTableFactor)
        self.DS.SysArgs.update(getattr(self.Config,"DSSysArgs",{}))
        print('数据源构建完成, 运行时间 : %.2f' % (time.clock()-StartT))
        # 计算估计收益率的日期序列
        self._genRegressDate()
        # 调整RiskESTDates
        self._adjustRiskESTDate()
        if self.RiskESTDates==[]:
            self.QSEnv.SysArgs['LastErrorMsg'] = "可以计算风险数据的日期序列为空!"
            return 0
        else:
            return 0
    # 生成因子收益率以及特异性收益率
    def _genFactorAndSpecificReturn(self):
        Args = {"DS":self.DS,
                "RiskDB":self.RiskDB,
                "RegressDates":self.RegressDates,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            FactorAndSpecificReturnGeneration(Args,self.QSEnv)
        else:
            nTask = len(self.RegressDates)
            nPrcs = min((nTask,self.Config.ModelArgs["子进程数"]))
            Procs,Main2SubQueue,Sub2MainQueue = self.QSEnv.startMultiProcess(n_prc=nPrcs, target_fun=FactorAndSpecificReturnGeneration,
                                                                             arg=Args, partition_arg="RegressDates", n_partition_tail=1,
                                                                             main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            with ProgressBar(max_value=nTask) as ProgBar:
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
            for iPID,iPrcs in Procs.items():
                iPrcs.join()
        return 0
    # 生成因子协方差矩阵
    def _genFactorCovariance(self):
        Args = {"RiskDB":self.RiskDB,
                "RiskESTDates":self.RiskESTDates,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs,
                "FactorCovESTArgs":self.Config.FactorCovESTArgs,
                "EigenfactorRiskAdjustmentArgs":self.Config.EigenfactorRiskAdjustmentArgs,
                "FactorVolatilityRegimeAdjustmentArgs":self.Config.FactorVolatilityRegimeAdjustmentArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            FactorCovarianceGeneration(Args,self.QSEnv)
        else:
            nTask = len(self.RiskESTDates)
            nPrcs = min((nTask,self.Config.ModelArgs["子进程数"]))
            ProgBar = ProgressBar(max_value=nTask)
            Procs,Main2SubQueue,Sub2MainQueue = self.QSEnv.startMultiProcess(n_prc=nPrcs, target_fun=FactorCovarianceGeneration,
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
        if self.Config.ModelArgs["FactorVolatilityRegimeAdjustment"]:# 未完待续
            print("Factor Volatility Regime Adjustment进行中...")
            FactorReturnDates = pd.Series(self.RiskDB.getFactorReturnDate(self.TargetTable))
            iFactorReturn = None
            iFactorReturnDates = []
            FactorVolatility = []# 用于FactorVolatilityRegimeAdjustment
            iSampleDates = []# 用于FactorVolatilityRegimeAdjustment
            for iDate in self.RiskESTDates:
                iLastDates = iSampleDates
                iSampleDates = self.getTableDate(self.TargetTable,end_date=iDate)
                if self.Config.FactorVolatilityRegimeAdjustmentArgs["样本长度"]>0:
                    iSampleDates = iSampleDates[max((0,len(iSampleDates)-1-self.Config.FactorVolatilityRegimeAdjustmentArgs["样本长度"])):]
                iNewSampleDates = list(set(iSampleDates).difference(set(iLastDates)))
                iNewSampleDates.sort()
                FactorVolatility = self.RiskDB.loadFactorReturn(self.TargetTable,dates=None)
                iFactorVolatility = pd.Series(np.diag(iFactorCov)**0.5)
            print("Factor Volatility Regime Adjustment完成")
        return 0
    # 生成特异性风险
    def _genSpecificRisk(self):
        Args = {"RiskDB":self.RiskDB,
                "RiskESTDates":self.RiskESTDates,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs,
                "FactorCovESTArgs":self.Config.FactorCovESTArgs,
                "SpecificRiskESTArgs":self.Config.SpecificRiskESTArgs}
        if Args["ModelArgs"]['运行模式']=='串行':
            SpecificRiskGeneration(Args,self.QSEnv)
        else:
            nTask = len(self.RiskESTDates)
            nPrcs = min((nTask,self.Config.ModelArgs["子进程数"]))
            ProgBar = ProgressBar(max_value=nTask)
            Procs,Main2SubQueue,Sub2MainQueue = self.QSEnv.startMultiProcess(n_prc=nPrcs, target_fun=SpecificRiskGeneration,
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
        print("截面回归进行中...")
        TotalStartT = time.clock()
        self._genFactorAndSpecificReturn()
        print("截面回归完成, 耗时 : %.2f" % (time.clock()-TotalStartT))
        print("因子协方差矩阵估计中...")
        StartT = time.clock()
        self._genFactorCovariance()
        print("因子协方差矩阵估计完成, 耗时 : %.2f" % (time.clock()-StartT))
        print("特异性风险估计中...")
        StartT = time.clock()
        self._genSpecificRisk()
        print("特异性风险估计完成, 耗时 : %.2f" % (time.clock()-StartT))
        print("风险数据计算完成, 耗时 : %.2f" % (time.clock()-TotalStartT))
        self.RiskDB.connect()
        return 0