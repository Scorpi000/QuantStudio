# coding=utf-8
"""尾部相关性模型"""
import time
import os
import imp

import pandas as pd
import numpy as np
from progressbar import ProgressBar

from . import RiskModelFun
from QuantStudio import __QS_LibPath__, __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import startMultiProcess
from QuantStudio.Matlab.MatlabEngine import MatlabEngine

# 估计尾部相关性
def _TailDependenceGeneration(args):
    ReturnDTs = pd.Series(args["FT"].getDateTime())
    args["FT"].start()
    iReturn = None
    iReturnDTs = []
    if args["ModelArgs"]['运行模式']=='串行':# 运行模式为串行
        ProgBar = ProgressBar(max_value=len(args["RiskESTDTs"]))
        ProgBar.start()
    else:
        ProgBar = None
    IDs = args["FT"].getID(ifactor_name=args["ModelArgs"]["收益率因子"])
    for i, iDT in enumerate(args["RiskESTDTs"]):
        iInd = (ReturnDTs<=iDT).sum()-1
        if iInd<args["TailDependenceArgs"]["样本长度"]-1:# 样本不足, 跳过
            if ProgBar is not None: ProgBar.update(i+1)
            else: args['Sub2MainQueue'].put((args["PID"], 1, None))
            continue
        args["FT"].move(iDT)
        iLastDates = iReturnDTs
        iReturnDTs = list(ReturnDTs.iloc[iInd-args["TailDependenceArgs"]["样本长度"]+1:iInd+1])
        iNewDTs = sorted(set(iReturnDTs).difference(set(iLastDates)))
        iNewReturn = args["FT"].readData(factor_names=[args["ModelArgs"]["收益率因子"]], ids=IDs, dts=iNewDTs).iloc[0]
        Eng = args["MatlabEng"].acquireEngine()
        Eng.workspace["NewReturn"] = matlab.double(iNewReturn.values.tolist())
        Eng.workspace["nDate"] = matlab.double([args["TailDependenceArgs"]["样本长度"]])
        Eng.genTailDependenceMatrix(nargout=0)
        TD = Eng.workspace['TD']
        args["MatlabEng"].releaseEngine()
        iCov = args["RiskDB"].readCov(args["ModelArgs"]["波动率来源表"], dts=iDT, ids=list(iNewReturn.columns))
        iD = np.diag(np.diag(iCov.values)**0.5)
        TD = pd.DataFrame(np.dot(np.dot(iD,np.array(TD)),iD), index=iCov.index, columns=iCov.columns)
        args["RiskDB"].writeData(args["TargetTable"], iDT, cov=TD)
        if ProgBar is not None: ProgBar.update(i+1)
        else: args['Sub2MainQueue'].put((args["PID"], 1, None))
    if ProgBar is not None: ProgBar.finish()
    args["FT"].end()
    return 0


class TailDependenceModel(object):
    """尾部相关性模型"""
    def __init__(self, name, factor_table, risk_db, table_name, config_file=None):
        self.ModelType = "尾部相关性模型"
        self.Name = name
        if config_file is None: config_file = __QS_LibPath__+os.sep+"SampleEstModelConfig.py"
        self.Config = imp.load_module(config_file, *imp.find_module(config_file, [os.path.split(config_file)[0]]))
        self.RiskESTDTs = []# 估计风险的时点序列
        self.RiskDB = risk_db# 风险数据库
        self.TargetTable = table_name# 风险数据存储的目标表
        self.FT = factor_table# 提供因子数据的因子表
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
    # 生成尾部相关性矩阵
    def _genTailDependence(self):
        Args = {"RiskDB":self.RiskDB,
                "FT":self.FT,
                "RiskESTDTs":self.RiskESTDTs,
                "TargetTable":self.TargetTable,
                "ModelArgs":self.Config.ModelArgs,
                "MatlabEng":MatlabEngine(),
                "TailDependenceArgs":self.Config.TailDependenceArgs}
        Args["MatlabEng"].connect()
        if Args["ModelArgs"]['运行模式']=='串行':
            _TailDependenceGeneration(Args)
        else:
            nTask = len(self.RiskESTDTs)
            nPrcs = min(nTask, self.Config.ModelArgs["子进程数"])
            ProgBar = ProgressBar(max_value=nTask)
            Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_TailDependenceGeneration,
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
            for iPID, iPrcs in Procs.items(): iPrcs.join()
        return 0
    # 生成数据
    def run(self):
        TotalStartT = time.clock()
        print("==========尾部相关性模型==========", "1. 初始化", sep="\n", end="\n")
        self._initInfo()
        print(('耗时 : %.2f' % (time.clock()-TotalStartT, )), "2. 估计尾部相关性矩阵", sep="\n", end="\n")
        StartT = time.clock()
        self._genTailDependence()
        print("耗时 : %.2f" % (time.clock()-StartT, ), ("总耗时 : %.2f" % (time.clock()-TotalStartT, )), "="*28, sep="\n", end="\n")
        return 0