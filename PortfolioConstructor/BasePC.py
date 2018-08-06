# -*- coding: utf-8 -*-
from collections import OrderedDict
import time
import os

import pandas as pd
import numpy as np
try:
    import matlab
except:
    pass

from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.Tools.AuxiliaryFun import getFactorList

# 约束条件
# Box约束：lb <= x <= ub,{'lb':array(n,1),'ub':array(n,1),'type':'Box'}
# 线性不等式约束：A * x <= b,{'A':array(m,n),'b':array(m,1),'type':'LinearIn'}
# 线性等式约束：Aeq * x == beq,{'Aeq':array(m,n),'beq':array(m,1),'type':'LinearEq'}
# 二次约束：x'*Sigma*x + Mu'*x <= q,{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'q':double,'type':'Quadratic'},其中，Sigma = X*F*X'+Delta
# L1范数约束：sum(abs(x-c)) <= l,{'c':array(n,1),'l':double,'type':'L1'}
# 正部总约束：sum((x-c_pos)^+) <= l_pos，{'c_pos':array(n,1),'l_pos':double,'type':'Pos'}
# 负部总约束：sum((x-c_neg)^-) <= l_neg，{'c_neg':array(n,1),'l_neg':double,'type':'Neg'}
# 非零数目约束：sum((x-b)!=0) <= N, {'b':array(n,1),'N':double,'type':'NonZeroNum'}

# 其他选项
# {"MaxIter":int,"tol":double,"Algorithm":str,"x0":array(n),...}

# 优化结果
# (array(n),{'fval':double,'IterNum':int,'tol':double,...})

# 投资组合构造器基类
class PortfolioConstructor(object):
    """投资组合构造器"""
    def __init__(self,name,qs_env=None):
        # 创建后必须立即赋值指定的变量
        self.Name = name# 投资组合构造器的名称，字符串
        self.QSEnv = qs_env
        self.AllFactorDataType = {}# 可用的所有因子数据类型，用于生成参数信息
        # 优化前必须指定的变量
        self.Holding = None# 当前持有的投资组合，Series(index=self.TargetIDs)
        self.BenchmarkHolding = None# 当前的基准投资组合，Series(index=self.TargetIDs)
        self.ExpectedReturn = None# 当前的预期收益率，Series(index=self.TargetIDs)
        self.AmountFactor = None# 成交额因子，Series(index=self.TargetIDs)
        self.CovMatrix = None# 当前股票收益率的协方差矩阵，DataFrame(index=self.TargetIDs,columns=self.TargetIDs)
        self.FactorCov = None# 当前因子收益率的协方差矩阵，DataFrame(index=[因子],columns=[因子])
        self.RiskFactorData = None# 当前的风险因子值，DataFrame(index=self.TargetIDs,columns=[因子])
        self.SpecificRisk = None# 当前个股的特异风险，Series(index=self.TargetIDs)
        self.FactorData = None# 当前使用的因子值，DataFrame(index=self.TargetIDs,columns=[因子])
        self.Wealth = None# 当前的财富值，double
        self.FilterID = {}# 可能用到过滤条件以及对应的ID,{ID过滤条件:[ID]}
        self.TargetIDs = None# 最终进入优化的股票池,[ID]
        self.nID = None# 最终进入优化的股票数量
        self.ObjectiveConstant = 0.0# 优化目标规范化后剩余的常数项
        
        self.ConstraintArgInfoFun = {"预算约束":self._genBudgetConstraintArgInfo,
                                     "因子暴露约束":self._genFactorExposeConstraintArgInfo,
                                     "权重约束":self._genWeightConstraintArgInfo,
                                     "预期收益约束":self._genExpectedReturnConstraintArgInfo,
                                     "波动率约束":self._genVolatilityConstraintArgInfo,
                                     "换手约束":self._genTurnoverConstraintArgInfo,
                                     "非零数目约束":self._genNonZeroNumConstraintArgInfo}
        self.ConstraintArgs = ([],[])# 已经添加的约束条件参数，([约束类型],[参数集])，启动前由外部指定
        self.ObjectArg = None# 优化的目标参数，启动前由外部指定
        self.OptionArg = None# 选项参数，启动前由外部指定
        
        # 共享信息
        # 优化前根据参数信息即可生成
        self.UseCovMatrix = False# 是否使用协方差矩阵
        self.UseBenchmark = False# 是否使用基准投资组合
        self.UseHolding = False# 是否使用当前投资组合
        self.UseExpectedReturn = False# 是否使用预期收益率
        self.UseAmount = False# 是否使用成交额
        self.UseWealth = False# 是否使用总财富
        self.UseFactor = []# 使用数据的因子，
        # 优化时根据具体数据生成
        self.BenchmarkExtraIDs = []# 基准相对于TargetIDs多出来的证券ID
        self.BenchmarkExtra = pd.Series()# 基准相对于TargetIDs多出来的证券ID权重，pd.Series(index=self.BenchmarkExtraIDs)
        self.BenchmarkExtraCov = pd.DataFrame()# 基准相对于TargetIDs多出来的证券ID对应的协方差阵，pd.DataFrame(index=self.BenchmarkExtraIDs,columns=self.BenchmarkExtraIDs)
        self.BenchmarkExtraCov1 = pd.DataFrame()# 基准相对于TargetIDs多出来的证券ID关于TargetIDs的协方差阵，pd.DataFrame(index=self.BenchmarkExtraIDs,columns=self.TargetIDs)
        self.BenchmarkExtraExpectedReturn = pd.Series()# 基准相对于TargetIDs多出来的证券ID对应的预期收益，pd.Series(index=self.BenchmarkExtraIDs)
        self.BenchmarkExtraFactorData = pd.DataFrame()# 基准相对于TargetIDs多出来的证券ID对应的因子数据，pd.DataFrame(index=self.BenchmarkExtraIDs,columns=[因子])
        self.HoldingExtraIDs = []# 当前持仓相对于TargetIDs多出来的证券ID
        self.HoldingExtra = pd.Series()# 当前持仓相对于TargetIDs多出来的证券ID权重，pd.Series(index=self.HoldingExtraIDs)
        self.HoldingExtraAmount = pd.Series()# 当前持仓相对于TargetIDs多出来的证券ID对应的因子数据，pd.Series(index=self.HoldingExtraIDs)
        return
    # 设置优化目标
    def setObject(self,object_arg=None):
        self.ObjectArg,_ = self.genObjectArgInfo(arg=object_arg)
        return 0
    # 获取支持的约束条件
    def getSupportedConstraint(self):
        return list(self.ConstraintArgInfoFun.keys())
    # 添加约束条件
    def addConstraint(self,constraint_type="因子暴露约束",constraint_arg=None):
        self.ConstraintArgs[0].append(constraint_type)
        if constraint_arg is None:
            self.ConstraintArgs[1].append(self.ConstraintArgInfoFun[constraint_type](constraint_arg)[0])
        else:
            self.ConstraintArgs[1].append(constraint_arg)
        return 0
    # 清空约束条件
    def clearConstraint(self):
        self.ConstraintArgs = ([],[])
        return 0
    # 设置其他选项参数
    def setOptionArg(self,option_arg=None):
        self.OptionArg,_ = self.genOptionArgInfo(arg=option_arg)
        return 0
    # 设置初始的目标选股池
    def setTargetID(self,target_ids=None):
        if target_ids is not None:
            self.TargetIDs = set(target_ids)
        else:
            self.TargetIDs = None
        return 0
    # 设置预期收益率
    def setExpectedReturn(self,expected_return):
        self.ExpectedReturn = expected_return.dropna()
        if self.TargetIDs is None:
            self.TargetIDs = set(self.ExpectedReturn.index)
        else:
            self.TargetIDs = self.TargetIDs.intersection(self.ExpectedReturn.index)
        return 0
    # 设置协方差矩阵
    def setCovMatrix(self,factor_cov=None,specific_risk=None,risk_factor_data=None,cov_matrix=None):
        if cov_matrix is None:
            self.FactorCov = factor_cov
            self.SpecificRisk = specific_risk.dropna()
            self.RiskFactorData = risk_factor_data.dropna(how='any',axis=0)
            if self.TargetIDs is not None:
                self.TargetIDs = self.TargetIDs.intersection(self.SpecificRisk.index).intersection(self.RiskFactorData.index)
            else:
                self.TargetIDs = set(self.SpecificRisk.index).intersection(self.RiskFactorData.index)
        else:
            cov_matrix = cov_matrix.dropna(how='all',axis=0)
            cov_matrix = cov_matrix.loc[:,cov_matrix.index]
            self.CovMatrix = cov_matrix.dropna(how='any',axis=0)
            if self.TargetIDs is not None:
                self.TargetIDs = self.TargetIDs.intersection(self.CovMatrix.index)
            else:
                self.TargetIDs = set(self.CovMatrix.index)
        return 0
    # 设置因子数据
    def setFactorData(self,factor_data):
        self.FactorData = factor_data.dropna(how='any',axis=0)
        if self.TargetIDs is None:
            self.TargetIDs = set(self.FactorData.index)
        else:
            self.TargetIDs = self.TargetIDs.intersection(self.FactorData.index)
        return 0
    # 设置成交额数据
    def setAmountData(self,amount_data):
        self.AmountFactor = amount_data.dropna()
        if self.TargetIDs is None:
            self.TargetIDs = set(self.AmountFactor.index)
        else:
            self.TargetIDs = self.TargetIDs.intersection(self.AmountFactor.index)
        return 0
    # 设置持仓数据
    def setHolding(self,holding_data):
        self.Holding = holding_data
        return 0
    # 设置基准持仓数据
    def setBenchmarkHolding(self,benchmark_holding):
        self.BenchmarkHolding = benchmark_holding
        return 0
    # 设置总财富
    def setWealth(self,wealth):
        self.Wealth = wealth
        return 0
    # 设置过滤ID
    def setFilteredID(self,ds,idt):
        for iIDFilterStr in self.FilterID:
            OldIDFilterStr,OldIDFilterFactors = ds.setIDFilter(iIDFilterStr)
            self.FilterID[iIDFilterStr] = ds.getID(idt=idt,is_filtered=True)
            ds.setIDFilter(OldIDFilterStr,id_filter_factors=OldIDFilterFactors)
        return 0
    # 启动投资组合构造器
    def initPC(self):
        self.UseCovMatrix = ("波动率约束" in self.ConstraintArgs[0])
        self.UseHolding = ("换手约束" in self.ConstraintArgs[0])
        self.UseExpectedReturn = ("预期收益约束" in self.ConstraintArgs[0])
        self.UseWealth = False
        self.UseAmount = False
        self.UseFactor = set()
        self.UseBenchmark = False
        self.FilterID = {}
        for i,iConstraintArgs in enumerate(self.ConstraintArgs[1]):
            self.UseBenchmark = (self.UseBenchmark or iConstraintArgs.get("相对基准",False))
            if self.ConstraintArgs[0][i]=='因子暴露约束':
                self.UseFactor = self.UseFactor.union(set(iConstraintArgs['因子名称']))
            elif (self.ConstraintArgs[0][i]=='换手约束') and (iConstraintArgs['限制类型'] in ['买卖限制','买入限制','卖出限制']) and (iConstraintArgs['成交额倍数']!=0.0):
                self.UseAmount = True
                self.UseWealth = True
            elif (self.ConstraintArgs[0][i]=='权重约束') and (iConstraintArgs['目标ID'] is not None):
                self.FilterID[iConstraintArgs['目标ID']] = None
        self.UseFactor = list(self.UseFactor)
        return 0
    # 结束投资组合构造器
    def endPC(self):
        return (1,{})
    # 生成目标函数参数信息
    def genObjectArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        return ({},{})
    # 生成目标的优化器条件形式
    def _genObject(self,object_arg):
        return {}
    # 生成预算约束条件参数信息,即：i'*(w-benchmark) <=(==,>=) a
    def _genBudgetConstraintArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        if arg is None:
            arg = {}
            arg['限制上限'] = 1.0
            arg['限制下限'] = 1.0
            arg['相对基准'] = False
            arg['舍弃优先级'] = -1
        ArgInfo = {}
        ArgInfo['限制上限'] = {'数据类型':'Double','取值范围':[-9999.0,9999,0.0001],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['限制下限'] = {'数据类型':'Double','取值范围':[-9999.0,9999,0.0001],'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':2,'是否可见':True}
        ArgInfo['舍弃优先级'] = {'数据类型':'Int','取值范围':[-1,9999,1],'是否刷新':False,'序号':3,'是否可见':True}
        return (arg,ArgInfo)
    # 生成预算约束条件的优化器条件形式
    def _genBudgetConstraint(self,constraint_arg):
        if constraint_arg['相对基准']:
            aAdj = self.BenchmarkHolding.sum()+self.BenchmarkExtra.sum()
        else:
            aAdj = 0.0
        Constraints = []
        if constraint_arg['限制上限']==constraint_arg['限制下限']:
            Constraints.append({"type":"LinearEq",
                               "Aeq":np.ones((1,self.nID)),
                               "beq":np.array([[constraint_arg['限制上限']+aAdj]])})
        else:
            if constraint_arg['限制下限']>-9999.0:
                Constraints.append({"type":"LinearIn",
                                   "A":-np.ones((1,self.nID)),
                                   "b":-np.array([[constraint_arg['限制下限']+aAdj]])})
            if constraint_arg['限制上限']<9999.0:
                Constraints.append({"type":"LinearIn",
                                   "A":np.ones((1,self.nID)),
                                   "b":np.array([[constraint_arg['限制上限']+aAdj]])})
        return Constraints
    # 生成因子暴露约束条件参数信息，即：f'*(w-benchmark) <=(==,>=) a
    def _genFactorExposeConstraintArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        DefaultNumFactorList,DefaultStrFactorList = getFactorList(self.AllFactorDataType)
        if arg is None:
            arg = {}
            arg['因子类型'] = '数值型'
            arg['因子名称'] = ([] if DefaultNumFactorList==[] else [DefaultNumFactorList[0]])
            arg['限制上限'] = 0.0
            arg['限制下限'] = 0.0
            arg['相对基准'] = False
            arg['舍弃优先级'] = -1
        elif arg['因子名称']!=[]:
            if (arg['因子类型']=='数值型') and (self.AllFactorDataType[arg['因子名称'][0]]=='string'):
                arg['因子名称'] = [DefaultNumFactorList[0]]
            elif (arg['因子类型']=='类别型') and (self.AllFactorDataType[arg['因子名称'][0]]!='string'):
                arg['因子名称'] = [DefaultStrFactorList[0]]
        ArgInfo = {}
        ArgInfo['因子类型'] = {'数据类型':'Str','取值范围':['数值型','类别型'],'是否刷新':True,'序号':0,'是否可见':True}
        if arg['因子类型']=='数值型':
            ArgInfo['因子名称'] = {'数据类型':'ArgList','取值范围':{"数据类型":"Str","取值范围":DefaultNumFactorList},'是否刷新':False,'序号':1,'是否可见':True}
        else:
            ArgInfo['因子名称'] = {'数据类型':'ArgList','取值范围':{"数据类型":"Str","取值范围":DefaultStrFactorList},'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['限制上限'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':2,'是否可见':True}
        ArgInfo['限制下限'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':3,'是否可见':True}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':4,'是否可见':True}
        ArgInfo['舍弃优先级'] = {'数据类型':'Int','取值范围':[-1,9999,1],'是否刷新':False,'序号':5,'是否可见':True}
        return (arg,ArgInfo)
    # 生成数值型因子暴露约束条件的优化器条件形式
    def _genNumFactorExposeConstraint(self,constraint_arg):
        Constraints = []
        nFactor = len(constraint_arg['因子名称'])
        if constraint_arg['相对基准']:
            aAdj = (np.dot(self.BenchmarkHolding.values,self.FactorData.loc[:,constraint_arg['因子名称']].values) + np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraFactorData.loc[:,constraint_arg['因子名称']].values)).reshape((nFactor,1))
        else:
            aAdj = np.zeros((nFactor,1))
        if constraint_arg['限制上限']==constraint_arg['限制下限']:
            Aeq = self.FactorData.loc[:,constraint_arg['因子名称']].values.T
            AeqSum = np.abs(Aeq).sum(axis=1)
            Mask = (AeqSum!=0.0)
            Constraints.append({"type":"LinearEq",
                               "Aeq":Aeq[Mask,:],
                               "beq":(constraint_arg['限制上限']+aAdj)[Mask,:]})
        else:
            A = self.FactorData.loc[:,constraint_arg['因子名称']].values.T
            ASum = np.abs(A).sum(axis=1)
            Mask = (ASum!=0.0)
            if constraint_arg['限制下限']>-9999.0:
                Constraints.append({"type":"LinearIn",
                                   "A":-A[Mask,:],
                                   "b":-(constraint_arg['限制下限']+aAdj)[Mask,:]})
            if constraint_arg['限制上限']<9999.0:
                Constraints.append({"type":"LinearIn",
                                   "A":A[Mask,:],
                                   "b":(constraint_arg['限制上限']+aAdj)[Mask,:]})
        return Constraints
    # 生成类别型因子暴露约束条件的优化器条件形式
    def _genClassFactorExposeConstraint(self,constraint_arg):
        Constraints = []
        if self.UseBenchmark:
            AllFactorData = self.FactorData.append(self.BenchmarkExtraFactorData)
        else:
            AllFactorData = self.FactorData
        for iFactor in constraint_arg['因子名称']:
            iFactorData = AllFactorData[iFactor]
            iFactorData = DummyVarTo01Var(iFactorData,ignore_na=True,ignore_nonstring=True)
            nFactor = iFactorData.shape[1]
            if constraint_arg['相对基准']:
                aAdj = (np.dot(self.BenchmarkHolding.values,iFactorData.loc[self.TargetIDs].values) + np.dot(self.BenchmarkExtra.values,iFactorData.loc[self.BenchmarkExtraIDs].values)).reshape((nFactor,1))
            else:
                aAdj = np.zeros((nFactor,1))
            if constraint_arg['限制上限']==constraint_arg['限制下限']:
                Aeq = iFactorData.loc[self.TargetIDs].values.T
                AeqSum = np.abs(Aeq).sum(axis=1)
                Mask = (AeqSum!=0.0)
                Constraints.append({"type":"LinearEq",
                                    "Aeq":Aeq[Mask,:],
                                    "beq":(constraint_arg['限制上限']+aAdj)[Mask,:]})
            else:
                A = iFactorData.loc[self.TargetIDs].values.T
                ASum = np.abs(A).sum(axis=1)
                Mask = (ASum!=0.0)
                if constraint_arg['限制下限']>-9999.0:
                    Constraints.append({"type":"LinearIn",
                                        "A":-A[Mask,:],
                                        "b":-(constraint_arg['限制下限']+aAdj)[Mask,:]})
                if constraint_arg['限制上限']<9999.0:
                    Constraints.append({"type":"LinearIn",
                                        "A":A[Mask,:],
                                        "b":(constraint_arg['限制上限']+aAdj)[Mask,:]})
        return Constraints
    # 生成因子暴露约束条件的优化器条件形式
    def _genFactorExposeConstraint(self,constraint_arg):
        if constraint_arg['因子类型']=='数值型':
            return self._genNumFactorExposeConstraint(constraint_arg)
        else:
            return self._genClassFactorExposeConstraint(constraint_arg)
    # 生成权重约束条件参数信息,即：(w-benchmark) <=(>=) a
    def _genWeightConstraintArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        if arg is None:
            arg = {}
            arg['目标ID'] = None
            arg['限制上限'] = 1.0
            arg['限制下限'] = 0.0
            arg['相对基准'] = False
            arg['舍弃优先级'] = -1
        ArgInfo = {}
        ArgInfo['目标ID'] = {'数据类型':'IDFilterStr','取值范围':list(self.AllFactorDataType.keys()),'是否刷新':False,'是否可见':True,'序号':0,'可否遍历':False}
        ArgInfo['限制上限'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['限制下限'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':2,'是否可见':True}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':3,'是否可见':True}
        ArgInfo['舍弃优先级'] = {'数据类型':'Int','取值范围':[-1,9999,1],'是否刷新':False,'序号':4,'是否可见':True}
        return (arg,ArgInfo)
    # 生成权重约束条件的优化器条件形式
    def _genWeightConstraint(self,constraint_arg):
        if constraint_arg['目标ID'] is None:
            UpConstraint = pd.Series(constraint_arg['限制上限'],index=self.TargetIDs)
            DownConstraint = pd.Series(constraint_arg['限制下限'],index=self.TargetIDs)
        else:
            UpConstraint = pd.Series(np.inf,index=self.TargetIDs)
            DownConstraint = pd.Series(-np.inf,index=self.TargetIDs)
            SpecialIDs = self.FilterID[constraint_arg['目标ID']]
            if SpecialIDs is None:
                return []
            SpecialIDs = list(set(SpecialIDs).intersection(set(self.TargetIDs)))
            UpConstraint[SpecialIDs] = constraint_arg['限制上限']
            DownConstraint[SpecialIDs] = constraint_arg['限制下限']
        UpConstraint[UpConstraint>=9999.0] = np.inf
        DownConstraint[DownConstraint<=-9999.0] = -np.inf
        if constraint_arg['相对基准']:
            UpConstraint += self.BenchmarkHolding
            DownConstraint += self.BenchmarkHolding
        return [{"type":"Box","lb":DownConstraint.values.reshape((self.nID,1)),"ub":UpConstraint.values.reshape((self.nID,1))}]
    # 产生换手率约束条件参数信息，即：sum(abs(w-w0)) <=(==) a
    def _genTurnoverConstraintArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        if arg is None:
            arg = {}
            arg['限制类型'] = '总换手限制'
            arg['成交额倍数'] = 1.0
            arg['限制上限'] = 0.7
            arg['舍弃优先级'] = 0
        ArgInfo = {}
        ArgInfo['限制类型'] = {'数据类型':'Str','取值范围':['总换手限制','总买入限制','总卖出限制','买卖限制','买入限制','卖出限制'],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['限制上限'] = {'数据类型':'Double','取值范围':[0.0,9999.0,0.0001],'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['成交额倍数'] = {'数据类型':'Double','取值范围':[0.0,9999.0,0.0001],'是否刷新':False,'序号':2,'是否可见':True}
        ArgInfo['舍弃优先级'] = {'数据类型':'Int','取值范围':[-1,9999,1],'是否刷新':False,'序号':3,'是否可见':True}
        return (arg,ArgInfo)
    # 生成换手率约束条件的优化器条件形式
    def _genTurnoverConstraint(self,constraint_arg):
        HoldingWeight = self.Holding.values.reshape((self.nID,1))
        if constraint_arg['限制类型']=='总换手限制':
            aAdj = constraint_arg['限制上限']-self.HoldingExtra.abs().sum()
            return [{"type":"L1","c":HoldingWeight,"l":aAdj}]
        elif constraint_arg['限制类型']=='总买入限制':
            aAdj = constraint_arg['限制上限']+self.HoldingExtra[self.HoldingExtra<0].sum()
            return [{"type":"Pos","c_pos":HoldingWeight,"l":aAdj}]
        elif constraint_arg['限制类型']=='总卖出限制':
            aAdj = constraint_arg['限制上限']-self.HoldingExtra[self.HoldingExtra>0].sum()
            return [{"type":"Neg","c_neg":HoldingWeight,"l":aAdj}]
        if constraint_arg['成交额倍数']==0.0:
            aAdj = np.zeros((self.nID,1))+constraint_arg['限制上限']
        else:
            aAdj = self.AmountFactor.values.reshape((self.nID,1))*constraint_arg['成交额倍数']/self.Wealth
        if constraint_arg['限制类型']=='买卖限制':
            return [{"type":"Box","ub":aAdj+HoldingWeight,"lb":-aAdj+HoldingWeight}]
        elif constraint_arg['限制类型']=='买入限制':
            return [{"type":"Box","ub":aAdj+HoldingWeight,"lb":np.zeros((self.nID,1))-np.inf}]
        elif constraint_arg['限制类型']=='买入限制':
            return [{"type":"Box","ub":np.zeros((self.nID,1))+np.inf,"lb":-aAdj+HoldingWeight}]
        return []
    # 生成波动率约束条件参数信息,即：(w-benchmark)'*Cov*(w-benchmark) <= a
    def _genVolatilityConstraintArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        if arg is None:
            arg = {}
            arg['限制上限'] = 0.06
            arg['相对基准'] = False
            arg['舍弃优先级'] = -1
        ArgInfo = {}
        ArgInfo['限制上限'] = {'数据类型':'Double','取值范围':[0.0,9999.0,0.0001],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['舍弃优先级'] = {'数据类型':'Int','取值范围':[-1,9999,1],'是否刷新':False,'序号':2,'是否可见':True}
        return (arg,ArgInfo)
    # 生成波动率约束条件的优化器条件形式
    def _genVolatilityConstraint(self,constraint_arg):
        if constraint_arg['相对基准']:
            Sigma = self.CovMatrix.values
            BenchmarkWeight = self.BenchmarkHolding.values
            Mu = -2*np.dot(BenchmarkWeight,Sigma)-2*np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov1.values)
            Mu = Mu.reshape((self.nID,1))
            q = constraint_arg['限制上限']**2-np.dot(np.dot(BenchmarkWeight,Sigma),BenchmarkWeight)
            q -= 2*np.dot(np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov1.values),BenchmarkWeight)
            q -= np.dot(np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov.values),self.BenchmarkExtra.values)
        else:
            Mu = np.zeros((self.nID,1))
            q = constraint_arg['限制上限']**2
        Constraint = {"type":"Quadratic","Mu":Mu,"q":q}
        if self.FactorCov is not None:
            Constraint["X"] = self.RiskFactorData.values
            Constraint["F"] = self.FactorCov.values
            Constraint["Delta"] = self.SpecificRisk.values.reshape((self.nID,1))**2
        else:
            Constraint['Sigma'] = self.CovMatrix.values
        return [Constraint]
    # 生成预期收益约束条件，即：r'*(w-benchmark) >= a
    def _genExpectedReturnConstraintArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        if arg is None:
            arg = {"限制下限":0.0,
                   "相对基准":False,
                   "舍弃优先级":-1}
        ArgInfo = {}
        ArgInfo['限制下限'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['舍弃优先级'] = {'数据类型':'Int','取值范围':[-1,9999,1],'是否刷新':False,'序号':2,'是否可见':True}
        return (arg,ArgInfo)
    # 生成预期收益约束条件的优化器条件形式
    def _genExpectedReturnConstraint(self,constraint_arg):
        r = self.ExpectedReturn.values.reshape((1,self.nID))
        if constraint_arg["相对基准"]:
            aAdj = -constraint_arg['限制下限']-np.dot(self.BenchmarkExtraExpectedReturn.values,self.BenchmarkExtra.values)-np.dot(r[0],self.BenchmarkHolding.values)
            return [{"type":"LinearIn","A":-r,"b":np.array([[aAdj]])}]
        else:
            return [{"type":"LinearIn","A":-r,"b":np.array([[-constraint_arg['限制下限']]])}]
    # 生成非零数目约束条件, 即: sum((w-benchmark!=0)<=N
    def _genNonZeroNumConstraintArgInfo(self,arg=None):
        if arg is None:
            arg = {"限制上限":150,
                   "相对基准":False,
                   "舍弃优先级":-1}
        ArgInfo = {}
        ArgInfo['限制上限'] = {'数据类型':'Int','取值范围':[1,9999,1],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['舍弃优先级'] = {'数据类型':'Int','取值范围':[-1,9999,1],'是否刷新':False,'序号':2,'是否可见':True}
        return (arg,ArgInfo)
    # 生成非零数目约束条件的优化器条件形式
    def _genNonZeroNumConstraint(self,constraint_arg):
        if constraint_arg["相对基准"]:
            N = constraint_arg['限制上限']-(self.BenchmarkExtra.values!=0).sum()
            return [{"type":"NonZeroNum","N":N,"b":self.BenchmarkHolding.values.reshape((self.nID,1))}]
        else:
            return [{"type":"NonZeroNum","N":constraint_arg["限制上限"],"b":np.zeros((self.nID,1))}]
    # 其他选项参数信息
    def genOptionArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        return ({},{})
    # 整理选项参数
    def _genOption(self,option_arg):
        return {}
    # 预处理数据
    def _preprocessData(self):
        TargetIDs = self.TargetIDs
        self.TargetIDs = list(self.TargetIDs)
        self.TargetIDs.sort()
        self.nID = len(self.TargetIDs)
        if self.UseBenchmark:
            self.BenchmarkExtraIDs = list(set(self.BenchmarkHolding[self.BenchmarkHolding>0].index).difference(TargetIDs))
            self.BenchmarkExtraIDs.sort()
            self.BenchmarkExtra = self.BenchmarkHolding[self.BenchmarkExtraIDs]
            self.BenchmarkHolding = self.BenchmarkHolding.ix[self.TargetIDs]
            self.BenchmarkHolding[pd.isnull(self.BenchmarkHolding)] = 0.0
        else:
            self.BenchmarkExtraIDs = []
            self.BenchmarkHolding = None
            self.BenchmarkExtra = pd.Series()
        TargetBenchmarkExtraIDs = self.TargetIDs+self.BenchmarkExtraIDs
        if self.UseCovMatrix:
            if self.FactorCov is not None:
                RiskFactorDataNAFillVal = self.RiskFactorData.mean()
                SpecialRiskNAFillVal = self.SpecificRisk.mean()
                self.RiskFactorData = self.RiskFactorData.ix[TargetBenchmarkExtraIDs]
                self.SpecificRisk = self.SpecificRisk.ix[TargetBenchmarkExtraIDs]
                self.RiskFactorData = self.RiskFactorData.fillna(RiskFactorDataNAFillVal)
                self.SpecificRisk = self.SpecificRisk.fillna(SpecialRiskNAFillVal)
                self.CovMatrix = np.dot(np.dot(self.RiskFactorData.values,self.FactorCov.values),self.RiskFactorData.values.T)+np.diag(self.SpecificRisk.values**2)
                self.CovMatrix = pd.DataFrame(self.CovMatrix,index=TargetBenchmarkExtraIDs,columns=TargetBenchmarkExtraIDs)
                self.RiskFactorData = self.RiskFactorData.loc[self.TargetIDs]
                self.SpecificRisk = self.SpecificRisk.loc[self.TargetIDs]
            self.BenchmarkExtraCov = self.CovMatrix.loc[self.BenchmarkExtraIDs,self.BenchmarkExtraIDs]
            self.BenchmarkExtraCov1 = self.CovMatrix.loc[self.BenchmarkExtraIDs,self.TargetIDs]
            self.CovMatrix = self.CovMatrix.loc[self.TargetIDs,self.TargetIDs]
        if self.UseExpectedReturn:
            self.BenchmarkExtraExpectedReturn = self.ExpectedReturn.ix[self.BenchmarkExtraIDs]
            self.BenchmarkExtraExpectedReturn = self.BenchmarkExtraExpectedReturn.fillna(0.0)
            self.ExpectedReturn = self.ExpectedReturn.ix[self.TargetIDs]
            self.ExpectedReturn = self.ExpectedReturn.fillna(0.0)
        if self.UseFactor!=[]:
            self.BenchmarkExtraFactorData = self.FactorData.ix[self.BenchmarkExtraIDs]
            self.BenchmarkExtraFactorData = self.BenchmarkExtraFactorData.fillna(0.0)
            self.FactorData = self.FactorData.ix[self.TargetIDs]
            self.FactorData = self.FactorData.fillna(0.0)
        if self.UseHolding:
            self.HoldingExtraIDs = list(set(self.Holding[self.Holding>0].index).difference(TargetIDs))
            self.HoldingExtraIDs.sort()
            self.HoldingExtra = self.Holding[self.HoldingExtraIDs]
            self.Holding = self.Holding.ix[self.TargetIDs]
            self.Holding[pd.isnull(self.Holding)] = 0.0
        else:
            self.HoldingExtraIDs = []
            self.Holding = None
            self.HoldingExtra = pd.Series()
        TargetHoldingExtraIDs = self.TargetIDs+self.HoldingExtraIDs
        if self.UseAmount:
            self.HoldingExtraAmount = self.AmountFactor.ix[self.HoldingExtraIDs]
            self.HoldingExtraAmount = self.HoldingExtraAmount.fillna(0.0)
            self.AmountFactor = self.AmountFactor.ix[self.TargetIDs]
            self.AmountFactor = self.AmountFactor.fillna(0.0)
        return 0
    # 整理条件，形成优化模型
    def _prepareModel(self,objective,contraints,option_arg):
        nVar = len(self.TargetIDs)
        PreparedConstraints = {}
        for iConstraint in contraints:
            if iConstraint is None:
                continue
            elif iConstraint['type'] == "Box":
                PreparedConstraints["Box"] = PreparedConstraints.get("Box",{"lb":np.zeros((nVar,1))-np.inf,"ub":np.zeros((nVar,1))+np.inf,"type":"Box"})
                PreparedConstraints["Box"]["lb"] = np.maximum(PreparedConstraints["Box"]["lb"],iConstraint["lb"])
                PreparedConstraints["Box"]["ub"] = np.minimum(PreparedConstraints["Box"]["ub"],iConstraint["ub"])
            elif iConstraint['type'] == "LinearIn":
                PreparedConstraints["LinearIn"] = PreparedConstraints.get("LinearIn",{"A":np.zeros((0,nVar)),"b":np.zeros((0,1)),"type":"LinearIn"})
                PreparedConstraints["LinearIn"]["A"] = np.vstack((PreparedConstraints["LinearIn"]["A"],iConstraint["A"]))
                PreparedConstraints["LinearIn"]["b"] = np.vstack((PreparedConstraints["LinearIn"]["b"],iConstraint["b"]))
            elif iConstraint["type"] == "LinearEq":
                PreparedConstraints["LinearEq"] = PreparedConstraints.get("LinearEq",{"Aeq":np.zeros((0,nVar)),"beq":np.zeros((0,1)),"type":"LinearEq"})
                PreparedConstraints["LinearEq"]["Aeq"] = np.vstack((PreparedConstraints["LinearEq"]["Aeq"],iConstraint["Aeq"]))
                PreparedConstraints["LinearEq"]["beq"] = np.vstack((PreparedConstraints["LinearEq"]["beq"],iConstraint["beq"]))
            else:
                PreparedConstraints[iConstraint["type"]] = PreparedConstraints.get(iConstraint["type"],[])
                PreparedConstraints[iConstraint["type"]].append(iConstraint)
        PreparedOption = self._genOption(option_arg)
        return (nVar,objective,PreparedConstraints,PreparedOption)    
    # 求解一次优化问题
    def _solve(self,nvar,prepared_objective,prepared_constraints,prepared_option):
        return (None,{})
    # 求解优化问题
    def solve(self):
        self._preprocessData()
        Objective = self._genObject(self.ObjectArg)
        Constraints = []
        DropedConstraintInds = {-1:[]}
        DropedConstraints = {-1:[]}
        iStartInd = -1
        for i,iConstraintType in enumerate(self.ConstraintArgs[0]):
            iConstraintArg = self.ConstraintArgs[1][i]
            if iConstraintType=='预算约束':
                iConstraints = self._genBudgetConstraint(iConstraintArg)
            elif iConstraintType=='因子暴露约束':
                iConstraints = self._genFactorExposeConstraint(iConstraintArg)
            elif iConstraintType=="权重约束":
                iConstraints = self._genWeightConstraint(iConstraintArg)
            elif iConstraintType=="预期收益约束":
                iConstraints = self._genExpectedReturnConstraint(iConstraintArg)
            elif iConstraintType=="波动率约束":
                iConstraints = self._genVolatilityConstraint(iConstraintArg)
            elif iConstraintType=="换手约束":
                iConstraints = self._genTurnoverConstraint(iConstraintArg)
            elif iConstraintType=='非零数目约束':
                iConstraints = self._genNonZeroNumConstraint(iConstraintArg)
            else:
                iConstraints = []
            Constraints += iConstraints
            iEndInd = iStartInd+len(iConstraints)
            iPriority = iConstraintArg.get('舍弃优先级',-1)
            if (iEndInd-iStartInd!=0) and (iPriority!=-1):
                DropedConstraintInds[iPriority] = DropedConstraintInds.get(iPriority,[])+[i for i in range(iStartInd+1,iEndInd+1)]
                DropedConstraints[iPriority] = DropedConstraints.get(iPriority,[])+[str(i)+'-'+iConstraintType]
            iStartInd = iEndInd
        ResultInfo = {}
        ReleasedConstraint = []
        Priority = list(DropedConstraintInds.keys())
        Priority.sort()
        while (ResultInfo.get("Status",0)!=1) and (Priority!=[]):
            iPriority = Priority.pop(0)
            for j in DropedConstraintInds[iPriority]:
                Constraints[j] = None
            nVar,PreparedObjective,PreparedConstraints,PreparedOption = self._prepareModel(Objective,Constraints,self.OptionArg)
            TargetWeight,ResultInfo = self._solve(nVar,PreparedObjective,PreparedConstraints,PreparedOption)
            ReleasedConstraint += DropedConstraints[iPriority]
        ResultInfo['ReleasedConstraint'] = ReleasedConstraint
        if TargetWeight is not None:
            return (pd.Series(TargetWeight,index=self.TargetIDs),ResultInfo)
        else:
            return (None,ResultInfo)
    # 检查当前的优化问题是否可解
    def _checkSolvability(self):
        for iConstraintType in self.ConstraintArgs[0]:
            if iConstraintType not in self.ConstraintArgInfoFun:
                return "不支持类型为"+iConstraintType+"的条件"
        return None
    # 计算波动率约束优化后的实现值
    def _calRealizedVolatilityConstraint(self,optimal_w,constraint_arg):
        if not constraint_arg['相对基准']:
            return np.dot(optimal_w.values,np.dot(optimal_w.values,self.CovMatrix.loc[self.TargetIDs,self.TargetIDs].values))**0.5
        else:
            IDs = list(set(self.BenchmarkHolding.index).intersection(set(self.CovMatrix.index)))
            Mu = -2*((self.BenchmarkHolding.loc[IDs]*self.CovMatrix.loc[IDs,IDs]).T.sum())
            temp = (-Mu/2*self.BenchmarkHolding).sum()
            if pd.isnull(temp):
                temp = 0.0
            Mu = Mu.loc[self.TargetIDs]
            Mu[pd.isnull(Mu)] = 0.0
            return (np.dot(optimal_w.values,np.dot(optimal_w.values,self.CovMatrix.loc[self.TargetIDs,self.TargetIDs].values))+np.dot(Mu.values,optimal_w.values)+temp)**0.5
    # 保存自身信息
    def saveInfo(self,container):
        container["Name"] = self.Name
        container["AllFactorDataType"] = self.AllFactorDataType
        return container
    # 恢复信息
    def loadInfo(self,container):
        self.Name = container["Name"]
        self.AllFactorDataType = container['AllFactorDataType']
        return 0

# 基于 MATLAB 的投资组合构造器
class MatlabPC(PortfolioConstructor):
    """基于 MATLAB 的投资组合构造器"""
    def __init__(self,name,qs_env=None):
        PortfolioConstructor.__init__(self,name,qs_env=qs_env)
        self.MatlabScript = ""
        return
    # 传递优化目标变量
    def _transmitObjective(self, prepared_objective, eng):
        MatlabVar = {}
        for iVar,iValue in prepared_objective.items():
            if isinstance(iValue,np.ndarray):
                MatlabVar[iVar] = matlab.double(iValue.tolist())
            elif isinstance(iValue,str):
                MatlabVar[iVar] = iValue
            else:
                MatlabVar[iVar] = matlab.double([iValue])
        eng.workspace['Objective'] = MatlabVar
        return 0
    # 传递约束条件变量
    def _transmitConstraint(self,prepared_constraints,eng):
        if isinstance(prepared_constraints,dict):
            MatlabVar = {}
            for iVar,iValue in prepared_constraints.items():
                if isinstance(iValue,np.ndarray):
                    MatlabVar[iVar] = matlab.double(iValue.tolist())
                elif isinstance(iValue,str):
                    MatlabVar[iVar] = iValue
                else:
                    MatlabVar[iVar] = matlab.double([iValue])
            eng.workspace[prepared_constraints['type']+'_Constraint'] = MatlabVar
            return 0
        else:
            MatlabVar = []
            for i,iConstraint in enumerate(prepared_constraints):
                iMatlabVar = {}
                for jVar,jValue in iConstraint.items():
                    if isinstance(jValue,np.ndarray):
                        iMatlabVar[jVar] = matlab.double(jValue.tolist())
                    elif isinstance(jValue,str):
                        iMatlabVar[jVar] = jValue
                    else:
                        iMatlabVar[jVar] = matlab.double([jValue])
                MatlabVar.append(iMatlabVar)
            eng.workspace[MatlabVar[0]['type']+'_Constraint'] = MatlabVar
            return 0
    # 调用 MATLAB 求解优化问题
    def _solve(self,nvar,prepared_objective,prepared_constraints,prepared_option):
        ErrorCode = self.QSEnv.MatlabEngine.connect(engine_name=None,option="-desktop")
        if ErrorCode!=1:
            return np.zeros(nvar)+np.nan
        Eng = self.QSEnv.MatlabEngine.acquireEngine()
        Eng.clear(nargout=0)
        self._transmitObjective(prepared_objective,Eng)
        for iType in prepared_constraints:
            self._transmitConstraint(prepared_constraints[iType],Eng)
        Eng.workspace['nVar'] = float(nvar)
        Eng.workspace["Options"] = prepared_option
        getattr(Eng,self.MatlabScript)(nargout=0)
        ResultInfo = Eng.workspace['ResultInfo']
        if ResultInfo["Status"]==1:
            x = Eng.workspace['x']
            x = np.array(x).reshape(nvar)
        else:
            x = None
        self.QSEnv.MatlabEngine.releaseEngine()
        return (x,ResultInfo)