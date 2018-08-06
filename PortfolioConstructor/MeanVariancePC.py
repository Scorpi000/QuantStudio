# -*- coding: utf-8 -*-
"""均值方差投资组合构建模型"""
# 目标的一般形式：gamma0*r'w - gamma*1/2*w'*Cov*w - lambda1*norm(w-w0,1) - lambda2*sum((w-w0)^+) - lambda3*sum((w-w0)^-)
from collections import OrderedDict

import pandas as pd
import numpy as np
try:
    import matlab
    OptClasses = OrderedDict()
except:
    OptClasses = None

from . import BasePC

# 优化目标
# 线性目标：f'*x,{'f':array(n,1),'type':'Linear'}
# 二次目标：x'Sigma*x + Mu'*x,{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'type':'Quadratic'},其中，Sigma = X*F*X'+Delta
# L1惩罚线性目标：f'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'f':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Linear'}
# L1惩罚二次目标：x'Sigma*x + Mu'*x + lambda1*sum(abs(x-c)) + lambda2*sum((x-c_pos)^+) + lambda3*sum((x-c_neg)^-),{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'Mu':array(n,1),'lambda1':double,'c':array(n,1),'lambda2':double,'c_pos':array(n,1),'lambda3':double,'c_neg':array(n,1),'type':'L1_Quadratic'}, 其中, Sigma = X*F*X'+Delta

# 其他选项
# {"MaxIter":int,"tol":double,"Algorithm":str,"x0":array(n),...}

# 优化结果
# (array(n),{'fval':double,'IterNum':int,'tol':double,...})

# 均值方差投资组合构造器基类
class MeanVariancePC(BasePC.PortfolioConstructor):
    """均值方差投资组合构造器"""
    # 启动投资组合构造器
    def initPC(self):
        Error = BasePC.PortfolioConstructor.initPC(self)
        self.UseCovMatrix = (self.UseCovMatrix or (self.ObjectArg['风险厌恶系数']!=0.0))
        self.UseHolding = (self.UseHolding or ((self.ObjectArg['换手惩罚系数']!=0.0) or (self.ObjectArg['买入惩罚系数']!=0.0) or (self.ObjectArg['卖出惩罚系数']!=0.0)))
        self.UseBenchmark = (self.UseBenchmark or self.ObjectArg['相对基准'])
        self.UseExpectedReturn = (self.UseExpectedReturn or (self.ObjectArg['收益项系数']!=0.0))
        return Error
    # 生成目标函数参数信息
    def genObjectArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        if arg is None:
            arg = {}
            arg['相对基准'] = False
            arg['收益项系数'] = 0.0
            arg['风险厌恶系数'] = 1.0
            arg['换手惩罚系数'] = 0.0
            arg['买入惩罚系数'] = 0.0
            arg['卖出惩罚系数'] = 0.0
        ArgInfo = {}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['收益项系数'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':1,'是否可见':True}
        ArgInfo['风险厌恶系数'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':2,'是否可见':True}
        ArgInfo['换手惩罚系数'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':3,'是否可见':True}
        ArgInfo['买入惩罚系数'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':4,'是否可见':True}
        ArgInfo['卖出惩罚系数'] = {'数据类型':'Double','取值范围':[-9999.0,9999.0,0.0001],'是否刷新':False,'序号':5,'是否可见':True}
        return (arg,ArgInfo)
    # 生成目标的优化器条件形式
    def _genObject(self,object_arg):
        Sign = -1.0
        self.ObjectiveConstant = 0.0
        if object_arg['收益项系数']!=0.0:
            Mu = object_arg['收益项系数']*self.ExpectedReturn.values.reshape((self.nID,1))
            if object_arg['相对基准']:
                self.ObjectiveConstant += -object_arg['收益项系数']*np.dot(self.ExpectedReturn.values,self.BenchmarkHolding.values)
                self.ObjectiveConstant += -object_arg['收益项系数']*np.dot(self.BenchmarkExtraExpectedReturn.values,self.BenchmarkExtra.values)
        else:
            Mu = np.zeros((self.nID,1))
        if object_arg['风险厌恶系数']!=0.0:
            Sigma = self.CovMatrix.values
            if object_arg['相对基准']:
                Mu += object_arg['风险厌恶系数']*np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov1.values).reshape((self.nID,1))
                Mu += object_arg['风险厌恶系数']*np.dot(self.BenchmarkHolding.values,Sigma).reshape((self.nID,1))
                self.ObjectiveConstant += -object_arg['风险厌恶系数']/2*np.dot(np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov.values),self.BenchmarkExtra.values)
                self.ObjectiveConstant += -object_arg['风险厌恶系数']*np.dot(np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov1.values),self.BenchmarkHolding.values)
                self.ObjectiveConstant += -object_arg['风险厌恶系数']/2*np.dot(np.dot(self.BenchmarkHolding.values,Sigma),self.BenchmarkHolding.values)
            if self.FactorCov is None:
                Objective = {"type":"Quadratic",
                             "Sigma":-Sign*object_arg['风险厌恶系数']/2*Sigma,
                             "Mu":Sign*Mu}
            else:
                Objective = {"type":"Quadratic",
                             "X":self.RiskFactorData.values,
                             "F":-Sign*object_arg['风险厌恶系数']/2*self.FactorCov.values,
                             "Delta":-Sign*object_arg['风险厌恶系数']/2*self.SpecificRisk.values.reshape((self.nID,1))**2,
                             "Mu":Sign*Mu}
        else:
            Objective = {"type":"Linear",
                         "f":Sign*Mu}
        if object_arg['换手惩罚系数']!=0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda1':Sign*object_arg['换手惩罚系数'],
                              "c":self.Holding.values.reshape((self.nID,1))})
            self.ObjectiveConstant += -object_arg['换手惩罚系数']*self.HoldingExtra.abs().sum()
        if object_arg['买入惩罚系数']!=0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda2':Sign*object_arg['买入惩罚系数'],
                              "c_pos":self.Holding.values.reshape((self.nID,1))})
            self.ObjectiveConstant += -object_arg['买入惩罚系数']*self.HoldingExtra[self.HoldingExtra>0].sum()
        if object_arg['卖出惩罚系数']!=0.0:
            Objective['type'] = "L1_" + Objective['type'].split("_")[-1]
            Objective.update({'lambda3':Sign*object_arg['卖出惩罚系数'],
                              "c_neg":self.Holding.values.reshape((self.nID,1))})
            self.ObjectiveConstant += -object_arg['卖出惩罚系数']*(-self.HoldingExtra[self.HoldingExtra<0].sum())
        return Objective

class MatlabMVPC(MeanVariancePC,BasePC.MatlabPC):
    """MATLAB均值方差组合构造器"""
    def __init__(self,name,qs_env=None):
        MeanVariancePC.__init__(self,name,qs_env)
        self.MatlabScript = "solveMeanVariance"
    # 产生选项参数信息    
    def genOptionArgInfo(self,arg=None):
        if arg is None:
            arg = {}
            arg['信息显示'] = 'Default'
            arg['求解器'] = 'Default'
        ArgInfo = {}
        ArgInfo['信息显示'] = {'数据类型':'Str','取值范围':['Default','0','1','2'],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['求解器'] = {'数据类型':'Str','取值范围':['Default','cplex','mosek','sedumi'],'是否刷新':False,'序号':0,'是否可见':True}
        return (arg,ArgInfo)
    # 整理选项参数信息
    def _genOption(self,option_arg):
        PreparedOption = {'Display':option_arg['信息显示'],'Solver':option_arg['求解器']}
        return PreparedOption
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        return BasePC.MatlabPC._solve(self,nvar, prepared_objective, prepared_constraints, prepared_option)

if OptClasses is not None:
    OptClasses["MATLAB均值方差组合构造器"] = MatlabMVPC
else:
    OptClasses = OrderedDict()