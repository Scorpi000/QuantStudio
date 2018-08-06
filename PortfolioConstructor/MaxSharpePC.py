# -*- coding: utf-8 -*-
"""最大夏普率投资组合构建模型"""
# 目标的一般形式：
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
# 最大夏普率：(f'*x + f0) / sqrt(x'Sigma*x + Mu'x + q),{'f':array(n,1),'f0':double,'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Mu':array(n,1),'Delta':array(n,1),'q':double,'type':'Sharpe'},其中，Sigma = X*F*X'+Delta

# 其他选项
# {"MaxIter":int,"tol":double,"Algorithm":str,"x0":array(n),...}

# 优化结果
# (array(n),{'fval':double,'IterNum':int,'tol':double,...})

# 最大夏普率投资组合构造器基类
class MaxSharpePC(BasePC.PortfolioConstructor):
    """最大夏普率投资组合构造器"""
    # 启动投资组合构造器
    def initPC(self):
        Error = BasePC.PortfolioConstructor.initPC(self)
        self.UseCovMatrix = True
        self.UseExpectedReturn = True
        return Error
    # 生成目标函数参数信息
    def genObjectArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        if arg is None:
            arg = {"相对基准":False}
        ArgInfo = {}
        ArgInfo['相对基准'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':0,'是否可见':True}
        return (arg,ArgInfo)
    # 生成目标的优化器条件形式
    def _genObject(self,object_arg):
        if not object_arg['相对基准']:
            if self.FactorCov is None:
                return {"type":"Sharpe","f":self.ExpectedReturn.values.reshape((self.nID,1)),"f0":0.0,
                        "Sigma":self.CovMatrix.values,"Mu":np.zeros((self.nID,1)),"q":0.0}
            else:
                return {"type":"Sharpe","f":self.ExpectedReturn.values.reshape((self.nID,1)),"f0":0.0,
                        "X":self.RiskFactorData.values,"F":self.FactorCov.values,"Delta":self.SpecificRisk.values.reshape((self.nID,1))**2,
                        "Mu":np.zeros((self.nID,1)),"q":0.0}
        Objective = {"type":"Sharpe","f":self.ExpectedReturn.values.reshape((self.nID,1)),
                     "f0":-np.dot(self.ExpectedReturn.values,self.BenchmarkHolding.values)-np.dot(self.BenchmarkExtraExpectedReturn.values,self.BenchmarkExtra.values)}
        Sigma = self.CovMatrix.values
        Mu = -np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov1.values).reshape((self.nID,1))
        Mu -= np.dot(self.BenchmarkHolding.values,Sigma).reshape((self.nID,1))
        q = np.dot(np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov.values),self.BenchmarkExtra.values)
        q += 2*np.dot(np.dot(self.BenchmarkExtra.values,self.BenchmarkExtraCov1.values),self.BenchmarkHolding.values)
        q += np.dot(np.dot(self.BenchmarkHolding.values,Sigma),self.BenchmarkHolding.values)
        Objective["Mu"] = Mu
        Objective["q"] = q
        if self.FactorCov is None:
            Objective["Sigma"] = Sigma
        else:
            Objective["X"] = self.RiskFactorData.values
            Objective["F"] = self.FactorCov.values
            Objective["Delta"] = self.SpecificRisk.values.reshape((self.nID,1))**2
        return Objective

class MatlabMSPC(MaxSharpePC,BasePC.MatlabPC):
    """MATLAB最大夏普率组合构造器"""
    def __init__(self,name,qs_env=None):
        MaxSharpePC.__init__(self,name,qs_env)
        self.MatlabScript = "solveMaxSharpe"
    # 产生选项参数信息    
    def genOptionArgInfo(self,arg=None):
        if arg is None:
            arg = {}
            arg['信息显示'] = 'Default'
        ArgInfo = {}
        ArgInfo['信息显示'] = {'数据类型':'Str','取值范围':['Default','0','1','2'],'是否刷新':False,'序号':0,'是否可见':True}
        return (arg,ArgInfo)
    # 整理选项参数信息
    def _genOption(self,option_arg):
        PreparedOption = {'Display':option_arg['信息显示']}
        return PreparedOption
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        return BasePC.MatlabPC._solve(self, nvar, prepared_objective, prepared_constraints, prepared_option)

if OptClasses is not None:
    OptClasses["MATLAB最大夏普率组合构造器"] = MatlabMSPC
else:
    OptClasses = OrderedDict()