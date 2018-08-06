# -*- coding: utf-8 -*-
"""风险预算投资组合构建模型"""
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
from QuantStudio.Tools.AuxiliaryFun import getFactorList

# 优化目标
# 风险预算：{'Sigma':array(n,n),'X':array(n,k),'F':array(k,k),'Delta':array(n,1),'b':array(n,1),type':'Risk_Budget'}

# 其他选项
# {"MaxIter":int,"tol":double,"Algorithm":str,"x0":array(n),...}

# 优化结果
# (array(n),{'fval':double,'IterNum':int,'tol':double,...})

# 风险预算投资组合构造器基类
class RiskBudgetPC(BasePC.PortfolioConstructor):
    """风险预算投资组合构造器"""
    def __init__(self,name,qs_env=None):
        BasePC.PortfolioConstructor.__init__(self,name,qs_env)
        self.ConstraintArgInfoFun = {"预算约束":self._genBudgetConstraintArgInfo,
                                     "权重约束":self._genWeightConstraintArgInfo}
        self.ConstraintAdjustive = False
        return
    # 生成预算约束条件参数信息
    def _genBudgetConstraintArgInfo(self,arg=None):
        arg,ArgInfo = BasePC.PortfolioConstructor._genBudgetConstraintArgInfo(self,arg=arg)
        for iArgName in ArgInfo:
            ArgInfo[iArgName]['是否可改'] = False
        return (arg,ArgInfo)
    # 生成权重约束条件参数信息
    def _genWeightConstraintArgInfo(self,arg=None):
        arg,ArgInfo = BasePC.PortfolioConstructor._genWeightConstraintArgInfo(self,arg=arg)
        for iArgName in ArgInfo:
            ArgInfo[iArgName]['是否可改'] = False
        return (arg,ArgInfo)
    # 启动投资组合构造器
    def initPC(self):
        Error = BasePC.PortfolioConstructor.initPC(self)
        self.UseCovMatrix = True
        return Error
    # 生成目标函数参数信息
    def genObjectArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        DefaultNumFactorList,DefaultStrFactorList = getFactorList(self.AllFactorDataType)
        if arg is None:
            arg = {"预算因子":"等权"}
        ArgInfo = {}
        ArgInfo['预算因子'] = {'数据类型':'Str','取值范围':DefaultNumFactorList+["等权"],'是否刷新':False,'序号':0,'是否可见':True}
        return (arg,ArgInfo)
    # 生成目标的优化器条件形式
    def _genObject(self,object_arg):
        if self.FactorCov is None:
            Objective = {"type":"Risk_Budget", "Sigma":self.CovMatrix.values}
        else:
            Objective = {"type":"Risk_Budget", "X":self.RiskFactorData.values, "F":self.FactorCov.values,
                         "Delta":self.SpecificRisk.values.reshape((self.nID,1))**2}
        if object_arg['预算因子']=="等权":
            Objective["b"] = np.zeros((self.nID,1))+1/self.nID
        else:
            Objective["b"] = self.FactorData.loc[:,[object_arg['预算因子']]].values
        return Objective

class MatlabRBPC(RiskBudgetPC,BasePC.MatlabPC):
    """MATLAB风险预算组合构造器"""
    def __init__(self,name,qs_env=None):
        RiskBudgetPC.__init__(self,name,qs_env)
        self.MatlabScript = "solveRiskBudget"
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
        return BasePC.MatlabPC._solve(self,nvar, prepared_objective, prepared_constraints, prepared_option)

if OptClasses is not None:
    OptClasses["MATLAB风险预算组合构造器"] = MatlabRBPC
else:
    OptClasses = OrderedDict()