# coding=utf-8
"""融券账户(TODO)"""
import os
import shelve
from copy import deepcopy

import pandas as pd
import numpy as np

from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.HistoryTest.StrategyTest.StrategyTestModule import Account, cutDateTime

class MarginAccount(Account):
    """融券账户"""
    def __init__(self, sys_args={}, **kwargs):
        self.QSEnv = qs_env
        super().__init__()
        self.__QS_Type__ = "ShortAccount"
        self.Dates = []# 日期序列, ['20050101']
        self.Position = []# 仓位序列, [pd.Series(股票数, index=[ID])]
        self.NominalAmount = []# 名义金额, [double]
        self.Margin = []# 保证金序列, [double]
        self.TurnOver = []# 换手率序列, [double]
        self.TradeDates = []# 发生调仓的日期, 与self.Dates可能不等长
        self.InitMargin = []# 再平衡时的初始Margin, 与self.TradeDates等长
        self.MarginPreTrade = []# 交易发生之前用成交价计算的保证金, 用于计算对冲和空头的保证金以及复算统计结果, 与self.TradeDates等长
        return
    # 生成系统参数信息集以及初始值
    def genSysArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        DefaultNumFactorList,DefaultStrFactorList = getFactorList(self.StdDataSource.DataType)
        if arg is None:
            arg = {}
            arg['启用卖空'] = False
            arg['可卖空条件'] = None
            arg['限买条件'] = getDefaultNontradableIDFilter(self.StdDataSource,True,self.QSEnv)
            arg['限卖条件'] = getDefaultNontradableIDFilter(self.StdDataSource,False,self.QSEnv)
            arg['成交价'] = searchNameInStrList(DefaultNumFactorList,['价','Price','price'])
            arg['结算价'] = arg['成交价']
            arg['交易费率'] = 0.003
            arg['保证金率'] = 0.0
        ArgInfo = {}
        ArgInfo['启用卖空'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['可卖空条件'] = {'数据类型':'IDFilterStr','取值范围':self.StdDataSource.FactorNames,'是否刷新':False,'序号':1}
        ArgInfo['限买条件'] = {'数据类型':'IDFilterStr','取值范围':self.StdDataSource.FactorNames,'是否刷新':False,'序号':2}
        ArgInfo['限卖条件'] = {'数据类型':'IDFilterStr','取值范围':self.StdDataSource.FactorNames,'是否刷新':False,'序号':3}
        ArgInfo['成交价'] = {'数据类型':'Str','取值范围':DefaultNumFactorList,'是否刷新':False,'序号':4,'是否可见':True}
        ArgInfo['结算价'] = {'数据类型':'Str','取值范围':DefaultNumFactorList,'是否刷新':False,'序号':5,'是否可见':True}
        ArgInfo['交易费率'] = {'数据类型':'Double','取值范围':[0,1,0.0005],'是否刷新':False,'序号':6,'是否可见':True}
        ArgInfo['保证金率'] = {'数据类型':'Double','取值范围':[0,1,0.0005],'是否刷新':False,'序号':7,'是否可见':True}
        return (arg,ArgInfo)
    # 回归初始化状态
    def start(self):
        self.Dates = []
        self.Position = []
        self.NominalAmount = []
        self.Margin = []
        self.TurnOver = []
        self.TradeDates = []
        self.InitMargin = []
        self.MarginPreTrade = []
        self.TempData = {'CompiledNonbuyableIDStr':self.SysArgs['限买条件'],'CompiledNonsellableIDStr':self.SysArgs["限卖条件"],'CompiledShortableIDStr':self.SysArgs['可卖空条件']}
        return 0
    # 时间跳转,并初始化账户,成功返回1
    def MoveOn(self,idt):
        if self.Dates==[]:
            self.Position.append(pd.Series([],dtype='float'))
            self.NominalAmount.append(0)
            self.Margin.append(0)
            self.TurnOver.append(0)
        else:
            self.Position.append(self.Position[-1].copy())
            self.NominalAmount.append(self.NominalAmount[-1])
            self.Margin.append(self.Margin[-1])
            self.TurnOver.append(0)
        self.Dates.append(idt)
        return 0
    # 结束账户，并生成特有的结果集
    def endAccount(self):
        return {}
    # 获取可卖空的证券ID
    def getShortableID(self,idt):
        if (self.SysArgs['可卖空条件'] is None) or (self.SysArgs['可卖空条件']==''):
            return self.StdDataSource.getID()
        OldIDFilterStr,OldIDFilterFactors = self.StdDataSource.setIDFilter(self.TempData.get("CompiledShortableIDStr"),self.TempData.get("ShortableIDFilterFactors"))
        IDs = self.StdDataSource.getID(idt,is_filtered=True)
        self.TempData["CompiledShortableIDStr"],self.TempData["ShortableIDFilterFactors"] = self.StdDataSource.setIDFilter(OldIDFilterStr,OldIDFilterFactors)
        return IDs
    # 获取不能买入的证券ID
    def getNonbuyableID(self,idt):
        if self.SysArgs['限买条件'] is None:
            return []
        OldIDFilterStr,OldIDFilterFactors = self.StdDataSource.setIDFilter(self.TempData.get("CompiledNonbuyableIDStr"),self.TempData.get("NonbuyableIDFilterFactors"))
        IDs = self.StdDataSource.getID(idt,is_filtered=True)
        self.TempData["CompiledNonbuyableIDStr"],self.TempData["NonbuyableIDFilterFactors"] = self.StdDataSource.setIDFilter(OldIDFilterStr,OldIDFilterFactors)
        return IDs
    # 获取不能卖出的证券ID
    def getNonsellableID(self,idt):
        if self.SysArgs['限卖条件'] is None:
            return []
        OldIDFilterStr,OldIDFilterFactors = self.StdDataSource.setIDFilter(self.TempData.get("CompiledNonsellableIDStr"),self.TempData.get("NonsellableIDFilterFactors"))
        IDs = self.StdDataSource.getID(idt,is_filtered=True)
        self.TempData["CompiledNonsellableIDStr"],self.TempData["NonsellableIDFilterFactors"] = self.StdDataSource.setIDFilter(OldIDFilterStr,OldIDFilterFactors)
        return IDs    
    # 获得仓位分布, 返回: pd.Series(股票数,index=[ID])
    def getPosition(self,date=None):
        if self.Position==[]:
            return pd.Series([],dtype='float')
        if date is None:
            Pos = -1
        else:
            try:
                Pos = self.Dates.index(date)
            except:
                Pos = (pd.Series(self.Dates)<=date).sum()-1
            if Pos==-1:
                return pd.Series([],dtype='float')
        return self.Position[Pos].copy()
    # 计算持仓金额分布, price: pd.Series(价格,index=[ID]), 默认使用最后一天和结算价参数提供的价格数据, 返回pd.Series(金额,index=[ID])
    def calcWealthDistribution(self,price=None,date=None):
        Position = self.getPosition(date=date)
        if Position.shape[0]==0:
            return pd.Series([],dtype='float')
        if price is None:
            price = self.StdDataSource.getDateTimeData(idt=date,factor_names=[self.SysArgs['结算价']])[self.SysArgs['结算价']]
        Rslt = (Position*price)
        return Rslt[pd.notnull(Rslt) & (Rslt!=0.0)]
    # 计算财富,给定日期和价格信息, price: pd.Series(价格,index=[ID]), 默认使用最后一天和结算价参数提供的价格数据
    def calcWealth(self,price=None,date=None):
        return self.calcWealthDistribution(price=price,date=date).sum()
    # 计算持仓投资组合, price: pd.Series(价格,index=[ID]), 默认使用最后一天和结算价参数提供的价格数据, 返回pd.Series(权重,index=[ID])
    def calcPortfolio(self,price=None,date=None):
        WealthDistribution = self.calcWealthDistribution(price=price,date=date)
        TotalWealth = WealthDistribution.sum()
        if TotalWealth==0:
            return pd.Series([],dtype='float')
        else:
            return WealthDistribution/abs(TotalWealth)
    # 保存自身信息
    def saveInfo(self,container):
        container['DSName'] = self.StdDataSource.Name
        container['SysArgs'] = self.SysArgs
        return container
    # 恢复信息
    def loadInfo(self,container):
        self.StdDataSource = self.DSs[container['DSName']]
        self.SysArgs = container['SysArgs']
        _,self.SysArgInfos = self.genSysArgInfo(self.SysArgs)
        return 0
