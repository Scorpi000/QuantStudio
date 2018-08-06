# coding=utf-8
import os
import shelve
import datetime
from collections import OrderedDict

import pandas as pd
import numpy as np

from QuantStudio.FunLib.AuxiliaryFun import getFactorList
from QuantStudio.FunLib.DateTimeFun import DateStr2Datetime,Datetime2DateStr,getMonthLastDay,getNaturalDay
from . import StrategyTestFun

class ResultReport(object):
    """结果报告"""
    def __init__(self,all_dss={},cash_account=None,long_account=None,arbitrage_account=None,short_account=None,signal_prc=None,qs_env=None):
        self.AllDSs = all_dss# 所有的数据源，{数据源名:数据源}
        self.CashAccount = cash_account
        self.LongAccount = long_account
        self.ArbitrageAccount = arbitrage_account
        self.ShortAccount = short_account
        self.SignalPrc = signal_prc
        self.QSEnv = qs_env
        if self.AllDSs!={}:
            self.SysArgs,self.SysArgInfos = self.genSysArgInfo(None)# 生成系统参数和系统参数描述
        else:
            self.SysArgs = None
            self.SysArgInfos = None
        # 基本的结果数据
        self.InitWealth = None# 初始资金
        self.Wealth = None# DataFrame,columns=['现金','证券','总财富']
        self.Turnover = None# DataFrame,columns=['换手率']
        return
    # 生成分类分析设置参数
    def genClassAnalysisArgInfo(self,arg=None):
        if arg is None:
            arg = {}
            arg['分类分析'] = False
            arg['类别数据源'] = self.AllDSNames[0]
            IndustryFactorList = self.DefaultStrFactorList
            arg['类别因子'] = IndustryFactorList[0]
        else:
            if arg['类别数据源'] not in self.AllDSNames:
                arg['类别数据源'] = self.AllDSNames[0]
            Temp,IndustryFactorList = getFactorList(self.AllDSs[arg['类别数据源']].DataType)
            if arg['类别因子'] not in IndustryFactorList:
                arg['类别因子'] = IndustryFactorList[0]
        ArgInfo = {}
        ArgInfo['分类分析'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['类别数据源'] = {'数据类型':'Str','取值范围':self.AllDSNames,'是否刷新':True,'序号':1,'是否可见':True}
        ArgInfo['类别因子'] = {'数据类型':'Str','取值范围':IndustryFactorList,'是否刷新':False,'序号':2,'是否可见':True}
        return (arg,ArgInfo)
    # 生成基准及对冲设置参数
    def genBenchmarkArgInfo(self,arg=None):
        if arg is None:
            arg = {}
            arg['基准数据源'] = self.AllDSNames[0]
            BenchmarkPriceFactorList = self.DefaultNumFactorList
            arg['基准价格因子'] = BenchmarkPriceFactorList[0]
            BenchmarkIDs = self.AllDSs[arg['基准数据源']].getID()
            arg['基准 ID'] = '无'
            arg['空头平衡时点'] = []
            arg['保证金率'] = 0.0
        else:
            ChangedKey = arg.pop("_ChangedKey_",None)
            if arg['基准数据源'] not in self.AllDSNames:
                arg['基准数据源'] = self.AllDSNames[0]
            BenchmarkPriceFactorList,Temp = getFactorList(self.AllDSs[arg['基准数据源']].DataType)
            if arg['基准价格因子'] not in BenchmarkPriceFactorList:
                arg['基准价格因子'] = BenchmarkPriceFactorList[0]
            BenchmarkIDs = self.AllDSs[arg['基准数据源']].getID()
            if arg['基准 ID'] not in BenchmarkIDs:
                arg['基准 ID'] = '无'
        ArgInfo = {}
        ArgInfo['基准数据源'] = {'数据类型':'Str','取值范围':self.AllDSNames,'是否刷新':True,'序号':0}
        ArgInfo['基准价格因子'] = {'数据类型':'Str','取值范围':BenchmarkPriceFactorList,'是否刷新':True,'序号':1}
        ArgInfo['基准 ID'] = {'数据类型':'Str','取值范围':BenchmarkIDs+['无'],'是否刷新':False,'序号':2}
        ArgInfo['空头平衡时点'] = {'数据类型':'DateList','取值范围':[],'是否刷新':False,'序号':3}
        ArgInfo['保证金率'] = {'数据类型':'Double','取值范围':[0,1,0.001],'是否刷新':False,'序号':4}
        return (arg,ArgInfo)
    # 生成日历统计参数及其初始值
    def genCalendarArgInfo(self,arg=None):
        if arg is None:
            arg = {}
            arg['年度统计'] = False
            arg['月度统计'] = '不统计'
            arg['日度统计'] = False
            arg['向前统计天数'] = 0
            arg['向后统计天数'] = 0
        ArgInfo = {}
        ArgInfo['年度统计'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['月度统计'] = {'数据类型':'Str','取值范围':['不统计',"计入下月统计","计入上月统计"],'是否刷新':False,'序号':1,'是否可见':False}
        ArgInfo['日度统计'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':2,'是否可见':True}
        ArgInfo['向前统计天数'] = {'数据类型':'Int','取值范围':[0,9999,1],'是否刷新':False,'序号':3,'是否可见':True}
        ArgInfo['向后统计天数'] = {'数据类型':'Bool','取值范围':[0,9999,1],'是否刷新':False,'序号':4,'是否可见':True}
        return (arg,ArgInfo)
    # 生成滚动分析参数及其初始值
    def genRollingAnalysisArgInfo(self,arg=None):
        if arg is None:
            arg = {}
            arg['滚动分析'] = False
            arg['最小窗口'] = 252
        ArgInfo = {}
        ArgInfo['滚动分析'] = {'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':0,'是否可见':True}
        ArgInfo['最小窗口'] = {'数据类型':'Int','取值范围':[1,9999,1],'是否刷新':False,'序号':1,'是否可见':True}
        return (arg,ArgInfo)
    # 生成系统参数信息集以及初始值
    def genSysArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        self.AllDSNames = list(self.AllDSs.keys())
        ArgInfo = {}
        ArgInfo['基准对冲'] = {'数据类型':'ArgSet','取值范围':[None,self.genBenchmarkArgInfo,None,{}],'是否刷新':True,'序号':0,'是否可见':True}
        ArgInfo['日历分析'] = {'数据类型':'ArgSet','取值范围':[None,self.genCalendarArgInfo,None,{}],'是否刷新':True,'序号':1,'是否可见':True}
        ArgInfo['分类分析'] = {'数据类型':'ArgSet','取值范围':[None,self.genClassAnalysisArgInfo,None,{}],'是否刷新':True,'序号':2,'是否可见':False}
        ArgInfo['滚动分析'] = {'数据类型':'ArgSet','取值范围':[None,self.genRollingAnalysisArgInfo,None,{}],'是否刷新':True,'序号':3,'是否可见':False}
        ArgInfo['统计日期序列'] = {'数据类型':'DateList','取值范围':[],'是否刷新':False,'序号':4,'是否可见':False}
        if arg is None:
            self.DefaultNumFactorList,self.DefaultStrFactorList = getFactorList(self.AllDSs[self.AllDSNames[0]].DataType)
            arg = {}
            arg['基准对冲'],ArgInfo['基准对冲']['取值范围'][0] = self.genBenchmarkArgInfo(None)
            arg['日历分析'],ArgInfo['日历分析']['取值范围'][0] = self.genCalendarArgInfo(None)
            arg['分类分析'],ArgInfo['分类分析']['取值范围'][0] = self.genClassAnalysisArgInfo(None)
            arg['滚动分析'],ArgInfo['滚动分析']['取值范围'][0] = self.genRollingAnalysisArgInfo(None)
            arg['统计日期序列'] = []# 最终用于生成统计数据的日期序列, 默认和回测序列保持一致
        else:
            ChangedKey = arg.pop("_ChangedKey_",None)
            arg['基准对冲'],ArgInfo['基准对冲']['取值范围'][0] = self.genBenchmarkArgInfo(arg['基准对冲'])
            arg['日历分析'],ArgInfo['日历分析']['取值范围'][0] = self.genCalendarArgInfo(arg['日历分析'])
            arg['分类分析'],ArgInfo['分类分析']['取值范围'][0] = self.genClassAnalysisArgInfo(arg['分类分析'])
            arg['滚动分析'],ArgInfo['滚动分析']['取值范围'][0] = self.genRollingAnalysisArgInfo(arg.get('滚动分析'))
        return (arg,ArgInfo)
    # 回归初始化状态
    def start(self):
        self.InitWealth = None
        self.Wealth = None
        self.Turnover = None
        return 0
    # 生成原始结果, 在调用其他非纯功能函数之前必须首先调用此函数
    def genPrimaryResult(self):
        self.Wealth = pd.DataFrame(self.CashAccount.Wealth,index=self.CashAccount.Dates,columns=['现金'])
        self.Wealth['证券'] = pd.Series(self.LongAccount.Wealth,index=self.CashAccount.Dates,name='证券')
        self.Turnover = pd.DataFrame(self.LongAccount.TurnOver,index=self.CashAccount.Dates,columns=['换手率'])
        self.Wealth['总财富'] = self.Wealth['现金']+self.Wealth['证券']
        self.InitWealth = self.CashAccount.SysArgs['初始资金']
        return 0
    # 获取基准的价格序列
    def getBenchmarkPrice(self,dates):
        Rslt = self.AllDSs[self.SysArgs['基准对冲']['基准数据源']].getFactorData(ifactor_name=self.SysArgs['基准对冲']['基准价格因子'],dates=dates,ids=[self.SysArgs['基准对冲']['基准 ID']])
        return Rslt.loc[:,self.SysArgs['基准对冲']['基准 ID']]
    # 获取基准的单位净值序列
    def getBenchmarkUnitizedWealth(self,price_seq):
        NotNaNPrice = price_seq[pd.notnull(price_seq)]
        if (NotNaNPrice.shape[0]==0) or (NotNaNPrice.iloc[0]==0):
            return price_seq
        else:
            return price_seq/NotNaNPrice.iloc[0]
    # 获取基准的收益率序列,Series
    def getBenchmarkYield(self,price_seq):
        YieldSeq = pd.Series(0.0,index=price_seq.index,name='基准收益率')
        YieldSeq.iloc[1:] = (price_seq.iloc[1:].values-price_seq.iloc[:-1].values)/price_seq.iloc[:-1].values
        return YieldSeq
    # 获取对冲组合收益率序列,Series,使用结果集里的基准来计算
    def getLBYield(self,yield_seq,benchmark_yield):
        AdjustDates = self.SysArgs['基准对冲']['空头平衡时点']
        if AdjustDates==[]:
            AdjustDates = self.SignalPrc.TradeDates
        if AdjustDates is None:
            return yield_seq-benchmark_yield
        UnitedWealth, BenchmarkUnitedWealth,LSUnitedWealth = 1, 1, 1
        LSYield = []
        TempInd = 1
        nTradeDate = len(AdjustDates)
        if nTradeDate==0:
            return pd.Series(1,index=yield_seq.index)
        for iDate in yield_seq.index:
            if iDate < AdjustDates[0]:
                LSYield.append(0)
                continue
            elif iDate == AdjustDates[0]:
                LSYield.append(yield_seq.loc[iDate])
                UnitedWealth *= (1+yield_seq.loc[iDate])
                BenchmarkUnitedWealth = UnitedWealth
                LSUnitedWealth = UnitedWealth+BenchmarkUnitedWealth*self.SysArgs['基准对冲']['保证金率']
                continue
            iRet = UnitedWealth*yield_seq.loc[iDate]-BenchmarkUnitedWealth*benchmark_yield.loc[iDate]
            LSYield.append(iRet/LSUnitedWealth)
            LSUnitedWealth += iRet
            UnitedWealth *= (1+yield_seq.loc[iDate])
            BenchmarkUnitedWealth *= (1+benchmark_yield.loc[iDate])
            if (TempInd<nTradeDate) and (iDate >= AdjustDates[TempInd]):
                UnitedWealth = LSUnitedWealth
                BenchmarkUnitedWealth = LSUnitedWealth
                LSUnitedWealth = UnitedWealth+BenchmarkUnitedWealth*self.SysArgs['基准对冲']['保证金率']
                TempInd += 1
        return pd.Series(LSYield,index=yield_seq.index)
    # 获取对冲组合收益率序列,Series,使用对冲账户的结果来计算
    def getArbitrageYield(self,long_wealth,init_wealth):
        nRebalanceDate = len(self.ArbitrageAccount.RebalanceDates)
        if nRebalanceDate==0:
            ArbitrageYield = long_wealth.diff()
            ArbitrageYield.iloc[0] = 0.0
            ArbitrageYield.iloc[1:] = ArbitrageYield.iloc[1:].values/long_wealth.iloc[:-1].values
            return ArbitrageYield
        ArbitrageYield = []
        Margin = pd.Series(self.ArbitrageAccount.Margin,index=self.ArbitrageAccount.Dates)
        MarginPreTrade = pd.Series(self.ArbitrageAccount.MarginPreTrade,index=self.ArbitrageAccount.RebalanceDates)
        InitMargin = pd.Series(self.ArbitrageAccount.InitMargin,index=self.ArbitrageAccount.RebalanceDates)
        LongWealthPreTrade = pd.Series(self.LongAccount.WealthPreTrade,index=self.LongAccount.Dates)+pd.Series(self.CashAccount.WealthPreTrade,index=self.CashAccount.Dates)
        if long_wealth.index[0]==self.ArbitrageAccount.RebalanceDates[0]:
            ArbitrageYield.append(((long_wealth.iloc[0]+Margin.iloc[0])-(init_wealth+InitMargin.iloc[0]))/abs(init_wealth+InitMargin.iloc[0]))
            RebalanceInd = 1
        else:
            ArbitrageYield.append((long_wealth.iloc[0]/init_wealth-1))
            RebalanceInd = 0
        for i in range(1,long_wealth.shape[0],1):
            iDate = long_wealth.index[i]
            if (RebalanceInd==nRebalanceDate) or (iDate<self.ArbitrageAccount.RebalanceDates[RebalanceInd]):# 没有发生交易和资金注入抽离, 跟踪
                ArbitrageYield.append(((long_wealth.iloc[i]+Margin.iloc[i]) - (long_wealth.iloc[i-1]+Margin.iloc[i-1])) / abs(long_wealth.iloc[i-1]+Margin.iloc[i-1]))
            elif iDate==self.ArbitrageAccount.RebalanceDates[RebalanceInd]:# 发生交易和资金注入抽离
                PreTradeYield = ((LongWealthPreTrade.iloc[i]+MarginPreTrade.loc[iDate]) - (long_wealth.iloc[i-1]+Margin.iloc[i-1])) / abs(long_wealth.iloc[i-1]+Margin.iloc[i-1])
                AfterTradeYield = ((long_wealth.iloc[i]+Margin.iloc[i]) - (LongWealthPreTrade.iloc[i]+InitMargin.loc[iDate])) / abs(LongWealthPreTrade.iloc[i]+InitMargin.loc[iDate])
                ArbitrageYield.append((1+PreTradeYield)*(1+AfterTradeYield)-1)
                RebalanceInd += 1
        return pd.Series(ArbitrageYield,index=long_wealth.index)
    # 获取多空组合收益率序列,Series,使用空头账户里的结果来计算
    def getLSYield(self,long_wealth,init_wealth):
        nTradeDate = len(self.ShortAccount.TradeDates)
        if nTradeDate==0:
            LSYield = long_wealth.diff()
            LSYield.iloc[0] = 0.0
            LSYield.iloc[1:] = LSYield.iloc[1:].values/long_wealth.iloc[:-1].values
            return LSYield
        LSYield = []
        Margin = pd.Series(self.ShortAccount.Margin,index=self.ShortAccount.Dates)
        MarginPreTrade = pd.Series(self.ShortAccount.MarginPreTrade,index=self.ShortAccount.TradeDates)
        InitMargin = pd.Series(self.ShortAccount.InitMargin,index=self.ShortAccount.TradeDates)
        LongWealthPreTrade = pd.Series(self.LongAccount.WealthPreTrade,index=self.LongAccount.Dates)+pd.Series(self.CashAccount.WealthPreTrade,index=self.CashAccount.Dates)
        if long_wealth.index[0]==self.ShortAccount.TradeDates[0]:
            LSYield.append(((long_wealth.iloc[0]+Margin.iloc[0])-(init_wealth+InitMargin.iloc[0]))/abs(init_wealth+InitMargin.iloc[0]))
            TradeInd = 1
        else:
            LSYield.append((long_wealth.iloc[0]/init_wealth-1))
            TradeInd = 0
        for i in range(1,long_wealth.shape[0],1):
            iDate = long_wealth.index[i]
            if (TradeInd>=nTradeDate) or (iDate<self.ShortAccount.TradeDates[TradeInd]):# 没有发生交易和资金注入抽离, 跟踪
                LSYield.append(((long_wealth.iloc[i]+Margin.iloc[i]) - (long_wealth.iloc[i-1]+Margin.iloc[i-1])) / abs(long_wealth.iloc[i-1]+Margin.iloc[i-1]))
            elif iDate==self.ShortAccount.TradeDates[TradeInd]:# 发生交易和资金注入抽离
                PreTradeYield = ((LongWealthPreTrade.iloc[i]+MarginPreTrade.loc[iDate]) - (long_wealth.iloc[i-1]+Margin.iloc[i-1])) / abs(long_wealth.iloc[i-1]+Margin.iloc[i-1])
                AfterTradeYield = ((long_wealth.iloc[i]+Margin.iloc[i]) - (LongWealthPreTrade.iloc[i]+InitMargin.loc[iDate])) / abs(LongWealthPreTrade.iloc[i]+InitMargin.loc[iDate])
                LSYield.append((1+PreTradeYield)*(1+AfterTradeYield)-1)
                TradeInd += 1
        return pd.Series(LSYield,index=long_wealth.index)
    # 生成报告
    def genReport(self):
        output = self.genSummaryReport()
        if self.SysArgs['日历分析']['年度统计']:
            output=self.genYearReport(output)
        if self.SysArgs['日历分析']['月度统计']!='不统计':
            output.update(self.genMonthReport(output))
        if self.SysArgs['日历分析']['日度统计']:
            output.update(self.genDayReport(output))
        if self.SysArgs['分类分析']['分类分析']:
            output.update(self.genClassAnalysisReport())
        if self.SysArgs['滚动分析']['滚动分析']:
            output.update(self.genRollingAnalysisReport(output))
        return output
    # 生成统计结果
    def genSummaryReport(self):
        output = {}
        output['换手率'] = self.Turnover.copy()
        if self.SysArgs['统计日期序列']!=[]:
            output['净值'] = self.Wealth.loc[self.SysArgs['统计日期序列']].copy()
            InitWealth = output['净值']['总财富'].iloc[0]
            ResultDate = self.SysArgs['统计日期序列']
        else:
            output['净值'] = self.Wealth.copy()
            InitWealth = self.InitWealth
            ResultDate = self.CashAccount.Dates
        output['净值']['策略多头'] = output['净值']['总财富']/InitWealth
        LongRet = StrategyTestFun.calcYieldSeq(output['净值']['总财富'].values, init_wealth=InitWealth)
        output['收益率'] = pd.DataFrame(index=output['净值']['总财富'].index)
        output['收益率']['策略多头'] = LongRet
        if self.ShortAccount.SysArgs['启用卖空']:
            LSYield = self.getLSYield(self.Wealth['总财富'], self.InitWealth)
            output['净值']['策略多空'] = StrategyTestFun.calcWealthSeq(LSYield.loc[ResultDate].values)
            output['收益率']['策略多空'] = StrategyTestFun.calcYieldSeq(output['净值']['策略多空'].values, init_wealth=1.0)
            InitData = (InitWealth,0,InitWealth,1,1)
        else:
            InitData = (InitWealth,0,InitWealth,1)
        InitDate = Datetime2DateStr(DateStr2Datetime(ResultDate[0])-datetime.timedelta(1))
        if self.ArbitrageAccount.SysArgs['启用对冲']:
            BenchmarkPrice = self.ArbitrageAccount.StdDataSource.getFactorData(ifactor_name=self.ArbitrageAccount.SysArgs['结算价'],dates=self.CashAccount.Dates,ids=[self.ArbitrageAccount.SysArgs['目标ID']])
            BenchmarkPrice = BenchmarkPrice[self.ArbitrageAccount.SysArgs['目标ID']].loc[ResultDate]
            output['收益率']['基准'] = self.getBenchmarkYield(price_seq=BenchmarkPrice)
            output['净值']['基准'] = self.getBenchmarkUnitizedWealth(price_seq=BenchmarkPrice)
            ArbitrageYield = self.getArbitrageYield(self.Wealth['总财富'],self.InitWealth)
            output['净值']['对冲'] = StrategyTestFun.calcWealthSeq(ArbitrageYield.loc[ResultDate].values)
            output['收益率']['对冲'] = StrategyTestFun.calcYieldSeq(output['净值']['对冲'].values, init_wealth=1.0)
            InitData = pd.DataFrame([InitData+(1,1)],index=[InitDate],columns=output['净值'].columns)
        elif self.SysArgs['基准对冲']['基准 ID']!='无':
            BenchmarkPrice = self.getBenchmarkPrice(dates=self.CashAccount.Dates)
            BenchmarkYield =  self.getBenchmarkYield(price_seq=BenchmarkPrice)
            output['净值']['基准'] = self.getBenchmarkUnitizedWealth(price_seq=BenchmarkPrice.loc[ResultDate])
            output['收益率']['基准'] = self.getBenchmarkYield(price_seq=output['净值']['基准'])
            ArbitrageYield = self.getLBYield(pd.Series(StrategyTestFun.calcYieldSeq(self.Wealth['总财富'].values, init_wealth=self.InitWealth), index=self.Wealth['总财富'].index),BenchmarkYield)
            output['净值']['对冲'] = StrategyTestFun.calcWealthSeq(ArbitrageYield.loc[ResultDate].values)
            output['收益率']['对冲'] = StrategyTestFun.calcYieldSeq(output['净值']['对冲'].values,1.0)
            InitData = pd.DataFrame([InitData+(1,1)],index=[InitDate],columns=output['净值'].columns)
        else:
            InitData = pd.DataFrame([InitData],index=[InitDate],columns=output['净值'].columns)
        output['净值'] = InitData.append(output['净值'])
        StartDate = output['净值'].index[0]
        EndDate = output['净值'].index[-1]
        SummaryData = OrderedDict()# 统计数据
        SummaryIndex = ['起始日','结束日']
        SummaryData['策略多头'] = [ResultDate[0],ResultDate[-1]]
        SummaryIndex.append('日期数')
        SummaryData['策略多头'].append(len(ResultDate))
        SummaryIndex.append('总收益')
        SummaryData['策略多头'].append(output['净值']['总财富'].iloc[-1]-InitWealth)
        SummaryIndex.append('总收益率')
        SummaryData['策略多头'].append(output['净值']['总财富'].iloc[-1]/InitWealth-1)
        SummaryIndex.append('年化收益率')
        SummaryData['策略多头'].append(StrategyTestFun.calcAnnualYield(output['净值']['策略多头'].values,start_date=StartDate,end_date=EndDate))
        SummaryIndex.append('年化波动率')
        SummaryData['策略多头'].append(StrategyTestFun.calcAnnualVolatility(output['净值']['策略多头'].values,start_date=StartDate,end_date=EndDate))
        SummaryIndex.append('Sharpe比率')
        SummaryData['策略多头'].append(SummaryData['策略多头'][5]/SummaryData['策略多头'][6])
        SummaryIndex.append('胜率')
        SummaryData['策略多头'].append((output['收益率']['策略多头']>=0).sum()/(SummaryData['策略多头'][2]-1))
        MaxDrawdownRate,MaxDrawdownStartPos,MaxDrawdownEndPos,DrawdownSeq = StrategyTestFun.calcDrawdown(wealth_seq=output['净值']['策略多头'].values)
        output['回撤'] = pd.DataFrame(DrawdownSeq,index=output['净值'].index,columns=['策略多头'])
        SummaryIndex.append('最大回撤率')
        SummaryData['策略多头'].append(abs(MaxDrawdownRate))
        SummaryIndex.append('最大回撤开始日期')
        SummaryData['策略多头'].append(output['净值'].index[MaxDrawdownStartPos])
        SummaryIndex.append('最大回撤结束日期')
        SummaryData['策略多头'].append(output['净值'].index[MaxDrawdownEndPos])
        if self.ShortAccount.SysArgs['启用卖空']:
            SummaryData['策略多空'] = SummaryData['策略多头'][:3]+[output['净值']['策略多空'][-1]-1,output['净值']['策略多空'][-1]-1]
            SummaryData['策略多空'].append(StrategyTestFun.calcAnnualYield(output['净值']['策略多空'].values,start_date=StartDate,end_date=EndDate))
            SummaryData['策略多空'].append(StrategyTestFun.calcAnnualVolatility(output['净值']['策略多空'].values,start_date=StartDate,end_date=EndDate))
            SummaryData['策略多空'].append(SummaryData['策略多空'][5]/SummaryData['策略多空'][6])
            SummaryData['策略多空'].append((output['收益率']['策略多空']>=0).sum()/(SummaryData['策略多空'][2]-1))
            MaxDrawdownRate,MaxDrawdownStartPos,MaxDrawdownEndPos,DrawdownSeq = StrategyTestFun.calcDrawdown(wealth_seq=output['净值']['策略多空'].values)
            output['回撤']['策略多空'] = DrawdownSeq
            SummaryData['策略多空'] += [abs(MaxDrawdownRate),output['净值'].index[MaxDrawdownStartPos],output['净值'].index[MaxDrawdownEndPos]]
        if self.ArbitrageAccount.SysArgs['启用对冲'] or (self.SysArgs['基准对冲']['基准 ID']!='无'):
            SummaryData['基准'] = SummaryData['策略多头'][:3]+[output['净值']['基准'][-1]-1,output['净值']['基准'][-1]-1]
            SummaryData['对冲'] = SummaryData['策略多头'][:3]+[output['净值']['对冲'][-1]-1,output['净值']['对冲'][-1]-1]
            SummaryData['基准'].append(StrategyTestFun.calcAnnualYield(output['净值']['基准'].values,start_date=StartDate,end_date=EndDate))
            SummaryData['对冲'].append(StrategyTestFun.calcAnnualYield(output['净值']['对冲'].values,start_date=StartDate,end_date=EndDate))
            SummaryData['基准'].append(StrategyTestFun.calcAnnualVolatility(output['净值']['基准'].values,start_date=StartDate,end_date=EndDate))
            SummaryData['对冲'].append(StrategyTestFun.calcAnnualVolatility(output['净值']['对冲'].values,start_date=StartDate,end_date=EndDate))
            SummaryData['基准'].append(SummaryData['基准'][5]/SummaryData['基准'][6])
            SummaryData['对冲'].append(SummaryData['对冲'][5]/SummaryData['对冲'][6])
            SummaryData['基准'].append((output['收益率']['基准']>=0).sum()/(SummaryData['基准'][2]-1))
            SummaryData['对冲'].append((output['收益率']['对冲']>=0).sum()/(SummaryData['对冲'][2]-1))
            MaxDrawdownRate,MaxDrawdownStartPos,MaxDrawdownEndPos,DrawdownSeq = StrategyTestFun.calcDrawdown(wealth_seq=output['净值']['基准'].values)
            output['回撤']['基准'] = DrawdownSeq
            SummaryData['基准'] += [abs(MaxDrawdownRate),output['净值'].index[MaxDrawdownStartPos],output['净值'].index[MaxDrawdownEndPos]]
            MaxDrawdownRate,MaxDrawdownStartPos,MaxDrawdownEndPos,DrawdownSeq = StrategyTestFun.calcDrawdown(wealth_seq=output['净值']['对冲'].values)
            output['回撤']['对冲'] = DrawdownSeq
            SummaryData['对冲'] += [abs(MaxDrawdownRate),output['净值'].index[MaxDrawdownStartPos],output['净值'].index[MaxDrawdownEndPos]]
        output['统计数据'] = pd.DataFrame(SummaryData,index=SummaryIndex)
        return output
    # 生成分年度报告
    def genYearReport(self,output):
        Data = StrategyTestFun.calcReturnPerYear(output['净值'].iloc[:,3:].values, list(output["净值"].index), self.QSEnv.SysArgs.get("DateRuler",None))
        Data.columns = output['净值'].columns[3:]
        output['年度统计'] = Data
        return output
    # 生成月度报告
    def genMonthReport(self,output):
        MonthYield = [pd.Series(0.0,index=output['净值'].columns[3:]) for i in range(12)]
        MonthNum = [0 for i in range(12)]
        PreDate = output['净值'].index[0]
        StartInd = 0
        for i,iDate in enumerate(output['净值'].index[1:]):
            if iDate[:6]!=PreDate[:6]:# 进入新的月度
                if self.SysArgs['日历分析']['月度统计']=='计入下月统计':
                    iTargetMonth = int(PreDate[4:6])
                else:
                    iTargetMonth = int(PreDate[4:6])-1
                    if iTargetMonth==0:
                        iTargetMonth = 12
                MonthYield[iTargetMonth-1] += output['净值'].iloc[i,3:]/output['净值'].iloc[StartInd,3:]-1
                MonthNum[iTargetMonth-1] += 1
                StartInd = i
            PreDate = iDate
        MonthYield[int(iDate[4:6])-1] += output['净值'].iloc[-1,3:]/output['净值'].iloc[StartInd,3:]-1
        MonthNum[int(iDate[4:6])-1] += 1
        for i in range(12):
            if MonthNum[i]==0:
                MonthYield[i] = pd.Series(np.nan,index=output['净值'].columns[3:])
            else:
                MonthYield[i] = MonthYield[i]/MonthNum[i]
        output['月度统计'] = pd.DataFrame(MonthYield,index=[i+1 for i in range(12)])
        return output
    # 生成日度统计报告
    def genDayReport(self,output):
        Price = output['净值'].iloc[:,3:]
        DayReturn = {iCol:{} for iCol in Price.columns}
        DayNum = {iCol:{} for iCol in Price.columns}
        for i,iDate in enumerate(Price.index[self.SysArgs['日历分析']['向后统计天数']:]):
            iDay = iDate[4:]
            iEnd = i+self.SysArgs['日历分析']['向后统计天数']+self.SysArgs['日历分析']['向前统计天数']+1
            if iEnd>=Price.shape[0]:
                break
            for jCol in Price.columns:
                DayReturn[jCol][iDay] = DayReturn[jCol].get(iDay,0.0)+Price[jCol].iloc[i+self.SysArgs['日历分析']['向后统计天数']+self.SysArgs['日历分析']['向前统计天数']+1]/Price[jCol].iloc[i]-1
                DayNum[jCol][iDay] = DayNum[jCol].get(iDay,0.0)+1
        DayReturn = pd.DataFrame({iCol:pd.Series(DayReturn[jCol]) for iCol in DayReturn})
        DayReturn = DayReturn.sort_index()
        DayNum = pd.DataFrame({iCol:pd.Series(DayNum[jCol]) for iCol in DayNum})
        DayNum = DayNum.sort_index()
        output['日度统计'] = DayReturn/DayNum
        return output
    # 生成分类分析报告
    def genClassAnalysisReport(self):
        IndustryData = self.AllDSs[self.SysArgs['分类分析']['类别数据源']].getFactorData(ifactor_name=self.SysArgs['分类分析']['类别因子'])
        Price = self.LongAccount.StdDataSource.getFactorData(ifactor_name=self.LongAccount.SysArgs['结算价'])
        Price = Price.where(pd.notnull(Price),np.nan)
        PriceDiff = Price.diff()
        PriceDiff = PriceDiff.where(pd.notnull(PriceDiff),0)
        Position = self.LongAccount.Position
        AllIndustries = list(np.unique(IndustryData.values[pd.notnull(IndustryData.values)]))
        AllIndustries += ['无分类']
        IndustryNum = []
        IndustryPosRatio = []
        IndustryUnitizedWealth = []
        IndustryWealth = []
        IndustryYield = []
        PreDate = self.LongAccount.Dates[0]
        Cash = pd.Series(self.CashAccount.Wealth,index=self.CashAccount.Dates)
        EquityWealth = pd.Series(self.LongAccount.Wealth,index=self.LongAccount.Dates)
        TotalWealth = Cash+EquityWealth
        for i,iDate in enumerate(self.LongAccount.Dates):
            if i>0:
                iPrePriceData = iAllPrice
            else:
                iIndustryPrePos = {}
            iPos = pd.Series(Position[i])
            iIDs = list(iPos.index)
            iWealth = TotalWealth.iloc[i]
            iCash = Cash.iloc[i]
            iIndustryData = IndustryData.loc[iDate,iIDs]
            iPriceData = Price.loc[iDate,iIDs]
            iAllPrice = Price.loc[iDate,:]
            iPriceDiff = PriceDiff.loc[iDate,:]
            iIndustryNum = []
            iIndustryPosRatio = []
            iIndustryYield = []
            for jIndustry in AllIndustries:
                if jIndustry!='无分类':
                    ijMask = (iIndustryData==jIndustry)
                else:
                    ijMask = pd.isnull(iIndustryData)
                ijIndustryPos = iPos[ijMask]
                ijPriceData = iPriceData[ijMask]
                # 计算i期j行业持仓数量
                iIndustryNum.append(ijMask.sum())
                # 计算i期j行业资金占比
                ijIndustryPosRatio = (ijPriceData*ijIndustryPos).sum()/iWealth
                if pd.notnull(ijIndustryPosRatio):
                    iIndustryPosRatio.append(ijIndustryPosRatio)
                else:
                    iIndustryPosRatio.append(0)
                if i>0:
                    ijIndustryProfit = (iIndustryPrePos[jIndustry]*iPriceDiff[iIndustryPrePos[jIndustry].index]).sum()
                    ijIndustryProfit -= StrategyTestFun.calcTurnover(iIndustryPrePos[jIndustry]*iAllPrice[iIndustryPrePos[jIndustry].index],ijIndustryPos*ijPriceData)*self.LongAccount.SysArgs['交易费率']
                    ijPreWealth = (iIndustryPrePos[jIndustry]*iPrePriceData[iIndustryPrePos[jIndustry].index]).sum()
                    if pd.notnull(ijPreWealth) and (ijPreWealth!=0):
                        iIndustryYield.append(ijIndustryProfit/ijPreWealth)
                    else:
                        iIndustryYield.append(0)
                else:
                    iIndustryYield.append(-self.LongAccount.SysArgs['交易费率'])
                iIndustryPrePos[jIndustry] = ijIndustryPos
            IndustryNum.append(iIndustryNum+[len(iIDs)])
            IndustryPosRatio.append(iIndustryPosRatio+[iCash/iWealth])
            IndustryYield.append(iIndustryYield+[self.CashAccount.SysArgs['资金收益率']])
        output = {'子类持仓数':pd.DataFrame(IndustryNum,index=self.LongAccount.Dates,columns=AllIndustries+['总数'])}
        output['子类资金占比'] = pd.DataFrame(IndustryPosRatio,index=self.LongAccount.Dates,columns=AllIndustries+['现金'])
        output['子类收益率'] = pd.DataFrame(IndustryYield,index=self.LongAccount.Dates,columns=AllIndustries+['现金'])
        output['子类收益贡献'] = pd.DataFrame(np.row_stack((output['子类资金占比'].iloc[0,:].values*output['子类收益率'].iloc[0,:].values,output['子类资金占比'].iloc[:-1,:].values*output['子类收益率'].iloc[1:,:].values)),index=self.LongAccount.Dates,columns=AllIndustries+['现金'])
        return output
    # 生成滚动分析报告
    def genRollingAnalysisReport(self,output):
        WealthSeq = output['净值'].iloc[:,3:]
        output['滚动年化收益率'] = pd.DataFrame(np.zeros(WealthSeq.shape),index=WealthSeq.index,columns=WealthSeq.columns)
        output['滚动年化波动率'] = pd.DataFrame(np.zeros(WealthSeq.shape),index=WealthSeq.index,columns=WealthSeq.columns)
        output['滚动夏普率'] = pd.DataFrame(np.zeros(WealthSeq.shape),index=WealthSeq.index,columns=WealthSeq.columns)
        NumPerYear = (WealthSeq.shape[0]-1)/((DateStr2Datetime(WealthSeq.index[-1])-DateStr2Datetime(WealthSeq.index[0])).days/365)
        for iCol in WealthSeq.columns:
            output['滚动年化收益率'][iCol] = StrategyTestFun.calcExpandingAnnualYieldSeq(WealthSeq[iCol].values,self.SysArgs['滚动分析']['最小窗口'],NumPerYear)
            output['滚动年化波动率'][iCol] = StrategyTestFun.calcExpandingAnnualVolatilitySeq(WealthSeq[iCol].values,self.SysArgs['滚动分析']['最小窗口'],NumPerYear)
            output['滚动夏普率'][iCol] = output['滚动年化收益率'][iCol]/output['滚动年化波动率'][iCol]
        return output
    # 获取可以用于参数分析结果输出的结果
    def getAvailableArgAnalyzeOutput(self):
        return ['统计数据','净值','收益率','换手率','回撤','年度统计']
    # 保存自身信息
    def saveInfo(self,container):
        container["SysArgs"] = self.SysArgs
        return container
    # 恢复信息
    def loadInfo(self,container):
        self.SysArgs = container["SysArgs"]
        _,self.SysArgInfos = self.genSysArgInfo(self.SysArgs)
        return 0