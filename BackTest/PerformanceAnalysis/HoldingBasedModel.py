# coding=utf-8
"""基于持仓数据的绩效分析模型(TODO)"""
import datetime as dt

import pandas as pd
import numpy as np
import statsmodels.api as sm
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Dict, Instance, on_trait_change

from QuantStudio import __QS_Error__
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.RiskModel.RiskDataSource import RiskDataSource
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var

# TODO
class FMPModel(BaseModule):
    """基于特征因子模拟组合的绩效分析模型"""
    Holding = Enum(None, arg_type="SingleOption", label="策略持仓", order=0)
    HoldingType = Enum("数量", "权重", arg_type="SingleOption", label="策略持仓类型", order=1)
    BenchmarkHolding = Enum(None, arg_type="SingleOption", label="基准持仓", order=2)
    BenchmarkHoldingType = Enum("数量", "权重", arg_type="SingleOption", label="基准持仓类型", order=3)
    AttributeFactors = ListStr(arg_type="MultiOption", label="特征因子", order=4, option_range=())
    IndustryFactor = Enum("无", arg_type="SingleOption", label="行业因子", order=5)
    PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=6)
    RiskDS = Instance(RiskDataSource, arg_type="RiskDS", label="风险数据源", order=7)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=8)
    def __init__(self, factor_table, name="因子模拟组合绩效分析模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("Holding", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="策略持仓", order=0))
        self.Holding = DefaultNumFactorList[0]
        self.add_trait("BenchmarkHolding", Enum(*(["无"]+DefaultNumFactorList), arg_type="SingleOption", label="基准持仓", order=2))
        self.add_trait("AttributeFactors", ListStr(arg_type="MultiOption", label="特征因子", order=4, option_range=tuple(DefaultNumFactorList)))
        self.AttributeFactors.append(DefaultNumFactorList[-1])
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=5))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=6))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        self.HoldingDates = list(self.Holding.keys())
        self.HoldingDates.sort()
        self.nHolding = len(self.HoldingDates)
        self.HoldingInd = -1
        self.BenchmarkHoldingDates = list(self.BenchmarkHolding.keys())
        self.BenchmarkHoldingDates.sort()
        self.nBenchmarkHolding = len(self.BenchmarkHoldingDates)
        self.BenchmarkHoldingInd = -1
        if self.IndustryFactor!="无":
            self._AllIndustries = self._FactorTable.readData(factor_names=[self.IndustryFactor], dts=self._FactorTable.getDateTime(ifactor_name=self.IndustryFactor), ids=self._FactorTable.getID(ifactor_name=self.IndustryFactor)).iloc[0]
            self._AllIndustries = sorted(set(pd.unique(self._AllIndustries.values.flatten())).difference({None, np.nan}))
            self._AttributeFactors = self.AttributeFactors+self._AllIndustries
        else:
            self._AttributeFactors = self.AttributeFactors
        self._Output = {}
        self._Output["因子暴露"] = pd.DataFrame(0.0, columns=self._AttributeFactors)
        self._Output["风险调整的因子暴露"] = pd.DataFrame(0.0,columns=self._AttributeFactors)
        self._Output["风险贡献"] = pd.DataFrame(0.0, columns=self._AttributeFactors+["Alpha"])
        self._Output["收益贡献"] = pd.DataFrame(0.0, columns=self._AttributeFactors+["Alpha"])
        self._Output["因子收益"] = pd.DataFrame(np.nan, columns=self._AttributeFactors)
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def _calPortfolioReturn(self, cur_date, next_date, portfolio):
        Price = self.DSs[self.SysArgs['归因数据']['主数据源']].getFactorData(ifactor_name=self.SysArgs['归因数据']['价格因子'],dates=[cur_date,next_date],ids=list(portfolio.index))
        Ret = (Price.loc[next_date]-Price.loc[cur_date])/Price.loc[cur_date]
        return (Ret*portfolio).sum()
    def _calPortfolioRisk(self, cov_matrix, portfolio):
        return np.sqrt(np.dot(np.dot(portfolio.values,cov_matrix),portfolio.values))
    def _estimateFactorReturn(self, cur_date, next_date, fmp):
        Price = self.DSs[self.SysArgs['归因数据']['主数据源']].getFactorData(ifactor_name=self.SysArgs['归因数据']['价格因子'],dates=[cur_date,next_date],ids=list(fmp.index))
        Ret = (Price.loc[next_date]-Price.loc[cur_date])/Price.loc[cur_date]
        FactorRet = pd.Series(np.nan,index=fmp.columns)
        for iFactor in fmp:
            FactorRet[iFactor] = (Ret*fmp.loc[:,iFactor]).sum()
        return FactorRet
    def _getCurPortfolio(self, idt):
        while (self.HoldingInd<self.nHolding-1) and (cur_date>=self.HoldingDates[self.HoldingInd+1]):
            self.HoldingInd += 1
        if self.HoldingInd==-1:
            return None
        HoldingDate = self.HoldingDates[self.HoldingInd]
        Holding = pd.Series(self.Holding[HoldingDate])
        IDs = list(Holding.index)
        Price = self.DSs[self.SysArgs['归因数据']['主数据源']].getFactorData(ifactor_name=self.SysArgs['归因数据']['价格因子'],dates=[cur_date])
        Price = Price.loc[cur_date]
        if self.SysArgs['策略持仓类型']=='数量':
            Portfolio = Price.loc[IDs]*Holding
        else:
            PrePrice = self.DSs[self.SysArgs['归因数据']['主数据源']].getFactorData(ifactor_name=self.SysArgs['归因数据']['价格因子'],dates=[HoldingDate])
            PrePrice = PrePrice.loc[HoldingDate]
            Portfolio = (1+(Price.loc[IDs]-PrePrice.loc[IDs])/PrePrice.loc[IDs])*Holding
        Portfolio = Portfolio[pd.notnull(Portfolio)]
        NegMask = (Portfolio<0)
        TotalNegWeight = Portfolio[NegMask].sum()
        if TotalNegWeight!=0:
            Portfolio[NegMask] = Portfolio[NegMask]/TotalNegWeight
        PosMask = (Portfolio>0)
        TotalPosWeight = Portfolio[PosMask].sum()
        if TotalPosWeight!=0:
            Portfolio[PosMask] = Portfolio[PosMask]/TotalPosWeight
            Portfolio[NegMask] = Portfolio[NegMask]*TotalNegWeight/TotalPosWeight
        return Portfolio
    def _getCurBenchmarkPortfolio(self, cur_date):
        while (self.BenchmarkHoldingInd<self.nBenchmarkHolding-1) and (cur_date>=self.BenchmarkHoldingDates[self.BenchmarkHoldingInd+1]):
            self.BenchmarkHoldingInd += 1
        if self.BenchmarkHoldingInd==-1:
            return None
        HoldingDate = self.BenchmarkHoldingDates[self.BenchmarkHoldingInd]
        Holding = pd.Series(self.BenchmarkHolding[HoldingDate])
        IDs = list(Holding.index)
        Price = self.DSs[self.SysArgs['归因数据']['主数据源']].getFactorData(ifactor_name=self.SysArgs['归因数据']['价格因子'],dates=[cur_date])
        Price = Price.loc[cur_date]
        if self.SysArgs['基准持仓类型']=='数量':
            Portfolio = Price.loc[IDs]*Holding
        else:
            PrePrice = self.DSs[self.SysArgs['归因数据']['主数据源']].getFactorData(ifactor_name=self.SysArgs['归因数据']['价格因子'],dates=[HoldingDate])
            PrePrice = PrePrice.loc[HoldingDate]
            Portfolio = (1+(Price.loc[IDs]-PrePrice.loc[IDs])/PrePrice.loc[IDs])*Holding
        Portfolio = Portfolio[pd.notnull(Portfolio)]
        NegMask = (Portfolio<0)
        TotalNegWeight = Portfolio[NegMask].sum()
        if TotalNegWeight!=0:
            Portfolio[NegMask] = Portfolio[NegMask]/TotalNegWeight
        PosMask = (Portfolio>0)
        TotalPosWeight = Portfolio[PosMask].sum()
        if TotalPosWeight!=0:
            Portfolio[PosMask] = Portfolio[PosMask]/TotalPosWeight
            Portfolio[NegMask] = Portfolio[NegMask]*TotalNegWeight/TotalPosWeight
        return Portfolio
    def __QS_move__(self, idt, *args, **kwargs):
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
        LastInd = self._CurCalcInd - 1
        LastDateTime = self._Model.DateTimeSeries[LastInd]
        Portfolio = self._getCurPortfolio(idt)
        if Portfolio is None: return 0
        Portfolio = Portfolio.astype('float')
        IDs = self.DSs[self.SysArgs['归因数据']['主数据源']].getID(cur_date,is_filtered=True)
        BenchmarkPortfolio = self._getCurBenchmarkPortfolio(cur_date)
        if BenchmarkPortfolio is not None:
            BenchmarkPortfolio = BenchmarkPortfolio.astype('float')
        TempPortfolio = pd.Series(0.0,index=IDs)
        TempPortfolio[Portfolio.index] = Portfolio
        if BenchmarkPortfolio is not None:
            TempBenchmarkPortfolio = pd.Series(0.0,index=IDs)
            TempBenchmarkPortfolio[BenchmarkPortfolio.index] = BenchmarkPortfolio
            Portfolio = TempPortfolio-TempBenchmarkPortfolio
        else:
            Portfolio = TempPortfolio
        # 计算因子模拟组合
        self.SysArgs['风险数据源'].MoveOn(cur_date)
        CovMatrix = self.SysArgs['风险数据源'].getDateCov(cur_date,drop_na=True)
        FactorExpose = self.DSs[self.SysArgs['归因数据']['主数据源']].getDateData(idate=cur_date,factor_names=self.SysArgs['归因数据']['归因因子'],ids=IDs)
        FactorExpose = FactorExpose.dropna(axis=0)
        IDs = list(set(FactorExpose.index).intersection(set(CovMatrix.index)))
        IDs.sort()
        CovMatrix = CovMatrix.loc[IDs,IDs]
        FactorExpose = FactorExpose.loc[IDs,:]
        if self.SysArgs['归因数据']['归因行业因子']!='无':
            DummyData = DummyVarTo01Var(FactorExpose.pop(self.SysArgs['归因数据']['归因行业因子']),ignore_nonstring=True)
            FactorExpose = pd.merge(FactorExpose,DummyData,left_index=True,right_index=True)
        CovMatrixInv = np.linalg.inv(CovMatrix)
        FMPHolding = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(FactorExpose.values),CovMatrixInv),FactorExpose.values)),np.transpose(FactorExpose.values)),CovMatrixInv)
        # 计算持仓对因子模拟组合的投资组合
        Portfolio = Portfolio.loc[IDs]
        NegMask = (Portfolio<0)
        TotalNegWeight = Portfolio[NegMask].sum()
        if TotalNegWeight!=0:
            Portfolio[NegMask] = Portfolio[NegMask]/TotalNegWeight
        PosMask = (Portfolio>0)
        TotalPosWeight = Portfolio[PosMask].sum()
        if TotalPosWeight!=0:
            Portfolio[PosMask] = Portfolio[PosMask]/TotalPosWeight
            Portfolio[NegMask] = Portfolio[NegMask]*TotalNegWeight/TotalPosWeight
        Beta = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(FMPHolding,CovMatrix.values),np.transpose(FMPHolding))),FMPHolding),CovMatrix.values),Portfolio.values)
        self._Output['因子暴露'].loc[cur_date,FactorExpose.columns] = Beta
        self._Output['风险调整的因子暴露'].loc[cur_date,FactorExpose.columns] = np.sqrt(np.diag(np.dot(np.dot(FMPHolding,CovMatrix.values),np.transpose(FMPHolding))))*Beta
        self._Output['风险贡献'].loc[cur_date,FactorExpose.columns] = np.dot(np.dot(FMPHolding,CovMatrix.values),Portfolio.values)/np.sqrt(np.dot(np.dot(Portfolio.values,CovMatrix.values),Portfolio.values))*Beta
        self._Output['风险贡献'].loc[cur_date,'残差项'] = self._calPortfolioRisk(CovMatrix,Portfolio)-self._Output['风险贡献'].loc[cur_date,FactorExpose.columns].sum()
        if cur_date!=self.Dates[-1]:
            FactorRet = self._estimateFactorReturn(cur_date,self.Dates[cur_ind+1],pd.DataFrame(np.transpose(FMPHolding),index=IDs,columns=FactorExpose.columns))
            self._Output['因子收益'].loc[self.Dates[cur_ind+1],FactorExpose.columns] = FactorRet
            self._Output['收益贡献'].loc[cur_date,FactorExpose.columns] = FactorRet*self._Output['因子暴露'].loc[cur_date,FactorExpose.columns]
            self._Output['收益贡献'].loc[cur_date,'残差项'] = self._calPortfolioReturn(cur_date,self.Dates[cur_ind+1],Portfolio)-self._Output['收益贡献'].loc[cur_date,FactorExpose.columns].sum()
        return 0
    def __QS_end__(self):
        self.SysArgs['风险数据源'].endDS()
        self._Output['风险贡献占比'] = self._Output['风险贡献'].divide(self._Output['风险贡献'].sum(axis=1),axis=0)
        AvgData = pd.DataFrame(columns=['因子暴露','风险调整的因子暴露','风险贡献','风险贡献占比','收益贡献'],index=self._AttributeFactors+['残差项'])
        AvgData['因子暴露'][self._AttributeFactors] = self._Output['因子暴露'].mean(axis=0)
        AvgData['风险调整的因子暴露'][self._AttributeFactors] = self._Output['风险调整的因子暴露'].mean(axis=0)
        AvgData['风险贡献'] = self._Output['风险贡献'].mean(axis=0)
        AvgData['风险贡献占比'] = self._Output['风险贡献占比'].mean(axis=0)
        AvgData['收益贡献'] = self._Output['收益贡献'].mean(axis=0)
        self._Output['指标均值'] = AvgData
        return self._Output

# TODO
class BrinsonModel(BaseModule):
    """Brinson 绩效分析模型"""
    def __init__(self, factor_table, name="Brinson绩效分析模型", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        return super().__init__(name=name, sys_args=sys_args, config_file=None, **kwargs)
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        pass
        return (self._FactorTable, )
    def __QS_move__(self, idt):
        return 0
    def __QS_end__(self):
        return 0