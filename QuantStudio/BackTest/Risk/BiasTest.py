# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from collections import OrderedDict

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, ListInt, Int, Str, Instance

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.RiskDataBase.RiskDB import RiskTable
from QuantStudio.Tools.StrategyTestFun import genRandomPortfolio
from QuantStudio.RiskModel.RiskModelFun import dropRiskMatrixNA

class BiasTest(BaseModule):
    """BiasTest"""
    RiskTable = Instance(RiskTable, arg_type="RiskTable", label="风险表", order=0)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=1)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=2)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=3)
    #WeightFactors = ListStr(arg_type="MultiOption", label="权重因子", order=4 option_range=())
    #StyleFactors = ListStr(arg_type="MultiOption", label="风格因子", order=5, option_range=())
    #IndustryFactor = Enum("无", arg_type="SingleOption", label="行业因子", order=6)
    #IndustryNeutralFactors = ListStr(arg_type="MultiOption", label="行业中性因子", order=7, option_range=())
    RandomNums = ListInt([20,50,100,200], arg_type="NultiOpotion", label="随机组合", order=8)
    LookBack = Int(12, arg_type="Integer", label="回溯期数", order=9)
    def __init__(self, factor_table, name="BiasTest", sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name=name, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=3))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("WeightFactors", ListStr(["等权"], arg_type="MultiOption", label="权重因子", order=4, option_range=tuple(["等权"]+DefaultNumFactorList)))
        self.add_trait("StyleFactors", ListStr(arg_type="MultiOption", label="风格因子", order=5, option_range=tuple(DefaultNumFactorList)))
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=6))
        self.add_trait("IndustryNeutralFactors", ListStr(arg_type="MultiOption", label="行业中性因子", order=7, option_range=tuple(DefaultNumFactorList)))
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._WeightFactors = list(self.WeightFactors)
        self._HasEW = ("等权" in self._WeightFactors)
        if self._HasEW: self._WeightFactors.remove("等权")
        self._Output = {}
        self._CurCalcInd = 0
        self._Portfolios = OrderedDict()
        self._CovMatrix = None
        return (self._FactorTable, )
    def _genPortfolio(self, idt, ids):
        PortfolioDict = OrderedDict()
        if self._WeightFactors:
            WeightData = self._FactorTable.readData(factor_names=self._WeightFactors, dts=[idt], ids=ids).iloc[:, 0]
        else:
            WeightData = pd.DataFrame()
        if self._HasEW:
            WeightData["等权"] = pd.Series(1, index=ids) / len(ids)
        if self.IndustryFactor!="无":
            Industry = self._FactorTable.readData(factor_names=[self.IndustryFactor], dts=[idt], ids=ids).iloc[0, 0]
            AllIndustries = Industry[pd.notnull(Industry)].unique()
            AllIndustries.sort()
        if self.StyleFactors:
            StyleFactorData = self._FactorTable.readData(factor_names=list(self.StyleFactors), dts=[idt], ids=ids).iloc[:, 0]
        if self.IndustryNeutralFactors:
            IndNeutralData = self._FactorTable.readData(factor_names=list(self.IndustryNeutralFactors), dts=[idt], ids=ids).iloc[:, 0]
        for iWeightFactor in WeightData:
            iWeightData = WeightData[iWeightFactor]
            iMask = (pd.notnull(iWeightData) & (iWeightData!=0))
            iWeightData = iWeightData[iMask]
            # 全部 ID 组合
            PortfolioDict["全体%s加权组合" % (iWeightFactor,)] = iWeightData / iWeightData.abs().sum()
            # 行业组合
            if self.IndustryFactor!="无":
                for jIndustry in AllIndustries:
                    ijMask = (Industry[iMask]==jIndustry)
                    ijWeightData = iWeightData[ijMask]
                    PortfolioDict["%s行业%s加权组合" % (jIndustry, iWeightFactor)] = ijWeightData / ijWeightData.abs().sum()
                    # 行业中性组合
                    for kFactor in self.IndustryNeutralFactors:
                        kTopPortfolio = ("%sTop%s加权组合" % (kFactor, iWeightFactor))
                        kBottomPortfolio = ("%sBottom%s加权组合" % (kFactor, iWeightFactor))
                        ijkIndNeutralData= IndNeutralData[kFactor][iMask][ijMask]
                        ijkThreshold = ijkIndNeutralData.median()
                        PortfolioDict[kTopPortfolio] = PortfolioDict.get(kTopPortfolio, []) + ijkIndNeutralData[ijkIndNeutralData>ijkThreshold].index.tolist()
                        PortfolioDict[kBottomPortfolio] = PortfolioDict.get(kBottomPortfolio, []) + ijkIndNeutralData[ijkIndNeutralData<=ijkThreshold].index.tolist()
                for kFactor in self.IndustryNeutralFactors:
                    kTopPortfolio = ("%sTop%s加权组合" % (kFactor, iWeightFactor))
                    kPortfolio = iWeightData.loc[PortfolioDict.pop(kTopPortfolio)]
                    PortfolioDict[kTopPortfolio] = kPortfolio / kPortfolio.abs().sum()
                    kBottomPortfolio = ("%sBottom%s加权组合" % (kFactor, iWeightFactor))
                    kPortfolio = iWeightData.loc[PortfolioDict.pop(kBottomPortfolio)]
                    PortfolioDict[kBottomPortfolio] = kPortfolio / kPortfolio.abs().sum()
            # 风格因子组合
            for jFactor in self.StyleFactors:
                jFactorData = StyleFactorData[jFactor][iMask]
                ijWeightData = iWeightData[jFactorData>=jFactorData.quantile(0.8)]
                PortfolioDict["%s风格Top%s加权组合" % (jFactor, iWeightFactor)] = ijWeightData / ijWeightData.abs().sum()
                ijWeightData = iWeightData[jFactorData<=jFactorData.quantile(0.2)]
                PortfolioDict["%s风格Bottom%s加权组合" % (jFactor, iWeightFactor)] = ijWeightData / ijWeightData.abs().sum()
            # 随机组合
            for jNum in self.RandomNums:
                PortfolioDict["随机%d%s加权组合" % (jNum, iWeightFactor)] = genRandomPortfolio(ids, target_num=20, weight=iWeightData)
        return PortfolioDict
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            LastInd = self._CurCalcInd - 1
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            LastInd = self._CurCalcInd - 1
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (LastInd<0): return 0
        IDs = self._FactorTable.getFilteredID(idt=idt, id_filter_str=self.IDFilter)
        LastCovMatrix, self._CovMatrix = self._CovMatrix, dropRiskMatrixNA(self.RiskTable.readCov(dts=[idt], ids=IDs).iloc[0])
        IDs = self._CovMatrix.index.tolist()
        LastPortfolios, self._Portfolios = self._Portfolios, self._genPortfolio(idt, IDs)
        if not LastPortfolios:
            AllPortfolioNames = list(self._Portfolios)
            self._Output["Z-Score"] = pd.DataFrame(columns=AllPortfolioNames)
            self._Output["Robust Z-Score"] = pd.DataFrame(columns=AllPortfolioNames)
            self._Output["Bias 统计量"] = pd.DataFrame(columns=AllPortfolioNames)
            self._Output["Robust Bias 统计量"] = pd.DataFrame(columns=AllPortfolioNames)
            return 0
        else:
            self._Output["Robust Bias 统计量"].loc[idt] = self._Output["Bias 统计量"].loc[idt] = self._Output["Robust Z-Score"].loc[idt] = self._Output["Z-Score"].loc[idt] = np.nan
        Price = self._FactorTable.readData(dts=[LastDateTime, idt], ids=self._FactorTable.getID(ifactor_name=self.PriceFactor), factor_names=[self.PriceFactor]).iloc[0]
        Return = Price.iloc[1] / Price.iloc[0] - 1
        for jPortfolioName, jPortfolio in LastPortfolios.items():
            jCovMatrix = LastCovMatrix.loc[jPortfolio.index, jPortfolio.index]
            jStd = np.dot(np.dot(jPortfolio.values, jCovMatrix.values), jPortfolio.values)**0.5
            jReturn = (Return[jPortfolio.index] * jPortfolio).sum()
            self._Output["Z-Score"].loc[idt, jPortfolioName] = jReturn / jStd
            self._Output["Robust Z-Score"].loc[idt, jPortfolioName] = max((-3, min((3, jReturn / jStd))))
            if self._Output["Z-Score"].shape[0]>=self.LookBack:
                self._Output["Bias 统计量"].loc[idt, jPortfolioName] = self._Output["Z-Score"][jPortfolioName].iloc[-self.LookBack:].std()
                self._Output["Robust Bias 统计量"].loc[idt, jPortfolioName] = self._Output["Robust Z-Score"][jPortfolioName].iloc[-self.LookBack:].std()
        AllPortfolioNames = list(LastPortfolios)
        self._Output["Z-Score"] = self._Output["Z-Score"].loc[:, AllPortfolioNames]
        self._Output["Robust Z-Score"] = self._Output["Robust Z-Score"].loc[:, AllPortfolioNames]
        self._Output["Bias 统计量"] = self._Output["Bias 统计量"].loc[:, AllPortfolioNames]
        self._Output["Robust Bias 统计量"] = self._Output["Robust Bias 统计量"].loc[:, AllPortfolioNames]
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        self._Output["汇总统计量"] = pd.DataFrame(index=self._Output["Bias 统计量"].columns) 
        self._Output["汇总统计量"]["RAD 统计量"] = (self._Output["Bias 统计量"] - 1).abs().mean()
        self._Output["汇总统计量"]["Robust RAD 统计量"] = (self._Output["Robust Bias 统计量"] - 1).abs().mean()
        self._Output["Bias 统计量"].insert(0, "95%置信下界", 1-(2/self.LookBack)**0.5)
        self._Output["Bias 统计量"].insert(0, "95%置信上界", 1+(2/self.LookBack)**0.5)
        self._Output["Robust Bias 统计量"].insert(0, "95%置信下界", 1-(2/self.LookBack)**0.5)
        self._Output["Robust Bias 统计量"].insert(0, "95%置信上界", 1+(2/self.LookBack)**0.5)
        Stats = self._Output["Bias 统计量"].iloc[:, 2:]
        SampleNum = pd.notnull(Stats).sum(axis=0)
        self._Output["汇总统计量"]["Bias 统计量高估比例"] = (Stats.T<self._Output["Bias 统计量"]["95%置信下界"]).sum(axis=1) / SampleNum
        self._Output["汇总统计量"]["Bias 统计量低估比例"] = (Stats.T>self._Output["Bias 统计量"]["95%置信上界"]).sum(axis=1) / SampleNum
        self._Output["汇总统计量"]["Bias 统计量准确度"] = 1 - self._Output["汇总统计量"]["Bias 统计量高估比例"]  - self._Output["汇总统计量"]["Bias 统计量低估比例"]
        Stats = self._Output["Robust Bias 统计量"].iloc[:, 2:]
        SampleNum = pd.notnull(Stats).sum(axis=0)
        self._Output["汇总统计量"]["Robust Bias 统计量高估比例"] = (Stats.T<self._Output["Robust Bias 统计量"]["95%置信下界"]).sum(axis=1) / SampleNum
        self._Output["汇总统计量"]["Robust Bias 统计量低估比例"] = (Stats.T>self._Output["Robust Bias 统计量"]["95%置信上界"]).sum(axis=1) / SampleNum
        self._Output["汇总统计量"]["Robust Bias 统计量准确度"] = 1 - self._Output["汇总统计量"]["Robust Bias 统计量高估比例"]  - self._Output["汇总统计量"]["Robust Bias 统计量低估比例"]
        return 0