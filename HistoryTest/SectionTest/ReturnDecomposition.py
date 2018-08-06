# coding=utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm

from QuantStudio import QSArgs, QSError, QSObject
from QuantStudio.FunLib.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.FunLib.DataTypeConversionFun import DummyVarTo01Var
from QuantStudio.FactorTest.FactorTestModel import FactorTestBaseModule

class FamaMacBethRegression(FactorTestBaseModule):
    """Fama-MacBeth 回归"""
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if self.QSEnv.DSs.isEmpty():
            return super().__QS_genSysArgs__(args=args, **kwargs)
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        PriceFactor = searchNameInStrList(DefaultNumFactorList,['价','Price','price'])
        if (args is None) or ("数据源" not in args):
            SysArgs = {"测试因子": [DefaultNumFactorList[0]],
                       "行业因子": "无",
                       "价格因子": PriceFactor,
                       "计算时点": DefaultDS.getDateTime(),
                       "筛选条件": None,
                       "滚动期数": 12,
                       "数据源": DefaultDS.Name}
            ArgInfo = {}
            ArgInfo["测试因子"] = {"type":"MultiOption","range":DefaultNumFactorList,"order":0}
            ArgInfo["行业因子"] = {"type":"SingleOption","range":["无"]+DefaultDS.FactorNames,"order":1}
            ArgInfo["价格因子"] = {"type":"SingleOption","range":DefaultNumFactorList,"order":2}
            ArgInfo["计算时点"] = {"type":"DateList","order":3}
            ArgInfo["筛选条件"] = {"type":"IDFilter","factor_list":DefaultDS.FactorNames,"order":5}
            ArgInfo["滚动期数"] = {"type":"Integer","min":1,"max":120,"order":8,"visible":False}
            ArgInfo["数据源"] = {"type":"SingleOption","range":list(self.QSEnv.DSs.keys()),"refresh":True,"order":6,"visible":False}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        args._QS_MonitorChange = True
        args["数据源"] = DefaultDS.Name
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        if args["行业因子"] not in DefaultDS.FactorNames:
            args["行业因子"] = "无"
        args.ArgInfo["行业因子"]["range"] = ["无"]+DefaultDS.FactorNames
        if args["价格因子"] not in DefaultNumFactorList:
            args["价格因子"] = PriceFactor
        args.ArgInfo["价格因子"]["range"] = DefaultNumFactorList
        if not set(args["测试因子"]).issubset(set(DefaultNumFactorList)):
            args["测试因子"] = [DefaultNumFactorList[0]]
        args.ArgInfo["测试因子"]["range"] = DefaultNumFactorList
        if not set(args["计算时点"]).issubset(set(DefaultDS.getDateTime())):
            args["计算时点"] = DefaultDS.getDateTime()
        if not set(args.ArgInfo["筛选条件"]["factor_list"]).issubset(set(DefaultDS.FactorNames)):
            args["筛选条件"] = None
        args.ArgInfo["筛选条件"]["factor_list"] = DefaultDS.FactorNames
        args._QS_MonitorChange = True
        return args
    def __QS_onSysArgChanged__(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="数据源"):# 数据源发生了变化
            Args["数据源"] = Value
            self.__QS_genSysArgs__(args=Args)
            return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_start__(self):
        self._Output = {"Pure Return":[],"Raw Return":[],"日期":[]}
        self._CurCalcInd = 0
        self._DS = self.QSEnv.DSs[self._SysArgs["数据源"]]
        self._CalcDateTimes = (set(self._SysArgs["计算时点"]) if self._SysArgs["计算时点"]!=[] else None)
        return 0
    def __QS_move__(self, idt):
        if (self._CalcDateTimes is not None) and (idt not in self._CalcDateTimes):
            return 0
        self._CurCalcInd = self._SysArgs["计算时点"][self._CurCalcInd:].index(idt)+self._CurCalcInd
        LastInd = self._CurCalcInd-1
        if LastInd<0:
            nFactor = len(self._SysArgs["测试因子"])
            self._Output["Pure Return"].append((np.nan,)*nFactor)
            self._Output["Raw Return"].append((np.nan,)*nFactor)
            self._Output["日期"].append(idt)
            return self._Output
        LastDate = self._SysArgs["计算时点"][LastInd]
        LastIDs = self._DS.getID(idt=LastDate, is_filtered=True, id_filter_str=self._SysArgs["筛选条件"])
        FactorData = self._DS.getDateTimeData(idt=LastDate,ids=LastIDs,factor_names=self._SysArgs["测试因子"])
        Price = self._DS.getFactorData(dates=[LastDate,self.CurDate],ids=LastIDs,ifactor_name=self._SysArgs["价格因子"])
        Ret = Price.loc[self.CurDate]/Price.loc[LastDate]-1
        # 展开Dummy因子
        if self._SysArgs["行业因子"]!="无":
            DummyFactorData = self._DS.getDateTimeData(idt=LastDate,ids=LastIDs,factor_names=[self._SysArgs["行业因子"]])[self._SysArgs["行业因子"]]
            Mask = pd.notnull(DummyFactorData)
            DummyFactorData = DummyVarTo01Var(DummyFactorData[Mask],ignore_na=True)
            FactorData = pd.merge(FactorData.loc[Mask],DummyFactorData,left_index=True,right_index=True)
        # 回归
        nFactor = len(self._SysArgs["测试因子"])
        yData = Ret[FactorData.index].values
        xData = FactorData.values
        if self._SysArgs["行业因子"]=="无":
            xData = sm.add_constant(xData,prepend=False)
            LastInds = [nFactor]
        else:
            LastInds = [nFactor+i for i in range(xData.shape[1]-nFactor)]
        try:
            Result = sm.OLS(yData,xData,missing="drop").fit()
            self._Output["Pure Return"].append(Result.params[0:nFactor])
        except:
            self._Output["Pure Return"].append(np.zeros(nFactor)+np.nan)
        self._Output["Raw Return"].append(np.zeros(nFactor)+np.nan)
        for i,iFactorName in enumerate(self._SysArgs["测试因子"]):
            iXData = xData[:,[i]+LastInds]
            try:
                Result = sm.OLS(yData,iXData,missing="drop").fit()
                self._Output["Raw Return"][-1][i] = Result.params[0]
            except:
                pass
        self._Output["日期"].append(idt)
        return 0
    def __QS_end__(self):
        self._Output["Pure Return"] = pd.DataFrame(self._Output["Pure Return"],index=self._Output["日期"],columns=self._SysArgs["测试因子"])
        self._Output["Raw Return"] = pd.DataFrame(self._Output["Raw Return"],index=self._Output["日期"],columns=self._SysArgs["测试因子"])
        nDate = self._Output["Raw Return"].shape[0]
        self._Output["滚动t统计量_Pure"] = pd.DataFrame(np.nan,index=self._Output["日期"],columns=self._SysArgs["测试因子"])
        self._Output["滚动t统计量_Raw"] = pd.DataFrame(np.nan,index=self._Output["日期"],columns=self._SysArgs["测试因子"])
        # 计算滚动t统计量
        for i in range(nDate):
            if i<self._SysArgs["滚动期数"]-1:
                continue
            iReturn = self._Output["Pure Return"].iloc[i-self._SysArgs["滚动期数"]+1:i+1,:]
            self._Output["滚动t统计量_Pure"].iloc[i] = iReturn.mean(axis=0)/iReturn.std(axis=0)*pd.notnull(iReturn).sum(axis=0)**0.5
            iReturn = self._Output["Raw Return"].iloc[i-self._SysArgs["滚动期数"]+1:i+1,:]
            self._Output["滚动t统计量_Raw"].iloc[i] = iReturn.mean(axis=0)/iReturn.std(axis=0)*pd.notnull(iReturn).sum(axis=0)**0.5
        nYear = (DateStr2Datetime(self._Output["日期"][-1])-DateStr2Datetime(self._Output["日期"][0])).days/365
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["Pure Return"].columns)
        self._Output["统计数据"]["年化收益率(Pure)"] = ((1+self._Output["Pure Return"]).prod())**(1/nYear)-1
        self._Output["统计数据"]["跟踪误差(Pure)"] = self._Output["Pure Return"].std()*np.sqrt(nDate/nYear)
        self._Output["统计数据"]["信息比率(Pure)"] = self._Output["统计数据"]["年化收益率(Pure)"]/self._Output["统计数据"]["跟踪误差(Pure)"]
        self._Output["统计数据"]["胜率(Pure)"] = (self._Output["Pure Return"]>0).sum()/nDate
        self._Output["统计数据"]["t统计量(Pure)"] = self._Output["Pure Return"].mean()/self._Output["Pure Return"].std()*np.sqrt(nDate)
        self._Output["统计数据"]["年化收益率(Raw)"] = (1+self._Output["Raw Return"]).prod()**(1/nYear)-1
        self._Output["统计数据"]["跟踪误差(Raw)"] = self._Output["Raw Return"].std()*np.sqrt(nDate/nYear)
        self._Output["统计数据"]["信息比率(Raw)"] = self._Output["统计数据"]["年化收益率(Raw)"]/self._Output["统计数据"]["跟踪误差(Raw)"]
        self._Output["统计数据"]["胜率(Raw)"] = (self._Output["Raw Return"]>0).sum()/nDate
        self._Output["统计数据"]["t统计量(Raw)"] = self._Output["Raw Return"].mean()/self._Output["Raw Return"].std()*np.sqrt(nDate)
        self._Output["统计数据"]["年化收益率(Pure-Naive)"] = (1+self._Output["Pure Return"]-self._Output["Raw Return"]).prod()**(1/nYear)-1
        self._Output["统计数据"]["跟踪误差(Pure-Naive)"] = (self._Output["Pure Return"]-self._Output["Raw Return"]).std()*np.sqrt(nDate/nYear)
        self._Output["统计数据"]["信息比率(Pure-Naive)"] = self._Output["统计数据"]["年化收益率(Pure-Naive)"]/self._Output["统计数据"]["跟踪误差(Pure-Naive)"]
        self._Output["统计数据"]["胜率(Pure-Naive)"] = (self._Output["Pure Return"]-self._Output["Raw Return"]>0).sum()/nDate
        self._Output["统计数据"]["t统计量(Pure-Naive)"] = (self._Output["Pure Return"]-self._Output["Raw Return"]).mean()/(self._Output["Pure Return"]-self._Output["Raw Return"]).std()*np.sqrt(nDate)
        self._Output.pop("日期")
        self._DS, self._CalcDateTimes = None, None
        return 0