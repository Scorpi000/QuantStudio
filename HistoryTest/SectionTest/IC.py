# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Str, Instance, Dict, on_trait_change
from traitsui.api import SetEditor, Item
import statsmodels.api as sm

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData
from QuantStudio.Tools.ExcelFun import copyChart
from QuantStudio.FactorDataBase.FactorDB import FactorTable
from QuantStudio.HistoryTest.HistoryTestModel import BaseModule

class IC(BaseModule):
    """IC"""
    TestFactors = ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=())
    FactorOrder = Dict(key_trait=Str(), value_trait=Enum("降序", "升序"), arg_type="ArgDict", label="排序方向", order=1)
    PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=2)
    IndustryFactor = Enum("无", arg_type="SingleOption", label="行业因子", order=3)
    WeightFactor = Enum("等权", arg_type="SingleOption", label="权重因子", order=4)
    CalcDTs = List(dt.datetime, arg_type="DateList", label="计算时点", order=5)
    LookBack = Int(1, arg_type="Integer", label="回溯期数", order=6)
    CorrMethod = Enum("spearman", "pearson", "kendall", arg_type="SingleOption", label="相关性算法", order=7)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=8)
    RollAvgPeriod = Int(12, arg_type="Integer", label="滚动平均期数", order=9)
    _FactorTable = Instance(FactorTable, allow_none=False)
    def __init__(self, factor_table, sys_args={}, **kwargs):
        self._FactorTable = factor_table
        super().__init__(name="IC", sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FactorTable.getFactorMetaData(key="DataType")))
        self.add_trait("TestFactors", ListStr(arg_type="MultiOption", label="测试因子", order=0, option_range=tuple(DefaultNumFactorList)))
        self.TestFactors.append(DefaultNumFactorList[0])
        self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=2))
        self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
        self.add_trait("IndustryFactor", Enum(*(["无"]+DefaultStrFactorList), arg_type="SingleOption", label="行业因子", order=3))
        self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=4))
    @on_trait_change("TestFactors[]")
    def _on_TestFactors_changed(self, obj, name, old, new):
        self.FactorOrder = {iFactorName:self.FactorOrder.get(iFactorName, "降序") for iFactorName in self.TestFactors}
    def getViewItems(self, context_name=""):
        Items = super().getViewItems(context_name=context_name)
        Items[0].editor = SetEditor(values=self.trait("TestFactors").option_range)
        return Items
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        self._Output = {}
        self._Output["IC"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["股票数"] = {iFactorName:[] for iFactorName in self.TestFactors}
        self._Output["时点"] = []
        self._CurCalcInd = 0
        return (self._FactorTable, )
    def __QS_move__(self, idt):
        if self.CalcDTs:
            if idt not in self.CalcDTs[self._CurCalcInd:]: return 0
            self._CurCalcInd = self.CalcDTs[self._CurCalcInd:].index(idt) + self._CurCalcInd
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self.CalcDTs[PreInd]
            LastDateTime = self.CalcDTs[LastInd]
        else:
            self._CurCalcInd = self._Model.DateTimeIndex
            PreInd = self._CurCalcInd - self.LookBack
            LastInd = self._CurCalcInd - 1
            PreDateTime = self._Model.DateTimeSeries[PreInd]
            LastDateTime = self._Model.DateTimeSeries[LastInd]
        if (PreInd<0) or (LastInd<0):
            for iFactorName in self.TestFactors:
                self._Output["IC"][iFactorName].append(np.nan)
                self._Output["股票数"][iFactorName].append(np.nan)
            self._Output["时点"].append(idt)
            return 0
        PreIDs = self._FactorTable.getFilteredID(idt=PreDateTime, id_filter_str=self.IDFilter)
        FactorExpose = self._FactorTable.readData(dts=[PreDateTime], ids=PreIDs, factor_names=list(self.TestFactors)).iloc[:, 0, :]
        CurPrice = self._FactorTable.readData(dts=[idt], ids=PreIDs, factor_names=[self.PriceFactor]).iloc[0, 0, :]
        LastPrice = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.PriceFactor]).iloc[0, 0, :]
        Ret = CurPrice/LastPrice-1
        if self.IndustryFactor!="无":# 进行收益率的行业调整
            IndustryData = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.IndustryFactor]).iloc[0, 0, :]
            AllIndustry = IndustryData.unique()
            if self.WeightFactor=="等权":
                for iIndustry in AllIndustry:
                    iMask = (IndustryData==iIndustry)
                    Ret[iMask] -= Ret[iMask].mean()
            else:
                WeightData = self._FactorTable.readData(dts=[LastDateTime], ids=PreIDs, factor_names=[self.WeightFactor]).iloc[0, 0, :]
                for iIndustry in AllIndustry:
                    iMask = (IndustryData==iIndustry)
                    iWeight = WeightData[iMask]
                    iRet = Ret[iMask]
                    Ret[iMask] -= (iRet*iWeight).sum() / iWeight[pd.notnull(iWeight) & pd.notnull(iRet)].sum(skipna=False)
        for iFactorName in self.TestFactors:
            self._Output["IC"][iFactorName].append(FactorExpose[iFactorName].corr(Ret, method=self.CorrMethod))
            self._Output["股票数"][iFactorName].append(pd.notnull(FactorExpose[iFactorName]).sum())
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        CalcDateTimes = self._Output.pop("时点")
        self._Output["股票数"] = pd.DataFrame(self._Output["股票数"], index=CalcDateTimes)
        self._Output["IC"] = pd.DataFrame(self._Output["IC"], index=CalcDateTimes)
        for i, iFactorName in enumerate(self.TestFactors):
            if self.FactorOrder[iFactorName]=="升序":
                self._Output["IC"][iFactorName] = -self._Output["IC"][iFactorName]
        self._Output["IC的移动平均"] = self._Output["IC"].copy()
        nDate = len(CalcDateTimes)
        for i in range(nDate):
            if i<self.RollAvgPeriod-1:
                self._Output["IC的移动平均"].iloc[i,:] = np.nan
            else:
                self._Output["IC的移动平均"].iloc[i,:] = self._Output["IC"].iloc[i-self.RollAvgPeriod+1:i+1, :].mean()
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["IC"].columns)
        self._Output["统计数据"]["平均值"] = self._Output["IC"].mean()
        self._Output["统计数据"]["标准差"] = self._Output["IC"].std()
        self._Output["统计数据"]["最小值"] = self._Output["IC"].min()
        self._Output["统计数据"]["最大值"] = self._Output["IC"].max()
        self._Output["统计数据"]["IC_IR"] = self._Output["统计数据"]["平均值"]/self._Output["统计数据"]["标准差"]
        self._Output["统计数据"]["t统计量"] = np.nan
        self._Output["统计数据"]["平均股票数"] = self._Output["股票数"].mean()
        self._Output["统计数据"]["IC×Sqrt(N)"] = self._Output["统计数据"]["平均值"]*np.sqrt(self._Output["统计数据"]["平均股票数"])
        self._Output["统计数据"]["有效期数"] = 0.0
        for iFactor in self._Output["IC"]:
            self._Output["统计数据"].loc[iFactor,"有效期数"] = pd.notnull(self._Output["IC"][iFactor]).sum()
        self._Output["统计数据"]["t统计量"] = (self._Output["统计数据"]["有效期数"]**0.5)*self._Output["统计数据"]["IC_IR"]
        return 0
    def genExcelReport(self, xl_book, sheet_name):
        xl_book.sheets["IC"].api.Copy(Before=xl_book.sheets[0].api)
        xl_book.sheets[0].name = sheet_name
        CurSheet = xl_book.sheets[sheet_name]
        nDate = self._Output["IC"].shape[0]
        # 写入统计数据
        CurSheet[1,0].expand().clear_contents()
        CurSheet[1,1].value = self._Output["统计数据"].values
        CurSheet[1,0].options(transpose=True).value = list(self._Output["统计数据"].index)
        Table = CurSheet.api.ListObjects(1)
        Table.Resize(CurSheet[0:self._Output["统计数据"].shape[0]+1,0:self._Output["统计数据"].shape[1]+1].api)
        # 写入日期序列
        CurSheet[0,11].expand().clear_contents()
        Dates = [iDate[:4]+"-"+iDate[4:6]+"-"+iDate[6:] for iDate in self._Output["IC"].index]
        CurSheet[0,11].value = "时点"
        CurSheet[1,11].options(transpose=True).value = Dates
        # 写入时间序列数据
        FormatFun1 = np.vectorize(lambda x:("%.2f%%" % x) if pd.notnull(x) else None)
        FormatFun2 = np.vectorize(lambda x:("%u" % x) if pd.notnull(x) else None)
        for i,iFactor in enumerate(self._Output["统计数据"].index):
            iCol = 13+i*3
            CurSheet[0,iCol-1].value = iFactor+"-IC"
            CurSheet[1,iCol-1].value = FormatFun1(self._Output["IC"][[iFactor]].values*100)
            CurSheet[0,iCol].value = iFactor+"-IC的移动平均"
            CurSheet[1,iCol].value = FormatFun1(self._Output["IC的移动平均"][[iFactor]].values*100)
            CurSheet[0,iCol+1].value = iFactor+"-股票数"
            CurSheet[1,iCol+1].value = FormatFun2(self._Output["股票数"][[iFactor]].values)
            # 绘制图线
            Chrt = copyChart(xl_book,sheet_name,"IC",6,iCol-1,sheet_name,iFactor+"-IC").api[1]
            Chrt.SeriesCollection(1).Values = CurSheet[1:nDate+1, iCol-1].api
            Chrt.SeriesCollection(1).Name = "='"+sheet_name+"'!R1C"+str(iCol)
            Chrt.SeriesCollection(2).Values = CurSheet[1:nDate+1, iCol].api
            Chrt.SeriesCollection(2).Name = "='"+sheet_name+"'!R1C"+str(iCol+1)
            Chrt.SeriesCollection(1).XValues = CurSheet[1:nDate+1, 11].api
            Chrt = copyChart(xl_book,sheet_name,"股票数",23,iCol-1,sheet_name,iFactor+"-股票数").api[1]
            Chrt.SeriesCollection(1).Values = CurSheet[1:nDate+1, iCol+1].api
            Chrt.SeriesCollection(1).Name = "='"+sheet_name+"'!R1C"+str(iCol+2)
            Chrt.SeriesCollection(1).XValues = CurSheet[1:nDate+1, 11].api
            Chrt.ChartTitle.Text = iFactor+"-股票数"
        CurSheet.charts["IC"].delete()
        CurSheet.charts["股票数"].delete()
        return 0

class RiskAdjustedIC(IC):    
    """风险调整的 IC"""
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if self.QSEnv.DSs.isEmpty():
            return super().__QS_genSysArgs__(args=args, **kwargs)
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        PriceFactor = searchNameInStrList(DefaultNumFactorList,['价','Price','price'])
        if (args is None) or ("数据源" not in args):
            SysArgs = {"测试因子": pd.DataFrame([("降序",)], index=[DefaultNumFactorList[0]], columns=["排序方向"], dtype="O"),
                       "价格因子": PriceFactor,
                       "风险因子": [DefaultNumFactorList[-1]],
                       "行业因子": "无",
                       "计算时点": DefaultDS.getDateTime(),
                       "回溯期数": 1,
                       "相关性算法": "spearman",
                       "筛选条件": None,
                       "滚动平均期数": 12,
                       "数据源": DefaultDS.Name}
            ArgInfo = {}
            ArgInfo["测试因子"] = {"type":"ArgFrame", "order":0, "row":DefaultNumFactorList, "row_removable":True,
                                  "colarg_info":{"排序方向":{"type":"SingleOption","range":["降序","升序"]}},
                                  "unchecked_value":{"排序方向":"降序"}}
            ArgInfo["价格因子"] = {"type":"SingleOption","range":DefaultNumFactorList,"order":1}
            ArgInfo["风险因子"] = {"type":"MultiOption","range":DefaultNumFactorList,"order":2}
            ArgInfo["行业因子"] = {"type":"SingleOption","range":["无"]+DefaultDS.FactorNames,"order":3}
            ArgInfo["计算时点"] = {"type":"DateList","order":4}
            ArgInfo["回溯期数"] = {"type":"Integer","min":1,"max":120,"order":5}
            ArgInfo["相关性算法"] = {"type":"SingleOption","range":["spearman","pearson","kendall"],"order":6}
            ArgInfo["筛选条件"] = {"type":"IDFilter","factor_list":DefaultDS.FactorNames,"order":7}
            ArgInfo["滚动平均期数"] = {"type":"Integer","min":1,"max":120,"order":8,"visible":False}
            ArgInfo["数据源"] = {"type":"SingleOption","range":list(self.QSEnv.DSs.keys()),"order":9,"refresh":True,"visible":False}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        args._QS_MonitorChange = False
        args["数据源"] = DefaultDS.Name
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        if args["行业因子"] not in DefaultDS.FactorNames:
            args["行业因子"] = "无"
        args.ArgInfo["行业因子"]["range"] = ["无"]+DefaultDS.FactorNames
        if args["价格因子"] not in DefaultNumFactorList:
            args["价格因子"] = PriceFactor
        args.ArgInfo["价格因子"]["range"] = DefaultNumFactorList
        if not set(args["测试因子"].index).issubset(set(DefaultNumFactorList)):
            args["测试因子"] = pd.DataFrame([("降序",)], index=[DefaultNumFactorList[0]], columns=["排序方向"], dtype="O")
        args.ArgInfo["测试因子"]["row"] = DefaultNumFactorList
        if not set(args["风险因子"]).issubset(set(DefaultNumFactorList)):
            args["风险因子"] = [DefaultNumFactorList[-1]]
        args.ArgInfo["风险因子"]["range"] = DefaultNumFactorList
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
        return super(FactorTestBaseModule, self).__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_move__(self, idt):
        if (self._CalcDateTimes is not None) and (idt not in self._CalcDateTimes):
            return 0
        self._CurCalcInd = self._SysArgs["计算时点"][self._CurCalcInd:].searchsorted(idt)+self._CurCalcInd
        PreInd = self._CurCalcInd-self._SysArgs["回溯期数"]
        LastInd = self._CurCalcInd-1
        if (PreInd<0) or (LastInd<0):
            return self._moveNone(idt)
        PreDateTime = self._SysArgs["计算时点"][PreInd]
        LastDateTime = self._SysArgs["计算时点"][LastInd]
        PreIDs = self._DS.getID(idt=PreDateTime, is_filtered=True, id_filter_str=self._SysArgs["筛选条件"])
        FactorExpose = self._DS.getDateTimeData(idt=PreDateTime, ids=PreIDs, factor_names=list(self._SysArgs["测试因子"].index))
        if self._SysArgs["风险因子"]!=[]:
            RiskExpose = self._DS.getDateTimeData(idt=PreDateTime, ids=PreIDs, factor_names=self._SysArgs["风险因子"])
            RiskExpose["constant"] = 1.0
        else:
            RiskExpose = pd.DataFrame(1.0,index=PreIDs,columns=["constant"])
        CurPrice = self._DS.getFactorData(dts=[idt], ids=PreIDs, ifactor_name=self._SysArgs["价格因子"]).iloc[0]
        LastPrice = self._DS.getFactorData(dts=[LastDateTime], ids=PreIDs, ifactor_name=self._SysArgs["价格因子"]).iloc[0]
        Ret = CurPrice/LastPrice-1
        Mask = (pd.isnull(RiskExpose).sum(axis=1)==0)
        # 展开Dummy因子
        if self._SysArgs["行业因子"]!="无":
            DummyFactorData = self._DS.getDateTimeData(idt=PreDateTime,ids=PreIDs,factor_names=[self._SysArgs["行业因子"]])[self._SysArgs["行业因子"]]
            _,_,_,DummyFactorData = prepareRegressData(np.ones(DummyFactorData.shape[0]),dummy_data=DummyFactorData.values)
        iMask = (pd.notnull(Ret) & Mask)
        Ret = Ret[iMask]
        iX = RiskExpose.loc[iMask].values
        if self._SysArgs["行业因子"]!="无":
            iDummy = DummyFactorData[iMask.values]
            iDummy = iDummy[:,(np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
            iX = np.hstack((iX,iDummy[:,:-1]))
        try:
            Result = sm.OLS(Ret.values, iX, missing="drop").fit()
        except:
            return self._moveNone(idt)
        RiskAdjustedRet = pd.Series(Result.resid, index=Ret.index)
        for iFactorName in self._SysArgs["测试因子"].index:
            iFactorExpose = FactorExpose[iFactorName]
            iMask = (Mask & pd.notnull(iFactorExpose))
            iFactorExpose = iFactorExpose[iMask]
            iX = RiskExpose.loc[iMask].values
            if self._SysArgs["行业因子"]!="无":
                iDummy = DummyFactorData[iMask.values]
                iDummy = iDummy[:,(np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
                iX = np.hstack((iX,iDummy[:,:-1]))
            try:
                Result = sm.OLS(iFactorExpose.values,iX,missing="drop").fit()
            except:
                self._Output["IC"][iFactorName].append(np.nan)
                self._Output["股票数"][iFactorName].append(0)
                continue
            iFactorExpose = pd.Series(Result.resid,index=iFactorExpose.index)
            self._Output["IC"][iFactorName].append(iFactorExpose.corr(RiskAdjustedRet,method=self._SysArgs["相关性算法"]))
            self._Output["股票数"][iFactorName].append(pd.notnull(iFactorExpose).sum())
        self._Output["时点"].append(idt)
        return 0

class ICDecay(BaseModule):
    """IC 衰减"""
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if self.QSEnv.DSs.isEmpty():
            return super().__QS_genSysArgs__(args=args, **kwargs)
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        PriceFactor = searchNameInStrList(DefaultNumFactorList,['价','Price','price'])
        if (args is None) or ("数据源" not in args):
            SysArgs = {"测试因子":DefaultNumFactorList[0],
                       "排序方向":"降序",
                       "价格因子":PriceFactor,
                       "行业因子":"无",
                       "权重因子":"等权",
                       "计算时点":DefaultDS.getDateTime(),
                       "回溯期数":list(np.arange(1,13)),
                       "相关性算法":"spearman",
                       "筛选条件":None,
                       "数据源":DefaultDS.Name}
            ArgInfo = {}
            ArgInfo["测试因子"] = {"type":"SingleOption","range":DefaultNumFactorList,"order":0}
            ArgInfo["排序方向"] = {"type":"SingleOption","range":["降序","升序"],"order":1}
            ArgInfo["价格因子"] = {"type":"SingleOption","range":DefaultNumFactorList,"order":2}
            ArgInfo["行业因子"] = {"type":"SingleOption","range":DefaultDS.FactorNames+["无"],"order":3}
            ArgInfo["权重因子"] = {"type":"SingleOption","range":DefaultNumFactorList+["等权"],"order":4}
            ArgInfo["回溯期数"] = {"type":"MultiOption","range":list(np.arange(1,25)),"order":5}
            ArgInfo["计算时点"] = {"type":"DateList","order":6}
            ArgInfo["相关性算法"] = {"type":"SingleOption","range":["spearman","pearson","kendall"],"order":7}
            ArgInfo["筛选条件"] = {"type":"IDFilter","factor_list":DefaultDS.FactorNames,"order":8}
            ArgInfo["数据源"] = {"type":"SingleOption","range":list(self.QSEnv.DSs.keys()),"refresh":True,"order":9,"visible":False}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        args._QS_MonitorChange = False
        args["数据源"] = DefaultDS.Name
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        if args["权重因子"] not in DefaultNumFactorList:
            args["权重因子"] = "等权"
        args.ArgInfo["权重因子"]["range"] = DefaultNumFactorList+["等权"]
        if args["行业因子"] not in DefaultDS.FactorNames:
            args["行业因子"] = "无"
        args.ArgInfo["行业因子"]["range"] = DefaultDS.FactorNames+["无"]
        if args["价格因子"] not in DefaultNumFactorList:
            args["价格因子"] = PriceFactor
        args.ArgInfo["价格因子"]["range"] = DefaultNumFactorList
        if args["测试因子"] not in DefaultNumFactorList:
            args["测试因子"] = DefaultNumFactorList[0]
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
    def __QS_onDSChanged__(self, change_type, change_info, **kwargs):
        return _QS_onDSChanged(self, change_type, change_info, **kwargs)
    def __QS_start__(self):
        self._Output = {"IC":[[] for i in args["回溯期数"]]}
        self._Output["时点"] = []
        self._CalcDateTimes = (set(self._SysArgs["计算时点"]) if self._SysArgs["计算时点"]!=[] else None)
        self._CurCalcInd = 0
        self._DS = self.QSEnv.DSs[self._SysArgs["数据源"]]
        return 0
    def __QS_move__(self, idt):
        if (self._CalcDateTimes is not None) and (idt not in self._CalcDateTimes):
            return 0
        self._CurCalcInd = self._SysArgs["计算时点"][self._CurCalcInd:].searchsorted(idt)+self._CurCalcInd
        LastInd = self._CurCalcInd-1
        if (LastInd<0):
            for i,iRollBack in enumerate(self._SysArgs["回溯期数"]):
                self._Output["IC"][i].append(np.nan)
            self._Output["时点"].append(idt)
            return self._Output
        LastDateTime = self._SysArgs["计算时点"][LastInd]
        Price = self._DS.getFactorData(dts=[LastDateTime, idt],ids=None,ifactor_name=self._SysArgs["价格因子"])
        Ret = Price.loc[idt]/Price.loc[LastDateTime]-1
        for i,iRollBack in enumerate(self._SysArgs["回溯期数"]):
            iPreInd = self._CurCalcInd-iRollBack
            if iPreInd<0:
                self._Output["IC"][i].append(np.nan)
                continue
            iPreDate = self._SysArgs["计算时点"][iPreInd]
            iPreIDs = self._DS.getID(idt=iPreDate, is_filtered=True, id_filter_str=self._SysArgs["筛选条件"])
            iRet = Ret.loc[iPreIDs].copy()
            if self._SysArgs["行业因子"]!="无":
                IndustryData = self._DS.getFactorData(dts=[iPreDate],ids=iPreIDs,ifactor_name=self._SysArgs["行业因子"]).loc[iPreDate]
                AllIndustry = IndustryData.unique()
                # 进行收益率的行业调整
                if self._SysArgs["权重因子"]=="等权":
                    for iIndustry in AllIndustry:
                        iRet[IndustryData==iIndustry] -= iRet[IndustryData==iIndustry].mean()
                else:
                    WeightData = self._DS.getFactorData(idt=iPreDate,ids=iPreIDs,ifactor_name=self._SysArgs["权重因子"]).loc[iPreDate]
                    for iIndustry in AllIndustry:
                        iWeight = WeightData[IndustryData==iIndustry]
                        iiRet = iRet[IndustryData==iIndustry]
                        iRet[IndustryData==iIndustry] -= (iiRet*iWeight).sum()/iWeight[pd.notnull(iWeight) & pd.notnull(iiRet)].sum(skipna=False)
            iFactorExpose = self._DS.getFactorData(dts=[iPreDate],ids=iPreIDs,ifactor_name=self._SysArgs["测试因子"]).loc[iPreDate]
            self._Output["IC"][i].append(iFactorExpose.corr(iRet,method=self._SysArgs["相关性算法"]))
        self._Output["时点"].append(idt)
        return 0
    def __QS_end__(self):
        self._Output["IC"] = pd.DataFrame(np.array(self._Output["IC"]).T,index=self._Output.pop("时点"),columns=args["回溯期数"])
        if args["排序方向"] == "升序":
            self._Output["IC"] = -self._Output["IC"]
        self._Output["统计数据"] = pd.DataFrame(index=self._Output["IC"].columns)
        self._Output["统计数据"]["IC平均值"] = self._Output["IC"].mean()
        nDates = pd.notnull(self._Output["IC"]).sum()
        self._Output["统计数据"]["标准差"] = self._Output["IC"].std()
        self._Output["统计数据"]["风险调整的IC"] = self._Output["统计数据"]["IC平均值"]/self._Output["统计数据"]["标准差"]
        self._Output["统计数据"]["胜率"] = (self._Output["IC"]>0).sum()/nDates
        self._Output["统计数据"]["t统计量"] = self._Output["统计数据"]["风险调整的IC"]*nDates**0.5
        self._DS, self._CalcDateTimes = None, None
        return 0
    def genICDecayStdReport(self, xl_book, sheet_name):
            xl_book.sheets["IC的衰减"].api.Copy(Before=xl_book.sheets[0].api)
            xl_book.sheets[0].name = sheet_name
            CurSheet = xl_book.sheets[sheet_name]
            nDate = self._Output["IC"].shape[0]
            # 写入统计数据
            CurSheet[1,0].expand().clear_contents()
            CurSheet[1,1].value = self._Output["统计数据"].values
            CurSheet[1,0].options(transpose=True).value = list(self._Output["统计数据"].index)
            Table = CurSheet.api.ListObjects(1)
            Table.Resize(CurSheet[0:self._Output["统计数据"].shape[0]+1,0:self._Output["统计数据"].shape[1]+1].api)
            Chrt = CurSheet.charts["IC的衰减"].api[1]
            Chrt.SeriesCollection(1).Values = CurSheet[1:self._Output["统计数据"].shape[0]+1,1].api
            Chrt.SeriesCollection(1).Name = args["测试因子"]+"-IC均值"
            Chrt.SeriesCollection(2).Values = CurSheet[1:self._Output["统计数据"].shape[0]+1,4].api
            Chrt.SeriesCollection(2).Name = args["测试因子"]+"-胜率"
            Chrt.SeriesCollection(1).XValues = CurSheet[1:self._Output["统计数据"].shape[0]+1,0].api
            # 写入日期序列
            CurSheet[0,7].expand().clear_contents()
            Dates = [iDate[:4]+"-"+iDate[4:6]+"-"+iDate[6:] for iDate in self._Output["IC"].index]
            CurSheet[0,7].value = "时点"
            CurSheet[1,7].options(transpose=True).value = Dates
            # 写入时间序列数据
            CurSheet[0,8].value = list(self._Output["IC"].columns)
            CurSheet[1,8].value = np.vectorize(lambda x:("%.2f%%" % x) if pd.notnull(x) else None)(self._Output["IC"].values*100)
            return 0