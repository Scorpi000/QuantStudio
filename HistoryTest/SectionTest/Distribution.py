# coding=utf-8

import numpy as np
import pandas as pd

from QuantStudio.QSEnvironment import QSArgs,QSError,QSObject
from QuantStudio.FunLib.AuxiliaryFun import getFactorList, genAvailableName
from QuantStudio.FunLib.ExcelFun import copyChart
from QuantStudio.FactorTest.FactorTestModel import FactorTestBaseModule

class IndustryDistribution(FactorTestBaseModule):
    """因子值行业分布"""
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if self.QSEnv.DSs.isEmpty():
            return super().__QS_genSysArgs__(args=args, **kwargs)
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        if (args is None) or ("数据源" not in args):
            SysArgs = {"测试因子": [DefaultNumFactorList[0]],
                       "行业因子": searchNameInStrList(DefaultStrFactorList,["行业","Ind","ind"]),
                       "阈值": "中位数",
                       "计算时点": DefaultDS.getDateTime(),
                       "筛选条件": None,
                       "数据源": DefaultDS.Name}
            ArgInfo = {}
            ArgInfo["测试因子"] = {"type":"MultiOption","range":DefaultNumFactorList,"order":0}
            ArgInfo["行业因子"] = {"type":"SingleOption","range":DefaultDS.FactorNames,"order":1}
            ArgInfo["阈值"] = {"type":"SingleOption","range":["中位数","平均数","25%分位数","75%分位数"],"order":2}
            ArgInfo["计算时点"] = {"type":"DateList","order":3}
            ArgInfo["筛选条件"] = {"type":"IDFilter","factor_list":DefaultDS.FactorNames,"order":4}
            ArgInfo["数据源"] = {"type":"SingleOption","range":list(self.QSEnv.DSs.keys()),"order":5,"refresh":True,"visible":False}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        args._QS_MonitorChange = False
        args["数据源"] = DefaultDS.Name
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        if not set(args["测试因子"].index).issubset(set(DefaultNumFactorList)):
            args["测试因子"] = [DefaultNumFactorList[0]]
        args.ArgInfo["测试因子"]["range"] = DefaultNumFactorList
        if args["行业因子"] not in FactorNames:
            args["行业因子"] = searchNameInStrList(DefaultStrFactorList,["行业","Ind","ind"])
        args.ArgInfo["行业因子"]["range"] = DefaultDS.FactorNames
        if not set(args["计算时点"]).issubset(set(DefaultDS.getDateTime())):
            args["计算时点"] = DefaultDS.getDateTime()
        if not set(args.ArgInfo["筛选条件"]["factor_list"]).issubset(set(DefaultDS.FactorNames)):
            args["筛选条件"] = None
        args.ArgInfo["筛选条件"]["factor_list"] = DefaultDS.FactorNames
        args._QS_MonitorChange = True
        return args
    def __QS_onSysArgChanged__(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if change_type=="set":
            if (Key=="数据源") and (Value!=Args["数据源"]):# 数据源发生了变化
                Args["数据源"] = Value
                self.__QS_genSysArgs__(args=Args, **kwargs)
                return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_start__(self):
        self._DS = self.DSs[self._SysArgs["数据源"]]
        AllIndustries = np.array(self._DS.getFactorUniqueData(ifactor_name=self._SysArgs["行业因子"]))
        Mask = pd.isnull(AllIndustries)
        if np.sum(Mask)>0:
            AllIndustries = list(AllIndustries[~Mask])+[None]
        self._Output = {iFactorName:{iIndustry:[] for iIndustry in AllIndustries} for iFactorName in self._SysArgs["测试因子"]}
        self._Output["历史平均值"] = {iFactorName:[] for iFactorName in self._SysArgs["测试因子"]}
        self._Output["历史标准差"] = {iFactorName:[] for iFactorName in self._SysArgs["测试因子"]}
        self._Output["行业分类"] = AllIndustries
        self._Output["日期"] = []
        self._CalcDateTimes = (set(self._SysArgs["计算时点"]) if self._SysArgs["计算时点"]!=[] else None)
        return 0
    def __QS_move__(self, idt):
        if (self._CalcDateTimes is not None) and (idt not in self._CalcDateTimes):
            return 0
        self._Output["日期"].append(idt)
        IDs = self._DS.getID(idt=idt, is_filtered=True, id_filter_str=self._SysArgs["筛选条件"])
        FactorExpose = self._DS.getDateTimeData(idt=idt,ids=IDs,factor_names=self._SysArgs["测试因子"])
        IndustryData = self._DS.getFactorData(dates=idt,ids=IDs,ifactor_name=self._SysArgs["行业因子"])
        IndustryData = IndustryData.loc[idt]
        Threshold = {}
        Mask = {}
        for iFactorName in self._SysArgs["测试因子"]:
            Mask[iFactorName] = pd.notnull(FactorExpose[iFactorName])
            if self._SysArgs["阈值"]=="中位数":
                Threshold[iFactorName] = FactorExpose[iFactorName].median()
            elif self._SysArgs["阈值"]=="平均值":
                Threshold[iFactorName] = FactorExpose[iFactorName].mean()
            elif self._SysArgs["阈值"]=="25%分位数":
                Threshold[iFactorName] = FactorExpose[iFactorName].quantile(0.25)
            elif self._SysArgs["阈值"]=="75%分位数":
                Threshold[iFactorName] = FactorExpose[iFactorName].quantile(0.75)
        for jIndustry in self._Output["行业分类"]:
            if pd.isnull(jIndustry):
                jMask = pd.isnull(IndustryData)
            else:
                jMask = (IndustryData==jIndustry)
            for iFactorName in self._SysArgs["测试因子"]:
                ijMask = (jMask & Mask[iFactorName])
                ijNum = ijMask.sum()
                if ijNum!=0:
                    self._Output[iFactorName][jIndustry].append((FactorExpose[iFactorName][ijMask]>=Threshold[iFactorName]).sum()/ijNum)
                else:
                    self._Output[iFactorName][jIndustry].append(np.nan)
        return 0
    def __QS_end__(self):
        for iFactorName in self._SysArgs["测试因子"]:
            self._Output[iFactorName] = pd.DataFrame(self._Output[iFactorName],index=pd.Index(self._Output["日期"]),columns=self._Output["行业分类"])
            self._Output["历史平均值"][iFactorName] = self._Output[iFactorName].mean()
            self._Output["历史标准差"][iFactorName] = self._Output[iFactorName].std()
        self._Output["历史平均值"] = pd.DataFrame(self._Output["历史平均值"],columns=self._SysArgs["测试因子"])
        self._Output["历史标准差"] = pd.DataFrame(self._Output["历史标准差"],columns=self._SysArgs["测试因子"])
        self._Output.pop("行业分类")
        self._Output.pop("日期")
        self._DS, self._CalcDateTimes = None, None
        return 0
    def __QS_genExcelReport__(self, xl_book, sheet_name):
        xl_book.sheets["因子值行业分布"].api.Copy(Before=xl_book.sheets[0].api)
        xl_book.sheets[0].name = sheet_name
        CurSheet = xlBook.sheets[sheet_name]
        nIndustry,nFactor = self._Output["历史平均值"].shape
        # 写入数据
        FormatFun = np.vectorize(lambda x:("%.2f%%" % x) if pd.notnull(x) else None)
        CurSheet[1,0].options(transpose=True).value = list(self._Output["历史平均值"].index)
        CurSheet[0,1].value = list(self._Output["历史平均值"].columns)
        CurSheet[1,1].value = FormatFun(self._Output["历史平均值"].values*100)
        CurSheet[0,nFactor+1].value = list(self._Output["历史标准差"].columns)
        CurSheet[1,nFactor+1].value = FormatFun(self._Output["历史标准差"].values*100)
        CurSheet[0,0].value = "行业"
        # 绘制图线
        StartRow = 4
        for j,jFactor in enumerate(self._Output["历史平均值"].columns):
            Chrt = ExcelFun.copyChart(xl_book,sheet_name,"行业分布",StartRow-1,1,sheet_name,jFactor+"-行业分布")
            ChrtArea = Chrt.api[1]
            ChrtArea.SeriesCollection(1).Values = CurSheet[1:nIndustry+1,j+1]
            ChrtArea.SeriesCollection(1).Name = jFactor+"-均值"
            ChrtArea.SeriesCollection(2).Values = CurSheet[1:nIndustry+1,j+1+nFactor]
            ChrtArea.SeriesCollection(2).Name = jFactor+"-标准差"
            ChrtArea.SeriesCollection(1).XValues = CurSheet[1:nIndustry+1,0]
            StartRow += 17
        CurSheet.charts["行业分布"].delete()
        return 0