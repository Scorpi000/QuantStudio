# coding=utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm

from QuantStudio.QSEnvironment import QSArgs, QSError, QSObject
from QuantStudio.FunLib.AuxiliaryFun import getFactorList, genAvailableName
from QuantStudio.FunLib.ExcelFun import copyChart
from QuantStudio.FactorTest.FactorTestModel import FactorTestBaseModule

class SectionCorrelation(FactorTestBaseModule):
    """因子截面相关性"""
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if self.QSEnv.DSs.isEmpty():
            return super().__QS_genSysArgs__(args=args, **kwargs)
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        if (args is None) or ("数据源" not in args):
            SysArgs = {"测试因子": pd.DataFrame([("降序",)], index=[DefaultNumFactorList[0]], columns=["排序方向"], dtype="O"),
                       "相关性算法": ["spearman"],
                       "计算时点": DefaultDS.getDateTime(),
                       "风险数据源": None,
                       "筛选条件": None,
                       "数据源": DefaultDS.Name}
            ArgInfo = {}
            ArgInfo["测试因子"] = {"type":"ArgFrame", "order":0, "row":DefaultNumFactorList, "row_removable":True,
                                  "colarg_info":{"排序方向":{"type":"SingleOption","range":["降序","升序"]}},
                                  "unchecked_value":{"排序方向":"降序"}}
            ArgInfo["相关性算法"] = {"type":"MultiOption","range":["spearman","pearson","kendall"],"order":1}
            ArgInfo["计算时点"] = {"type":"DateList","order":2}
            ArgInfo["风险数据源"] = {"type":"RiskDataSource","order":3,"refresh":True}
            ArgInfo["筛选条件"] = {"type":"IDFilter","factor_list":DefaultDS.FactorNames,"order":4}
            ArgInfo["数据源"] = {"type":"SingleOption","range":list(self.QSEnv.DSs.keys()),"order":5,"refresh":True,"visible":False}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        args._QS_MonitorChange = False
        args["数据源"] = DefaultDS.Name
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        if not set(args["测试因子"].index).issubset(set(DefaultNumFactorList)):
            args["测试因子"] = pd.DataFrame([("降序",)], index=[DefaultNumFactorList[0]], columns=["排序方向"], dtype="O")
        args.ArgInfo["测试因子"]["row"] = DefaultNumFactorList
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
            elif Key=="风险数据源":# 风险数据源发生了变化
                if Value is not None:
                    Args.ArgInfo["相关性算法"]["range"] = ["spearman","pearson","kendall","factor-score correlation","factor-portfolio correlation"]
                else:
                    Args.ArgInfo["相关性算法"]["range"] = ["spearman","pearson","kendall"]
                    Args["相关性算法"] = list(set(Args["相关性算法"]).intersection(set(Args.ArgInfo["相关性算法"]["range"])))
                Args["风险数据源"] = Value
                return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_start__(self):
        self._Output = {"FactorPair":[]}
        for i,iFactor in enumerate(self._SysArgs["测试因子"].index):
            for j,jFactor in enumerate(self._SysArgs["测试因子"].index):
                if j>i:
                    self._Output["FactorPair"].append(iFactor+"-"+jFactor)
        nPair = len(self._Output["FactorPair"])
        self._Output.update({iMethod:[[] for i in range(nPair)] for iMethod in self._SysArgs["相关性算法"]})
        self._CorrMatrixNeeded = (("factor-score correlation" in self._SysArgs["相关性算法"]) or ("factor-portfolio correlation" in self._SysArgs["相关性算法"]))
        if (self._SysArgs["风险数据源"] is not None) and self._CorrMatrixNeeded:
            self._SysArgs["风险数据源"].start()
        self._Output["日期"] = []
        self._CalcDateTimes = (set(self._SysArgs["计算时点"]) if self._SysArgs["计算时点"]!=[] else None)
        self._CurCalcInd = 0
        return 0
    def __QS_move__(self, idt):
        if (self._CalcDateTimes is not None) and (idt not in self._CalcDateTimes):
            return 0
        self._Output["日期"].append(idt)
        IDs = self.QSEnv.DSs[self._SysArgs["数据源"]].getID(idt=idt, is_filtered=True, id_filter_str=self._SysArgs["筛选条件"])
        FactorExpose = self.QSEnv.DSs[self._SysArgs["数据源"]].getDateTimeData(idt=idt,ids=IDs,factor_names=list(self._SysArgs["测试因子"].index)).astype("float")
        if self._CorrMatrixNeeded and (self._SysArgs["风险数据源"] is not None):
            self._SysArgs["风险数据源"].move(idt)
            CovMatrix = self._SysArgs["风险数据源"].getDateCov(idt, ids=IDs, drop_na=True)
            FactorIDs = {}
        else:
            CovMatrix = None
        PairInd = 0
        for i,iFactor in enumerate(self._SysArgs["测试因子"].index):
            iFactorExpose = FactorExpose[iFactor]
            if self._CorrMatrixNeeded:
                iIDs = FactorIDs.get(iFactor)
                if iIDs is None:
                    if CovMatrix is not None:
                        FactorIDs[iFactor] = list(set(CovMatrix.index).intersection(set(iFactorExpose[pd.notnull(iFactorExpose)].index)))
                    else:
                        FactorIDs[iFactor] = list(iFactorExpose[pd.notnull(iFactorExpose)].index)
                    iIDs = FactorIDs[iFactor]
            for j,jFactor in enumerate(self._SysArgs["测试因子"].index):
                if j>i:
                    jFactorExpose = FactorExpose[jFactor]
                    if self._CorrMatrixNeeded:
                        jIDs = FactorIDs.get(jFactor)
                        if jIDs is None:
                            if CovMatrix is not None:
                                FactorIDs[jFactor] = list(set(CovMatrix.index).intersection(set(jFactorExpose[pd.notnull(jFactorExpose)].index)))
                            else:
                                FactorIDs[jFactor] = list(jFactorExpose[pd.notnull(jFactorExpose)].index)
                            jIDs = FactorIDs[jFactor]
                        IDs = list(set(iIDs).intersection(set(jIDs)))
                        iTempExpose = iFactorExpose.loc[IDs].values
                        jTempExpose = jFactorExpose.loc[IDs].values
                        if CovMatrix is not None:
                            TempCovMatrix = CovMatrix.loc[IDs,IDs].values
                        else:
                            nID = len(IDs)
                            TempCovMatrix = np.eye(nID,nID)
                    for kMethod in self._SysArgs["相关性算法"]:
                        if kMethod=="factor-score correlation":
                            ijCov = np.dot(iTempExpose.T,np.dot(TempCovMatrix,jTempExpose))
                            iStd = np.sqrt(np.dot(iTempExpose.T,np.dot(TempCovMatrix,iTempExpose)))
                            jStd = np.sqrt(np.dot(jTempExpose.T,np.dot(TempCovMatrix,jTempExpose)))
                            self._Output[kMethod][PairInd].append(ijCov/iStd/jStd)
                        elif kMethod=="factor-portfolio correlation":
                            TempCovMatrixInv = np.linalg.inv(TempCovMatrix)
                            ijCov = np.dot(iTempExpose.T,np.dot(TempCovMatrixInv,jTempExpose))
                            iStd = np.sqrt(np.dot(iTempExpose.T,np.dot(TempCovMatrixInv,iTempExpose)))
                            jStd = np.sqrt(np.dot(jTempExpose.T,np.dot(TempCovMatrixInv,jTempExpose)))
                            self._Output[kMethod][PairInd].append(ijCov/iStd/jStd)
                        else:
                            self._Output[kMethod][PairInd].append(FactorExpose[iFactor].corr(FactorExpose[jFactor],method=kMethod))
                    PairInd += 1
        return 0
    def __QS_end__(self):
        for iMethod in self._SysArgs["相关性算法"]:
            self._Output[iMethod] = pd.DataFrame(np.array(self._Output[iMethod]).T,columns=self._Output.pop("FactorPair"),index=self._Output.pop("日期"))
            iAvgName = iMethod+"均值"
            self._Output[iAvgName] = pd.DataFrame(index=self._SysArgs["测试因子"].index,columns=self._SysArgs["测试因子"].index,dtype="float")
            for i,iFactor in enumerate(self._SysArgs["测试因子"].index):
                for j,jFactor in enumerate(self._SysArgs["测试因子"].index):
                    if j>i:
                        if self._SysArgs["测试因子"].loc[iFactor,"排序方向"]!=self._SysArgs["测试因子"].loc[jFactor,"排序方向"]:
                            self._Output[iMethod][iFactor+"-"+jFactor] = -self._Output[iMethod][iFactor+"-"+jFactor]
                        self._Output[iAvgName].loc[iFactor,jFactor] = self._Output[iMethod][iFactor+"-"+jFactor].mean()
                    elif j<i:
                        self._Output[iAvgName].loc[iFactor,jFactor] = self._Output[iAvgName].loc[jFactor,iFactor]
                    else:
                        self._Output[iAvgName].loc[iFactor,jFactor] = 1
        if (self._SysArgs["风险数据源"] is not None) and self._CorrMatrixNeeded:
            self._SysArgs["风险数据源"].end()
        self._CalcDateTimes = None
        return 0
    def __QS_genExcelReport__(self, xl_book, sheet_name):
        xl_book.sheets.add(name=sheet_name)
        CurSheet = xl_book.sheets[sheet_name]
        nFactor = self._SysArgs["测试因子"].shape[0]
        for j,jMethod in enumerate(self._SysArgs["相关性算法"]):
            jData = self._Output[jMethod+"均值"]
            # 写入统计数据
            CurSheet[(nFactor+1)*j,0].value = jData
            CurSheet[(nFactor+1)*j,0].value = jMethod+"相关性"
            # 形成热图
            CurSheet[(nFactor+1)*j+1:(nFactor+1)*j+1+nFactor,1:1+nFactor].select()
            iFormatConditions = xl_book.app.selection.api.FormatConditions
            iFormatConditions.AddColorScale(2)
            iFormatConditions(iFormatConditions.Count).SetFirstPriority()
            iFormatConditions(1).ColorScaleCriteria(1).Type = 1
            iFormatConditions(1).ColorScaleCriteria(1).FormatColor.Color = 16776444
            iFormatConditions(1).ColorScaleCriteria(1).FormatColor.TintAndShade = 0
            iFormatConditions(1).ColorScaleCriteria(2).Type = 2
            iFormatConditions(1).ColorScaleCriteria(2).FormatColor.Color = 7039480
            iFormatConditions(1).ColorScaleCriteria(2).FormatColor.TintAndShade = 0
        return 0

class FactorTurnover(FactorTestBaseModule):
    """因子换手率"""
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if self.QSEnv.DSs.isEmpty():
            return super().__QS_genSysArgs__(args=args, **kwargs)
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        if (args is None) or ("数据源" not in args):
            SysArgs = {"数据源":DefaultDS.Name,
                       "测试因子":[DefaultNumFactorList[0]],
                       "计算时点":DefaultDS.getDateTime(),
                       "筛选条件":None}
            ArgInfo = {"测试因子":{"type":"MultiOption","range":DefaultNumFactorList,"order":0},
                       "计算时点":{"type":"DateList","order":1},
                       "筛选条件":{"type":"IDFilter","factor_list":DefaultDS.FactorNames,"order":2},
                       "数据源":{"type":"SingleOption","range":list(self.DSs.keys()),"refresh":True,"order":3,"visible":False}}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        args._QS_MonitorChange = False
        args["数据源"] = DefaultDS.Name
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        if not set(Args["测试因子"]).issubset(set(DefaultDS.FactorNames)):
            Args["测试因子"] = pd.DataFrame([("降序",)], index=[DefaultNumFactorList[0]], columns=["排序方向"], dtype="O")
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
        if change_type=="set":
            if (Key=="数据源") and (Value!=Args["数据源"]):# 数据源发生了变化
                Args["数据源"] = Value
                self.__QS_genSysArgs__(args=Args, **kwargs)
                return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_start__(self):
        self._Output = {iFactorName:[] for iFactorName in self.SysArgs["测试因子"]}
        self._Output["日期"] = []
        self._CalcDateTimes = (set(self._SysArgs["计算时点"]) if self._SysArgs["计算时点"]!=[] else None)
        self._CurCalcInd = 0
        self._DS = self.QSEnv.DSs[self._SysArgs["数据源"]]
        return 0
    def __QS_move__(self, idt):
        if (self._CalcDateTimes is not None) and (idt not in self._CalcDateTimes):
            return 0
        self._CurCalcInd = self._SysArgs["计算时点"][self._CurCalcInd:].searchsorted(idt)+self._CurCalcInd
        PreInd = self._CurCalcInd-1
        if (PreInd<0):
            for iFactorName in self.SysArgs["测试因子"]:
                self._Output[iFactorName].append(0.0)
            self._Output["日期"].append(idt)
            return self._Output
        PreDate = self._SysArgs["计算时点"][PreInd]
        PreIDs = self._DS.getID(idt=PreDate, is_filtered=True, id_filter_str=self._SysArgs["筛选条件"])
        PreFactorExpose = self._DS.getDateTimeData(idt=PreDate,ids=PreIDs,factor_names=self._SysArgs["测试因子"]).astype("float")
        CurFactorExpose = self._DS.getDateTimeData(idt=idt,ids=PreIDs,factor_names=self._SysArgs["测试因子"]).astype("float")
        for iFactorName in self._SysArgs["测试因子"]:
            self._Output[iFactorName].append(CurFactorExpose[iFactorName].corr(PreFactorExpose[iFactorName]))
        self._Output["日期"].append(idt)
        return 0
    def __QS_end__(self):
        Dates = self._Output.pop("日期")
        self._Output = {"因子换手率":pd.DataFrame(self._Output, index=Dates)}
        self._Output["平均换手率"] = pd.DataFrame(self._Output["因子换手率"].mean(), columns=["平均换手率"])
        self._DS, self._CalcDateTimes = None, None
        return self._Output
    def __QS_genExcelReport__(self, xl_book, sheet_name):
        xl_book.sheets["因子换手率"].api.Copy(Before=xl_book.sheets[0].api)
        xl_book.sheets[0].name = sheet_name
        CurSheet = xl_book.sheets[sheet_name]
        nDate = self._Output["因子换手率"].shape[0]
        # 写入统计数据
        CurSheet[1,0].expand().clear_contents()
        CurSheet[1,0].options(transpose=True).value = list(self._Output["平均换手率"].index)
        FormatFun = np.vectorize(lambda x:("%.2f%%" % x) if pd.notnull(x) else None)
        CurSheet[1,1].value = FormatFun(self._Output["平均换手率"].values*100)
        Table = CurSheet.api.ListObjects(1)
        Table.Resize(CurSheet[0:self._Output["平均换手率"].shape[0]+1,0:self._Output["平均换手率"].shape[1]+1].api)
        # 写入日期序列
        CurSheet[0,3].expand().clear_contents()
        Dates = [iDate[:4]+"-"+iDate[4:6]+"-"+iDate[6:] for iDate in self._Output["因子换手率"].index]
        CurSheet[0,3].value = "日期"
        CurSheet[1,3].options(transpose=True).value = Dates
        # 写入时间序列数据
        for i,iFactor in enumerate(self._Output["平均换手率"].index):
            iCol = 5+i
            CurSheet[0,iCol-1].value = iFactor+"-截面相关性"
            CurSheet[1,iCol-1].value = FormatFun(self._Output["因子换手率"][[iFactor]].values*100)
            # 绘制图线
            Chrt = copyChart(xl_book,sheet_name,"因子换手率",6,iCol-1,sheet_name,iFactor+"-因子换手率").api[1]
            Chrt.SeriesCollection(1).Values = CurSheet[1:nDate+1,iCol-1].api
            Chrt.SeriesCollection(1).Name = iFactor+"-截面相关性"
            Chrt.SeriesCollection(1).XValues = CurSheet[1:nDate+1,3].api
        CurSheet.charts["因子换手率"].delete()
        return 0