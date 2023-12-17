# coding=utf-8
"""Wind 量化研究数据库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Int, Str, List, ListStr, Dict, Callable, File

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.QSObjects import QSSQLObject
from QuantStudio import __QS_Error__, __QS_LibPath__, __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB
from QuantStudio.FactorDataBase.FDBFun import updateInfo, adjustDateTime, SQL_Table, SQL_WideTable, SQL_FeatureTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable, SQL_ConstituentTable
from QuantStudio.Tools.api import Panel

def RollBackNPeriod(report_date, n_period):
    Date = report_date
    for i in range(1, n_period+1):
        if Date[-4:]=='1231':
            Date = Date[0:4]+'0930'
        elif Date[-4:]=='0930':
            Date = Date[0:4]+'0630'
        elif Date[-4:]=='0630':
            Date = Date[0:4]+'0331'
        elif Date[-4:]=='0331':
            Date = str(int(Date[0:4])-1)+'1231'
    return Date
# 查找某个报告期对应的公告期
def findNoteDate(report_date, report_note_dates):
    for i in range(0, report_note_dates.shape[0]):
        if report_date==report_note_dates['报告期'].iloc[i]: return report_note_dates['公告日期'].iloc[i]
    return None
# 生成报告期-公告日期 SQL 查询语句
def genANN_ReportSQLStr(table_prefix, ids, report_period="1231"):
    DBTableName = table_prefix+"AShareIssuingDatePredict"
    # 提取财报的公告期数据, ID, 公告日期, 报告期
    SQLStr = "SELECT "+DBTableName+".s_info_windcode, "
    SQLStr += DBTableName+".s_stm_actual_issuingdate, "
    SQLStr += DBTableName+".report_period "
    SQLStr += "FROM "+DBTableName+" "
    SQLStr += "WHERE ("+genSQLInCondition(DBTableName+".s_info_windcode", ids, is_str=True, max_num=1000)+") "
    if report_period is not None:
        SQLStr += "AND "+DBTableName+".report_period LIKE '%"+report_period+"' "
    SQLStr += "ORDER BY "+DBTableName+".s_info_windcode, "
    SQLStr += DBTableName+".s_stm_actual_issuingdate, "+DBTableName+".report_period"
    return SQLStr
def _prepareReportANNRawData(fdb, ids):
    SQLStr = genANN_ReportSQLStr(fdb.TablePrefix, ids, report_period="1231")
    RawData = fdb.fetchall(SQLStr)
    if not RawData: return pd.DataFrame(columns=["ID", "公告日期", "报告期"])
    else: return pd.DataFrame(np.array(RawData), columns=["ID", "公告日期", "报告期"])
def _saveRawDataWithReportANN(ft, report_ann_file, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock):
    isANNReport = raw_data._QS_ANNReport
    if isANNReport:
        PID = sorted(pid_lock)[0]
        ANN_ReportFilePath = raw_data_dir+os.sep+PID+os.sep+report_ann_file+("."+ft._ANN_ReportFileSuffix if ft._ANN_ReportFileSuffix else "")
        pid_lock[PID].acquire()
        if not os.path.isfile(ANN_ReportFilePath):# 没有报告期-公告日期数据, 提取该数据
            with pd.HDFStore(ANN_ReportFilePath) as ANN_ReportFile: pass
            pid_lock[PID].release()
            IDs = []
            for iPID in sorted(pid_ids): IDs.extend(pid_ids[iPID])
            RawData = _prepareReportANNRawData(ft.FactorDB, ids=IDs)
            super(type(ft), ft).__QS_saveRawData__(RawData, [], raw_data_dir, pid_ids, report_ann_file, pid_lock)
        else:
            pid_lock[PID].release()
    raw_data = raw_data.set_index(['ID'])
    CommonCols = list(raw_data.columns.difference(set(factor_names)))
    AllIDs = set(raw_data.index)
    for iPID, iIDs in pid_ids.items():
        with pd.HDFStore(raw_data_dir+os.sep+iPID+os.sep+file_name+ft._QSArgs.OperationMode._FileSuffix) as iFile:
            iInterIDs = sorted(AllIDs.intersection(set(iIDs)))
            iData = raw_data.loc[iInterIDs]
            for jFactorName in factor_names:
                ijData = iData[CommonCols+[jFactorName]].reset_index()
                if isANNReport: ijData.columns.name = raw_data_dir+os.sep+iPID+os.sep+report_ann_file
                iFile[jFactorName] = ijData
            iFile["_QS_IDs"] = pd.Series(iIDs)
    return 0

# f: 该算子所属的因子对象或因子表对象
# idt: 当前所处的时点
# iid: 当前待计算的 ID
# x: 当期的数据, 分析师评级时为: DataFrame(columns=["日期", ...]), 分析师盈利预测时为: [DataFrame(columns=["日期", "报告期", "预测基准股本", ...])], list的长度为向前年数
# args: 参数, {参数名:参数值}
def _DefaultOperator(f, idt, iid, x, args):
    return np.nan


class _AnalystConsensusTable(SQL_Table):
    """分析师汇总表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        CalcType = Enum("FY0", "FY1", "FY2", "Fwd12M", label="计算方法", arg_type="SingleOption", order=0, option_range=["FY0", "FY1", "FY2", "Fwd12M"])
        Period = Enum("263001000", "263002000", "263003000", "263004000", label="周期", arg_type="SingleOption", order=1, option_range=["263001000", "263002000", "263003000", "263004000"])
        LookBack = Int(180, arg_type="Integer", label="回溯天数", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._PeriodField = FactorInfo[FactorInfo["FieldType"]=="Period"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = 'W2财务年报-公告日期'
        self._ANN_ReportFileSuffix = "h5"
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        CalcType = args.get("计算方法", self._QSArgs.CalcType)
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self._QSArgs.LookBack))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._DateField, self._ReportDateField, self._PeriodField]+factor_names]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成SQL语句, 日期, ID, 报告期, 数据
        SQLStr = 'SELECT '+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._ReportDateField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._PeriodField]+"='"+args.get("周期", self._QSArgs.Period)+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += 'ORDER BY '+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self._DateField]+", "+DBTableName+"."+FieldDict[self._ReportDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=['日期','ID','报告期']+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData), columns=['日期','ID','报告期']+factor_names)
        RawData._QS_ANNReport = (CalcType!="Fwd12M")
        return RawData
    def __QS_saveRawData__(self, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, **kwargs):
        return _saveRawDataWithReportANN(self, self._ANN_ReportFileName, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock)
    def __QS_genGroupInfo__(self, factors, operation_mode):
        PeriodGroup = {}
        for iFactor in factors:
            iPeriod = iFactor.Period
            if iPeriod not in PeriodGroup:
                PeriodGroup[iPeriod] = {"FactorNames":[iFactor.Name], 
                                        "RawFactorNames":{iFactor._NameInFT}, 
                                        "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                        "args":{"周期":iPeriod, "计算方法":iFactor.CalcType, "回溯天数":iFactor.LookBack}}
            else:
                PeriodGroup[iPeriod]["FactorNames"].append(iFactor.Name)
                PeriodGroup[iPeriod]["RawFactorNames"].add(iFactor._NameInFT)
                PeriodGroup[iPeriod]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], PeriodGroup[iPeriod]["StartDT"])
                if iFactor.CalcType!="Fwd12M": PeriodGroup[iPeriod]["args"]["计算方法"] = iFactor.CalcType
                PeriodGroup[iPeriod]["args"]["回溯天数"] = max(PeriodGroup[iPeriod]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iPeriod in PeriodGroup:
            StartInd = operation_mode.DTRuler.index(PeriodGroup[iPeriod]["StartDT"])
            Groups.append((self, PeriodGroup[iPeriod]["FactorNames"], list(PeriodGroup[iPeriod]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], PeriodGroup[iPeriod]["args"]))
        return Groups
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return Panel(np.nan, items=factor_names, major_axis=dts, minor_axis=ids)
        Dates = sorted({iDT.strftime("%Y%m%d") for iDT in dts})
        CalcType, LookBack = args.get("计算方法", self._QSArgs.CalcType), args.get("回溯天数", self._QSArgs.LookBack)
        if CalcType=="Fwd12M":
            CalcFun, FYNum, ANNReportData = self._calcIDData_Fwd12M, None, None
        else:
            CalcFun, FYNum = self._calcIDData_FY, int(CalcType[-1])
            ANNReportPath = raw_data.columns.name+("."+self._ANN_ReportFileSuffix if self._ANN_ReportFileSuffix else "")
            if (ANNReportPath is not None) and os.path.isfile(ANNReportPath):
                with pd.HDFStore(ANNReportPath, mode="r") as ANN_ReportFile:
                    ANNReportData = ANN_ReportFile["RawData"]
            else:
                ANNReportData = _prepareReportANNRawData(self._FactorDB, ids)
            ANNReportData = ANNReportData.set_index(["ID"])
        raw_data = raw_data.set_index(["ID"])
        Data = {}
        for iID in raw_data.index.unique():
            if ANNReportData is not None:
                if iID in ANNReportData.index:
                    iANNReportData = ANNReportData.loc[[iID]]
                else:
                    continue
            else:
                iANNReportData = None
            Data[iID] = CalcFun(Dates, raw_data.loc[[iID]], iANNReportData, factor_names, LookBack, FYNum)
        Data = Panel(Data, major_axis=Dates, minor_axis=factor_names)
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Dates]
        Data = Data.swapaxes(0, 2)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(set(dts)).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]
    def _calcIDData_FY(self, date_seq, raw_data, report_ann_data, factor_names, lookback, fy_num):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1
        tempLen = raw_data.shape[0]
        NoteDate = None
        for i, iDate in enumerate(date_seq):
            while (tempInd<tempLen-1) and (iDate>=raw_data['日期'].iloc[tempInd+1]): tempInd = tempInd+1
            if tempInd<0: continue
            LastYear = str(int(iDate[0:4])-1)
            NoteDate = findNoteDate(LastYear+'1231', report_ann_data)
            if (NoteDate is None) or (NoteDate>iDate):
                ObjectDate = str(int(LastYear)+fy_num)+'1231'
            else:
                ObjectDate = str(int(iDate[0:4])+fy_num)+'1231'
            iDate = dt.date(int(iDate[0:4]), int(iDate[4:6]), int(iDate[6:])) 
            for j in range(0, tempInd+1):
                if raw_data['报告期'].iloc[tempInd-j]==ObjectDate:
                    FYNoteDate = raw_data['日期'].iloc[tempInd-j]
                    if (iDate - dt.date(int(FYNoteDate[0:4]), int(FYNoteDate[4:6]), int(FYNoteDate[6:]))).days<=lookback:
                        StdData[i] = raw_data[factor_names].iloc[tempInd-j].values
                        break
        return StdData
    def _calcIDData_Fwd12M(self, date_seq, raw_data, report_ann_data, factor_names, lookback, fy_num):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1
        tempLen = raw_data.shape[0]
        NoteDate = None
        for i, iDate in enumerate(date_seq):
            while (tempInd<tempLen-1) and (iDate>=raw_data['日期'].iloc[tempInd+1]): tempInd = tempInd+1
            if tempInd<0: continue
            ObjectDate1 = iDate[0:4]+'1231'
            ObjectDate2 = str(int(iDate[0:4])+1)+'1231'
            ObjectData1 = None
            ObjectData2 = None
            iDate = dt.date(int(iDate[0:4]), int(iDate[4:6]), int(iDate[6:]))
            for j in range(0, tempInd+1):
                if (ObjectData1 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectDate1):
                    NoteDate = raw_data['日期'].iloc[tempInd-j]
                    if (iDate-dt.date(int(NoteDate[0:4]), int(NoteDate[4:6]), int(NoteDate[6:]))).days<=lookback:
                        ObjectData1 = raw_data[factor_names].iloc[tempInd-j].values
                if (ObjectData2 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectDate2):
                    NoteDate = raw_data['日期'].iloc[tempInd-j]
                    if (iDate-dt.date(int(NoteDate[0:4]), int(NoteDate[4:6]), int(NoteDate[6:]))).days<=lookback:
                        ObjectData2 = raw_data[factor_names].iloc[tempInd-j].values
                if (ObjectData1 is not None) and (ObjectData2 is not None):
                    break
            if (ObjectData1 is not None) and (ObjectData2 is not None):
                Weight1 = (dt.date(int(ObjectDate1[0:4]), 12, 31) - iDate).days
                if (iDate.month==2) and (iDate.day==29): Weight1 = Weight1/366
                else:
                    Weight1 = Weight1/(dt.date(iDate.year+1, iDate.month, iDate.day)-iDate).days
                StdData[i] = Weight1*ObjectData1.astype("float") + (1-Weight1)*ObjectData2.astype("float")
        return StdData
class _AnalystRatingDetailTable(SQL_Table):
    """分析师投资评级明细表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        Operator = Callable(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
        AdditionalFields = ListStr(arg_type="ListStr", label="附加字段", order=2)
        Deduplication = ListStr(arg_type="ListStr", label="去重字段", order=3)
        Period = Int(180, arg_type="Integer", label="周期", order=4)
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=5, option_range=["double", "string", "object"])
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.Deduplication = [self._Owner._InstituteField]
    
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._InstituteField = FactorInfo[FactorInfo["FieldType"]=="Institute"].index[0]
        self._AnalystField = FactorInfo[FactorInfo["FieldType"]=="Analyst"].index[0]
        self._TempData = {}
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_genGroupInfo__(self, factors, operation_mode):
        FactorNames, RawFactorNames, StartDT = [], set(), dt.datetime.now()
        Args = {"附加字段": set(), "去重字段": set(), "周期":0}
        for iFactor in factors:
            FactorNames.append(iFactor.Name)
            RawFactorNames.add(iFactor._NameInFT)
            Args["附加字段"] = Args["附加字段"].union(set(iFactor.AdditionalFields))
            Args["去重字段"] = Args["去重字段"].union(set(iFactor.Deduplication))
            Args["周期"] = max(Args["周期"], iFactor.Period)
            StartDT = min(operation_mode._FactorStartDT[iFactor.Name], StartDT)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        StartInd = operation_mode.DTRuler.index(StartDT)
        return [(self, FactorNames, list(RawFactorNames), operation_mode.DTRuler[StartInd:EndInd+1], Args)]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("周期", self._QSArgs.Period))
        AdditiveFields = args.get("附加字段", self._QSArgs.AdditionalFields)
        DeduplicationFields = args.get("去重字段", self._QSArgs.Deduplication)
        AllFields = list(set(factor_names+AdditiveFields+DeduplicationFields))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._DateField]+AllFields]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成SQL语句, 日期, ID, 其他字段
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in AllFields: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += 'ORDER BY '+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self._DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID"]+AllFields)
        else: RawData = pd.DataFrame(np.array(RawData), columns=["日期", "ID"]+AllFields)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return Panel(np.nan, items=factor_names, major_axis=dts, minor_axis=ids)
        Dates = sorted({dt.datetime.combine(iDT.date(), dt.time(0)) for iDT in dts})
        DeduplicationFields = args.get("去重字段", self._QSArgs.Deduplication)
        AdditionalFields = list(set(args.get("附加字段", self._QSArgs.AdditionalFields)+DeduplicationFields))
        AllFields = list(set(factor_names+AdditionalFields))
        raw_data = raw_data.loc[:, ["日期", "ID"]+AllFields].set_index(["ID"])
        Period = args.get("周期", self._QSArgs.Period)
        ModelArgs = args.get("参数", self._QSArgs.ModelArgs)
        Operator = args.get("算子", self._QSArgs.Operator)
        DataType = args.get("数据类型", self._QSArgs.DataType)
        AllIDs = set(raw_data.index)
        Data = {}
        for kFactorName in factor_names:
            if DataType=="double": kData = np.full(shape=(len(Dates), len(ids)), fill_value=np.nan)
            else: kData = np.full(shape=(len(Dates), len(ids)), fill_value=None, dtype="O")
            kFields = ["日期", kFactorName]+AdditionalFields
            for j, jID in enumerate(ids):
                if jID not in AllIDs: continue
                jRawData = raw_data.loc[[jID]][kFields]
                for i, iDate in enumerate(Dates):
                    iStartDate = (iDate - dt.timedelta(Period)).strftime("%Y%m%d")
                    ijRawData = jRawData[(jRawData["日期"]<=iDate.strftime("%Y%m%d")) & (jRawData["日期"]>iStartDate)]
                    if DeduplicationFields:
                        ijTemp = ijRawData.groupby(by=DeduplicationFields)[["日期"]].max()
                        ijTemp = ijTemp.reset_index()
                        ijRawData = pd.merge(ijTemp, ijRawData, how='left', left_on=DeduplicationFields+["日期"], right_on=DeduplicationFields+["日期"])
                    kData[i, j] = Operator(self, iDate, jID, ijRawData, ModelArgs)
            Data[kFactorName] = kData
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)

class _AnalystEstDetailTable(SQL_Table):
    """分析师盈利预测明细表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        Operator = Callable(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
        ForwardYears = List(default=[0], label="向前年数", arg_type="ArgList", order=2)
        AdditionalFields = ListStr(arg_type="ListStr", label="附加字段", order=3)
        Deduplication = ListStr(arg_type="ListStr", label="去重字段", order=4)
        Period = Int(180, arg_type="Integer", label="周期", order=5)
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=6, option_range=["double", "string", "object"])
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.Deduplication = [self._Owner._InstituteField]
    
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._InstituteField = FactorInfo[FactorInfo["FieldType"]=="Institute"].index[0]
        self._AnalystField = FactorInfo[FactorInfo["FieldType"]=="Analyst"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._CapitalField = FactorInfo[FactorInfo["FieldType"]=="Capital"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = 'W2财务年报-公告日期'
        self._ANN_ReportFileSuffix = "h5"
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_genGroupInfo__(self, factors, operation_mode):
        FactorNames, RawFactorNames, StartDT = [], set(), dt.datetime.now()
        Args = {"附加字段": set(), "去重字段": set(), "周期":0}
        for iFactor in factors:
            FactorNames.append(iFactor.Name)
            RawFactorNames.add(iFactor._NameInFT)
            Args["附加字段"] = Args["附加字段"].union(set(iFactor.AdditionalFields))
            Args["去重字段"] = Args["去重字段"].union(set(iFactor.Deduplication))
            Args["周期"] = max(Args["周期"], iFactor.Period)
            StartDT = min(operation_mode._FactorStartDT[iFactor.Name], StartDT)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        StartInd = operation_mode.DTRuler.index(StartDT)
        Args["附加字段"], Args["去重字段"] = list(Args["附加字段"]), list(Args["去重字段"])
        return [(self, FactorNames, list(RawFactorNames), operation_mode.DTRuler[StartInd:EndInd+1], Args)]
    def __QS_saveRawData__(self, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, **kwargs):
        return _saveRawDataWithReportANN(self, self._ANN_ReportFileName, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("周期", self._QSArgs.Period))
        AdditiveFields = args.get("附加字段", self._QSArgs.AdditionalFields)
        DeduplicationFields = args.get("去重字段", self._QSArgs.Deduplication)
        AllFields = list(set(factor_names+AdditiveFields+DeduplicationFields))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._DateField, self._ReportDateField, self._CapitalField]+AllFields]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成SQL语句, 日期, ID, 报告期, 研究机构名称, 分析师名称, 预测基准股本(万股), 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._ReportDateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._CapitalField]+", "
        for iField in AllFields: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self._DateField]+", "+DBTableName+"."+FieldDict[self._ReportDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID", self._ReportDateField, self._CapitalField]+AllFields)
        else: RawData = pd.DataFrame(np.array(RawData), columns=["日期", "ID", self._ReportDateField, self._CapitalField]+AllFields)
        RawData._QS_ANNReport = True
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Dates = sorted({dt.datetime.combine(iDT.date(), dt.time(0)) for iDT in dts})
        DeduplicationFields = args.get("去重字段", self._QSArgs.Deduplication)
        AdditionalFields = list(set(args.get("附加字段", self._QSArgs.AdditionalFields)+DeduplicationFields))
        AllFields = list(set(factor_names+AdditionalFields))
        ANNReportPath = raw_data.columns.name+("."+self._ANN_ReportFileSuffix if self._ANN_ReportFileSuffix else "")
        raw_data = raw_data.loc[:, ["日期", "ID", self._ReportDateField, self._CapitalField]+AllFields].set_index(["ID"])
        Period = args.get("周期", self._QSArgs.Period)
        ForwardYears = args.get("向前年数", self._QSArgs.ForwardYears)
        ModelArgs = args.get("参数", self._QSArgs.ModelArgs)
        Operator = args.get("算子", self._QSArgs.Operator)
        DataType = args.get("数据类型", self._QSArgs.DataType)
        if (ANNReportPath is not None) and os.path.isfile(ANNReportPath):
            with pd.HDFStore(ANNReportPath, mode="r") as ANN_ReportFile:
                ANNReportData = ANN_ReportFile["RawData"]
        else:
            ANNReportData = _prepareReportANNRawData(self._FactorDB, ids)
        ANNReportData = ANNReportData.set_index(["ID"])
        raw_data[self._CapitalField] = raw_data[self._CapitalField].astype("float")
        AllIDs = set(raw_data.index)
        Data = {}
        for kFactorName in factor_names:
            if DataType=="double": kData = np.full(shape=(len(Dates), len(ids)), fill_value=np.nan)
            else: kData = np.full(shape=(len(Dates), len(ids)), fill_value=None, dtype="O")
            kFields = ["日期", self._ReportDateField, self._CapitalField, kFactorName]+AdditionalFields
            for j, jID in enumerate(ids):
                if jID not in AllIDs:
                    x = [pd.DataFrame(columns=kFields)]*len(ForwardYears)
                    for i, iDate in enumerate(Dates):
                        kData[i, j] = Operator(self, iDate, jID, x, ModelArgs)
                    continue
                if jID in ANNReportData.index:
                    jReportNoteDate = ANNReportData.loc[[jID]].reset_index()
                else:
                    jReportNoteDate = pd.DataFrame(columns=ANNReportData.columns)
                jRawData = raw_data.loc[[jID], kFields]
                ijNoteDate = None
                for i, iDate in enumerate(Dates):
                    iStartDate = (iDate - dt.timedelta(Period)).strftime("%Y%m%d")
                    ijRawData = jRawData[(jRawData["日期"]<=iDate.strftime("%Y%m%d")) & (jRawData["日期"]>iStartDate)]
                    iLastYear = str(iDate.year-1)
                    ijNoteDate = findNoteDate(iLastYear+"1231", jReportNoteDate)
                    x = []
                    for iiNFY in ForwardYears:
                        if (ijNoteDate is None) or ((ijNoteDate is not None) and (ijNoteDate>iDate.strftime("%Y%m%d"))):
                            ObjectDate = str(int(iLastYear)+iiNFY)+"1231"
                        else:
                            ObjectDate = str(iDate.year+iiNFY)+"1231"
                        iijRawData = ijRawData[ijRawData[self._ReportDateField]==ObjectDate].copy()
                        if iijRawData.shape[0]==0: x.append(iijRawData)
                        else:
                            if DeduplicationFields:
                                ijTemp = iijRawData.groupby(by=DeduplicationFields)[["日期"]].max()
                                ijTemp = ijTemp.reset_index()
                                iijRawData = pd.merge(ijTemp, iijRawData, how="left", left_on=DeduplicationFields+["日期"], right_on=DeduplicationFields+["日期"])
                            x.append(iijRawData)
                    kData[i, j] = Operator(self, iDate, jID, x, ModelArgs)
            Data[kFactorName] = kData
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)

class _WideTable(SQL_WideTable):
    """WindDB2 宽因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _NarrowTable(SQL_NarrowTable):
    """WindDB2 窄因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _FeatureTable(SQL_FeatureTable):
    """WindDB2 特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _TimeSeriesTable(SQL_TimeSeriesTable):
    """WindDB2 时序因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _MappingTable(SQL_MappingTable):
    """WindDB2 映射因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

class _ConstituentTable(SQL_ConstituentTable):
    """WindDB2 成份因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=None, exchange_info=None, **kwargs)

# 财务因子表, 表结构特征:
# 报告期字段, 表示财报的报告期
# 公告日期字段, 表示财报公布的日期
class _FinancialTable(SQL_Table):
    """财务因子表"""
    class __QS_ArgClass__(SQL_Table.__QS_ArgClass__):
        ReportDate = Enum("所有", "年报", "中报", "一季报", "三季报", label="报告期", arg_type="SingleOption", order=0, option_range=["所有", "年报", "中报", "一季报", "三季报"])
        ReportType = ListStr(["408001000", "408004000", "408005000"], label="报表类型", arg_type="MultiOption", order=1, option_range=("408001000", "408004000", "408005000"))
        CalcType = Enum("最新", "单季度", "TTM", label="计算方法", arg_type="SingleOption", order=2, option_range=["最新", "单季度", "TTM"])
        YearLookBack = Int(0, label="回溯年数", arg_type="Integer", order=3)
        PeriodLookBack = Int(0, label="回溯期数", arg_type="Integer", order=4)
        ExprFactor = Str("", label="业绩快报因子", arg_type="String", order=5)
        NoticeFactor = Str("", label="业绩预告因子", arg_type="String", order=6)
        IgnoreMissing = Enum(True, False, label="忽略缺失", arg_type="Bool", order=7)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._ANNDateField = FactorInfo[FactorInfo["FieldType"]=="AnnDate"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._ReportTypeField = FactorInfo[FactorInfo["FieldType"]=="AdjustType"].index
        self._TempData = {}
        if self._ReportTypeField.shape[0]==0: self._ReportTypeField = None
        else: self._ReportTypeField = self._ReportTypeField[0]
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        FactorNames = FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
        return [self._ANNDateField, self._ReportDateField, self._ReportTypeField]+FactorNames
    # 返回在给定时点 idt 之前有财务报告的 ID
    # 如果 idt 为 None, 将返回所有有财务报告的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._ANNDateField]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._ANNDateField]+"<='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有财务报告的公告时点
    # 如果 iid 为 None, 将返回所有有财务报告的公告时点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._ANNDateField]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._ANNDateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._ANNDateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._ANNDateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._ANNDateField]
        return list(map(lambda x: dt.datetime.strptime(x[0], "%Y%m%d"), self._FactorDB.fetchall(SQLStr)))
    # 生成业绩快报SQL查询语句
    def _genExpressSQLStr(self, expr_factor, ids):
        DBTableName = self._FactorDB.TablePrefix + "AShareProfitExpress"
        # 将字段名转换成Wind内部的字段名
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc["中国A股业绩快报"].loc[['公告日期','Wind代码','报告期',expr_factor]]
        # 形成SQL语句, ID, 公告日期, 报告期, 财务数据
        SQLStr = 'SELECT '+DBTableName+'.'+FieldDict["Wind代码"]+', '
        SQLStr += DBTableName+'.'+FieldDict['公告日期']+', '
        SQLStr += DBTableName+'.'+FieldDict['报告期']+', '
        SQLStr += DBTableName+'.'+FieldDict[expr_factor]+' '
        SQLStr += 'FROM '+DBTableName+' '
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict["Wind代码"], ids, is_str=True, max_num=1000)+") "
        SQLStr += 'AND '+DBTableName+'.'+FieldDict['公告日期']+' IS NOT NULL '
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict['Wind代码']+', '
        SQLStr += DBTableName+'.'+FieldDict['公告日期']+', '
        SQLStr += DBTableName+'.'+FieldDict['报告期']
        return SQLStr
    # 生成业绩预告SQL查询语句
    def _genNoticeSQLStr(self, notice_factor, ids):
        DBTableName = self._FactorDB.TablePrefix + "AShareProfitNotice"
        # 将字段名转换成Wind内部的字段名
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc["中国A股业绩预告"].loc[['公告日期','Wind代码','报告期',notice_factor]]
        # 形成SQL语句, ID, 公告日期, 报告期, 财务数据
        SQLStr = 'SELECT '+DBTableName+'.'+FieldDict["Wind代码"]+', '
        SQLStr += DBTableName+'.'+FieldDict['公告日期']+', '
        SQLStr += DBTableName+'.'+FieldDict['报告期']+', '
        SQLStr += DBTableName+'.'+FieldDict[notice_factor]+'*10000 '
        SQLStr += 'FROM '+DBTableName+' '
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict["Wind代码"], ids, is_str=True, max_num=1000)+") "
        SQLStr +=     'AND '+DBTableName+'.'+FieldDict['公告日期']+' IS NOT NULL '
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict['Wind代码']+', '
        SQLStr += DBTableName+'.'+FieldDict['公告日期']+', '
        SQLStr += DBTableName+'.'+FieldDict['报告期']
        return SQLStr
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ExprNoticeConditionGroup = {}
        for iFactor in factors:
            iExprNoticeConditions = (iFactor.ExprFactor, iFactor.NoticeFactor)
            if iExprNoticeConditions not in ExprNoticeConditionGroup:
                ExprNoticeConditionGroup[iExprNoticeConditions] = {"FactorNames":[iFactor.Name], 
                                                                   "RawFactorNames":{iFactor._NameInFT}, 
                                                                   "args":iFactor.Args.copy()}
            else:
                ExprNoticeConditionGroup[iExprNoticeConditions]["FactorNames"].append(iFactor.Name)
                ExprNoticeConditionGroup[iExprNoticeConditions]["RawFactorNames"].add(iFactor._NameInFT)
        Groups = []
        for iExprNoticeConditions in ExprNoticeConditionGroup:
            Groups.append((self, ExprNoticeConditionGroup[iExprNoticeConditions]["FactorNames"], list(ExprNoticeConditionGroup[iExprNoticeConditions]["RawFactorNames"]), [], ExprNoticeConditionGroup[iExprNoticeConditions]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        Fields = list(set([self._IDField, self._ANNDateField, self._ReportDateField, self._ReportTypeField]+factor_names))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        if self._ReportTypeField is not None:
            # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
            SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
            SQLStr += "CASE WHEN ("+DBTableName+"."+FieldDict[self._ReportTypeField]+" = '408004000') OR (AShareIssuingDatePredict.s_stm_actual_issuingdate IS NULL) THEN "
            SQLStr += DBTableName+"."+FieldDict[self._ANNDateField]+" "
            SQLStr += "ELSE "+self._FactorDB.TablePrefix+"AShareIssuingDatePredict.s_stm_actual_issuingdate END AS ANNDate, "
            SQLStr += DBTableName+"."+FieldDict[self._ReportDateField]+", "
            SQLStr += DBTableName+"."+FieldDict[self._ReportTypeField]+", "
        else:
            # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
            SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
            SQLStr += DBTableName+"."+FieldDict[self._ANNDateField]+" AS ANNDate, "
            SQLStr += DBTableName+"."+FieldDict[self._ReportDateField]+", "
            SQLStr += "NULL AS ReportType, "
        for iField in factor_names:
            if iField in (self._IDField, self._ANNDateField, self._ReportDateField):
                SQLStr += DBTableName+"."+FieldDict[iField]+" AS "+iField+", "
            else:
                SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" "
        SQLStr += "FROM "+DBTableName+" "
        if self._ReportTypeField is not None:
            SQLStr += "LEFT JOIN "+self._FactorDB.TablePrefix+"AShareIssuingDatePredict ON ("+DBTableName+"."+FieldDict[self._IDField]+"="+self._FactorDB.TablePrefix+"AShareIssuingDatePredict.s_info_windcode AND "+self._FactorDB.TablePrefix+"AShareIssuingDatePredict.report_period="+DBTableName+"."+FieldDict[self._ReportDateField]+") "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        if self._ReportTypeField is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._ReportTypeField]+" IN ('"+"','".join(args.get("报表类型", self._QSArgs.ReportType))+"')"
        SQLStr = "SELECT t.* FROM ("+SQLStr+") t WHERE t.ANNDate IS NOT NULL "
        SQLStr += "ORDER BY t."+FieldDict[self._IDField]+", t.ANNDate, t."+FieldDict[self._ReportDateField]
        if self._ReportTypeField is not None: SQLStr += ", t."+FieldDict[self._ReportTypeField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData), columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        # 拼接业绩快报数据
        ExprFactor = args.get("业绩快报因子", self._QSArgs.ExprFactor)
        if ExprFactor:
            SQLStr = self._genExpressSQLStr(ExprFactor, ids)
            ExpressRawData = self._FactorDB.fetchall(SQLStr)
            if not ExpressRawData: ExpressRawData = pd.DataFrame(columns=['ID', 'AnnDate', 'ReportDate', ExprFactor])
            else: ExpressRawData = pd.DataFrame(np.array(ExpressRawData), columns=['ID', 'AnnDate', 'ReportDate', ExprFactor])
            ExpressRawData["ReportType"] = "1ProfitExpress"
            for iField in factor_names: ExpressRawData[iField] = ExpressRawData[ExprFactor]
            ExpressRawData.pop(ExprFactor)
            RawData = pd.concat([RawData, ExpressRawData])
        NoticeFactor = args.get("业绩预告因子", self._QSArgs.NoticeFactor)
        if NoticeFactor:
            SQLStr = self._genNoticeSQLStr(NoticeFactor, ids)
            NoticeRawData = self._FactorDB.fetchall(SQLStr)
            if not NoticeRawData: NoticeRawData = pd.DataFrame(columns=['ID', 'AnnDate', 'ReportDate', NoticeFactor])
            else: NoticeRawData = pd.DataFrame(np.array(NoticeRawData), columns=['ID', 'AnnDate', 'ReportDate', NoticeFactor])
            NoticeRawData['ReportType'] = "0ProfitNotice"
            for iField in factor_names: NoticeRawData[iField] = NoticeRawData[NoticeFactor]
            NoticeRawData.pop(NoticeFactor)
            RawData = pd.concat([RawData, NoticeRawData])
        if ExprFactor or NoticeFactor: RawData = RawData.sort_values(by=["ID", "AnnDate", "ReportDate", "ReportType"])
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Dates = sorted({iDT.strftime("%Y%m%d") for iDT in dts})
        CalcType, YearLookBack, PeriodLookBack, ReportDate, IgnoreMissing = args.get("计算方法", self._QSArgs.CalcType), args.get("回溯年数", self._QSArgs.YearLookBack), args.get("回溯期数", self._QSArgs.PeriodLookBack), args.get("报告期", self._QSArgs.ReportDate), args.get("忽略缺失", self._QSArgs.IgnoreMissing)
        if (YearLookBack==0) and (PeriodLookBack==0):
            if CalcType=="最新": CalcFun = self._calcIDData_LR
            elif CalcType=="单季度": CalcFun = self._calcIDData_SQ
            elif CalcType=="TTM": CalcFun = self._calcIDData_TTM
        elif YearLookBack>0:
            if CalcType=="最新": CalcFun = self._calcIDData_LR_NYear
            elif CalcType=="单季度": CalcFun = self._calcIDData_SQ_NYear
            elif CalcType=="TTM": CalcFun = self._calcIDData_TTM_NYear
        elif PeriodLookBack>0:
            if CalcType=="最新": CalcFun = self._calcIDData_LR_NPeriod
            elif CalcType=="单季度": CalcFun = self._calcIDData_SQ_NPeriod
            elif CalcType=="TTM": CalcFun = self._calcIDData_TTM_NPeriod
        raw_data = raw_data.set_index(["ID"])
        Data = {}
        for iID in raw_data.index.unique():
            Data[iID] = CalcFun(Dates, raw_data.loc[[iID]], factor_names, ReportDate, YearLookBack, PeriodLookBack, IgnoreMissing)
        Data = Panel(Data, major_axis=Dates, minor_axis=factor_names)
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Dates]
        Data.minor_axis = factor_names
        Data = Data.swapaxes(0, 2)
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        NewData = {}
        for i, iFactorName in enumerate(factor_names):
            if FactorInfo.loc[iFactorName, "DataType"].find("NUMBER")!=-1:
                NewData[iFactorName] = Data.iloc[i].astype("float")
            else:
                NewData[iFactorName] = Data.iloc[i]
        Data = adjustDateTime(Panel(NewData, items=factor_names, major_axis=Data.major_axis, minor_axis=Data.minor_axis), dts, fillna=False)
        Data = Data.loc[:, :, ids]
        return Data
    # 检索最大报告期的位置
    def _findMaxReportDateInd(self, idate, raw_data, report_date, MaxReportDateInd, MaxNoteDateInd, PreMaxNoteDateInd):
        if isinstance(report_date, dict):
            TargetReportDateDict = self._TempData.get("TargetReportDateDict",{})
            TargetReportDate = TargetReportDateDict.get(idate)
            if TargetReportDate is None:
                MonthDay = idate[-4:]
                for iStartDay, iEndDay in report_date:
                    if (iStartDay<=MonthDay) and (MonthDay<=iEndDay):
                        iBack, iMonthDay = report_date[(iStartDay,iEndDay)]
                        TargetReportDate = str(int(idate[:4])+iBack)+iMonthDay
                TargetReportDateDict[idate] = TargetReportDate
                self._TempData["TargetReportDateDict"] = TargetReportDateDict
            LastTargetReportDate = self._TempData.get("LastTargetReportDate")
            if LastTargetReportDate != TargetReportDate:
                MaxReportDateInd = -1
                for i in range(0, MaxNoteDateInd+1):
                    if (raw_data['ReportDate'].iloc[MaxNoteDateInd-i]==TargetReportDate):
                        MaxReportDateInd = MaxNoteDateInd-i
                        break
                if MaxReportDateInd==-1:
                    return (MaxReportDateInd, False)
                else:
                    self._TempData["LastTargetReportDate"] = TargetReportDate
                    self._TempData["LastTargetReportInd"] = MaxReportDateInd
                    return (MaxNoteDateInd,True)
            elif MaxNoteDateInd!=PreMaxNoteDateInd:
                NewMaxReportDateInd = MaxReportDateInd
                for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                    if (raw_data['ReportDate'].iloc[MaxNoteDateInd-i]==TargetReportDate):
                        NewMaxReportDateInd = MaxNoteDateInd-i
                        break
                self._TempData["LastTargetReportDate"] = TargetReportDate
                self._TempData["LastTargetReportInd"] = NewMaxReportDateInd
                return (NewMaxReportDateInd, (NewMaxReportDateInd!=MaxReportDateInd))
            else:
                MaxReportDateInd = self._TempData["LastTargetReportInd"]
                return (MaxReportDateInd, False)
        elif MaxNoteDateInd==PreMaxNoteDateInd:
            return (MaxReportDateInd, False)
        elif report_date == '所有':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (MaxReportDateInd<0) or (raw_data['ReportDate'].iloc[MaxNoteDateInd-i]>=raw_data['ReportDate'].iloc[MaxReportDateInd]):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '年报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['ReportDate'].iloc[MaxNoteDateInd-i][-4:]=='1231') and ((MaxReportDateInd<0) or (raw_data['ReportDate'].iloc[MaxNoteDateInd-i]>=raw_data['ReportDate'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '中报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['ReportDate'].iloc[MaxNoteDateInd-i][-4:]=='0630') and ((MaxReportDateInd<0) or (raw_data['ReportDate'].iloc[MaxNoteDateInd-i]>=raw_data['ReportDate'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '一季报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['ReportDate'].iloc[MaxNoteDateInd-i][-4:]=='0331') and ((MaxReportDateInd<0) or (raw_data['ReportDate'].iloc[MaxNoteDateInd-i]>=raw_data['ReportDate'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '三季报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['ReportDate'].iloc[MaxNoteDateInd-i][-4:]=='0930') and ((MaxReportDateInd<0) or (raw_data['ReportDate'].iloc[MaxNoteDateInd-i]>=raw_data['ReportDate'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        return (MaxReportDateInd, Changed)
    def _calcIDData_LR(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan, dtype="O")
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:# 最大报告期没有变化
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iRawData = raw_data.iloc[:tempInd+1]
            StdData[i] = iRawData[iRawData["ReportDate"]==MaxReportDate][factor_names].fillna(method=FillnaMethod).values[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_SQ(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        raw_data[factor_names] = raw_data[factor_names].astype("float")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iRawData = raw_data.iloc[:tempInd+1]
            if MaxReportDate[-4:]=="1231": iPreReportDate = MaxReportDate[0:4]+"0930"
            elif MaxReportDate[-4:]=="0930": iPreReportDate = MaxReportDate[0:4]+"0630"
            elif MaxReportDate[-4:]=="0630": iPreReportDate = MaxReportDate[0:4]+"0331"
            else:
                StdData[i] = iRawData[iRawData["ReportDate"]==MaxReportDate][factor_names].fillna(method=FillnaMethod).values[-1]
                continue
            iPreReportData = iRawData[iRawData["ReportDate"]==iPreReportDate][factor_names].fillna(method=FillnaMethod).values# 前一个报告期数据
            if iPreReportData.shape[0]==0: continue
            StdData[i] = iRawData[iRawData["ReportDate"]==MaxReportDate][factor_names].fillna(method=FillnaMethod).values[-1] - iPreReportData[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_TTM(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        raw_data[factor_names] = raw_data[factor_names].astype("float")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iRawData = raw_data.iloc[:tempInd+1]
            if MaxReportDate[-4:]=='1231':# 最新财报为年报
                StdData[i] = iRawData[iRawData["ReportDate"]==MaxReportDate][factor_names].fillna(method=FillnaMethod).values[-1]
            else:
                iLastYear = str(int(MaxReportDate[0:4])-1)
                iPreYearReport = iRawData[iRawData["ReportDate"]==iLastYear+"1231"][factor_names].fillna(method=FillnaMethod).values# 去年年报数据
                iPreReportData = iRawData[iRawData["ReportDate"]==iLastYear+MaxReportDate[-4:]][factor_names].fillna(method=FillnaMethod).values# 去年同期数据
                if (iPreReportData.shape[0]==0) or (iPreYearReport.shape[0]==0): continue
                StdData[i] = iRawData[iRawData["ReportDate"]==MaxReportDate][factor_names].fillna(method=FillnaMethod).values[-1] + iPreYearReport[-1] - iPreReportData[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_LR_NYear(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan, dtype="O")
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iLastNYear = str(int(MaxReportDate[0:4])-year_lookback)
            iRawData = raw_data.iloc[:tempInd+1]
            iPreData = iRawData[iRawData["ReportDate"]==iLastNYear+MaxReportDate[-4:]][factor_names].fillna(method=FillnaMethod).values
            if iPreData.shape[0]>0: StdData[i] = iPreData[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_SQ_NYear(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        raw_data[factor_names] = raw_data[factor_names].astype("float")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iLastNYear = str(int(MaxReportDate[0:4])-year_lookback)
            iRawData = raw_data.iloc[:tempInd+1]
            if MaxReportDate[-4:]=="1231":
                iPreReportDate1 = iLastNYear+"1231"
                iPreReportDate2 = iLastNYear+"0930"
            elif MaxReportDate[-4:]=="0930":
                iPreReportDate1 = iLastNYear+"0930"
                iPreReportDate2 = iLastNYear+"0630"
            elif MaxReportDate[-4:]=="0630":
                iPreReportDate1 = iLastNYear+"0630"
                iPreReportDate2 = iLastNYear+"0331"
            else:
                iPreReportData1 = iRawData[iRawData["ReportDate"]==iLastNYear+"0331"][factor_names].fillna(method=FillnaMethod).values
                if iPreReportData1.shape[0]>0: StdData[i] = iPreReportData1[-1]
                continue
            iPreReportData1 = iRawData[iRawData["ReportDate"]==iPreReportDate1][factor_names].fillna(method=FillnaMethod).values# 上N年同期财报数据
            iPreReportData2 = iRawData[iRawData["ReportDate"]==iPreReportDate2][factor_names].fillna(method=FillnaMethod).values# 上N年同期的上一期财报数据
            if (iPreReportData1.shape[0]==0) or (iPreReportData2.shape[0]==0): continue
            StdData[i] = iPreReportData1[-1] - iPreReportData2[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_TTM_NYear(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        raw_data[factor_names] = raw_data[factor_names].astype("float")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iLastNYear = int(MaxReportDate[0:4])-year_lookback
            iRawData = raw_data.iloc[:tempInd+1]
            if MaxReportDate[-4:]=="1231":# 最新财报为年报
                iPreNReportData = iRawData[iRawData["ReportDate"]==str(iLastNYear)+"1231"][factor_names].fillna(method=FillnaMethod).values
                if iPreNReportData.shape[0]>0: StdData[i] = iPreNReportData[-1]
                continue
            iPreNReportData = iRawData[iRawData["ReportDate"]==str(iLastNYear)+MaxReportDate[-4:]][factor_names].fillna(method=FillnaMethod).values# 上N年同期数据
            iPreN_1YearReportData = iRawData[iRawData["ReportDate"]==str(iLastNYear-1)+"1231"][factor_names].fillna(method=FillnaMethod).values# 上N+1年年报数据
            iPreN_1ReportData = iRawData[iRawData["ReportDate"]==str(iLastNYear-1)+MaxReportDate[-4:]][factor_names].fillna(method=FillnaMethod).values# 上N+1年同期数据
            if (iPreNReportData.shape[0]==0) or (iPreN_1YearReportData.shape[0]==0) or (iPreN_1ReportData.shape[0]==0): continue
            StdData[i] = iPreNReportData[-1] + iPreN_1YearReportData[-1] - iPreN_1ReportData[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_LR_NPeriod(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan, dtype="O")
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iRawData = raw_data.iloc[:tempInd+1]
            ObjectReportDate = RollBackNPeriod(MaxReportDate, period_lookback)
            iPreData = iRawData[iRawData["ReportDate"]==ObjectReportDate][factor_names].fillna(method=FillnaMethod).values
            if iPreData.shape[0]>0: StdData[i] = iPreData[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_SQ_NPeriod(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        raw_data[factor_names] = raw_data[factor_names].astype("float")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            ObjectReportDate = RollBackNPeriod(MaxReportDate, period_lookback)# 上N期报告期
            iRawData = raw_data.iloc[:tempInd+1]
            if ObjectReportDate[-4:]=="1231":
                iPreReportDate = ObjectReportDate[0:4]+"0930"
            elif ObjectReportDate[-4:]=="0930":
                iPreReportDate = ObjectReportDate[0:4]+"0630"
            elif ObjectReportDate[-4:]=="0630":
                iPreReportDate = ObjectReportDate[0:4]+"0331"
            else:
                iPreReportData1 = iRawData[iRawData["ReportDate"]==ObjectReportDate][factor_names].fillna(method=FillnaMethod).values
                if iPreReportData1.shape[0]>0: StdData[i] = iPreReportData1[-1]
                continue
            iPreReportData1 = iRawData[iRawData["ReportDate"]==ObjectReportDate][factor_names].fillna(method=FillnaMethod).values# 上N期财报数据
            iPreReportData2 = iRawData[iRawData["ReportDate"]==iPreReportDate][factor_names].fillna(method=FillnaMethod).values# 上N+1期财报数据
            if (iPreReportData1.shape[0]==0) or (iPreReportData2.shape[0]==0): continue
            StdData[i] = iPreReportData1[-1] - iPreReportData2[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_TTM_NPeriod(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback, ignore_missing):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        FillnaMethod = ("pad" if ignore_missing else "bfill")
        raw_data[factor_names] = raw_data[factor_names].astype("float")
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['AnnDate'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
            MaxReportDate = raw_data['ReportDate'].iloc[MaxReportDateInd]# 当前最大报告期
            iRawData = raw_data.iloc[:tempInd+1]
            ObjectReportDate = RollBackNPeriod(MaxReportDate, period_lookback)
            if ObjectReportDate[-4:]=='1231':# 上N期财报为年报
                iPreNPeriodReportData = iRawData[iRawData["ReportDate"]==ObjectReportDate][factor_names].fillna(method=FillnaMethod).values
                if iPreNPeriodReportData.shape[0]>0: StdData[i] = iPreNPeriodReportData[-1]
                continue
            iPreNPeriodReportData = iRawData[iRawData["ReportDate"]==ObjectReportDate][factor_names].fillna(method=FillnaMethod).values# 上N期数据
            iPreNPeriodYear_1YearReportData = iRawData[iRawData["ReportDate"]==str(int(ObjectReportDate[0:4])-1)+"1231"][factor_names].fillna(method=FillnaMethod).values# 上N期上一年年报数据
            iPreNPeriodYear_1ReportData = iRawData[iRawData["ReportDate"]==str(int(ObjectReportDate[0:4])-1)+ObjectReportDate[-4:]][factor_names].fillna(method=FillnaMethod).values# 上N期上一年同期数据
            if (iPreNPeriodReportData.shape[0]==0) or (iPreNPeriodYear_1YearReportData.shape[0]==0) or (iPreNPeriodYear_1ReportData.shape[0]==0): continue
            StdData[i] = iPreNPeriodReportData[-1] + iPreNPeriodYear_1YearReportData[-1] - iPreNPeriodYear_1ReportData[-1]
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData

class WindDB2(QSSQLObject, FactorDB):
    """Wind 量化研究数据库"""
    class __QS_ArgClass__(QSSQLObject.__QS_ArgClass__, FactorDB.__QS_ArgClass__):
        Name = Str("WindDB2", arg_type="String", label="名称", order=-100)
        DBInfoFile = File(label="库信息文件", arg_type="File", order=100)
        FTArgs = Dict({"时点格式": "%Y%m%d", "日期格式": "%Y%m%d"}, label="因子表参数", arg_type="Dict", order=101)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"WindDB2Config.json" if config_file is None else config_file), **kwargs)
        self._InfoFilePath = __QS_LibPath__+os.sep+"WindDB2Info.hdf5"# 数据库信息文件路径
        if not os.path.isfile(self._QSArgs.DBInfoFile):
            if self._QSArgs.DBInfoFile: self._QS_Logger.warning("找不到指定的库信息文件 : '%s'" % self._QSArgs.DBInfoFile)
            self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"WindDB2Info.xlsx"# 默认数据库信息源文件路径
            self._TableInfo, self._FactorInfo = updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger, out_info=False)# 数据库表信息, 数据库字段信息
        else:
            self._InfoResourcePath = self._QSArgs.DBInfoFile
            self._TableInfo, self._FactorInfo = updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger, out_info=True)# 数据库表信息, 数据库字段信息
        return
    @property
    def TableNames(self):
        if self._TableInfo is not None: return self._TableInfo[pd.notnull(self._TableInfo["TableClass"])].index.tolist()
        else: return []
    def getTable(self, table_name, args={}):
        if table_name in self._TableInfo.index:
            TableClass = args.get("因子表类型", self._TableInfo.loc[table_name, "TableClass"])
            if pd.notnull(TableClass) and (TableClass!=""):
                DefaultArgs = self._TableInfo.loc[table_name, "DefaultArgs"]
                if pd.isnull(DefaultArgs): DefaultArgs = {}
                else: DefaultArgs = eval(DefaultArgs)
                Args = self._QSArgs.FTArgs.copy()
                Args.update(DefaultArgs)
                Args.update(args)
                return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=Args, logger=self._QS_Logger)")
        Msg = ("因子库 '%s' 目前尚不支持因子表: '%s'" % (self.Name, table_name))
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)
    # -----------------------------------------数据提取---------------------------------
    # 给定起始日期和结束日期, 获取交易所交易日期, 目前支持: "SSE", "SZSE", "SHFE", "DCE", "CZCE", "INE", "CFEEX"
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if start_date is None: start_date = dt.datetime(1900, 1, 1)
        if end_date is None: end_date = dt.datetime.today()
        ExchangeInfo = self._TableInfo.loc[["香港交易所交易日历", "中国A股交易日历", "中国期货交易日历", "中国债券市场交易日", "中国期权交易日历"]]
        ExchangeInfo = ExchangeInfo[ExchangeInfo["Description"].str.contains(exchange)]
        if ExchangeInfo.shape[0]==0: raise __QS_Error__("不支持交易所: '%s' 的交易日序列!" % exchange)
        else: Dates = self.getTable(ExchangeInfo.index[0]).getDateTime(iid=exchange, start_dt=start_date, end_dt=end_date)
        if kwargs.get("output_type", "datetime")=="date": return list(map(lambda x: x.date(), Dates))
        else: return Dates
    # 获取指定日 date 的全体 A 股 ID
    # date: 指定日, datetime.date
    # is_current: False 表示上市日在指定日之前的 A 股, True 表示上市日在指定日之前且尚未退市的 A 股
    def _getAllAStock(self, date, is_current=True):
        if is_current:
            SQLStr = "SELECT S_INFO_WINDCODE FROM {Prefix}AShareDescription WHERE (S_INFO_DELISTDATE is NULL OR S_INFO_DELISTDATE>'{Date}') AND S_INFO_LISTDATE<='{Date}' ORDER BY S_INFO_WINDCODE"
        else:
            SQLStr = "SELECT S_INFO_WINDCODE FROM {Prefix}AShareDescription WHERE S_INFO_LISTDATE<='{Date}' ORDER BY S_INFO_WINDCODE"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y%m%d")))]
    # 获取指定日 date 指数 index_id 的成份股 ID
    # index_id: 指数 ID, 默认值 "全体A股"
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示进入指数的日期在指定日之前的成份股, True 表示进入指数的日期在指定日之前且尚未剔出指数的 A 股
    def getStockID(self, index_id="全体A股", date=None, is_current=True):
        if date is None: date = dt.date.today()
        if index_id=="全体A股": return self._getAllAStock(date=date, is_current=is_current)
        for iTableName in self._TableInfo[(self._TableInfo["TableClass"]=="ConstituentTable") & self._TableInfo.index.str.contains("A股")].index:
            IDs = self.getTable(iTableName).getID(ifactor_name=index_id, idt=date, is_current=is_current)
            if IDs: return IDs
        else: return []
    
    # 获取指定日 date 基金 ID
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示成立日在指定日之前的基金, True 表示成立日在指定日之前且尚未清盘的基金
    def getMutualFundID(self, exchange=None, date=None, is_current=True, start_date=None, **kwargs):
        if date is None: date = dt.date.today()
        if start_date is not None: start_date = start_date.strftime("%Y%m%d")
        SQLStr = "SELECT f_info_windcode AS ID FROM {Prefix}ChinaMutualFundDescription "
        SQLStr += "WHERE {Prefix}ChinaMutualFundDescription.f_info_setupdate <= '{Date}' "
        if start_date is not None:
            SQLStr += "AND (({Prefix}ChinaMutualFundDescription.f_info_maturitydate IS NULL) OR ({Prefix}ChinaMutualFundDescription.f_info_maturitydate >= '{StartDate}')) "
        if is_current:
            if start_date is None:
                SQLStr += "AND (({Prefix}ChinaMutualFundDescription.f_info_maturitydate IS NULL) OR ({Prefix}ChinaMutualFundDescription.f_info_maturitydate >= '{Date}')) "
            else:
                SQLStr += "AND {Prefix}ChinaMutualFundDescription.f_info_setupdate <= '{StartDate}' "
                SQLStr += "AND (({Prefix}ChinaMutualFundDescription.f_info_maturitydate IS NULL) OR ({Prefix}ChinaMutualFundDescription.f_info_maturitydate >= '{Date}')) "
        if exchange:
            if isinstance(exchange, str):
                SQLStr += f"AND {self.TablePrefix}ChinaMutualFundDescription.f_info_exchmarket = '{exchange}' "
            else:
                SQLStr += "AND {Prefix}ChinaMutualFundDescription.f_info_exchmarket IN ('"+"', '".join(exchange)+"') "
        SQLStr += "ORDER BY ID"
        Rslt = np.array(self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y%m%d"), StartDate=start_date)))
        if Rslt.shape[0]>0: return Rslt[:, 0].tolist()
        else: return []
    
    # 给定期货代码 future_code, 获取指定日 date 的期货 ID
    # future_code: 期货代码(str)或者期货代码列表(list(str)), None 表示所有期货代码
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日在指定日之前的期货, True 表示上市日在指定日之前且尚未退市的期货
    # kwargs:
    # contract_type: 合约类型, 可选 "月合约", "连续合约", "所有", 默认值 "月合约"
    # include_simulation: 是否包括仿真合约, 默认值 False
    def getFutureID(self, future_code="IF", date=None, is_current=True, **kwargs):
        if date is None: date = dt.date.today()
        SQLStr = "SELECT DISTINCT s_info_windcode FROM {Prefix}CFuturesDescription "
        if future_code:
            if isinstance(future_code, str): SQLStr += "WHERE fs_info_sccode='"+future_code+"' "
            else: SQLStr += "WHERE fs_info_sccode IN ('"+"', '".join(future_code)+"') "
        else: SQLStr += "WHERE fs_info_sccode IS NOT NULL "
        if not kwargs.get("include_simulation", False): SQLStr += "AND s_info_name NOT LIKE '%仿真%' "
        ContractType = kwargs.get("contract_type", "月合约")
        if ContractType!="所有": SQLStr += "AND fs_info_type="+("2" if ContractType=="连续合约" else "1")+" "
        if ContractType!="连续合约":
            SQLStr += "AND ((s_info_listdate<='{Date}') OR (s_info_listdate IS NULL)) "
            if is_current: SQLStr += "AND ((s_info_delistdate>='{Date}') OR (s_info_delistdate IS NULL)) "
        SQLStr += "ORDER BY s_info_windcode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y%m%d"), FutureCode=future_code))]
    # 获取指定交易所 exchange 指定日 date 的期货代码
    # exchange: 交易所(str)或者交易所列表(list(str)), 默认值 None 表示支持的所有交易所
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日在指定日之前的期货代码, True 表示上市日在指定日之前且尚未退市的期货代码
    # kwargs:
    # include_simulation: 是否包括仿真合约, 默认值 False
    def getFutureCode(self, exchange=None, date=None, is_current=True, **kwargs):
        if date is None: date = dt.date.today()
        SQLStr = "SELECT DISTINCT fs_info_sccode FROM {Prefix}CFuturesDescription "
        SQLStr += "WHERE s_info_listdate<='{Date}' "
        if is_current: SQLStr += "AND s_info_delistdate>='{Date}' "
        if exchange:
            if isinstance(exchange, str): SQLStr += "AND s_info_exchmarket='"+exchange+"' "
            else: SQLStr += "AND s_info_exchmarket IN ('"+"', '".join(exchange)+"') "
        if not kwargs.get("include_simulation", False): SQLStr += "AND s_info_name NOT LIKE '%仿真%' "
        SQLStr += "ORDER BY fs_info_sccode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y%m%d")))]
    # 给定期权代码 option_code, 获取指定日 date 的期权代码
    # option_code: 期权代码(str)
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日在指定日之前的期权, True 表示上市日在指定日之前且尚未退市的期权
    def getOptionID(self, option_code="510050OP", date=None, is_current=True, **kwargs):
        if date is None: date = dt.date.today()
        SQLStr = "SELECT DISTINCT s_info_windcode FROM {Prefix}ChinaOptionDescription "
        SQLStr += "WHERE s_info_sccode LIKE '{OptionCode}%%' "
        SQLStr += "AND s_info_ftdate<='{Date}' "
        if is_current: SQLStr += "AND s_info_lasttradingdate>='{Date}' "
        SQLStr += "ORDER BY s_info_windcode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y%m%d"), OptionCode=option_code))]

if __name__=="__main__":
    iDB = WindDB2()
    iDB.getStockID()