# coding=utf-8
"""Wind 量化研究数据库"""
import re
import os
import shelve
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Int, Str, Range, Bool, List, ListStr, Dict, Function, Password, Either

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries, getDateSeries
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.FileFun import getShelveFileSuffix
from QuantStudio import __QS_Object__, __QS_Error__, __QS_LibPath__, __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import updateInfo, adjustDateTime

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
        ANN_ReportFilePath = raw_data_dir+os.sep+PID+os.sep+report_ann_file
        pid_lock[PID].acquire()
        if not os.path.isfile(ANN_ReportFilePath+("."+ft._ANN_ReportFileSuffix if ft._ANN_ReportFileSuffix else "")):# 没有报告期-公告日期数据, 提取该数据
            with shelve.open(ANN_ReportFilePath) as ANN_ReportFile: pass
            pid_lock[PID].release()
            IDs = []
            for iPID in sorted(pid_ids): IDs.extend(pid_ids[iPID])
            RawData = _prepareReportANNRawData(ft.FactorDB, ids=IDs)
            super(_DBTable, ft).__QS_saveRawData__(RawData, [], raw_data_dir, pid_ids, report_ann_file, pid_lock)
        else:
            pid_lock[PID].release()
    raw_data = raw_data.set_index(['ID'])
    CommonCols = list(raw_data.columns.difference(set(factor_names)))
    AllIDs = set(raw_data.index)
    for iPID, iIDs in pid_ids.items():
        with shelve.open(raw_data_dir+os.sep+iPID+os.sep+file_name) as iFile:
            iInterIDs = sorted(AllIDs.intersection(set(iIDs)))
            iData = raw_data.loc[iInterIDs]
            for jFactorName in factor_names:
                ijData = iData[CommonCols+[jFactorName]].reset_index()
                if isANNReport: ijData.columns.name = raw_data_dir+os.sep+iPID+os.sep+report_ann_file
                iFile[jFactorName] = ijData
            iFile["_QS_IDs"] = iIDs
    return 0

# f: 该算子所属的因子对象或因子表对象
# idt: 当前所处的时点
# iid: 当前待计算的 ID
# x: 当期的数据, 分析师评级时为: DataFrame(columns=["日期", ...]), 分析师盈利预测时为: [DataFrame(columns=["日期", "报告期", "预测基准股本", ...])], list的长度为向前年数
# args: 参数, {参数名:参数值}
def _DefaultOperator(f, idt, iid, x, args):
    return np.nan

class _DBTable(FactorTable):
    def getMetaData(self, key=None):
        TableInfo = self._FactorDB._TableInfo.loc[self.Name]
        if key is None:
            return TableInfo
        else:
            return TableInfo.get(key, None)
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None:
            factor_names = self.FactorNames
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        if key=="DataType":
            if hasattr(self, "_DataType"): return self._DataType.loc[factor_names]
            MetaData = FactorInfo["DataType"].loc[factor_names]
            for i in range(MetaData.shape[0]):
                iDataType = MetaData.iloc[i].lower()
                if iDataType.find("number")!=-1: MetaData.iloc[i] = "double"
                else: MetaData.iloc[i] = "string"
            return MetaData
        elif key=="Description": return FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType"),
                                 "Description":self.getFactorMetaData(factor_names, key="Description")})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))

class _CalendarTable(_DBTable):
    """交易日历因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._DataType = pd.Series("double", index=["交易日"])
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    @property
    def FactorNames(self):
        return ["交易日"]
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType":
            return self._DataType.loc[factor_names]
        elif key=="Description": return pd.Series(["0 or nan: 非交易日; 1: 交易日"]*len(factor_names), index=factor_names)
        elif key is None:
            return pd.DataFrame({"DataType": self.getFactorMetaData(factor_names, key="DataType"),
                                 "Description": self.getFactorMetaData(factor_names, key="Description")})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 返回给定时点 idt 有交易的交易所列表
    # 如果 idt 为 None, 将返回表中的所有交易所
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._DateField, self._IDField]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._DateField]+"='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回交易所为 iid 的交易日列表
    # 如果 iid 为 None, 将返回表中有记录数据的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._DateField]]
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else: SQLStr = "SELECT DISTINCT "+SQLStr[7:]+"WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._DateField]
        return list(map(lambda x: dt.datetime.strptime(x[0], "%Y%m%d"), self._FactorDB.fetchall(SQLStr)))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._DateField, self._IDField]]
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+dts[0].strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+dts[-1].strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"])
        return pd.DataFrame(np.array(RawData), columns=["日期", "ID"])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data["交易日"] = 1
        Data = pd.Panel({"交易日":raw_data.set_index(["日期", "ID"])["交易日"].unstack()}).loc[factor_names]
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Data.major_axis]
        return adjustDateTime(Data, dts, fillna=True, value=0)
# 行情因子表, 表结构特征:
# 日期字段, 表示数据填充的时点; 可能存在多个日期字段, 必须指定其中一个作为数据填充的时点
# 条件字段, 作为条件过滤记录; 可能存在多个条件字段
# 在设定某些条件下, 数据填充时点和 ID 可以唯一标志一行记录
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class _MarketTable(_DBTable):
    """行情因子表"""
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=0)
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=1)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateFields = FactorInfo[FactorInfo["FieldType"]=="Date"].index.tolist()# 所有的日期字段列表
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        iFactorInfo = FactorInfo[pd.notnull(FactorInfo["Supplementary"]) & (FactorInfo["FieldType"]=="Date")]
        self._UniqueDateField = iFactorInfo[iFactorInfo["Supplementary"].str.contains("UniqueDate")].index# 具有唯一性的日期字段, 如果未设置, 则默认所有日期字段都有唯一性
        if self._UniqueDateField.shape[0]==0: self._UniqueDateField = None
        else: self._UniqueDateField = self._UniqueDateField[0]
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        self.add_trait("DateField", Enum(*self._DateFields, arg_type="SingleOption", label="日期字段", order=1))
        iFactorInfo = FactorInfo[(FactorInfo["FieldType"]=="Date") & pd.notnull(FactorInfo["Supplementary"])]
        iFactorInfo = iFactorInfo[iFactorInfo["Supplementary"].str.contains("DefaultDate")]
        if iFactorInfo.shape[0]>0: self.DateField = iFactorInfo.index[0]
        else: self.DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+2))
            self[iCondition] = str(FactorInfo.loc[iCondition, "Supplementary"])
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()+self._DateFields
    def getCondition(self, icondition, ids=None, dts=None):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField, self._IDField, icondition]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[icondition]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None: SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if dts is not None:
            Dates = list({iDT.strftime("%Y%m%d") for iDT in dts})
            SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict[self.DateField], Dates, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[icondition]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField, self._IDField]+self._ConditionFields]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self.DateField]+"='"+idt.strftime("%Y%m%d")+"' "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self.DateField]+" IS NOT NULL "
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for iConditionField in self._ConditionFields:
            if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
            else:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+args.get(iConditionField, self[iConditionField])+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField, self._IDField]+self._ConditionFields]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self.DateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for iConditionField in self._ConditionFields:
            if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
            else:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+args.get(iConditionField, self[iConditionField])+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self.DateField]
        return list(map(lambda x: dt.datetime.strptime(x[0], "%Y%m%d"), self._FactorDB.fetchall(SQLStr)))
    def __QS_genGroupInfo__(self, factors, operation_mode):
        DateConditionGroup = {}
        for iFactor in factors:
            iDateConditions = (iFactor.DateField, ";".join([iArgName+":"+iFactor[iArgName] for iArgName in iFactor.ArgNames if iArgName!="回溯天数"]))
            if iDateConditions not in DateConditionGroup:
                DateConditionGroup[iDateConditions] = {"FactorNames":[iFactor.Name], 
                                                       "RawFactorNames":{iFactor._NameInFT}, 
                                                       "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                       "args":iFactor.Args.copy()}
            else:
                DateConditionGroup[iDateConditions]["FactorNames"].append(iFactor.Name)
                DateConditionGroup[iDateConditions]["RawFactorNames"].add(iFactor._NameInFT)
                DateConditionGroup[iDateConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], DateConditionGroup[iDateConditions]["StartDT"])
                DateConditionGroup[iDateConditions]["args"]["回溯天数"] = max(DateConditionGroup[iDateConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iDateConditions in DateConditionGroup:
            StartInd = operation_mode.DTRuler.index(DateConditionGroup[iDateConditions]["StartDT"])
            Groups.append((self, DateConditionGroup[iDateConditions]["FactorNames"], list(DateConditionGroup[iDateConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], DateConditionGroup[iDateConditions]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField, self._IDField]+self._ConditionFields+factor_names]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self.DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        if (self._UniqueDateField is not None) and (self._UniqueDateField!=self.DateField):
            UniqueDateField = FactorInfo.loc[self._UniqueDateField, "DBFieldName"]
            SubSQLStr = "SELECT MAX("+DBTableName+"."+UniqueDateField+") AS UniqueDateField, "
            SubSQLStr += DBTableName+"."+FieldDict[self.DateField]+", "
            SubSQLStr += DBTableName+"."+FieldDict[self._IDField]+" "
            SubSQLStr += "FROM "+DBTableName+" "
            SubSQLStr += "GROUP BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self.DateField]
            SQLStr += "INNER JOIN ("+SubSQLStr+") t ON ("
            SQLStr += DBTableName+"."+FieldDict[self._IDField]+"=t."+FieldDict[self._IDField]+" "
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"=t."+FieldDict[self.DateField]+" "
            SQLStr += "AND "+DBTableName+"."+UniqueDateField+"=t.UniqueDateField) "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        for iConditionField in self._ConditionFields:
            if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
            else:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+args.get(iConditionField, self[iConditionField])+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self.DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["日期", "ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Data.major_axis]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]

class _DividendTable(_DBTable):
    """分红因子表"""
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=0)
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=1)
    Operator = Either(Function(None), None, arg_type="Function", label="算子", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateFields = FactorInfo[FactorInfo["FieldType"]=="Date"].index.tolist()# 所有的日期字段列表
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        self.add_trait("DateField", Enum(*self._DateFields, arg_type="SingleOption", label="日期字段", order=0))
        iFactorInfo = FactorInfo[(FactorInfo["FieldType"]=="Date") & pd.notnull(FactorInfo["Supplementary"])]
        iFactorInfo = iFactorInfo[iFactorInfo["Supplementary"].str.contains("DefaultDate")]
        if iFactorInfo.shape[0]>0: self.DateField = iFactorInfo.index[0]
        else: self.DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+3))
            setattr(self, "Condition"+str(i), str(FactorInfo.loc[iCondition, "Supplementary"]))
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()+self._DateFields+self._ConditionFields
    def getCondition(self, icondition, ids=None, dts=None):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._DateField, self._IDField, icondition]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[icondition]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None: SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if dts is not None:
            Dates = list({iDT.strftime("%Y%m%d") for iDT in dts})
            SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict[self._DateField], Dates, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[icondition]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField, self._IDField]+self._ConditionFields]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self.DateField]+"='"+idt.strftime("%Y%m%d")+"' "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self.DateField]+" IS NOT NULL "
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for iConditionField in self._ConditionFields:
            if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
            else:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+args.get(iConditionField, self[iConditionField])+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField, self._IDField]+self._ConditionFields]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self.DateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for iConditionField in self._ConditionFields:
            if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
            else:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+args.get(iConditionField, self[iConditionField])+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self.DateField]
        return list(map(lambda x: dt.datetime.strptime(x[0], "%Y%m%d"), self._FactorDB.fetchall(SQLStr)))
    def __QS_genGroupInfo__(self, factors, operation_mode):
        DateConditionGroup = {}
        for iFactor in factors:
            iDateConditions = (iFactor.DateField, ";".join([iArgName+":"+iFactor[iArgName] for iArgName in iFactor.ArgNames if iArgName not in ("回溯天数", "算子")]))
            if iDateConditions not in DateConditionGroup:
                DateConditionGroup[iDateConditions] = {"FactorNames":[iFactor.Name], 
                                                       "RawFactorNames":{iFactor._NameInFT}, 
                                                       "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                       "args":iFactor.Args.copy()}
            else:
                DateConditionGroup[iDateConditions]["FactorNames"].append(iFactor.Name)
                DateConditionGroup[iDateConditions]["RawFactorNames"].add(iFactor._NameInFT)
                DateConditionGroup[iDateConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], DateConditionGroup[iDateConditions]["StartDT"])
                DateConditionGroup[iDateConditions]["args"]["回溯天数"] = max(DateConditionGroup[iDateConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iDateConditions in DateConditionGroup:
            StartInd = operation_mode.DTRuler.index(DateConditionGroup[iDateConditions]["StartDT"])
            Groups.append((self, DateConditionGroup[iDateConditions]["FactorNames"], list(DateConditionGroup[iDateConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], DateConditionGroup[iDateConditions]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField, self._IDField]+self._ConditionFields+factor_names]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self.DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        for iConditionField in self._ConditionFields:
            if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
            else:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+args.get(iConditionField, self[iConditionField])+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self.DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["日期", "ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Operator = args.get("算子", self.Operator)
        if Operator is None: Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in raw_data.columns:
            Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        Data = pd.Panel(Data).loc[factor_names, :, ids]
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Data.major_axis]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, :]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]

class _ConstituentTable(_DBTable):
    """成份因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._GroupField = FactorInfo[FactorInfo["FieldType"]=="Group"].index[0]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._InDateField = FactorInfo[FactorInfo["FieldType"]=="InDate"].index[0]
        self._OutDateField = FactorInfo[FactorInfo["FieldType"]=="OutDate"].index[0]
        self._CurSignField = FactorInfo[FactorInfo["FieldType"]=="CurSign"].index
        if self._CurSignField.shape[0]==0: self._CurSignField = None
        else: self._CurSignField = self._CurSignField[0]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        if not hasattr(self, "_IndexIDs"):# [指数 ID]
            DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
            FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._GroupField]]
            SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._GroupField]+" "# 指数 ID
            SQLStr += "FROM "+DBTableName+" "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._GroupField]
            self._IndexIDs = [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        return self._IndexIDs
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        if key=="DataType":
            return pd.Series("double", index=factor_names)
        elif key=="Description": return pd.Series(["0 or nan: 非成分; 1: 是成分"]*len(factor_names), index=factor_names)
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType"),
                                 "Description":self.getFactorMetaData(factor_names, key="Description")})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 返回指数 ID 为 ifactor_name 在给定时点 idt 的所有成份股
    # 如果 idt 为 None, 将返回指数 ifactor_name 的所有历史成份股
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有 ID
    def getID(self, ifactor_name=None, idt=None, args={}, **kwargs):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        Fields = [self._IDField, self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        if ifactor_name is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._GroupField]+"='"+ifactor_name+"' "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._GroupField]+" IS NOT NULL "
        if idt is not None:
            idt = idt.strftime("%Y%m%d")
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+idt+"' "
            if kwargs.get("is_current", True):
                SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+idt+"') "
                if self._CurSignField:
                    SQLStr += "OR ("+DBTableName+"."+FieldDict[self._CurSignField]+"=1)) "
                else:
                    SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回指数 ID 为 ifactor_name 包含成份股 iid 的时间点序列
    # 如果 iid 为 None, 将返回指数 ifactor_name 的有记录数据的时间点序列
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有时间点
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        Fields = [self._IDField, self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        if iid is not None:
            SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._InDateField]+" "# 纳入日期
            SQLStr += DBTableName+"."+FieldDict[self._OutDateField]+" "# 剔除日期
            SQLStr += "FROM "+DBTableName+" "
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._InDateField]+" IS NOT NULL "
            if ifactor_name is not None:
                SQLStr += "AND "+DBTableName+"."+FieldDict[self._GroupField]+"='"+ifactor_name+"' "
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
            if start_dt is not None:
                SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+start_dt.strftime("%Y%m%d")+"') "
                SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL))"
            if end_dt is not None:
                SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._InDateField]
            Data = self._FactorDB.fetchall(SQLStr)
            DateTimes = set()
            for iStartDate, iEndDate in Data:
                iStartDT = dt.datetime.strptime(iStartDate, "%Y%m%d")
                if iEndDate is None: iEndDT = (dt.datetime.now() if end_dt is None else end_dt)
                else: iEndDT = dt.datetime.strptime(iEndDate, "%Y%m%d")
                DateTimes = DateTimes.union(getDateTimeSeries(start_dt=iStartDT, end_dt=iEndDT, timedelta=dt.timedelta(1)))
            return sorted(DateTimes)
        SQLStr = "SELECT MIN("+DBTableName+"."+FieldDict[self._InDateField]+") "# 纳入日期
        SQLStr += "FROM "+DBTableName
        if ifactor_name is not None:
            SQLStr += " WHERE "+DBTableName+"."+FieldDict[self._GroupField]+"='"+ifactor_name+"'"
        StartDT = dt.datetime.strptime(self._FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d")
        if start_dt is not None:
            StartDT = max((StartDT, start_dt))
        if end_dt is None:
            end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        Fields = [self._GroupField, self._IDField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        # 指数中成份股 ID, 指数证券 ID, 纳入日期, 剔除日期, 最新标志
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._GroupField]+", "# 指数证券 ID
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "# ID
        SQLStr += DBTableName+"."+FieldDict[self._InDateField]+", "# 纳入日期
        SQLStr += DBTableName+"."+FieldDict[self._OutDateField]+" "# 剔除日期
        if self._CurSignField: SQLStr = SQLStr[:-1]+", "+DBTableName+"."+FieldDict[self._CurSignField]+" "# 最新标志
        SQLStr += 'FROM '+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._GroupField], factor_names, is_str=True, max_num=1000)+") "
        SQLStr += 'AND ('+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+') '
        SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+StartDate.strftime("%Y%m%d")+"') "
        SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += 'ORDER BY '+DBTableName+"."+FieldDict[self._GroupField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._InDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=Fields)
        else: return pd.DataFrame(np.array(RawData), columns=Fields)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DateSeries = getDateSeries(StartDate, EndDate)
        Data = {}
        for iIndexID in factor_names:
            iRawData = raw_data[raw_data[self._GroupField]==iIndexID].set_index([self._IDField])
            iData = pd.DataFrame(0, index=DateSeries, columns=pd.unique(iRawData.index))
            for jID in iData.columns:
                jIDRawData = iRawData.loc[[jID]]
                for k in range(jIDRawData.shape[0]):
                    kStartDate = dt.datetime.strptime(jIDRawData[self._InDateField].iloc[k], "%Y%m%d").date()
                    kEndDate = (dt.datetime.strptime(jIDRawData[self._OutDateField].iloc[k], "%Y%m%d").date()-dt.timedelta(1) if jIDRawData[self._OutDateField].iloc[k] is not None else dt.date.today())
                    iData[jID].loc[kStartDate:kEndDate] = 1
            Data[iIndexID] = iData
        Data = pd.Panel(Data)
        if Data.minor_axis.intersection(ids).shape[0]==0: return pd.Panel(0.0, items=factor_names, major_axis=dts, minor_axis=ids)
        Data = Data.loc[factor_names, :, ids]
        Data.major_axis = [dt.datetime.combine(iDate, dt.time(0)) for iDate in Data.major_axis]
        Data.fillna(value=0, inplace=True)
        return adjustDateTime(Data, dts, fillna=True, method="bfill")
class _MappingTable(_DBTable):
    """映射因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._StartDateField = FactorInfo[FactorInfo["FieldType"]=="StartDate"].index[0]
        self._EndDateField = FactorInfo[FactorInfo["FieldType"]=="EndDate"].index[0]
        self._EndDateIncluded = FactorInfo[FactorInfo["FieldType"]=="EndDate"]["Supplementary"].iloc[0]
        self._EndDateIncluded = (pd.isnull(self._EndDateIncluded) or (self._EndDateIncluded=="包含"))
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    # 返回给定时点 idt 有数据的所有 ID
    # 如果 idt 为 None, 将返回所有有记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._StartDateField, self._EndDateField]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._StartDateField]+"<='"+idt.strftime("%Y%m%d")+"' "
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._EndDateField]+">='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回给定 ID iid 的起始日期距今的时点序列
    # 如果 idt 为 None, 将以表中最小的起始日期作为起点
    # 忽略 ifactor_name    
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._StartDateField]]
        SQLStr = "SELECT MIN("+DBTableName+"."+FieldDict[self._StartDateField]+") "# 起始日期
        SQLStr += "FROM "+DBTableName
        if iid is not None: SQLStr += " WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"'"
        StartDT = dt.datetime.strptime(self._FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d")
        if start_dt is not None: StartDT = max((StartDT, start_dt))
        if end_dt is None: end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1))
        # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._StartDateField, self._EndDateField]+factor_names]
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._StartDateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._EndDateField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += 'WHERE ('+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+') '
        SQLStr += "AND (("+DBTableName+"."+FieldDict[self._EndDateField]+">='"+StartDate.strftime("%Y%m%d")+"') "
        SQLStr += "OR ("+DBTableName+"."+FieldDict[self._EndDateField]+" IS NULL)) "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._StartDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += 'ORDER BY '+DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._StartDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", self._StartDateField, self._EndDateField]+factor_names)
        else: return pd.DataFrame(np.array(RawData), columns=["ID", self._StartDateField, self._EndDateField]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data[self._EndDateField] = raw_data[self._EndDateField].where(pd.notnull(raw_data[self._EndDateField]), dt.date.today().strftime("%Y%m%d"))
        raw_data.set_index(["ID"], inplace=True)
        DeltaDT = dt.timedelta(int(not self._EndDateIncluded))
        Data, nFactor = {}, len(factor_names)
        for iID in raw_data.index.unique():
            iRawData = raw_data.loc[[iID]]
            iData = pd.DataFrame(index=dts, columns=factor_names)
            for j in range(iRawData.shape[0]):
                ijRawData = iRawData.iloc[j]
                jStartDate, jEndDate = dt.datetime.strptime(ijRawData[self._StartDateField], "%Y%m%d"), dt.datetime.strptime(ijRawData[self._EndDateField], "%Y%m%d")-DeltaDT
                iData.loc[jStartDate:jEndDate] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iData.loc[jStartDate:jEndDate].shape[0], axis=0)
            Data[iID] = iData
        return pd.Panel(Data).swapaxes(0, 2).loc[:, :, ids]

class _IndustryTable(_DBTable):
    """行业因子表"""
    Level = Enum(1, 2, 3, 4, label="分类级别", arg_type="SingleOption", order=0)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IndustryCodeField = FactorInfo[FactorInfo["FieldType"]=="IndustryCode"].index[0]
        self._IndustryCodeStart = FactorInfo[FactorInfo["FieldType"]=="IndustryCode"]["Supplementary"].iloc[0][1:]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._InDateField = FactorInfo[FactorInfo["FieldType"]=="InDate"].index[0]
        self._OutDateField = FactorInfo[FactorInfo["FieldType"]=="OutDate"].index[0]
        if fdb.DBType=="Oracle": self._SubStrFun = 'SUBSTR'
        else: self._SubStrFun = 'SUBSTRING'
        self._DataType = pd.Series("string", index=["行业名称", "行业代码"])
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        return ["行业名称", "行业代码"]
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        if key=="DataType": return self._DataType.loc[factor_names]
        elif key is None: return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType")})
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 返回在给定时点 idt 的有行业分类的证券
    # 如果 idt 为 None, 将返回所有曾经有行业分类的证券
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        IndustryLevel = args.get("分类级别", self.Level)
        # 获取行业代码列表
        SubSQLStr = "SELECT "+self._SubStrFun+"("+self._FactorDB.TablePrefix+"AShareIndustriesCode.industriescode,1,"+str(2*(IndustryLevel+1))+") "
        SubSQLStr += "FROM "+self._FactorDB.TablePrefix+"AShareIndustriesCode "
        SubSQLStr += "WHERE "+self._FactorDB.TablePrefix+"AShareIndustriesCode.industriescode LIKE '"+self._IndustryCodeStart+"%' "
        SubSQLStr += "AND "+self._FactorDB.TablePrefix+"AShareIndustriesCode.levelnum="+str(IndustryLevel+1)
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._IndustryCodeField, self._InDateField, self._OutDateField]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]# ID
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE "+self._SubStrFun+"("+DBTableName+"."+FieldDict[self._IndustryCodeField]+",1,"+str(2*(IndustryLevel+1))+") IN ("+SubSQLStr+") "
        if idt is not None:
            idt = idt.strftime("%Y%m%d")
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+idt+"' "
            SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+idt+"') "
            SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回给定 iid 的证券有行业分类的时间点序列
    # 如果 iid 为 None, 将返回表里有记录的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        IndustryLevel = args.get("分类级别", self.Level)
        # 获取行业代码列表
        SubSQLStr = "SELECT "+self._SubStrFun+"("+self._FactorDB.TablePrefix+"AShareIndustriesCode.industriescode,1,"+str(2*(IndustryLevel+1))+") "
        SubSQLStr += "FROM "+self._FactorDB.TablePrefix+"AShareIndustriesCode "
        SubSQLStr += "WHERE "+self._FactorDB.TablePrefix+"AShareIndustriesCode.industriescode LIKE '"+self._IndustryCodeStart+"%' "
        SubSQLStr += "AND "+self._FactorDB.TablePrefix+"AShareIndustriesCode.levelnum="+str(IndustryLevel+1)
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._IndustryCodeField, self._InDateField, self._OutDateField]]
        if iid is not None:
            SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._InDateField]+" "# 纳入日期
            SQLStr += DBTableName+"."+FieldDict[self._OutDateField]+" "# 剔除日期
            SQLStr += "FROM "+DBTableName+" "
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._InDateField]+" IS NOT NULL "
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._IndustryCodeField]+" IN ("+SubSQLStr+") "
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
            if start_dt is not None:
                SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+start_dt.strftime("%Y%m%d")+"') "
                SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL))"
            if end_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._InDateField]
            Data = self._FactorDB.fetchall(SQLStr)
            DateTimes = set()
            for iStartDate, iEndDate in Data:
                iStartDT = dt.datetime.strptime(iStartDate, "%Y%m%d")
                if iEndDate is None: iEndDT = (dt.datetime.now() if end_dt is None else end_dt)
                else: iEndDT = dt.datetime.strptime(iEndDate, "%Y%m%d")
                DateTimes = DateTimes.union(getDateTimeSeries(start_dt=iStartDT, end_dt=iEndDT, timedelta=dt.timedelta(1)))
            return sorted(DateTimes)
        SQLStr = "SELECT MIN("+DBTableName+"."+FieldDict[self._InDateField]+") "# 纳入日期
        SQLStr += "FROM "+DBTableName
        SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IndustryCodeField]+" IN ("+SubSQLStr+") "
        StartDT = dt.datetime.strptime(self._FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d")
        if start_dt is not None: StartDT = max((StartDT, start_dt))
        if end_dt is None: end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1))
    def __QS_genGroupInfo__(self, factors, operation_mode):
        LevelGroup = {}
        StartDT = dt.datetime.now()
        FactorNames, RawFactorNames = [], set()
        for iFactor in factors:
            iLevel = iFactor.Level
            if iLevel not in LevelGroup:
                LevelGroup[iLevel] = {"FactorNames":[iFactor.Name], 
                                      "RawFactorNames":{iFactor._NameInFT}, 
                                      "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                      "args":{"分类级别":iLevel}}
            else:
                LevelGroup[iLevel]["FactorNames"].append(iFactor.Name)
                LevelGroup[iLevel]["RawFactorNames"].add(iFactor._NameInFT)
                LevelGroup[iLevel]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], LevelGroup[iLevel]["StartDT"])
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iLevel in LevelGroup:
            StartInd = operation_mode.DTRuler.index(LevelGroup[iLevel]["StartDT"])
            Groups.append((self, LevelGroup[iLevel]["FactorNames"], list(LevelGroup[iLevel]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], LevelGroup[iLevel]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IndustryLevel = args.get("分类级别", self.Level)
        # 获取行业代码对应的行业名称
        SQLStr = "SELECT "+self._SubStrFun+"("+self._FactorDB.TablePrefix+"AShareIndustriesCode.industriescode,1,"+str(2*(IndustryLevel+1))+"), "
        SQLStr += self._FactorDB.TablePrefix+"AShareIndustriesCode.industriesname "
        SQLStr += "FROM "+self._FactorDB.TablePrefix+"AShareIndustriesCode "
        SQLStr += "WHERE "+self._FactorDB.TablePrefix+"AShareIndustriesCode.industriescode LIKE '"+self._IndustryCodeStart+"%' "
        SQLStr += "AND "+self._FactorDB.TablePrefix+"AShareIndustriesCode.levelnum="+str(IndustryLevel+1)
        IndustryCodeName = {iCode:iName for iCode, iName in self._FactorDB.fetchall(SQLStr)}
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._IndustryCodeField, self._InDateField, self._OutDateField]]
        # ID, 行业分类代码, 纳入日期, 剔除日期
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += self._SubStrFun+"("+DBTableName+"."+FieldDict[self._IndustryCodeField]+",1,"+str(2*(IndustryLevel+1))+"), "
        SQLStr += DBTableName+"."+FieldDict[self._InDateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._OutDateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE "+self._SubStrFun+"("+DBTableName+"."+FieldDict[self._IndustryCodeField]+",1,"+str(2*(IndustryLevel+1))+") IN ('"+"', '".join(IndustryCodeName.keys())+"') "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+StartDate.strftime("%Y%m%d")+"') "
        SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "        
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self._InDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["ID", "行业代码", self._InDateField, self._OutDateField])
        else: RawData = pd.DataFrame(np.array(RawData), columns=["ID", "行业代码", self._InDateField, self._OutDateField])
        RawData["行业名称"] = RawData["行业代码"]
        for iCode, iName in IndustryCodeName.items(): RawData["行业名称"][RawData["行业代码"]==iCode] = iName
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(np.full(shape=(len(factor_names), len(dts), len(ids)), fill_value=None, dtype="O"), items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data[self._OutDateField] = raw_data[self._OutDateField].where(pd.notnull(raw_data[self._OutDateField]), dt.date.today().strftime("%Y%m%d"))
        raw_data.set_index(["ID"], inplace=True)
        DeltaDT = dt.timedelta(1)
        Data, nFactor = {}, len(factor_names)
        for iID in raw_data.index.unique():
            iRawData = raw_data.loc[[iID]]
            iData = pd.DataFrame(index=dts, columns=factor_names, dtype="O")
            for j in range(iRawData.shape[0]):
                ijRawData = iRawData.iloc[j]
                jStartDate, jEndDate = dt.datetime.strptime(ijRawData[self._InDateField], "%Y%m%d"), dt.datetime.strptime(ijRawData[self._OutDateField], "%Y%m%d")-DeltaDT
                iData.loc[jStartDate:jEndDate] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iData.loc[jStartDate:jEndDate].shape[0], axis=0)
            Data[iID] = iData
        return pd.Panel(Data).swapaxes(0, 2).loc[:, :, ids]

class _FeatureTable(_DBTable):
    """特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField]]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
        # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField]+factor_names]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成SQL语句, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data = raw_data.set_index(["ID"])
        if raw_data.index.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        Data = pd.Panel(raw_data.values.T.reshape((raw_data.shape[1], raw_data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=raw_data.index, minor_axis=dts).swapaxes(1, 2)
        return Data.loc[:, :, ids]
# 财务因子表, 表结构特征:
# 报告期字段, 表示财报的报告期
# 公告日期字段, 表示财报公布的日期
class _FinancialTable(_DBTable):
    """财务因子表"""
    ReportDate = Enum("所有", "年报", "中报", "一季报", "三季报", Dict(), Function(), label="报告期", arg_type="SingleOption", order=0)
    ReportType = List(["408001000", "408004000", "408005000"], label="报表类型", arg_type="MultiOption", order=1, option_range=("408001000", "408004000", "408005000"))
    CalcType = Enum("最新", "单季度", "TTM", label="计算方法", arg_type="SingleOption", order=2)
    YearLookBack = Int(0, label="回溯年数", arg_type="Integer", order=3)
    PeriodLookBack = Int(0, label="回溯期数", arg_type="Integer", order=4)
    ExprFactor = Str("", label="业绩快报因子", arg_type="String", order=5)
    NoticeFactor = Str("", label="业绩预告因子", arg_type="String", order=6)
    IgnoreMissing = Bool(True, label="忽略缺失", arg_type="Bool", order=7)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._ANNDateField = FactorInfo[FactorInfo["FieldType"]=="ANNDate"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._ReportTypeField = FactorInfo[FactorInfo["FieldType"]=="ReportType"].index
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
        if self._ReportTypeField is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._ReportTypeField]+" IN ('"+"','".join(args.get("报表类型", self.ReportType))+"')"
        SQLStr = "SELECT t.* FROM ("+SQLStr+") t WHERE t.ANNDate IS NOT NULL "
        SQLStr += "ORDER BY t."+FieldDict[self._IDField]+", t.ANNDate, t."+FieldDict[self._ReportDateField]
        if self._ReportTypeField is not None: SQLStr += ", t."+FieldDict[self._ReportTypeField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData), columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        # 拼接业绩快报数据
        ExprFactor = args.get("业绩快报因子", self.ExprFactor)
        if ExprFactor:
            SQLStr = self._genExpressSQLStr(ExprFactor, ids)
            ExpressRawData = self._FactorDB.fetchall(SQLStr)
            if not ExpressRawData: ExpressRawData = pd.DataFrame(columns=['ID', 'AnnDate', 'ReportDate', ExprFactor])
            else: ExpressRawData = pd.DataFrame(np.array(ExpressRawData), columns=['ID', 'AnnDate', 'ReportDate', ExprFactor])
            ExpressRawData["ReportType"] = "1ProfitExpress"
            for iField in factor_names: ExpressRawData[iField] = ExpressRawData[ExprFactor]
            ExpressRawData.pop(ExprFactor)
            RawData = pd.concat([RawData, ExpressRawData])
        NoticeFactor = args.get("业绩预告因子", self.NoticeFactor)
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
        CalcType, YearLookBack, PeriodLookBack, ReportDate, IgnoreMissing = args.get("计算方法", self.CalcType), args.get("回溯年数", self.YearLookBack), args.get("回溯期数", self.PeriodLookBack), args.get("报告期", self.ReportDate), args.get("忽略缺失", self.IgnoreMissing)
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
        Data = pd.Panel(Data)
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
        Data = adjustDateTime(pd.Panel(NewData).loc[factor_names], dts, fillna=True, method="pad")
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

class _AnalystConsensusTable(_DBTable):
    """分析师汇总表"""
    CalcType = Enum("FY0", "FY1", "FY2", "Fwd12M", label="计算方法", arg_type="SingleOption", order=0)
    Period = Enum("263001000", "263002000", "263003000", "263004000", label="周期", arg_type="SingleOption", order=1)
    LookBack = Int(180, arg_type="Integer", label="回溯天数", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._PeriodField = FactorInfo[FactorInfo["FieldType"]=="Period"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = 'W2财务年报-公告日期'
        self._ANN_ReportFileSuffix = getShelveFileSuffix()
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        CalcType = args.get("计算方法", self.CalcType)
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._DateField, self._ReportDateField, self._PeriodField]+factor_names]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成SQL语句, 日期, ID, 报告期, 数据
        SQLStr = 'SELECT '+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._ReportDateField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._PeriodField]+"='"+args.get("周期", self.Period)+"' "
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
        StartDT = dt.datetime.now()
        FactorNames, RawFactorNames = [], set()
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
        if raw_data.shape[0]==0: return pd.Panel(np.nan, items=factor_names, major_axis=dts, minor_axis=ids)
        Dates = sorted({iDT.strftime("%Y%m%d") for iDT in dts})
        CalcType, LookBack = args.get("计算方法", self.CalcType), args.get("回溯天数", self.LookBack)
        if CalcType=="Fwd12M":
            CalcFun, FYNum, ANNReportData = self._calcIDData_Fwd12M, None, None
        else:
            CalcFun, FYNum = self._calcIDData_FY, int(CalcType[-1])
            ANNReportPath = raw_data.columns.name
            if (ANNReportPath is not None) and os.path.isfile(ANNReportPath+("."+self._ANN_ReportFileSuffix if self._ANN_ReportFileSuffix else "")):
                with shelve.open(ANNReportPath) as ANN_ReportFile:
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
        Data = pd.Panel(Data, minor_axis=factor_names)
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
class _AnalystRatingDetailTable(_DBTable):
    """分析师投资评级明细表"""
    Operator = Function(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
    ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
    AdditionalFields = ListStr(arg_type="MultiOption", label="附加字段", order=2, option_range=())
    Deduplication = ListStr(arg_type="MultiOption", label="去重字段", order=3, option_range=())
    Period = Int(180, arg_type="Integer", label="周期", order=4)
    DataType = Enum("double", "string", arg_type="SingleOption", label="数据类型", order=5)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._InstituteField = FactorInfo[FactorInfo["FieldType"]=="Institute"].index[0]
        self._AnalystField = FactorInfo[FactorInfo["FieldType"]=="Analyst"].index[0]
        self._TempData = {}
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.Deduplication = [self._InstituteField]
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
        StartDate -= dt.timedelta(args.get("周期", self.Period))
        AdditiveFields = args.get("附加字段", self.AdditionalFields)
        DeduplicationFields = args.get("去重字段", self.Deduplication)
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
        if raw_data.shape[0]==0: return pd.Panel(np.nan, items=factor_names, major_axis=dts, minor_axis=ids)
        Dates = sorted({dt.datetime.combine(iDT.date(), dt.time(0)) for iDT in dts})
        DeduplicationFields = args.get("去重字段", self.Deduplication)
        AdditionalFields = list(set(args.get("附加字段", self.AdditionalFields)+DeduplicationFields))
        AllFields = list(set(factor_names+AdditionalFields))
        raw_data = raw_data.loc[:, ["日期", "ID"]+AllFields].set_index(["ID"])
        Period = args.get("周期", self.Period)
        ModelArgs = args.get("参数", self.ModelArgs)
        Operator = args.get("算子", self.Operator)
        DataType = args.get("数据类型", self.DataType)
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
        return pd.Panel(Data, major_axis=Dates, minor_axis=ids).loc[factor_names, dts]

class _AnalystEstDetailTable(_DBTable):
    """分析师盈利预测明细表"""
    Operator = Function(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
    ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
    ForwardYears = List(default=[0], label="向前年数", arg_type="ArgList", order=2)
    AdditionalFields = ListStr(arg_type="MultiOption", label="附加字段", order=3, option_range=())
    Deduplication = ListStr(arg_type="MultiOption", label="去重字段", order=4, option_range=())
    Period = Int(180, arg_type="Integer", label="周期", order=5)
    DataType = Enum("double", "string", arg_type="SingleOption", label="数据类型", order=6)
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
        self._ANN_ReportFileSuffix = getShelveFileSuffix()
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.Deduplication = [self._InstituteField]
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
        StartDate -= dt.timedelta(args.get("周期", self.Period))
        AdditiveFields = args.get("附加字段", self.AdditionalFields)
        DeduplicationFields = args.get("去重字段", self.Deduplication)
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
        DeduplicationFields = args.get("去重字段", self.Deduplication)
        AdditionalFields = list(set(args.get("附加字段", self.AdditionalFields)+DeduplicationFields))
        AllFields = list(set(factor_names+AdditionalFields))
        ANNReportPath = raw_data.columns.name
        raw_data = raw_data.loc[:, ["日期", "ID", self._ReportDateField, self._CapitalField]+AllFields].set_index(["ID"])
        Period = args.get("周期", self.Period)
        ForwardYears = args.get("向前年数", self.ForwardYears)
        ModelArgs = args.get("参数", self.ModelArgs)
        Operator = args.get("算子", self.Operator)
        DataType = args.get("数据类型", self.DataType)
        if (ANNReportPath is not None) and os.path.isfile(ANNReportPath+("."+self._ANN_ReportFileSuffix if self._ANN_ReportFileSuffix else "")):
            with shelve.open(ANNReportPath) as ANN_ReportFile:
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
        return pd.Panel(Data, major_axis=Dates, minor_axis=ids).loc[factor_names, dts]
# 公告信息表, 表结构特征:
# 公告日期, 表示获得信息的时点;
# 截止日期, 表示信息有效的时点, 该字段可能没有;
# 如果存在截止日期, 以截止日期和公告日期的最大值作为数据填充的时点; 如果不存在截止日期, 以公告日期作为数据填充的时点;
# 数据填充时点和 ID 不能唯一标志一行记录, 对于每个 ID 每个数据填充时点可能存在多个数据, 将所有的数据以 list 组织, 如果算子参数不为 None, 以该算子作用在数据 list 上的结果为最终填充结果, 否则以数据 list 填充;
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class _AnnTable(_DBTable):
    """公告信息表"""
    #ANNDate = Enum(None, arg_type="SingleOption", label="公告日期", order=0)
    Operator = Function(lambda x: x.tolist(), arg_type="Function", label="算子", order=1)
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._AnnDateField = FactorInfo[FactorInfo["FieldType"]=="ANNDate"].index[0]# 所有的公告日期
        self._EndDateField = FactorInfo[FactorInfo["FieldType"]=="EndDate"].index# 截止日期
        self._EndDateField = (self._EndDateField[0] if self._EndDateField.shape[0]>0 else None)
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+2))
            self[iCondition] = str(FactorInfo.loc[iCondition, "Supplementary"])
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()+[self._AnnDateField, self._EndDateField]
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._AnnDateField, self._EndDateField, self._IDField]+self._ConditionFields]
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if idt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._AnnDateField]+"<='"+idt.strftime("%Y%m%d")+"' "
            if self._EndDateField is not None:
                SQLStr += "AND "+DBTableName+"."+FieldDict[self._EndDateField]+"<='"+idt.strftime("%Y%m%d")+"' "
        for iConditionField in self._ConditionFields:
            if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+str(args.get(iConditionField, self[iConditionField]))+"' "
            else:
                SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+str(args.get(iConditionField, self[iConditionField]))+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._AnnDateField, self._EndDateField, self._IDField]+self._ConditionFields]
        if self._EndDateField is not None:
            SQLStr = "SELECT DISTINCT CASE WHEN "+FieldDict[self._AnnDateField]+">="+FieldDict[self._EndDateField]+" THEN "+FieldDict[self._AnnDateField]+" "
            SQLStr += "WHEN "+FieldDict[self._AnnDateField]+"<"+FieldDict[self._EndDateField]+" THEN "+FieldDict[self._EndDateField]+" END AS DT "
            SQLStr += "FROM "+DBTableName+" "
            if iid is not None: SQLStr += "WHERE "+FieldDict[self._IDField]+"='"+iid+"' "
            else: SQLStr += "WHERE "+FieldDict[self._IDField]+" IS NOT NULL "
            for iConditionField in self._ConditionFields:
                if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                    SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+str(args.get(iConditionField, self[iConditionField]))+"' "
                else:
                    SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+str(args.get(iConditionField, self[iConditionField]))+" "
            SQLStr = "SELECT t.DT FROM ("+SQLStr+") t "
            if start_dt is not None: SQLStr += "AND t.DT>='"+start_dt.strftime("%Y%m%d")+"' "
            if end_dt is not None: SQLStr += "AND t.DT<='"+end_dt.strftime("%Y%m%d")+"' "
            SQLStr += "ORDER BY t.DT"
        else:
            SQLStr = "SELECT DISTINCT "+FieldDict[self._AnnDateField]+" FROM "+DBTableName+" "
            if iid is not None: SQLStr += "WHERE "+FieldDict[self._IDField]+"='"+iid+"' "
            else: SQLStr += "WHERE "+FieldDict[self._IDField]+" IS NOT NULL "
            if start_dt is not None: SQLStr += "AND "+FieldDict[self._AnnDateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
            if end_dt is not None: SQLStr += "AND "+FieldDict[self._AnnDateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
            for iConditionField in self._ConditionFields:
                if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                    SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+str(args.get(iConditionField, self[iConditionField]))+"' "
                else:
                    SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+str(args.get(iConditionField, self[iConditionField]))+" "
            SQLStr += "ORDER BY "+FieldDict[self._AnnDateField]
        return list(map(lambda x: dt.datetime.strptime(x[0], "%Y%m%d"), self._FactorDB.fetchall(SQLStr)))
    def getCondition(self, icondition, ids=None, dts=None):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._AnnDateField, self._EndDateField, self._IDField, icondition]]
        if (self._EndDateField is None) or (dts is None):
            SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[icondition]+" "
            SQLStr += "FROM "+DBTableName+" "
            if ids is not None: SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
            else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
            if dts is not None:
                Dates = list({iDT.strftime("%Y%m%d") for iDT in dts})
                SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict[self._AnnDateField], Dates, is_str=True, max_num=1000)+") "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[icondition]
        else:
            SQLStr = "SELECT CASE WHEN "+FieldDict[self._AnnDateField]+">="+FieldDict[self._EndDateField]+" THEN "+FieldDict[self._AnnDateField]+" "
            SQLStr += "WHEN "+FieldDict[self._AnnDateField]+"<"+FieldDict[self._EndDateField]+" THEN "+FieldDict[self._EndDateField]+" END AS DT, "
            SQLStr += FieldDict[icondition]+" "
            SQLStr += "FROM "+DBTableName
            if ids is not None: SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
            else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
            Dates = list({iDT.strftime("%Y%m%d") for iDT in dts})
            SQLStr += "AND ("+genSQLInCondition("DT", Dates, is_str=True, max_num=1000)+") "
            SQLStr = "SELECT DISTINCT t."+FieldDict[icondition]+" FROM ("+SQLStr+") t "
            SQLStr += "ORDER BY t."+FieldDict[icondition]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+iFactor[iArgName] for iArgName in iFactor.ArgNames if iArgName not in ("回溯天数", "算子")])
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {"FactorNames":[iFactor.Name], 
                                               "RawFactorNames":{iFactor._NameInFT}, 
                                               "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                               "args":iFactor.Args.copy()}
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
                ConditionGroup[iConditions]["args"]["回溯天数"] = max(ConditionGroup[iConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._AnnDateField, self._EndDateField, self._IDField]+self._ConditionFields+factor_names]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        if self._EndDateField is not None:
            SubSQLStr = "SELECT "+FieldDict[self._IDField]+", "
            SubSQLStr += FieldDict[self._AnnDateField]+", "
            SubSQLStr += FieldDict[self._EndDateField]+", "
            SubSQLStr += "CASE WHEN "+FieldDict[self._AnnDateField]+">="+FieldDict[self._EndDateField]+" THEN "+FieldDict[self._AnnDateField]+" "
            SubSQLStr += "WHEN "+FieldDict[self._AnnDateField]+"<"+FieldDict[self._EndDateField]+" THEN "+FieldDict[self._EndDateField]+" END AS DT, "
            for iField in factor_names: SubSQLStr += FieldDict[iField]+", "
            SubSQLStr = SubSQLStr[:-2]+" FROM "+DBTableName+" "
            SubSQLStr1 = "SELECT "+FieldDict[self._IDField]+", "
            SubSQLStr1 += FieldDict[self._AnnDateField]+", "
            SubSQLStr1 += "MAX("+FieldDict[self._EndDateField]+") AS EndDT "
            SubSQLStr1 += "FROM "+DBTableName+" "
            SubSQLStr1 += "WHERE ("+genSQLInCondition(FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
            for iConditionField in self._ConditionFields:
                if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                    SubSQLStr1 += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+str(args.get(iConditionField, self[iConditionField]))+"' "
                else:
                    SubSQLStr1 += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+str(args.get(iConditionField, self[iConditionField]))+" "            
            SubSQLStr1 += "GROUP BY "+FieldDict[self._IDField]+", "+FieldDict[self._AnnDateField]
            SQLStr = "SELECT t."+FieldDict[self._IDField]+", "
            SQLStr += "t.DT, "
            for iField in factor_names: SQLStr += "t."+FieldDict[iField]+", "
            SQLStr = SQLStr[:-2]+" FROM ("+SubSQLStr+") t "
            SQLStr += "INNER JOIN ("+SubSQLStr1+") t1 "
            SQLStr += "ON (t."+FieldDict[self._IDField]+"=t1."+FieldDict[self._IDField]+" "
            SQLStr += "AND t."+FieldDict[self._AnnDateField]+"=t1."+FieldDict[self._AnnDateField]+" "
            SQLStr += "AND t."+FieldDict[self._EndDateField]+"=t1.EndDT) "
            SQLStr += "WHERE t.DT>='"+StartDate.strftime("%Y%m%d")+"' "
            SQLStr += "AND t.DT<='"+EndDate.strftime("%Y%m%d")+"' "
            SQLStr += "ORDER BY t."+FieldDict[self._IDField]+", t.DT"
        else:
            SQLStr = "SELECT "+FieldDict[self._IDField]+", "
            SQLStr += FieldDict[self._AnnDateField]+", "
            for iField in factor_names: SQLStr += FieldDict[iField]+", "
            SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
            SQLStr += "WHERE ("+genSQLInCondition(FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
            SQLStr += "AND "+FieldDict[self._AnnDateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
            SQLStr += "AND "+FieldDict[self._AnnDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
            for iConditionField in self._ConditionFields:
                if FactorInfo.loc[iConditionField, "DataType"].find("CHAR")!=-1:
                    SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+str(args.get(iConditionField, self[iConditionField]))+"' "
                else:
                    SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"="+str(args.get(iConditionField, self[iConditionField]))+" "
            SQLStr += "ORDER BY "+FieldDict[self._IDField]+", "+FieldDict[self._AnnDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "日期"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["ID", "日期"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Operator = args.get("算子", self.Operator)
        if Operator is None: Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        Data = pd.Panel(Data).loc[factor_names, :, ids]
        Data.major_axis = [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Data.major_axis]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]

class WindDB2(FactorDB):
    """Wind 量化研究数据库"""
    DBType = Enum("SQL Server", "Oracle", "MySQL", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("wind", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=1521, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"WindDB2Config.json" if config_file is None else config_file), **kwargs)
        self._Connection = None# 数据库链接
        self._AllTables = []# 数据库中的所有表名, 用于查询时解决大小写敏感问题
        self._InfoFilePath = __QS_LibPath__+os.sep+"WindDB2Info.hdf5"# 数据库信息文件路径
        self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"WindDB2Info.xlsx"# 数据库信息源文件路径
        self._TableInfo, self._FactorInfo = updateInfo(self._InfoFilePath, self._InfoResourcePath)# 数据库表信息, 数据库字段信息
        self._PID = None# 保存数据库连接创建时的进程号
        self.Name = "WindDB2"
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state["_Connection"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._Connection:
            self.connect()
        else:
            self._Connection = None
        self._AllTables = state.get("_AllTables", [])
    # -------------------------------------------数据库相关---------------------------
    def connect(self):
        self._Connection = None
        if (self.Connector=='cx_Oracle') or ((self.Connector=='default') and (self.DBType=='Oracle')):
            try:
                import cx_Oracle
                self._Connection = cx_Oracle.connect(self.User, self.Pwd, cx_Oracle.makedsn(self.IPAddr, str(self.Port), self.DBName))
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='pymssql') or ((self.Connector=='default') and (self.DBType=='SQL Server')):
            try:
                import pymssql
                self._Connection = pymssql.connect(server=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                if self.Connector!='default': raise e
        elif (self.Connector=='mysql.connector') or ((self.Connector=='default') and (self.DBType=='MySQL')):
            try:
                import mysql.connector
                self._Connection = mysql.connector.connect(host=self.IPAddr, port=str(self.Port), user=self.User, password=self.Pwd, database=self.DBName, charset=self.CharSet)
            except Exception as e:
                if self.Connector!='default': raise e
        if self._Connection is None:
            if self.Connector not in ('default', 'pyodbc'):
                self._Connection = None
                raise __QS_Error__("不支持该连接器(connector) : "+self.Connector)
            else:
                import pyodbc
                if self.DSN:
                    self._Connection = pyodbc.connect('DSN=%s;PWD=%s' % (self.DSN, self.Pwd))
                else:
                    self._Connection = pyodbc.connect('DRIVER={%s};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s' % (self.DBType, self.DBName, self.IPAddr+","+str(self.Port), self.User, self.Pwd))
        self._Connection.autocommit = True
        self._AllTables = []
        self._PID = os.getpid()
        return 0
    def disconnect(self):
        if self._Connection is not None:
            try:
                self._Connection.close()
            except Exception as e:
                raise e
            finally:
                self._Connection = None
        return 0
    def isAvailable(self):
        return (self._Connection is not None)
    def cursor(self, sql_str=None):
        if self._Connection is None: raise __QS_Error__("%s尚未连接!" % self.__doc__)
        if os.getpid()!=self._PID: self.connect()# 如果进程号发生变化, 重连
        Cursor = self._Connection.cursor()
        if sql_str is None: return Cursor
        if not self._AllTables:
            if self.DBType=="SQL Server":
                Cursor.execute("SELECT Name FROM SysObjects Where XType='U'")
                self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
            elif self.DBType=="MySQL":
                Cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='"+self.DBName+"' AND table_type='base table'")
                self._AllTables = [rslt[0] for rslt in Cursor.fetchall()]
        for iTable in self._AllTables:
            sql_str = re.sub(iTable, iTable, sql_str, flags=re.IGNORECASE)
        Cursor.execute(sql_str)
        return Cursor
    def fetchall(self, sql_str):
        Cursor = self.cursor(sql_str=sql_str)
        Data = Cursor.fetchall()
        Cursor.close()
        return Data
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        if self._TableInfo is not None: return self._TableInfo.index.tolist()
        else: return []
    def getTable(self, table_name, args={}):
        if table_name in self._TableInfo.index:
            TableClass = self._TableInfo.loc[table_name, "TableClass"]
            if pd.notnull(TableClass):
                return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=args)")
        raise __QS_Error__("因子库目前尚不支持表: '%s'" % table_name)
    # -----------------------------------------数据提取---------------------------------
    # 给定起始日期和结束日期, 获取交易所交易日期, 目前支持: "SSE", "SZSE", "SHFE", "DCE", "CZCE", "INE", "CFEEX"
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if start_date is None: start_date = dt.datetime(1900, 1, 1)
        if end_date is None: end_date = dt.datetime.today()
        ExchangeInfo = self._TableInfo[self._TableInfo["TableClass"]=="CalendarTable"]
        ExchangeInfo = ExchangeInfo[ExchangeInfo["Description"].str.contains(exchange)]
        if ExchangeInfo.shape[0]==0: raise __QS_Error__("不支持交易所: '%s' 的交易日序列!" % exchange)
        else: Dates = self.getTable(ExchangeInfo.index[0]).getDateTime(iid=exchange, start_dt=start_date, end_dt=end_date)
        if kwargs.get("output_type", "date")=="date": return list(map(lambda x: x.date(), Dates))
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