# coding=utf-8
"""Wind 量化研究数据库"""
import os
import shelve
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Int, Str, Range, Bool, List, Dict, Function

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries, getDateSeries
from QuantStudio.Tools.FileFun import readJSONFile
from QuantStudio import __QS_Error__, __QS_LibPath__
from QuantStudio.FactorDataBase.WindDB import WindDB, _DBTable, _adjustDateTime
from QuantStudio.FactorDataBase.WindDB import _MarketTable as WindMarketTable

def fillna(df, limit_ns):
    Ind = pd.DataFrame(np.r_[0, np.diff(df.index.values).astype("float")].reshape((df.shape[0], 1)).repeat(df.shape[1], axis=1).cumsum(axis=0))
    Ind1 = Ind.where(pd.notnull(df.values), other=np.nan)
    Ind1.fillna(method="pad", inplace=True)
    df = df.fillna(method="pad")
    df.where((Ind.values-Ind1.values<=limit_ns), np.nan, inplace=True)
    return df

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
# 查找某个报告期对应的公告期
def findNoteDate(report_date, report_note_dates):
    for i in range(0, report_note_dates.shape[0]):
        if report_date==report_note_dates['报告期'].iloc[i]: return report_note_dates['公告日期'].iloc[i]
    return None
# f: 该算子所属的因子, 因子对象
# idt: 当前所处的时点
# iid: 当前待计算的 ID
# x: 描述子当期的数据, [DataFrame(columns=['预测日期', '报告期', '研究机构名称', '分析师名称', '预测基准股本(万股)']+SysArgs['字段'])], list的长度为向前年数
# args: 参数, {参数名:参数值}
def _DefaultOperator(f, idt, iid, x, args):
    return np.nan

class _CalendarTable(_DBTable):
    """交易日历因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    @property
    def FactorNames(self):
        return ["交易日"]
    # 返回给定时点 idt 有交易的交易所列表
    # 如果 idt 为 None, 将返回表中的所有交易所
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._DateField]+"='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回交易所为 iid 的交易日列表
    # 如果 iid 为 None, 将返回表中有记录数据的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField])
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else:
            SQLStr = "SELECT DISTINCT "+SQLStr[7:]+"WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._DateField]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self._FactorDB.fetchall(SQLStr)))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField])
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+dts[0].strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+dts[-1].strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]: return pd.DataFrame(columns=["日期", "ID"])
        return pd.DataFrame(np.array(RawData), columns=["日期", "ID"])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data["交易日"] = 1
        Data = pd.Panel({"交易日":raw_data.set_index(["日期", "ID"])["交易日"].unstack()}).loc[factor_names]
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]), 23, 59, 59, 999999) for iDate in Data.major_axis]
        return _adjustDateTime(Data, dts, fillna=True, value=0)

class _MarketTable(_DBTable):
    """行情因子表"""
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=0)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        for i, iCondition in enumerate(self._ConditionFields): self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+2))
    def getCondition(self, icondition, ids=None, dts=None):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField, icondition])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[icondition]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if dts is not None:
            Dates = list({iDT.strftime("%Y%m%d") for iDT in dts})
            SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict[self._DateField], Dates, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[icondition]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField]+self._ConditionFields)
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._DateField]+"='"+idt.strftime("%Y%m%d")+"' "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._DateField]+" IS NOT NULL "
        for iConditionField in self._ConditionFields: SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField]+self._ConditionFields)
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._DateField]+" "# 日期
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        for iConditionField in self._ConditionFields: SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._DateField]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self._FactorDB.fetchall(SQLStr)))
        # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField]+self._ConditionFields+factor_names)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        for iConditionField in self._ConditionFields: SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self._DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["日期", "ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data = raw_data.set_index(["日期", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]), 23, 59, 59, 999999) for iDate in Data.major_axis]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.ix[:, dts, ids]
        AllDTs = Data.major_axis.union(set(dts)).sort_values()
        Data = Data.ix[:, AllDTs, ids]
        LimitNs = LookBack*24.0*3600*10**9
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillna(Data.iloc[i], limit_ns=LimitNs)
        return Data.loc[:, dts]

class _ConstituentTable(_DBTable):
    """成份因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
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
            DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
            FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._GroupField])
            SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._GroupField]+" "# 指数 ID
            SQLStr += "FROM "+DBTableName+" "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._GroupField]
            self._IndexIDs = [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        return self._IndexIDs
    # 返回指数 ID 为 ifactor_name 在给定时点 idt 的所有成份股
    # 如果 idt 为 None, 将返回指数 ifactor_name 的所有历史成份股
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有 ID
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        Fields = [self._IDField, self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=Fields)
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]# ID
        SQLStr += "FROM "+DBTableName+" "
        if ifactor_name is not None:
            SQLStr += "WHERE "+DBTableName+'.'+FieldDict[self._GroupField]+"='"+ifactor_name+"' "
        else:
            SQLStr += "WHERE "+DBTableName+'.'+FieldDict[self._GroupField]+" IS NOT NULL "
        if idt is not None:
            idt = idt.strftime("%Y%m%d")
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+idt+"' "
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
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        Fields = [self._IDField, self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=Fields)
        if iid is not None:
            SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._InDateField]+" "# 纳入日期
            SQLStr += DBTableName+"."+FieldDict[self._OutDateField]+" "# 剔除日期
            SQLStr += "FROM "+DBTableName+" "
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._InDateField]+" IS NOT NULL "
            if ifactor_name is not None:
                SQLStr += "AND "+DBTableName+'.'+FieldDict[self._GroupField]+"='"+ifactor_name+"' "
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
            if start_dt is not None:
                SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+start_dt.strftime("%Y%m%d")+"') "
                SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL))"
            if end_dt is not None:
                SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._InDateField]
            Data = self._FactorDB.fetchall(SQLStr)
            TimeDelta = dt.timedelta(seconds=59, microseconds=999999, minutes=59, hours=23)
            DateTimes = set()
            for iStartDate, iEndDate in Data:
                iStartDT = dt.datetime.strptime(iStartDate, "%Y%m%d") + TimeDelta
                if iEndDate is None:
                    iEndDT = (dt.datetime.now() if end_dt is None else end_dt)
                DateTimes = DateTimes.union(set(getDateTimeSeries(start_dt=iStartDT, end_dt=iEndDT, timedelta=dt.timedelta(1))))
            return sorted(DateTimes)
        SQLStr = "SELECT MIN("+DBTableName+"."+FieldDict[self._InDateField]+") "# 纳入日期
        SQLStr += "FROM "+DBTableName
        if ifactor_name is not None:
            SQLStr += " WHERE "+DBTableName+'.'+FieldDict[self._GroupField]+"='"+ifactor_name+"'"
        StartDT = dt.datetime.strptime(self._FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d") + dt.timedelta(seconds=59, microseconds=999999, minutes=59, hours=23)
        if start_dt is not None:
            StartDT = max((StartDT, start_dt))
        if end_dt is None:
            end_dt = dt.datetime.combine(dt.date.today(), dt.time(23,59,59,999999))
        return list(getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1)))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        Fields = [self._IDField, self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=Fields)
        # 指数中成份股 ID, 指数证券 ID, 纳入日期, 剔除日期, 最新标志
        SQLStr = "SELECT "+DBTableName+'.'+FieldDict[self._GroupField]+', '# 指数证券 ID
        SQLStr += DBTableName+'.'+FieldDict[self._IDField]+', '# ID
        SQLStr += DBTableName+'.'+FieldDict[self._InDateField]+', '# 纳入日期
        SQLStr += DBTableName+'.'+FieldDict[self._OutDateField]+' '# 剔除日期
        if self._CurSignField: SQLStr = SQLStr[:-1]+", "+DBTableName+'.'+FieldDict[self._CurSignField]+' '# 最新标志
        SQLStr += 'FROM '+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+'.'+FieldDict[self._GroupField], factor_names, is_str=True, max_num=1000)+") "
        SQLStr += 'AND ('+genSQLInCondition(DBTableName+'.'+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+') '
        SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+StartDate.strftime("%Y%m%d")+"') "
        SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict[self._GroupField]+", "
        SQLStr += DBTableName+'.'+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+'.'+FieldDict[self._InDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]: return pd.DataFrame(columns=Fields)
        return pd.DataFrame(np.array(RawData), columns=Fields)
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
        Data = pd.Panel(Data).ix[factor_names, :, ids]
        Data.major_axis = [dt.datetime.combine(iDate, dt.time(23, 59, 59, 999999)) for iDate in Data.major_axis]
        Data.fillna(value=0, inplace=True)
        return _adjustDateTime(Data, dts, fillna=True, method="bfill")

class _DividendTable(_DBTable):
    """分红配股因子表"""
    DateField = Enum("除权除息日", "股权登记日", "派息日", "红股上市日", arg_type="SingleOption", label="日期字段", order=0)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)    
    def __QS_initArgs__(self):
        if self.Name=="中国A股分红":
            self.add_trait("DateField", Enum("除权除息日", "股权登记日", "派息日", "红股上市日", arg_type="SingleOption", label="日期字段", order=0))
        elif self.Name=="中国A股配股":
            self.add_trait("DateField", Enum("除权除息日", "股权登记日", "配股上市日", arg_type="SingleOption", label="日期字段", order=0))
        elif self.Name=="中国共同基金分红":
            self.add_trait("DateField", Enum("除息日", "派息日", "净值除权日", "权益登记日", "收益支付日", arg_type="SingleOption", label="日期字段", order=0))
    # 返回除权除息日为给定时点 idt 的所有 ID
    # 如果 idt 为 None, 将返回所有有记录已经实施分红的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self.DateField, self._IDField, "方案进度"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["方案进度"]+"=3 "
        if idt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回给定 iid 的所有除权除息日
    # 如果 iid 为 None, 将返回所有有记录已经实施分红的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self.DateField, self._IDField])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self.DateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self.DateField]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self._FactorDB.fetchall(SQLStr)))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self.DateField, self._IDField, "方案进度"]+factor_names)
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self.DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["方案进度"]+"=3 "
        SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField],ids,is_str=True,max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self.DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]: return pd.DataFrame(columns=["日期","ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["日期","ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data.set_index(["日期", "ID"], inplace=True)
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data:
            Data[iFactorName] = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double":
                Data[iFactorName] = Data[iFactorName].astype("float")
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]), 23, 59, 59, 999999) for iDate in Data.major_axis]
        Data = _adjustDateTime(Data, dts, fillna=False)
        Data = Data.ix[:, :, ids]
        return Data

class _MappingTable(_DBTable):
    """映射因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._StartDateField = FactorInfo[FactorInfo["FieldType"]=="StartDate"].index[0]
        self._EndDateField = FactorInfo[FactorInfo["FieldType"]=="EndDate"].index[0]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    # 返回给定时点 idt 有数据的所有 ID
    # 如果 idt 为 None, 将返回所有有记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField, self._StartDateField, self._EndDateField])
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
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField, self._StartDateField])
        SQLStr = "SELECT MIN("+DBTableName+"."+FieldDict[self._StartDateField]+") "# 起始日期
        SQLStr += "FROM "+DBTableName
        if iid is not None: SQLStr += " WHERE "+DBTableName+'.'+FieldDict[self._IDField]+"='"+iid+"'"
        StartDT = dt.datetime.strptime(self._FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d") + dt.timedelta(seconds=59, microseconds=999999, minutes=59, hours=23)
        if start_dt is not None: StartDT = max((StartDT, start_dt))
        if end_dt is None: end_dt = dt.datetime.combine(dt.date.today(), dt.time(23,59,59,999999))
        return list(getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1)))
        # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name,fields=[self._IDField, self._StartDateField, self._EndDateField]+factor_names)
        SQLStr = "SELECT "+DBTableName+'.'+FieldDict[self._IDField]+', '
        SQLStr += DBTableName+'.'+FieldDict[self._StartDateField]+', '
        SQLStr += DBTableName+'.'+FieldDict[self._EndDateField]+', '
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += 'WHERE ('+genSQLInCondition(DBTableName+'.'+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+') '
        SQLStr += "AND (("+DBTableName+"."+FieldDict[self._EndDateField]+">='"+StartDate.strftime("%Y%m%d")+"') "
        SQLStr += "OR ("+DBTableName+"."+FieldDict[self._EndDateField]+" IS NULL)) "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._StartDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+'.'+FieldDict[self._StartDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        return pd.DataFrame((np.array(RawData) if RawData else RawData), columns=["ID", '起始日期', '截止日期']+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return None
        StartDT = dt.datetime.combine(dt.datetime.strptime(raw_data["起始日期"].min(), "%Y%m%d").date(), dt.time(23,59,59,999999))
        raw_data["截止日期"] = raw_data["截止日期"].where(pd.notnull(raw_data["截止日期"]), dt.date.today().strftime("%Y%m%d"))
        EndDT = dt.datetime.combine(dt.datetime.strptime(raw_data["截止日期"].max(), "%Y%m%d").date(), dt.time(23,59,59,999999))
        DTs = getDateTimeSeries(StartDT, EndDT, timedelta=dt.timedelta(1))
        IDs = pd.unique(raw_data["ID"])
        raw_data["截止日期"] = [dt.datetime.combine(dt.datetime.strptime(iDate, "%Y%m%d").date(), dt.time(23,59,59,999999)) for iDate in raw_data["截止日期"]]
        raw_data.set_index(["截止日期", "ID"], inplace=True)
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = pd.DataFrame(index=DTs, columns=IDs)
            iData = raw_data[iFactorName].unstack()
            Data[iFactorName].loc[iData.index, iData.columns] = iData
            Data[iFactorName].fillna(method="bfill", inplace=True)
        Data = pd.Panel(Data).ix[factor_names, :, ids]
        return _adjustDateTime(Data, dts, fillna=True, method="bfill")

class _FeatureTable(_DBTable):
    """特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
        # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField]+factor_names)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]: return pd.DataFrame(columns=["ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data = raw_data.set_index(["ID"])
        Data = pd.Panel({dt.datetime.today(): raw_data.T}).swapaxes(0, 1)
        Data = _adjustDateTime(Data, dts, fillna=True, method="bfill")
        Data = Data.ix[:, :, ids]
        return Data
class _FinancialTable(_DBTable):
    """财务因子表"""
    ReportDate = Enum("所有", "年报", "中报", "一季报", "三季报", Dict(), Function(), label="报告期", arg_type="SingleOption", order=0)
    ReportType = List(["408001000", "408004000"], label="报表类型", arg_type="MultiOption", order=1, option_range=("408001000", "408004000"))
    CalcType = Enum("最新", "单季度", "TTM", label="计算方法", arg_type="SingleOption", order=2)
    YearLookBack = Int(0, label="回溯年数", arg_type="Integer", order=3)
    PeriodLookBack = Int(0, label="回溯期数", arg_type="Integer", order=4)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._ANNDateField = FactorInfo[FactorInfo["FieldType"]=="ANNDate"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._ReportTypeField = FactorInfo[FactorInfo["FieldType"]=="ReportType"].index
        self._TempData = {}
        if self._ReportTypeField.shape[0]==0: self._ReportTypeField = None
        else: self._ReportTypeField = self._ReportTypeField[0]
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    # 返回在给定时点 idt 之前有财务报告的 ID
    # 如果 idt 为 None, 将返回所有有财务报告的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField, self._ANNDateField])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._ANNDateField]+"<='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有财务报告的公告时点
    # 如果 iid 为 None, 将返回所有有财务报告的公告时点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._ANNDateField, self._IDField])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self._ANNDateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
        else: SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._ANNDateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._ANNDateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._ANNDateField]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self._FactorDB.fetchall(SQLStr)))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField, self._ANNDateField, self._ReportDateField, self._ReportTypeField]+factor_names)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        if self._ReportTypeField is not None:
            # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
            SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
            SQLStr += "CASE WHEN "+DBTableName+"."+FieldDict[self._ReportTypeField]+"='408001000' THEN "+self._FactorDB.TablePrefix+"AShareIssuingDatePredict.s_stm_actual_issuingdate "
            SQLStr += "WHEN "+DBTableName+"."+FieldDict[self._ReportTypeField]+"='408004000' THEN "+DBTableName+"."+FieldDict[self._ANNDateField]+" END AS ANNDate, "
            SQLStr += DBTableName+"."+FieldDict[self._ReportDateField]+", "
            SQLStr += DBTableName+"."+FieldDict[self._ReportTypeField]+", "
        else:
            # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
            SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
            SQLStr += DBTableName+"."+FieldDict[self._ANNDateField]+" AS ANNDate, "
            SQLStr += DBTableName+"."+FieldDict[self._ReportDateField]+", "
            SQLStr += "NULL AS ReportType, "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" "
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "INNER JOIN "+self._FactorDB.TablePrefix+"AShareIssuingDatePredict ON ("+DBTableName+"."+FieldDict[self._IDField]+"="+self._FactorDB.TablePrefix+"AShareIssuingDatePredict.s_info_windcode AND "+self._FactorDB.TablePrefix+"AShareIssuingDatePredict.report_period="+DBTableName+"."+FieldDict[self._ReportDateField]+") "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        if self._ReportTypeField is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._ReportTypeField]+" IN ('"+"','".join(args.get("报表类型", self.ReportType))+"')"
        SQLStr = "SELECT t.* FROM ("+SQLStr+") t WHERE t.ANNDate IS NOT NULL "
        SQLStr += "ORDER BY t."+FieldDict[self._IDField]+", t.ANNDate, t."+FieldDict[self._ReportDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "公告日期", "报告期", "报表类型"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["ID", "公告日期", "报告期", "报表类型"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Dates = sorted({iDT.strftime("%Y%m%d") for iDT in dts})
        CalcType, YearLookBack, PeriodLookBack, ReportDate = args.get("计算方法", self.CalcType), args.get("回溯年数", self.YearLookBack), args.get("回溯期数", self.PeriodLookBack), args.get("报告期", self.ReportDate)
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
            Data[iID] = CalcFun(Dates, raw_data.loc[iID], factor_names, ReportDate, YearLookBack, PeriodLookBack)
        Data = pd.Panel(Data)
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:]), 23, 59, 59, 999999) for iDate in Dates]
        Data.minor_axis = factor_names
        Data = Data.swapaxes(0, 2)
        Data = _adjustDateTime(Data, dts, fillna=True, method="pad")
        Data = Data.ix[:, :, ids]
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
                    if (raw_data['报告期'].iloc[MaxNoteDateInd-i]==TargetReportDate):
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
                    if (raw_data['报告期'].iloc[MaxNoteDateInd-i]==TargetReportDate):
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
                if (MaxReportDateInd<0) or (raw_data['报告期'].iloc[MaxNoteDateInd-i]>=raw_data['报告期'].iloc[MaxReportDateInd]):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '年报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['报告期'].iloc[MaxNoteDateInd-i][-4:]=='1231') and ((MaxReportDateInd<0) or (raw_data['报告期'].iloc[MaxNoteDateInd-i]>=raw_data['报告期'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '中报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['报告期'].iloc[MaxNoteDateInd-i][-4:]=='0630') and ((MaxReportDateInd<0) or (raw_data['报告期'].iloc[MaxNoteDateInd-i]>=raw_data['报告期'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '一季报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['报告期'].iloc[MaxNoteDateInd-i][-4:]=='0331') and ((MaxReportDateInd<0) or (raw_data['报告期'].iloc[MaxNoteDateInd-i]>=raw_data['报告期'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        elif report_date == '三季报':
            Changed = False
            for i in range(0,MaxNoteDateInd-PreMaxNoteDateInd):
                if (raw_data['报告期'].iloc[MaxNoteDateInd-i][-4:]=='0930') and ((MaxReportDateInd<0) or (raw_data['报告期'].iloc[MaxNoteDateInd-i]>=raw_data['报告期'].iloc[MaxReportDateInd])):
                    MaxReportDateInd = MaxNoteDateInd-i
                    Changed = True
        return (MaxReportDateInd, Changed)
    def _calcIDData_LR(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:# 最大报告期没有变化
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            if MaxReportDateInd>=0: StdData[i] = raw_data[factor_names].iloc[MaxReportDateInd].values
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_SQ(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd,Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            preReportData = None# 前一个报告期数据
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            if MaxReportDate[-4:]=='1231':
                for j in range(0, tempInd+1):
                    if raw_data['报告期'].iloc[tempInd-j]==MaxReportDate[0:4]+'0930':
                        preReportData = raw_data[factor_names].iloc[tempInd-j].values
                        break
            elif MaxReportDate[-4:]=='0930':
                for j in range(0, tempInd+1):
                    if raw_data['报告期'].iloc[tempInd-j]==MaxReportDate[0:4]+'0630':
                        preReportData = raw_data[factor_names].iloc[tempInd-j].values
                        break
            elif MaxReportDate[-4:]=='0630':
                for j in range(0, tempInd+1):
                    if raw_data['报告期'].iloc[tempInd-j]==MaxReportDate[0:4]+'0331':
                        preReportData = raw_data[factor_names].iloc[tempInd-j].values
                        break
            elif MaxReportDate[-4:]=='0331':
                preReportData = 0
            if preReportData is not None:
                StdData[i] = raw_data[factor_names].iloc[MaxReportDateInd].values - preReportData
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_TTM(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            preReportData = None# 去年同期数据
            preYearReport = None# 去年年报数据
            if MaxReportDate[-4:]=='1231':# 最新财报为年报
                StdData[i] = raw_data[factor_names].iloc[MaxReportDateInd].values
            else:
                Year = MaxReportDate[0:4]
                LastYear = str(int(Year)-1)
                for j in range(0, tempInd+1):
                    if (preYearReport is not None) and (preReportData is not None):
                        break
                    elif (preYearReport is None) and (raw_data['报告期'].iloc[tempInd-j]==LastYear+'1231'):
                        preYearReport = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preReportData is None) and (raw_data['报告期'].iloc[tempInd-j]==LastYear+MaxReportDate[-4:]):
                        preReportData = raw_data[factor_names].iloc[tempInd-j].values
                if (preYearReport is not None) and (preReportData is not None):
                    StdData[i] = raw_data[factor_names].iloc[MaxReportDateInd].values + preYearReport - preReportData
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_LR_NYear(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            Year = MaxReportDate[0:4]
            LastNYear = str(int(Year)-year_lookback)
            for j in range(0, tempInd+1):
                if raw_data['报告期'].iloc[tempInd-j]==LastNYear+MaxReportDate[-4:]:
                    StdData[i] = raw_data[factor_names].iloc[tempInd-j].values
                    break
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_SQ_NYear(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            preReportData1 = None# 上N年同期财报数据
            preReportData2 = None# 上N年同期的上一期财报数据
            Year = MaxReportDate[0:4]
            LastNYear = str(int(Year)-year_lookback)
            if MaxReportDate[-4:]=='1231':
                for j in range(0, tempInd+1):
                    if (preReportData1 is not None) and (preReportData2 is not None):
                        break
                    elif (preReportData1 is None) and (raw_data['报告期'].iloc[tempInd-j]==LastNYear+'1231'):# 找到了上N年同期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preReportData2 is None) and (raw_data['报告期'].iloc[tempInd-j]==LastNYear+'0930'):# 找到了上N年同期的上一期数据
                        preReportData2 = raw_data[factor_names].iloc[tempInd-j].values
            elif MaxReportDate[-4:]=='0930':
                for j in range(0, tempInd+1):
                    if (preReportData1 is not None) and (preReportData2 is not None):
                        break
                    elif (preReportData1 is None) and (raw_data['报告期'].iloc[tempInd-j]==LastNYear+'0930'):# 找到了上N年同期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preReportData2 is None) and (raw_data['报告期'].iloc[tempInd-j]==LastNYear+'0630'):# 找到了上N年同期的上一期数据
                        preReportData2 = raw_data[factor_names].iloc[tempInd-j].values
            elif MaxReportDate[-4:]=='0630':
                for j in range(0, tempInd+1):
                    if (preReportData1 is not None) and (preReportData2 is not None):
                        break
                    if (preReportData1 is None) and (raw_data['报告期'].iloc[tempInd-j]==LastNYear+'0630'):# 找到了上N年同期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preReportData2 is None) and (raw_data['报告期'].iloc[tempInd-j]==LastNYear+'0331'):# 找到了上N年同期的上一期数据
                        preReportData2 = raw_data[factor_names].iloc[tempInd-j].values
            elif MaxReportDate[-4:]=='0331':
                for j in range(0, tempInd+1):
                    if raw_data['报告期'].iloc[tempInd-j]==LastNYear+'0331':# 找到了上N年同期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                        preReportData2 = 0
                        break
            if (preReportData1 is not None) and (preReportData2 is not None):
                StdData[i] = preReportData1 - preReportData2
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_TTM_NYear(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            preNYearReportData = None# 上N年同期数据
            preN_1YearYearReport = None# 上N+1年年报数据
            preN_1YearReportData = None# 上N+1年同期数据
            Year = MaxReportDate[0:4]
            LastNYear = str(int(Year)-year_lookback)
            if MaxReportDate[-4:]=='1231':# 最新财报为年报
                for j in range(0, tempInd+1):
                    if (raw_data['报告期'].iloc[tempInd-j]==LastNYear+'1231'):
                        StdData[i] = raw_data[factor_names].iloc[tempInd-j].values
                        break
            else:
                for j in range(0, tempInd+1):
                    if (preN_1YearYearReport is not None) and (preNYearReportData is not None) and (preN_1YearReportData is not None):
                        break
                    elif (preN_1YearYearReport is None) and (raw_data['报告期'].iloc[tempInd-j]==str(int(LastNYear)-1)+'1231'):
                        preN_1YearYearReport = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preNYearReportData is None) and (raw_data['报告期'].iloc[tempInd-j]==LastNYear+MaxReportDate[-4:]):
                        preNYearReportData = raw_data[factor_names][tempInd-j].values
                    elif (preN_1YearReportData is None) and (raw_data['报告期'].iloc[tempInd-j]==str(int(LastNYear)-1)+MaxReportDate[-4:]):
                        preN_1YearReportData = raw_data[factor_names].iloc[tempInd-j].values
                if (preN_1YearYearReport is not None) and (preNYearReportData is not None) and (preN_1YearReportData is not None):
                    StdData[i] = preNYearReportData + preN_1YearYearReport - preN_1YearReportData
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_LR_NPeriod(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            ObjectReportDate = RollBackNPeriod(MaxReportDate, period_lookback)
            for j in range(0, tempInd+1):
                if raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate:
                    StdData[i] = raw_data[factor_names].iloc[tempInd-j].values
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_SQ_NPeriod(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
                continue
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            preReportData1 = None# 上N期财报数据
            preReportData2 = None# 上N+1期财报数据
            ObjectReportDate = RollBackNPeriod(MaxReportDate, period_lookback)# 上N期报告期
            if ObjectReportDate[-4:]=='1231':
                for j in range(0, tempInd+1):
                    if (preReportData1 is not None) and (preReportData2 is not None):
                        break
                    elif (preReportData1 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate):# 找到了上N期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preReportData2 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate[0:4]+'0930'):# 找到了上N+1期数据
                        preReportData2 = raw_data[factor_names].iloc[tempInd-j].values
            elif ObjectReportDate[-4:]=='0930':
                for j in range(0, tempInd+1):
                    if (preReportData1 is not None) and (preReportData2 is not None):
                        break
                    elif (preReportData1 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate):# 找到了上N年同期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preReportData2 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate[0:4]+'0630'):# 找到了上N年同期的上一期数据
                        preReportData2 = raw_data[factor_names].iloc[tempInd-j].values
            elif ObjectReportDate[-4:]=='0630':
                for j in range(0, tempInd+1):
                    if (preReportData1 is not None) and (preReportData2 is not None):
                        break
                    elif (preReportData1 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate):# 找到了上N年同期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preReportData2 is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate[0:4]+'0331'):# 找到了上N年同期的上一期数据
                        preReportData2 = raw_data[factor_names].iloc[tempInd-j].values
            elif ObjectReportDate[-4:]=='0331':
                for j in range(0, tempInd+1):
                    if raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate:# 找到了上N年同期数据
                        preReportData1 = raw_data[factor_names].iloc[tempInd-j].values
                        preReportData2 = 0
                        break
            if (preReportData1 is not None) and (preReportData2 is not None):
                StdData[i] = preReportData1 - preReportData2
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
    def _calcIDData_TTM_NPeriod(self, date_seq, raw_data, factor_names, report_date, year_lookback, period_lookback):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        tempInd = -1# 指向目前看到的最大的公告期
        tempLen = raw_data.shape[0]
        MaxReportDateInd = -1# 指向目前看到的最大的报告期
        for i, iDate in enumerate(date_seq):
            tempPreInd = tempInd# 指向先前的最大公告期
            while (tempInd<tempLen-1) and (iDate>=raw_data['公告日期'].iloc[tempInd+1]): tempInd = tempInd+1
            MaxReportDateInd, Changed = self._findMaxReportDateInd(iDate, raw_data, report_date, MaxReportDateInd, tempInd, tempPreInd)
            if not Changed:
                if MaxReportDateInd>=0: StdData[i] = StdData[i-1]
            MaxReportDate = raw_data['报告期'].iloc[MaxReportDateInd]# 当前最大报告期
            preNPeriodReportData = None# 上N期数据
            preNPeriodYear_1YearReport = None# 上N期上一年年报数据
            preNPeriodYear_1ReportData = None# 上N期上一年同期数据
            ObjectReportDate = RollBackNPeriod(MaxReportDate, period_lookback)
            if ObjectReportDate[-4:]=='1231':# 上N期财报为年报
                for j in range(0, tempInd+1):
                    if (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate):
                        StdData[i] = raw_data[factor_names].iloc[tempInd-j].values
                        break
            else:
                for j in range(0, tempInd+1):
                    if (preNPeriodReportData is not None) and (preNPeriodYear_1YearReport is not None) and (preNPeriodYear_1ReportData is not None):
                        break
                    elif (preNPeriodReportData is None) and (raw_data['报告期'].iloc[tempInd-j]==ObjectReportDate):
                        preNPeriodReportData = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preNPeriodYear_1YearReport is None) and (raw_data['报告期'].iloc[tempInd-j]==str(int(ObjectReportDate[0:4])-1)+'1231'):
                        preNPeriodYear_1YearReport = raw_data[factor_names].iloc[tempInd-j].values
                    elif (preNPeriodYear_1ReportData is None) and (raw_data['报告期'].iloc[tempInd-j]==str(int(ObjectReportDate[0:4])-1)+ObjectReportDate[-4:]):
                        preNPeriodYear_1ReportData = raw_data[factor_names].iloc[tempInd-j].values
                if (preNPeriodReportData is not None) and (preNPeriodYear_1YearReport is not None) and (preNPeriodYear_1ReportData is not None):
                    StdData[i] = preNPeriodReportData + preNPeriodYear_1YearReport - preNPeriodYear_1ReportData
        self._TempData.pop("LastTargetReportDate", None)
        self._TempData.pop("LastTargetReportInd", None)
        return StdData
class _AnalystConsensusTable(_DBTable):
    """分析师汇总表"""
    CalcType = Enum("FY0", "FY1", "FY2", "Fwd12M", label="计算方法", arg_type="SingleOption", order=0)
    Period = Enum("263001000", "263002000", "263003000", "263004000", label="汇总有效期", arg_type="SingleOption", order=1)
    LookBack = Int(180, arg_type="Integer", label="回溯天数", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.ix[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._PeriodField = FactorInfo[FactorInfo["FieldType"]=="Period"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = 'W2财务年报-公告日期'
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def _prepareReportANNRawData(self, ids):
        SQLStr = genANN_ReportSQLStr(self._FactorDB.TablePrefix, ids, report_period="1231")
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "公告日期", "报告期"])
        else: return pd.DataFrame(np.array(RawData), columns=["ID", "公告日期", "报告期"])
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        CalcType = args.get("计算方法", self.CalcType)
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField, self._DateField, self._ReportDateField, self._PeriodField]+factor_names)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, 日期, ID, 报告期, 数据
        SQLStr = 'SELECT '+DBTableName+'.'+FieldDict[self._DateField]+', '
        SQLStr += DBTableName+'.'+FieldDict[self._IDField]+', '
        SQLStr += DBTableName+'.'+FieldDict[self._ReportDateField]+', '
        for iField in factor_names: SQLStr += DBTableName+'.'+FieldDict[iField]+', '
        SQLStr = SQLStr[:-2]+' '
        SQLStr += 'FROM '+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._PeriodField]+"='"+args.get("汇总有效期", self.Period)+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict[self._IDField]+', '+DBTableName+'.'+FieldDict[self._DateField]+', '+DBTableName+'.'+FieldDict[self._ReportDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=['日期','ID','报告期']+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData), columns=['日期','ID','报告期']+factor_names)
        RawData._QS_ANNReport = (CalcType!="Fwd12M")
        return RawData
    def __QS_saveRawData__(self, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock):
        isANNReport = raw_data._QS_ANNReport
        if isANNReport:
            PID = sorted(pid_lock)[0]
            ANN_ReportFilePath = raw_data_dir+os.sep+PID+os.sep+self._ANN_ReportFileName
            pid_lock[PID].acquire()
            if not os.path.isfile(ANN_ReportFilePath+'.dat'):# 没有报告期-公告日期数据, 提取该数据
                with shelve.open(ANN_ReportFilePath) as ANN_ReportFile: pass
                pid_lock[PID].release()
                IDs = []
                for iPID in sorted(pid_ids): IDs.extend(pid_ids[iPID])
                RawData = self._prepareReportANNRawData(ids=IDs)
                super().__QS_saveRawData__(RawData, [], raw_data_dir, pid_ids, self._ANN_ReportFileName, pid_lock)
            else:
                pid_lock[PID].release()
        raw_data = raw_data.set_index(['ID'])
        CommonCols = list(raw_data.columns.difference(set(factor_names)))
        AllIDs = set(raw_data.index)
        for iPID, iIDs in pid_ids.items():
            with shelve.open(raw_data_dir+os.sep+iPID+os.sep+file_name) as iFile:
                iIDs = sorted(AllIDs.intersection(set(iIDs)))
                iData = raw_data.loc[iIDs]
                for jFactorName in factor_names:
                    ijData = iData[CommonCols+[jFactorName]].reset_index()
                    if isANNReport:
                        ijData._QS_ANNReportPath = raw_data_dir+os.sep+iPID+os.sep+self._ANN_ReportFileName
                    iFile[jFactorName] = ijData
        return 0
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
                                        "args":{"汇总有效期":iPeriod, "计算方法":iFactor.CalcType, "回溯天数":iFactor.LookBack}}
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
            ANNReportPath = getattr(raw_data, "_QS_ANNReportPath", None)
            if (ANNReportPath is not None) and os.path.isfile(ANNReportPath):
                with shelve.open(ANNReportPath) as ANN_ReportFile:
                    ANNReportData = ANN_ReportFile["RawData"]
            else:
                ANNReportData = self._prepareReportANNRawData(ids)
            ANNReportData = ANNReportData.set_index(["ID"])
        raw_data = raw_data.set_index(["ID"])
        Data = {}
        for iID in raw_data.index.unique():
            if ANNReportData is not None:
                if iID in ANNReportData.index:
                    iANNReportData = ANNReportData.loc[iID]
                else:
                    continue
            else:
                iANNReportData = None
            Data[iID] = CalcFun(Dates, raw_data.loc[iID], iANNReportData, factor_names, LookBack, FYNum)
        Data = pd.Panel(Data)
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:]), 23, 59, 59, 999999) for iDate in Dates]
        Data = Data.swapaxes(0, 2)
        if LookBack==0: return Data.ix[:, dts, ids]
        AllDTs = Data.major_axis.union(set(dts)).sort_values()
        Data = Data.ix[:, AllDTs, ids]
        LimitNs = LookBack*24.0*3600*10**9
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillna(Data.iloc[i], limit_ns=LimitNs)
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

class WindDB2(WindDB):
    """Wind 量化研究数据库"""
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args, **kwargs)
        self.Name = "WindDB2"
    def __QS_initArgs__(self):
        ConfigFilePath = __QS_LibPath__+os.sep+"WindDB2Config.json"# 配置文件路径
        self._InfoFilePath = __QS_LibPath__+os.sep+"WindDB2Info.hdf5"# 数据库信息文件路径
        Config = readJSONFile(ConfigFilePath)
        ArgNames = self.ArgNames
        for iArgName, iArgVal in Config.items():
            if iArgName in ArgNames: self[iArgName] = iArgVal
        if not os.path.isfile(self._InfoFilePath):
            InfoResourcePath = __QS_MainPath__+os.sep+"Rescource"+os.sep+"WindDB2Info.xlsx"# 数据库信息源文件路径
            print("缺失数据库信息文件: '%s', 尝试从 '%s' 中导入信息." % (self._InfoFilePath, InfoResourcePath))
            if not os.path.isfile(InfoResourcePath): raise __QS_Error__("缺失数据库信息文件: %s" % InfoResourcePath)
            self.importInfo(InfoResourcePath)
        self._TableInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/TableInfo")
        self._FactorInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/FactorInfo")
    def getTable(self, table_name, args={}):
        TableClass = self._TableInfo.loc[table_name, "TableClass"]
        return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=args)")
    # -----------------------------------------数据提取---------------------------------
    # 给定起始日期和结束日期，获取交易所交易日期, 目前仅支持："上海证券交易所", "深圳证券交易所", "香港交易所"
    def getTradeDay(self, start_date=None, end_date=None, exchange="上海证券交易所"):
        if exchange not in ("上海证券交易所", "深圳证券交易所", "香港交易所"):
            raise __QS_Error__("不支持交易所：%s的交易日序列！" % exchange)
        if start_date is None:
            start_date = dt.date(1970,1,1)
        if end_date is None:
            end_date = dt.date.today()
        if exchange in ("上海证券交易所", "深圳证券交易所"):
            SQLStr = "SELECT TRADE_DAYS FROM {Prefix}AShareCalendar WHERE S_INFO_EXCHMARKET=\"{Exchange}\" "
            SQLStr += "AND TRADE_DAYS<=\"{EndDate}\" "
            SQLStr += "AND TRADE_DAYS>=\"{StartDate}\" "
            SQLStr += "ORDER BY TRADE_DAYS"
            SQLStr = SQLStr.format(Prefix=self.TablePrefix, StartDate=start_date.strftime("%Y%m%d"), 
                                   EndDate=end_date.strftime("%Y%m%d"), Exchange=("SSE" if exchange=="上海证券交易所" else "SZSE"))
        elif exchange=="香港交易所":
            SQLStr = "SELECT TRADE_DAYS FROM {Prefix}HKEXCalendar WHERE S_INFO_EXCHMARKET=\"HKEX\" "
            SQLStr += "AND TRADE_DAYS<=\"{EndDate}\" "
            SQLStr += "AND TRADE_DAYS>=\"{StartDate}\" "
            SQLStr += "ORDER BY TRADE_DAYS"
            SQLStr = SQLStr.format(Prefix=self.TablePrefix,StartDate=start_date.strftime("%Y%m%d"),EndDate=end_date.strftime("%Y%m%d"))
        return list(map(lambda x: dt.date(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8])), self.fetchall(SQLStr)))
    # 获取指定日当前在市或者历史上出现过的全体 A 股 ID
    def _getAllAStock(self, date, is_current=True):
        if is_current:
            SQLStr = "SELECT S_INFO_WINDCODE FROM {Prefix}AShareDescription WHERE (S_INFO_DELISTDATE is NULL OR S_INFO_DELISTDATE>'{Date}') AND S_INFO_LISTDATE<='{Date}' ORDER BY S_INFO_WINDCODE"
        else:
            SQLStr = "SELECT S_INFO_WINDCODE FROM {Prefix}AShareDescription WHERE S_INFO_LISTDATE<='{Date}' ORDER BY S_INFO_WINDCODE"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y%m%d")))]
    # 获取指定日当前或历史上的指数成份股ID, is_current=True: 获取指定日当天的ID, False:获取截止指定日历史上出现的 ID
    def getID(self, index_id="全体A股", date=None, is_current=True):
        if date is None:
            date = dt.date.today()
        if index_id=="全体A股":
            return self._getAllAStock(date=date, is_current=is_current)
        # 查询该指数所在的表
        for iTable in table_vs_index:
            if index_id in table_vs_index[iTable]:
                TargetTable = iTable
                break
        else:
            raise __QS_Error__("不支持提取指数代码为：%s的成份股!")
        TargetWindTable = self.TableName2DBTableName([TargetTable])[TargetTable]
        WindFieldNames = self.FieldName2DBFieldName(table=TargetTable,fields=["指数Wind代码","成份股Wind代码","纳入日期","剔除日期","最新标志"])
        # 获取指数中的股票ID
        SQLStr = "SELECT "+WindFieldNames["成份股Wind代码"]+" FROM {Prefix}"+TargetWindTable+" WHERE "+WindFieldNames["指数Wind代码"]+"='"+index_id+"'"
        SQLStr += " AND "+WindFieldNames["纳入日期"]+"<='"+date.strftime("%Y%m%d")+"'"# 纳入日期在date之前
        if is_current:
            SQLStr += " AND ("+WindFieldNames["最新标志"]+"=1 OR "+WindFieldNames["剔除日期"]+">'"+date.strftime("%Y%m%d")+"')"# 剔除日期在date之后
        SQLStr += " ORDER BY "+WindFieldNames["成份股Wind代码"]
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix))]
    # --------------------------------------------信息转换-----------------------------------
    # 获取行业名称和Wind内部查询代码
    def getIndustryDBInnerID(self,industry_class_name="中信行业",level=1):
        with shelve.open(self.LibPath+os.sep+"WindIndustryCode") as LibFile:
            if industry_class_name=="中信一级行业":
                IndustryWindID = LibFile["中信行业1"]
            elif industry_class_name=="申万一级行业":
                IndustryWindID = LibFile["申万行业1"]
            elif industry_class_name=="Wind一级行业":
                IndustryWindID = LibFile["Wind行业1"]
            else:
                IndustryWindID = LibFile.get(industry_class_name+str(level))
        if IndustryWindID is not None:
            return {IndustryWindID[iKey]:iKey for iKey in IndustryWindID.index}
        else:
            return None

if __name__=="__main__":
    import time

    # 功能测试
    WDB = WindDB2()
    #WDB.importInfo("C:\\HST\\QuantStudio\\Resource\\WindDB2Info.xlsx")
    WDB.connect()
    print(WDB.TableNames)
    StartDT = dt.datetime(2007,1,1)
    EndDT = dt.datetime(2007,12,31,23,59,59,999999)

    #FT = WDB.getTable("中国国债期货标的券", args={"月合约Wind代码":"T1803.CFE"})
    #DTs = FT.getDateTime(start_dt=StartDT, end_dt=EndDT)
    #CF = WDB.getTable("中国国债期货标的券", args={"回溯天数":366}).readData(factor_names=["转换因子"], ids=FutureInfo.index.tolist(), dts=[DTs[-1]]).iloc[:, 0, :]
    #FT = WDB.getTable("中国国债期货最便宜可交割券")
    #DTs = FT.getDateTime(iid="T1803.CFE", start_dt=StartDT, end_dt=EndDT)
    #Data = FT.readData(factor_names=["CTD证券Wind代码", "IRR"], ids=["T1803.CFE", "T1806.CFE"], dts=DTs)

    #DTs = WDB.getTable("中国A股交易日历").getDateTime(iid="SSE", start_dt=StartDT, end_dt=EndDT)
    #Data = WDB.getTable("中国A股利润表").readData(factor_names=["利润总额", "营业收入"], ids=["000001.SZ", "600000.SH"], dts=DTs)

    DTs = WDB.getTable("中国A股交易日历").getDateTime(iid="SSE", start_dt=StartDT, end_dt=EndDT)
    Data = WDB.getTable("中国A股盈利预测汇总").readData(factor_names=["每股现金流平均值", "净利润平均值(万元)"], ids=["000001.SZ", "600000.SH"], dts=DTs)

    #DTs = WDB.getTable("中国封闭式基金日行情").getDateTime(start_dt=StartDT, end_dt=EndDT)    
    #FutureIDMap = WDB.getTable("中国期货连续(主力)合约和月合约映射表").readData(factor_names=["映射月合约Wind代码"], ids=["IH.CFE"], dts=DTs)
    #FutureInfo = WDB.getTable("中国期货基本资料").readData(factor_names=["上市日期", "最后交易日期"], ids=["IH1801.CFE"], dts=[dt.datetime.today()]).iloc[:, 0, :]

    WDB.disconnect()
    
    #import QuantStudio.api as QS
    #HDB = QS.FactorDB.HDF5DB()
    #HDB.connect()
    #df = HDB.getTable("TestTable10").readData(factor_names=["净利润_FY0"], ids=None, dts=None).iloc[0]
    #df = df.loc[QS.Tools.DateTime.getMonthLastDateTime(df.index.tolist())]
    #df.iloc[1, 0] = np.nan
    #df.iloc[2, 1] = np.nan
    #df.iloc[4, 0] = np.nan
    #df1 = fillna(df, 30*24*3600*10**9)
    #pass