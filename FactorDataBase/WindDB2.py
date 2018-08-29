# coding=utf-8
"""Wind 量化研究数据库"""
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Int, Str, Range, Bool

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries, getDateSeries
from QuantStudio.Tools.FileFun import readJSONFile
from QuantStudio import __QS_Error__, __QS_LibPath__
from QuantStudio.FactorDataBase.WindDB import WindDB, _DBTable, _adjustDateTime
from QuantStudio.FactorDataBase.WindDB import _MarketTable as WindMarketTable

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
    def __QS_prepareRawData__(self, factor_names, ids=None, dts=None, args={}):
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField])
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if dts:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+dts[0].strftime("%Y%m%d")+"' "
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+dts[-1].strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["日期", "ID"])
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["日期", "ID"])
    def __QS_calcData__(self, raw_data, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        raw_data["交易日"] = 1
        Data = pd.Panel({"交易日":raw_data.set_index(["日期", "ID"])["交易日"].unstack()}).loc[factor_names]
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]), 23, 59, 59, 999999) for iDate in Data.major_axis]
        return _adjustDateTime(Data, dts, fillna=True, value=0)

class _MarketTable(_DBTable):
    """行情因子表"""
    FillNa = Bool(False, arg_type="Bool", label="缺失填充", order=0)
    FillNaLookBack = Int(0, arg_type="Integer", label="缺失填充回溯期数", order=1)
    RawDataLookBack = Int(0, arg_type="Integer", label="原始数据回溯天数", order=2)
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
    def __QS_prepareRawData__(self, factor_names=None, ids=None, dts=None, args={}):
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = None, None
        if factor_names is None: factor_names=self.FactorNames
        if args.get("缺失填充", self.FillNa): StartDate -= dt.timedelta(args.get("原始数据回溯天数", self.RawDataLookBack))
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._DateField, self._IDField]+self._ConditionFields+factor_names)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        if StartDate is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        if EndDate is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        for iConditionField in self._ConditionFields: SQLStr += "AND "+DBTableName+"."+FieldDict[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self._DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+factor_names)
        return pd.DataFrame(np.array(RawData), columns=["日期", "ID"]+factor_names)
    def __QS_calcData__(self, raw_data, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        raw_data = raw_data.set_index(["日期", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]), 23, 59, 59, 999999) for iDate in Data.major_axis]
        Data = _adjustDateTime(Data, dts, fillna=args.get("缺失填充", self.FillNa), method="pad", limit=args.get("缺失填充回溯期数", self.FillNaLookBack))
        if ids is not None: Data = Data.ix[:, :, ids]
        return Data

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
    def __QS_prepareRawData__(self, factor_names=None, ids=None, dts=None, args={}):
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = None, None
        if factor_names is None: factor_names=self.FactorNames
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
        if ids is not None: SQLStr += 'AND ('+genSQLInCondition(DBTableName+'.'+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+') '
        if StartDate is not None:
            SQLStr += "AND (("+DBTableName+"."+FieldDict[self._OutDateField]+">'"+StartDate.strftime("%Y%m%d")+"') "
            SQLStr += "OR ("+DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        if EndDate is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        else:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._InDateField]+" IS NOT NULL "
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict[self._GroupField]+", "
        SQLStr += DBTableName+'.'+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+'.'+FieldDict[self._InDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=Fields)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=Fields)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names=None, ids=None, dts=None, args={}):
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = dt.datetime.strptime(raw_data[self._InDateField].min(), "%Y%m%d").date(), dt.date.today()
        if factor_names is None: factor_names = self.FactorNames
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
        Data = pd.Panel(Data).ix[factor_names]
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
    def __QS_prepareRawData__(self, factor_names=None, ids=None, dts=None, args={}):
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = None, dt.date.today()
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self.DateField, self._IDField, "方案进度"]+factor_names)
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self.DateField]+", "
        SQLStr += DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
            
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["方案进度"]+"=3 "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField],ids,is_str=True,max_num=1000)+") "
        if StartDate is not None: SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+StartDate.strftime("%Y%m%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]+", "+DBTableName+"."+FieldDict[self.DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["日期","ID"]+factor_names)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["日期","ID"]+factor_names)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
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
        if ids is not None: Data = Data.ix[:, :, ids]
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
    def __QS_prepareRawData__(self, factor_names=None, ids=None, dts=None, args={}):
        if dts: StartDate, EndDate = dts[0].date(), dts[-1].date()
        else: StartDate, EndDate = None, dt.date.today()
        if factor_names is None: factor_names=self.FactorNames
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name,fields=[self._IDField, self._StartDateField, self._EndDateField]+factor_names)
        SQLStr = "SELECT "+DBTableName+'.'+FieldDict[self._IDField]+', '
        SQLStr += DBTableName+'.'+FieldDict[self._StartDateField]+', '
        SQLStr += DBTableName+'.'+FieldDict[self._EndDateField]+', '
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += 'WHERE ('+genSQLInCondition(DBTableName+'.'+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+') '
        else:
            SQLStr += 'WHERE '+DBTableName+'.'+FieldDict[self._IDField]+' IS NOT NULL '
        if StartDate is not None:
            SQLStr += "AND (("+DBTableName+"."+FieldDict[self._EndDateField]+">='"+StartDate.strftime("%Y%m%d")+"') "
            SQLStr += "OR ("+DBTableName+"."+FieldDict[self._EndDateField]+" IS NULL)) "
        if EndDate is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._StartDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        else:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self._StartDateField]+" IS NOT NULL "
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict[self._IDField]+", "
        SQLStr += DBTableName+'.'+FieldDict[self._StartDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        return pd.DataFrame((np.array(RawData) if RawData else RawData), columns=["ID", '起始日期', '截止日期']+factor_names)
    def __QS_calcData__(self, raw_data, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if raw_data.shape[0]==0: return None
        if factor_names is None: factor_names = self.FactorNames
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
        Data = pd.Panel(Data)
        return _adjustDateTime(Data, dts, fillna=True, method="bfill").loc[factor_names]

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
    def __QS_prepareRawData__(self, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names=self.FactorNames
        FieldDict = self._FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self._IDField]+factor_names)
        DBTableName = self._FactorDB.TablePrefix+self._FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self._IDField]+", "
        for iField in factor_names: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict[self._IDField], ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict[self._IDField]+" IS NOT NULL "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self._IDField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["ID"]+factor_names)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["ID"]+factor_names)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names=None, ids=None, dts=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        raw_data = raw_data.set_index(["ID"])
        Data = pd.Panel({dt.datetime.today(): raw_data.T}).swapaxes(0, 1)
        Data = _adjustDateTime(Data, dts, fillna=True, method="bfill")
        if ids is not None: Data = Data.ix[:, :, ids]
        return Data

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
    StartDT = dt.datetime(2017,1,1)
    EndDT = dt.datetime(2017,12,31,23,59,59,999999)
    
    #FT = WDB.getTable("中国国债期货标的券", args={"月合约Wind代码":"T1803.CFE"})
    #DTs = FT.getDateTime(start_dt=StartDT, end_dt=EndDT)
    #CF = WDB.getTable("中国国债期货标的券", args={"回溯天数":366}).readData(factor_names=["转换因子"], ids=FutureInfo.index.tolist(), dts=[DTs[-1]]).iloc[:, 0, :]
    FT = WDB.getTable("中国国债期货最便宜可交割券")
    DTs = FT.getDateTime(iid="T1803.CFE", start_dt=StartDT, end_dt=EndDT)
    Data = FT.readData(factor_names=["CTD证券Wind代码", "IRR"], ids=["T1803.CFE", "T1806.CFE"], dts=DTs)
    
    #DTs = WDB.getTable("中国封闭式基金日行情").getDateTime(start_dt=StartDT, end_dt=EndDT)    
    #FutureIDMap = WDB.getTable("中国期货连续(主力)合约和月合约映射表").readData(factor_names=["映射月合约Wind代码"], ids=["IH.CFE"], dts=DTs)
    #FutureInfo = WDB.getTable("中国期货基本资料").readData(factor_names=["上市日期", "最后交易日期"], ids=["IH1801.CFE"], dts=[dt.datetime.today()]).iloc[:, 0, :]
    WDB.disconnect()