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
from QuantStudio.FactorDataBase.WindDB import WindDB, _DBTable, _genStartEndDate, _adjustDateTime
from QuantStudio.FactorDataBase.WindDB import _MarketTable as WindMarketTable

class _CalendarTable(_DBTable):
    """交易日历因子表"""
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(name=name, sys_args=sys_args, **kwargs)
        self._Exchanges = None# array([交易所])
    @property
    def FactorNames(self):
        return ["交易日"]
    # 返回给定时点 idt 有交易的交易所列表
    # 如果 idt 为 None, 将返回表中的所有交易所
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=['日期', '交易所'])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["交易所"]+" "
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["日期"]+"='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["交易所"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    # 返回交易所为 iid 的交易日列表
    # 如果 iid 为 None, 将返回表中有记录数据的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=['日期', '交易所'])
        SQLStr = "SELECT "+DBTableName+"."+FieldDict["日期"]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["交易所"]+"='"+iid+"' "
        else:
            SQLStr = "SELECT DISTINCT "+SQLStr[7:]+"WHERE "+DBTableName+"."+FieldDict["交易所"]+" IS NOT NULL "
        if start_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["日期"]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["日期"]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["日期"]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self.FactorDB.fetchall(SQLStr)))
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        StartDate, EndDate = _genStartEndDate(dts, start_dt, end_dt)
        if factor_names is None:
            factor_names = self.FactorNames
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["日期", "交易所"])
        SQLStr = "SELECT "+DBTableName+"."+FieldDict["日期"]+", "
        SQLStr += DBTableName+"."+FieldDict["交易所"]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict["交易所"], ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["交易所"]+" IS NOT NULL "
        if StartDate is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["日期"]+">='"+StartDate.strftime("%Y%m%d")+"' "
        if EndDate is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["日期"]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["交易所"]+", "
        SQLStr += DBTableName+"."+FieldDict["日期"]
        RawData = self.FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["日期", "ID"])
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["日期", "ID"])
        RawData["交易日"] = 1
        Data = pd.Panel({"交易日":RawData.set_index(["日期", "ID"])["交易日"].unstack()}).loc[factor_names]
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]), 23, 59, 59, 999999) for iDate in Data.major_axis]
        return _adjustDateTime(Data, dts, start_dt, end_dt, fillna=True, value=0)

class _MarketTable(WindMarketTable):
    """行情因子表"""
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["交易日期", "Wind代码"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["Wind代码"]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["交易日期"]+"='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["Wind代码"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["交易日期", "Wind代码"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["交易日期"]+" "# 日期
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["Wind代码"]+"='"+iid+"' "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["Wind代码"]+" IS NOT NULL "
        if start_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["交易日期"]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self.FactorDB.fetchall(SQLStr)))
    def _getRawData(self, fields, ids=None, start_date=None, end_date=None, args={}):
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["交易日期","Wind代码"]+fields)
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict["交易日期"]+", "
        SQLStr += DBTableName+"."+FieldDict["Wind代码"]+", "
        for iField in fields:
            SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict["Wind代码"],ids,is_str=True,max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["Wind代码"]+" IS NOT NULL "
        if start_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+">='"+start_date.strftime("%Y%m%d")+"' "
        if end_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+"<='"+end_date.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["Wind代码"]+", "+DBTableName+"."+FieldDict["交易日期"]
        RawData = self.FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["日期","ID"]+fields)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["日期","ID"]+fields)
        return RawData

class _ConstituentTable(_DBTable):
    """成份因子表"""
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(name=name, sys_args=sys_args, **kwargs)
        self._IndexIDs = None# (指数 ID)
    @property
    def FactorNames(self):
        if self._IndexIDs is None:
            DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
            FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["指数Wind代码"])
            SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["指数Wind代码"]+" "# 指数 ID
            SQLStr += "FROM "+DBTableName+" "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["指数Wind代码"]
            self._IndexIDs = [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
        return self._IndexIDs
    # 返回指数 ID 为 ifactor_name 在给定时点 idt 的所有成份股
    # 如果 idt 为 None, 将返回指数 ifactor_name 的所有历史成份股
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有 ID
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=['成份股Wind代码', '指数Wind代码', '纳入日期', '剔除日期', '最新标志'])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["成份股Wind代码"]# ID
        SQLStr += "FROM "+DBTableName+" "
        if ifactor_name is not None:
            SQLStr += "WHERE "+DBTableName+'.'+FieldDict['指数Wind代码']+"='"+ifactor_name+"' "
        else:
            SQLStr += "WHERE "+DBTableName+'.'+FieldDict['指数Wind代码']+" IS NOT NULL "
        if idt is not None:
            idt = idt.strftime("%Y%m%d")
            SQLStr += "AND "+DBTableName+"."+FieldDict["纳入日期"]+"<='"+idt+"' "
            SQLStr += "AND (("+DBTableName+"."+FieldDict["剔除日期"]+">'"+idt+"') "
            SQLStr += "OR ("+DBTableName+"."+FieldDict["最新标志"]+"=1)) "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["成份股Wind代码"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    # 返回指数 ID 为 ifactor_name 包含成份股 iid 的时间点序列
    # 如果 iid 为 None, 将返回指数 ifactor_name 的有记录数据的时间点序列
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有时间点
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=['成份股Wind代码', '指数Wind代码', '纳入日期', '剔除日期', '最新标志'])
        if iid is not None:
            SQLStr = "SELECT "+DBTableName+"."+FieldDict["纳入日期"]+" "# 纳入日期
            SQLStr += DBTableName+"."+FieldDict["剔除日期"]+" "# 剔除日期
            SQLStr += "FROM "+DBTableName+" "
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["纳入日期"]+" IS NOT NULL "
            if ifactor_name is not None:
                SQLStr += "AND "+DBTableName+'.'+FieldDict['指数Wind代码']+"='"+ifactor_name+"' "
            SQLStr += "AND "+DBTableName+"."+FieldDict["成份股Wind代码"]+"='"+iid+"' "
            if start_dt is not None:
                SQLStr += "AND (("+DBTableName+"."+FieldDict["剔除日期"]+">'"+start_dt.strftime("%Y%m%d")+"') "
                SQLStr += "OR ("+DBTableName+"."+FieldDict["剔除日期"]+" IS NULL))"
            if end_dt is not None:
                SQLStr += "AND "+DBTableName+"."+FieldDict["纳入日期"]+"<='"+end_dt.strftime("%Y%m%d")+"' "
            SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["纳入日期"]
            Data = self.FactorDB.fetchall(SQLStr)
            TimeDelta = dt.timedelta(seconds=59, microseconds=999999, minutes=59, hours=23)
            DateTimes = set()
            for iStartDate, iEndDate in Data:
                iStartDT = dt.datetime.strptime(iStartDate, "%Y%m%d") + TimeDelta
                if iEndDate is None:
                    iEndDT = (dt.datetime.now() if end_dt is None else end_dt)
                DateTimes = DateTimes.union(set(getDateTimeSeries(start_dt=iStartDT, end_dt=iEndDT, timedelta=dt.timedelta(1))))
            return sorted(DateTimes)
        SQLStr = "SELECT MIN("+DBTableName+"."+FieldDict["纳入日期"]+") "# 纳入日期
        SQLStr += "FROM "+DBTableName
        if ifactor_name is not None:
            SQLStr += " WHERE "+DBTableName+'.'+FieldDict['指数Wind代码']+"='"+ifactor_name+"'"
        StartDT = dt.datetime.strptime(self.FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d") + dt.timedelta(seconds=59, microseconds=999999, minutes=59, hours=23)
        if start_dt is not None:
            StartDT = max((StartDT, start_dt))
        if end_dt is None:
            end_dt = dt.datetime.combine(dt.date.today(), dt.time(23,59,59,999999))
        return list(getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1)))
    def _getRawData(self, fields, ids=None, start_date=None, end_date=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=['成份股Wind代码', '指数Wind代码', '纳入日期', '剔除日期', '最新标志'])
        # 指数中成份股 ID, 指数证券 ID, 纳入日期, 剔除日期, 最新标志
        SQLStr = "SELECT "+DBTableName+'.'+FieldDict['指数Wind代码']+', '# 指数证券 ID
        SQLStr += DBTableName+'.'+FieldDict['成份股Wind代码']+', '# ID
        SQLStr += DBTableName+'.'+FieldDict['纳入日期']+', '# 纳入日期
        SQLStr += DBTableName+'.'+FieldDict['剔除日期']+', '# 剔除日期
        SQLStr += DBTableName+'.'+FieldDict['最新标志']+' '# 最新标志
        SQLStr += 'FROM '+DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(DBTableName+'.'+FieldDict['指数Wind代码'], fields, is_str=True, max_num=1000)+") "
        if ids is not None:
            SQLStr += 'AND ('+genSQLInCondition(DBTableName+'.'+FieldDict['成份股Wind代码'], ids, is_str=True, max_num=1000)+') '
        if start_date is not None:
            SQLStr += "AND (("+DBTableName+"."+FieldDict["剔除日期"]+">'"+start_date.strftime("%Y%m%d")+"') "
            SQLStr += "OR ("+DBTableName+"."+FieldDict["剔除日期"]+" IS NULL)) "
        if end_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["纳入日期"]+"<='"+end_date.strftime("%Y%m%d")+"' "
        else:
            SQLStr += "AND "+DBTableName+"."+FieldDict["纳入日期"]+" IS NOT NULL "
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict['指数Wind代码']+", "
        SQLStr += DBTableName+'.'+FieldDict['成份股Wind代码']+", "
        SQLStr += DBTableName+'.'+FieldDict['纳入日期']
        RawData = self.FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["指数ID", 'ID', '纳入日期', '剔除日期', '最新标志'])
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["指数ID", 'ID', '纳入日期', '剔除日期', '最新标志'])
        return RawData
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        StartDate, EndDate = _genStartEndDate(dts, start_dt, end_dt)
        if factor_names is None:
            factor_names = self.FactorNames
        RawData = self._getRawData(factor_names, ids, StartDate, EndDate, args=args)
        if StartDate is None:
            StartDate = dt.datetime.strptime(RawData["纳入日期"].min(), "%Y%m%d").date()
        if EndDate is None:
            EndDate = dt.date.today()
        DateSeries = getDateSeries(StartDate, EndDate)
        Data = {}
        for iIndexID in factor_names:
            iRawData = RawData[RawData["指数ID"]==iIndexID].set_index(["ID"])
            iData = pd.DataFrame(0, index=DateSeries, columns=pd.unique(iRawData.index))
            for jID in iData.columns:
                jIDRawData = iRawData.loc[[jID]]
                for k in range(jIDRawData.shape[0]):
                    kStartDate = dt.datetime.strptime(jIDRawData["纳入日期"].iloc[k], "%Y%m%d").date()
                    kEndDate = (dt.datetime.strptime(jIDRawData["剔除日期"].iloc[k], "%Y%m%d").date()-dt.timedelta(1) if jIDRawData["剔除日期"].iloc[k] is not None else dt.date.today())
                    iData[jID].loc[kStartDate:kEndDate] = 1
            Data[iIndexID] = iData
        Data = pd.Panel(Data).ix[factor_names]
        Data.major_axis = [dt.datetime.combine(iDate, dt.time(23, 59, 59, 999999)) for iDate in Data.major_axis]
        Data.fillna(value=0, inplace=True)
        return _adjustDateTime(Data, dts, start_dt, end_dt, fillna=True, method="bfill")

class _ETFPCFTable(WindMarketTable):
    """ETF 申购赎回成份因子表"""
    FundID = Str("510050.SH", arg_type="String", label="基金ID", order=0)
    def getFundID(self, ids=None, dts=None):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["交易日期", "成份股Wind代码", "基金Wind代码"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["基金Wind代码"]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict["成份股Wind代码"], ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["成份股Wind代码"]+" IS NOT NULL "
        if dts is not None:
            Dates = list({iDT.strftime("%Y%m%d") for iDT in dts})
            SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict["交易日期"], Dates, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["基金Wind代码"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    # 返回 ETF 基金 ID 为给定参数的在给定时点 idt 的所有申购赎回成份
    # 如果 idt 为 None, 将返回该 ETF 基金的所有历史申购赎回成份
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["交易日期", "成份股Wind代码", "基金Wind代码"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["成份股Wind代码"]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["基金Wind代码"]+"='"+args.get("基金ID", self.FundID)+"' "
        if idt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+"='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["成份股Wind代码"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    # 返回 ETF 基金 ID 为给定参数的其申购赎回成份中包含 iid 的时间点序列
    # 如果 iid 为 None, 将返回 ETF 基金 ifactor_name 的有记录数据的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["交易日期", "成份股Wind代码", "基金Wind代码"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["交易日期"]+" "# 日期
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["基金Wind代码"]+"='"+args.get("基金ID", self.FundID)+"' "
        if iid is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["成份股Wind代码"]+"='"+iid+"' "
        if start_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["交易日期"]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self.FactorDB.fetchall(SQLStr)))
    def _getRawData(self, fields, ids=None, start_date=None, end_date=None, args={}):
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["交易日期", "成份股Wind代码", "基金Wind代码"]+fields)
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict["交易日期"]+", "
        SQLStr += DBTableName+"."+FieldDict["成份股Wind代码"]+", "
        for iField in fields:
            SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["基金Wind代码"]+"='"+args.get("基金ID", self.FundID)+"' "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict["成份股Wind代码"], ids, is_str=True, max_num=1000)+") "
        if start_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+">='"+start_date.strftime("%Y%m%d")+"' "
        if end_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["交易日期"]+"<='"+end_date.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["成份股Wind代码"]+", "
        SQLStr += DBTableName+"."+FieldDict["交易日期"]
        RawData = self.FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["日期", "ID"]+fields)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["日期", "ID"]+fields)
        return RawData

class _DividendTable(_DBTable):
    """分红配股因子表"""
    DateField = Enum("除权除息日", "股权登记日", "派息日", "红股上市日", arg_type="SingleOption", label="日期字段", order=0)
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
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self.DateField, "Wind代码", "方案进度"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["Wind代码"]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["方案进度"]+"=3 "
        if idt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["Wind代码"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    # 返回给定 iid 的所有除权除息日
    # 如果 iid 为 None, 将返回所有有记录已经实施分红的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self.DateField, "Wind代码"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict[self.DateField]+" "
        SQLStr += "FROM "+DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["Wind代码"]+"='"+iid+"' "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["Wind代码"]+" IS NOT NULL "
        if start_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+start_dt.strftime("%Y%m%d")+"' "
        if end_dt is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict[self.DateField]
        return list(map(lambda x: dt.datetime(int(x[0][:4]), int(x[0][4:6]), int(x[0][6:8]), 23, 59, 59, 999999), self.FactorDB.fetchall(SQLStr)))
    def _getRawData(self, fields, ids=None, start_date=None, end_date=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=[self.DateField,"Wind代码","方案进度"]+fields)
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict[self.DateField]+", "
        SQLStr += DBTableName+"."+FieldDict["Wind代码"]+", "
        for iField in fields:
            SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "WHERE "+DBTableName+"."+FieldDict["方案进度"]+"=3 "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(DBTableName+"."+FieldDict["Wind代码"],ids,is_str=True,max_num=1000)+") "
        if start_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+">='"+start_date.strftime("%Y%m%d")+"' "
        if end_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict[self.DateField]+"<='"+end_date.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["Wind代码"]+", "+DBTableName+"."+FieldDict[self.DateField]
        RawData = self.FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["日期","ID"]+fields)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["日期","ID"]+fields)
        return RawData
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        StartDate, EndDate = _genStartEndDate(dts, start_dt, end_dt)
        if factor_names is None: factor_names = self.FactorNames
        RawData = self._getRawData(factor_names, ids, StartDate, EndDate, args=args)
        RawData.set_index(["日期", "ID"], inplace=True)
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in RawData:
            Data[iFactorName] = RawData[iFactorName].unstack()
            if DataType[iFactorName]=="double":
                Data[iFactorName] = Data[iFactorName].astype("float")
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8]), 23, 59, 59, 999999) for iDate in Data.major_axis]
        Data = _adjustDateTime(Data, dts, start_dt, end_dt, fillna=False)
        if ids is not None: Data = Data.ix[:, :, ids]
        return Data

class _MappingTable(_DBTable):
    """映射因子表"""
    # 返回给定时点 idt 有数据的所有 ID
    # 如果 idt 为 None, 将返回所有有记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["Wind代码", "起始日期", "截止日期"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["Wind代码"]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        if idt is not None:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["起始日期"]+"<='"+idt.strftime("%Y%m%d")+"' "
            SQLStr += "AND "+DBTableName+"."+FieldDict["截止日期"]+">='"+idt.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["Wind代码"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    # 返回给定 ID iid 的起始日期距今的时点序列
    # 如果 idt 为 None, 将以表中最小的起始日期作为起点
    # 忽略 ifactor_name    
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["Wind代码", "起始日期"])
        SQLStr = "SELECT MIN("+DBTableName+"."+FieldDict["起始日期"]+") "# 起始日期
        SQLStr += "FROM "+DBTableName
        if iid is not None:
            SQLStr += " WHERE "+DBTableName+'.'+FieldDict['Wind代码']+"='"+iid+"'"
        StartDT = dt.datetime.strptime(self.FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d") + dt.timedelta(seconds=59, microseconds=999999, minutes=59, hours=23)
        if start_dt is not None:
            StartDT = max((StartDT, start_dt))
        if end_dt is None:
            end_dt = dt.datetime.combine(dt.date.today(), dt.time(23,59,59,999999))
        return list(getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1)))
     # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def _getRawData(self, fields, ids=None, start_date=None, end_date=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name,fields=["Wind代码", "起始日期", "截止日期"]+fields)
        SQLStr = "SELECT "+DBTableName+'.'+FieldDict['Wind代码']+', '
        SQLStr += DBTableName+'.'+FieldDict['起始日期']+', '
        SQLStr += DBTableName+'.'+FieldDict['截止日期']+', '
        for iField in fields:
            SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += 'WHERE ('+genSQLInCondition(DBTableName+'.'+FieldDict['Wind代码'], ids, is_str=True, max_num=1000)+') '
        else:
            SQLStr += 'WHERE '+DBTableName+'.'+FieldDict['Wind代码']+' IS NOT NULL '
        if start_date is not None:
            SQLStr += "AND (("+DBTableName+"."+FieldDict["截止日期"]+">='"+start_date.strftime("%Y%m%d")+"') "
            SQLStr += "OR ("+DBTableName+"."+FieldDict["截止日期"]+" IS NULL)) "
        if end_date is not None:
            SQLStr += "AND "+DBTableName+"."+FieldDict["起始日期"]+"<='"+end_date.strftime("%Y%m%d")+"' "
        else:
            SQLStr += "AND "+DBTableName+"."+FieldDict["起始日期"]+" IS NOT NULL "
        SQLStr += 'ORDER BY '+DBTableName+'.'+FieldDict['Wind代码']+", "
        SQLStr += DBTableName+'.'+FieldDict['起始日期']
        RawData = self.FactorDB.fetchall(SQLStr)
        return pd.DataFrame((np.array(RawData) if RawData else RawData), columns=["ID", '起始日期', '截止日期']+fields)
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        StartDate, EndDate = _genStartEndDate(dts, start_dt, end_dt)
        if factor_names is None: factor_names = self.FactorNames
        RawData = self._getRawData(factor_names, ids, StartDate, EndDate, args=args)
        if RawData.shape[0]==0: return None
        StartDT = dt.datetime.combine(dt.datetime.strptime(RawData["起始日期"].min(), "%Y%m%d").date(), dt.time(23,59,59,999999))
        RawData["截止日期"] = RawData["截止日期"].where(pd.notnull(RawData["截止日期"]), dt.date.today().strftime("%Y%m%d"))
        EndDT = dt.datetime.combine(dt.datetime.strptime(RawData["截止日期"].max(), "%Y%m%d").date(), dt.time(23,59,59,999999))
        DTs = getDateTimeSeries(StartDT, EndDT, timedelta=dt.timedelta(1))
        IDs = pd.unique(RawData["ID"])
        RawData["截止日期"] = [dt.datetime.combine(dt.datetime.strptime(iDate, "%Y%m%d").date(), dt.time(23,59,59,999999)) for iDate in RawData["截止日期"]]
        RawData.set_index(["截止日期", "ID"], inplace=True)
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = pd.DataFrame(index=DTs, columns=IDs)
            iData = RawData[iFactorName].unstack()
            Data[iFactorName].loc[iData.index, iData.columns] = iData
            Data[iFactorName].fillna(method="bfill", inplace=True)
        Data = pd.Panel(Data)
        return _adjustDateTime(Data, dts, start_dt, end_dt, fillna=True, method="bfill").loc[factor_names]

class _FeatureTable(_DBTable):
    """特征因子表"""
    def getID(self, ifactor_name=None, idt=None, args={}):
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["Wind代码"])
        SQLStr = "SELECT DISTINCT "+DBTableName+"."+FieldDict["Wind代码"]+" "# ID
        SQLStr += "FROM "+DBTableName+" "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["Wind代码"]
        return [iRslt[0] for iRslt in self.FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
     # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def _getRawData(self, fields, ids=None, args={}):
        FieldDict = self.FactorDB.FieldName2DBFieldName(table=self.Name, fields=["Wind代码"]+fields)
        DBTableName = self.FactorDB.TablePrefix+self.FactorDB.TableName2DBTableName([self.Name])[self.Name]
        # 形成SQL语句, ID, 因子数据
        SQLStr = "SELECT "+DBTableName+"."+FieldDict["Wind代码"]+", "
        for iField in fields:
            SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" "
        SQLStr += "FROM "+DBTableName+" "
        if ids is not None:
            SQLStr += "WHERE ("+genSQLInCondition(DBTableName+"."+FieldDict["Wind代码"], ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+DBTableName+"."+FieldDict["Wind代码"]+" IS NOT NULL "
        SQLStr += "ORDER BY "+DBTableName+"."+FieldDict["Wind代码"]
        RawData = self.FactorDB.fetchall(SQLStr)
        if RawData==[]:
            RawData = pd.DataFrame(columns=["ID"]+fields)
        else:
            RawData = pd.DataFrame(np.array(RawData), columns=["ID"]+fields)
        return RawData
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        RawData = self._getRawData(factor_names, ids, args=args)
        RawData = RawData.set_index(["ID"])
        Data = pd.Panel({dt.datetime.today(): RawData.T}).swapaxes(0, 1)
        Data = _adjustDateTime(Data, dts, start_dt, end_dt, fillna=True, method="bfill")
        if ids is not None:
            Data = Data.ix[:, :, ids]
        return Data

class WindDB2(WindDB):
    """Wind 量化研究数据库"""
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args, **kwargs)
        self.Name = "WindDB2"
    def __QS_initArgs__(self):
        Config = readJSONFile(__QS_LibPath__+os.sep+"WindDB2Config.json")
        for iArgName, iArgVal in Config.items(): self[iArgName] = iArgVal
        self._InfoFilePath = __QS_LibPath__+os.sep+"WindDB2Info.hdf5"
        self._TableInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/TableInfo")
        self._FactorInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/FactorInfo")
    def getTable(self, table_name):
        TableClass = self._TableInfo.loc[table_name, "TableClass"]
        FT = eval("_"+TableClass+"('"+table_name+"')")
        FT.FactorDB = self
        return FT
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
    # 功能测试
    WDB = WindDB2()
    WDB.connect()
    print(WDB.TableNames)
    
    # 导入数据库信息
    #from QuantStudio import __QS_MainPath__
    #WDB.importInfo(__QS_MainPath__+os.sep+"Resource\\WindDB2Info.xlsx")
    
    ## 交易日
    #FT = WDB.getTable("中国期货交易日历")
    #print(FT.FactorNames)
    #IDs = FT.getID()
    #DateTimes = FT.getDateTime(iid="CFFEX")
    #TradeDayData = FT.readData(factor_names=["交易日"], ids=["INE", "CFFEX"], start_dt=dt.datetime(2018,1,1))
    
    ## 日行情
    #FT = WDB.getTable("货币市场日行情")
    #print(FT.FactorNames)
    ##print(FT.getID())
    #MarketData = FT.readData(factor_names=["收盘利率"], ids=["IBO007.IB"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999))
    
    # 指数成份
    #FT = WDB.getTable("中国A股指数成份股")
    #print(FT.FactorNames)
    #IDs = FT.readData(factor_names=["000016.SH"], ids=None, start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999))
    #ConstituentData = FT.readData(factor_names=["000016.SH"], ids=None, start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999))
    
    ## ETF
    #FT = WDB.getTable("中国ETF申购赎回清单")
    #print(FT.FactorNames)
    #PCFData = FT.readData(ids=["510050.SH"], start_dt=dt.datetime(2018,4,26), end_dt=dt.datetime(2018,4,27,23,59,59,999999))
    #FT = WDB.getTable("中国ETF申购赎回成份")
    #print(FT.FactorNames)
    #PCFData = FT.readData(factor_names=["股票数量", "现金替代标志"], start_dt=dt.datetime(2017,6,1), end_dt=dt.datetime(2017,6,30,23,59,59,999999), args={"基金ID":"510050.SH"})
    
    # 分红, 配股
    FT = WDB.getTable("中国A股分红")
    print(FT.FactorNames)
    DividendData = FT.readData(factor_names=["每股派息(税前)", "每股送股比例", "每股转增比例"], ids=["600000.SH", "000001.SZ"], start_dt=dt.datetime(2017,1,1))
    #FT = WDB.getTable("中国A股配股")
    #print(FT.FactorNames)
    #RightIssueData = FT.readData(factor_names=["配股价格", "配股比例"], ids=["600000.SH", "300110.SZ"], start_dt=dt.datetime(2017,1,1))
    #FT = WDB.getTable("中国共同基金分红")
    #print(FT.FactorNames)
    #FundDividendData = FT.readData(factor_names=["每股派息(元)", "可分配收益(元)", "收益分配金额(元)", "基准基金份额(万份)"], ids=["510050.SH"], start_dt=dt.datetime(2017,1,1))
    
    # 映射
    #FT = WDB.getTable("中国期货连续(主力)合约和月合约映射表")
    #print(FT.FactorNames)
    ##IDs = FT.getID()
    #Data = FT.readData(factor_names=["映射月合约Wind代码"], ids=["T00.CFE", "T.CFE"], start_dt=dt.datetime(2017,1,1), end_dt=dt.datetime(2017,12,31,23,59,59,999999))
    #FT = WDB.getTable("中国共同基金被动型基金跟踪指数")
    #print(FT.FactorNames)
    #Data = FT.readData(factor_names=["跟踪指数Wind代码"], ids=["510050.SH", "510680.SH"], start_dt=dt.datetime(2018,1,1))
    
    ## 特征
    #FT = WDB.getTable("中国A股指数基本资料")
    #print(FT.FactorNames)
    #Data = FT.readData(factor_names=["发布方", "发布日期"], ids=["000001.SH", "000016.SH"]).iloc[:, 0]
    WDB.disconnect()