# coding=utf-8
"""基于 tushare 的因子库"""
import re
import os
import datetime as dt

import numpy as np
import pandas as pd
import tushare as ts
from traits.api import Enum, Int, Str, Range, Bool

from QuantStudio.Tools.AuxiliaryFun import searchNameInStrList
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio import __QS_Object__, __QS_Error__, __QS_LibPath__, __QS_MainPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable

class _FactorTable(FactorTable):
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
                if iDataType.find("str")!=-1: MetaData.iloc[i] = "string"
                else: MetaData.iloc[i] = "double"
            return MetaData
        elif key=="Description": return FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType"),
                                 "Description":self.getFactorMetaData(factor_names, key="Description")})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))


class _CalendarTable(_FactorTable):
    """交易日历因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    # 返回交易所列表
    # 忽略 ifactor_name, idt
    def getID(self, ifactor_name=None, idt=None, args={}):
        return ["SSE", "SZSE"]
    # 返回交易所为 iid 的交易日列表
    # 如果 iid 为 None, 将返回表中有记录数据的时间点序列
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        DateField = self._FactorDB._FactorInfo['DBFieldName'].loc[self.Name].loc[self._DateField]
        if start_dt is None: start_dt = dt.date(1900, 1, 1)
        start_dt = start_dt.strftime("%Y%m%d")
        if end_dt is None: end_dt = dt.date.today()
        end_dt = end_dt.strftime("%Y%m%d")
        if iid is None: iid="SSE"
        Dates = self._FactorDB._ts.query(DBTableName, exchange_id=iid, start_date=start_dt, end_date=end_dt, fields=DateField, is_open="1")
        return [dt.datetime(int(iDate[:4]), int(iDate[4:6]), int(iDate[6:8])) for iDate in Dates[DateField].values]
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo['DBFieldName'].loc[self.Name].loc[[self._DateField, self._IDField]+factor_names]
        Fields = FieldDict.tolist()
        StartDate, EndDate = dts[0].strftime("%Y%m%d"), dts[-1].strftime("%Y%m%d")
        RawData = None
        for iID in ids:
            iData = self._FactorDB._ts.query(DBTableName, exchange_id=iID, start_date=StartDate, end_date=EndDate, fields=Fields)
            iData.index = [iID] * iData.shape[0]
            if RawData is None: RawData = iData
            else: RawData = RawData.append(iData)
        if RawData is None: return pd.DataFrame(columns=["日期", "ID"]+factor_names)
        RawData.index, RawData.columns = np.arange(RawData.shape[0]), ["日期", "ID"]+factor_names
        return RawData
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
        return Data.loc[:, dts, ids]

class _MarketTable(_FactorTable):
    """行情因子表"""
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=0)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateFields = FactorInfo[FactorInfo["FieldType"]=="Date"].index.tolist()
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        self.add_trait("DateField", Enum(*self._DateFields, arg_type="SingleOption", label="日期字段", order=1))
        DefaultDateField = FactorInfo[(FactorInfo["Supplementary"]=="DefaultDate") & (FactorInfo["FieldType"]=="Date")]
        if DefaultDateField.shape[0]>0: self.DateField = DefaultDateField.index[0]
        else: self.DateField = self._DateFields[0]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+2))
            self[iCondition] = str(FactorInfo.loc[iCondition, "Supplementary"])        
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()+self._DateFields
    def getID(self, ifactor_name=None, idt=None, args={}):
        return self._FactorDB.getID(index_id="全体A股", is_current=False)
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if start_dt is not None: start_dt = start_dt.date()
        if end_dt is not None: end_dt = end_dt.date()
        return self._FactorDB.getTradeDay(start_date=start_dt, end_date=end_dt, exchange="SSE", output_type="datetime")
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
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        Fields = [self.DateField, self._IDField]+factor_names
        DBFields = FactorInfo['DBFieldName'].loc[Fields].tolist()
        DBTableName = self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        RawData = pd.DataFrame(columns=DBFields)
        FieldStr = ",".join(DBFields)
        Conditions = {}
        for iConditionField in self._ConditionFields:
            iDataType = FactorInfo.loc[iConditionField, "DataType"]
            if iDataType=="str":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = args.get(iConditionField, self[iConditionField])
            elif iDataType=="float":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = float(args.get(iConditionField, self[iConditionField]))
            elif iDataType=="int":
                Conditions[FactorInfo.loc[iConditionField, "DBFieldName"]] = int(args.get(iConditionField, self[iConditionField]))
        if (len(ids)<=(EndDate-StartDate).days) and (self._IDField=="ts_code"):
            StartDate, EndDate = StartDate.strftime("%Y%m%d"), EndDate.strftime("%Y%m%d")
            for iID in ids:
                iData = self._FactorDB._ts.query(DBTableName, ts_code=iID, start_date=StartDate, end_date=EndDate, fields=FieldStr, **Conditions)
                RawData = RawData.append(iData)
        else:
            for i in range((EndDate-StartDate).days+1):
                iDate = StartDate + dt.timedelta(i)
                iData = self._FactorDB._ts.query(DBTableName, trade_date=iDate.strftime("%Y%m%d"), fields=FieldStr, **Conditions)
                RawData = RawData.append(iData)
        RawData = RawData.loc[:, DBFields]
        RawData.columns = ["日期", "ID"]+factor_names
        return RawData.sort_values(by=["ID", "日期"])
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
        if Data.minor_axis.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]

class TushareDB(FactorDB):
    """tushare"""
    Token = Str("", label="Token", arg_type="String", order=0)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"TushareDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "TushareDB"
        self._ts = None
        self._InfoFilePath = __QS_LibPath__+os.sep+"TushareDBInfo.hdf5"# 数据库信息文件路径
        self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"TushareDBInfo.xlsx"# 数据库信息源文件路径
        self._updateInfo()
        return
    def _updateInfo(self):
        if not os.path.isfile(self._InfoFilePath): 
            print("缺失数据库信息文件: '%s', 尝试从 '%s' 中导入信息." % (self._InfoFilePath, self._InfoResourcePath))
            if not os.path.isfile(self._InfoResourcePath): raise __QS_Error__("缺失数据库信息文件: %s" % self._InfoResourcePath)
            self.importInfo(self._InfoResourcePath)
        elif os.path.isfile(self._InfoResourcePath) and (os.path.getmtime(self._InfoResourcePath)>os.path.getmtime(self._InfoFilePath)):
            print("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % self._InfoResourcePath)
            self.importInfo(self._InfoResourcePath)
        self._TableInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/TableInfo")
        self._FactorInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/FactorInfo")
    def connect(self):
        ts.set_token(self.Token)
        self._ts = ts.pro_api()
        return 0
    def disconnect(self):
        self._ts = None
        return 0
    def isAvailable(self):
        return (self._ts is not None)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ts"] = self.isAvailable()
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._ts:
            self._ts = None
            self.connect()
        else:
            self._ts = None
    @property
    def TableNames(self):
        if self._TableInfo is not None: return self._TableInfo.index.tolist()
        else: return []
    def getTable(self, table_name, args={}):
        TableClass = self._TableInfo.loc[table_name, "TableClass"]
        return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=args)")
    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if exchange not in ("SSE", "SZSE"): raise __QS_Error__("不支持交易所: '%s' 的交易日序列!" % exchange)
        if start_date is None: start_date = dt.date(1900, 1, 1)
        start_date = start_date.strftime("%Y%m%d")
        if end_date is None: end_date = dt.date.today()
        end_date = end_date.strftime("%Y%m%d")
        Dates = self._ts.trade_cal(exchange_id=exchange, start_date=start_date, end_date=end_date, fields="cal_date", is_open="1")
        if kwargs.get("output_type", "date")=="date":
            return [dt.datetime.strptime(iDate, "%Y%m%d").date() for iDate in Dates["cal_date"]]
        else:
            return [dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Dates["cal_date"]]
    # 获取指定日当前在市或者历史上出现过的全体 A 股 ID
    def _getAllAStock(self, date, is_current=True):
        Data = self._ts.stock_basic(exchange_id="", is_hs="", list_status="L", fields="ts_code, list_date, delist_date")
        Data = Data.append(self._ts.stock_basic(exchange_id="", is_hs="", list_status="D", fields="ts_code, list_date, delist_date"))
        Data = Data.append(self._ts.stock_basic(exchange_id="", is_hs="", list_status="P", fields="ts_code, list_date, delist_date"))
        date = date.strftime("%Y%m%d")
        Data = Data[Data["list_date"]<=date]
        if is_current: Data = Data[pd.isnull(Data["delist_date"]) | (Data["delist_date"]>date)]
        return sorted(Data["ts_code"])
    # 获取指定日当前或历史上的指数成份股ID, is_current=True: 获取指定日当天的ID, False:获取截止指定日历史上出现的 ID
    def getID(self, index_id="全体A股", date=None, is_current=True):
        if date is None: date = dt.date.today()
        if index_id=="全体A股": return self._getAllAStock(date=date, is_current=is_current)
        if not is_current: raise __QS_Error__("不支持提取 '%s' 的历史成分股!" % index_id)
        return self._ts.index_weight(index_code=index_id, trade_date=date.strftime("%Y%m%d"), fields="con_code")["con_code"].tolist()
    # 将 Excel 文件中的表和字段信息导入信息文件
    def importInfo(self, excel_file_path):
        DF = pd.read_excel(excel_file_path, "TableInfo").set_index(["TableName"])
        writeNestedDict2HDF5(DF, self._InfoFilePath, "/TableInfo")
        DF = pd.read_excel(excel_file_path, 'FactorInfo').set_index(['TableName', 'FieldName'])
        writeNestedDict2HDF5(DF, self._InfoFilePath, "/FactorInfo")