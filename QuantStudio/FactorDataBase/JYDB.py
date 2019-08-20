# coding=utf-8
"""聚源数据库"""
import re
import os
import shelve
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Int, Str, Range, Bool, List, ListStr, Dict, Function, Password, Either, Float

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries, getDateSeries
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.FileFun import getShelveFileSuffix
from QuantStudio import __QS_Object__, __QS_Error__, __QS_LibPath__, __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable
from QuantStudio.FactorDataBase.FDBFun import adjustDateTime

# 将信息源文件中的表和字段信息导入信息文件
def _importInfo(info_file, info_resource):
    TableInfo = pd.read_excel(info_resource, "TableInfo").set_index(["TableName"])
    FactorInfo = pd.read_excel(info_resource, "FactorInfo").set_index(['TableName', 'FieldName'])
    ExchangeInfo = pd.read_excel(info_resource, "ExchangeInfo", dtype={"ExchangeCode":"O"}).set_index(["ExchangeCode"])
    try:
        from QuantStudio.Tools.DataTypeFun import writeNestedDict2HDF5
        writeNestedDict2HDF5(TableInfo, info_file, "/TableInfo")
        writeNestedDict2HDF5(FactorInfo, info_file, "/FactorInfo")
        writeNestedDict2HDF5(ExchangeInfo, info_file, "/ExchangeInfo")
    except:
        pass
    return (TableInfo, FactorInfo, ExchangeInfo)

# 更新信息文件
def _updateInfo(info_file, info_resource):
    if not os.path.isfile(info_file):
        print("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    elif (os.path.getmtime(info_resource)>os.path.getmtime(info_file)):
        print("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % info_resource)
    else:
        try:
            from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
            return (readNestedDictFromHDF5(info_file, ref="/TableInfo"), readNestedDictFromHDF5(info_file, ref="/FactorInfo"), readNestedDictFromHDF5(info_file, ref="/ExchangeInfo"))
        except:
            print("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    if not os.path.isfile(info_resource): raise __QS_Error__("缺失数据库信息源文件: %s" % info_resource)
    return _importInfo(info_file, info_resource)

# 给 ID 去后缀
def deSuffixID(ids, sep='.'):
    return [(".".join(iID.split(".")[:-1]) if iID.find(".")!=-1 else iID) for iID in ids]
# 根据字段的数据类型确定 QS 的数据类型
def _identifyDataType(field_data_type):
    if (field_data_type.find("number")!=-1) or (field_data_type.find("int")!=-1) or (field_data_type.find("decimal")!=-1) or (field_data_type.find("float")!=-1):
        return "double"
    elif field_data_type.find("date")!=-1:
        return "datetime"
    else:
        return "string"
class _DBTable(FactorTable):
    def _getIDField(self):
        if self._MainTableName is None: return self._DBTableName+"."+self._IDField
        Exchange = self._FactorDB._TableInfo.loc[self.Name, "Exchange"]
        if pd.isnull(Exchange): return self._MainTableName+"."+self._MainTableID
        ExchangeField, ExchangeCodes = Exchange.split(":")
        ExchangeField = self._MainTableName + "." + ExchangeField
        ExchangeCodes = ExchangeCodes.split(",")
        ExchangeInfo = self._FactorDB._ExchangeInfo
        IDField = "CASE "
        for iCode in ExchangeCodes:
            IDField += "WHEN "+ExchangeField+"="+iCode+" THEN CONCAT("+self._MainTableName+"."+self._MainTableID+", '"+ExchangeInfo.loc[iCode, "Suffix"]+"') "
        DefaultSuffix = self._FactorDB._TableInfo.loc[self.Name, "DefaultSuffix"]
        if pd.isnull(DefaultSuffix):
            IDField += "ELSE "+self._MainTableName+"."+self._MainTableID+" END"
        else:
            IDField += "ELSE CONCAT("+self._MainTableName+"."+self._MainTableID+", '"+DefaultSuffix+"') END"
        return IDField
    def _adjustRawDataByRelatedField(self, raw_data, fields):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        RelatedFields = FactorInfo["RelatedSQL"].loc[fields]
        RelatedFields = RelatedFields[pd.notnull(RelatedFields)]
        if RelatedFields.shape[0]==0: return raw_data
        for iField in RelatedFields.index:
            iOldData = raw_data.pop(iField)
            iDataType = _identifyDataType(FactorInfo.loc[iField, "DataType"])
            if iDataType=="double":
                iNewData = pd.Series(np.nan, index=raw_data.index, dtype="float")
            else:
                iNewData = pd.Series(None, index=raw_data.index, dtype="O")
            iSQLStr = FactorInfo.loc[iField, "RelatedSQL"]
            if iSQLStr[0]=="{":
                iMapInfo = eval(iSQLStr).items()
            else:
                if iSQLStr.find("{Keys}")==-1:
                    iMapInfo = self._FactorDB.fetchall(iSQLStr.format(TablePrefix=self._FactorDB.TablePrefix))
                else:
                    Keys = ", ".join([str(iKey) for iKey in iOldData[pd.notnull(iOldData)].unique()])
                    iMapInfo = self._FactorDB.fetchall(iSQLStr.format(TablePrefix=self._FactorDB.TablePrefix, Keys=Keys))
            for jVal, jRelatedVal in iMapInfo:
                if pd.notnull(jVal):
                    iNewData[iOldData==jVal] = jRelatedVal
                else:
                    iNewData[pd.isnull(iOldData)] = jRelatedVal
            raw_data[iField] = iNewData
        return raw_data
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
                if (iDataType.find("number")!=-1) or (iDataType.find("int")!=-1) or (iDataType.find("decimal")!=-1) or (iDataType.find("float")!=-1): MetaData.iloc[i] = "double"
                else: MetaData.iloc[i] = "string"
            return MetaData
        elif key=="Description": return FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType"),
                                 "Description":self.getFactorMetaData(factor_names, key="Description")})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))


class _FeatureTable(_DBTable):
    """特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = self._getIDField()
        SQLStr = "SELECT DISTINCT "+IDField+" AS ID "
        SQLStr += "FROM "+self._DBTableName+" "
        if self._MainTableName!=self._DBTableName:
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
        if pd.notnull(self._MainTableCondition): SQLStr += "WHERE "+self._MainTableCondition+" "
        SQLStr += "ORDER BY ID"
        return [str(iRslt[0]) for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
        # 时间点默认是当天, ID 默认是 [000001.SH], 特别参数: 回溯天数
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        IDField = self._getIDField()
        # 形成SQL语句, ID, 因子数据
        SQLStr = "SELECT "+IDField+" AS ID, "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        ids = deSuffixID(ids)
        if self._MainTableName==self._DBTableName:
            SQLStr += "WHERE ("+genSQLInCondition(self._DBTableName+"."+self._IDField, ids, is_str=True, max_num=1000)+") "
        else:
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
            SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, ids, is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "ORDER BY ID"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID"]+factor_names)
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        RawData["ID"] = [str(iID) for iID in RawData["ID"]]
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        raw_data = raw_data.set_index(["ID"])
        if raw_data.index.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.loc[:, factor_names]
        Data = pd.Panel(raw_data.values.T.reshape((raw_data.shape[1], raw_data.shape[0], 1)).repeat(len(dts), axis=2), items=factor_names, major_axis=raw_data.index, minor_axis=dts).swapaxes(1, 2)
        return Data.loc[:, :, ids]


class _MappingTable(_DBTable):
    """映射因子表"""
    OnlyStartFilled = Bool(False, label="只填起始日", arg_type="Bool", order=0)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]
        self._StartDateField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="StartDate"].iloc[0]
        self._EndDateField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="EndDate"].iloc[0]
        self._EndDateIncluded = FactorInfo[FactorInfo["FieldType"]=="EndDate"]["Supplementary"].iloc[0]
        self._EndDateIncluded = (pd.isnull(self._EndDateIncluded) or (self._EndDateIncluded=="包含"))
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Either(Str(""), ListStr([]), arg_type="String", label=iCondition, order=i+2))
            iConditionVal = FactorInfo.loc[iCondition, "Supplementary"]
            if not iConditionVal:
                self[iCondition] = ""
            else:
                iConditionVal = str(iConditionVal).strip().split(",")
                if len(iConditionVal)==1: self[iCondition] = iConditionVal[0]
                else: self[iCondition] = iConditionVal
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        return FactorInfo.index.tolist()
    def _genConditionSQLStr(self, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        SQLStr = ""
        for iConditionField in self._ConditionFields:
            iConditionVal = args.get(iConditionField, self[iConditionField])
            if not iConditionVal: continue
            if _identifyDataType(FactorInfo.loc[iConditionField, "DataType"])=="string":
                SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iConditionField]+"='"+iConditionVal+"' AND "
            else:
                SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iConditionField]+"="+iConditionVal+" AND "
        return SQLStr[:-5]
    def getCondition(self, icondition, ids=None, dts=None):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[icondition]+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        if ids is not None:
            ids = deSuffixID(ids)
            SQLStr += "AND ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, ids, is_str=True, max_num=1000)+") "
        else: SQLStr += "AND "+self._MainTableName+"."+self._IDField+" IS NOT NULL "        
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        if dts is not None:
            Dates = list({iDT.strftime("%Y-%m-%d") for iDT in dts})
            SQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+self._DateField, Dates, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[icondition]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回给定时点 idt 有数据的所有 ID
    # 如果 idt 为 None, 将返回所有有记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = self._getIDField()
        SQLStr = "SELECT DISTINCT "+IDField+" AS ID, "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        if idt is not None:
            SQLStr += "WHERE "+self._DBTableName+"."+self._StartDateField+"<='"+idt.strftime("%Y-%m-%d")+"' "
            if self._EndDateIncluded:
                SQLStr += "AND "+self._DBTableName+"."+self._EndDateField+">='"+idt.strftime("%Y-%m-%d")+"' "
            else:
                SQLStr += "AND "+self._DBTableName+"."+self._EndDateField+">'"+idt.strftime("%Y-%m-%d")+"' "        
        else: SQLStr += "WHERE "+self._DBTableName+"."+self._StartDateField+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SubSQLStr = self._genConditionSQLStr(args=args)
        if SubSQLStr: SQLStr += "AND "+SubSQLStr+" "
        SQLStr += "ORDER BY ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回给定 ID iid 的起始日期距今的时点序列
    # 如果 idt 为 None, 将以表中最小的起始日期作为起点
    # 忽略 ifactor_name    
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._IDField, self._StartDateField]]
        SQLStr = "SELECT MIN("+self._DBTableName+"."+self._StartDateField+") "# 起始日期
        SQLStr += "FROM "+self._DBTableName
        if iid is not None:
            iID = deSuffixID([iid])[0]
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+iID+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else: SQLStr += "WHERE "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        SubSQLStr = self._genConditionSQLStr(args=args)
        if SubSQLStr: SQLStr += "AND "+SubSQLStr
        StartDT = dt.datetime.strptime(self._FactorDB.fetchall(SQLStr)[0][0], "%Y-%m-%d")
        if start_dt is not None: StartDT = max((StartDT, start_dt))
        if end_dt is None: end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=StartDT, end_dt=end_dt, timedelta=dt.timedelta(1))
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in iFactor.ArgNames if iArgName not in ("只填起始日",)])
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {"FactorNames":[iFactor.Name], 
                                               "RawFactorNames":{iFactor._NameInFT}, 
                                               "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                               "args":iFactor.Args.copy()}
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IDField = self._getIDField()
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        # 形成SQL语句, ID, 开始日期, 结束日期, 因子数据
        SQLStr = "SELECT "+IDField+" AS ID, "
        SQLStr += self._DBTableName+"."+self._StartDateField+", "
        SQLStr += self._DBTableName+"."+self._EndDateField+", "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        ids = deSuffixID(ids)
        SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, ids, is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND (("+self._DBTableName+"."+self._EndDateField+">='"+StartDate.strftime("%Y-%m-%d")+"') "
        SQLStr += "OR ("+self._DBTableName+"."+self._EndDateField+" IS NULL) "
        SQLStr += "OR ("+self._DBTableName+"."+self._EndDateField+"<"+self._DBTableName+"."+self._StartDateField+")) "
        SQLStr += "AND "+self._DBTableName+"."+self._StartDateField+"<='"+EndDate.strftime("%Y-%m-%d")+"' "
        SubSQLStr = self._genConditionSQLStr(args=args)
        if SubSQLStr: SQLStr += "AND "+SubSQLStr+" "
        SQLStr += "ORDER BY ID, "
        SQLStr += self._DBTableName+"."+self._StartDateField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "QS_起始日", "QS_结束日"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "QS_起始日", "QS_结束日"]+factor_names)
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData
    def _checkMultiMapping(self, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for iConditionField in self._ConditionFields:
            iConditionVal = args.get(iConditionField, self[iConditionField])
            if (not isinstance(iConditionVal, str)) or (iConditionVal==""):
                return True
        return False
    def _calcMultiMappingData(self, raw_data, factor_names, ids, dts, args={}):
        Data, nFactor = {}, len(factor_names)
        raw_data.set_index(["ID"], inplace=True)
        raw_data["QS_结束日"] = raw_data["QS_结束日"].where(pd.notnull(raw_data["QS_结束日"]), dts[-1]+dt.timedelta(1))
        if args.get("只填起始日", self.OnlyStartFilled):
            raw_data["QS_起始日"] = raw_data["QS_起始日"].where(raw_data["QS_起始日"]>=dts[0], dts[0])
            for iID in raw_data.index.unique():
                iRawData = raw_data.loc[[iID]].set_index(["QS_起始日"])
                iData = pd.DataFrame(index=dts, columns=factor_names, dtype="O")
                for jStartDate in iRawData.index.drop_duplicates():
                    iData.iloc[iData.index.searchsorted(jStartDate)] = iRawData.loc[[jStartDate], factor_names].values.T.tolist()
                Data[iID] = iData
                return pd.Panel(Data).swapaxes(0, 2).loc[:, :, ids]
        else:
            DeltaDT = dt.timedelta(int(not self._EndDateIncluded))
            for iID in raw_data.index.unique():
                iRawData = raw_data.loc[[iID]].set_index(["QS_起始日", "QS_结束日"])
                iData = pd.DataFrame(index=dts, columns=factor_names, dtype="O")
                for jStartDate, jEndDate in iRawData.index.drop_duplicates():
                    ijRawData = iRawData.loc[jStartDate].loc[[jEndDate], factor_names].values.T.tolist()
                    if pd.isnull(jEndDate) or (jEndDate<jStartDate):
                        iData.loc[jStartDate:] = [ijRawData] * iData.loc[jStartDate:].shape[0]
                    else:
                        jEndDate -= DeltaDT
                        iData.loc[jStartDate:jEndDate] = [ijRawData] * iData.loc[jStartDate:jEndDate].shape[0]
                Data[iID] = iData
                return pd.Panel(Data).swapaxes(0, 2).loc[:, :, ids]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        if self._checkMultiMapping(args=args): return self._calcMultiMappingData(raw_data, factor_names, ids, dts, args=args)
        raw_data.set_index(["ID"], inplace=True)
        Data, nFactor = {}, len(factor_names)
        if args.get("只填起始日", self.OnlyStartFilled):
            raw_data["QS_起始日"] = raw_data["QS_起始日"].where(raw_data["QS_起始日"]>=dts[0], dts[0])
            for iID in raw_data.index.unique():
                iRawData = raw_data.loc[[iID]].set_index(["QS_起始日"])
                iData = pd.DataFrame(index=dts, columns=factor_names)
                for jStartDate in iRawData.index:
                    iData.iloc[iData.index.searchsorted(jStartDate)] = iRawData.loc[jStartDate, factor_names]
                Data[iID] = iData
                return pd.Panel(Data).swapaxes(0, 2).loc[:, :, ids]
        else:
            DeltaDT = dt.timedelta(int(not self._EndDateIncluded))
            for iID in raw_data.index.unique():
                iRawData = raw_data.loc[[iID]]
                iData = pd.DataFrame(index=dts, columns=factor_names)
                for j in range(iRawData.shape[0]):
                    ijRawData = iRawData.iloc[j]
                    jStartDate, jEndDate = ijRawData["QS_起始日"], ijRawData["QS_结束日"]
                    if pd.isnull(jEndDate) or (jEndDate<jStartDate):
                        iData.loc[jStartDate:] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iData.loc[jStartDate:].shape[0], axis=0)
                    else:
                        jEndDate -= DeltaDT
                        iData.loc[jStartDate:jEndDate] = np.repeat(ijRawData[factor_names].values.reshape((1, nFactor)), iData.loc[jStartDate:jEndDate].shape[0], axis=0)
                Data[iID] = iData
            return pd.Panel(Data).swapaxes(0, 2).loc[:, :, ids]
class _ConstituentTable(_DBTable):
    """成份因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]        
        FactorInfo = fdb._FactorInfo.loc[name]
        self._GroupField = FactorInfo[FactorInfo["FieldType"]=="Group"].index[0]
        self._IDField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]
        self._InDateField = FactorInfo[FactorInfo["FieldType"]=="InDate"].index[0]
        self._OutDateField = FactorInfo[FactorInfo["FieldType"]=="OutDate"].index[0]
        self._CurSignField = FactorInfo[FactorInfo["FieldType"]=="CurSign"].index
        if self._CurSignField.shape[0]==0: self._CurSignField = None
        else: self._CurSignField = self._CurSignField[0]
        #self._IndexIDField = self._MainTableName+".SecuCode AS IndexID"
        #self._IndexIDField = "CASE WHEN ("+self._MainTableName+".SecuMarket = 83) THEN CONCAT("+self._MainTableName+"."+self._MainTableID+", '.SH') "
        #self._IndexIDField += "CASE WHEN ("+self._MainTableName+".SecuMarket = 90) THEN CONCAT("+self._MainTableName+"."+self._MainTableID+", '.SZ') "
        #self._IndexIDField += "ELSE "+self._MainTableName+"."+self._MainTableID+" END AS IndexID"
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        if not hasattr(self, "_IndexIDs"):# [指数 ID]
            FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._GroupField]]
            IndexID = self._MainTableName+"."+self._MainTableID
            SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+FieldDict[self._GroupField]+" "
            SQLStr += "FROM "+self._DBTableName+" "
            #SQLStr += "INNER JOIN "+self._MainTableName+" "
            #SQLStr += "ON "+self._DBTableName+"."+FieldDict[self._GroupField]+"="+self._MainTableName+".InnerCode "
            SQLStr += "ORDER BY "+self._DBTableName+"."+FieldDict[self._GroupField]
            self._IndexIDs = [str(iRslt[0]) for iRslt in self._FactorDB.fetchall(SQLStr)]
            #self._IndexIDs = pd.DataFrame(np.array(self._FactorDB.fetchall(SQLStr)), columns=["IndexID", "InnerCode"])
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
    def getID(self, ifactor_name=None, idt=None, args={}, **kwargs):# TODO
        Fields = [self._IDField, self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+FieldDict[self._IDField]+" "# ID
        SQLStr += "FROM "+self._DBTableName+" "
        if ifactor_name is not None:
            SQLStr += "WHERE "+self._DBTableName+"."+FieldDict[self._GroupField]+"='"+ifactor_name+"' "
        else:
            SQLStr += "WHERE "+self._DBTableName+"."+FieldDict[self._GroupField]+" IS NOT NULL "
        if idt is not None:
            idt = idt.strftime("%Y%m%d")
            SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._InDateField]+"<='"+idt+"' "
            if kwargs.get("is_current", True):
                SQLStr += "AND (("+self._DBTableName+"."+FieldDict[self._OutDateField]+">'"+idt+"') "
                if self._CurSignField:
                    SQLStr += "OR ("+self._DBTableName+"."+FieldDict[self._CurSignField]+"=1)) "
                else:
                    SQLStr += "OR ("+self._DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        SQLStr += "ORDER BY "+self._DBTableName+"."+FieldDict[self._IDField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回指数 ID 为 ifactor_name 包含成份股 iid 的时间点序列
    # 如果 iid 为 None, 将返回指数 ifactor_name 的有记录数据的时间点序列
    # 如果 ifactor_name 为 None, 返回数据库表中有记录的所有时间点
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):# TODO
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        Fields = [self._IDField, self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        if iid is not None:
            SQLStr = "SELECT "+self._DBTableName+"."+FieldDict[self._InDateField]+" "# 纳入日期
            SQLStr += self._DBTableName+"."+FieldDict[self._OutDateField]+" "# 剔除日期
            SQLStr += "FROM "+self._DBTableName+" "
            SQLStr += "WHERE "+self._DBTableName+"."+FieldDict[self._InDateField]+" IS NOT NULL "
            if ifactor_name is not None:
                SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._GroupField]+"='"+ifactor_name+"' "
            SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._IDField]+"='"+iid+"' "
            if start_dt is not None:
                SQLStr += "AND (("+self._DBTableName+"."+FieldDict[self._OutDateField]+">'"+start_dt.strftime("%Y%m%d")+"') "
                SQLStr += "OR ("+self._DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL))"
            if end_dt is not None:
                SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._InDateField]+"<='"+end_dt.strftime("%Y%m%d")+"' "
            SQLStr += "ORDER BY "+self._DBTableName+"."+FieldDict[self._InDateField]
            Data = self._FactorDB.fetchall(SQLStr)
            DateTimes = set()
            for iStartDate, iEndDate in Data:
                iStartDT = dt.datetime.strptime(iStartDate, "%Y%m%d")
                if iEndDate is None: iEndDT = (dt.datetime.now() if end_dt is None else end_dt)
                else: iEndDT = dt.datetime.strptime(iEndDate, "%Y%m%d")
                DateTimes = DateTimes.union(getDateTimeSeries(start_dt=iStartDT, end_dt=iEndDT, timedelta=dt.timedelta(1)))
            return sorted(DateTimes)
        SQLStr = "SELECT MIN("+self._DBTableName+"."+FieldDict[self._InDateField]+") "# 纳入日期
        SQLStr += "FROM "+self._DBTableName
        if ifactor_name is not None:
            SQLStr += " WHERE "+self._DBTableName+"."+FieldDict[self._GroupField]+"='"+ifactor_name+"'"
        StartDT = dt.datetime.strptime(self._FactorDB.fetchall(SQLStr)[0][0], "%Y%m%d")
        if start_dt is not None:
            StartDT = max((StartDT, start_dt))
        if end_dt is None:
            end_dt = dt.datetime.combine(dt.date.today(), dt.time(0))
        return getDateTimeSeries(start_dt=start_dt, end_dt=end_dt, timedelta=dt.timedelta(1))
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IDField = self._getIDField()
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        Fields = [self._GroupField, self._InDateField, self._OutDateField]
        if self._CurSignField: Fields.append(self._CurSignField)
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        # 指数中成份股 ID, 指数证券 ID, 纳入日期, 剔除日期, 最新标志
        SQLStr = "SELECT "+self._DBTableName+"."+FieldDict[self._GroupField]+", "# 指数证券 ID
        SQLStr += IDField+" AS ID, "# ID
        SQLStr += self._DBTableName+"."+FieldDict[self._InDateField]+", "# 纳入日期
        SQLStr += self._DBTableName+"."+FieldDict[self._OutDateField]+", "# 剔除日期
        if self._CurSignField:
            SQLStr += self._DBTableName+"."+FieldDict[self._CurSignField]+" "# 最新标志
        else:
            SQLStr += "NULL AS CurSign "# 最新标志
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        ids = deSuffixID(ids)
        SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, ids, is_str=True, max_num=1000)+")                 "
        SQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+FieldDict[self._GroupField], factor_names, is_str=False, max_num=1000)+") "
        SQLStr += "AND (("+self._DBTableName+"."+FieldDict[self._OutDateField]+">'"+StartDate.strftime("%Y%m%d")+"') "
        SQLStr += "OR ("+self._DBTableName+"."+FieldDict[self._OutDateField]+" IS NULL)) "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._InDateField]+"<='"+EndDate.strftime("%Y%m%d")+"' "
        SQLStr += "ORDER BY "+self._DBTableName+"."+FieldDict[self._GroupField]+", ID, "
        SQLStr += self._DBTableName+"."+FieldDict[self._InDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["IndexID", "SecurityID", "InDate", "OutDate", "CurSign"])
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["IndexID", "SecurityID", "InDate", "OutDate", "CurSign"])
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        DateSeries = getDateSeries(StartDate, EndDate)
        Data = {}
        for iIndexID in factor_names:
            iRawData = raw_data[raw_data["IndexID"]==int(iIndexID)].set_index(["SecurityID"])
            iData = pd.DataFrame(0, index=DateSeries, columns=pd.unique(iRawData.index))
            for jID in iData.columns:
                jIDRawData = iRawData.loc[[jID]]
                for k in range(jIDRawData.shape[0]):
                    kStartDate = jIDRawData["InDate"].iloc[k].date()
                    kEndDate = (jIDRawData["OutDate"].iloc[k].date()-dt.timedelta(1) if jIDRawData["OutDate"].iloc[k] is not None else dt.date.today())
                    iData[jID].loc[kStartDate:kEndDate] = 1
            Data[iIndexID] = iData
        Data = pd.Panel(Data)
        if Data.minor_axis.intersection(ids).shape[0]==0: return pd.Panel(0.0, items=factor_names, major_axis=dts, minor_axis=ids)
        Data = Data.loc[factor_names, :, ids]
        Data.major_axis = [dt.datetime.combine(iDate, dt.time(0)) for iDate in Data.major_axis]
        Data.fillna(value=0, inplace=True)
        return adjustDateTime(Data, dts, fillna=True, method="bfill")


# 行情因子表, 表结构特征:
# 日期字段, 表示数据填充的时点;
# 条件字段, 作为条件过滤记录; 可能存在多个条件字段
# 在设定某些条件下, 数据填充时点和 ID 可以唯一标志一行记录
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class _MarketTable(_DBTable):
    """行情因子表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]# ID 字段
        self._DateField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="Date"].iloc[0]# 发布日期字段
        self._ConditionFields = FactorInfo[FactorInfo["FieldType"]=="Condition"].index.tolist()# 所有的条件字段列表
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+3))
            iConditionVal = FactorInfo.loc[iCondition, "Supplementary"]
            if iConditionVal: self[iCondition] = str(iConditionVal)
    def _genConditionSQLStr(self, args={}):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        SQLStr = ""
        for iConditionField in self._ConditionFields:
            iConditionVal = args.get(iConditionField, self[iConditionField])
            if iConditionVal:
                if _identifyDataType(FactorInfo.loc[iConditionField, "DataType"])=="string":
                    SQLStr += "AND "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iConditionField]+"='"+iConditionVal+"' "
                else:
                    SQLStr += "AND "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iConditionField]+"="+iConditionVal+" "
        return SQLStr[:-1]
    def getCondition(self, icondition, ids=None, dts=None):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[icondition]+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        if ids is not None:
            if self._SecurityType=="A股": ids = deSuffixID(ids)
            SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, ids, is_str=True, max_num=1000)+") "
        else: SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "        
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        if dts is not None:
            Dates = list({iDT.strftime("%Y-%m-%d") for iDT in dts})
            SQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+self._DateField, Dates, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[icondition]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = self._getIDField()
        SQLStr = "SELECT DISTINCT "+IDField+" AS ID "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        if idt is not None: SQLStr += "WHERE "+self._DBTableName+"."+self._DateField+"='"+idt.strftime("%Y-%m-%d")+"' "
        else: SQLStr += "WHERE "+self._DBTableName+"."+self._DateField+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+self._DateField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        if iid is not None:
            iID = deSuffixID([iid])[0]
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+iID+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else: SQLStr += "WHERE "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+self._DBTableName+"."+self._DateField+">='"+start_dt.strftime("%Y-%m-%d")+"' "
        if end_dt is not None: SQLStr += "AND "+self._DBTableName+"."+self._DateField+"<='"+end_dt.strftime("%Y-%m-%d")+"' "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+self._DateField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        DateConditionGroup = {}
        for iFactor in factors:
            iDateConditions = (self._DateField, ";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in iFactor.ArgNames if iArgName!="回溯天数"]))
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
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        IDField = self._getIDField()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        SubSQLStr = "SELECT "+self._MainTableName+"."+self._MainTableID+", "
        SubSQLStr += "MAX("+self._DBTableName+"."+self._DateField+") "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "INNER JOIN "+self._MainTableName+" "
        SubSQLStr += "ON "+self._JoinCondition+" "
        SubSQLStr += "WHERE "+self._DBTableName+"."+self._DateField+"<'"+end_date.strftime("%Y-%m-%d")+"' "
        SubSQLStr += "AND ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SubSQLStr += "AND "+self._MainTableCondition+" "
        SubSQLStr += "GROUP BY "+self._MainTableName+"."+self._MainTableID
        SQLStr = "SELECT "+self._DBTableName+"."+self._DateField+", "
        SQLStr += IDField+" AS ID, "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "WHERE ("+self._MainTableName+"."+self._MainTableID+", "+self._DBTableName+"."+self._DateField+") IN ("+SubSQLStr+")"
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += self._genConditionSQLStr(args=args)
        return SQLStr
    def _genSQLStr(self, factor_names, ids, start_date, end_date, args={}):
        IDField = self._getIDField()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+self._DBTableName+"."+self._DateField+", "
        SQLStr += IDField+" AS ID, "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "WHERE "+self._DBTableName+"."+self._DateField+">='"+start_date.strftime("%Y-%m-%d")+"' "
        SQLStr += "AND "+self._DBTableName+"."+self._DateField+"<='"+end_date.strftime("%Y-%m-%d")+"' "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID, "+self._DBTableName+"."+self._DateField
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        LookBack = args.get("回溯天数", self.LookBack)
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        RawData = self._FactorDB.fetchall(self._genSQLStr(factor_names, ids, start_date=StartDate, end_date=EndDate, args=args))
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID"]+factor_names)
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["日期"]==dt.datetime.combine(StartDate,dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["日期", "ID"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "日期"])
        if RawData.shape[0]==0: return RawData
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Data = {}
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            iDataType = _identifyDataType(FactorInfo.loc[iFactorName, "DataType"])
            if iDataType=="double": iRawData = iRawData.astype("float")
            #elif iDataType=="datetime": iRawData = iRawData.applymap(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else None)
            Data[iFactorName] = iRawData
        Data = pd.Panel(Data).loc[factor_names]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        if np.isinf(LookBack):
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = Data.iloc[i].fillna(method="pad")
        else:
            Limits = LookBack*24.0*3600
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]
# 信息发布表, 表结构特征:
# 公告日期, 表示信息发布的时点;
# 截止日期, 表示信息有效的时点;
# 如果不忽略公告日期, 则以截止日期和公告日期的最大值作为数据填充的时点, 同一填充时点存在多个截止日期时以最大截止日期的记录值填充
# 如果忽略公告日期, 则以截止日期作为数据填充的时点, 必须保证截至日期具有唯一性
# 条件字段, 作为条件过滤记录; 可能存在多个条件字段
# 在设定某些条件下, 数据填充时点和 ID 可以唯一标志一行记录
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class _InfoPublTable(_MarketTable):
    """信息发布表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    IgnorePublDate = Bool(False, label="忽略公告日", arg_type="Bool", order=1)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        FactorInfo = fdb._FactorInfo.loc[name]
        self._AnnDateField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="AnnDate"].iloc[0]# 公告日期
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = self._getIDField()
        AnnDateField, EndDateField = self._DBTableName+"."+self._AnnDateField, self._DBTableName+"."+self._DateField
        SQLStr = "SELECT DISTINCT "+IDField+" AS ID "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        if idt is not None:
            SQLStr += "WHERE (CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END)='"+idt.strftime("%Y-%m-%d")+"' "
        else:
            SQLStr += "WHERE "+AnnDateField+" IS NOT NULL AND "+EndDateField+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        AnnDateField, EndDateField = self._DBTableName+"."+self._AnnDateField, self._DBTableName+"."+self._DateField
        SQLStr = "SELECT DISTINCT CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END AS DT"
        SQLStr += "FROM "+self._DBTableName+" "
        if iid is not None:
            iID = deSuffixID([iid])[0]
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+iID+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else: SQLStr += "WHERE "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        if start_dt is not None:
            SQLStr += "AND ("+AnnDateField+">='"+start_dt.strftime("%Y-%m-%d")+"' "
            SQLStr += "OR "+EndDateField+">='"+start_dt.strftime("%Y-%m-%d")+"') "
        if end_dt is not None:
            SQLStr += "AND ("+AnnDateField+"<='"+end_dt.strftime("%Y-%m-%d")+"' "
            SQLStr += "AND "+EndDateField+"<='"+end_dt.strftime("%Y-%m-%d")+"') "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY DT"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def _genNullIDSQLStr_InfoPubl(self, factor_names, ids, end_date, args={}):
        IDField = self._getIDField()
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        AnnDateField, EndDateField = self._DBTableName+"."+self._AnnDateField, self._DBTableName+"."+self._DateField
        SubSQLStr = "SELECT "+IDField+" AS ID, "
        SubSQLStr += self._DBTableName+"."+self._IDField+", "
        SubSQLStr += AnnDateField+" AS AnnDate, "
        SubSQLStr += "MAX("+EndDateField+") AS MaxEndDate "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "INNER JOIN "+self._MainTableName+" "
        SubSQLStr += "ON "+self._JoinCondition+" "
        SubSQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SubSQLStr += "AND "+self._MainTableCondition+" "
        SubSQLStr += self._genConditionSQLStr(args=args)+" "
        SubSQLStr += "AND ("+AnnDateField+"<'"+end_date.strftime("%Y-%m-%d")+"' "
        SubSQLStr += "AND "+EndDateField+"<'"+end_date.strftime("%Y-%m-%d")+"') "
        SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField+", "+AnnDateField
        SubSQLStr1 = "SELECT t.ID, "
        SubSQLStr1 += "t."+self._IDField+", "
        SubSQLStr1 += "MAX(CASE WHEN t.AnnDate>=t.MaxEndDate THEN t.AnnDate ELSE t.MaxEndDate END) AS DT "
        SubSQLStr1 += "FROM ("+SubSQLStr+") t GROUP BY t."+self._IDField
        SQLStr = "SELECT t1.DT, "
        SQLStr += "t1.ID, "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN ("+SubSQLStr1+") t1 "
        SQLStr += "ON (t1."+self._IDField+"="+self._DBTableName+"."+self._IDField+") "
        SQLStr += "AND (CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END=t1.DT)"
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if self.IgnorePublDate: return super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args={})
        IDField = self._getIDField()
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        LookBack = args.get("回溯天数", self.LookBack)
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        AnnDateField, EndDateField = self._DBTableName+"."+self._AnnDateField, self._DBTableName+"."+self._DateField
        SubSQLStr = "SELECT "+IDField+" AS ID, "
        SubSQLStr += self._DBTableName+"."+self._IDField+", "
        SubSQLStr += AnnDateField+" AS AnnDate, "
        SubSQLStr += "MAX("+EndDateField+") AS MaxEndDate "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "INNER JOIN "+self._MainTableName+" "
        SubSQLStr += "ON "+self._JoinCondition+" "
        SubSQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SubSQLStr += "AND "+self._MainTableCondition+" "
        ConditionSQLStr = self._genConditionSQLStr(args=args)
        SubSQLStr += ConditionSQLStr+" "
        SubSQLStr += "AND ("+AnnDateField+">='"+StartDate.strftime("%Y-%m-%d")+"' "
        SubSQLStr += "OR "+EndDateField+">='"+StartDate.strftime("%Y-%m-%d")+"') "
        SubSQLStr += "AND ("+AnnDateField+"<='"+EndDate.strftime("%Y-%m-%d")+"' "
        SubSQLStr += "AND "+EndDateField+"<='"+EndDate.strftime("%Y-%m-%d")+"') "
        SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField+", "+AnnDateField
        SQLStr = "SELECT CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END AS DT, "
        SQLStr += "t.ID, "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t."+self._IDField+"="+self._DBTableName+"."+self._IDField+") "
        SQLStr += "AND (t.MaxEndDate="+EndDateField+") "
        SQLStr += "WHERE TRUE "
        SQLStr += ConditionSQLStr+" "
        SQLStr += "ORDER BY t.ID, DT"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID"]+factor_names)
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["日期"]==dt.datetime.combine(StartDate,dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr_InfoPubl(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["日期", "ID"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "日期"])
        if RawData.shape[0]==0: return RawData
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData
# 多重信息发布表, 表结构特征:
# 公告日期, 表示获得信息的时点;
# 截止日期, 表示信息有效的时点;
# 如果不忽略公告日期, 则以截止日期和公告日期的最大值作为数据填充的时点, 同一填充时点存在多个截止日期时以最大截止日期的记录值填充
# 如果忽略公告日期, 则以截止日期作为数据填充的时点
# 条件字段, 作为条件过滤记录; 可能存在多个条件字段
# 数据填充时点和 ID 不能唯一标志一行记录, 对于每个 ID 每个数据填充时点可能存在多个数据, 将所有的数据以 list 组织, 如果算子参数不为 None, 以该算子作用在数据 list 上的结果为最终填充结果, 否则以数据 list 填充;
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class _MultiInfoPublTable(_InfoPublTable):
    """多重信息发布表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    IgnorePublDate = Bool(False, label="忽略公告日", arg_type="Bool", order=1)
    Operator = Function(lambda x: x.tolist(), arg_type="Function", label="算子", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        FactorInfo = fdb._FactorInfo.loc[name]
        self._OrderFields = FactorInfo[FactorInfo["Supplementary"]=="OrderField"].index.tolist()
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        FactorNames = list(set(factor_names).union(self._OrderFields))
        RawData = super().__QS_prepareRawData__(FactorNames, ids, dts, args=args)
        RawData = RawData.sort_values(by=["ID", "日期"]+self._OrderFields)
        return RawData.loc[:, ["日期", "ID"]+factor_names]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Operator = args.get("算子", self.Operator)
        if Operator is None: Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in factor_names:
            Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        Data = pd.Panel(Data).loc[factor_names, :, ids]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, ids]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        if np.isinf(LookBack):
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = Data.iloc[i].fillna(method="pad")
        else:
            Limits = LookBack*24.0*3600
            for i, iFactorName in enumerate(Data.items): Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]

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
# 财务因子表, 表结构特征:
# 报告期字段, 表示财报的报告期
# 公告日期字段, 表示财报公布的日期
class _FinancialTable(_DBTable):
    """财务因子表"""
    ReportDate = Enum("所有", "年报", "中报", "一季报", "三季报", Dict(), Function(), label="报告期", arg_type="SingleOption", order=0)
    ReportType = ListStr(["1"], label="报表类型", arg_type="MultiOption", order=1, option_range=("1", "2"))
    CalcType = Enum("最新", "单季度", "TTM", label="计算方法", arg_type="SingleOption", order=2)
    YearLookBack = Int(0, label="回溯年数", arg_type="Integer", order=3)
    PeriodLookBack = Int(0, label="回溯期数", arg_type="Integer", order=4)
    ExprFactor = Str("", label="业绩快报因子", arg_type="String", order=5)
    NoticeFactor = Str("", label="业绩预告因子", arg_type="String", order=6)
    IgnoreMissing = Bool(True, label="忽略缺失", arg_type="Bool", order=7)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]# ID 字段
        self._ANNDateField = FactorInfo[FactorInfo["FieldType"]=="ANNDate"].index
        if self._ANNDateField.shape[0]>0: self._ANNDateField = self._ANNDateField[0]
        else: self._ANNDateField = None
        self._ANNTypeField = FactorInfo[FactorInfo["FieldType"]=="ANNType"].index
        if self._ANNTypeField.shape[0]>0: self._ANNTypeField = self._ANNTypeField[0]
        else: self._ANNTypeField = None        
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._ReportTypeField = FactorInfo[FactorInfo["FieldType"]=="ReportType"].index
        if self._ReportTypeField.shape[0]==0: self._ReportTypeField = None
        else: self._ReportTypeField = self._ReportTypeField[0]
        self._TempData = {}
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        FactorNames = FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
        return [self._ANNDateField, self._ReportDateField, self._ReportTypeField]+FactorNames
    # 返回在给定时点 idt 之前有财务报告的 ID
    # 如果 idt 为 None, 将返回所有有财务报告的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = self._getIDField()
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._ANNDateField]]
        SQLStr = "SELECT DISTINCT "+IDField+" AS ID "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        if idt is not None: SQLStr += "WHERE "+self._DBTableName+"."+FieldDict[self._ANNDateField]+"<='"+idt.strftime("%Y-%m-%d")+"' "
        else: SQLStr += "WHERE "+self._DBTableName+"."+FieldDict[self._ANNDateField]+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "ORDER BY ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    # 返回在给定 ID iid 的有财务报告的公告时点
    # 如果 iid 为 None, 将返回所有有财务报告的公告时点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._ANNDateField]]
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+FieldDict[self._ANNDateField]+" "
        SQLStr += "FROM "+self._DBTableName+" "
        if iid is not None:
            iID = deSuffixID([iid])[0]
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+iID+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else: SQLStr += "WHERE "+self._DBTableName+"."+self._IDField+" IS NOT NULL         "
        if start_dt is not None: SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._ANNDateField]+">='"+start_dt.strftime("%Y-%m-%d")+"' "
        if end_dt is not None: SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._ANNDateField]+"<='"+end_dt.strftime("%Y-%m-%d")+"' "
        SQLStr += "ORDER BY "+self._DBTableName+"."+FieldDict[self._ANNDateField]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
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
        IDField = self._getIDField()
        if self._ReportTypeField is not None:
            Fields = list(set([self._ANNDateField, self._ANNTypeField, self._ReportDateField, self._ReportTypeField]+factor_names))
        else:
            Fields = list(set([self._ANNDateField, self._ANNTypeField, self._ReportDateField]+factor_names))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
        SQLStr = "SELECT "+IDField+" AS ID, "
        if self._FactorDB.DBType=="SQL Server":
            SQLStr += "TO_CHAR("+self._DBTableName+"."+FieldDict[self._ANNDateField]+",'YYYYMMDD'), "
            SQLStr += "TO_CHAR("+self._DBTableName+"."+FieldDict[self._ReportDateField]+",'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr += "DATE_FORMAT("+self._DBTableName+"."+FieldDict[self._ANNDateField]+",'%Y%m%d'), "
            SQLStr += "DATE_FORMAT("+self._DBTableName+"."+FieldDict[self._ReportDateField]+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr += "TO_CHAR("+self._DBTableName+"."+FieldDict[self._ANNDateField]+",'yyyyMMdd'), "
            SQLStr += "TO_CHAR("+self._DBTableName+"."+FieldDict[self._ReportDateField]+",'yyyyMMdd'), "
        if self._ReportTypeField is not None:
            SQLStr += self._DBTableName+"."+FieldDict[self._ReportTypeField]+", "
        else:
            SQLStr += "NULL AS ReportType, "
        for iField in factor_names:
            if iField in (self._IDField, self._ANNDateField, self._ReportDateField):
                SQLStr += self._DBTableName+"."+FieldDict[iField]+" AS "+iField+", "
            else:
                SQLStr += self._DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "WHERE "+self._DBTableName+"."+FieldDict[self._ANNTypeField]+" = 20 "
        if self._ReportTypeField is not None:
            SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._ReportTypeField]+" IN ("+", ".join(self.ReportType)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        ids = deSuffixID(ids)
        SQLStr += "AND ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, ids, is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY ID, "+self._DBTableName+"."+FieldDict[self._ANNDateField]+", "
        SQLStr += self._DBTableName+"."+FieldDict[self._ReportDateField]
        if self._ReportTypeField is not None:
            SQLStr += ", "+self._DBTableName+"."+FieldDict[self._ReportTypeField]+" DESC"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
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
        Data = pd.Panel(Data, major_axis=[dt.datetime.strptime(iDate, "%Y%m%d") for iDate in Dates], minor_axis=factor_names).swapaxes(0, 2)
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        NewData = {}
        for i, iFactorName in enumerate(factor_names):
            iDataType = _identifyDataType(FactorInfo.loc[iFactorName, "DataType"])
            if iDataType=="double": NewData[iFactorName] = Data.iloc[i].astype("float")
            #elif iDataType=="datetime": NewData[iFactorName] = Data.iloc[i].applymap(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else None)
            else: NewData[iFactorName] = Data.iloc[i]
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


# 财务指标因子表, 表结构特征:
# 报告期字段, 表示财报的报告期
# 无公告日期字段, 需另外补充完整
class _FinancialIndicatorTable(_FinancialTable):
    """财务指标因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        FactorNames = FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
        return [self._ReportDateField]+FactorNames
    def getID(self, ifactor_name=None, idt=None, args={}):# TODO
        return []
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):# TODO
        return []
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IDField = self._getIDField()
        Fields = list(set([self._ReportDateField]+factor_names))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[Fields]
        # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
        SQLStr = "SELECT "+IDField+" AS ID, "
        if self._FactorDB.DBType=="SQL Server":
            SQLStr += "TO_CHAR(LC_BalanceSheetAll.InfoPublDate, 'YYYYMMDD'), "
            SQLStr += "TO_CHAR("+self._DBTableName+"."+FieldDict[self._ReportDateField]+", 'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr += "DATE_FORMAT(LC_BalanceSheetAll.InfoPublDate, '%Y%m%d'), "
            SQLStr += "DATE_FORMAT("+self._DBTableName+"."+FieldDict[self._ReportDateField]+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr += "TO_CHAR(LC_BalanceSheetAll.InfoPublDate, 'yyyyMMdd'), "
            SQLStr += "TO_CHAR("+self._DBTableName+"."+FieldDict[self._ReportDateField]+",'yyyyMMdd'), "
        SQLStr += "NULL AS ReportType, "
        for iField in factor_names:
            if iField in (self._IDField, self._ReportDateField):
                SQLStr += self._DBTableName+"."+FieldDict[iField]+" AS "+iField+", "
            else:
                SQLStr += self._DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "INNER JOIN LC_BalanceSheetAll "
        SQLStr += "ON ("+self._DBTableName+"."+self._IDField+"=LC_BalanceSheetAll.CompanyCode "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._ReportDateField]+"=LC_BalanceSheetAll.EndDate) "
        SQLStr += "WHERE LC_BalanceSheetAll.BulletinType = 20 "
        SQLStr += "AND LC_BalanceSheetAll.IfMerged = 1 "
        SQLStr += "AND LC_BalanceSheetAll.IfAdjusted = 2 "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        SQLStr += "ORDER BY ID, LC_BalanceSheetAll.InfoPublDate, "
        SQLStr += self._DBTableName+"."+FieldDict[self._ReportDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "AnnDate", "ReportDate", "ReportType"]+factor_names)
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData

# 生成报告期-公告日期 SQL 查询语句
def genANN_ReportSQLStr(db_type, table_prefix, ids, report_period="1231"):
    DBTableName = table_prefix+"LC_BalanceSheetAll"
    # 提取财报的公告期数据, ID, 公告日期, 报告期
    SQLStr = "SELECT CASE WHEN "+table_prefix+"SecuMain.SecuMarket=83 THEN CONCAT("+table_prefix+"SecuMain.SecuCode, '.SH') "
    SQLStr += "WHEN "+table_prefix+"SecuMain.SecuMarket=90 THEN CONCAT("+table_prefix+"SecuMain.SecuCode, '.SZ') "
    SQLStr += "ELSE "+table_prefix+"SecuMain.SecuCode END AS ID, "
    if db_type=="SQL Server":
        SQLStr += "TO_CHAR("+DBTableName+".InfoPublDate,'YYYYMMDD'), "
        ReportDateStr = "TO_CHAR("+DBTableName+".EndDate,'YYYYMMDD')"
    elif db_type=="MySQL":
        SQLStr += "DATE_FORMAT("+DBTableName+".InfoPublDate,'%Y%m%d'), "
        ReportDateStr = "DATE_FORMAT("+DBTableName+".EndDate,'%Y%m%d')"
    elif db_type=="Oracle":
        SQLStr += "TO_CHAR("+DBTableName+".InfoPublDate,'yyyyMMdd'), "
        ReportDateStr = "TO_CHAR("+DBTableName+".EndDate,'yyyyMMdd')"
    SQLStr += ReportDateStr+" "
    SQLStr += "FROM "+DBTableName+" "
    SQLStr += "INNER JOIN "+table_prefix+"SecuMain "
    SQLStr += "ON "+table_prefix+"SecuMain.CompanyCode="+DBTableName+".CompanyCode "
    SQLStr += "WHERE "+table_prefix+"SecuMain.SecuCategory=1 AND "+table_prefix+"SecuMain.SecuMarket IN (83,90) "    
    SQLStr += "AND ("+genSQLInCondition(table_prefix+"SecuMain.SecuCode", deSuffixID(ids), is_str=True, max_num=1000)+") "
    if report_period is not None:
        SQLStr += "AND "+ReportDateStr+" LIKE '%"+report_period+"' "
    SQLStr += "ORDER BY "+table_prefix+"SecuMain.SecuCode, "
    SQLStr += DBTableName+".InfoPublDate, "+DBTableName+".EndDate"
    return SQLStr
def _prepareReportANNRawData(fdb, ids):
    SQLStr = genANN_ReportSQLStr(fdb.DBType, fdb.TablePrefix, ids, report_period="1231")
    RawData = fdb.fetchall(SQLStr)
    if not RawData: RawData =  pd.DataFrame(columns=["ID", "公告日期", "报告期"])
    else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "公告日期", "报告期"])
    return RawData
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
            iInterIDs = sorted(AllIDs.intersection(iIDs))
            iData = raw_data.loc[iInterIDs]
            for jFactorName in factor_names:
                ijData = iData[CommonCols+[jFactorName]].reset_index()
                if isANNReport: ijData.columns.name = raw_data_dir+os.sep+iPID+os.sep+report_ann_file
                iFile[jFactorName] = ijData
            iFile["_QS_IDs"] = iIDs
    return 0
class _AnalystConsensusTable(_DBTable):
    """分析师汇总表"""
    CalcType = Enum("FY0", "FY1", "FY2", "Fwd12M", label="计算方法", arg_type="SingleOption", order=0)
    Period = Enum(30,60,90,180, label="周期", arg_type="SingleOption", order=1)
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]# ID 字段
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._PeriodField = FactorInfo[FactorInfo["FieldType"]=="Period"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = 'JY财务年报-公告日期'
        self._ANN_ReportFileSuffix = getShelveFileSuffix()
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        return
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        IDField = self._getIDField()
        CalcType = args.get("计算方法", self.CalcType)
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._DateField, self._ReportDateField, self._PeriodField]+factor_names]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成SQL语句, 日期, ID, 报告期, 数据
        if self._FactorDB.DBType=="SQL Server":
            SQLStr = "SELECT TO_CHAR("+self._DBTableName+"."+FieldDict[self._DateField]+",'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr = "SELECT DATE_FORMAT("+self._DBTableName+"."+FieldDict[self._DateField]+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr = "SELECT TO_CHAR("+self._DBTableName+"."+FieldDict[self._DateField]+",'yyyyMMdd'), "
        SQLStr += IDField+" AS ID, "
        SQLStr += "CONCAT("+self._DBTableName+"."+FieldDict[self._ReportDateField]+", '1231') AS ReportDate, "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._PeriodField]+"="+str(args.get("周期", self.Period))+" "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y-%m-%d")+"' "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y-%m-%d")+"' "
        SQLStr += "ORDER BY ID, "+self._DBTableName+"."+FieldDict[self._DateField]+", "+self._DBTableName+"."+FieldDict[self._ReportDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=['日期','ID','报告期']+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=['日期','ID','报告期']+factor_names)
        RawData._QS_ANNReport = (CalcType!="Fwd12M")
        if RawData.shape[0]==0: return RawData
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
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

# f: 该算子所属的因子对象或因子表对象
# idt: 当前所处的时点
# iid: 当前待计算的 ID
# x: 当期的数据, 分析师评级时为: DataFrame(columns=["日期", ...]), 分析师盈利预测时为: [DataFrame(columns=["日期", "报告期", ...])], list的长度为向前年数
# args: 参数, {参数名:参数值}
def _DefaultOperator(f, idt, iid, x, args):
    return np.nan
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
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]# ID 字段
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._InstituteField = FactorInfo[FactorInfo["FieldType"]=="Institute"].index[0]
        self._ReportDateField = FactorInfo[FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = "JY财务年报-公告日期"
        self._ANN_ReportFileSuffix = getShelveFileSuffix()
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
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
        IDField = self._getIDField()
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("周期", self.Period))
        AdditiveFields = args.get("附加字段", self.AdditionalFields)
        DeduplicationFields = args.get("去重字段", self.Deduplication)
        AllFields = list(set(factor_names+AdditiveFields+DeduplicationFields))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._DateField, self._ReportDateField]+AllFields]
        # 形成SQL语句, 日期, ID, 报告期, 研究机构, 因子数据
        if self._FactorDB.DBType=="SQL Server":
            SQLStr = "SELECT TO_CHAR("+self._DBTableName+"."+FieldDict[self._DateField]+",'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr = "SELECT DATE_FORMAT("+self._DBTableName+"."+FieldDict[self._DateField]+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr = "SELECT TO_CHAR("+self._DBTableName+"."+FieldDict[self._DateField]+",'yyyyMMdd'), "
        SQLStr += IDField+" AS ID, "
        SQLStr += "CONCAT("+self._DBTableName+"."+FieldDict[self._ReportDateField]+", '1231') AS ReportDate, "
        for iField in AllFields: SQLStr += self._DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y-%m-%d")+"' "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y-%m-%d")+"' "
        SQLStr += "ORDER BY ID, "+self._DBTableName+"."+FieldDict[self._DateField]+", "+self._DBTableName+"."+FieldDict[self._ReportDateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID", self._ReportDateField]+AllFields)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID", self._ReportDateField]+AllFields)
        RawData._QS_ANNReport = True
        if RawData.shape[0]==0: return RawData
        RawData = self._adjustRawDataByRelatedField(RawData, AllFields)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Dates = sorted({dt.datetime.combine(iDT.date(), dt.time(0)) for iDT in dts})
        DeduplicationFields = args.get("去重字段", self.Deduplication)
        AdditionalFields = list(set(args.get("附加字段", self.AdditionalFields)+DeduplicationFields))
        AllFields = list(set(factor_names+AdditionalFields))
        ANNReportPath = raw_data.columns.name
        raw_data = raw_data.loc[:, ["日期", "ID", self._ReportDateField]+AllFields].set_index(["ID"])
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
        AllIDs = set(raw_data.index)
        Data = {}
        for kFactorName in factor_names:
            if DataType=="double": kData = np.full(shape=(len(Dates), len(ids)), fill_value=np.nan)
            else: kData = np.full(shape=(len(Dates), len(ids)), fill_value=None, dtype="O")
            kFields = ["日期", self._ReportDateField, kFactorName]+AdditionalFields
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
                                ijTemp[DeduplicationFields] = ijTemp[DeduplicationFields].astype("O")
                                iijRawData = pd.merge(ijTemp, iijRawData, how="left", left_on=DeduplicationFields+["日期"], right_on=DeduplicationFields+["日期"])
                            x.append(iijRawData)
                    kData[i, j] = Operator(self, iDate, jID, x, ModelArgs)
            Data[kFactorName] = kData
        return pd.Panel(Data, major_axis=Dates, minor_axis=ids).loc[factor_names, dts]

class _AnalystRatingDetailTable(_DBTable):
    """分析师投资评级明细表"""
    Operator = Function(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
    ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
    AdditionalFields = ListStr(arg_type="MultiOption", label="附加字段", order=2, option_range=())
    Deduplication = ListStr(arg_type="MultiOption", label="去重字段", order=3, option_range=())
    Period = Int(180, arg_type="Integer", label="周期", order=4)
    DataType = Enum("double", "string", arg_type="SingleOption", label="数据类型", order=5)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = FactorInfo[FactorInfo["FieldType"]=="ID"].index[0]
        self._DateField = FactorInfo[FactorInfo["FieldType"]=="Date"].index[0]
        self._InstituteField = FactorInfo[FactorInfo["FieldType"]=="Institute"].index[0]
        self._TempData = {}
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
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
        IDField = self._getIDField()
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("周期", self.Period))
        AdditiveFields = args.get("附加字段", self.AdditionalFields)
        DeduplicationFields = args.get("去重字段", self.Deduplication)
        AllFields = list(set(factor_names+AdditiveFields+DeduplicationFields))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self._DateField]+AllFields]
        DBTableName = self._FactorDB.TablePrefix + self._FactorDB._TableInfo.loc[self.Name, "DBTableName"]
        # 形成SQL语句, 日期, ID, 其他字段
        if self._FactorDB.DBType=="SQL Server":
            SQLStr = "SELECT TO_CHAR("+DBTableName+"."+FieldDict[self._DateField]+",'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr = "SELECT DATE_FORMAT("+DBTableName+"."+FieldDict[self._DateField]+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr = "SELECT TO_CHAR("+DBTableName+"."+FieldDict[self._DateField]+",'yyyyMMdd'), "
        SQLStr += IDField+" AS ID, "
        for iField in AllFields: SQLStr += DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+">='"+StartDate.strftime("%Y-%m-%d")+"' "
        SQLStr += "AND "+DBTableName+"."+FieldDict[self._DateField]+"<='"+EndDate.strftime("%Y-%m-%d")+"' "
        SQLStr += "ORDER BY "+self._FactorDB.TablePrefix+"SecuMain.SecuCode, "+DBTableName+"."+FieldDict[self._DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+AllFields)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID"]+AllFields)
        RawData = self._adjustRawDataByRelatedField(RawData, AllFields)
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
                    if (ijRawData.shape[0]>0) and (DeduplicationFields):
                        ijTemp = ijRawData.groupby(by=DeduplicationFields)[["日期"]].max()
                        ijTemp = ijTemp.reset_index()
                        ijTemp[DeduplicationFields] = ijTemp[DeduplicationFields].astype("O")
                        ijRawData = pd.merge(ijTemp, ijRawData, how='left', left_on=DeduplicationFields+["日期"], right_on=DeduplicationFields+["日期"])
                    kData[i, j] = Operator(self, iDate, jID, ijRawData, ModelArgs)
            Data[kFactorName] = kData
        return pd.Panel(Data, major_axis=Dates, minor_axis=ids).loc[factor_names, dts]

class _DividendTable(_DBTable):
    """分红因子表"""
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=0)
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=1)
    Operator = Either(Function(None), None, arg_type="Function", label="算子", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        self._MainTableName = fdb._TableInfo.loc[name, "MainTableName"]
        self._MainTableName = fdb.TablePrefix + self._MainTableName
        self._MainTableID = fdb._TableInfo.loc[name, "MainTableID"]
        self._JoinCondition = fdb._TableInfo.loc[name, "JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
        self._MainTableCondition = fdb._TableInfo.loc[name, "MainTableCondition"]
        if pd.notnull(self._MainTableCondition):
            self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
        self._SecurityType = fdb._TableInfo.loc[name, "SecurityType"]
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
    def getCondition(self, icondition, ids=None, dts=None):# TODO
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
    def getID(self, ifactor_name=None, idt=None, args={}):# TODO
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
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):# TODO
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
        IDField = self._getIDField()
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("回溯天数", self.LookBack))
        FieldDict = self._FactorDB._FactorInfo["DBFieldName"].loc[self.Name].loc[[self.DateField]+self._ConditionFields+factor_names]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+self._DBTableName+"."+FieldDict[self.DateField]+", "
        SQLStr += IDField+" AS ID, "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FieldDict[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "INNER JOIN "+self._MainTableName+" "
        SQLStr += "ON "+self._JoinCondition+" "
        SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=True, max_num=1000)+") "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self.DateField]+">='"+StartDate.strftime("%Y-%m-%d")+"' "
        SQLStr += "AND "+self._DBTableName+"."+FieldDict[self.DateField]+"<='"+EndDate.strftime("%Y-%m-%d")+"' "
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        for iConditionField in self._ConditionFields:
            if _identifyDataType(FactorInfo.loc[iConditionField, "DataType"])=="string":
                SQLStr += "AND "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iConditionField]+"='"+args.get(iConditionField, self[iConditionField])+"' "
            else:
                SQLStr += "AND "+self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iConditionField]+"="+args.get(iConditionField, self[iConditionField])+" "
        SQLStr += "ORDER BY ID, "+self._DBTableName+"."+FieldDict[self.DateField]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID"]+factor_names)
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Operator = args.get("算子", self.Operator)
        if Operator is None: Operator = (lambda x: x.tolist())
        Data = {}
        for iFactorName in raw_data.columns:
            Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
        Data = pd.Panel(Data).loc[factor_names, :, ids]
        LookBack = args.get("回溯天数", self.LookBack)
        if LookBack==0: return Data.loc[:, dts, :]
        AllDTs = Data.major_axis.union(dts).sort_values()
        Data = Data.loc[:, AllDTs, ids]
        Limits = LookBack*24.0*3600
        for i, iFactorName in enumerate(Data.items):
            Data.iloc[i] = fillNaByLookback(Data.iloc[i], lookback=Limits)
        return Data.loc[:, dts]


# 宏观因子表
class _MacroTable(_DBTable):
    """宏观因子表"""
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=0)
    IgnoreTime = Bool(True, label="忽略时间", arg_type="Bool", order=1)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DBTableName = fdb.TablePrefix + fdb._TableInfo.loc[name, "DBTableName"]
        FactorInfo = fdb._FactorInfo.loc[name]
        self._IDField = "CAST("+self._DBTableName+"."+FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="ID"].iloc[0]+" AS CHAR)"# ID 字段
        self._DateField = self._DBTableName+"."+FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="Date"].iloc[0]# 发布日期字段
        self._EndDateField = self._DBTableName+"."+FactorInfo["DBFieldName"][FactorInfo["FieldType"]=="EndDate"].iloc[0]# 截止日期字段
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    @property
    def FactorNames(self):
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        FactorNames = FactorInfo[FactorInfo["FieldType"]=="因子"].index.tolist()
        EndDate = FactorInfo[FactorInfo["FieldType"]=="EndDate"].index[0]
        return [EndDate]+FactorNames
    def getID(self, ifactor_name=None, idt=None, args={}):
        SQLStr = "SELECT DISTINCT "+self._IDField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        if idt is not None:
            if args.get("忽略时间", self.IgnoreTime):
                SQLStr += "WHERE DATE_FORMAT("+self._DateField+", '%Y-%m-%d')='"+idt.strftime("%Y-%m-%d")+"' "
            else:
                SQLStr += "WHERE "+self._DateField+"='"+idt.strftime("%Y-%m-%d %H:%M:%S")+"' "
        else:
            SQLStr += "WHERE "+self._DateField+" IS NOT NULL "
        SQLStr += "ORDER BY "+self._IDField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        if IgnoreTime:
            DateField = "DATE("+self._DateField+")"
        else:
            DateField = self._DateField
        SQLStr = "SELECT DISTINCT "+DateField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        if iid is not None:
            SQLStr += "WHERE "+self._IDField+"='"+iid+"' "
        else:
            SQLStr += "WHERE "+self._IDField+" IS NOT NULL "
        if IgnoreTime:
            if start_dt is not None: SQLStr += "AND "+DateField+">='"+start_dt.strftime("%Y-%m-%d")+"' "
            if end_dt is not None: SQLStr += "AND "+DateField+"<='"+end_dt.strftime("%Y-%m-%d")+"' "
        else:
            if start_dt is not None: SQLStr += "AND "+DateField+">='"+start_dt.strftime("%Y-%m-%d %H:%M:%S")+"' "
            if end_dt is not None: SQLStr += "AND "+DateField+"<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S")+"' "
        SQLStr += "ORDER BY "+self._DateField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        DateConditionGroup = {}
        for iFactor in factors:
            iDateConditions = (self._DateField, ";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in iFactor.ArgNames if iArgName!="回溯天数"]))
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
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        # 形成SQL语句, 发布日期, 截止日期, ID, 因子数据
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        if IgnoreTime:
            SQLStr = "SELECT DATE("+self._DateField+"), "
        else:
            SQLStr = "SELECT "+self._DateField+", "
        SQLStr += self._IDField+" AS ID, "
        SQLStr += self._EndDateField+", "
        for iField in factor_names: SQLStr += self._DBTableName+"."+FactorInfo["DBFieldName"].loc[iField]+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "WHERE ("+genSQLInCondition(self._IDField, ids, is_str=True, max_num=1000)+") "
        if IgnoreTime:
            SQLStr += "AND DATE("+self._DateField+")>='"+StartDate.strftime("%Y-%m-%d")+"' "
            SQLStr += "AND DATE("+self._DateField+")<='"+EndDate.strftime("%Y-%m-%d")+"' "
        else:
            SQLStr += "AND "+self._DateField+">='"+StartDate.strftime("%Y-%m-%d %H:%M:%S")+"' "
            SQLStr += "AND "+self._DateField+"<='"+EndDate.strftime("%Y-%m-%d %H:%M:%S")+"' "
        SQLStr += "ORDER BY ID, "+self._DateField+", "+self._EndDateField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["AnnDate", "ID", "EndDate"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["AnnDate", "ID", "EndDate"]+factor_names)
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)        
        return RawData
    # 检索最大截止日期的位置
    def _findMaxReportDateInd(self, raw_data, MaxEndDateInd, MaxNoteDateInd, PreMaxNoteDateInd):
        if MaxNoteDateInd==PreMaxNoteDateInd:
            return (MaxEndDateInd, False)
        Changed = False
        for i in range(0, MaxNoteDateInd-PreMaxNoteDateInd):
            if (MaxEndDateInd<0) or (raw_data['EndDate'].iloc[MaxNoteDateInd-i]>=raw_data['EndDate'].iloc[MaxEndDateInd]):
                MaxEndDateInd = MaxNoteDateInd-i
                Changed = True
        return (MaxEndDateInd, Changed)
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        DTs = sorted(dts)
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        raw_data = raw_data.set_index(["ID"])
        Data = {}
        for jID in raw_data.index.unique():
            jData = np.full(shape=(len(DTs), len(factor_names)), fill_value=np.nan, dtype="O")
            jRawData = raw_data.loc[[jID]]
            tempInd = -1# 指向目前看到的最大的公告期
            tempLen = jRawData.shape[0]
            MaxEndDateInd = -1# 指向目前看到的最大的截止日期
            for i, iDT in enumerate(DTs):
                tempPreInd = tempInd# 指向先前的最大公告期
                if IgnoreTime:
                    while (tempInd<tempLen-1) and (iDT.date()>=jRawData["AnnDate"].iloc[tempInd+1]): tempInd = tempInd+1
                else:
                    while (tempInd<tempLen-1) and (iDT>=jRawData["AnnDate"].iloc[tempInd+1]): tempInd = tempInd+1
                MaxEndDateInd, Changed = self._findMaxReportDateInd(jRawData, MaxEndDateInd, tempInd, tempPreInd)
                if not Changed:# 最大截止日期没有变化
                    if MaxEndDateInd>=0: jData[i] = jData[i-1]
                    continue
                MaxEndDate = jRawData['EndDate'].iloc[MaxEndDateInd]# 当前最大截止日期
                jSubRawData = jRawData.iloc[:tempInd+1]
                jData[i] = jSubRawData[jSubRawData["EndDate"]==MaxEndDate][factor_names].values[-1]
            Data[jID] = jData
        Data = pd.Panel(Data, major_axis=DTs, minor_axis=factor_names).swapaxes(0, 2)
        FactorInfo = self._FactorDB._FactorInfo.loc[self.Name]
        NewData = {}
        for i, iFactorName in enumerate(factor_names):
            iDataType = _identifyDataType(FactorInfo.loc[iFactorName, "DataType"])
            if iDataType=="double": NewData[iFactorName] = Data.iloc[i].astype("float")
            #elif iDataType=="datetime": NewData[iFactorName] = Data.iloc[i].applymap(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else None)
            else: NewData[iFactorName] = Data.iloc[i]
        Data = adjustDateTime(pd.Panel(NewData).loc[factor_names], dts, fillna=True, method="pad")
        Data = Data.loc[:, :, ids]
        return Data

class JYDB(FactorDB):
    """聚源数据库"""
    DBType = Enum("MySQL", "SQL Server", "Oracle", arg_type="SingleOption", label="数据库类型", order=0)
    DBName = Str("jydb1", arg_type="String", label="数据库名", order=1)
    IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    Port = Range(low=0, high=65535, value=3306, arg_type="Integer", label="端口", order=3)
    User = Str("root", arg_type="String", label="用户名", order=4)
    Pwd = Password("", arg_type="String", label="密码", order=5)
    TablePrefix = Str("", arg_type="String", label="表名前缀", order=6)
    CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=7)
    Connector = Enum("default", "cx_Oracle", "pymssql", "mysql.connector", "pymysql", "pyodbc", arg_type="SingleOption", label="连接器", order=8)
    DSN = Str("", arg_type="String", label="数据源", order=9)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"JYDBConfig.json" if config_file is None else config_file), **kwargs)
        self._Connection = None# 数据库链接
        self._AllTables = []# 数据库中的所有表名, 用于查询时解决大小写敏感问题
        self._InfoFilePath = __QS_LibPath__+os.sep+"JYDBInfo.hdf5"# 数据库信息文件路径
        self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"JYDBInfo.xlsx"# 数据库信息源文件路径
        self._TableInfo, self._FactorInfo, self._ExchangeInfo = _updateInfo(self._InfoFilePath, self._InfoResourcePath)# 数据库表信息, 数据库字段信息
        self._PID = None# 保存数据库连接创建时的进程号
        self.Name = "JYDB"
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
        elif self.Connector=='pymysql':
            try:
                import pymysql
                self._Connection = pymysql.connect(host=self.IPAddr, port=self.Port, user=self.User, password=self.Pwd, db=self.DBName, charset=self.CharSet)
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
        if self._TableInfo is not None: return self._TableInfo[pd.notnull(self._TableInfo["TableClass"])].index.tolist()
        else: return []
    def getTable(self, table_name, args={}):
        if table_name in self._TableInfo.index:
            TableClass = self._TableInfo.loc[table_name, "TableClass"]
            if pd.notnull(TableClass):
                return eval("_"+TableClass+"(name='"+table_name+"', fdb=self, sys_args=args)")
        raise __QS_Error__("因子库目前尚不支持表: '%s'" % table_name)
    # -----------------------------------------数据提取---------------------------------
    # 给定起始日期和结束日期, 获取交易所交易日期, 目前支持: "SSE", "SZSE", "SHFE", "DCE", "CZCE", "INE", "CFFEX"
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if start_date is None: start_date = dt.datetime(1900, 1, 1)
        if end_date is None: end_date = dt.datetime.today()
        ExchangeCode = self._ExchangeInfo[self._ExchangeInfo["Exchange"]==exchange].index
        if ExchangeCode.shape[0]==0: raise __QS_Error__("不支持的交易所: %s" % exchange)
        else: ExchangeCode = ExchangeCode[0]
        SQLStr = "SELECT {Prefix}QT_TradingDayNew.TradingDate FROM {Prefix}QT_TradingDayNew "
        SQLStr += "WHERE {Prefix}QT_TradingDayNew.TradingDate>='{StartDate}' AND {Prefix}QT_TradingDayNew.TradingDate<='{EndDate}' "
        SQLStr += "AND {Prefix}QT_TradingDayNew.IfTradingDay=1 "
        SQLStr += "AND {Prefix}QT_TradingDayNew.SecuMarket={ExchangeCode} "
        SQLStr += "ORDER BY {Prefix}QT_TradingDayNew.TradingDate"
        SQLStr = SQLStr.format(Prefix=self.TablePrefix, ExchangeCode=ExchangeCode,
                               StartDate=start_date.strftime("%Y-%m-%d"), EndDate=end_date.strftime("%Y-%m-%d"))
        Rslt = self.fetchall(SQLStr)
        if kwargs.get("output_type", "date")=="date": return [iRslt[0].date() for iRslt in Rslt]
        else: return [iRslt[0] for iRslt in Rslt]
    # 获取指定日 date 的全体 A 股 ID
    # date: 指定日, datetime.date
    # is_current: False 表示上市日在指定日之前的 A 股, True 表示上市日在指定日之前且尚未退市的 A 股
    def _getAllAStock(self, date, is_current=True):
        SQLStr = "SELECT CASE WHEN {Prefix}SecuMain.SecuMarket=83 THEN CONCAT({Prefix}SecuMain.SecuCode, '.SH') "
        SQLStr += "WHEN {Prefix}SecuMain.SecuMarket=90 THEN CONCAT({Prefix}SecuMain.SecuCode, '.SZ') "
        SQLStr += "ELSE {Prefix}SecuMain.SecuCode END FROM {Prefix}SecuMain "
        SQLStr += "WHERE {Prefix}SecuMain.SecuCategory = 1 "
        SQLStr += "AND {Prefix}SecuMain.SecuMarket IN (83, 90) "
        SQLStr += "AND {Prefix}SecuMain.ListedDate <= '{Date}' "
        if is_current:
            SubSQLStr = "SELECT DISTINCT {Prefix}LC_ListStatus.InnerCode FROM {Prefix}LC_ListStatus "
            SubSQLStr += "WHERE {Prefix}LC_ListStatus.ChangeType = 4 "
            SubSQLStr += "AND {Prefix}LC_ListStatus.ChangeDate <= '{Date}' "
            SQLStr += "AND {Prefix}SecuMain.InnerCode NOT IN ("+SubSQLStr+") "
        SQLStr += "ORDER BY {Prefix}SecuMain.SecuCode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y-%m-%d")))]
    # 获取指定日 date 指数 index_id 的成份股 ID
    # index_id: 指数 ID, 默认值 "全体A股"
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示进入指数的日期在指定日之前的成份股, True 表示进入指数的日期在指定日之前且尚未剔出指数的 A 股
    def getStockID(self, index_id="全体A股", date=None, is_current=True):# TODO
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
    def getFutureID(self, future_code="IF", exchange="CFFEX", date=None, is_current=True, **kwargs):# TODO
        if date is None: date = dt.date.today()
        SQLStr = "SELECT DISTINCT s_info_windcode FROM {Prefix}Fut_ContractMain "
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
    # 获取指定交易所 exchange 的期货代码
    # exchange: 交易所(str)或者交易所列表(list(str))
    # date: 指定日, 默认值 None 表示今天
    # is_listed: True 表示只返回当前上市的期货代码
    # kwargs:
    # include_simulation: 是否包括仿真合约, 默认值 False
    def getFutureCode(self, exchange=["SHFE", "INE", "DCE", "CZCE", "CFFEX"], is_listed=True, **kwargs):# TODO
        SQLStr = "SELECT DISTINCT TradingCode FROM {Prefix}Fut_FuturesContract "
        if exchange:
            if isinstance(exchange, str): exchange = [exchange]
            ExchgCodes = set()
            for iExchg in exchange:
                iExchgCode = self._ExchangeInfo[self._ExchangeInfo["Exchange"]==iExchg]
                if iExchgCode.shape[0]==0: raise __QS_Error__("不支持的交易所: %s" % iExchg)
                ExchgCodes.add(str(iExchgCode[0]))
            SQLStr += "WHERE Exchange IN ("+", ".join(ExchgCodes)+") "
        else:
            SQLStr += "WHERE Exchange IS NOT NULL "
        if is_listed: SQLStr += "AND ContractState=1 "
        SQLStr += "ORDER BY TradingCode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix))]
    # 给定期权代码 option_code, 获取指定日 date 的期权代码
    # option_code: 期权代码(str)
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日在指定日之前的期权, True 表示上市日在指定日之前且尚未退市的期权
    def getOptionID(self, option_code="510050", date=None, is_current=True, **kwargs):
        if date is None: date = dt.date.today()
        SQLStr = "SELECT DISTINCT TradingCode FROM {Prefix}Opt_OptionContract "
        SQLStr += "WHERE TradingCode LIKE '{OptionCode}%%' "
        SQLStr += "AND ListingDate<='{Date}' "
        SQLStr += "AND IfReal=1 "
        if is_current: SQLStr += "AND LastTradingDate>='{Date}' "
        SQLStr += "ORDER BY TradingCode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y-%m-%d"), OptionCode=option_code))]
    # 获取指定日 date 基金 ID
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示成立日在指定日之前的基金, True 表示成立日在指定日之前且尚未清盘的基金
    def getMutualFundID(self, date=None, is_current=True, **kwargs):
        if date is None: date = dt.date.today()
        SQLStr = "SELECT CASE {Prefix}SecuMain.SecuMarket WHEN 83 THEN CONCAT({Prefix}SecuMain.SecuCode, '.SH') "
        SQLStr += "WHEN 90 THEN CONCAT({Prefix}SecuMain.SecuCode, '.SZ') "
        SQLStr += "ELSE CONCAT({Prefix}SecuMain.SecuCode, '.MF') END AS ID FROM {Prefix}mf_fundarchives "
        SQLStr += "INNER JOIN {Prefix}SecuMain ON {Prefix}SecuMain.InnerCode={Prefix}mf_fundarchives.InnerCode "
        SQLStr += "WHERE {Prefix}mf_fundarchives.EstablishmentDate <= '{Date}' "
        if is_current:
            SQLStr += "AND ({Prefix}mf_fundarchives.EnClearingDate IS NULL "
            SQLStr += "OR {Prefix}mf_fundarchives.EnClearingDate > '{Date}') "
        SQLStr += "ORDER BY ID"
        Rslt = np.array(self.fetchall(SQLStr.format(Prefix=self.TablePrefix, Date=date.strftime("%Y-%m-%d"))))
        if Rslt.shape[0]>0: return Rslt[:, 0].tolist()
        else: return []
    # 获取宏观指标名称对应的指标 ID, TODO
    def getMacroIndicatorID(self, indicators, table_name=None):
        pass