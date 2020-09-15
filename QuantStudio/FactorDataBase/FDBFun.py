# coding=utf-8
import os

import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.FactorDataBase.FactorDB import FactorTable

# 将信息源文件中的表和字段信息导入信息文件
def importInfo(info_file, info_resource):
    TableInfo = pd.read_excel(info_resource, "TableInfo").set_index(["TableName"])
    FactorInfo = pd.read_excel(info_resource, 'FactorInfo').set_index(['TableName', 'FieldName'])
    try:
        from QuantStudio.Tools.DataTypeFun import writeNestedDict2HDF5
        writeNestedDict2HDF5(TableInfo, info_file, "/TableInfo")
        writeNestedDict2HDF5(FactorInfo, info_file, "/FactorInfo")
    except:
        pass
    return (TableInfo, FactorInfo)

# 更新信息文件
def updateInfo(info_file, info_resource, logger):
    if not os.path.isfile(info_file):
        logger.warning("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    elif (os.path.getmtime(info_resource)>os.path.getmtime(info_file)):
        logger.warning("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % info_resource)
    else:
        try:
            from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
            return (readNestedDictFromHDF5(info_file, ref="/TableInfo"), readNestedDictFromHDF5(info_file, ref="/FactorInfo"))
        except:
            logger.warning("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    if not os.path.isfile(info_resource): raise __QS_Error__("缺失数据库信息源文件: %s" % info_resource)
    return importInfo(info_file, info_resource)

def adjustDateTime(data, dts, fillna=False, **kwargs):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.shape[0]==0:
            if isinstance(data, pd.DataFrame): data = pd.DataFrame(index=dts, columns=data.columns)
            else: data = pd.Series(index=dts)
        else:
            if fillna:
                AllDTs = data.index.union(dts)
                AllDTs = AllDTs.sort_values()
                data = data.loc[AllDTs]
                data = data.fillna(**kwargs)
            data = data.loc[dts]
    else:
        if data.shape[1]==0:
            data = pd.Panel(items=data.items, major_axis=dts, minor_axis=data.minor_axis)
        else:
            FactorNames = data.items
            if fillna:
                AllDTs = data.major_axis.union(dts)
                AllDTs = AllDTs.sort_values()
                data = data.loc[:, AllDTs, :]
                data = pd.Panel({data.items[i]:data.iloc[i].fillna(axis=0, **kwargs) for i in range(data.shape[0])})
            data = data.loc[FactorNames, dts, :]
    return data

def adjustDataDTID(data, look_back, factor_names, ids, dts, only_start_lookback=False, only_lookback_nontarget=False, only_lookback_dt=False, logger=None):
    if look_back==0:
        try:
            return data.loc[:, dts, ids]
        except KeyError as e:
            if logger is not None:
                logger.warning("待提取的因子 %s 数据超出了原始数据的时点或 ID 范围, 将填充缺失值!" % (str(list(data.items)), ))
            return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
    AllDTs = data.major_axis.union(dts).sort_values()
    AdjData = data.loc[:, AllDTs, ids]
    if only_start_lookback:# 只在起始时点回溯填充缺失
        AllAdjData = AdjData
        AdjData = AllAdjData.loc[:, :dts[0], :]
        TargetDTs = dts[:1]
    else:
        TargetDTs = dts
    if only_lookback_dt:
        TargetDTs = sorted(set(TargetDTs).difference(data.major_axis))
    if TargetDTs:
        Limits = look_back*24.0*3600
        if only_lookback_nontarget:# 只用非目标时间序列的数据回溯填充
            Mask = pd.Series(np.full(shape=(AdjData.shape[1], ), fill_value=False, dtype=np.bool), index=AdjData.major_axis)
            Mask[TargetDTs] = True
            FillMask = Mask.copy()
            FillMask[Mask.astype("int").diff()!=1] = False
            TimeDelta = pd.Series(np.r_[0, np.diff(Mask.index.values) / np.timedelta64(1, "D")], index=Mask.index)
            TimeDelta[(Mask & (~FillMask)) | (Mask.astype("int").diff()==-1)] = 0
            TimeDelta = TimeDelta.cumsum().loc[TargetDTs]
            FirstDelta = TimeDelta.iloc[0]
            TimeDelta = TimeDelta.diff().fillna(value=0)
            TimeDelta.iloc[0] = FirstDelta
            NewLimits = np.minimum(TimeDelta.values*24.0*3600, Limits).reshape((TimeDelta.shape[0], 1)).repeat(AdjData.shape[2], axis=1)
            Limits = pd.DataFrame(0, index=AdjData.major_axis, columns=AdjData.minor_axis)
            Limits.loc[TargetDTs, :] = NewLimits
        if only_lookback_dt:
            Mask = pd.Series(np.full(shape=(AdjData.shape[1], ), fill_value=False, dtype=np.bool), index=AdjData.major_axis)
            Mask[TargetDTs] = True
            FillMask = Mask.copy()
            FillMask[Mask.astype("int").diff()!=1] = False
            FillMask = FillMask.loc[TargetDTs]
            TimeDelta = pd.Series(np.r_[0, np.diff(Mask.index.values) / np.timedelta64(1, "D")], index=Mask.index).loc[TargetDTs]
            NewLimits = TimeDelta.cumsum().loc[TargetDTs]
            Temp = NewLimits.copy()
            Temp[~FillMask] = np.nan
            Temp = Temp.fillna(method="pad")
            TimeDelta[~FillMask] = np.nan
            NewLimits = NewLimits - Temp + TimeDelta.fillna(method="pad")
            if isinstance(Limits, pd.DataFrame):
                Limits.loc[TargetDTs, :] = np.minimum(NewLimits.values.reshape((NewLimits.shape[0], 1)).repeat(AdjData.shape[2], axis=1), Limits.loc[TargetDTs].values)
            else:
                NewLimits = np.minimum(NewLimits.values*24.0*3600, Limits).reshape((NewLimits.shape[0], 1)).repeat(AdjData.shape[2], axis=1)
                Limits = pd.DataFrame(0, index=AdjData.major_axis, columns=AdjData.minor_axis)
                Limits.loc[TargetDTs, :] = NewLimits
        if np.isinf(look_back) and (not only_lookback_nontarget) and (not only_lookback_dt):
            for i, iFactorName in enumerate(AdjData.items): AdjData.iloc[i].fillna(method="pad", inplace=True)
        else:
            AdjData = dict(AdjData)
            for iFactorName in AdjData: AdjData[iFactorName] = fillNaByLookback(AdjData[iFactorName], lookback=Limits)
            AdjData = pd.Panel(AdjData).loc[factor_names]
    if only_start_lookback:
        AllAdjData.loc[:, dts[0], :] = AdjData.loc[:, dts[0], :]
        return AllAdjData.loc[:, dts]
    else:
        return AdjData.loc[:, dts]


# 基于 SQL 数据库表的因子表
# table_info: Series(index=["DBTableName", "SecurityType"]), 可选的 index=["MainTableName", "MainTableID", "JoinCondition", "MainTableCondition", "DefaultSuffix", "Exchange", "SecurityCategory"]
# factor_info: DataFrame(index=[], columns=["DBFieldName", "FieldType", "DataType", "Supplementary", "RelatedSQL"]), 可选的 columns=[]
# security_info: DataFrame(index=[], columns=["Suffix"])
# exchange_info: DataFrame(index=[], columns=["Suffix"])
# 参数编号分配:
# 0 - 100: 因子表特定参数
# 100 - 199: 条件参数, 100: 通用筛选条件
# 200 - 299: 通用参数
class SQL_Table(FactorTable):
    FilterCondition = Str("", arg_type="Dict", label="筛选条件", order=100)
    TableType = Str("", arg_type="SingleOption", label="因子表类型", order=200)
    PreFilterID = Bool(True, arg_type="Bool", label="预筛选ID", order=201)
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        self._TablePrefix = table_prefix
        self._TableInfo = table_info
        self._FactorInfo = factor_info
        self._SecurityInfo = security_info
        self._ExchangeInfo = exchange_info
        self._QS_IgnoredGroupArgs = ("遍历模式", )
        self._DTFormat = "'%Y-%m-%d'"
        self._DBTableName = self._TablePrefix + self._TableInfo.loc["DBTableName"]
        self._SecurityType = self._TableInfo.loc["SecurityType"]
        # 解析 ID 字段, 至多一个 ID 字段
        Idx = np.where(self._FactorInfo["FieldType"]=="ID")[0]
        if Idx.shape[0]==0:
            self._IDField = None
            self._IDFieldIsStr = True
        else:
            self._IDField = self._FactorInfo["DBFieldName"].iloc[Idx[0]]
            self._IDFieldIsStr = (self.__QS_identifyDataType(self._FactorInfo["DataType"].iloc[Idx[0]])!="double")
        # 解析条件字段
        self._ConditionFields = self._FactorInfo[self._FactorInfo["FieldType"]=="Condition"].index.tolist()
        # 解析主表
        self._MainTableName = self._TableInfo.get("MainTableName", None)
        if pd.isnull(self._MainTableName):
            self._MainTableName = self._DBTableName
            self._MainTableID = self._IDField
            self._MainTableCondition = None
        else:
            self._MainTableName = self._TablePrefix + self._MainTableName
            self._MainTableID = self._TableInfo.loc["MainTableID"]
            self._JoinCondition = self._TableInfo.loc["JoinCondition"].format(DBTable=self._DBTableName, MainTable=self._MainTableName)
            self._MainTableCondition = self._TableInfo.loc["MainTableCondition"]
            if pd.notnull(self._MainTableCondition):
                self._MainTableCondition = self._MainTableCondition.format(MainTable=self._MainTableName)
            self._IDFieldIsStr = True# TODO
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        for i, iCondition in enumerate(self._ConditionFields):
            self.add_trait("Condition"+str(i), Str("", arg_type="String", label=iCondition, order=i+101))
            iConditionVal = self._FactorInfo.loc[iCondition, "Supplementary"]
            if pd.isnull(iConditionVal) or (isinstance(iConditionVal, str) and (iConditionVal.lower() in ("", "nan"))):
                self[iCondition] = ""
            else:
                self[iCondition] = str(iConditionVal).strip()
    def __QS_identifyDataType(self, data_type):
        field_data_type = field_data_type.lower()
        if (field_data_type.find("num")!=-1) or (field_data_type.find("int")!=-1) or (field_data_type.find("decimal")!=-1) or (field_data_type.find("double")!=-1) or (field_data_type.find("float")!=-1):
            return "double"
        elif field_data_type.find("date")!=-1:
            return "object"
        else:
            return "string"
    def __QS_adjustID(self, ids):
        return ids
    def __QS_restoreID(self, ids):
        return ids
    def __QS_adjustDT(self, dts):
        return [iDT.strftime(self._DTFormat) for iDT in dts]
    def _genFromSQLStr(self, setable_join_str=[]):
        SQLStr = "FROM "+self._DBTableName+" "
        for iJoinStr in setable_join_str: SQLStr += iJoinStr+" "
        if self._DBTableName!=self._MainTableName:
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
        return SQLStr[:-1]
    def _getIDField(self, args={}):
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            if not self._IDFieldIsStr:
                RawIDField = "CAST("+self._DBTableName+"."+self._IDField+" AS CHAR)"
            else:
                RawIDField = self._DBTableName+"."+self._IDField
        else:
            RawIDField = self._MainTableName+"."+self._MainTableID
        DefaultSuffix = self._TableInfo.get("DefaultSuffix", None)
        Exchange = self._TableInfo.get("Exchange", None)
        SecurityCategory = self._TableInfo.get("SecurityCategory", None)
        Suffix = "{ElseSuffix}"
        if pd.notnull(SecurityCategory):
            SecuCategoryField, SecuCategoryCodes = SecurityCategory.split(":")
            if self._MainTableName is None:
                SecuCategoryField = self._DBTableName + "." + SecuCategoryField
            else:
                SecuCategoryField = self._MainTableName + "." + SecuCategoryField
            SecuCategoryCodes = SecuCategoryCodes.split(",")
            SecurityInfo = self._SecurityInfo
            iSuffix = "CASE "+SecuCategoryField+" "
            for iCode in SecuCategoryCodes:
                iSuffix += "WHEN "+iCode+" THEN '"+SecurityInfo.loc[iCode, "Suffix"]+"' "
            iSuffix += "ELSE {ElseSuffix} END"
            Suffix = Suffix.format(ElseSuffix=iSuffix)
        if pd.notnull(Exchange):
            ExchangeField, ExchangeCodes = Exchange.split(":")
            if self._MainTableName is None:
                ExchangeField = self._DBTableName + "." + ExchangeField
            else:
                ExchangeField = self._MainTableName + "." + ExchangeField
            ExchangeCodes = ExchangeCodes.split(",")
            ExchangeInfo = self._ExchangeInfo
            iSuffix = "CASE "+ExchangeField+" "
            for iCode in ExchangeCodes:
                iSuffix += "WHEN "+iCode+" THEN '"+ExchangeInfo.loc[iCode, "Suffix"]+"' "
            iSuffix += "ELSE {ElseSuffix} END"
            Suffix = Suffix.format(ElseSuffix=iSuffix)
        Suffix = Suffix.format(ElseSuffix=("''" if pd.isnull(DefaultSuffix) else "'"+DefaultSuffix+"'"))
        return "CONCAT("+RawIDField+", "+Suffix+")"
    def _adjustRawDataByRelatedField(self, raw_data, fields):# TODO JYDB
        RelatedFields = self._FactorInfo["RelatedSQL"].loc[fields]
        RelatedFields = RelatedFields[pd.notnull(RelatedFields)]
        if RelatedFields.shape[0]==0: return raw_data
        for iField in RelatedFields.index:
            iOldData = raw_data.pop(iField)
            iOldDataType = self.__QS_identifyDataType(self._FactorInfo.loc[iField[:-2], "DataType"])
            iDataType = self.__QS_identifyDataType(self._FactorInfo.loc[iField, "DataType"])
            if iDataType=="double":
                iNewData = pd.Series(np.nan, index=raw_data.index, dtype="float")
            else:
                iNewData = pd.Series(np.full(shape=(raw_data.shape[0], ), fill_value=None, dtype="O"), index=raw_data.index, dtype="O")
            iSQLStr = self._FactorInfo.loc[iField, "RelatedSQL"]
            if iSQLStr[0]=="{":
                iMapInfo = eval(iSQLStr).items()
            else:
                iStartIdx = iSQLStr.find("{KeyCondition}")
                if iStartIdx!=-1:
                    iEndIdx = iSQLStr[iStartIdx:].find(" ")
                    if iEndIdx==-1: iEndIdx = len(iSQLStr)
                    else: iEndIdx += iStartIdx
                    iStartIdx += 14
                    KeyField = iSQLStr[iStartIdx:iEndIdx]
                    iKeys = iOldData[pd.notnull(iOldData)].unique().tolist()
                    if iKeys:
                        KeyCondition = genSQLInCondition(KeyField, iKeys, is_str=(iOldDataType!="double"))
                    else:
                        KeyCondition = KeyField+" IN (NULL)"
                    iSQLStr = iSQLStr.replace("{KeyCondition}"+KeyField, "{KeyCondition}")
                else:
                    KeyCondition = ""
                if iSQLStr.find("{Keys}")!=-1:
                    Keys = ", ".join([str(iKey) for iKey in iOldData[pd.notnull(iOldData)].unique()])
                    if not Keys: Keys = "NULL"
                else:
                    Keys = ""
                iMapInfo = self._FactorDB.fetchall(iSQLStr.format(TablePrefix=self._TablePrefix, Keys=Keys, KeyCondition=KeyCondition))
            for jVal, jRelatedVal in iMapInfo:
                if pd.notnull(jVal):
                    if iOldDataType!="double":
                        iNewData[iOldData==str(jVal)] = jRelatedVal
                    elif isinstance(jVal, str):
                        iNewData[iOldData==float(jVal)] = jRelatedVal
                    else:
                        iNewData[iOldData==jVal] = jRelatedVal
                else:
                    iNewData[pd.isnull(iOldData)] = jRelatedVal
            raw_data[iField] = iNewData
        return raw_data
    def _genFieldSQLStr(self, factor_names):# TODO JYDB
        SQLStr = ""
        JoinStr = []
        SETables = set()
        for iField in factor_names:
            iInfo = self._FactorInfo.loc[iField, "Supplementary"]
            if isinstance(iInfo, str) and (iInfo.find("从表")!=-1):
                iInfo = iInfo.split(":")
                iSETable, iJoinField = iInfo[-2:]
                SQLStr += iSETable+"."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
                if iSETable not in SETables:
                    JoinStr.append("LEFT JOIN "+iSETable+" ON "+self._DBTableName+".ID="+iSETable+"."+iJoinField)
                    SETables.add(iSETable)
            else:
                SQLStr += self._DBTableName+"."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
        return (SQLStr[:-2], JoinStr)
    def _genConditionSQLStr(self, args={}):
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr = "AND "+FilterStr.format(Table=self._DBTableName, TablePrefix=self._TablePrefix)+" "
        else: SQLStr = ""
        for iConditionField in self._ConditionFields:
            iConditionVal = args.get(iConditionField, self[iConditionField])
            if iConditionVal:
                if self.__QS_identifyDataType(self._FactorInfo.loc[iConditionField, "DataType"])!="double":
                    SQLStr += "AND "+self._DBTableName+"."+self._FactorInfo.loc[iConditionField, "DBFieldName"]+" IN ('"+"','".join(iConditionVal.split(","))+"') "
                else:
                    SQLStr += "AND "+self._DBTableName+"."+self._FactorInfo.loc[iConditionField, "DBFieldName"]+" IN ("+iConditionVal+") "
        return SQLStr[:-1]
    def getCondition(self, icondition, ids=None, dts=None, args={}):
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+self._FactorInfo.loc[icondition, "DBFieldName"]+" "
        SQLStr += "FROM "+self._DBTableName+" "
        if self._MainTableName!=self._DBTableName:
            SQLStr += "INNER JOIN "+self._MainTableName+" "
            SQLStr += "ON "+self._JoinCondition+" "
        if ids is not None: SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, self.__QS_adjustID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
        else: SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        if (dts is not None) and hasattr(self, "DateField"):
            DateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
            SQLStr += "AND ("+genSQLInCondition(DateField, self.__QS_adjustDT(dts), is_str=False, max_num=1000)+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+self._FactorInfo.loc[icondition, "DBFieldName"]
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getMetaData(self, key=None, args={}):
        if key is None:
            return self._TableInfo
        else:
            return self._TableInfo.get(key, None)
    @property
    def FactorNames(self):
        return self._FactorInfo[pd.notnull(self._FactorInfo["FieldType"])].index.tolist()
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key=="DataType":
            if hasattr(self, "_DataType"): return self._DataType.loc[factor_names]
            return self._FactorInfo["DataType"].loc[factor_names].apply(self.__QS_identifyDataType)
        elif key=="Description": return self._FactorInfo["Description"].loc[factor_names]
        elif key is None:
            return pd.DataFrame({"DataType":self.getFactorMetaData(factor_names, key="DataType", args=args),
                                 "Description":self.getFactorMetaData(factor_names, key="Description", args=args)})
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            #iConditions = ";".join([iCondition+":"+str(iFactor[iCondition]) for i, iCondition in enumerate(self._ConditionFields)]+["筛选条件:"+iFactor["筛选条件"]])
            iConditions = ";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in iFactor.ArgNames if iArgName not in self._QS_IgnoredGroupArgs])
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

# 行情因子表, 表结构特征:
# 日期字段, 表示数据填充的时点;
# 条件字段, 作为条件过滤记录; 可能存在多个条件字段
# 在设定某些条件下, 数据填充时点和 ID 可以唯一标志一行记录
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class SQL_MarketTable(SQL_Table):
    """行情因子表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    OnlyStartLookBack = Bool(False, label="只起始日回溯", arg_type="Bool", order=1)
    OnlyLookBackNontarget = Bool(False, label="只回溯非目标日", arg_type="Bool", order=2)
    OnlyLookBackDT = Bool(False, label="只回溯时点", arg_type="Bool", order=3)
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=4)
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "回溯天数", "只起始日回溯", "只回溯非目标日", "只回溯时点")
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        DateFields = self._FactorInfo[self._FactorInfo["FieldType"].str.contains("Date")].index.tolist()# 所有的日期字段列表
        self.add_trait("DateField", Enum(*DateFields, arg_type="SingleOption", label="日期字段", order=4))
        iFactorInfo = self._FactorInfo[self._FactorInfo["FieldType"].str.contains("Date") & pd.notnull(self._FactorInfo["Supplementary"])]
        iFactorInfo = iFactorInfo[iFactorInfo["Supplementary"].str.contains("Default")]
        if iFactorInfo.shape[0]>0: self.DateField = iFactorInfo.index[0]
        else: self.DateField = DateFields[0]
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    # 返回在给定时点 idt 的有数据记录的 ID
    # 如果 idt 为 None, 将返回所有有历史数据记录的 ID
    # 忽略 ifactor_name
    def getID(self, ifactor_name=None, idt=None, args={}):
        DateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr()+" "
        if idt is not None: SQLStr += "WHERE "+DateField+"="+idt.strftime(self._DTFormat)+" "
        else: SQLStr += "WHERE "+DateField+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID"
        return self.__QS_restoreID([iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)])
    # 返回在给定 ID iid 的有数据记录的时间点
    # 如果 iid 为 None, 将返回所有有历史数据记录的时间点
    # 忽略 ifactor_name
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        SQLStr = "SELECT DISTINCT "+DateField+" "
        if iid is not None:
            SQLStr += self._genFromSQLStr()+" "
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+self.__QS_adjustID([iid])[0]+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else:
            SQLStr += "FROM "+self._DBTableName+" "
            SQLStr += "WHERE "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        if start_dt is not None: SQLStr += "AND "+DateField+">="+start_dt.strftime(self._DTFormat)+" "
        if end_dt is not None: SQLStr += "AND "+DateField+"<="+end_dt.strftime(self._DTFormat)+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY "+DateField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in iFactor.ArgNames if iArgName not in self._QS_IgnoredGroupArgs])
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
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        IDField = self._getIDField(args=args)
        DateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        SubSQLStr = "SELECT "+self._MainTableName+"."+self._MainTableID+", "
        SubSQLStr += "MAX("+DateField+") "
        SubSQLStr += self._genFromSQLStr()+" "
        SubSQLStr += "WHERE "+DateField+"<"+end_date.strftime(self._DTFormat)+" "
        if args.get("预筛选ID", self.PreFilterID):
            SubSQLStr += "AND ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, self.__QS_adjustID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
        else:
            SubSQLStr += "AND "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SubSQLStr += "AND "+self._MainTableCondition+" "
        ConditionSQLStr = self._genConditionSQLStr(args=args)
        SubSQLStr += ConditionSQLStr+" "
        SubSQLStr += "GROUP BY "+self._MainTableName+"."+self._MainTableID
        SQLStr = "SELECT "+DateField+", "
        SQLStr += IDField+" AS ID, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "WHERE ("+self._MainTableName+"."+self._MainTableID+", "+DateField+") IN ("+SubSQLStr+") "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += ConditionSQLStr
        return SQLStr
    def _genSQLStr(self, factor_names, ids, start_date, end_date, args={}):
        IDField = self._getIDField(args=args)
        DateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        # 形成SQL语句, 日期, ID, 因子数据
        SQLStr = "SELECT "+DateField+", "
        SQLStr += IDField+" AS ID, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "WHERE "+DateField+">="+start_date.strftime(self._DTFormat)+" "
        SQLStr += "AND "+DateField+"<="+end_date.strftime(self._DTFormat)+" "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        if args.get("预筛选ID", self.PreFilterID):
            SQLStr += "AND ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, self.__QS_adjustID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
        else:
            SQLStr += "AND "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID, "+DateField
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
        if args.get("只回溯时点", self.OnlyLookBackDT):
            RowIdxMask = pd.Series(False, index=raw_data.index).unstack(fill_value=True).astype(bool)
            RawIDs = RowIdxMask.columns
            if RawIDs.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
            RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
            RowIdx[RowIdxMask] = np.nan
            RowIdx = adjustDataDTID(pd.Panel({"RowIdx": RowIdx}), args.get("回溯天数", self.LookBack), ["RowIdx"], RowIdx.columns.tolist(), dts, 
                                              args.get("只起始日回溯", self.OnlyStartLookBack), 
                                              args.get("只回溯非目标日", self.OnlyLookBackNontarget), 
                                              logger=self._QS_Logger).iloc[0].values
            RowIdx[pd.isnull(RowIdx)] = -1
            RowIdx = RowIdx.astype(int)
            ColIdx = np.arange(RowIdx.shape[1]).reshape((1, RowIdx.shape[1])).repeat(RowIdx.shape[0], axis=0)
            RowIdxMask = (RowIdx==-1)
            Data = {}
            for iFactorName in raw_data.columns:
                iRawData = raw_data[iFactorName].unstack()
                iDataType = self.__QS_identifyDataType(self._FactorInfo.loc[iFactorName, "DataType"])
                if iDataType=="double": iRawData = iRawData.astype("float")
                iRawData = iRawData.values[RowIdx, ColIdx]
                iRawData[RowIdxMask] = None
                Data[iFactorName] = pd.DataFrame(iRawData, index=dts, columns=RawIDs)
            return pd.Panel(Data).loc[factor_names, :, ids]
        else:
            Data = {}
            for iFactorName in raw_data.columns:
                iRawData = raw_data[iFactorName].unstack()
                iDataType = self.__QS_identifyDataType(self._FactorInfo.loc[iFactorName, "DataType"])
                if iDataType=="double": iRawData = iRawData.astype("float")
                Data[iFactorName] = iRawData
            Data = pd.Panel(Data).loc[factor_names]
            return adjustDataDTID(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts, 
                                  args.get("只起始日回溯", self.OnlyStartLookBack), 
                                  args.get("只回溯非目标日", self.OnlyLookBackNontarget), 
                                  logger=self._QS_Logger)


# 信息发布表, 表结构特征:
# 公告日期, 表示信息发布的时点;
# 截止日期, 表示信息有效的时点;
# 如果不忽略公告日期, 则以截止日期和公告日期的最大值作为数据填充的时点, 同一填充时点存在多个截止日期时以最大截止日期的记录值填充
# 如果忽略公告日期, 则以截止日期作为数据填充的时点, 必须保证截至日期具有唯一性
# 条件字段, 作为条件过滤记录; 可能存在多个条件字段
# 在设定某些条件下, 数据填充时点和 ID 可以唯一标志一行记录
# 先填充表中已有的数据, 然后根据回溯天数参数填充缺失的时点
class SQL_InfoPublTable(SQL_MarketTable):
    """信息发布表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    OnlyStartLookBack = Bool(False, label="只起始日回溯", arg_type="Bool", order=1)
    OnlyLookBackNontarget = Bool(False, label="只回溯非目标日", arg_type="Bool", order=2)
    OnlyLookBackDT = Bool(False, label="只回溯时点", arg_type="Bool", order=3)
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=4)
    IgnorePublDate = Bool(False, label="忽略公告日", arg_type="Bool", order=5)
    IgnoreTime = Bool(True, label="忽略时间", arg_type="Bool", order=6)
    EndDateASC = Bool(False, label="截止日期递增", arg_type="Bool", order=7)
    def __init__(self, name, fdb, sys_args={}, table_prefix="", table_info=None, factor_info=None, security_info=None, exchange_info=None, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=table_prefix, table_info=table_info, factor_info=factor_info, security_info=security_info, exchange_info=exchange_info, **kwargs)
        self._AnnDateField = self._FactorInfo["DBFieldName"][self._FactorInfo["FieldType"]=="AnnDate"]
        if self._AnnDateField.shape[0]>0: self._AnnDateField = self._AnnDateField.iloc[0]# 公告日期
        else: self._AnnDateField = None
    def getID(self, ifactor_name=None, idt=None, args={}):
        EndDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        if self._AnnDateField is None: AnnDateField = EndDateField
        else: AnnDateField = self._DBTableName+"."+self._AnnDateField
        SQLStr = "SELECT DISTINCT "+self._getIDField(args=args)+" AS ID "
        SQLStr += self._genFromSQLStr()+" "
        if args.get("忽略时间", self.IgnoreTime): DTFormat = self._DTFormat
        else: DTFormat = "'%Y-%m-%d %H:%M:%S'"
        if AnnDateField!=EndDateField:
            if idt is not None:
                SQLStr += "WHERE (CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END)="+idt.strftime(DTFormat)+" "
            else:
                SQLStr += "WHERE "+AnnDateField+" IS NOT NULL AND "+EndDateField+" IS NOT NULL "
        else:
            if idt is not None:
                SQLStr += "WHERE "+AnnDateField+"="+idt.strftime(DTFormat)+" "
            else:
                SQLStr += "WHERE "+AnnDateField+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY ID"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if args.get("忽略公告日", self.IgnorePublDate) or (self._AnnDateField is None): return super().getDateTime(ifactor_name=ifactor_name, iid=iid, start_dt=start_dt, end_dt=end_dt, args=args)
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        EndDateField = self._DBTableName+"."+self._FactorDB._FactorInfo.loc[self.Name].loc[args.get("日期字段", self.DateField), "DBFieldName"]
        if self._AnnDateField is None: AnnDateField = EndDateField
        else: AnnDateField = self._DBTableName+"."+self._AnnDateField
        if AnnDateField!=EndDateField:
            if IgnoreTime:
                SQLStr = "SELECT DISTINCT STR_TO_DATE(CONCAT(DATE(CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END), ' ', TIME(0)), '%Y-%m-%d %H:%i:%s') AS DT "
            else:
                SQLStr = "SELECT DISTINCT CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END AS DT "
        else:
            if IgnoreTime:
                SQLStr = "SELECT DISTINCT STR_TO_DATE(CONCAT(DATE("+AnnDateField+"), ' ', TIME(0)), '%Y-%m-%d %H:%i:%s') AS DT "
            else:
                SQLStr = "SELECT DISTINCT "+AnnDateField+" AS DT "
        if iid is not None:
            SQLStr += self._genFromSQLStr()+" "
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+"='"+self.__QS_adjustID([iid])[0]+"' "
            if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        else:
            SQLStr += "FROM "+self._DBTableName+" "
            SQLStr += "WHERE "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        if IgnoreTime: DTFormat = self._DTFormat
        else: DTFormat = "'%Y-%m-%d %H:%M:%S'"
        if AnnDateField!=EndDateField:
            if start_dt is not None:
                SQLStr += "AND ("+AnnDateField+">="+start_dt.strftime(DTFormat)+" "
                SQLStr += "OR "+EndDateField+">="+start_dt.strftime(DTFormat)+") "
            if end_dt is not None:
                SQLStr += "AND ("+AnnDateField+"<="+end_dt.strftime(DTFormat)+" "
                SQLStr += "AND "+EndDateField+"<="+end_dt.strftime(DTFormat)+") "
        else:
            if start_dt is not None:
                SQLStr += "AND "+AnnDateField+">="+start_dt.strftime(DTFormat)+" "
            if end_dt is not None:
                SQLStr += "AND "+AnnDateField+"<="+end_dt.strftime(DTFormat)+" "
        SQLStr += self._genConditionSQLStr(args=args)+" "
        SQLStr += "ORDER BY DT"
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def _genNullIDSQLStr_InfoPubl(self, factor_names, ids, end_date, args={}):
        EndDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        if self._AnnDateField is None: AnnDateField = EndDateField
        else: AnnDateField = self._DBTableName+"."+self._AnnDateField
        SubSQLStr = "SELECT "+self._DBTableName+"."+self._IDField+", "
        SubSQLStr += "MAX("+EndDateField+") AS MaxEndDate "
        SubSQLStr += "FROM "+self._DBTableName+" "
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        if IgnoreTime: DTFormat = self._DTFormat
        else: DTFormat = "'%Y-%m-%d %H:%M:%S'"
        SubSQLStr += "WHERE ("+AnnDateField+"<"+end_date.strftime(DTFormat)+" "
        SubSQLStr += "AND "+EndDateField+"<"+end_date.strftime(DTFormat)+") "
        ConditionSQLStr = self._genConditionSQLStr(args=args)
        SubSQLStr += ConditionSQLStr+" "
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            if args.get("预筛选ID", self.PreFilterID):
                SubSQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+self._IDField, self.__QS_adjustID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SubSQLStr += "AND "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField
        if IgnoreTime:
            SQLStr = "SELECT DATE(CASE WHEN "+AnnDateField+">=t.MaxEndDate THEN "+AnnDateField+" ELSE t.MaxEndDate END) AS DT, "
        else:
            SQLStr = "SELECT CASE WHEN "+AnnDateField+">=t.MaxEndDate THEN "+AnnDateField+" ELSE t.MaxEndDate END AS DT, "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t."+self._IDField+"="+self._DBTableName+"."+self._IDField+" "
        SQLStr += "AND "+EndDateField+"=t.MaxEndDate) "
        if not ((self._MainTableName is None) or (self._MainTableName==self._DBTableName)):
            if args.get("预筛选ID", self.PreFilterID):
                SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, self.__QS_adjustID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        else:
            SQLStr += "WHERE TRUE "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += ConditionSQLStr
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if args.get("忽略公告日", self.IgnorePublDate) or (self._AnnDateField is None): return super().__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args)
        IgnoreTime = args.get("忽略时间", self.IgnoreTime)
        if IgnoreTime: DTFormat = self._DTFormat
        else: DTFormat = "'%Y-%m-%d %H:%M:%S'"
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        LookBack = args.get("回溯天数", self.LookBack)
        if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        EndDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("日期字段", self.DateField), "DBFieldName"]
        AnnDateField = self._DBTableName+"."+self._AnnDateField
        SubSQLStr = "SELECT "+self._DBTableName+"."+self._IDField+", "
        if IgnoreTime:
            SubSQLStr += "DATE(CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END) AS AnnDate, "
        else:
            SubSQLStr += "CASE WHEN "+AnnDateField+">="+EndDateField+" THEN "+AnnDateField+" ELSE "+EndDateField+" END AS AnnDate, "
        SubSQLStr += "MAX("+EndDateField+") AS MaxEndDate "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "WHERE ("+AnnDateField+">="+StartDate.strftime(DTFormat)+" "
        SubSQLStr += "OR "+EndDateField+">="+StartDate.strftime(DTFormat)+") "
        SubSQLStr += "AND ("+AnnDateField+"<="+EndDate.strftime(DTFormat)+" "
        SubSQLStr += "AND "+EndDateField+"<="+EndDate.strftime(DTFormat)+") "
        ConditionSQLStr = self._genConditionSQLStr(args=args)
        SubSQLStr += ConditionSQLStr+" "
        if (self._MainTableName is None) or (self._MainTableName==self._DBTableName):
            if args.get("预筛选ID", self.PreFilterID):
                SubSQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+self._IDField, self.__QS_adjustID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SubSQLStr += "AND "+self._DBTableName+"."+self._IDField+" IS NOT NULL "
        if IgnoreTime:
            SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField+", DATE(AnnDate)"
        else:
            SubSQLStr += "GROUP BY "+self._DBTableName+"."+self._IDField+", AnnDate"
        SQLStr = "SELECT t.AnnDate AS DT, "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "t.MaxEndDate AS MaxEndDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "INNER JOIN ("+SubSQLStr+") t "
        SQLStr += "ON (t."+self._IDField+"="+self._DBTableName+"."+self._IDField+") "
        SQLStr += "AND (t.MaxEndDate="+EndDateField+") "
        if not ((self._MainTableName is None) or (self._MainTableName==self._DBTableName)):
            if args.get("预筛选ID", self.PreFilterID):
                SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, self.__QS_adjustID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
            else:
                SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        else:
            SQLStr += "WHERE TRUE "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += ConditionSQLStr+" "
        SQLStr += "ORDER BY ID, DT"
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID", "MaxEndDate"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID", "MaxEndDate"]+factor_names)
        if np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["日期"]==dt.datetime.combine(StartDate,dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr_InfoPubl(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["日期", "ID", "MaxEndDate"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["ID", "日期"])
        if RawData.shape[0]==0: return RawData.loc[:, ["日期", "ID"]+factor_names]
        if args.get("截止日期递增", self.EndDateASC):# 删除截止日期非递增的记录
            DTRank = RawData.loc[:, ["ID", "日期", "MaxEndDate"]].set_index(["ID"]).astype(np.datetime64).groupby(axis=0, level=0).rank(method="min")
            RawData = RawData[(DTRank["日期"]<=DTRank["MaxEndDate"]).values]
        RawData = RawData.loc[:, ["日期", "ID"]+factor_names]
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
class SQL_MultiInfoPublTable(SQL_InfoPublTable):
    """多重信息发布表"""
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=0)
    OnlyStartLookBack = Bool(False, label="只起始日回溯", arg_type="Bool", order=1)
    OnlyLookBackNontarget = Bool(False, label="只回溯非目标日", arg_type="Bool", order=2)
    OnlyLookBackDT = Bool(False, label="只回溯时点", arg_type="Bool", order=3)
    #DateField = Enum(None, arg_type="SingleOption", label="日期字段", order=4)
    IgnorePublDate = Bool(False, label="忽略公告日", arg_type="Bool", order=5)
    IgnoreTime = Bool(True, label="忽略时间", arg_type="Bool", order=6)
    EndDateASC = Bool(False, label="截止日期递增", arg_type="Bool", order=7)
    Operator = Either(Function(None), None, arg_type="Function", label="算子", order=8)
    OperatorDataType = Enum("object", "double", "string", arg_type="SingleOption", label="算子数据类型", order=9)
    OrderFields = List(arg_type="List", label="排序字段", order=10)# [("字段名", "ASC" 或者 "DESC")]
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        self._QS_IgnoredGroupArgs = ("遍历模式", "回溯天数", "只起始日回溯", "只回溯非目标日", "只回溯时点", "算子", "算子数据类型")
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if key=="DataType":
            if factor_names is None: factor_names = self.FactorNames
            if args.get("算子", self.Operator) is None:
                return pd.Series(["object"]*len(factor_names), index=factor_names)
            else:
                return pd.Series([args.get("算子数据类型", self.OperatorDataType)]*len(factor_names), index=factor_names)
        else:
            return super().getFactorMetaData(factor_names=factor_names, key=key, args=args)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        OrderFields = args.get("排序字段", self.OrderFields)
        if OrderFields:
            OrderFields, Orders = np.array(OrderFields).T.tolist()
        else:
            OrderFields, Orders = [], []
        FactorNames = list(set(factor_names).union(OrderFields))
        RawData = super().__QS_prepareRawData__(FactorNames, ids, dts, args=args)
        RawData = RawData.sort_values(by=["ID", "日期"]+OrderFields, ascending=[True, True]+[(iOrder.lower()=="asc") for iOrder in Orders])
        return RawData.loc[:, ["日期", "ID"]+factor_names]
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["日期", "ID"])
        Operator = args.get("算子", self.Operator)
        if Operator is None: Operator = (lambda x: x.tolist())
        if args.get("只回溯时点", self.OnlyLookBackDT):
            DeduplicatedIndex = raw_data.index(~raw_data.index.duplicated())
            RowIdxMask = pd.Series(False, index=DeduplicatedIndex).unstack(fill_value=True).astype(bool)
            RawIDs = RowIdxMask.columns
            if RawIDs.intersection(ids).shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
            RowIdx = pd.DataFrame(np.arange(RowIdxMask.shape[0]).reshape((RowIdxMask.shape[0], 1)).repeat(RowIdxMask.shape[1], axis=1), index=RowIdxMask.index, columns=RawIDs)
            RowIdx[RowIdxMask] = np.nan
            RowIdx = adjustDataDTID(pd.Panel({"RowIdx": RowIdx}), args.get("回溯天数", self.LookBack), ["RowIdx"], RowIdx.columns.tolist(), dts, 
                                              args.get("只起始日回溯", self.OnlyStartLookBack), 
                                              args.get("只回溯非目标日", self.OnlyLookBackNontarget), 
                                              logger=self._QS_Logger).iloc[0].values
            RowIdx[pd.isnull(RowIdx)] = -1
            RowIdx = RowIdx.astype(int)
            ColIdx = np.arange(RowIdx.shape[1]).reshape((1, RowIdx.shape[1])).repeat(RowIdx.shape[0], axis=0)
            RowIdxMask = (RowIdx==-1)
            Data = {}
            for iFactorName in factor_names:
                iRawData = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
                iRawData = iRawData.values[RowIdx, ColIdx]
                iRawData[RowIdxMask] = None
                Data[iFactorName] = pd.DataFrame(iRawData, index=dts, columns=RawIDs)
            return pd.Panel(Data).loc[factor_names, :, ids]
        else:
            Data = {}
            for iFactorName in factor_names:
                Data[iFactorName] = raw_data[iFactorName].groupby(axis=0, level=[0, 1]).apply(Operator).unstack()
            Data = pd.Panel(Data).loc[factor_names, :, ids]
            return adjustDataDTID(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts, 
                                  args.get("只起始日回溯", self.OnlyStartLookBack), 
                                  args.get("只回溯非目标日", self.OnlyLookBackNontarget),
                                  logger=self._QS_Logger)