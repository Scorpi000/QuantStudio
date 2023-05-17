# coding=utf-8
"""聚源数据库"""
import re
import os
import json
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, Int, Str, List, ListStr, Dict, Callable, File

from QuantStudio.Tools.api import Panel
from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.QSObjects import QSSQLObject
from QuantStudio import __QS_Error__, __QS_LibPath__, __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB
from QuantStudio.FactorDataBase.FDBFun import adjustDataDTID, SQL_Table, SQL_FeatureTable, SQL_WideTable, SQL_MappingTable, SQL_NarrowTable, SQL_TimeSeriesTable, SQL_ConstituentTable, SQL_FinancialTable

# 将信息源文件中的表和字段信息导入信息文件
def _importInfo(info_file, info_resource, logger, out_info=False):
    Suffix = info_resource.split(".")[-1]
    if Suffix in ("xlsx", "xls"):
        TableInfo = pd.read_excel(info_resource, "TableInfo", engine="openpyxl").set_index(["TableName"])
        FactorInfo = pd.read_excel(info_resource, "FactorInfo", engine="openpyxl").set_index(['TableName', 'FieldName'])
        ExchangeInfo = pd.read_excel(info_resource, "ExchangeInfo", dtype={"ExchangeCode":"O"}, engine="openpyxl").set_index(["ExchangeCode"])
        SecurityInfo = pd.read_excel(info_resource, "SecurityInfo", dtype={"SecurityCategoryCode":"O"}, engine="openpyxl").set_index(["SecurityCategoryCode"])
    elif Suffix=="json":
        Info = json.load(open(info_resource, "r"))
        TableInfo = pd.DataFrame(Info["TableInfo"]).T
        ExchangeInfo = pd.DataFrame(Info["ExchangeInfo"]).T
        SecurityInfo = pd.DataFrame(Info["SecurityInfo"]).T
        TableNames = sorted(Info["FactorInfo"].keys())
        FactorInfo = pd.concat([pd.DataFrame(Info["FactorInfo"][iTableName]).T for iTableName in TableNames], keys=TableNames)
    else:
        Msg = ("不支持的库信息文件 : '%s'" % (info_resource, ))
        logger.error(Msg)
        raise __QS_Error__(Msg)
    if not out_info:
        try:
            from QuantStudio.Tools.DataTypeFun import writeNestedDict2HDF5
            writeNestedDict2HDF5(TableInfo, info_file, "/TableInfo")
            writeNestedDict2HDF5(FactorInfo, info_file, "/FactorInfo")
            writeNestedDict2HDF5(ExchangeInfo, info_file, "/ExchangeInfo")
            writeNestedDict2HDF5(SecurityInfo, info_file, "/SecurityInfo")
        except Exception as e:
            logger.warning("更新数据库信息文件 '%s' 失败 : %s" % (info_file, str(e)))
    return (TableInfo, FactorInfo, ExchangeInfo, SecurityInfo)

# 更新信息文件
def _updateInfo(info_file, info_resource, logger, out_info=False):
    if out_info: return _importInfo(info_file, info_resource, logger, out_info=out_info)
    if not os.path.isfile(info_file):
        logger.warning("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    elif (os.path.getmtime(info_resource)>os.path.getmtime(info_file)):
        logger.warning("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % info_resource)
    else:
        try:
            from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
            return (readNestedDictFromHDF5(info_file, ref="/TableInfo"), readNestedDictFromHDF5(info_file, ref="/FactorInfo"), readNestedDictFromHDF5(info_file, ref="/ExchangeInfo"), readNestedDictFromHDF5(info_file, ref="/SecurityInfo"))
        except:
            logger.warning("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    if not os.path.isfile(info_resource): raise __QS_Error__("缺失数据库信息源文件: %s" % info_resource)
    return _importInfo(info_file, info_resource, logger, out_info=out_info)

# 给 ID 去后缀
def deSuffixID(ids, sep='.'):
    return [(".".join(iID.split(".")[:-1]) if iID.find(".")!=-1 else iID) for iID in ids]
# 根据字段的数据类型确定 QS 的数据类型
def _identifyDataType(field_data_type):
    field_data_type = field_data_type.lower()
    if (field_data_type.find("num")!=-1) or (field_data_type.find("int")!=-1) or (field_data_type.find("decimal")!=-1) or (field_data_type.find("double")!=-1) or (field_data_type.find("float")!=-1):
        return "double"
    elif field_data_type.find("date")!=-1:
        return "object"
    else:
        return "string"


class _JY_SQL_Table(SQL_Table):
    def _genFieldSQLStr(self, factor_names):
        SQLStr = ""
        JoinStr = []
        SETables = set()
        for iField in factor_names:
            iInfo = self._FactorInfo.loc[iField, "Supplementary"]
            if isinstance(iInfo, str) and (iInfo.find("从表")!=-1):
                iInfo = iInfo.split(":")[1:]
                if len(iInfo)==2:
                    iSETable, iJoinField = iInfo
                    iTypeCode = None
                else:
                    iSETable, iJoinField, iTypeCode = iInfo
                SQLStr += iSETable+".Code, "
                if iSETable not in SETables:
                    if iTypeCode is None:
                        JoinStr.append("LEFT JOIN "+iSETable+" ON ("+self._DBTableName+".ID="+iSETable+"."+iJoinField+")")
                    else:
                        JoinStr.append("LEFT JOIN "+iSETable+" ON ("+self._DBTableName+".ID="+iSETable+"."+iJoinField+" AND "+iSETable+".TypeCode="+iTypeCode+")")
                    SETables.add(iSETable)
            else:
                SQLStr += self._DBTableName+"."+self._FactorInfo.loc[iField, "DBFieldName"]+", "
        return (SQLStr[:-2], JoinStr)
    def __QS_adjustID__(self, ids):
        return [(".".join(iID.split(".")[:-1]) if iID.find(".")!=-1 else iID) for iID in ids]
    def _getSecuMainIDField(self):
        ExchangeInfo = self._FactorDB._ExchangeInfo
        IDField = "CASE SecuMarket "
        for iCode in ExchangeInfo[pd.notnull(ExchangeInfo["Suffix"])].index:
            IDField += "WHEN "+iCode+" THEN CONCAT(SecuCode, '"+ExchangeInfo.loc[iCode, "Suffix"]+"') "
        IDField += "ELSE SecuCode END"
        return IDField
    def _adjustRawDataByRelatedField(self, raw_data, fields):
        RelatedFields = self._FactorInfo["RelatedSQL"].reindex(index=fields)
        RelatedFields = RelatedFields[pd.notnull(RelatedFields)]
        if RelatedFields.shape[0]==0: return raw_data
        for iField in RelatedFields.index:
            iOldData = raw_data.pop(iField)
            iDataMask = pd.notnull(iOldData)
            iOldDataType = _identifyDataType(self._FactorInfo.loc[iField[:-2], "DataType"])
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
                    iKeys = iOldData[iDataMask].unique().tolist()
                    if iKeys:
                        KeyCondition = genSQLInCondition(KeyField, iKeys, is_str=(iOldDataType!="double"))
                    else:
                        KeyCondition = KeyField+" IN (NULL)"
                    iSQLStr = iSQLStr.replace("{KeyCondition}"+KeyField, "{KeyCondition}")
                else:
                    KeyCondition = ""
                if iSQLStr.find("{Keys}")!=-1:
                    if iOldDataType!="double":
                        Keys = "'"+"', '".join([str(iKey) for iKey in iOldData[pd.notnull(iOldData)].unique()])+"'"
                    else:
                        Keys = ", ".join([str(iKey) for iKey in iOldData[pd.notnull(iOldData)].unique()])
                    if not Keys: Keys = "NULL"
                else:
                    Keys = ""
                if iSQLStr.find("{SecuCode}")!=-1:
                    SecuCode = self._getSecuMainIDField()
                else:
                    SecuCode = ""
                iMapInfo = self._FactorDB.fetchall(iSQLStr.format(TablePrefix=self._FactorDB._QSArgs.TablePrefix, Keys=Keys, KeyCondition=KeyCondition, SecuCode=SecuCode))
            iDataType = _identifyDataType(self._FactorInfo.loc[iField, "DataType"])
            if iDataType=="double":
                iNewData = pd.Series(np.nan, index=raw_data.index, dtype="float")
            else:
                iNewData = pd.Series(np.full(shape=(raw_data.shape[0], ), fill_value=None, dtype="O"), index=raw_data.index, dtype="O")
            #for jVal, jRelatedVal in iMapInfo:
                #if pd.notnull(jVal):
                    #if iOldDataType!="double":
                        #iNewData[iOldData==str(jVal)] = jRelatedVal
                    #elif isinstance(jVal, str):
                        #iNewData[iOldData==float(jVal)] = jRelatedVal
                    #else:
                        #iNewData[iOldData==jVal] = jRelatedVal
                #else:
                    #iNewData[pd.isnull(iOldData)] = jRelatedVal
            iMapInfo = pd.DataFrame(iMapInfo, columns=["Old", "New"])
            iMapMask = pd.isnull(iMapInfo.iloc[:, 0])
            if iMapMask.sum()>1:
                raise __QS_Error__("数据映射错误 : 缺失数据对应多个值!")
            elif iMapMask.sum()==1:
                iNewData[~iDataMask] = iMapInfo.iloc[:, 1][iMapMask].iloc[0]
            iMapInfo = iMapInfo[~iMapMask]
            if iOldDataType!="double":
                iMapInfo["Old"] = iMapInfo["Old"].astype(str)
            else:
                iMapInfo["Old"] = iMapInfo["Old"].astype(float)
                iOldData = iOldData.astype(float)
            iMapInfo = iMapInfo.set_index(["Old"]).iloc[:, 0]
            iNewData[iDataMask] = iOldData.map(iMapInfo)[iDataMask]
            raw_data[iField] = iNewData
        return raw_data

class _WideTable(_JY_SQL_Table, SQL_WideTable):
    """聚源宽因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)

class _NarrowTable(_JY_SQL_Table, SQL_NarrowTable):
    """聚源窄因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)

class _FeatureTable(_JY_SQL_Table, SQL_FeatureTable):
    """聚源特征因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)

class _TimeSeriesTable(_JY_SQL_Table, SQL_TimeSeriesTable):
    """聚源时序因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)

class _MappingTable(_JY_SQL_Table, SQL_MappingTable):
    """聚源映射因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)

class _ConstituentTable(_JY_SQL_Table, SQL_ConstituentTable):
    """聚源成份因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)

class _FinancialTable(_JY_SQL_Table, SQL_FinancialTable):
    """聚源财务因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)

# 财务指标因子表, 表结构特征:
# 报告期字段, 表示财报的报告期
# 无公告日期字段, 需另外补充完整
class _FinancialIndicatorTable(_FinancialTable):
    """财务指标因子表"""
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
        if self._TableInfo["SecurityType"] not in ("A股", "公募基金"):
            raise __QS_Error__("FinancialIndicatorTable 类型的因子表 '%s' 中的证券为不支持的证券类型!" % (name, ))
        return
    def getID(self, ifactor_name=None, idt=None, args={}):# TODO
        return []
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):# TODO
        return []
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if self._TableInfo["SecurityType"]=="A股":
            return self._prepareRawDataAStock(factor_names=factor_names, ids=ids, dts=dts, args=args)
        elif self._TableInfo["SecurityType"]=="公募基金":
            return self._prepareRawDataMF(factor_names=factor_names, ids=ids, dts=dts, args=args)
        else:
            raise __QS_Error__("FinancialIndicatorTable 类型的因子表 '%s' 中的证券为不支持的证券类型!" % (self.Name, ))
    def _prepareRawDataAStock(self, factor_names, ids, dts, args={}):
        ReportDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
        SQLStr = "SELECT "+self._getIDField(args=args)+" AS ID, "
        SQLStr += "LC_BalanceSheetAll.InfoPublDate, "
        SQLStr += ReportDateField+", "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "INNER JOIN LC_BalanceSheetAll "
        SQLStr += "ON ("+IDField+"=LC_BalanceSheetAll.CompanyCode "
        SQLStr += "AND "+ReportDateField+"=LC_BalanceSheetAll.EndDate) "
        SQLStr += "WHERE LC_BalanceSheetAll.BulletinType = 20 "
        SQLStr += "AND LC_BalanceSheetAll.IfMerged = 1 "
        SQLStr += "AND LC_BalanceSheetAll.IfAdjusted = 2 "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += self._genIDSQLStr(ids, args=args)+" "
        SQLStr += "ORDER BY ID, LC_BalanceSheetAll.InfoPublDate, "
        SQLStr += ReportDateField
        #RawData = self._FactorDB.fetchall(SQLStr)
        #if not RawData: return pd.DataFrame(columns=["ID", "AnnDate", "ReportDate"]+factor_names)
        #else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "AnnDate", "ReportDate"]+factor_names)
        RawData = pd.read_sql_query(SQLStr, self._FactorDB.Connection)
        RawData.columns = ["ID", "AnnDate", "ReportDate"]+factor_names
        RawData["AdjustType"] = 0
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData
    def _prepareRawDataMF(self, factor_names, ids, dts, args={}):
        IDField = args.get("ID字段", self._QSArgs.IDField)
        IDField = self._DBTableName+"."+self._FactorInfo.loc[(IDField if IDField is not None else self._IDField), "DBFieldName"]
        ReportDateField = self._DBTableName+"."+self._FactorInfo.loc[args.get("时点字段", self._QSArgs.DTField), "DBFieldName"]
        # 形成 SQL 语句, ID, 公告日期, 报告期, 报表类型, 财务因子
        SQLStr = "SELECT "+self._getIDField(args=args)+" AS ID, "
        SQLStr += "MF_BalanceSheetNew.InfoPublDate, "
        SQLStr += ReportDateField+", "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        SQLStr += "INNER JOIN MF_BalanceSheetNew "
        SQLStr += "ON ("+IDField+"=MF_BalanceSheetNew.InnerCode "
        SQLStr += "AND "+ReportDateField+"=MF_BalanceSheetNew.EndDate) "
        SQLStr += "WHERE MF_BalanceSheetNew.Mark = 2 "
        SQLStr += self._genConditionSQLStr(use_main_table=True, args=args)+" "
        SQLStr += self._genIDSQLStr(ids, args=args)+" "
        SQLStr += "ORDER BY ID, MF_BalanceSheetNew.InfoPublDate, "
        SQLStr += ReportDateField
        RawData = pd.read_sql_query(SQLStr, self._FactorDB.Connection)
        RawData.columns = ["ID", "AnnDate", "ReportDate"]+factor_names
        RawData["AdjustType"] = 0
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData

# 查找某个报告期对应的公告期
def findNoteDate(report_date, report_note_dates):
    for i in range(0, report_note_dates.shape[0]):
        if report_date==report_note_dates['报告期'].iloc[i]: return report_note_dates['公告日期'].iloc[i]
    return None
# 生成报告期-公告日期 SQL 查询语句
def genANN_ReportSQLStr(db_type, table_prefix, ids, report_period="1231", pre_filter_id=True):
    DBTableName = table_prefix+"LC_BalanceSheetAll"
    # 提取财报的公告期数据, ID, 公告日期, 报告期
    SQLStr = "SELECT CASE "+table_prefix+"SecuMain.SecuMarket WHEN 83 THEN CONCAT("+table_prefix+"SecuMain.SecuCode, '.SH') "
    SQLStr += "WHEN 90 THEN CONCAT("+table_prefix+"SecuMain.SecuCode, '.SZ') "
    SQLStr += "WHEN 18 THEN CONCAT("+table_prefix+"SecuMain.SecuCode, '.BJ') "
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
    SQLStr += "WHERE "+table_prefix+"SecuMain.SecuCategory IN (1,41) AND "+table_prefix+"SecuMain.SecuMarket IN (18,83,90) "
    if pre_filter_id:
        SQLStr += "AND ("+genSQLInCondition(table_prefix+"SecuMain.SecuCode", deSuffixID(ids), is_str=True, max_num=1000)+") "
    else:
        SQLStr += "AND "+table_prefix+"SecuMain.SecuCode IS NOT NULL "
    if report_period is not None:
        SQLStr += "AND "+ReportDateStr+" LIKE '%"+report_period+"' "
    SQLStr += "ORDER BY "+table_prefix+"SecuMain.SecuCode, "
    SQLStr += DBTableName+".InfoPublDate, "+DBTableName+".EndDate"
    return SQLStr
def _prepareReportANNRawData(fdb, ids, pre_filter_id=True):
    SQLStr = genANN_ReportSQLStr(fdb.DBType, fdb._QSArgs.TablePrefix, ids, report_period="1231", pre_filter_id=pre_filter_id)
    RawData = fdb.fetchall(SQLStr)
    if not RawData: RawData =  pd.DataFrame(columns=["ID", "公告日期", "报告期"])
    else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["ID", "公告日期", "报告期"])
    return RawData
def _saveRawDataWithReportANN(ft, report_ann_file, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, pre_filter_id=True):
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
            RawData = _prepareReportANNRawData(ft.FactorDB, ids=IDs, pre_filter_id=pre_filter_id)
            super(_JY_SQL_Table, ft).__QS_saveRawData__(RawData, [], raw_data_dir, pid_ids, report_ann_file, pid_lock)
        else:
            pid_lock[PID].release()
    raw_data = raw_data.set_index(['ID'])
    CommonCols = list(raw_data.columns.difference(set(factor_names)))
    AllIDs = set(raw_data.index)
    for iPID, iIDs in pid_ids.items():
        with pd.HDFStore(raw_data_dir+os.sep+iPID+os.sep+file_name+ft._QSArgs.OperationMode._FileSuffix) as iFile:
            iInterIDs = sorted(AllIDs.intersection(iIDs))
            iData = raw_data.loc[iInterIDs]
            for jFactorName in factor_names:
                ijData = iData[CommonCols+[jFactorName]].reset_index()
                if isANNReport: ijData.columns.name = raw_data_dir+os.sep+iPID+os.sep+report_ann_file
                iFile[jFactorName] = ijData
            iFile["_QS_IDs"] = pd.Series(iIDs)
    return 0
class _AnalystConsensusTable(_JY_SQL_Table):
    """分析师汇总表"""
    class __QS_ArgClass__(_JY_SQL_Table.__QS_ArgClass__):
        CalcType = Enum("FY0", "FY1", "FY2", "Fwd12M", label="计算方法", arg_type="SingleOption", order=0, option_range=["FY0", "FY1", "FY2", "Fwd12M"])
        Period = Enum(30,60,90,180, label="周期", arg_type="SingleOption", order=1, option_range=[30,60,90,180])
        LookBack = Int(0, arg_type="Integer", label="回溯天数", order=2)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)
        self._DateField = self._FactorInfo[self._FactorInfo["FieldType"]=="Date"].index[0]
        self._ReportDateField = self._FactorInfo[self._FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._PeriodField = self._FactorInfo[self._FactorInfo["FieldType"]=="Period"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = 'JY财务年报-公告日期'
        self._ANN_ReportFileSuffix = "h5"
        return
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        if StartDT is not None: StartDT -= dt.timedelta(args.get("回溯天数", self._QSArgs.LookBack))
        DateField = self._DBTableName+"."+self._FactorInfo.loc[self._DateField, "DBFieldName"]
        ReportDateField = self._DBTableName+"."+self._FactorInfo.loc[self._ReportDateField, "DBFieldName"]
        PeriodField = self._DBTableName+"."+self._FactorInfo.loc[self._PeriodField, "DBFieldName"]
        # 形成SQL语句, 日期, ID, 报告期, 数据
        if self._FactorDB.DBType=="SQL Server":
            SQLStr = "SELECT TO_CHAR("+DateField+",'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr = "SELECT DATE_FORMAT("+DateField+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr = "SELECT TO_CHAR("+DateField+",'yyyyMMdd'), "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "CONCAT("+ReportDateField+", '1231') AS ReportDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(factor_names)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        if args.get("预筛选ID", self._QSArgs.PreFilterID):
            SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        SQLStr += "AND "+PeriodField+"="+str(args.get("周期", self._QSArgs.Period))+" "
        if StartDT is not None:
            SQLStr += "AND "+DateField+">='"+StartDT.strftime("%Y-%m-%d %H:%M:%S")+"' "
        if EndDT is not None:
            SQLStr += "AND "+DateField+"<='"+EndDT.strftime("%Y-%m-%d %H:%M:%S")+"' "
        SQLStr += "ORDER BY ID, "+DateField+", "+ReportDateField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=['日期','ID','报告期']+factor_names)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=['日期','ID','报告期']+factor_names)
        RawData._QS_ANNReport = (args.get("计算方法", self._QSArgs.CalcType)!="Fwd12M")
        RawData._QS_PreFilterID = args.get("预筛选ID", self._QSArgs.PreFilterID)
        if RawData.shape[0]==0: return RawData
        RawData = self._adjustRawDataByRelatedField(RawData, factor_names)
        return RawData
    def __QS_saveRawData__(self, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, **kwargs):
        return _saveRawDataWithReportANN(self, self._ANN_ReportFileName, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, pre_filter_id=raw_data._QS_PreFilterID)
    def __QS_genGroupInfo__(self, factors, operation_mode):
        PeriodGroup = {}
        for iFactor in factors:
            iConditions = (iFactor.Period, iFactor.PreFilterID)
            if iConditions not in PeriodGroup:
                PeriodGroup[iConditions] = {"FactorNames":[iFactor.Name], 
                                        "RawFactorNames":{iFactor._NameInFT}, 
                                        "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                        "args":{"周期": iFactor.Period, "计算方法":iFactor.CalcType, "回溯天数":iFactor.LookBack}}
            else:
                PeriodGroup[iConditions]["FactorNames"].append(iFactor.Name)
                PeriodGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                PeriodGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], PeriodGroup[iConditions]["StartDT"])
                if iFactor.CalcType!="Fwd12M": PeriodGroup[iConditions]["args"]["计算方法"] = iFactor.CalcType
                PeriodGroup[iConditions]["args"]["回溯天数"] = max(PeriodGroup[iConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in PeriodGroup:
            StartInd = operation_mode.DTRuler.index(PeriodGroup[iConditions]["StartDT"])
            Groups.append((self, PeriodGroup[iConditions]["FactorNames"], list(PeriodGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], PeriodGroup[iConditions]["args"]))
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
                ANNReportData = _prepareReportANNRawData(self._FactorDB, ids, pre_filter_id=args.get("预筛选ID", self._QSArgs.PreFilterID))
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
        return adjustDataDTID(Data, LookBack, factor_names, ids, dts, logger=self._QS_Logger)
    def _calcIDData_FY(self, date_seq, raw_data, report_ann_data, factor_names, lookback, fy_num):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        for k, kFactorName in enumerate(factor_names):
            kRawData = raw_data[pd.notnull(raw_data[kFactorName])]
            tempInd = -1
            tempLen = kRawData.shape[0]
            NoteDate = None
            for i, iDate in enumerate(date_seq):
                while (tempInd<tempLen-1) and (iDate>=kRawData['日期'].iloc[tempInd+1]): tempInd = tempInd+1
                if tempInd<0: continue
                LastYear = str(int(iDate[0:4])-1)
                NoteDate = findNoteDate(LastYear+'1231', report_ann_data)
                if (NoteDate is None) or (NoteDate>iDate):
                    ObjectDate = str(int(LastYear)+fy_num)+'1231'
                else:
                    ObjectDate = str(int(iDate[0:4])+fy_num)+'1231'
                iDate = dt.date(int(iDate[0:4]), int(iDate[4:6]), int(iDate[6:])) 
                for j in range(0, tempInd+1):
                    if kRawData['报告期'].iloc[tempInd-j]==ObjectDate:
                        FYNoteDate = kRawData['日期'].iloc[tempInd-j]
                        if (iDate - dt.date(int(FYNoteDate[0:4]), int(FYNoteDate[4:6]), int(FYNoteDate[6:]))).days<=lookback:
                            StdData[i, k] = kRawData[kFactorName].iloc[tempInd-j]
                            break
        return StdData
    def _calcIDData_Fwd12M(self, date_seq, raw_data, report_ann_data, factor_names, lookback, fy_num):
        StdData = np.full(shape=(len(date_seq), len(factor_names)), fill_value=np.nan)
        for k, kFactorName in enumerate(factor_names):
            kRawData = raw_data[pd.notnull(raw_data[kFactorName])]
            tempInd = -1
            tempLen = kRawData.shape[0]
            NoteDate = None
            for i, iDate in enumerate(date_seq):
                while (tempInd<tempLen-1) and (iDate>=kRawData['日期'].iloc[tempInd+1]): tempInd = tempInd+1
                if tempInd<0: continue
                ObjectDate1 = iDate[0:4]+'1231'
                ObjectDate2 = str(int(iDate[0:4])+1)+'1231'
                ObjectData1 = None
                ObjectData2 = None
                iDate = dt.date(int(iDate[0:4]), int(iDate[4:6]), int(iDate[6:]))
                for j in range(0, tempInd+1):
                    if (ObjectData1 is None) and (kRawData['报告期'].iloc[tempInd-j]==ObjectDate1):
                        NoteDate = kRawData['日期'].iloc[tempInd-j]
                        if (iDate-dt.date(int(NoteDate[0:4]), int(NoteDate[4:6]), int(NoteDate[6:]))).days<=lookback:
                            ObjectData1 = kRawData[kFactorName].iloc[tempInd-j]
                    if (ObjectData2 is None) and (kRawData['报告期'].iloc[tempInd-j]==ObjectDate2):
                        NoteDate = kRawData['日期'].iloc[tempInd-j]
                        if (iDate-dt.date(int(NoteDate[0:4]), int(NoteDate[4:6]), int(NoteDate[6:]))).days<=lookback:
                            ObjectData2 = kRawData[kFactorName].iloc[tempInd-j]
                    if (ObjectData1 is not None) and (ObjectData2 is not None):
                        break
                if (ObjectData1 is not None) and (ObjectData2 is not None):
                    Weight1 = (dt.date(int(ObjectDate1[0:4]), 12, 31) - iDate).days
                    if (iDate.month==2) and (iDate.day==29): Weight1 = Weight1/366
                    else:
                        Weight1 = Weight1/(dt.date(iDate.year+1, iDate.month, iDate.day)-iDate).days
                    StdData[i, k] = Weight1*float(ObjectData1) + (1-Weight1)*float(ObjectData2)
        return StdData

# f: 该算子所属的因子对象或因子表对象
# idt: 当前所处的时点
# iid: 当前待计算的 ID
# x: 当期的数据, 分析师评级时为: DataFrame(columns=["日期", ...]), 分析师盈利预测时为: [DataFrame(columns=["日期", "报告期", ...])], list的长度为向前年数
# args: 参数, {参数名:参数值}
def _DefaultOperator(f, idt, iid, x, args):
    return np.nan
class _AnalystEstDetailTable(_JY_SQL_Table):
    """分析师盈利预测明细表"""
    class __QS_ArgClass__(_JY_SQL_Table.__QS_ArgClass__):
        Operator = Callable(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
        ForwardYears = List(default=[0], label="向前年数", arg_type="ArgList", order=2)
        AdditionalFields = ListStr(arg_type="ListStr", label="附加字段", order=3)
        Deduplication = ListStr(arg_type="ListStr", label="去重字段", order=4)
        Period = Int(180, arg_type="Integer", label="周期", order=5)
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=6, option_range=["double", "string", "object"])
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            self._Owner._InstituteField = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"]=="Institute"].index[0]
            self.Deduplication = [self._Owner._InstituteField]
    
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)
        self._DateField = self._FactorInfo[self._FactorInfo["FieldType"]=="Date"].index[0]
        self._ReportDateField = self._FactorInfo[self._FactorInfo["FieldType"]=="ReportDate"].index[0]
        self._TempData = {}
        self._ANN_ReportFileName = "JY财务年报-公告日期"
        self._ANN_ReportFileSuffix = "h5"
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
        return _saveRawDataWithReportANN(self, self._ANN_ReportFileName, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, pre_filter_id=raw_data._QS_PreFilterID)
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        if StartDT is not None: StartDT -= dt.timedelta(args.get("周期", self._QSArgs.Period))
        AllFields = list(set(factor_names+args.get("附加字段", self._QSArgs.AdditionalFields)+args.get("去重字段", self._QSArgs.Deduplication)))
        DateField = self._DBTableName+"."+self._FactorInfo.loc[self._DateField, "DBFieldName"]
        ReportDateField = self._DBTableName+"."+self._FactorInfo.loc[self._ReportDateField, "DBFieldName"]
        # 形成SQL语句, 日期, ID, 报告期, 研究机构, 因子数据
        if self._FactorDB.DBType=="SQL Server":
            SQLStr = "SELECT TO_CHAR("+DateField+",'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr = "SELECT DATE_FORMAT("+DateField+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr = "SELECT TO_CHAR("+DateField+",'yyyyMMdd'), "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        SQLStr += "CONCAT("+ReportDateField+", '1231') AS ReportDate, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(AllFields)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        if args.get("预筛选ID", self._QSArgs.PreFilterID):
            SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        if StartDT is not None:
            SQLStr += "AND "+DateField+">='"+StartDT.strftime("%Y-%m-%d %H:%M:%S")+"' "
        if EndDT is not None:
            SQLStr += "AND "+DateField+"<='"+EndDT.strftime("%Y-%m-%d %H:%M:%S")+"' "
        SQLStr += "ORDER BY ID, "+DateField+", "+ReportDateField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["日期", "ID", self._ReportDateField]+AllFields)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID", self._ReportDateField]+AllFields)
        RawData._QS_ANNReport = True
        RawData._QS_PreFilterID = args.get("预筛选ID", self._QSArgs.PreFilterID)
        if RawData.shape[0]==0: return RawData
        RawData = self._adjustRawDataByRelatedField(RawData, AllFields)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        Dates = sorted({dt.datetime.combine(iDT.date(), dt.time(0)) for iDT in dts})
        DeduplicationFields = args.get("去重字段", self._QSArgs.Deduplication)
        AdditionalFields = list(set(args.get("附加字段", self._QSArgs.AdditionalFields)+DeduplicationFields))
        AllFields = list(set(factor_names+AdditionalFields))
        ANNReportPath = raw_data.columns.name+("."+self._ANN_ReportFileSuffix if self._ANN_ReportFileSuffix else "")
        raw_data = raw_data.loc[:, ["日期", "ID", self._ReportDateField]+AllFields].set_index(["ID"])
        Period = args.get("周期", self._QSArgs.Period)
        ForwardYears = args.get("向前年数", self._QSArgs.ForwardYears)
        ModelArgs = args.get("参数", self._QSArgs.ModelArgs)
        Operator = args.get("算子", self._QSArgs.Operator)
        DataType = args.get("数据类型", self._QSArgs.DataType)
        if (ANNReportPath is not None) and os.path.isfile(ANNReportPath):
            with pd.HDFStore(ANNReportPath, mode="r") as ANN_ReportFile:
                ANNReportData = ANN_ReportFile["RawData"]
        else:
            ANNReportData = _prepareReportANNRawData(self._FactorDB, ids, pre_filter_id=args.get("预筛选ID", self._QSArgs.PreFilterID))
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
        return Panel(Data, items=factor_names, major_axis=Dates, minor_axis=ids).loc[:, dts]

class _AnalystRatingDetailTable(_JY_SQL_Table):
    """分析师投资评级明细表"""
    class __QS_ArgClass__(_JY_SQL_Table.__QS_ArgClass__):
        Operator = Callable(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
        AdditionalFields = ListStr(arg_type="ListStr", label="附加字段", order=2)
        Deduplication = ListStr(arg_type="ListStr", label="去重字段", order=3)
        Period = Int(180, arg_type="Integer", label="周期", order=4)
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=5, option_range=["double", "string", "object"])
        def __QS_initArgs__(self):
            super().__QS_initArgs__()
            self._Owner._InstituteField = self._Owner._FactorInfo[self._Owner._FactorInfo["FieldType"]=="Institute"].index[0]
            self.Deduplication = [self._Owner._InstituteField]
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        super().__init__(name=name, fdb=fdb, sys_args=sys_args, table_prefix=fdb._QSArgs.TablePrefix, table_info=fdb._TableInfo.loc[name], factor_info=fdb._FactorInfo.loc[name], security_info=fdb._SecurityInfo, exchange_info=fdb._ExchangeInfo, **kwargs)
        self._DateField = self._FactorInfo[self._FactorInfo["FieldType"]=="Date"].index[0]
        self._TempData = {}
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
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if dts is not None:
            StartDT, EndDT = dts[0], dts[-1]
        else:
            StartDT = EndDT = None
        if StartDT is not None: StartDT -= dt.timedelta(args.get("周期", self._QSArgs.Period))
        AllFields = list(set(factor_names+args.get("附加字段", self._QSArgs.AdditionalFields)+args.get("去重字段", self._QSArgs.Deduplication)))
        DateField = self._DBTableName+"."+self._FactorInfo.loc[self._DateField, "DBFieldName"]
        # 形成SQL语句, 日期, ID, 其他字段
        if self._FactorDB.DBType=="SQL Server":
            SQLStr = "SELECT TO_CHAR("+DateField+",'YYYYMMDD'), "
        elif self._FactorDB.DBType=="MySQL":
            SQLStr = "SELECT DATE_FORMAT("+DateField+",'%Y%m%d'), "
        elif self._FactorDB.DBType=="Oracle":
            SQLStr = "SELECT TO_CHAR("+DateField+",'yyyyMMdd'), "
        SQLStr += self._getIDField(args=args)+" AS ID, "
        FieldSQLStr, SETableJoinStr = self._genFieldSQLStr(AllFields)
        SQLStr += FieldSQLStr+" "
        SQLStr += self._genFromSQLStr(setable_join_str=SETableJoinStr)+" "
        if args.get("预筛选ID", self._QSArgs.PreFilterID):
            SQLStr += "WHERE ("+genSQLInCondition(self._MainTableName+"."+self._MainTableID, deSuffixID(ids), is_str=self._IDFieldIsStr, max_num=1000)+") "
        else:
            SQLStr += "WHERE "+self._MainTableName+"."+self._MainTableID+" IS NOT NULL "
        if pd.notnull(self._MainTableCondition): SQLStr += "AND "+self._MainTableCondition+" "
        if StartDT is not None:
            SQLStr += "AND "+DateField+">='"+StartDT.strftime("%Y-%m-%d %H:%M:%S")+"' "
        if EndDT is not None:
            SQLStr += "AND "+DateField+"<='"+EndDT.strftime("%Y-%m-%d %H:%M:%S")+"' "
        SQLStr += "ORDER BY ID, "+DateField
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: return pd.DataFrame(columns=["日期", "ID"]+AllFields)
        else: RawData = pd.DataFrame(np.array(RawData, dtype="O"), columns=["日期", "ID"]+AllFields)
        RawData = self._adjustRawDataByRelatedField(RawData, AllFields)
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        #if raw_data.shape[0]==0: return Panel(np.nan, items=factor_names, major_axis=dts, minor_axis=ids)
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
                if jID not in AllIDs:
                    ijRawData = pd.DataFrame(columns=kFields)
                    for i, iDate in enumerate(Dates):
                        kData[i, j] = Operator(self, iDate, jID, ijRawData, ModelArgs)
                    continue
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
        return Panel(Data, items=factor_names, major_axis=Dates, minor_axis=ids).loc[:, dts]

class JYDB(QSSQLObject, FactorDB):
    """聚源数据库"""
    class __QS_ArgClass__(QSSQLObject.__QS_ArgClass__, FactorDB.__QS_ArgClass__):
        Name = Str("JYDB", arg_type="String", label="名称", order=-100)
        DBInfoFile = File(label="库信息文件", arg_type="File", order=100)
        FTArgs = Dict(label="因子表参数", arg_type="Dict", order=101)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"JYDBConfig.json" if config_file is None else config_file), **kwargs)
        self._InfoFilePath = __QS_LibPath__+os.sep+"JYDBInfo.hdf5"# 数据库信息文件路径
        if not os.path.isfile(self._QSArgs.DBInfoFile):
            if self._QSArgs.DBInfoFile: self._QS_Logger.warning("找不到指定的库信息文件 : '%s'" % self._QSArgs.DBInfoFile)
            self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"JYDBInfo.xlsx"# 默认数据库信息源文件路径
            self._TableInfo, self._FactorInfo, self._ExchangeInfo, self._SecurityInfo = _updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger)# 数据库表信息, 数据库字段信息
        else:
            self._InfoResourcePath = self._QSArgs.DBInfoFile
            self._TableInfo, self._FactorInfo, self._ExchangeInfo, self._SecurityInfo = _updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger, out_info=True)# 数据库表信息, 数据库字段信息
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
        Msg = ("因子库 '%s' 目前尚不支持因子表: '%s'" % (self._QSArgs.Name, table_name))
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)
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
        SQLStr = SQLStr.format(Prefix=self._QSArgs.TablePrefix, ExchangeCode=ExchangeCode,
                               StartDate=start_date.strftime("%Y-%m-%d %H:%M:%S"), EndDate=end_date.strftime("%Y-%m-%d %H:%M:%S"))
        Rslt = self.fetchall(SQLStr)
        if kwargs.get("output_type", "datetime")=="date": return [iRslt[0].date() for iRslt in Rslt]
        else: return [iRslt[0] for iRslt in Rslt]
    # 获取指定日 date 的全体 A 股 ID
    # date: 指定日, datetime.date
    # is_current: False 表示上市日在指定日之前的股票, True 表示上市日在指定日之前且尚未退市的股票
    # start_date: 起始日, 如果非 None, is_current=False 表示提取在 start_date 至 date 之间上市过的股票 ID, is_current=True 表示提取在 start_date 至 date 之间均保持上市的股票
    def _getAllAStock(self, date, is_current=True, start_date=None, exchange=("SSE", "SZSE", "BSE")):
        if start_date is not None: start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        SQLStr = "SELECT CASE {Prefix}SecuMain.SecuMarket WHEN 83 THEN CONCAT({Prefix}SecuMain.SecuCode, '.SH') "
        SQLStr += "WHEN 90 THEN CONCAT({Prefix}SecuMain.SecuCode, '.SZ') "
        SQLStr += "WHEN 18 THEN CONCAT({Prefix}SecuMain.SecuCode, '.BJ') "
        SQLStr += "ELSE {Prefix}SecuMain.SecuCode END FROM {Prefix}SecuMain "
        SQLStr += "WHERE {Prefix}SecuMain.SecuCategory IN (1,41) "
        SecuMarket = ", ".join(str(self._ExchangeInfo[self._ExchangeInfo["Exchange"]==iExchange].index[0]) for iExchange in exchange)
        SQLStr += "AND {Prefix}SecuMain.SecuMarket IN " + f"({SecuMarket}) "
        SQLStr += "AND {Prefix}SecuMain.ListedDate <= '{Date}' "
        if is_current or (start_date is not None):
            SubSQLStr = "SELECT DISTINCT {Prefix}LC_ListStatus.InnerCode FROM {Prefix}LC_ListStatus "
            SubSQLStr += "WHERE {Prefix}LC_ListStatus.ChangeType = 4 "
            if start_date is not None:
                SubSQLStr += "AND {Prefix}LC_ListStatus.ChangeDate <= '{StartDate}' "
            if is_current:
                SubSQLStr += "AND {Prefix}LC_ListStatus.ChangeDate <= '{Date}' "
                if start_date is not None:
                    SQLStr += "AND {Prefix}SecuMain.ListedDate <= '{StartDate}' "
            SQLStr += "AND {Prefix}SecuMain.InnerCode NOT IN ("+SubSQLStr+") "
        SQLStr += "ORDER BY {Prefix}SecuMain.SecuCode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y-%m-%d %H:%M:%S"), StartDate=start_date))]
    # 获取指定日 date 的全体港股 ID
    # date: 指定日, datetime.date
    # is_current: False 表示上市日在指定日之前的港股, True 表示上市日在指定日之前且尚未退市的港股
    # start_date: 起始日, 如果非 None, is_current=False 表示提取在 start_date 至 date 之间上市过的股票 ID, is_current=True 表示提取在 start_date 至 date 之间均保持上市的股票
    def _getAllHKStock(self, date, is_current=True, start_date=None):
        if start_date is not None: start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        SQLStr = "SELECT CONCAT({Prefix}HK_SecuMain.SecuCode, '.HK') "
        SQLStr += "FROM {Prefix}HK_SecuMain "
        SQLStr += "WHERE {Prefix}HK_SecuMain.SecuCategory IN (3,51,53,55,78) AND {Prefix}HK_SecuMain.SecuMarket = 72 "
        SQLStr += "AND {Prefix}HK_SecuMain.ListedDate <= '{Date}' "
        if start_date is not None:
            SQLStr += "AND (({Prefix}HK_SecuMain.DelistingDate IS NULL) OR ({Prefix}HK_SecuMain.DelistingDate > '{StartDate}')) "
        if is_current:
            if start_date is None:
                SQLStr += "AND (({Prefix}HK_SecuMain.DelistingDate IS NULL) OR ({Prefix}HK_SecuMain.DelistingDate > '{Date}')) "
            else:
                SQLStr += "AND {Prefix}HK_SecuMain.ListedDate <= '{StartDate}' "
                SQLStr += "AND (({Prefix}HK_SecuMain.DelistingDate IS NULL) OR ({Prefix}HK_SecuMain.DelistingDate > '{Date}')) "
        SQLStr += "ORDER BY {Prefix}HK_SecuMain.SecuCode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y-%m-%d %H:%M:%S"), StartDate=start_date))]
    # 获取指定日 date 的股票 ID
    # exchange: 交易所(str)或者交易所列表(list(str))
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日期在指定日之前的股票, True 表示上市日期在指定日之前且尚未退市的股票
    # start_date: 起始日, 如果非 None, is_current=False 表示提取在 start_date 至 date 之间上市过的股票 ID, is_current=True 表示提取在 start_date 至 date 之间均保持上市的股票
    def getStockID(self, exchange=("SSE", "SZSE", "BSE"), date=None, is_current=True, start_date=None, **kwargs):
        if date is None: date = dt.date.today()
        if isinstance(exchange, str):
            exchange = {exchange}
        else:
            exchange = set(exchange)
        IDs = []
        # 港股
        if "HKEX" in exchange:
            IDs += self._getAllHKStock(date=date, is_current=is_current, start_date=start_date)
            exchange.remove("HKEX")
        # A 股
        iExchange = {"SSE", "SZSE", "BSE"}
        if not exchange.isdisjoint(iExchange):
            IDs += self._getAllAStock(exchange=exchange.intersection(iExchange), date=date, is_current=is_current, start_date=start_date)
            exchange = exchange.difference(iExchange)
        if exchange:
            Msg = f"外部因子库 '{self._QSArgs.Name}' 调用 getStockID 时错误: 尚不支持交易所 {str(exchange)}"
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        return IDs
    # 获取指定日 date 的债券 ID
    # exchange: 交易所(str)或者交易所列表(list(str))
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示存续起始日在指定日之前的债券, True 表示存续起始日在指定日之前且尚未到期的债券
    # start_date: 起始日, 如果非 None, is_current=False 表示提取在 start_date 至 date 之间存续过的债券 ID, is_current=True 表示提取在 start_date 至 date 之间均保持存续的债券
    def getBondID(self, exchange=None, date=None, is_current=True, start_date=None, **kwargs):
        if date is None: date = dt.date.today()
        if start_date is not None: start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        Exchange = self._TableInfo.loc["债券证券主表", "Exchange"]
        ExchangeField, ExchangeCodes = Exchange.split(":")
        ExchangeCodes = ExchangeCodes.split(",")
        ExchangeInfo = self._ExchangeInfo
        IDField = "CASE tt."+ExchangeField+" "
        for iCode in ExchangeCodes:
            IDField += "WHEN "+iCode+" THEN CONCAT(tt.SecuCode, '"+ExchangeInfo.loc[iCode, "Suffix"]+"') "
        DefaultSuffix = self._TableInfo.loc["债券证券主表", "DefaultSuffix"]
        if pd.isnull(DefaultSuffix):
            IDField += "ELSE tt.SecuCode END"
        else:
            IDField += "ELSE CONCAT(tt.SecuCode, '"+DefaultSuffix+"') END"
        if not exchange:
            ExchgCodes = []
        else:
            if isinstance(exchange, str): exchange = [exchange]
            ExchgCodes = set()
            for iExchg in exchange:
                iExchgCode = ExchangeInfo[ExchangeInfo["Exchange"]==iExchg]
                if iExchgCode.shape[0]==0:
                    Msg = ("外部因子库 '%s' 调用 getBondID 时错误: 尚不支持交易所 '%s'" % (self._QSArgs.Name, iExchg))
                    self._QS_Logger.error(Msg)
                    raise __QS_Error__(Msg)
                ExchgCodes.add(str(iExchgCode[0]))
        # 先使用债券代码对照表进行查询, 出错后使用证券主表查询
        SubSQLStr = "SELECT t.SecuCode, t.SecuMarket, (CASE WHEN t2.StartDate IS NULL THEN t1.ValueDate ELSE t2.StartDate END) AS StartDate, (CASE WHEN t.InterestEndDate IS NULL THEN t1.EndDate ELSE t.InterestEndDate END) AS EndDate "
        SubSQLStr += "FROM {Prefix}Bond_Code t "
        SubSQLStr += "LEFT JOIN {Prefix}Bond_BasicInfoN t1 ON (t.InnerCode = t1.InnerCode) "
        SubSQLStr += "LEFT JOIN {Prefix}Bond_ConBDBasicInfo t2 ON (t.InnerCode = t2.InnerCode) "
        SQLStr = f"SELECT {IDField} AS ID FROM ({SubSQLStr}) tt "
        SQLStr += "WHERE ((tt.StartDate IS NULL) OR (tt.StartDate <= '{Date}')) "
        if ExchgCodes: SQLStr += "AND tt.SecuMarket IN ("+", ".join(ExchgCodes)+") "
        if start_date is not None:
            SQLStr += "AND ((tt.EndDate IS NULL) OR (tt.EndDate >= '{StartDate}')) "
        if is_current:
            if start_date is None:
                SQLStr += "AND ((tt.EndDate IS NULL) OR (tt.EndDate >= '{Date}')) "
            else:
                SQLStr += "AND ((tt.StartDate IS NULL) OR (tt.StartDate <= '{StartDate}')) "
                SQLStr += "AND ((tt.EndDate IS NULL) OR (tt.EndDate >= '{Date}')) "
        SQLStr += "ORDER BY ID"
        try:
            return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y-%m-%d %H:%M:%S"), StartDate=start_date))]
        except Exception as e:
            self._QS_Logger.warning("使用债券代码对照表(Bond_Code)提取债券 ID 失败: %s, 将使用证券主表(SecuMain)提取债券 ID, 将忽略参数 date, is_current, start_date!" % (str(e), ))
        SQLStr = "SELECT "+IDField+" AS ID FROM {Prefix}SecuMain tt "
        SQLStr += "WHERE tt.SecuCategory IN (6,7,9,11,14,17,18,23,28,29,31,33) "
        if not ExchgCodes:
            SQLStr += "AND tt.SecuMarket IN (16,71,73,83,84,89,90,310) "
        else:
            SQLStr += "AND tt.SecuMarket IN ("+", ".join(ExchgCodes)+") "
        SQLStr += "ORDER BY ID"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix))]
    # 给定期货代码 future_code, 获取指定日 date 的期货 ID
    # exchange: 交易所(str)或者交易所列表(list(str))
    # future_code: 期货代码(str)或者期货代码列表(list(str)), None 表示所有期货代码
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日在指定日之前的期货, True 表示上市日在指定日之前且尚未退市的期货
    # kwargs:
    # contract_type: 合约类型, 可选 "月合约", "连续合约", "所有", 默认值 "月合约"
    # continue_contract_type: 连续合约类型, list(str), 可选 "主力合约", "期货指数", "次主力合约", "连续合约", "连一合约", "连二合约", "连三合约", "连四合约", "当月连续合约", "次月连续合约", "当季连续合约", "下季连续合约", "隔季连续合约"
    def getFutureID(self, exchange="CFFEX", future_code="IF", date=None, is_current=True, start_date=None, **kwargs):
        if date is None: date = dt.date.today()
        if start_date is not None: start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        ExchangeInfo = self._ExchangeInfo
        Exchange = self._TableInfo.loc["期货合约", "Exchange"]
        ExchangeField, ExchangeCodes = Exchange.split(":")
        if exchange:
            if isinstance(exchange, str): exchange = [exchange]
            ExchangeCodes = set()
            for iExchg in exchange:
                iExchgCode = ExchangeInfo[ExchangeInfo["Exchange"]==iExchg].index
                if iExchgCode.shape[0]==0: raise __QS_Error__("不支持的交易所: %s" % iExchg)
                ExchangeCodes.add(str(iExchgCode[0]))
        else:
            ExchangeCodes = ExchangeCodes.split(",")
        Suffix = "CASE "+ExchangeField+" "
        for iCode in ExchangeCodes:
            Suffix += "WHEN "+iCode+" THEN '"+ExchangeInfo.loc[iCode, "Suffix"]+"' "
        DefaultSuffix = self._TableInfo.loc["期货合约", "DefaultSuffix"]
        if pd.isnull(DefaultSuffix):
            Suffix += "ELSE '' END"
        else:
            Suffix += "ELSE '"+DefaultSuffix+"' END"
        SQLStr = "SELECT DISTINCT CONCAT(ContractCode, "+Suffix+") AS ID FROM {Prefix}Fut_ContractMain "
        SQLStr += "WHERE "+ExchangeField+" IN ("+",".join(ExchangeCodes)+") "
        if future_code:
            if isinstance(future_code, str):
                SQLStr += "AND LEFT(ContractCode, "+str(len(future_code))+")='"+future_code+"' "
            else:
                SQLStr += "AND (LEFT(ContractCode, 1) IN ('"+"','".join(future_code)+"') OR LEFT(ContractCode, 2) IN ('"+"','".join(future_code)+"')) "
        ContractType = kwargs.get("contract_type", "月合约")
        ContinueContractType = kwargs.get("continue_contract_type", None)
        if ContractType!="所有": SQLStr += "AND IfReal="+("2" if ContractType=="连续合约" else "1")+" "
        if ContractType!="连续合约":
            SQLStr += "AND ((EffectiveDate IS NULL) OR (EffectiveDate <= '{Date}')) "
            if start_date is not None:
                SQLStr += "AND ((LastTradingDate IS NULL) OR (LastTradingDate >= '{StartDate}')) "
            if is_current:
                if start_date is None:
                    SQLStr += "AND ((LastTradingDate IS NULL) OR (LastTradingDate >= '{Date}')) "
                else:
                    SQLStr += "AND ((EffectiveDate IS NULL) OR (EffectiveDate <= '{StartDate}')) "
                    SQLStr += "AND ((LastTradingDate IS NULL) OR (LastTradingDate >= '{Date}')) "
        elif ContinueContractType is not None:
            if isinstance(ContinueContractType, str):
                SubSQLStr = f"SELECT DM FROM {{Prefix}}CT_SystemConst WHERE LB = 2352 AD MS ='{ContinueContractType}'"
            else:
                SubSQLStr = "SELECT DM FROM {Prefix}CT_SystemConst WHERE LB = 2352 AD MS IN ("+"', '".join(ContinueContractType)+"')"
            SQLStr += f"AND ContinueContType IN ({SubSQLStr}) "
        SQLStr += "ORDER BY ID"
        if future_code:
            if isinstance(future_code, str):
                return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y%m%d"), StartDate=start_date)) if re.findall("\D+", iRslt[0][:2])[0] == future_code]
            else:
                return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y%m%d"), StartDate=start_date)) if re.findall("\D+", iRslt[0][:2])[0] in future_code]
        else:
            return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y%m%d"), StartDate=start_date))]
    # 获取指定交易所 exchange 的期货代码
    # exchange: 交易所(str)或者交易所列表(list(str))
    # date: 指定日, 默认值 None 表示今天
    # is_current: True 表示只返回当前上市的期货代码
    # kwargs:
    def getFutureCode(self, exchange=("SHFE", "INE", "DCE", "CZCE", "CFFEX"), date=None, is_current=True, **kwargs):
        if date is not None:
            raise __QS_Error__("尚不支持获取指定日期上市的期货品种!")
        SQLStr = "SELECT DISTINCT TradingCode FROM {Prefix}Fut_FuturesContract "
        if exchange:
            if isinstance(exchange, str): exchange = [exchange]
            ExchgCodes = set()
            for iExchg in exchange:
                iExchgCode = self._ExchangeInfo[self._ExchangeInfo["Exchange"]==iExchg].index
                if iExchgCode.shape[0]==0: raise __QS_Error__("不支持的交易所: %s" % iExchg)
                ExchgCodes.add(str(iExchgCode[0]))
            SQLStr += "WHERE Exchange IN ("+", ".join(ExchgCodes)+") "
        else:
            SQLStr += "WHERE Exchange IS NOT NULL "
        if is_current: SQLStr += "AND ContractState<>5 "
        SQLStr += "ORDER BY TradingCode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix))]
    # 给定期权代码 option_code, 获取指定日 date 的期权代码
    # option_code: 期权代码(str)
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示上市日在指定日之前的期权, True 表示上市日在指定日之前且尚未退市的期权
    def getOptionID(self, option_code="510050", date=None, is_current=True, start_date=None, **kwargs):
        if date is None: date = dt.date.today()
        if start_date is not None: start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        SQLStr = "SELECT DISTINCT TradingCode FROM {Prefix}Opt_OptionContract "
        SQLStr += "WHERE TradingCode LIKE '{OptionCode}%%' "
        SQLStr += "AND IfReal=1 "
        SQLStr += "AND ListingDate <= '{Date}' "
        if start_date is not None:
            SQLStr += "AND ((LastTradingDate IS NULL) OR (LastTradingDate >= '{StartDate}')) "
        if is_current:
            if start_date is None:
                SQLStr += "AND ((LastTradingDate IS NULL) OR (LastTradingDate >= '{Date}')) "
            else:
                SQLStr += "AND ListingDate <= '{StartDate}' "
                SQLStr += "AND ((LastTradingDate IS NULL) OR (LastTradingDate >= '{Date}')) "        
        SQLStr += "ORDER BY TradingCode"
        return [iRslt[0] for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y-%m-%d %H:%M:%S"), StartDate=start_date, OptionCode=option_code))]
    # 获取指定日 date 基金 ID
    # date: 指定日, 默认值 None 表示今天
    # is_current: False 表示成立日在指定日之前的基金, True 表示成立日在指定日之前且尚未清盘的基金
    def getMutualFundID(self, exchange=None, date=None, is_current=True, start_date=None, **kwargs):
        if date is None: date = dt.date.today()
        if start_date is not None: start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        SQLStr = "SELECT CONCAT({Prefix}SecuMain.SecuCode, '.OF') AS ID FROM {Prefix}mf_fundarchives "
        SQLStr += "INNER JOIN {Prefix}SecuMain ON {Prefix}SecuMain.InnerCode={Prefix}mf_fundarchives.InnerCode "
        SQLStr += "WHERE {Prefix}mf_fundarchives.EstablishmentDate <= '{Date}' "
        if start_date is not None:
            SQLStr += "AND (({Prefix}mf_fundarchives.ExpireDate IS NULL) OR ({Prefix}mf_fundarchives.ExpireDate >= '{StartDate}')) "
        if is_current:
            if start_date is None:
                SQLStr += "AND (({Prefix}mf_fundarchives.ExpireDate IS NULL) OR ({Prefix}mf_fundarchives.ExpireDate >= '{Date}')) "
            else:
                SQLStr += "AND {Prefix}mf_fundarchives.EstablishmentDate <= '{StartDate}' "
                SQLStr += "AND (({Prefix}mf_fundarchives.ExpireDate IS NULL) OR ({Prefix}mf_fundarchives.ExpireDate >= '{Date}')) "
        if exchange:
            if isinstance(exchange, str): exchange = [exchange]
            ExchgCodes = set()
            for iExchg in exchange:
                iExchgCode = self._ExchangeInfo[self._ExchangeInfo["Exchange"]==iExchg].index
                if iExchgCode.shape[0]==0: raise __QS_Error__("不支持的交易所: %s" % iExchg)
                ExchgCodes.add(str(iExchgCode[0]))
            SQLStr += "AND {Prefix}SecuMain.SecuMarket IN ("+", ".join(ExchgCodes)+") "
        SQLStr += "ORDER BY ID"
        Rslt = np.array(self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y-%m-%d %H:%M:%S"), StartDate=start_date)))
        if Rslt.shape[0]>0: return Rslt[:, 0].tolist()
        else: return []
    # 获取行业 ID
    def getIndustryID(self, standard="中信行业分类", level=1, date=None, is_current=True, start_date=None, **kwargs):
        SQLStr = ("SELECT DM FROM {Prefix}CT_SystemConst WHERE LB=1081 AND MS='%s'" % (standard, ))
        Standard = self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix))
        if len(Standard)!=1:
            SQLStr = "SELECT DISTINCT MS FROM {Prefix}CT_SystemConst WHERE LB=1081"
            AllStandards = self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix))
            raise __QS_Error__("无法识别的行业分类标准 : %s, 支持的行业分类标准有 : %s" % (standard, ", ".join(iStandard[0] for iStandard in AllStandards)))
        if date is None: date = dt.date.today()
        if start_date is not None: start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        SQLStr = "SELECT IndustryNum FROM {Prefix}CT_IndustryType WHERE Standard="+str(Standard[0][0])+" "
        if pd.notnull(level):
            SQLStr += "AND Classification="+str(int(level))+" "
        SQLStr += "AND ((EffectiveDate IS NULL) OR (EffectiveDate <= '{Date}')) "
        if start_date is not None:
            SQLStr += "AND ((CancelDate IS NULL) OR (CancelDate > '{StartDate}')) "
        if is_current:
            if start_date is None:
                SQLStr += "AND ((CancelDate IS NULL) OR (CancelDate > '{Date}')) "
            else:
                SQLStr += "AND ((EffectiveDate IS NULL) OR (EffectiveDate <= '{StartDate}')) "
                SQLStr += "AND ((CancelDate IS NULL) OR (CancelDate > '{Date}')) "
        SQLStr += "ORDER BY IndustryNum"
        return [str(iRslt[0]) for iRslt in self.fetchall(SQLStr.format(Prefix=self._QSArgs.TablePrefix, Date=date.strftime("%Y-%m-%d %H:%M:%S"), StartDate=start_date))]
    # 获取宏观指标名称对应的指标 ID, TODO
    def getMacroIndicatorID(self, indicators, table_name=None):
        pass

if __name__=="__main__":
    iDB = JYDB()
    iDB.getStockID()