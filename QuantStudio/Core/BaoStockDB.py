# coding=utf-8
"""基于 BaoStock 的因子库(http://baostock.com/baostock/)(TODO)"""
import os
import datetime as dt
from typing import Optional, Literal, List, Any

import numpy as np
import pandas as pd
import baostock as bs
from pydantic import FilePath, Field

from QuantStudio.Core import __QS_Error__
from QuantStudio import __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.Core.FactorDB import FactorDB
from QuantStudio.Core.FactorTable import FactorTable
from QuantStudio.Core.Factor import FactorContext, FactorLocalContext
from QuantStudio.Core.Node import Context
from QuantStudio.Core.utils import _QS_calcData_WideTable
from QuantStudio.Tools.IDFun import suffixAShareID
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries
from QuantStudio.Tools.DataTypeFun import dict2id


# 将信息源文件中的表和字段信息导入信息文件
def _importInfo(info_file, info_resource, logger, out_info=False):
    Suffix = info_resource.split(".")[-1]
    if Suffix in ("xlsx", "xls"):
        TableInfo = pd.read_excel(info_resource, "TableInfo").set_index(["TableName"])
        FactorInfo = pd.read_excel(info_resource, "FactorInfo").set_index(['TableName', 'FieldName'])
        ArgInfo = pd.read_excel(info_resource, "ArgInfo").set_index(["TableName", "ArgName"])
    else:
        Msg = ("不支持的库信息文件 : '%s'" % (info_resource,))
        logger.error(Msg)
        raise __QS_Error__(Msg)
    if not out_info:
        try:
            from QuantStudio.Tools.DataTypeFun import writeNestedDict2HDF5
            writeNestedDict2HDF5(TableInfo, info_file, "/TableInfo")
            writeNestedDict2HDF5(FactorInfo, info_file, "/FactorInfo")
            writeNestedDict2HDF5(ArgInfo, info_file, "/ArgInfo")
        except Exception as e:
            logger.warning("更新数据库信息文件 '%s' 失败 : %s" % (info_file, str(e)))
    return (TableInfo, FactorInfo, ArgInfo)


# 更新信息文件
def _updateInfo(info_file, info_resource, logger, out_info=False):
    if out_info: return _importInfo(info_file, info_resource, logger, out_info=out_info)
    if not os.path.isfile(info_file):
        logger.warning("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    elif (os.path.getmtime(info_resource) > os.path.getmtime(info_file)):
        logger.warning("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % info_resource)
    else:
        try:
            from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
            return (readNestedDictFromHDF5(info_file, ref="/TableInfo"),
                    readNestedDictFromHDF5(info_file, ref="/FactorInfo"),
                    readNestedDictFromHDF5(info_file, ref="/ArgInfo"))
        except:
            logger.warning("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    if not os.path.isfile(info_resource): raise __QS_Error__("缺失数据库信息源文件: %s" % info_resource)
    return _importInfo(info_file, info_resource, logger, out_info=out_info)


class _BSTable(FactorTable):
    class __QS_ArgClass__(FactorTable.__QS_ArgClass__):
        IDAdj: Literal["无", "前缀"] = Field(default="无", title="ID调整", frozen=True)
        DTFmt: str = Field(default="", title="时点格式", frozen=True)
        APIArgs: dict = Field(default={}, title="API参数", frozen=True)

    def __init__(self, fdb, args={}, **kwargs):
        super().__init__(fdb=fdb, args=args, **kwargs)
        self._TableInfo = fdb._TableInfo.loc[self._QSArgs.Name]
        self._FactorInfo = fdb._FactorInfo.loc[self._QSArgs.Name]
        if self._QSArgs.Name in fdb._ArgInfo.index.get_level_values(0):
            self._ArgInfo = fdb._ArgInfo.loc[self._QSArgs.Name]
        else:
            self._ArgInfo = pd.DataFrame(columns=fdb._ArgInfo.columns)

    def model_dump(self):
        DumpedMdl = super().model_dump()
        DumpedMdl["__module__"] = "BaoStock"
        DumpedMdl["__qsargs__"]["APIArgs"] = self._getAPIArgs()
        return DumpedMdl

    @property
    def FactorNames(self):
        FactorInfo = self._FactorInfo
        return FactorInfo[FactorInfo["FieldType"] == "因子"].index.tolist()

    def _getAPIArgs(self):
        APIArgs = {}
        for iArgName in self._ArgInfo.index[self._ArgInfo["FieldType"] == "QSArg"]:
            iArgInfo = eval(self._ArgInfo.loc[iArgName, "ArgInfo"])
            if iArgName in self._QSArgs.APIArgs:
                iArgVal = self._QSArgs.APIArgs[iArgName]
                if iArgInfo["arg_type"]=="SingleOption":
                    if iArgVal not in iArgInfo["option_range"]:
                        raise __QS_Error__(f"不支持的参数值: {iArgVal}, 所有可选值: {iArgInfo['option_range']}")
                else:
                    raise __QS_Error__(f"不支持的参数类型: {iArgInfo['arg_type']}")
            else:
                if iArgInfo["arg_type"]=="SingleOption":
                    iArgVal = self._ArgInfo.loc[iArgName, "DefaultValue"]
                    iDataType = self._ArgInfo.loc[iArgName, "DataType"]
                    if iDataType == "int":
                        iArgVal = int(iArgVal)
                    elif iDataType != "str":
                        raise __QS_Error__(f"不支持的参数数据类型: {iDataType}")
                else:
                    raise __QS_Error__(f"不支持的参数类型: {iArgInfo['arg_type']}")
            APIArgs[iArgName] = iArgVal
        return APIArgs

    def __QS_adjustID__(self, ids):
        IDAdj = self._QSArgs.IDAdj
        if IDAdj == "无":
            return ids
        elif IDAdj == "前缀":
            return [iID.split(".")[-1].lower() + "." + ".".join(iID.split(".")[:-1]) if "." in iID else iID for iID in ids]
        else:
            raise __QS_Error__(f"BaoStockDB._BSTable: 不支持的 ID 调整方法 '{IDAdj}'")

    def __QS_restoreID__(self, ids):
        return ids

    def __QS_adjustDT__(self, dts):
        DTFmt = self._QSArgs.DTFmt
        if not DTFmt:
            return dts
        else:
            return [dt.datetime.strptime(iDT, DTFmt) if pd.notnull(iDT) else pd.NaT for iDT in dts]

    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        return None

    def __QS_saveRawData__(self, raw_data, key, target_fields, pid_ids, context: FactorContext, **kwargs):
        if raw_data is None: return 0
        Cache = context.FactorDataCache
        MaskCols = raw_data.columns.intersection(self._QS_RawDataMaskCols).tolist()
        CommonCols = raw_data.columns.difference(target_fields).tolist()
        for iFactorName in target_fields:
            iRawData = raw_data.loc[:, CommonCols+[iFactorName]]
            iKey = key+"-"+iFactorName
            iOldData = Cache.readRawData(iKey, target_fields=None, pids=None)
            if iOldData:
                iOldData = iOldData["RawData"]
                iOldData["QS_Mask"] = 1
                iRawData = pd.merge(iRawData, iOldData.loc[:, [*MaskCols, "QS_Mask"]], how="left", left_on=MaskCols, right_on=MaskCols)
                iOldData.pop("QS_Mask")
                iRawData = pd.concat([iOldData, iRawData[iRawData.pop("QS_Mask").isnull()]], ignore_index=True).sort_values(MaskCols)
            Cache.writeRawData(iKey, {"RawData": iRawData}, pid_ids, id_col="QS_ID", if_exists="replace")

    def getMetaData(self, key=None):
        TableInfo = self._FactorDB._TableInfo.loc[self._QSArgs.Name]
        if key is None:
            return TableInfo
        else:
            return TableInfo.get(key, None)

    def getFactorMetaData(self, factor_name, key=None):
        FactorInfo = self._FactorDB._FactorInfo.loc[self._QSArgs.Name]
        if key == "DataType":
            if hasattr(self, "_DataType"): return self._DataType.loc[factor_name]
            iDataType = FactorInfo.loc[factor_name, "DataType"].lower()
            if iDataType.find("str") != -1:
                iDataType = "string"
            else:
                iDataType = "double"
            return iDataType
        elif key == "Description":
            return FactorInfo.loc[factor_name, "Description"]
        elif key is None:
            return {
                "DataType": self.getFactorMetaData(factor_name, key="DataType"),
                "Description": self.getFactorMetaData(factor_name, key="Description")
            }
        else:
            return None


class _DTTable(_BSTable):
    """DTTable"""

    class __QS_ArgClass__(_BSTable.__QS_ArgClass__):
        LookBack: int = Field(default=0, title="回溯天数", frozen=True, ge=0)

    def __init__(self, fdb, args={}, **kwargs):
        super().__init__(fdb=fdb, args=args, **kwargs)
        self._QS_PrepareIgnoredArgs += ("LookBack",)

    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDT = dts[0] - dt.timedelta(args.get("回溯天数", self._QSArgs.LookBack))
        DTs = getDateTimeSeries(StartDT, dts[0]) + dts[1:]
        APIName = self._TableInfo.loc["DBTableName"]
        ArgInfo = self._ArgInfo
        DTArg = ArgInfo.index[ArgInfo["FieldType"] == "Date"][0]
        APIArgs = {DTArg: None}
        APIArgs.update(self._getAPIArgs())
        RawData = []
        for iDT in DTs:
            APIArgs[DTArg] = iDT.strftime("%Y-%m-%d")
            try:
                iRawData = getattr(self._FactorDB._bs, APIName)(**APIArgs)
            except:
                continue
            iRawData["QS_DT"] = iDT
            RawData.append(iRawData)
        IDField = self._FactorInfo.index[self._FactorInfo["FieldType"] == "ID"][0]
        if RawData:
            RawData = pd.concat(RawData, axis=0, ignore_index=True)
            RawData = RawData.rename(columns={IDField: "ID"}).reindex(columns=["ID", "QS_DT"] + factor_names)
            RawData["ID"] = RawData["ID"].apply(suffixAShareID)
            return RawData.sort_values(by=["ID", "QS_DT"])
        else:
            return pd.DataFrame(columns=["ID", "QS_DT"] + factor_names)

    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType", args=args)
        Args = self.Args.to_dict()
        Args.update(args)
        ErrorFmt = {
            "DuplicatedIndex": "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (
                self._FactorDB.Name, self.Name)}
        return _QS_calcData_WideTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger,
                                      error_fmt=ErrorFmt)


class _DTRangeTable(_BSTable):
    """DTRangeTable"""

    class __QS_ArgClass__(_BSTable.__QS_ArgClass__):
        LookBack: int = Field(default=0, title="回溯天数", frozen=True, ge=0)

    def __init__(self, fdb, args={}, **kwargs):
        super().__init__(fdb=fdb, args=args, **kwargs)
        self._QS_PrepareIgnoredArgs += ("LookBack",)

    def __QS_prepareRawData__(self, ids, dts, args={}, factor_names=None, **kwargs):
        if not factor_names: factor_names = self.FactorNames
        StartDate, EndDate = dts[0].date(), dts[-1].date()
        StartDate -= dt.timedelta(args.get("LookBack", self._QSArgs.LookBack))
        APIName = self._TableInfo.loc["DBTableName"]
        ArgInfo = self._ArgInfo
        DTField = self._FactorInfo.index[self._FactorInfo["FieldType"] == "Date"][0]
        StartDTArg = ArgInfo.index[ArgInfo["FieldType"] == "StartDate"][0]
        EndDTArg = ArgInfo.index[ArgInfo["FieldType"] == "EndDate"][0]
        IDArg = ArgInfo.index[ArgInfo["FieldType"] == "ID"][0]
        FieldArg = ArgInfo.index[ArgInfo["FieldType"] == "Field"][0]
        APIArgs = {
            StartDTArg: StartDate.strftime("%Y-%m-%d"),
            EndDTArg: EndDate.strftime("%Y-%m-%d"),
            FieldArg: DTField+","+",".join(factor_names)
        }
        APIArgs.update(self._getAPIArgs())
        AdjustedIDs = self.__QS_adjustID__(ids)
        RawData = []
        for i, iID in enumerate(AdjustedIDs):
            APIArgs[IDArg] = iID
            try:
                iRawData = getattr(bs, APIName)(**APIArgs).get_data()
            except:
                continue
            iRawData["QS_ID"] = ids[i]
            RawData.append(iRawData)
        if RawData:
            RawData = pd.concat(RawData, axis=0, ignore_index=True)
            RawData = RawData.rename(columns={DTField: "QS_DT"}).reindex(columns=["QS_ID", "QS_DT"] + factor_names)
            RawData["QS_DT"] = self.__QS_adjustDT__(RawData["QS_DT"])
            return RawData.sort_values(by=["QS_ID", "QS_DT"])
        else:
            return pd.DataFrame(columns=["QS_ID", "QS_DT"] + factor_names)

    def __QS_calcData__(self, raw_data, ids, dts, factor_names=None, **kwargs):
        if not factor_names: factor_names = self.FactorNames
        DataType = pd.Series({iFactorName: self.getFactorMetaData(factor_name=iFactorName, key="DataType") for iFactorName in factor_names})
        Args = self._QSArgs.to_dict(repr=False)
        ErrorFmt = {"DuplicatedIndex": "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        return _QS_calcData_WideTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)


class BaoStockDB(FactorDB):
    """BaoStockDB"""

    class __QS_ArgClass__(FactorDB.__QS_ArgClass__):
        Name: str = Field(default="BaoStockDB", title="名称", frozen=True)
        UserID: str = Field(default="anonymous", title="用户ID", frozen=True)
        Pwd: str = Field(default="123456", title="密码", frozen=True)
        DBInfoFile: Optional[FilePath] = Field(default=None, title="库信息文件", frozen=True)
        FTArgs: dict = Field(default={}, title="因子表参数", frozen=True)

    def __init__(self, args={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=(__QS_ConfigPath__ + os.sep + "BaoStockDBConfig.json" if config_file is None else config_file), **kwargs)
        self._InfoFilePath = __QS_MainPath__ + os.sep + "Lib" + os.sep + "BaoStockDBInfo.hdf5"  # 数据库信息文件路径
        if (not self._QSArgs.DBInfoFile) or (not os.path.isfile(self._QSArgs.DBInfoFile)):
            if self._QSArgs.DBInfoFile: self._QS_Logger.warning("找不到指定的库信息文件 : '%s'" % self._QSArgs.DBInfoFile)
            self._InfoResourcePath = __QS_MainPath__ + os.sep + "Resource" + os.sep + "BaoStockDBInfo.xlsx"  # 默认数据库信息源文件路径
            self._TableInfo, self._FactorInfo, self._ArgInfo = _updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger)  # 数据库表信息, 数据库字段信息
        else:
            self._InfoResourcePath = self._QSArgs.DBInfoFile
            self._TableInfo, self._FactorInfo, self._ArgInfo = _updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger, out_info=True)  # 数据库表信息, 数据库字段信息

    @property
    def TableNames(self):
        if self._TableInfo is not None:
            return self._TableInfo.index.tolist()
        else:
            return []

    def connect(self):
        LG = bs.login(user_id=self._QSArgs.UserID, password=self._QSArgs.Pwd)
        if LG.error_code != "0":
            raise __QS_Error__(f"BaoStockDB.connect 登录错误码: {LG.error_code}, 错误信息: {LG.error_msg}")
        return self

    def disconnect(self):
        LG = bs.logout(user_id=self._QSArgs.UserID)
        if LG.error_code != "0":
            self._QS_Logger.error(f"BaoStockDB.disconnect 登出错误码: {LG.error_code}, 错误信息: {LG.error_msg}")
        return 0

    def getTable(self, table_name, args={}):
        if table_name in self._TableInfo.index:
            TableClass = args.get("因子表类型", self._TableInfo.loc[table_name, "TableClass"])
            if pd.notnull(TableClass) and (TableClass != ""):
                DefaultArgs = self._TableInfo.loc[table_name, "DefaultArgs"]
                if pd.isnull(DefaultArgs):
                    DefaultArgs = {}
                else:
                    DefaultArgs = eval(DefaultArgs)
                Args = self._QSArgs.FTArgs.copy()
                Args.update(DefaultArgs)
                Args.update(args)
                Args["Name"] = table_name
                return eval("_" + TableClass + "(fdb=self, args=Args, logger=self._QS_Logger)")
        Msg = ("因子库 '%s' 目前尚不支持因子表: '%s'" % (self.Name, table_name))
        self._QS_Logger.error(Msg)
        raise __QS_Error__(Msg)

    def _getResult(self, rs):
        Rslt = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            Rslt.append(rs.get_row_data())
        Rslt = pd.DataFrame(Rslt, columns=rs.fields)
        return Rslt

    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if exchange != "SSE":
            self._QS_Logger.warning(f"BaoStockDB.getTradeDay 的参数 exchange 暂不支持除了 'SSE' 外的其他选项: '{exchange}', 该参数将被忽略!")
        self.connect()
        rs = bs.query_trade_dates(start_date=(None if start_date is None else start_date.strftime("%Y-%m-%d")), end_date=(None if end_date is None else end_date.strftime("%Y-%m-%d")))
        if rs.error_code != "0":
            raise __QS_Error__(f"BaoStockDB.getTradeDay query_trade_dates 错误码: {rs.error_code}, 错误信息: {rs.error_msg}")
        Rslt = rs.get_data()
        return Rslt["calendar_date"][Rslt["is_trading_day"] == "1"].apply(lambda d: dt.datetime.strptime(d, "%Y-%m-%d")).tolist()


if __name__ == "__main__":
    BSDB = BaoStockDB().connect()
    print(BSDB.TableNames)

    # DTs = BSDB.getTradeDay(start_date=dt.datetime(2022, 1, 1), end_date=dt.datetime(2022, 1, 31))
    # print(DTs)

    # CF = BSDB.getFactor("A股K线数据", args={})
    # Data = CF.readData(
    #     factor_names=["open", "close"],
    #     ids=["600000.SH"],
    #     dts=[dt.datetime(2022, 10, 28), dt.datetime(2022, 10, 31)]
    # )
    # print(Data)

    CF1 = BSDB.getTable("A股K线数据", args={"APIArgs": {"frequency": "d"}, "LookBack": 1})
    print(CF1.model_dump())
    print(CF1.QSID)
    print(CF1.PrepareID)
    CF2 = BSDB.getTable("A股K线数据", args={"LookBack": 0})
    print(CF2.model_dump())
    print(CF2.QSID)
    print(CF2.PrepareID)

    # CF = BSDB.getFactor("A股K线数据", args={})
    # F = CF.getFactor("open")
    # Data = F.readData(
    #     ids=["600000.SH"],
    #     dts=[dt.datetime(2022, 10, 28), dt.datetime(2022, 10, 31)]
    # )
    # print(Data)

    print("===")