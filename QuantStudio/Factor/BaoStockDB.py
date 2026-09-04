# coding=utf-8
"""基于 BaoStock 的因子库(http://baostock.com/baostock/)(TODO)"""
import os
import datetime as dt
from typing import Optional, Literal, Union, List, Tuple, Any

import numpy as np
import pandas as pd
import baostock as bs
from pydantic import FilePath, Field

from QuantStudio.Core import __QS_Error__
from QuantStudio import __QS_MainPath__, __QS_ConfigPath__
from QuantStudio.Factor.FactorDB import FactorDB
from QuantStudio.Factor.FactorTable import FactorTable
from QuantStudio.Factor.FactorUtils import _QS_calcData_WideTable
from QuantStudio.Tools.DateTimeFun import getDateTimeSeries


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


_DATATYPE_MAPPING = {
    "string": "string",
    "float": "double",
    "datetime": "object"
}


class _BSTable(FactorTable):
    """BaoStockDB 库中因子表"""

    class __QS_ArgClass__(FactorTable.__QS_ArgClass__):
        TableType: str = Field(default="BSTable", title="因子表类型", frozen=True, description="""只能在 getTable 时传入，因子表创建后不可改变, 用于指明形成的因子表的类型""")
        IDAdj: Literal["无", "前缀"] = Field(default="无", title="ID调整", frozen=True, repr=False)
        DTFmt: str = Field(default="", title="时点格式", frozen=True, repr=False)
        APIArgs: dict = Field(default={}, title="API参数", frozen=True, repr=False)

    def __init__(self, fdb:"BaoStockDB", args:dict={}, **kwargs):
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
    def FactorNames(self) -> List[str]:
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

    def __QS_adjustID__(self, ids: list[str]):
        IDAdj = self._QSArgs.IDAdj
        if IDAdj == "无":
            return ids
        elif IDAdj == "前缀":
            return [iID.split(".")[-1].lower() + "." + ".".join(iID.split(".")[:-1]) if "." in iID else iID for iID in ids]
        else:
            raise __QS_Error__(f"BaoStockDB._BSTable: 不支持的 ID 调整方法 '{IDAdj}'")

    def __QS_restoreID__(self, ids: pd.Series):
        ids = ids.str.split(".", expand=True)
        ids[0] = ids[0].str.upper()
        return ids[1] + "." + ids[0]

    def __QS_adjustDT__(self, dts):
        DTFmt = self._QSArgs.DTFmt
        if not DTFmt:
            return dts
        else:
            return [dt.datetime.strptime(iDT, DTFmt) if pd.notnull(iDT) else pd.NaT for iDT in dts]

    def getMetaData(self, key:Optional[str]=None) -> Union[Any, pd.Series]:
        if key is None:
            return self._TableInfo
        else:
            return self._TableInfo.get(key, None)

    def getFactorMetaData(self, factor_names:Optional[List[str]]=None, key:Optional[str]=None) -> Union[pd.DataFrame, pd.Series]:
        if factor_names is None: factor_names = self.FactorNames
        if key == "DataType":
            iDataType = self._FactorInfo.loc[factor_names, "DataType"].str.lower()
            iDataType = iDataType.replace(_DATATYPE_MAPPING).where(iDataType.isin(_DATATYPE_MAPPING), "object")
            return iDataType
        elif key == "Description":
            return self._FactorInfo.loc[factor_names, "Description"]
        elif key is None:
            MetaData = self._FactorInfo.loc[factor_names, ["DataType", "Description"]]
            MetaData["DataType"] = MetaData["DataType"].replace(_DATATYPE_MAPPING).where(MetaData["DataType"].isin(_DATATYPE_MAPPING), "object")
            return MetaData
        else:
            return None


class _DTTable(_BSTable):
    """BaoStockDB 库中基于取单个时点数据 API 的因子表"""

    class __QS_ArgClass__(_BSTable.__QS_ArgClass__):
        TableType: Literal["DTTable"] = Field(default="DTTable", title="因子表类型", frozen=True)
        LookBack: int = Field(default=0, title="回溯天数", frozen=True, ge=0)

    def __init__(self, fdb:"BaoStockDB", args:dict={}, **kwargs):
        super().__init__(fdb=fdb, args=args, **kwargs)
        self._QS_PrepareIgnoredArgs += ("LookBack",)

    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        StartDT = dts[0] - dt.timedelta(args.get("LookBack", self._QSArgs.LookBack))
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
                iRawData = getattr(bs, APIName)(**APIArgs).get_data()
            except:
                continue
            iRawData["QS_DT"] = iDT
            RawData.append(iRawData)
        IDField = self._FactorInfo.index[self._FactorInfo["FieldType"] == "ID"][0]
        if RawData:
            RawData = pd.concat(RawData, axis=0, ignore_index=True)
            RawData = RawData.rename(columns={IDField: "QS_ID"}).reindex(columns=["QS_ID", "QS_DT"] + factor_names)
            RawData["QS_ID"] = self.__QS_restoreID__(RawData["QS_ID"])
            return RawData.sort_values(by=["QS_ID", "QS_DT"])
        else:
            return pd.DataFrame(columns=["QS_ID", "QS_DT"] + factor_names)

    def __QS_calcData__(self, raw_data, factor_names, ids, dts):
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Args = self._QSArgs.to_dict(repr=False)
        ErrorFmt = {"DuplicatedIndex": "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        return _QS_calcData_WideTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)


class _DTRangeTable(_BSTable):
    """BaoStockDB 库中基于取时间区间数据 API 的因子表"""

    class __QS_ArgClass__(_BSTable.__QS_ArgClass__):
        TableType: Literal["DTRangeTable"] = Field(default="DTRangeTable", title="因子表类型", frozen=True)
        LookBack: int = Field(default=0, title="回溯天数", frozen=True, ge=0)

    def __init__(self, fdb: "BaoStockDB", args:dict={}, **kwargs):
        super().__init__(fdb=fdb, args=args, **kwargs)
        self._QS_PrepareIgnoredArgs += ("LookBack",)

    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
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

    def __QS_calcData__(self, raw_data, factor_names, ids, dts):
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Args = self._QSArgs.to_dict(repr=False)
        ErrorFmt = {"DuplicatedIndex": "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        return _QS_calcData_WideTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)


class BaoStockDB(FactorDB):
    """基于 BaoStock 的因子库
    API: http://baostock.com/baostock/
    库配置信息文件在 QuantStudio 包目录下 Resource 目录下的 BaoStockDBInfo.xlsx, 记录了相关配置信息
    """

    class __QS_ArgClass__(FactorDB.__QS_ArgClass__):
        Name: str = Field(default="BaoStockDB", title="名称", frozen=True)
        UserID: str = Field(default="anonymous", title="用户ID", frozen=True, repr=False)
        Pwd: str = Field(default="123456", title="密码", frozen=True, repr=False, json_schema_extra={"secret": True})
        DBInfoFile: Optional[FilePath] = Field(default=None, title="库信息文件", frozen=True, repr=False)
        FTArgs: dict = Field(default={}, title="因子表参数", frozen=True, repr=False)

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 BaoStockDB

        Args:
            args: 指定的对象参数集
            config_file: 配置文件路径, 默认配置文件为 "~/QuantStudioConfig/BaoStockDBConfig.json"
        """
        if (not config_file) and os.path.isfile(__QS_ConfigPath__ + os.sep + "BaoStockDBConfig.json"):
            config_file = __QS_ConfigPath__ + os.sep + "BaoStockDBConfig.json"
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._InfoFilePath = __QS_MainPath__ + os.sep + "Resource" + os.sep + "BaoStockDBInfo.hdf5"  # 数据库信息文件路径
        if (not self._QSArgs.DBInfoFile) or (not os.path.isfile(self._QSArgs.DBInfoFile)):
            if self._QSArgs.DBInfoFile: self._QS_Logger.warning("找不到指定的库信息文件 : '%s'" % self._QSArgs.DBInfoFile)
            self._InfoResourcePath = __QS_MainPath__ + os.sep + "Resource" + os.sep + "BaoStockDBInfo.xlsx"  # 默认数据库信息源文件路径
            self._TableInfo, self._FactorInfo, self._ArgInfo = _updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger)  # 数据库表信息, 数据库字段信息
        else:
            self._InfoResourcePath = self._QSArgs.DBInfoFile
            self._TableInfo, self._FactorInfo, self._ArgInfo = _updateInfo(self._InfoFilePath, self._InfoResourcePath, self._QS_Logger, out_info=True)  # 数据库表信息, 数据库字段信息

    @property
    def TableNames(self) -> List[str]:
        if self._TableInfo is not None:
            return self._TableInfo.index.tolist()
        else:
            return []

    def connect(self):
        LG = bs.login(user_id=self._QSArgs.UserID, password=self._QSArgs.Pwd)
        if LG.error_code != "0":
            raise __QS_Error__(f"BaoStockDB.connect 登录错误码: {LG.error_code}, 错误信息: {LG.error_msg}")
        return self

    def disconnect(self) -> int:
        LG = bs.logout(user_id=self._QSArgs.UserID)
        if LG.error_code != "0":
            self._QS_Logger.error(f"BaoStockDB.disconnect 登出错误码: {LG.error_code}, 错误信息: {LG.error_msg}")
        return 0

    def getTable(self, table_name:str, args:dict={}) -> _BSTable:
        if table_name in self._TableInfo.index:
            TableClass = args.get("TableType", self._TableInfo.loc[table_name, "TableClass"])
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

    def getTradeDay(self, start_date:Optional[dt.datetime]=None, end_date:Optional[dt.datetime]=None, exchange:Literal["SSE"]="SSE", **kwargs) -> List[dt.datetime]:
        """给定交易所、起始日和结束日, 获取交易日序列

        Args:
            start_date: 起始日, None 表示从可取的最早日期开始
            end_date: 结束日, None 表示当前日期
            exchange: 交易所, 默认 SSE(上交所)

        Returns:
            交易日序列
        """
        if exchange != "SSE":
            self._QS_Logger.warning(f"BaoStockDB.getTradeDay 的参数 exchange 暂不支持除了 'SSE' 外的其他选项: '{exchange}', 该参数将被忽略!")
        self.connect()
        rs = bs.query_trade_dates(start_date=(None if start_date is None else start_date.strftime("%Y-%m-%d")), end_date=(None if end_date is None else end_date.strftime("%Y-%m-%d")))
        if rs.error_code != "0":
            raise __QS_Error__(f"BaoStockDB.getTradeDay query_trade_dates 错误码: {rs.error_code}, 错误信息: {rs.error_msg}")
        Rslt = rs.get_data()
        return Rslt["calendar_date"][Rslt["is_trading_day"] == "1"].apply(lambda d: dt.datetime.strptime(d, "%Y-%m-%d")).tolist()

    # 获取指定日 date 的全体 A 股 ID
    # date: 指定日, datetime.date
    # is_current: False 表示上市日在指定日之前的股票, True 表示上市日在指定日之前且尚未退市的股票
    def _getAllAStock(self, date, is_current=True, exchange=("SSE", "SZSE")):
        ExchgSuffix = {"SSE": "SH", "SZSE": "SZ"}
        ExchgSuffix = {ExchgSuffix[Exchg] for Exchg in exchange}
        DTs = self.getTradeDay(start_date=date-dt.timedelta(30), end_date=date)
        for iDT in reversed(DTs):
            rs = bs.query_all_stock(day=iDT.strftime("%Y-%m-%d"))# 当参数“day”为空时，默认取当天日期。闭市后日K线数据更新，该接口才会返回当天数据，否则返回空。
            if rs.error_code != "0":
                raise __QS_Error__(f"BaoStockDB.getTradeDay query_trade_dates 错误码: {rs.error_code}, 错误信息: {rs.error_msg}")
            Rslt = rs.get_data()
            if not Rslt.empty:
                break
        else:
            Rslt = pd.DataFrame(columns=["code", "tradeStatus"])
        if is_current:
            Rslt = Rslt[Rslt["tradeStatus"]=="1"]
        Rslt = Rslt["code"].str.split(".", expand=True)
        Rslt[0] = Rslt[0].str.upper()
        Initial = Rslt[1].str.slice(0, 1)
        Rslt = Rslt[((Rslt[0]=="SH") & (Initial=="6")) | ((Rslt[0]=="SZ") & (Initial.isin(("0", "3"))))]
        return sorted(Rslt[1] + "." + Rslt[0])

    def getStockID(self, exchange:Optional[Union[str, Tuple[str]]]=("SSE", "SZSE"), date:Optional[dt.datetime]=None, is_current:bool=True, **kwargs) -> List[str]:
        """给定交易所和日期, 获取股票证券 ID 序列

        Args:
            exchange: 交易所(str)或者交易所列表(tuple), 默认 ("SSE", "SZSE") 表示上交所、深交所
            date: 指定日, 默认值 None 表示当前日期
            is_current: False 表示上市日期在指定日之前的股票, True 表示上市日期在指定日之前且尚未退市的股票
            
        Returns:
            股票证券 ID 序列
        """
        if date is None: date = dt.date.today()
        if isinstance(exchange, str):
            exchange = {exchange}
        else:
            exchange = set(exchange)
        IDs = []
        # A 股
        iExchange = {"SSE", "SZSE"}
        if not exchange.isdisjoint(iExchange):
            IDs += self._getAllAStock(exchange=exchange.intersection(iExchange), date=date, is_current=is_current)
            exchange = exchange.difference(iExchange)
        if exchange:
            Msg = f"外部因子库 '{self._QSArgs.Name}' 调用 getStockID 时错误: 尚不支持交易所 {str(exchange)}"
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        return IDs
    

if __name__ == "__main__":
    BSDB = BaoStockDB().connect()
    print(BSDB.TableNames)

    #DTs = BSDB.getTradeDay(start_date=dt.datetime(2022, 1, 1), end_date=dt.datetime(2022, 1, 31))
    #print(DTs)
    
    #IDs = BSDB.getStockID()
    #print(IDs)
    
    ## 测试 QSID
    #FT1 = BSDB.getTable("A股K线数据", args={"APIArgs": {"frequency": "d"}, "LookBack": 1})
    #print(FT1.model_dump())
    #print(FT1.QSID)
    #print(FT1.PrepareID)
    #FT2 = BSDB.getTable("A股K线数据", args={"LookBack": 0})
    #print(FT2.model_dump())
    #print(FT2.QSID)
    #print(FT2.PrepareID)
    
    #FT = BSDB.getTable("A股K线数据", args={})
    #Data = FT.readData(
        #factor_names=["open", "close"],
        #ids=["600000.SH"],
        #dts=[dt.datetime(2022, 10, 28), dt.datetime(2022, 10, 31)]
    #)
    #print(Data)
    
    FT = BSDB.getTable("行业分类", args={})
    Data = FT.readData(
        factor_names=["industry"],
        ids=["600000.SH"],
        dts=[dt.datetime(2022, 10, 28), dt.datetime(2022, 10, 31)]
    )
    print(Data.iloc[0])
    
    print("===")