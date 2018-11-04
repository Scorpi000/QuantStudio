# coding=utf-8
"""Wind 插件(TODO)"""
import sys
import re
import os
import datetime as dt
import tempfile

import numpy as np
import pandas as pd
from traits.api import File, Str, Int, on_trait_change

from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.FileFun import readJSONFile, listDirFile
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5, writeNestedDict2HDF5
from QuantStudio.Tools.DateTimeFun import getDateSeries, getDateTimeSeries
from QuantStudio import __QS_Error__, __QS_MainPath__, __QS_LibPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable, _adjustDateTime

class _WSD(FactorTable):
    CMD = Str('w.wsd("000001.SZ", "close", "2016-1-1", "2016-1-1", "")', arg_type="String", label="Wind命令", order=0)
    MaxIDNum = Int(100, arg_type="Integer", label="单次读取最大ID数量", order=1)
    def __QS_initArgs__(self):
        self._FactorInfo = self._parseCode(self.CMD)
        return super().__QS_initArgs__()
    @on_trait_change("CMD")
    def _on_CMD_changed(self, obj, name, old, new):
        self._FactorInfo = self._parseCode(new)
    @property
    def FactorNames(self):
        return self._FactorInfo.index.tolist()
    def _parseCode(self, codes):
        FactorInfo = pd.Series()
        codes = codes.strip()
        codes = codes.replace("\r", "\n")
        codes = codes.replace(" ", "")
        codes = re.split("\n+", codes)
        for iWindCode in codes:
            iPos = iWindCode.find("(")
            iCmd = iWindCode[:iPos].split(".")[-1]
            iCmdArgs = iWindCode[iPos+1:-1]
            iCmdArgs = iCmdArgs[1:-1].split('","')
            iWindFactors = iCmdArgs[1].split(",")
            iArgs = iCmdArgs[4]
            for ijWindFactor in iWindFactors:
                if ijWindFactor in FactorInfo.index:
                    if iArgs!=FactorInfo.loc[ijWindFactor]: raise __QS_Error__("Wind 指标有重复!")
                    continue
                FactorInfo[ijWindFactor] = iArgs
        return FactorInfo
    def _getFactorData(self, ifactor_name, ids, dts, args, max_id_num):
        StartDate, EndDate = dts[0].strftime("%Y%m%d"), dts[-1].strftime("%Y%m%d")
        nID = len(ids)
        i = 0
        FactorData = None
        while i<nID:
            iData = self._FactorDB.w.wsd(ids[i:min(nID, i+max_id_num)], ifactor_name, beginTime=StartDate, endTime=EndDate, options=args)
            iData = pd.DataFrame(list(zip(*iData.Data)), index=iData.Times, columns=iData.Codes)
            i += max_id_num
            if FactorData is None: FactorData = iData
            else: FactorData = pd.merge(FactorData, iData, left_index=True, right_index=True)
        return FactorData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if "Wind命令" in args: FactorInfo = self._parseCode(args["Wind命令"])
        else: FactorInfo = self._FactorInfo
        Data = {}
        MaxIDNum = args.get("单次读取最大ID数量", self.MaxIDNum)
        for iFactorName in factor_names:
            Data[iFactorName] = self._getFactorData(iFactorName, ids=ids, dts=dts, args=FactorInfo[iFactorName], max_id_num=MaxIDNum)
        Data = pd.Panel(Data).loc[factor_names]
        Data.major_axis = [dt.datetime.combine(iDate, dt.time(23, 59, 59, 999999)) for iDate in Data.major_axis]
        return Data.loc[:, dts, :]

class _WSS(FactorTable):
    pass
class _WSET(FactorTable):
    pass
class _EDB(FactorTable):
    pass
class WindAddinDB(FactorDB):
    """Wind 插件"""
    NavigatorPath = File("", arg_type="File", label="代码生成器路径", order=0)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self.w = None
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"WindDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "WindAddinDB"
        return
    def startNavigator(self):
        if not os.path.isfile(self.NavigatorPath): raise __QS_Error__("命令导航器文件路径有误!")
        os.popen(self.NavigatorPath)
        return 1    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["w"] = (True if self.isDBAvailable() else False)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.w:
            self.w = None
            self.connect()
        else:
            self.w = None
    def _updateInfo(self):
        if not os.path.isfile(self._InfoResourcePath):
            print("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (self._InfoFilePath, self._InfoResourcePath))
        elif (os.path.getmtime(self._InfoResourcePath)>os.path.getmtime(self._InfoFilePath)):
            print("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % self._InfoResourcePath)
        else:
            try:
                self._TableInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/TableInfo")
                self._FactorInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/FactorInfo")
                return 0
            except:
                print("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (self._InfoFilePath, self._InfoResourcePath))
        if not os.path.isfile(self._InfoResourcePath): raise __QS_Error__("缺失数据库信息文件: %s" % self._InfoResourcePath)
        self.importInfo(self._InfoResourcePath)
        self._TableInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/TableInfo")
        self._FactorInfo = readNestedDictFromHDF5(self._InfoFilePath, ref="/FactorInfo")
        return 0
    def connect(self):
        if self.w is None:
            try:
                from WindPy import w
                self.w = w
            except:
                self.w = None
                raise __QS_Error__("没有安装 Wind 插件, 或者插件安装失败!")
        with tempfile.TemporaryFile() as LogFile:
            Stdout = sys.stdout
            sys.stdout = LogFile
            Data = self.w.start()
            sys.stdout = Stdout
        if Data.ErrorCode!=0:
            self.w = None
            raise __QS_Error__("Wind 插件启动故障: %s, 请自行修复!" % (Data.Data[0]))
        return 1
    def disconnect(self):
        try:
            self.w.close()
        except:
            print("Wind 插件关闭时出现异常!")
        return 1
    def isAvailable(self):
        if self.w is not None: return self.w.isconnected()
        return False
    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        return ["WSD", "WSS", "WSET", "EDB"]
    def getTable(self, table_name, args={}):
        return eval("_"+table_name+"(name='"+table_name+"', fdb=self, sys_args=args)")
    # -----------------------------------------数据提取---------------------------------
    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE"):
        Data = self.w.tdays(beginTime=start_date, endTime=end_date, options="TradingCalendar="+exchange)
        return Data.Times
    # 获取指定日当前在市或者历史上出现过的全体 A 股 ID
    def _getAllAStock(self, date, is_current=True):
        Rslt = self.w.wset("sectorconstituent", "date="+date.strftime("%Y%m%d")+";sectorid=a001010100000000;field=wind_code")
        if is_current: return Rslt.Data[0]
        isToday = (dt.date.today()==date)
        IDs = set(Rslt.Data[0])
        Rslt = self.w.wset("delistsecurity", "field=wind_code, sec_type")
        Rslt = np.array(Rslt.Data)
        IDs = sorted(IDs.union(set(Rslt[0, :][Rslt[1, :]=="A股"])))
        if isToday: return IDs
        Rslt = self.w.wss(IDs, "ipo_date")
        date = dt.datetime.combine(date, dt.time(20, 0, 0, 5000))
        return [IDs[i] for i, iDate in enumerate(Rslt.Data[0]) if iDate<=date]
    # 给定指数名称和ID，获取指定日当前或历史上的指数中的股票ID，is_current=True:获取指定日当天的ID，False:获取截止指定日历史上出现的ID
    def getID(self, index_id="全体A股", date=None, is_current=True):
        if date is None: date = dt.date.today()
        if index_id=="全体A股": return self._getAllAStock(date=date, is_current=is_current)
        Res = self.w.wset("sectorconstituent","date="+date.strftime("%Y%m%d")+";windcode="+index_id)
        IDs = [iID for iID in Res.Data[1]]
        return IDs
    # 获取报告期对应的披露日期
    def getIssuingDate(self, start_date, end_date, ids=[]):
        Data = self.w.wsd(ids, "stm_issuingdate", start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), "Period=Q;Days=Alldays")
        Dates = Data.Times
        Data = np.array(Data.Data, dtype="O")
        if Dates[-1].strftime("%m%d") not in ("0331", "0630", "0930", "1231"):
            Dates.pop(-1)
            Data = Data[:, :-1]
        return pd.DataFrame(Data.T, index=Dates, columns=ids, dtype="O")

if __name__=="__main__":
    WADB = WindAddinDB()
    WADB.connect()
    FT = WADB.getTable("WSD")
    print(FT.FactorNames)
    Data = FT.readData(factor_names=["close"], ids=["000001.SZ"], dts=[dt.datetime(2018, 10, 31, 23, 59, 59, 999999)])
    print("====")