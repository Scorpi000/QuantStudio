# coding=utf-8
"""基于天软的因子库(TODO)"""
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str, Range, Directory, Password

from QuantStudio import __QS_Error__, __QS_LibPath__, __QS_MainPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB, FactorTable

class TinySoftDB(FactorDB):
    """TinySoft"""
    InstallDir = Directory(label="安装目录", arg_type="Directory", order=0)
    IPAddr = Str("tsl.tinysoft.com.cn", arg_type="String", label="IP地址", order=1)
    Port = Range(low=0, high=65535, value=443, arg_type="Integer", label="端口", order=2)
    User = Str("Scorpio", arg_type="String", label="用户名", order=3)
    Pwd = Password("shuntai11", arg_type="String", label="密码", order=4)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_LibPath__+os.sep+"TinySoftDBConfig.json" if config_file is None else config_file), **kwargs)
        self.Name = "TinySoftDB"
        self.TSLPy = None
        #self._InfoFilePath = __QS_LibPath__+os.sep+"TinySoftDBInfo.hdf5"# 数据库信息文件路径
        #self._InfoResourcePath = __QS_MainPath__+os.sep+"Resource"+os.sep+"TinySoftDBInfo.xlsx"# 数据库信息源文件路径
        #self._updateInfo()
        return
    def __getstate__(self):
        state = self.__dict__.copy()
        state["TSLPy"] = (True if self.isAvailable() else False)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.TSLPy: self.connect()
        else: self.TSLPy = None
    def connect(self):
        if not (os.path.isdir(self.InstallDir)): raise __QS_Error__("TinySoft 的安装目录设置有误!")
        elif self.InstallDir not in sys.path: sys.path.append(self.InstallDir)
        import TSLPy3
        self.TSLPy = TSLPy3
        ErrorCode = self.TSLPy.ConnectServer(self.IPAddr, int(self.Port))
        if ErrorCode!=0:
            self.TSLPy = None
            raise __QS_Error__("TinySoft 服务器连接失败!")
        Rslt = self.TSLPy.LoginServer(self.User, self.Pwd)
        if Rslt is not None:
            ErrorCode, Msg = Rslt
            if ErrorCode!=0:
                self.TSLPy = None
                raise __QS_Error__("TinySoft 登录失败: "+Msg)
        else:
            raise __QS_Error__("TinySoft 登录失败!")
        return 0
    def disconnect(self):
        self.TSLPy.Disconnect()
        self.TSLPy = None
    def isAvailable(self):
        if self.TSLPy is not None:
            return self.TSLPy.Logined()
        else:
            return False
    # 给定起始日期和结束日期, 获取交易所交易日期
    def getTradeDay(self, start_date=None, end_date=None, exchange="SSE", **kwargs):
        if exchange not in ("SSE", "SZSE"): raise __QS_Error__("不支持交易所: '%s' 的交易日序列!" % exchange)
        if start_date is None: start_date = dt.date(1900, 1, 1)
        if end_date is None: end_date = dt.date.today()
        CodeStr = "SetSysParam(pn_cycle(), cy_day());return MarketTradeDayQk(inttodate({StartDate}), inttodate({EndDate}));"
        CodeStr = CodeStr.format(StartDate=start_date.strftime("%Y%m%d"), EndDate=end_date.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        return list(map(lambda x: dt.date(*self.TSLPy.DecodeDate(x)), Data))
    # 获取指定日当前或历史上的全体 A 股 ID，返回在市场上出现过的所有A股, 目前仅支持提取当前的所有 A 股
    def _getAllAStock(self, date=None, is_current=True):# TODO
        if date is None: date = dt.date.today()
        CodeStr = "return getBK('深证A股;中小企业板;创业板;上证A股');"
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        IDs = []
        for iID in Data:
            iID = iID.decode("gbk")
            IDs.append(iID[2:]+"."+iID[:2])
        return IDs
    # 给定指数名称和ID，获取指定日当前或历史上的指数中的股票ID，is_current=True:获取指定日当天的ID，False:获取截止指定日历史上出现的ID, 目前仅支持提取当前的指数成份股
    def getID(self, index_id, date=None, is_current=True):# TODO
        if index_id=="全体A股": return self._getAllAStock(date=date, is_current=is_current)
        if date is None: date = dt.date.today()
        CodeStr = "return GetBKByDate('{IndexID}',IntToDate({Date}));"
        CodeStr = CodeStr.format(IndexID="".join(reversed(index_id.split("."))), Date=date.strftime("%Y%m%d"))
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        IDs = []
        for iID in Data:
            iID = iID.decode("gbk")
            IDs.append(iID[2:]+"."+iID[:2])
        return IDs
    # 将 DataFrame values 和 columns 中的 bytes 类型解码成字符串
    def _decodeDataFrame(self, df):
        Cols = []
        DecodeFun = (lambda x: x.decode("gbk"))
        for i, iCol in enumerate(df.columns):
            if isinstance(iCol, bytes): Cols.append(iCol.decode("gbk"))
            else: Cols.append(iCol)
            if df.dtypes.iloc[i]==np.dtype("O"): df[iCol] = df[iCol].apply(DecodeFun)
        df.columns = Cols
        return df
    # 获取 Level 1数据
    def getLevel1Data(self, target_id, start_date, end_date, fields=None):
        CodeStr = "return select {Fields} from tradetable datekey inttodate({StartDate}) to (inttodate({EndDate})+0.9999) of '{ID}' end;"
        target_id = "".join(reversed(target_id.split(".")))
        if fields is None:
            fields = ["yclose","vol","amount","cjbs",
                      "buy1","buy2","buy3","buy4","buy5","bc1","bc2","bc3","bc4","bc5",
                      "sale1","sale2","sale3","sale4","sale5","sc1","sc2","sc3","sc4","sc5",
                      "zmm","buy_vol","buy_amount","sale_vol","sale_amount",
                      "w_buy","w_sale","wb","lb"]
        elif "date" in fields:
            fields.remove("date")
        CodeStr = CodeStr.format(StartDate=start_date.strftime("%Y%m%d"), EndDate=end_date.strftime("%Y%m%d"), ID=target_id, 
                                 Fields="['date'],"+("['"+("'],['".join(fields))+"']" if fields is not None else "*"))
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        Data = pd.DataFrame(Data)
        Data = self._decodeDataFrame(Data)
        Data["date"] = Data["date"].apply(lambda x : dt.datetime(*self.TSLPy.DecodeDateTime(x)))
        Data = Data.set_index(["date"])
        return Data.loc[:, fields]
    # 获取 Tick 数据
    def getTickData(self, target_id, start_date, end_date, fields=None):
        if fields is None:
            fields = ["yclose","vol","amount",
                      "cjbs","buy1","bc1","sale1","sc1",
                      "zmm","buy_vol","buy_amount","sale_vol","sale_amount",
                      "w_buy","w_sale","wb","lb"]
        return self.getLevel1Data(target_id, start_date, end_date, fields=fields)
    # 获取 Bar 数据, cycle: 周期, int 或者字符串 : day, week, month, quarter, halfyear, year 等
    # cycle_unit: 周期单位, 支持: s(秒), d(天)
    def getBarData(self, target_id, start_date, end_date, fields=None, cycle=60, cycle_unit="s"):
        if isinstance(cycle, str): CycleStr = "cy_"+cycle+"()"
        elif cycle_unit=="s": CycleStr = ("cy_trailingseconds(%d)" % cycle)
        elif cycle_unit=="d": CycleStr = ("cy_trailingdays(%d)" % cycle)
        if fields is None: fields = ["open", "high", "low", "price", "vol", "amount", "cjbs"]
        elif "date" in fields: fields.remove("date")
        target_id = "".join(reversed(target_id.split(".")))
        CodeStr = "SetSysParam(pn_cycle(),{Cycle});return select {Fields} from markettable datekey inttodate({StartDate}) to (inttodate({EndDate})+0.9999) of '{ID}' end;"
        CodeStr = CodeStr.format(StartDate=start_date.strftime("%Y%m%d"), EndDate=end_date.strftime("%Y%m%d"), ID=target_id, Cycle=CycleStr,
                                 Fields="['date'],"+("['"+("'],['".join(fields))+"']" if fields is not None else "*"))
        ErrorCode, Data, Msg = self.TSLPy.RemoteExecute(CodeStr,{})
        if ErrorCode!=0: raise __QS_Error__("TinySoft 执行错误: "+Msg.decode("gbk"))
        Data = pd.DataFrame(Data)
        Data = self._decodeDataFrame(Data)
        Data["date"] = Data["date"].apply(lambda x : dt.datetime(*self.TSLPy.DecodeDateTime(x)))
        Data = Data.set_index(["date"])
        return Data.loc[:, fields]

if __name__=="__main__":
    TSDB = TinySoft()
    TSDB.connect()
    
    # 功能测试
    Dates = TSDB.getTradeDay(start_date=dt.date(2018,1,1), end_date=dt.date(2018,2,1))
    #IDs1 = TSDB._getAllAStock()
    IDs2 = TSDB.getID("000300.SH", dt.datetime(2018,6,7))
    #Data1 = TSDB.getLevel1Data("000002.SZ", dt.datetime(2018,6,7), dt.datetime(2018,6,7))
    #Data2 = TSDB.getTickData("000002.SZ", dt.datetime(2018,6,7), dt.datetime(2018,6,7))
    #Data3 = TSDB.getBarData("000002.SZ", dt.datetime(2018,1,1), dt.datetime(2018,6,7), cycle="day")
    TSDB.disconnect()