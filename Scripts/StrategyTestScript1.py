# coding=utf-8
import datetime as dt
import os
from copy import deepcopy

import pandas as pd
import numpy as np

from QuantStudio.FunLib.AuxiliaryFun import getFactorList,searchNameInStrList
from QuantStudio.FunLib.IDFun import testIDFilterStr
from QuantStudio.FunLib import DateTimeFun
from QuantStudio.FunLib.DataTypeFun import readNestedDictFromHDF5
from QuantStudio import QSError, QSArgs, QSObject
from QuantStudio.StrategyTest.StrategyTestModel import Account, StartedValidator, cutDateTime


def _getDefaultNontradableIDFilter(ft=None, nonbuyable=True, qs_env=None):
    if ft is None:
        return None
    if nonbuyable:
        DefaultNontradableIDFilter = readNestedDictFromHDF5(qs_env.SysArgs['LibPath']+os.sep+'IDFilter.hdf5', "/默认限买条件")
    else:
        DefaultNontradableIDFilter = readNestedDictFromHDF5(qs_env.SysArgs['LibPath']+os.sep+'IDFilter.hdf5', "/默认限卖条件")
    if DefaultNontradableIDFilter is not None:
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(DefaultNontradableIDFilter, ft.FactorNames)
        if CompiledIDFilterStr is not None:
            return DefaultNontradableIDFilter
    AllFactorNames = set(ft.FactorNames)
    if {'交易状态','涨跌停'}.issubset(AllFactorNames):
        if nonbuyable:
            DefaultNontradableIDFilter = "(@交易状态==0) | (@交易状态==-2) | pd.isnull(@交易状态) | (@涨跌停==1)"
        else:
            DefaultNontradableIDFilter = "(@交易状态==0) | (@交易状态==-2) | pd.isnull(@交易状态) | (@涨跌停==-1)"
    elif '交易状态' in AllFactorNames:
        DefaultNontradableIDFilter = "(@交易状态==0) | (@交易状态==-2) | pd.isnull(@交易状态)"
    elif {'是否在市','涨跌停'}.issubset(AllFactorNames):
        if nonbuyable:
            DefaultNontradableIDFilter = "(@是否在市!=1) | (@涨跌停==1)"
        else:
            DefaultNontradableIDFilter = "(@是否在市!=1) | (@涨跌停==-1)"
    elif '是否在市' in AllFactorNames:
        DefaultNontradableIDFilter = "(@是否在市!=1)"
    elif '涨跌停' in AllFactorNames:
        if nonbuyable:
            DefaultNontradableIDFilter = "(@涨跌停==1)"
        else:
            DefaultNontradableIDFilter = "(@涨跌停==-1)"
    else:
        DefaultNontradableIDFilter = None
    return DefaultNontradableIDFilter

class TimeBarAccount(Account):
    """Time Bar 账户"""
    def __init__(self, name, qs_env):
        super().__init__(name, qs_env)
        self.__QS_Type__ = "Time Bar 账户"
        # 继承自 Account 的属性
        #self._Cash = np.array([])# 剩余现金, 等于时间点长度+1, >=0
        #self._Debt = np.array([])# 负债, 等于时间点长度+1, >=0
        #self._CashRecord = pd.DataFrame(columns=["时间点", "现金流"])# 现金流记录, pd.DataFrame(columns=["日期", "时间点", "现金流"]), 现金流入为正, 现金流出为负
        #self._DebtRecord = pd.DataFrame(columns=["时间点", "融资"])# 融资记录, pd.DataFrame(columns=["日期", "时间点", "融资"]), 增加负债为正, 减少负债为负
        #self._TradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])# 交易记录
        self._IDs = []# 本账户支持交易的证券 ID, []
        self._Position = np.array([])# 仓位, array(index=dts+1, columns=self._IDs)
        self._PositionAmount = np.array([])# 持仓金额, array(index=dts+1, columns=self._IDs)
        self._EquityRecord = pd.DataFrame(columns=["时间点", "ID", "进出数量", "进出金额"])# 证券进出流水记录, 提取为负, 增加为正
        self._OrderQueue = []# 当前接收到的订单队列, [pd.DataFrame(columns=["ID", "数量", "目标价"])], 账户每个时点从队列头部取出订单处理, 并在尾部添加空订单
        
        self._DB = None# CloverDB 对象
        self._Simulator = None# Clover 模拟器
        self._TimeBars = None# Time Bar 对象
        self._SecIDs = pd.Series([])# 证券ID
        self._TimeBarMatrix = None# Time Bar 数据
        self._LastPrice = None# 最新价
        self._CurDate = None# 当前的日期
        return
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(None, **kwargs)
        if args is None:
            nSysArgs = len(SysArgs)
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"目标ID":[], "交易延迟":-1, "初始资金":0, "负债上限":np.inf})
            SysArgs.ArgInfo["目标ID"] = {"type":"MultiOption","range":[],"order":nSysArgs}
            SysArgs._QS_MonitorChange = True
            return SysArgs
        return args
    def __QS_start__(self, dts=None, dates=None, times=None):
        super().__QS_start__(dts=dts, dates=dates, times=times)
        self._IDs = self.SysArgs["目标ID"]
        nDT = dts.shape[0]
        nID = len(self._IDs)
        #self._Cash = np.zeros(nDT)
        #self._Debt = np.zeros(nDT)
        self._Position = np.zeros((nDT+1, nID))
        self._PositionAmount = np.zeros((nDT+1, nID))
        self._iTradingRecord = []# 暂存的交易记录
        # 创建模拟器等对象
        self._DB = self.QSEnv.getDB("CloverDB")
        jpckg = self._DB.jpype.JPackage('clover.model.beta.lowfreq')
        self._Simulator = jpckg.TimeBarSimulator.createSimulator()
        self._SecIDs = pd.Series(np.nan, index=self._IDs)
        tms_inv = int(np.diff(dts).min().total_seconds()*1000)
        self._TimeBars = self._DB.getTimeBar(self._IDs, dts[0], dts[-1], tms_inv, 5)
        self._TimeBarMatrix = jpckg.TimeBarUtil.toMatrix(self._TimeBars)
        TimeBarDateTimes = [dt.datetime.fromtimestamp(iTMS/1000) for iTMS in np.array(self._TimeBarMatrix.time)]
        self._LastPrice = pd.DataFrame((np.array(self._TimeBarMatrix.bid)+np.array(self._TimeBarMatrix.ask))/2, index=TimeBarDateTimes, columns=self._IDs)
        self._LastPrice[self._LastPrice==0] = np.nan
        self._LastPrice.fillna(method="pad", inplace=True)
        return 0
    def __QS_move__(self, idt, *args, **kwargs):
        super().__QS_move__(idt, *args, **kwargs)
        # 更新当前的账户信息
        iIndex = self.QSEnv.STM.DateTimeIndex
        self._Position[iIndex+1] = self._Position[iIndex]
        if self._CurDate!=idt.date():# 更新证券 ID
            self._CurDate = idt.date()
            SecIDs = pd.Series(self._DB.ID2SecurityID(self._IDs, idt=idt), index=self._IDs)
            NewSecIDs = [int(SecIDs[iID]) for iID in SecIDs.index if SecIDs[iID]!=self._SecIDs[iID]]
            self._Simulator.registerSecurity(NewSecIDs)
            self._SecIDs = SecIDs
        self._Simulator.update(self._TimeBars[self._LastPrice.index.searchsorted(idt)])
        # 撮合成交
        TradingRecord = self._matchOrder()
        TradingRecord.extend(self._iTradingRecord)
        TradingRecord = pd.DataFrame(TradingRecord, columns=self._TradingRecord.columns, 
                                     index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)))
        self._TradingRecord = self._TradingRecord.append(TradingRecord)
        # 更新当前的账户信息
        self._PositionAmount[iIndex+1] = self._Position[iIndex+1]*self.LastPrice.values
        return TradingRecord
    def __QS_after_move__(self, idt, *args, **kwargs):
        super().__QS_after_move__(self, idt, *args, **kwargs)
        self._iTradingRecord = self._matchOrder()
        if self._iTradingRecord:
            iIndex = self.QSEnv.STM.DateTimeIndex
            self._PositionAmount[iIndex+1] = self._Position[iIndex+1]*self.LastPrice.values
        return 0
    def __QS_end__(self):
        self._DB = None# CloverDB 对象
        self._Simulator = None# Clover 模拟器
        self._TimeBars = None# Time Bar 对象
        self._SecIDs = pd.Series([])# 证券ID
        self._LastPrice = None# 最新价
        self._Output = None# 缓存的输出结果
        return 0
    def output(self):
        if self._Output is not None:
            return self._Output
        self._Output = super().output()
        self._Output["证券进出记录"] = self._EquityRecord
        self._Output["持仓"] = self.getPositionSeries()
        self._Output["持仓金额"] = self.getPositionAmountSeries()
        return self._Output
    # 当前账户价值
    @property
    def AccountValue(self):
        return super().AccountValue + np.nansum(self._PositionAmount[self.QSEnv.STM.DateTimeIndex+1])
    # 当前账户的持仓
    @property
    def Position(self):
        return pd.Series(self._Position[self.QSEnv.STM.DateTimeIndex+1], index=self._IDs)
    # 当前账户的持仓金额
    @property
    def PositionAmount(self):
        return pd.Series(self._PositionAmount[self.QSEnv.STM.DateTimeIndex+1], index=self._IDs)
    # 当前账户的投资组合
    @property
    def Portfolio(self):
        return self.PositionAmount/self.AccountValue
    # 本账户支持交易的证券 ID
    @property
    def IDs(self):
        return np.array(self._IDs)
    # 当前最新价
    @property
    def LastPrice(self):
        iDateTime = self.QSEnv.STM.DateTime
        return self._LastPrice.iloc[self._LastPrice.index.searchsorted(iDateTime)]
    # 本账户是否可以卖空
    @property
    def isShortAllowed(self):
        return True
    # 获取持仓的历史序列, 以时间点为索引, 返回: pd.DataFrame(持仓, index=[时间点], columns=[ID])
    def getPositionSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.DataFrame(self._Position[1:self.QSEnv.STM.DateTimeIndex+2], index=self.QSEnv.STM.DateTimeSeries, columns=self._IDs)
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取持仓证券的金额历史序列, 以时间点为索引, 返回: pd.DataFrame(持仓金额, index=[时间点], columns=[ID])
    def getPositionAmountSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.DataFrame(self._PositionAmount[1:self.QSEnv.STM.DateTimeIndex+2], index=self.QSEnv.STM.DateTimeSeries, columns=self._IDs)
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 计算持仓投资组合历史序列, 以时间点为索引, 返回: pd.DataFrame(权重, index=[日期], columns=[ID])
    def getPortfolioSeries(self, dts=None, start_dt=None, end_dt=None):
        PositionAmount = self.getPositionAmountSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        AccountValue = self.getAccountValueSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        AccountValue[AccountValue==0] += 0.0001
        return (PositionAmount.T/AccountValue).T
    # 获取账户价值的历史序列, 以时间点为索引
    def getAccountValueSeries(self, dts=None, start_dt=None, end_dt=None):
        CashSeries = self.getCashSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        PositionAmountSeries = self.getPositionAmountSeries(dts=dts, start_dt=start_dt, end_dt=end_dt).sum(axis=1)
        DebtSeries = self.getDebtSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        return CashSeries + PositionAmountSeries - DebtSeries
    # 执行给定数量的证券委托单, target_id: 目标证券 ID, num: 待买卖的数量, target_price: nan 表示市价单, 
    # combined_order: 组合订单, DataFrame(index=[ID],columns=[数量, 目标价])
    # 基本的下单函数, 必须实现, 目前只实现了市价单
    def order(self, target_id=None, num=0, target_price=np.nan, combined_order=None):
        if target_id is not None:
            if num>0:
                OrderID = self._Simulator.reqNewOrderSingle(int(self._SecIDs[target_id]), 9999999, int(num))
            elif num<0:
                OrderID = self._Simulator.reqNewOrderSingle(int(self._SecIDs[target_id]), 1, int(num))
        if combined_order is not None:
            for iID in combined_order.index:
                iNum = int(combined_order.loc[iID, "数量"])
                if iNum>0:
                    OrderID = self._Simulator.reqNewOrderSingle(int(self._SecIDs[iID]), 9999999, iNum)
                elif iNum<0:
                    OrderID = self._Simulator.reqNewOrderSingle(int(self._SecIDs[iID]), 1, iNum)
        return 0
    # 撮合成交订单, 订单执行方式是先卖后买, 可用现金或股票不足时实际成交数量将少于目标数量
    def _matchOrder(self):
        TradingRecord = []
        if self._Simulator.hasExecutionReport():
            iIndex = self.QSEnv.STM.DateTimeIndex
            CurDateTime = self.QSEnv.STM.DateTime
            CashChanged = 0.0
            Position = self.Position
            while self._Simulator.hasExecutionReport():
                iTradingReport = self._Simulator.nextExecutionReport()
                iSecID = iTradingReport.secid
                iID = self._SecIDs[self._SecIDs==iSecID].index[0]
                iDateTime = dt.datetime.fromtimestamp(iTradingReport.sendingTime/1000)
                iNum = iTradingReport.lastsz
                iPrice = iTradingReport.lastpx
                iCashChanged = -iNum*iPrice
                iPosition = Position.get(iID, 0)
                if iPosition==0:
                    iType = "open"
                elif iPosition*iNum<0:
                    iType = "close"
                else:
                    iType = "open"
                TradingRecord.append((iDateTime, iID, iNum, iPrice, 0, iCashChanged, iType))
                Position[iID] = iPosition + iNum
                CashChanged += iCashChanged
            # 更新账户信息
            DebtDelta = max((- CashChanged - self.Cash, 0))
            self._Debt[iIndex+1] += DebtDelta
            if DebtDelta>0:
                self._DebtRecord.loc[self._DebtRecord.shape[0]] = (CurDateTime, DebtDelta)
            self._Cash[iIndex+1] -= min((- CashChanged, self.Cash))
            Position[Position.abs()<1e-6]=0
            self._Position[iIndex+1] = Position.values
        return TradingRecord

if __name__=="__main__":
    from QuantStudio.StrategyTest import Strategy
    
    # for each trading day:
    # - @ 10:00 a.m.:   buy 3 stocks that has smallest return in the morning
    #                   sell 3 stocks that has biggest return in the morning
    # - @ 14:55 p.m.:   flat all positions (suppose we can trade intraday)
    class DemoStrategy(Strategy):
        def init(self):
            pass
        def trade(self, idt, trading_record, signal):
            iHour = idt.hour
            iMinute = idt.minute
            if (iHour==9) and (iMinute==30):
                self.UserData["LastPrice"] = self.QSEnv.STM.Accounts["TimeBar账户"].LastPrice
            elif (iHour==10) and (iMinute==0):
                Price = self.QSEnv.STM.Accounts["TimeBar账户"].LastPrice
                Ret = np.log(Price) - np.log(self.UserData["LastPrice"])
                RetOrder = Ret.argsort()
                for iID in Ret.index[RetOrder.iloc[:3].values]:
                    self.QSEnv.STM.Accounts["TimeBar账户"].order(iID, 100)
                for iID in Ret.index[RetOrder.iloc[-3:].values]:
                    self.QSEnv.STM.Accounts["TimeBar账户"].order(iID, -100)
            elif (iHour==14) and (iMinute==56):
                Position = self.QSEnv.STM.Accounts["TimeBar账户"].Position
                Position = Position[Position!=0]
                for iID in Position.index:
                    self.QSEnv.STM.Accounts["TimeBar账户"].order(iID, -Position[iID])
    
    from QuantStudio import QSEnv
    from QuantStudio.FactorDataBase.CloverDB import CloverDB
    from QuantStudio.FunLib.DateTimeFun import getTimeSeries, combineDateTime
    from QuantStudio.GUI.QtGUIFun import showOutput
    QSE = QSEnv()
    
    # 创建因子数据库
    MainDB = CloverDB(QSE)
    MainDB.connect()
    QSE.addFactorDataBase(MainDB)
    IDs = ['600000.SH', '600016.SH', '600019.SH', '600028.SH', '600029.SH',
           '600030.SH', '600036.SH', '600048.SH', '600050.SH', '600104.SH',
           '600111.SH', '600340.SH', '600518.SH', '600519.SH', '600547.SH',
           '600606.SH', '600837.SH', '600887.SH', '600919.SH', '600958.SH',
           '600999.SH', '601006.SH', '601088.SH', '601166.SH', '601169.SH',
           '601186.SH', '601211.SH', '601229.SH', '601288.SH', '601318.SH',
           '601328.SH', '601336.SH', '601390.SH', '601398.SH', '601601.SH',
           '601628.SH', '601668.SH', '601669.SH', '601688.SH', '601766.SH',
           '601800.SH', '601818.SH', '601857.SH', '601878.SH', '601881.SH',
           '601985.SH', '601988.SH', '601989.SH', '603993.SH']
    
    # 测试时间序列
    TestDates = MainDB.getTradeDay(start_date=dt.date(2018,1,1), end_date=dt.date(2018,3,31))
    TestDateTimes = list(combineDateTime(TestDates, 
                                         np.append(getTimeSeries(dt.time(9,30), dt.time(11,30), dt.timedelta(minutes=1)), 
                                                   getTimeSeries(dt.time(13), dt.time(15), dt.timedelta(minutes=1)))))
    
    # 添加账户
    iAccount = TimeBarAccount("TimeBar账户",QSE)
    iAccount.SysArgs["目标ID"] = IDs
    QSE.STM.addAccount(iAccount)# 将账户添加入 QS 系统的策略测试模型
    
    
    # 构建策略
    QSE.STM.Strategy = DemoStrategy("主策略", QSE)# 将策略添加入 QS 系统的策略测试模型
    
    # 测试
    QSE.STM.run(test_dts=TestDateTimes)
    
    # 生成报告
    Rslt = QSE.STM.output()
    showOutput(QSE, Rslt)
    QSE.close()