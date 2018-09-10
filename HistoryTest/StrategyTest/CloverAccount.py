# coding=utf-8
import datetime as dt

import pandas as pd
import numpy as np

from QuantStudio import __QS_Error__
from QuantStudio.HistoryTest.StrategyTest.StrategyTestModule import Account, cutDateTime

class TimeBarAccount(Account):
    """Time Bar 账户"""
    def __init__(self, name, sys_args={}, **kwargs):
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
        self._EquityRecord = pd.DataFrame(columns=["时间点", "ID", "进出数量", "进出金额", "备注"])# 证券进出流水记录, 提取为负, 增加为正
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])# 当前接收到的订单
        self._LastPrice = np.array([])# 最新价, array(len(self._IDs))
        self._DB = None# CloverDB 对象
        self._Simulator = None# Clover 模拟器
        self._SecurityIDs = None# 证券ID, DataFrame(证券ID, index=[日期], columns=[ID])
        self._PriceMultiplier = None# 价格乘数, DataFrame(价格乘数, index=[日期], columns=[ID])
        self._TimeBars = None# Time Bar 对象
        self._CurDate = None# 当前的日期
        return
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(None, **kwargs)
        if args is None:
            nSysArgs = len(SysArgs)
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"目标ID":[], "初始资金":0, "负债上限":np.inf})
            SysArgs.ArgInfo["目标ID"] = {"type":"MultiOption","range":[],"order":nSysArgs}
            SysArgs._QS_MonitorChange = True
            return SysArgs
        return args
    def __QS_start__(self, dts=None, dates=None, times=None):
        super().__QS_start__(dts=dts, dates=dates, times=times)
        self._IDs = self.SysArgs["目标ID"]
        nDT, nID = dts.shape[0], len(self._IDs)
        #self._Cash = np.zeros(nDT)
        #self._Debt = np.zeros(nDT)
        self._Position = np.zeros((nDT+1, nID))
        self._PositionAmount = np.zeros((nDT+1, nID))
        self._iTradingRecord = []# 暂存的交易记录
        self._EquityRecord = pd.DataFrame(columns=["时间点", "ID", "进出数量", "进出金额", "备注"])
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])
        # 创建模拟器等对象
        self._DB = self.QSEnv.getDB("CloverDB")
        jpckg = self._DB.jpype.JPackage('clover.model.beta.lowfreq')
        self._Simulator = jpckg.TimeBarSimulator.createSimulator()
        tms_inv = int(np.diff(dts).min().total_seconds()*1000)
        self._TimeBars, AllSecurityIDs, self._SecurityIDs, self._PriceMultiplier = self._DB.getTimeBar(self._IDs, dts[0], dts[-1], tms_inv, 5)
        self._Simulator.registerSecurity(AllSecurityIDs)
        self._LastPrice = self._DB.fetchTimeBarData(["mid"], self._TimeBars, AllSecurityIDs, self._SecurityIDs, self._PriceMultiplier).iloc[0]
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
        self._Simulator.update(self._TimeBars[self._LastPrice.index.searchsorted(idt)])
        # 撮合成交
        TradingRecord = self._matchOrder(idt)
        TradingRecord.extend(self._iTradingRecord)
        TradingRecord = pd.DataFrame(TradingRecord, columns=self._TradingRecord.columns, 
                                     index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)))
        self._TradingRecord = self._TradingRecord.append(TradingRecord)
        # 更新当前的账户信息
        self._PositionAmount[iIndex+1] = self._Position[iIndex+1]*self.LastPrice.values
        return TradingRecord
    def __QS_after_move__(self, idt, *args, **kwargs):
        super().__QS_after_move__(self, idt, *args, **kwargs)
        self._iTradingRecord = self._matchOrder(idt)
        if self._iTradingRecord:
            iIndex = self.QSEnv.STM.DateTimeIndex
            self._PositionAmount[iIndex+1] = self._Position[iIndex+1]*self.LastPrice.values
        return 0
    def __QS_end__(self):
        self._DB = None# CloverDB 对象
        self._Simulator = None# Clover 模拟器
        self._TimeBars = None# Time Bar 对象
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
    # 执行给定数量的证券委托单, target_id: 目标证券 ID, num: 待买卖的数量, target_price: nan 表示市价单
    # combined_order: 组合订单, DataFrame(index=[ID],columns=[数量, 目标价])
    # 基本的下单函数, 必须实现, 目前只实现了市价单
    def order(self, target_id=None, num=0, target_price=np.nan, combined_order=None):
        iDT = self.QSEnv.STM.DateTime
        if target_id is not None:
            if pd.isnull(target_price):
                if num>0:
                    target_price = 9999999
                else:
                    target_price = 1
            else:
                target_price = int(target_price/self._PriceMultiplier.loc[iDT.date(), target_id])
            if num!=0:
                OrderID = self._Simulator.reqNewOrderSingle(int(self._SecurityIDs.loc[iDT.date(), target_id]), target_price, int(num))
        if combined_order is not None:
            for i, iID in enumerate(combined_order.index):
                iNum = combined_order["数量"].iloc[i]
                iTargetPrice = combined_order["目标价"].iloc[i]
                self.order(target_id=iID, num=iNum, target_price=iTargetPrice, combined_order=None)
        return 0
    # 更新账户信息
    def _updateAccount(self, cash_changed, position):
        iIndex = self.QSEnv.STM.DateTimeIndex
        DebtDelta = max((- cash_changed - self.Cash, 0))
        self._Debt[iIndex+1] += DebtDelta
        if DebtDelta>0:
            self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self.QSEnv.STM.DateTime, DebtDelta)
        self._Cash[iIndex+1] -= min((- cash_changed, self.Cash))
        position[position.abs()<1e-6] = 0.0
        self._Position[iIndex+1] = position.values
        return 0
    # 撮合成交订单, 订单执行方式是先卖后买, 可用现金或股票不足时实际成交数量将少于目标数量
    def _matchOrder(self, idt):
        TradingRecord = []
        if self._Simulator.hasExecutionReport():
            iIndex = self.QSEnv.STM.DateTimeIndex
            CashChanged = 0.0
            Position = self.Position
            while self._Simulator.hasExecutionReport():
                iTradingReport = self._Simulator.nextExecutionReport()
                iDateTime = dt.datetime.fromtimestamp(iTradingReport.sendingTime/1000)
                iSecID = iTradingReport.secid
                iSecurityIDs = self._SecurityIDs.loc[iDateTime.date()]
                iID = iSecurityIDs[iSecurityIDs==iSecID].index[0]
                iNum = iTradingReport.lastsz
                iPrice = iTradingReport.lastpx*self._PriceMultiplier.loc[iDateTime.date(), iID]
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
            self._updateAccount(CashChanged, Position)# 更新账户信息
        return TradingRecord