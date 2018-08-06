# -*- coding: utf-8 -*-
"""择时买卖型策略"""
import os
import datetime

import pandas as pd
import numpy as np

from QuantStudio.StrategyTest.StrategyTestFun import loadCSVFileTimingSignal
from QuantStudio import QSArgs, QSError, QSObject
from QuantStudio.StrategyTest.StrategyTestModel import Strategy

# 信号数据格式: -1~1 的仓位水平或者 None(表示无信号, 默认值)

# 自定义择时策略
class TimingStrategy(Strategy):
    def __init__(self, name, qs_env):
        super().__init__(name, qs_env)
        self.AllSignals = {}# 存储所有生成的信号, {时间戳: 信号}
        return
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(args, **kwargs)
        Accounts = list(self.QSEnv.STM.Accounts)
        if args is None:
            nSysArgs = len(SysArgs)
            AccountIDs = (Accounts[0].IDs if Accounts!=[] else [])
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"信号滞后期":0, 
                            "信号有效期":1,
                            "信号触发日":[],
                            "信号触发时":[],
                            "信号间期":"不发信号",
                            "目标账户":(Accounts[0] if Accounts!=[] else "无"),
                            "目标ID":(AccountIDs[0] if AccountIDs!=[] else "无"),
                            "交易目标":"锁定目标仓位"})
            SysArgs.ArgInfo.update({"信号滞后期":{"type":"Integer","order":nSysArgs,"min":0,"max":np.inf,"single_step":1},
                                    "信号有效期":{"type":"Integer","order":nSysArgs+1,"min":1,"max":np.inf,"single_step":1},
                                    "信号触发日":{'type':'DateList','order':nSysArgs+2},
                                    "信号触发时":{"type":"DateList","order":nSysArgs+3},
                                    "信号间期":{"type":"SingleOption","order":nSysArgs+4,"range":['不发信号','延续上期']},
                                    "目标账户":{"type":"SingleOption","order":nSysArgs+5,"range":["无"]+Accounts},
                                    "目标ID":{"type":"SingleOption","order":nSysArgs+6,"range":["无"]+AccountIDs},
                                    "交易目标":{"type":"SingleOption","order":nSysArgs+7,"range":["锁定买卖金额","锁定目标仓位","锁定目标金额"]}})
            SysArgs._QS_MonitorChange = True
            return SysArgs
        args._QS_MonitorChange = False
        if args["目标账户"] not in Accounts:
            args["目标账户"] = (Accounts[0] if Accounts!=[] else "无")
        args.ArgInfo["目标账户"]["range"] = ["无"]+Accounts
        AccountIDs = (self.QSEnv.STM.Accounts[args["目标账户"]].IDs if args["目标账户"]!="无" else [])
        if args["目标ID"] not in AccountIDs:
            args["目标ID"] = (AccountIDs[0] if AccountIDs!=[] else "无")
        args.ArgInfo["目标ID"]["range"] = ["无"]+AccountIDs
        args._QS_MonitorChange = True
        return args
    def __QS_start__(self):
        self.AllSignals = {}
        self._DS = self.QSEnv.DSs[self.SysArgs["数据源"]]
        self._Account = (self.QSEnv.STM.Accounts[self.SysArgs["目标账户"]] if self.SysArgs["目标账户"]!="无" else None)
        self._TargetID = self.SysArgs["目标ID"]
        self._TradeTarget = None# 锁定的交易目标
        self._SignalExcutePeriod = 0# 信号已经执行的期数
        self._TradeDates = (set(self._SysArgs["信号触发日"]) if self._SysArgs["信号触发日"]!=[] else None)
        self._TradeTimes = (set(self._SysArgs["信号触发时"]) if self._SysArgs["信号触发时"]!=[] else None)
        # 初始化信号滞后发生的控制变量
        self._TempData = {}
        self._TempData['StoredSignal'] = []# 暂存的信号，用于滞后发出信号
        self._TempData['LagNum'] = []# 当前日距离信号生成日的日期数
        self._TempData['LastSignal'] = None# 上次生成的信号
        return super().__QS_start__()
    def __QS_move__(self, idt, timestamp, trading_record, *args, **kwargs):
        if self._Account is not None:
            Signal = None
            CurTime = datetime.datetime.fromtimestamp(timestamp).time()
            if ((self._TradeDates is None) or (idt in self._TradeDates)) and ((self._TradeTimes is None) or (CurTime in self._TradeTimes)):
                Signal = self.genSignal(idt, timestamp, trading_record)
            self.trade(idt, timestamp, trading_record, Signal)
        return 0
    def __QS_end__(self):
        self._DS, self._Account, self._TradeDates, self._TradeTarget = None, None, None, None
        self._SignalExcutePeriod = 0
        self._TempData = {}
        return 0
    def genSignal(self, idt, timestamp, trading_record):
        return None
    def trade(self, idt, timestamp, trading_record, signal):
        if signal is not None:
            self.AllSignals[timestamp] = signal
        signal = self._bufferSignal(signal)
        if not self._Account.isShortAllowed:
            signal = max((0.0, signal))
        AccountValue = self._Account.AccountValue
        PositionAmount = self._Account.PositionAmount.get(self._TargetID, 0.0)
        if signal is not None:# 有新的信号, 形成新的交易目标
            if self.SysArgs["交易目标"]=="锁定买卖金额":
                self._TradeTarget = AccountValue*signal - PositionAmount
            elif self.SysArgs["交易目标"]=="锁定目标金额":
                self._TradeTarget = AccountValue*signal
            elif self.SysArgs["交易目标"]=="锁定目标仓位":
                self._TradeTarget = signal
            self._SignalExcutePeriod = 0
        elif self._TradeTarget is not None:# 没有新的信号, 根据交易记录调整交易目标
            self._SignalExcutePeriod += 1
            if self._SignalExcutePeriod>=self.SysArgs["信号有效期"]:
                self._TradeTarget = None
                self._SignalExcutePeriod = 0
            else:
                iTradingRecord = trading_record[self.SysArgs["目标账户"]]
                iTradingRecord = iTradingRecord.set_index(["ID"]).loc[self._TargetID]
                if self.SysArgs["交易目标"]=="锁定买卖金额":
                    TargetChanged = iTradingRecord["数量"]*iTradingRecord["价格"]
                    TargetChanged[pd.isnull(TargetChanged)] = 0.0
                    self._TradeTarget = self._TradeTarget - TargetChanged
        # 根据交易目标下订单
        if self._TradeTarget is not None:
            if self.SysArgs["交易目标"]=="锁定买卖金额":
                self._Account.orderAmount(target_id=self._TargetID, amount=self._TradeTarget)
            elif self.SysArgs["交易目标"]=="锁定目标仓位":
                self._Account.orderAmount(target_id=self._TargetID, amount=self._TradeTarget*AccountValue-PositionAmount)
            elif self.SysArgs["交易目标"]=="锁定目标金额":
                self._Account.orderAmount(target_id=self._TargetID, amount=self._TradeTarget-PositionAmount)
        return 0
    # 将信号缓存，并弹出滞后期到期的信号
    def _bufferSignal(self, signal):
        if signal is not None:
            self._TempData['StoredSignal'].append(signal)
            self._TempData['LagNum'].append(-1)
        for i,iLagNum in enumerate(self._TempData['LagNum']):
            self._TempData['LagNum'][i] = iLagNum+1
        signal = None
        while self._TempData['StoredSignal']!=[]:
            if self._TempData['LagNum'][0]>=self.SysArgs['信号滞后期']:
                signal = self._TempData['StoredSignal'].pop(0)
                self._TempData['LagNum'].pop(0)
            else:
                break
        return signal

# CSV文件择时策略, TODO: 关于时间戳文件的读写和判别
class CSVFileTimingStrategy(TimingStrategy):
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(args, **kwargs)
        if args is None:
            nSysArgs = len(SysArgs)
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"信号文件":""})
            SysArgs.ArgInfo.update({"信号文件":{"type":"Path","order":nSysArgs,"operation":"Open","filter":"Excel (*.csv)"}})
            SysArgs._QS_MonitorChange = True
            return SysArgs
        return super().__QS_genSysArgs__(args, **kwargs)
    def __QS_start__(self):
        Rslt = super().__QS_start__()
        # 加载信号文件
        with self.QSEnv.CacheLock:
            self.TempData['FileSignals'] = loadCSVFileTimingSignal(self.SysArgs['信号文件'])
        return Rslt
    def genSignal(self, idt, timestamp, trading_record):
        return self.TempData['FileSignals'].get(timestamp)