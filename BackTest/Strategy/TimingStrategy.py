# -*- coding: utf-8 -*-
"""择时交易型策略"""
import pandas as pd
import numpy as np
from traits.api import Enum, List, Int, Instance, on_trait_change

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.BackTest.Strategy.StrategyModule import Strategy, Account

# 信号数据格式: float, -inf~inf 的仓位水平或者 None(表示无信号, 默认值)

# 择时策略
class TimingStrategy(Strategy):
    SigalDelay = Int(0, label="信号滞后期", arg_type="Integer", order=0)
    SigalValidity = Int(1, label="信号有效期", arg_type="Integer", order=1)
    SigalDTs = List(label="信号触发时点", arg_type="DateTimeList", order=2)
    SignalInterval = Enum("不发信号", "延续上期", label="信号间期", arg_type="ArgObject", order=3)
    TargetAccount = Instance(Account, label="目标账户", arg_type="ArgObject", order=4)
    TargetID = Enum(None, label="目标ID", arg_type="SingleOption", order=5)
    TradeTarget = Enum("锁定买卖金额", "锁定目标仓位", "锁定目标金额", label="交易目标", arg_type="SingleOption", order=6)
    def __init__(self, name, factor_table=None, sys_args={}, config_file=None, **kwargs):
        self._FT = factor_table# 因子表
        self._AllSignals = {}# 存储所有生成的信号, {时点:信号}
        return super().__init__(name, sys_args=sys_args, config_file=config_file, **kwargs)
    @on_trait_change("TargetAccount")
    def on_TargetAccount_changed(self, obj, name, old, new):
        if self.TargetAccount is not None:
            self.add_trait("目标ID", Enum(*self.TargetAccount.IDs, label="目标ID", arg_type="SingleOption", order=5))
            if self.TargetAccount not in self.Accounts: self.Accounts.append(self.TargetAccount)
        else:
            self.add_trait("目标ID", Enum(None, label="目标ID", arg_type="SingleOption", order=5))
            self.Accounts.remove(old)
    @property
    def MainFactorTable(self):
        return self._FT
    def __QS_start__(self, mdl, dts, **kwargs):
        self._AllSignals = {}
        self._TradeTarget = None# 锁定的交易目标
        self._SignalExcutePeriod = 0# 信号已经执行的期数
        # 初始化信号滞后发生的控制变量
        self._TempData = {}
        self._TempData['StoredSignal'] = []# 暂存的信号，用于滞后发出信号
        self._TempData['LagNum'] = []# 当前日距离信号生成日的日期数
        self._TempData['LastSignal'] = None# 上次生成的信号
        return (self._FT, )+super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
    def __QS_move__(self, idt, **kwargs):
        iTradingRecord = {iAccount.Name:iAccount.__QS_move__(idt, **kwargs) for iAccount in self.Accounts}
        if (not self.SigalDTs) or (idt in self.SigalDTs):
            Signal = self.genSignal(idt, iTradingRecord)
        else:
            Signal = None
        self.trade(idt, iTradingRecord, Signal)
        for iAccount in self.Accounts: iAccount.__QS_after_move__(idt, **kwargs)
        return 0
    def genSignal(self, idt, trading_record):
        return None
    def trade(self, idt, trading_record, signal):
        if signal is not None: self._AllSignals[idt] = signal
        signal = self._bufferSignal(signal)
        AccountValue = self.TargetAccount.AccountValue
        PositionAmount = self.TargetAccount.PositionAmount.get(self.TargetID, 0.0)
        if signal is not None:# 有新的信号, 形成新的交易目标
            if self.TradeTarget=="锁定买卖金额":
                self._TradeTarget = AccountValue*signal - PositionAmount
            elif self.TradeTarget=="锁定目标金额":
                self._TradeTarget = AccountValue*signal
            elif self.TradeTarget=="锁定目标仓位":
                self._TradeTarget = signal
            self._SignalExcutePeriod = 0
        elif self._TradeTarget is not None:# 没有新的信号, 根据交易记录调整交易目标
            self._SignalExcutePeriod += 1
            if self._SignalExcutePeriod>=self.SigalValidity:
                self._TradeTarget = None
                self._SignalExcutePeriod = 0
            else:
                iTradingRecord = trading_record[self.TargetAccount]
                iTradingRecord = iTradingRecord.set_index(["ID"]).loc[self.TargetID]
                if self.TradeTarget=="锁定买卖金额":
                    TargetChanged = iTradingRecord["数量"] * iTradingRecord["价格"]
                    TargetChanged[pd.isnull(TargetChanged)] = 0.0
                    self._TradeTarget = self._TradeTarget - TargetChanged
        # 根据交易目标下订单
        if self._TradeTarget is not None:
            LastPrice = self.TargetAccount.LastPrice.loc[self.TargetID]
            if self.TradeTarget=="锁定买卖金额":
                self.TargetAccount.order(target_id=self.TargetID, num=self._TradeTarget/LastPrice)
            elif self.TradeTarget=="锁定目标仓位":
                self.TargetAccount.order(target_id=self.TargetID, num=(self._TradeTarget*AccountValue-PositionAmount)/LastPrice)
            elif self.TradeTarget=="锁定目标金额":
                self.TargetAccount.order(target_id=self.TargetID, num=(self._TradeTarget-PositionAmount)/LastPrice)
        return 0
    # 将信号缓存，并弹出滞后期到期的信号
    def _bufferSignal(self, signal):
        if signal is not None:
            self._TempData['StoredSignal'].append(signal)
            self._TempData['LagNum'].append(-1)
        for i, iLagNum in enumerate(self._TempData['LagNum']):
            self._TempData['LagNum'][i] = iLagNum+1
        signal = None
        while self._TempData['StoredSignal']!=[]:
            if self._TempData['LagNum'][0]>=self.SigalDelay:
                signal = self._TempData['StoredSignal'].pop(0)
                self._TempData['LagNum'].pop(0)
            else:
                break
        return signal