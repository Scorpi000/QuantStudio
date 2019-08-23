# -*- coding: utf-8 -*-
"""择时型策略"""
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, List, ListStr, Int, Instance, on_trait_change, Str

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.BackTest.Strategy.StrategyModule import Strategy, Account

# 信号数据格式: Series(float, -inf~inf 的仓位水平或者 nan 表示维持目前仓位, index=[ID]) 或者 None(表示无信号, 默认值)

# 择时策略
class TimingStrategy(Strategy):
    SignalDelay = Int(0, label="信号滞后期", arg_type="Integer", order=1)
    SigalValidity = Int(1, label="信号有效期", arg_type="Integer", order=2)
    SigalDTs = List(label="信号触发时点", arg_type="DateTimeList", order=3)
    TargetAccount = Instance(Account, label="目标账户", arg_type="ArgObject", order=4)
    ValueAllocated = Instance(pd.Series, arg_type="Series", label="资金分配", order=5)
    TradeTarget = Enum("锁定买卖金额", "锁定目标仓位", "锁定目标金额", label="交易目标", arg_type="SingleOption", order=6)
    def __init__(self, name, factor_table=None, sys_args={}, config_file=None, **kwargs):
        self._FT = factor_table# 因子表
        self._AllAllocationReset = {}# 存储所有的资金分配信号, {时点: 信号}
        self._ValueAllocated = None
        self._CashAllocated = None
        return super().__init__(name=name, accounts=[], fts=([] if self._FT is None else [self._FT]), sys_args=sys_args, config_file=config_file, **kwargs)
    @on_trait_change("TargetAccount")
    def on_TargetAccount_changed(self, obj, name, old, new):
        if (self.TargetAccount is not None) and (self.TargetAccount not in self.Accounts): self.Accounts.append(self.TargetAccount)
        elif (self.TargetAccount is None) and (old in self.Accounts): self.Accounts.remove(old)
    @on_trait_change("ValueAllocated")
    def on_ValueAllocated_changed(self, obj, name, old, new):
        self._isAllocationReseted = True
    @property
    def MainFactorTable(self):
        return self._FT
    @property
    def TargetIDs(self):# 当前有分配资金的 ID 列表
        if self._ValueAllocated is None:
            if self.ValueAllocated is None:
                if self.TargetAccount is not None: return self.TargetAccount.IDs
                else: return []
            return self.ValueAllocated.index.tolist()
        else:
            return self._ValueAllocated[self._ValueAllocated!=0].index.tolist()
    @property
    def PositionLevel(self):# 当前目标账户中所有 ID 的仓位水平
        if self.TargetAccount is None: raise __QS_Error__("尚未设置目标账户!")
        if self._CashAllocated is None: return pd.Series(0.0, index=self.TargetAccount.IDs)
        PositionAmount = self.TargetAccount.PositionAmount
        PositionValue = PositionAmount + self._CashAllocated
        PositionLevel = PositionAmount / PositionValue
        PositionLevel[PositionValue==0] = 0.0
        Mask = ((PositionAmount!=0) & (PositionValue==0))
        PositionLevel[Mask] = np.sign(PositionAmount)[Mask]
        return PositionLevel
    # 重新设置资金分配
    def _resetAllocation(self, new_allocation):
        IDs = self.TargetAccount.IDs
        if new_allocation is None:
            return pd.Series(self.TargetAccount.AccountValue / len(IDs), index=IDs)
        elif new_allocation.index.intersection(IDs).shape[0]==0:
            return pd.Series(0.0, index=IDs)
        else:
            return new_allocation.loc[IDs].fillna(0.0)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        Rslt = super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self._TradeTarget = None# 锁定的交易目标
        self._SignalExcutePeriod = 0# 信号已经执行的期数
        self._ValueAllocated = self._resetAllocation(self.ValueAllocated)
        self._CashAllocated = self._ValueAllocated - self.TargetAccount.PositionAmount.fillna(0.0)
        self._AllAllocationReset = {dts[0]-dt.timedelta(1): self._ValueAllocated}
        self._isAllocationReseted = False
        # 初始化信号滞后发生的控制变量
        self._TempData = {}
        self._TempData['StoredSignal'] = []# 暂存的信号, 用于滞后发出信号
        self._TempData['LagNum'] = []# 当前时点距离信号触发时点的期数
        self._TempData['LastSignal'] = None# 上次生成的信号
        self._TempData['StoredAllocation'] = []# 暂存的资金分配信号, 用于滞后发出信号
        self._TempData['AllocationLagNum'] = []# 当前时点距离信号触发时点的期数
        self._isStarted = True
        return (self._FT, )+Rslt
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        TradingRecord = {iAccount.Name:iAccount.__QS_move__(idt, **kwargs) for iAccount in self.Accounts}
        if (not self.SigalDTs) or (idt in self.SigalDTs):
            Signal = self.genSignal(idt, TradingRecord)
            if Signal is not None: self._AllSignals[idt] = Signal
        else: Signal = None
        Signal = self._bufferSignal(Signal)
        NewAllocation = None
        if self._isAllocationReseted:
            NewAllocation = self._resetAllocation(self.ValueAllocated)
            self._AllAllocationReset[idt] = NewAllocation
            self._isAllocationReseted = False
        NewAllocation = self._bufferAllocationReset(NewAllocation)
        if NewAllocation is not None:
            self._ValueAllocated = NewAllocation
            self._CashAllocated = self._ValueAllocated - self.TargetAccount.PositionAmount.fillna(0.0)
        else:# 更新资金分配
            iTradingRecord = TradingRecord[self.TargetAccount.Name]
            if iTradingRecord.shape[0]>0:
                CashChanged = pd.Series((iTradingRecord["买卖数量"] * iTradingRecord["价格"] + iTradingRecord["交易费"]).values, index=iTradingRecord["ID"].values)
                CashChanged = CashChanged.groupby(axis=0, level=0).sum().loc[self._CashAllocated.index]
                self._CashAllocated -= CashChanged.fillna(0.0)
        self.trade(idt, TradingRecord, Signal)
        for iAccount in self.Accounts: iAccount.__QS_after_move__(idt, **kwargs)
        return 0
    def output(self, recalculate=False):
        Output = super().output(recalculate=recalculate)
        if recalculate:
            Output["Strategy"]["择时信号"] = pd.DataFrame(self._AllSignals).T
            Output["Strategy"]["资金分配"] = pd.DataFrame(self._AllAllocationReset).T
        return Output
    def genSignal(self, idt, trading_record):
        return None
    def trade(self, idt, trading_record, signal):
        PositionAmount = self.TargetAccount.PositionAmount
        PositionValue = PositionAmount + self._CashAllocated
        if signal is not None:# 有新的信号, 形成新的交易目标
            if signal.shape[0]>0:
                signal = signal.loc[PositionValue.index]
            else:
                signal = pd.Series(np.nan, index=PositionValue.index)
            signal[self._ValueAllocated==0] = 0.0
            if self.TradeTarget=="锁定买卖金额":
                self._TradeTarget = signal * PositionValue.abs() - PositionAmount
            elif self.TradeTarget=="锁定目标金额":
                self._TradeTarget = PositionValue.abs() * signal
            elif self.TradeTarget=="锁定目标仓位":
                self._TradeTarget = signal
            self._SignalExcutePeriod = 0
        elif self._TradeTarget is not None:# 没有新的信号, 根据交易记录调整交易目标
            self._SignalExcutePeriod += 1
            if self._SignalExcutePeriod>=self.SigalValidity:
                self._TradeTarget = None
                self._SignalExcutePeriod = 0
            else:
                iTradingRecord = trading_record[self.TargetAccount.Name]
                if iTradingRecord.shape[0]>0:
                    if self.TradeTarget=="锁定买卖金额":
                        TargetChanged = pd.Series((iTradingRecord["买卖数量"] * iTradingRecord["价格"]).values, index=iTradingRecord["ID"].values)
                        TargetChanged = TargetChanged.groupby(axis=0, level=0).sum().loc[self._TradeTarget.index]
                        TargetChanged.fillna(0.0, inplace=True)
                        TradeTarget = self._TradeTarget - TargetChanged
                        TradeTarget[np.sign(self._TradeTarget)*np.sign(TradeTarget)<0] = 0.0
                        self._TradeTarget = TradeTarget
        # 根据交易目标下订单
        if self._TradeTarget is not None:
            if self.TradeTarget=="锁定买卖金额":
                Orders = self._TradeTarget
            elif self.TradeTarget=="锁定目标仓位":
                Orders = self._TradeTarget * PositionValue.abs() - PositionAmount
            elif self.TradeTarget=="锁定目标金额":
                Orders = self._TradeTarget - PositionAmount
            Orders = Orders / self.TargetAccount.LastPrice
            Orders = Orders[pd.notnull(Orders) & (Orders!=0)]
            if Orders.shape[0]==0: return 0
            Orders = pd.DataFrame(Orders.values, index=Orders.index, columns=["数量"])
            Orders["目标价"] = np.nan
            self.TargetAccount.order(combined_order=Orders)
        return 0
    # 将信号缓存, 并弹出滞后期到期的信号
    def _bufferSignal(self, signal):
        if self.SignalDelay<=0: return signal
        if signal is not None:
            self._TempData['StoredSignal'].append(signal)
            self._TempData['LagNum'].append(-1)
        for i, iLagNum in enumerate(self._TempData['LagNum']):
            self._TempData['LagNum'][i] = iLagNum + 1
        signal = None
        while self._TempData['StoredSignal']!=[]:
            if self._TempData['LagNum'][0]>=self.SignalDelay:
                signal = self._TempData['StoredSignal'].pop(0)
                self._TempData['LagNum'].pop(0)
            else:
                break
        return signal
    # 将资金分配信号缓存, 并弹出滞后期到期的资金分配信号
    def _bufferAllocationReset(self, allocation):
        if self.SignalDelay<=0: return allocation
        if allocation is not None:
            self._TempData['StoredAllocation'].append(allocation)
            self._TempData['AllocationLagNum'].append(-1)
        for i, iLagNum in enumerate(self._TempData['AllocationLagNum']):
            self._TempData['AllocationLagNum'][i] = iLagNum + 1
        allocation = None
        while self._TempData['StoredAllocation']!=[]:
            if self._TempData['AllocationLagNum'][0]>=self.SignalDelay:
                allocation = self._TempData['StoredAllocation'].pop(0)
                self._TempData['AllocationLagNum'].pop(0)
            else:
                break
        return allocation