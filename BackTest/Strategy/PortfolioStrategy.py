# -*- coding: utf-8 -*-
"""投资组合配置型策略"""
import os

import pandas as pd
import numpy as np
from traits.api import Enum, List, ListStr, Int, Float, Str, Bool, Dict, Instance, on_trait_change

from QuantStudio.Tools.AuxiliaryFun import searchNameInStrList, getFactorList, genAvailableName, match2Series
from QuantStudio.Tools.FileFun import listDirFile, writeFun2File
from QuantStudio.Tools.MathFun import CartesianProduct
from QuantStudio.Tools.StrategyTestFun import loadCSVFilePortfolioSignal, writePortfolioSignal2CSV
from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.BackTest.Strategy.StrategyModule import Strategy, Account

# 信号数据格式: (多头信号, 空头信号)
# 多头信号: Series(多头权重, index=[ID]) 或者 None(表示无信号, 默认值)
# 空头信号: Series(空头权重, index=[ID]) 或者 None(表示无信号, 默认值)

class _WeightAllocation(__QS_Object__):
    """权重分配"""
    ReAllocWeight = Bool(False, label="重配权重", arg_type="Bool", order=0)
    WeightFactor = Enum("等权", label="权重因子", arg_type="SingleOption", order=1)
    GroupFactors = List(label="分类因子", arg_type="MultiOption", order=2, option_range=())
    GroupWeight = Enum("等权", label="类别权重", arg_type="SingleOption", order=3)
    GroupMiss = Enum("忽略","全配", label="类别缺失", arg_type="SingleOption", order=4)
    WeightMiss = Enum("舍弃", "填充均值", label="权重缺失", arg_type="SingleOption", order=5)
    EmptyAlloc = Enum('清空持仓', '保持仓位', '市场组合', label="零股处理", arg_type="SingleOption", order=6)
    def __init__(self, ft=None, sys_args={}, config_file=None, **kwargs):
        self._FT = ft
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        if self._FT is not None:
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FT.getFactorMetaData(key="DataType")))
            self.add_trait("WeightFactor", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="权重因子", order=1))
            self.add_trait("GroupFactor", Enum(*self._FT.FactorNames, arg_type="MultiOption", label="类别因子", order=2, option_range=("等权", )+tuple(DefaultNumFactorList)))
            self.add_trait("GroupWeight", Enum(*(["等权"]+DefaultNumFactorList), arg_type="SingleOption", label="类别权重", order=3))

# 投资组合策略
class PortfolioStrategy(Strategy):
    SigalDelay = Int(0, label="信号滞后期", arg_type="Integer", order=0)
    SigalValidity = Int(1, label="信号有效期", arg_type="Integer", order=1)
    LongSigalDTs = List(label="多头信号时点", arg_type="DateTimeList", order=2)
    ShortSigalDTs = List(label="空头信号时点", arg_type="DateTimeList", order=3)
    LongWeightAlloction = Instance(_WeightAllocation, label="多头权重配置", arg_type="ArgObject", order=4)
    ShortWeightAlloction = Instance(_WeightAllocation, label="空头权重配置", arg_type="ArgObject", order=5)
    LongAccount = Instance(Account, label="多头账户", arg_type="ArgObject", order=6)
    ShortAccount = Instance(Account, label="空头账户", arg_type="ArgObject", order=7)
    TradeTarget = Enum("锁定买卖金额", "锁定目标权重", "锁定目标金额", label="交易目标", arg_type="SingleOption", order=8)
    def __init__(self, name, factor_table=None, sys_args={}, config_file=None, **kwargs):
        self._FT = factor_table# 因子表
        super().__init__(name, sys_args=sys_args, config_file=config_file, **kwargs)
        self._AllLongSignals = {}# 存储所有生成的多头信号, {时点:信号}
        self._AllShortSignals = {}# 存储所有生成的空头信号, {时点:信号}
        return
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.LongWeightAlloction = _WeightAllocation(ft=self._FT)
        self.ShortWeightAlloction = _WeightAllocation(ft=self._FT)
    @on_trait_change("LongAccount")
    def _on_LongAccount_changed(self, obj, name, old, new):
        if self.LongAccount and (self.LongAccount not in self.Accounts): self.Accounts.append(self.LongAccount)
        elif self.LongAccount is None: self.Accounts.remove(old)
    @on_trait_change("ShortAccount")
    def _on_ShortAccount_changed(self, obj, name, old, new):
        if self.ShortAccount and (self.ShortAccount not in self.Accounts): self.Accounts.append(self.ShortAccount)
        elif self.ShortAccount is None: self.Accounts.remove(old)
    @property
    def MainFactorTable(self):
        return self._FT
    def __QS_start__(self, mdl, dts, **kwargs):
        self._AllLongSignals = {}
        self._AllShortSignals = {}
        self._LongTradeTarget = None# 锁定的多头交易目标
        self._ShortTradeTarget = None# 锁定的空头交易目标
        self._LongSignalExcutePeriod = 0# 多头信号已经执行的期数
        self._ShortSignalExcutePeriod = 0# 空头信号已经执行的期数
        # 初始化信号滞后发生的控制变量
        self._TempData = {}
        self._TempData['StoredSignal'] = []# 暂存的信号，用于滞后发出信号
        self._TempData['LagNum'] = []# 当前日距离信号生成日的日期数
        return (self._FT, ) + super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
    def output(self, recalculate=False):
        Output = super().output(recalculate=recalculate)
        if recalculate:
            Output["Strategy"]["多头信号"] = pd.DataFrame(self._AllLongSignals).T
            Output["Strategy"]["空头信号"] = pd.DataFrame(self._AllShortSignals).T
        return Output
    # 生成多头信号, 用户实现
    def genLongSignal(self, idt, trading_record):
        return None
    # 生成空头信号, 用户实现
    def genShortSignal(self, idt, trading_record):
        return None
    def genSignal(self, idt, trading_record):
        LongSignal, ShortSignal = None, None
        if (self.LongAccount is not None) and ((not self.LongSigalDTs) or (idt in self.LongSigalDTs)):
            LongSignal = self.genLongSignal(idt, trading_record)
        if (self.ShortAccount is not None) and ((not self.ShortSigalDTs) or (idt in self.ShortSigalDTs)):
            ShortSignal = self.genShortSignal(idt, trading_record)
        return (LongSignal, ShortSignal)
    def trade(self, idt, trading_record, signal):
        LongSignal, ShortSignal = signal
        if LongSignal is not None:
            if self.LongWeightAlloction.ReAllocWeight:
                LongSignal = self._allocateWeight(idt, LongSignal.index.tolist(), self.LongAccount.IDs, self.LongWeightAlloction)
            self._AllLongSignals[idt] = LongSignal
        if ShortSignal is not None:
            if self.ShortWeightAlloction.ReAllocWeight:
                ShortSignal = self._allocateWeight(idt, ShortSignal.index.tolist(), self.ShortAccount.IDs, self.ShortWeightAlloction)
            self._AllShortSignals[idt] = ShortSignal
        LongSignal, ShortSignal = self._bufferSignal(LongSignal, ShortSignal)
        self._processLongSignal(idt, trading_record, LongSignal)
        self._processShortSignal(idt, trading_record, ShortSignal)
        return 0
    def _processLongSignal(self, idt, trading_record, long_signal):
        if self.LongAccount is None: return 0
        AccountValue = self.LongAccount.AccountValue
        pHolding = self.LongAccount.PositionAmount
        pHolding = pHolding[pHolding>0]
        pHolding = pHolding / AccountValue
        if long_signal is not None:# 有新的信号, 形成新的交易目标
            if self.TradeTarget=="锁定买卖金额":
                pSignalHolding, long_signal = match2Series(pHolding, long_signal, fillna=0.0)
                self._LongTradeTarget = (long_signal - pSignalHolding) * AccountValue
            elif self.TradeTarget=="锁定目标权重":
                self._LongTradeTarget = long_signal
            elif self.TradeTarget=="锁定目标金额":
                self._LongTradeTarget = long_signal*AccountValue
            self._LongSignalExcutePeriod = 0
        elif self._LongTradeTarget is not None:# 没有新的信号, 根据交易记录调整交易目标
            self._LongSignalExcutePeriod += 1
            if self._LongSignalExcutePeriod>=self.SigalValidity:
                self._LongTradeTarget = None
                self._LongSignalExcutePeriod = 0
            else:
                iTradingRecord = trading_record[self.LongAccount.Name].set_index(["ID"])
                if iTradingRecord.index.intersection(self._LongTradeTarget.index).shape[0]>0:
                    iTradingRecord = iTradingRecord.loc[self._LongTradeTarget.index]
                    if self.TradeTarget=="锁定买卖金额":
                        TargetChanged = iTradingRecord["数量"] * iTradingRecord["价格"]
                        TargetChanged[pd.isnull(TargetChanged)] = 0.0
                        self._LongTradeTarget = self._LongTradeTarget - TargetChanged
        # 根据交易目标下订单
        if self._LongTradeTarget is not None:
            if self.TradeTarget=="锁定买卖金额":
                Orders = self._LongTradeTarget
            elif self.TradeTarget=="锁定目标权重":
                pHolding, LongTradeTarget = match2Series(pHolding, self._LongTradeTarget, fillna=0.0)
                Orders = (LongTradeTarget - pHolding) * AccountValue
            elif self.TradeTarget=="锁定目标金额":
                pHolding, LongTradeTarget = match2Series(pHolding, self._LongTradeTarget, fillna=0.0)
                Orders = LongTradeTarget - pHolding*AccountValue
            LastPrice = self.LongAccount.LastPrice
            Orders = Orders / LastPrice[Orders.index]
            Orders = Orders[pd.notnull(Orders) & (Orders!=0)]
            if Orders.shape[0]==0: return 0
            Orders = pd.DataFrame(Orders, columns=["数量"])
            Orders["目标价"] = np.nan
            self.LongAccount.order(combined_order=Orders)
        return 0
    def _processShortSignal(self, idt, trading_record, short_signal):
        return 0# TODO
    # 配置权重
    def _allocateWeight(self, idt, ids, original_ids, args):
        nID = len(ids)
        if nID==0:# 零股处理
            if args.EmptyAlloc=='保持仓位': return None
            elif args.EmptyAlloc=='清空持仓': return pd.Series([])
            elif args.EmptyAlloc=='市场组合': ids=original_ids
        if not args.GroupFactors:# 没有类别因子
            if args.WeightFactor=='等权': NewSignal = pd.Series(1/nID, index=ids)
            else:
                WeightData = self._FT.readData(factor_names=[args.WeightFactor], dts=[idt], ids=ids).iloc[0,0,:]
                if args.WeightMiss=='舍弃': WeightData = WeightData[pd.notnull(WeightData)]
                else: WeightData[pd.notnull(WeightData)] = WeightData.mean()
                WeightData = WeightData / WeightData.sum()
                NewSignal = WeightData
        else:
            GroupData = self._FT.readData(factor_names=args.GroupFactors, dts=[idt], ids=original_ids).iloc[:,0,:]
            GroupData[pd.isnull(GroupData)] = np.nan
            AllGroups = [GroupData[iGroup].unique().tolist() for iGroup in args.GroupFactors]
            AllGroups = CartesianProduct(AllGroups)
            nGroup = len(AllGroups)
            if args.GroupWeight=='等权': GroupWeight = pd.Series(np.ones(nGroup)/nGroup, dtype='float')
            else:
                GroupWeight = pd.Series(index=np.arange(nGroup), dtype='float')
                GroupWeightData = self._FT.readData(factor_names=[args.GroupWeight], dts=[idt], ids=original_ids).iloc[0,0,:]
                for i, iGroup in enumerate(AllGroups):
                    if pd.notnull(iGroup[0]): iMask = (GroupData[args.GroupFactors[0]]==iGroup[0])
                    else: iMask = pd.isnull(GroupData[args.GroupFactors[0]])
                    for j, jSubGroup in enumerate(iGroup[1:]):
                        if pd.notnull(jSubGroup): iMask = (iMask & (GroupData[args.GroupFactors[j+1]]==jSubGroup))
                        else: iMask = (iMask & pd.isnull(GroupData[args.GroupFactors[j+1]]))
                    GroupWeight.iloc[i] = GroupWeightData[iMask].sum()
                GroupWeight[pd.isnull(GroupWeight)] = 0
                GroupTotalWeight = GroupWeight.sum()
                if GroupTotalWeight!=0: GroupWeight = GroupWeight/GroupTotalWeight
            if args.WeightFactor=='等权': WeightData = pd.Series(1.0, index=original_ids)
            else: WeightData = self._FT.readData(factor_names=[args.WeightFactor], dts=[idt], ids=original_ids).iloc[0,0,:]
            SelectedGroupData = GroupData.loc[ids]
            NewSignal = pd.Series()
            for i, iGroup in enumerate(AllGroups):
                if pd.notnull(iGroup[0]): iMask = (SelectedGroupData[args.GroupFactors[0]]==iGroup[0])
                else: iMask = pd.isnull(SelectedGroupData[args.GroupFactors[0]])
                for j, jSubGroup in enumerate(iGroup[1:]):
                    if pd.notnull(jSubGroup): iMask = (iMask & (SelectedGroupData[args.GroupFactors[j+1]]==jSubGroup))
                    else: iMask = (iMask & pd.isnull(SelectedGroupData[args.GroupFactors[j+1]]))
                iIDs = SelectedGroupData[iMask].index.tolist()
                if (iIDs==[]) and (args.GroupMiss=='全配'):
                    if pd.notnull(iGroup[0]): iMask = (GroupData[args.GroupFactors[0]]==iGroup[0])
                    else: iMask = pd.isnull(GroupData[args.GroupFactors[0]])
                    for k, kSubClass in enumerate(iGroup[1:]):
                        if pd.notnull(kSubClass): iMask = (iMask & (GroupData[args.GroupFactors[k+1]]==kSubClass))
                        else: iMask = (iMask & pd.isnull(GroupData[args.GroupFactors[k+1]]))
                    iIDs = GroupData[iMask].index.tolist()
                elif (iIDs==[]) and (args.GroupMiss=='忽略'): continue
                iSignal = WeightData.loc[iIDs]
                iSignalWeight = iSignal.sum()
                if iSignalWeight!=0: iSignal = iSignal / iSignalWeight * GroupWeight.iloc[i]
                else: iSignal = iSignal*0.0
                if args.WeightMiss=='填充均值': iSignal[pd.isnull(iSignal)] = iSignal.mean()
                NewSignal = NewSignal.append(iSignal[pd.notnull(iSignal) & (iSignal!=0)])
            NewSignal = NewSignal / NewSignal.sum()
        return NewSignal
    # 将信号缓存，并弹出滞后期到期的信号
    def _bufferSignal(self, long_signal, short_signal):
        if (long_signal is not None) or (short_signal is not None):
            self._TempData['StoredSignal'].append((long_signal, short_signal))
            self._TempData['LagNum'].append(-1)
        for i,iLagNum in enumerate(self._TempData['LagNum']):
            self._TempData['LagNum'][i] = iLagNum+1
        LongSignal, ShortSignal = None, None
        while self._TempData['StoredSignal']!=[]:
            if self._TempData['LagNum'][0]>=self.SigalDelay:
                LongSignal, ShortSignal = self._TempData['StoredSignal'].pop(0)
                self._TempData['LagNum'].pop(0)
            else:
                break
        return (LongSignal, ShortSignal)

# 分层筛选投资组合策略
class _TurnoverBuffer(__QS_Object__):
    """换手缓冲"""
    isBuffer = Bool(False, label="是否缓冲", arg_type="Bool", order=0)
    FilterUpBuffer = Float(0.1, label="筛选上限缓冲区", arg_type="Double", order=1)
    FilterDownBuffer = Float(0.1, label="筛选下限缓冲区", arg_type="Double", order=2)
    FilterNumBuffer = Int(0, label="筛选数目缓冲区", arg_type="Integer", order=3)

class _Filter(__QS_Object__):
    """筛选"""
    SignalType = Enum("多头信号", "空头信号", label="信号类型", arg_type="SingleOption", order=0)
    IDFilter = Str(arg_type="IDFilter", label="筛选条件", order=1)
    TargetFactor = Enum(None, label="目标因子", arg_type="SingleOption", order=2)
    FactorOrder = Enum("降序", "升序", label="排序方向", arg_type="SingleOption", order=3)
    FilterType = Enum("定量", "定比", "定量&定比", label="筛选方式", arg_type="SingleOption", order=4)
    FilterNum = Int(30, label="筛选数目", arg_type="Integer", order=5)
    GroupFactors = List(label="分类因子", arg_type="MultiOption", order=6, option_range=())
    TurnoverBuffer = Instance(_TurnoverBuffer, arg_type="ArgObject", label="换手缓冲", order=7)
    def __init__(self, ft=None, sys_args={}, config_file=None, **kwargs):
        self._FT = ft
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        self.TurnoverBuffer = _TurnoverBuffer()
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self._FT.getFactorMetaData(key="DataType")))
        self.add_trait("TargetFactor", Enum(*DefaultNumFactorList, label="目标因子", arg_type="SingleOption", order=2))
        self.add_trait("GroupFactors", List(arg_type="MultiOption", label="类别因子", order=6, option_range=tuple(self._FT.FactorNames)))
    @on_trait_change("FilterType")
    def _on_FilterType_changed(self, obj, name, old, new):
        if new=='定量':
            if "FilterUpLimit" in self.ArgNames:
                self.remove_trait("FilterUpLimit")
                self.remove_trait("FilterDownLimit")
            if "FilterNum" not in self.ArgNames:
                self.add_trait("FilterNum", Int(30, label="筛选数目", arg_type="Integer", order=5))
        elif new=='定比':
            if "FilterNum" in self.ArgNames:
                self.remove_trait("FilterNum")
            if "FilterUpLimit" not in self.ArgNames:
                self.add_trait("FilterUpLimit", Float(0.1, label="筛选上限", arg_type="Double", order=5.1))
                self.add_trait("FilterDownLimit", Float(0.0, label="筛选下限", arg_type="Double", order=5.2))
        else:
            if "FilterNum" not in self.ArgNames:
                self.add_trait("FilterNum", Int(30, label="筛选数目", arg_type="Integer", order=5))
            if "FilterUpLimit" not in self.ArgNames:
                self.add_trait("FilterUpLimit", Float(0.1, label="筛选上限", arg_type="Double", order=5.1))
                self.add_trait("FilterDownLimit", Float(0.0, label="筛选下限", arg_type="Double", order=5.2))
class HierarchicalFiltrationStrategy(PortfolioStrategy):
    FilterLevel = Int(1, label="筛选层数", arg_type="Integer", order=9)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        self.add_trait("Level0", Instance(_Filter, label="第0层", arg_type="ArgObject", order=10))
        self.Level0 = _Filter(ft=self._FT)
        self.LongWeightAlloction.ReAllocWeight = True
        self.ShortWeightAlloction.ReAllocWeight = True
    @on_trait_change("FilterLevel")
    def on_FilterLevel_changed(self, obj, name, old, new):
        ArgNames = self.ArgNames
        if new>old:# 增加了筛选层数
            for i in range(max(0, old), max(0, new)):
                self.add_trait("Level"+str(i), Instance(_Filter, label="第"+str(i)+"层", arg_type="ArgObject", order=10+i))
                setattr(self, "Level"+str(i), _Filter(ft=self._FT))
        elif new<old:# 减少了筛选层数
            for i in range(max(0, old)-1, max(0, new)-1, -1):
                self.remove_trait("Level"+str(i))
    def _filtrateID(self, idt, ids, args):
        FactorData = self._FT.readData(dts=[idt], ids=ids, factor_names=[args.TargetFactor]).iloc[0,0,:]
        FactorData = FactorData[pd.notnull(FactorData)]
        if args.FactorOrder=='降序': FactorData = -FactorData
        FactorData = FactorData.sort_values(ascending=True)
        if args.FilterType=='定比':
            UpLimit = FactorData.quantile(args.FilterUpLimit)
            DownLimit = FactorData.quantile(args.FilterDownLimit)
            NewIDs = FactorData[(FactorData>=DownLimit) & (FactorData<=UpLimit)].index.tolist()
        elif args.FilterType=='定量':
            NewIDs = FactorData.iloc[:args.FilterNum].index.tolist()
        elif args.FilterType=='定量&定比':
            UpLimit = FactorData.quantile(args.FilterUpLimit)
            DownLimit = FactorData.quantile(args.FilterDownLimit)
            NewIDs = FactorData.iloc[:args.FilterNum].index.intersection(FactorData[(FactorData>=DownLimit) & (FactorData<=UpLimit)].index).tolist()
        if not args.TurnoverBuffer.isBuffer: return NewIDs
        SignalIDs = set(NewIDs)
        nSignalID = len(SignalIDs)
        if args.SignalType=="多头信号":
            if self._AllLongSignals=={}: LastIDs = set()
            else: LastIDs = set(self._AllLongSignals[max(self._AllLongSignals)].index)
        else:
            if self._AllShortSignals=={}: LastIDs = set()
            else: LastIDs = set(self._AllShortSignals[max(self._AllShortSignals)].index)
        if args.FilterType=='定比':
            UpLimit = FactorData.quantile(min(1.0, args.FilterUpLimit+args.TurnoverBuffer.FilterUpBuffer))
            DownLimit = FactorData.quantile(max(0.0, args.FilterDownLimit-args.TurnoverBuffer.FilterDownBuffer))
            NewIDs = LastIDs.intersection(FactorData[(FactorData>=DownLimit) & (FactorData<=UpLimit)].index)
        elif args.FilterType=='定量':
            NewIDs = LastIDs.intersection(FactorData.iloc[:args.FilterNum+args.TurnoverBuffer.FilterNumBuffer].index)
        elif args.FilterType=='定量&定比':
            UpLimit = FactorData.quantile(min(1.0, args.FilterUpLimit+args.TurnoverBuffer.FilterUpBuffer))
            DownLimit = FactorData.quantile(max(0.0, args.FilterDownLimit-args.TurnoverBuffer.FilterDownBuffer))
            NewIDs = LastIDs.intersection(FactorData.iloc[:args.FilterNum+args.TurnoverBuffer.FilterNumBuffer].index).intersection(FactorData[(FactorData>=DownLimit) & (FactorData<=UpLimit)].index)
        if len(NewIDs)>=nSignalID:# 当前持有的股票已经满足要求
            FactorData = FactorData[list(NewIDs)].copy()
            FactorData.sort_values(inplace=True, ascending=True)
            return FactorData.iloc[:nSignalID].index.tolist()
        SignalIDs = list(SignalIDs.difference(NewIDs))
        FactorData = FactorData[SignalIDs].copy()
        FactorData.sort_values(inplace=True, ascending=True)
        return list(NewIDs)+FactorData.iloc[:(nSignalID-len(NewIDs))].index.tolist()
    def _genSignalIDs(self, idt, original_ids, signal_type):
        IDs = original_ids
        for i in range(self.FilterLevel):
            iArgs = self["第"+str(i)+"层"]
            if iArgs.SignalType!=signal_type: continue
            if iArgs.IDFilter:
                iIDs = self._FT.getFilteredID(idt, id_filter_str=iArgs.IDFilter)
                IDs = sorted(set(iIDs).intersection(set(IDs)))
            if iArgs.GroupFactors!=[]:
                GroupData = self._FT.readData(dts=[idt], ids=IDs, factor_names=iArgs.GroupFactors).iloc[:,0,:]
                if GroupData.shape[0]>0: GroupData[pd.isnull(GroupData)] = np.nan
                AllGroups = [GroupData[iGroup].unique().tolist() for iGroup in iArgs.GroupFactors]
                AllGroups = CartesianProduct(AllGroups)
                IDs = []
                for jGroup in AllGroups:
                    jMask = pd.Series(True, index=GroupData.index)
                    for k, kSubGroup in enumerate(jGroup):
                        if pd.notnull(kSubGroup): jMask = (jMask & (GroupData[iArgs.GroupFactors[k]]==kSubGroup))
                        else: jMask = (jMask & pd.isnull(GroupData[iArgs.GroupFactors[k]]))
                    jIDs = self._filtrateID(idt, GroupData[jMask].index.tolist(), iArgs)
                    IDs += jIDs
            else:
                IDs = self._filtrateID(idt, IDs, iArgs)
        return IDs
    def genLongSignal(self, idt, trading_record):
        if self.LongAccount is None: return None
        OriginalIDs = self.LongAccount.IDs
        IDs = self._genSignalIDs(idt, OriginalIDs, '多头信号')
        return pd.Series(1/len(IDs), index=IDs)
    def genShortSignal(self, idt, trading_record):
        if self.ShortAccount is None: return None
        OriginalIDs = self.ShortAccount.IDs
        IDs = self._genSignalIDs(idt, OriginalIDs, '空头信号')
        return pd.Series(1/len(IDs), index=IDs)

# 基于优化器的投资组合策略
class OptimizerStrategy(PortfolioStrategy):
    def __init__(self, name, pc, factor_table=None, sys_args={}, config_file=None, **kwargs):
        self._PC = pc
        self._SharedInfo = {}
        return super().__init__(name=name, factor_table=factor_table, sys_args=sys_args, config_file=config_file, **kwargs)
    # 生成约束条件参数信息以及初始值
    def _genConstraintArgInfo(self,arg=None):
        AllConstraintType = list(self.PC.ConstraintArgInfoFun.keys())
        AllConstraintType.sort()
        if arg is None:
            arg = {"约束类型":AllConstraintType[0]}
        InitConstraintArg,SubArgInfo = self.PC.ConstraintArgInfoFun[arg['约束类型']](None)
        if ("条件参数" not in arg) or (set(arg['条件参数'].keys())!=set(InitConstraintArg.keys())):
            arg['条件参数'] = InitConstraintArg
        ArgInfo = {"约束类型":{'数据类型':'Str','取值范围':AllConstraintType,'是否刷新':True,'序号':0,'是否可见':True,"是否可改":getattr(self.PC,"ConstraintAdjustive",True)},
                   "条件参数":{'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.ConstraintArgInfoFun[arg['约束类型']],None,{}],'是否刷新':False,'序号':1,'是否可见':True,'可否遍历':False}}
        return (arg,ArgInfo)
    # 生成信号后期调整参数集
    def _genSignalAdjustArgInfo(self,arg=None):
        if arg is None:
            arg = {"调整方式":"权重下限调整","权重下限幂次":-5,"权重累计阈值":0.97,"是否归一":False,"打印信息":False}
        ArgInfo = {"调整方式":{'数据类型':'Str','取值范围':["权重下限调整","权重累计调整"],'是否刷新':False,'序号':0,'是否可见':True},
                   "权重下限幂次":{'数据类型':'Int','取值范围':[-16,-1,1],'是否刷新':False,'序号':1,'是否可见':True},
                   "权重累计阈值":{'数据类型':'Double','取值范围':[0.0,1.0,0.00001],'是否刷新':False,'序号':2,'是否可见':True},
                   "是否归一":{'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':3,'是否可见':True},
                   "打印信息":{'数据类型':'Bool','取值范围':[True,False],'是否刷新':False,'序号':4,'是否可见':True}}
        return (arg,ArgInfo)
    # 生成系统参数信息集以及初始值
    def genSysArgInfo(self,arg=None):
        # arg=None 表示初始化参数
        DefaultNumFactorList,DefaultStrFactorList = getFactorList(self.StdDataSource.DataType)
        ArgInfo = {}
        if arg is None:# 初始化参数
            self.PC.AllFactorDataType = self.StdDataSource.DataType
            arg = {"信号滞后期":0,
                   "组合模型":"均值方差模型",
                   "构造器":self.PC.__doc__,
                   "目标ID":None,
                   "预期收益":DefaultNumFactorList[0],
                   "风险数据源":None,
                   '基准权重':searchNameInStrList(DefaultNumFactorList,['权重','Weight','weight']),
                   "成交额":searchNameInStrList(DefaultNumFactorList,['成交']),
                   "成交价":searchNameInStrList(DefaultNumFactorList,['价','Price','price']),
                   "约束个数":2,
                   "多头信号日":[],
                   "空头信号日":[]}
            arg['优化目标'],SubArgInfo = self.PC.genObjectArgInfo(None)
            ArgInfo['优化目标'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genObjectArgInfo,None,{}],'是否刷新':False,'序号':9}
            arg["0-预算约束"],iInfo = self._genConstraintArgInfo({"约束类型":"预算约束"})
            ArgInfo["0-预算约束"] = {'数据类型':'ArgSet','取值范围':[iInfo,self._genConstraintArgInfo,None,{}],'是否刷新':True,'序号':11,'可否遍历':False}
            arg["1-权重约束"],iInfo = self._genConstraintArgInfo({"约束类型":"权重约束"})
            ArgInfo["1-权重约束"] = {'数据类型':'ArgSet','取值范围':[iInfo,self._genConstraintArgInfo,None,{}],'是否刷新':True,'序号':12,'可否遍历':False}
            self._SharedInfo["约束条件"] = ['0-预算约束','1-权重约束']
            arg['构造器参数'],SubArgInfo = self.PC.genOptionArgInfo(None)
            ArgInfo['构造器参数'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genOptionArgInfo,None,{}],'是否刷新':False,'序号':13}
            arg['信号调整'],SubArgInfo = self._genSignalAdjustArgInfo(None)
            ArgInfo['信号调整'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self._genSignalAdjustArgInfo,None,{}],'是否刷新':False,'序号':14}
        else:
            ChangedKey = arg.pop("_ChangedKey_",None)
            if (self.PC.__doc__!=arg["构造器"]) or (arg["构造器"] not in self.ModelClass[arg['组合模型']]):# 调整组合模型或者更换了构造器
                NewArg = {"信号滞后期":arg["信号滞后期"],
                          "组合模型":arg['组合模型'],
                          "构造器":arg["构造器"],
                          "目标ID":arg["目标ID"],
                          "预期收益":arg['预期收益'],
                          "风险数据源":arg['风险数据源'],
                          '基准权重':arg['基准权重'],
                          "成交额":arg['成交额'],
                          "成交价":arg['成交价'],
                          "约束个数":0,
                          "空头信号日":arg['空头信号日'],
                          "多头信号日":arg['多头信号日']}
                AllPCClasses = list(self.ModelClass[NewArg['组合模型']].keys())
                if NewArg['构造器'] not in AllPCClasses:
                    NewArg['构造器'] = AllPCClasses[0]
                self.PC = self.ModelClass[NewArg['组合模型']][NewArg["构造器"]]("MainPC",qs_env=self.QSEnv)
                self.PC.AllFactorDataType = self.StdDataSource.DataType
                NewArg['优化目标'],SubArgInfo = self.PC.genObjectArgInfo(None)
                ArgInfo['优化目标'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genObjectArgInfo,None,{}],'是否刷新':False,'序号':9}            
                NewConstraintList = []
                for i in range(arg["约束个数"]):
                    iConstraintType = arg[self._SharedInfo['约束条件'][i]]['约束类型']
                    if iConstraintType in self.PC.ConstraintArgInfoFun:
                        NewArg[self._SharedInfo['约束条件'][i]],iInfo = self._genConstraintArgInfo(arg.get(self._SharedInfo['约束条件'][i]))
                        ArgInfo[self._SharedInfo['约束条件'][i]] = {'数据类型':'ArgSet','取值范围':[iInfo,self._genConstraintArgInfo,None,{}],'是否刷新':True,'序号':NewArg['约束个数']+11,'可否遍历':False}
                        NewConstraintList.append(str(NewArg['约束个数'])+"-"+iConstraintType)
                        NewArg['约束个数'] += 1
                self._SharedInfo["约束条件"] = NewConstraintList
                NewArg['构造器参数'],SubArgInfo = self.PC.genOptionArgInfo(None)
                ArgInfo['构造器参数'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genOptionArgInfo,None,{}],'是否刷新':False,'序号':NewArg['约束个数']+11}
                NewArg['信号调整'],SubArgInfo = self._genSignalAdjustArgInfo(arg['信号调整'])
                ArgInfo['信号调整'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self._genSignalAdjustArgInfo,None,{}],'是否刷新':False,'序号':NewArg['约束个数']+12}
                arg = NewArg
            elif len(arg)!=15+arg['约束个数']:# 调整了约束个数
                NewArg = {"信号滞后期":arg["信号滞后期"],
                          "组合模型":arg.get("组合模型","均值方差模型"),
                          "构造器":arg.get("构造器",self.PC.__doc__),
                          "目标ID":arg["目标ID"],
                          "预期收益":arg['预期收益'],
                          "风险数据源":arg['风险数据源'],
                          '基准权重':arg['基准权重'],
                          "成交额":arg['成交额'],
                          "成交价":arg['成交价'],
                          "优化目标":arg['优化目标'],
                          "约束个数":arg["约束个数"],
                          "空头信号日":arg['空头信号日'],
                          "多头信号日":arg['多头信号日']}
                NewArg['优化目标'],SubArgInfo = self.PC.genObjectArgInfo(arg['优化目标'])
                ArgInfo['优化目标'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genObjectArgInfo,None,{}],'是否刷新':False,'序号':9}
                nConstraintNoChg = min((len(self._SharedInfo['约束条件']),arg['约束个数']))
                self._SharedInfo['约束条件'] = self._SharedInfo['约束条件'][:nConstraintNoChg]
                for i in range(nConstraintNoChg):
                    NewArg[self._SharedInfo['约束条件'][i]],iInfo = self._genConstraintArgInfo(arg.get(self._SharedInfo['约束条件'][i]))
                    ArgInfo[self._SharedInfo['约束条件'][i]] = {'数据类型':'ArgSet','取值范围':[iInfo,self._genConstraintArgInfo,None,{}],'是否刷新':True,'序号':i+11,'可否遍历':False}
                for i in range(arg['约束个数']-nConstraintNoChg):
                    iArgName = str(i+nConstraintNoChg)+"-因子暴露约束"
                    NewArg[iArgName],iInfo = self._genConstraintArgInfo({"约束类型":"因子暴露约束"})
                    ArgInfo[iArgName] = {'数据类型':'ArgSet','取值范围':[iInfo,self._genConstraintArgInfo,None,{}],'是否刷新':True,'序号':i+nConstraintNoChg+11,'可否遍历':False}
                    self._SharedInfo['约束条件'].append(iArgName)
                NewArg['构造器参数'],SubArgInfo = self.PC.genOptionArgInfo(arg['构造器参数'])
                ArgInfo['构造器参数'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genOptionArgInfo,None,{}],'是否刷新':False,'序号':arg['约束个数']+11}
                NewArg['信号调整'],SubArgInfo = self._genSignalAdjustArgInfo(arg['信号调整'])
                ArgInfo['信号调整'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self._genSignalAdjustArgInfo,None,{}],'是否刷新':False,'序号':arg['约束个数']+12}
                arg = NewArg
            else:# 调整了约束条件
                arg['优化目标'],SubArgInfo = self.PC.genObjectArgInfo(arg['优化目标'])
                ArgInfo['优化目标'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genObjectArgInfo,None,{}],'是否刷新':False,'序号':9}
                for i in range(arg['约束个数']):
                    iNewConstraintType = str(i)+"-"+arg[self._SharedInfo['约束条件'][i]]['约束类型']
                    arg[iNewConstraintType],iArgInfo = self._genConstraintArgInfo(arg.pop(self._SharedInfo['约束条件'][i]))
                    self._SharedInfo['约束条件'][i] = iNewConstraintType
                    ArgInfo[iNewConstraintType] = {'数据类型':'ArgSet','取值范围':[iArgInfo,self._genConstraintArgInfo,None,{}],'是否刷新':True,'序号':i+11,'可否遍历':False}
                arg['构造器参数'],SubArgInfo = self.PC.genOptionArgInfo(arg['构造器参数'])
                ArgInfo['构造器参数'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self.PC.genOptionArgInfo,None,{}],'是否刷新':False,'序号':arg['约束个数']+11}
                arg['信号调整'],SubArgInfo = self._genSignalAdjustArgInfo(arg['信号调整'])
                ArgInfo['信号调整'] = {'数据类型':'ArgSet','取值范围':[SubArgInfo,self._genSignalAdjustArgInfo,None,{}],'是否刷新':False,'序号':arg['约束个数']+12}
        ArgInfo['信号滞后期'] = {'数据类型':'Int','取值范围':[0,9999,1],'是否刷新':False,'序号':0,'可否遍历':True}
        ArgInfo['组合模型'] = {'数据类型':'Str','取值范围':list(self.ModelClass.keys()),'是否刷新':True,'序号':1}
        ArgInfo['构造器'] = {'数据类型':'Str','取值范围':list(self.ModelClass[arg["组合模型"]].keys()),'是否刷新':True,'序号':2}
        ArgInfo["目标ID"] = {'数据类型':'IDFilterStr','取值范围':self.StdDataSource.FactorNames,'是否刷新':False,'序号':3,'可否遍历':False}
        ArgInfo['预期收益'] = {'数据类型':'Str','取值范围':DefaultNumFactorList,'是否刷新':False,'序号':4}
        ArgInfo['风险数据源'] = {'数据类型':'RiskDS','取值范围':[],'是否刷新':False,'序号':5}
        ArgInfo['基准权重'] = {'数据类型':'Str','取值范围':DefaultNumFactorList,'是否刷新':False,'序号':6}
        ArgInfo['成交额'] = {'数据类型':'Str','取值范围':DefaultNumFactorList,'是否刷新':False,'序号':7}
        ArgInfo['成交价'] = {'数据类型':'Str','取值范围':DefaultNumFactorList,'是否刷新':False,'序号':8}
        ArgInfo['约束个数'] = {'数据类型':'Int','取值范围':[0,9999,1],'是否刷新':True,'序号':10,'是否可改':getattr(self.PC,"ConstraintAdjustive",True)}
        ArgInfo['多头信号日'] = {'数据类型':'DateList','取值范围':self.StdDataSource.getDate(),'是否刷新':False,'序号':arg['约束个数']+13}
        ArgInfo['空头信号日'] = {'数据类型':'DateList','取值范围':self.StdDataSource.getDate(),'是否刷新':False,'序号':arg['约束个数']+14}
        return (arg,ArgInfo)
    # 启动并初始化信号生成器
    def start(self):
        if self.PC.__doc__!=self.SysArgs["构造器"]:
            self.PC = self.ModelClass[self.SysArgs['组合模型']][self.SysArgs["构造器"]]("MainPC",qs_env=self.QSEnv)
        self.PC.setObject(object_arg=self.SysArgs['优化目标'])
        self.PC.clearConstraint()
        for iConstraintArgName in self._SharedInfo['约束条件']:
            self.PC.addConstraint(constraint_type=self.SysArgs[iConstraintArgName]["约束类型"],constraint_arg=self.SysArgs[iConstraintArgName]["条件参数"])
        self.PC.setOptionArg(option_arg=self.SysArgs['构造器参数'])
        self.PC.initPC()
        if self.SysArgs['风险数据源'] is not None:
            self.SysArgs['风险数据源'].start()
        Rslt = SignalGenerator.start(self)
        self.TempData["CompiledIDFilterStr"] = self.SysArgs['目标ID']# 暂存编译好的过滤条件，加快运行速度
        self.TempData["IDFilterFactors"] = None
        self.Status = []# 记录求解失败的优化问题信息, [(日期, 结果信息)]
        self.ReleasedConstraint = []# 记录被舍弃的约束条件, [(日期, [舍弃的条件])]
        return Rslt
    # 调整信号
    def _adjustSignal(self,raw_signal):
        if raw_signal is not None:
            if self.SysArgs['信号调整']["调整方式"]=="权重下限调整":
                LongSignal = raw_signal[raw_signal>10**self.SysArgs['信号调整']["权重下限幂次"]]
                ShortSignal = raw_signal[raw_signal<-10**self.SysArgs['信号调整']["权重下限幂次"]]
            else:
                LongSignal = raw_signal[raw_signal>0]
                ShortSignal = raw_signal[raw_signal<0]
                LongSignal = LongSignal.sort_values(ascending=False)
                Pos = min((LongSignal.shape[0]-1,(LongSignal.cumsum()<self.SysArgs['信号调整']["权重累计阈值"]*LongSignal.sum()).sum()))
                LongSignal = LongSignal.iloc[:Pos+1]
                ShortSignal = ShortSignal.sort_values(ascending=True)
                Pos = min((ShortSignal.shape[0]-1,(ShortSignal.cumsum()>self.SysArgs['信号调整']["权重累计阈值"]*ShortSignal.sum()).sum()))
                ShortSignal = ShortSignal.iloc[:Pos+1]
            if self.SysArgs['信号调整']["是否归一"]:
                LongSignal = LongSignal/LongSignal.sum()
                ShortSignal = ShortSignal/ShortSignal.abs().sum()
            return {"多头信号":LongSignal,"空头信号":ShortSignal}
        else:
            return {"多头信号":None,"空头信号":None}
    # 产生信号(字典)，无信号则返回None
    def genSignal(self,cur_date):
        isLongSignalDate,isShortSignalDate = self._isSignalDate(cur_date)
        if isLongSignalDate or isShortSignalDate:
            if self.SysArgs['风险数据源'] is not None:
                self.SysArgs['风险数据源'].MoveOn(cur_date)
            if self.SysArgs["目标ID"] is None:
                self.PC.setTargetID(self.StdDataSource.getID(idate=cur_date,is_filtered=True))
            else:
                OldIDFilterStr,OldIDFilterFactors = self.StdDataSource.setIDFilter(self.TempData.get('CompiledIDFilterStr'),self.TempData.get('IDFilterFactors'))
                self.PC.setTargetID(self.StdDataSource.getID(idate=cur_date,is_filtered=True))
                self.TempData['CompiledIDFilterStr'],self.TempData['IDFilterFactors'] = self.StdDataSource.setIDFilter(OldIDFilterStr,id_filter_factors=OldIDFilterFactors)
            if self.PC.UseAmount:
                self.PC.setAmountData(self.StdDataSource.getFactorData(dates=cur_date,ids=None,ifactor_name=self.SysArgs['成交额']).loc[cur_date])
            if self.PC.UseExpectedReturn:
                self.PC.setExpectedReturn(self.StdDataSource.getFactorData(dates=cur_date,ids=None,ifactor_name=self.SysArgs['预期收益']).loc[cur_date])
            if self.PC.UseCovMatrix:
                if self.SysArgs["风险数据源"].RiskDB.DBType=='FRDB':
                    self.PC.setCovMatrix(factor_cov=self.SysArgs["风险数据源"].getDateFactorCov(cur_date),
                                         specific_risk=self.SysArgs["风险数据源"].getDateSpecificRisk(cur_date),
                                         risk_factor_data=self.SysArgs["风险数据源"].getDateFactorData(cur_date))
                else:
                    self.PC.setCovMatrix(cov_matrix=self.SysArgs["风险数据源"].getDateCov(cur_date))                    
            if self.PC.UseFactor!=[]:
                self.PC.setFactorData(factor_data=self.StdDataSource.getDateData(idate=cur_date,factor_names=self.PC.UseFactor,ids=None))
            if self.PC.UseHolding or self.PC.UseAmount or self.PC.UseWealth:
                TradePrice = self.StdDataSource.getFactorData(dates=cur_date,ids=None,ifactor_name=self.SysArgs['成交价']).loc[cur_date]# 获取当前成交价信息
                WealthDistribution = self.LongAccount.calcWealthDistribution(price=TradePrice)
                EquityWealth = WealthDistribution.sum()# 当前证券账户的财富
                CashWealth = self.CashAccount.getWealth()# 当前资金账户财富
                TotalWealth = EquityWealth+CashWealth# 总财富
                self.PC.setHolding(WealthDistribution/TotalWealth)# 当前的投资组合
                self.PC.setWealth(TotalWealth)
            if self.PC.UseBenchmark:
                self.PC.setBenchmarkHolding(self.StdDataSource.getFactorData(dates=cur_date,ids=None,ifactor_name=self.SysArgs['基准权重']).loc[cur_date])
            self.PC.setFilteredID(self.StdDataSource,cur_date)
            RawSignal,ResultInfo = self.PC.solve()
            if ResultInfo['Status']!=1:
                self.Status.append((cur_date,ResultInfo))# debug
                if self.SysArgs['信号调整'].get("打印信息",False):
                    print(cur_date+" : 错误代码-"+str(ResultInfo["ErrorCode"])+"    "+ResultInfo["Msg"])
            if ResultInfo['ReleasedConstraint']!=[]:
                self.ReleasedConstraint.append((cur_date,ResultInfo['ReleasedConstraint']))
                if self.SysArgs['信号调整'].get("打印信息",False):
                    print(cur_date+" : 舍弃约束-"+str(ResultInfo['ReleasedConstraint']))
            Signal = self._adjustSignal(RawSignal)
        else:
            Signal = {"多头信号":None,"空头信号":None}
        Signal = self._complementSignal(Signal)
        self._saveSignal(cur_date,Signal)
        return self._bufferSignal(Signal)
    # 结束信号生成器，并生成特有的结果集
    def endSG(self):
        self.PC.endPC()
        if self.SysArgs['风险数据源'] is not None:
            self.SysArgs['风险数据源'].endDS()
        if self.Status!=[]:
            print("以下日期组合优化问题的求解出现问题: ")
            for iDate,iResultInfo in self.Status:
                print(iDate+" : 错误代码-"+str(iResultInfo["ErrorCode"])+"    "+iResultInfo["Msg"])
        if self.ReleasedConstraint!=[]:
            print("以下日期组合优化问题的求解舍弃了约束条件: ")
            for iDate,iReleasedConstraint in self.ReleasedConstraint:
                print(iDate+" : 舍弃约束-"+str(iReleasedConstraint))
        return SignalGenerator.endSG(self)
    # 保存自身信息
    def saveInfo(self,container):
        RiskDS = self.SysArgs['风险数据源']
        if RiskDS is not None:
            self.SysArgs['风险数据源'] = RiskDS.saveInfo({})
        container = SignalGenerator.saveInfo(self,container)
        container['PC'] = self.PC.saveInfo({})
        container['_SharedInfo'] = self._SharedInfo
        self.SysArgs['风险数据源'] = RiskDS
        return container
    # 恢复信息
    def loadInfo(self,container):
        self._SharedInfo = container['_SharedInfo']
        self.PC.loadInfo(container['PC'])
        if "构造器" not in container['SysArgs']:# 兼容老版本
            SysArgs = container['SysArgs']
            SysArgs["构造器"] = self.PC.__doc__
            SysArgs["组合模型"] = "均值方差模型"
            SysArgs['构造器参数'] = SysArgs.pop("优化器参数")
            SysArgs["构造器参数"].pop("优化器",None)
            SysArgs["优化目标"].pop("优化目标")
            container["SysArgs"] = SysArgs
        Error = SignalGenerator.loadInfo(self,container)
        if self.SysArgs['风险数据源'] is not None:
            import RiskDataSource
            RiskDSInfo = self.SysArgs['风险数据源']
            self.SysArgs['风险数据源'] = RiskDataSource.RDSClasses[RiskDSInfo['RiskDB']][RiskDSInfo['DSType']](RiskDSInfo['Name'],getattr(self.QSEnv,RiskDSInfo['RiskDB'],self.QSEnv.RDB),qs_env=self.QSEnv)
            self.SysArgs['风险数据源'].loadInfo(RiskDSInfo)
        return Error
