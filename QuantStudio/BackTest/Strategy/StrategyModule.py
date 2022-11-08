# -*- coding: utf-8 -*-
import base64
from io import BytesIO
from copy import deepcopy
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Enum, List, Float, ListInt, Instance, on_trait_change
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from QuantStudio import QSArgs
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import summaryStrategy, calcYieldSeq, calcLSYield, formatStrategySummary
from QuantStudio.FactorDataBase.FactorDB import FactorTable
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage

_QS_MinPositionNum = 1e-8# 会被忽略掉的最小持仓数量
_QS_MinCash = 1e-8# 会被忽略掉的最小现金量

def cutDateTime(df, dts=None, start_dt=None, end_dt=None):
    if dts is not None: df = df.reindex(index=dts)
    if start_dt is not None: df = df[df.index>=start_dt]
    if end_dt is not None: df = df[df.index<=end_dt]
    return df
def genAccountOutput(init_cash, cash_series, debt_series, account_value_series, cash_record, debt_record, date_index, risk_free_rate=0.0):
    Output = {}
    # 以时间点为索引的序列
    Output["时间序列"] = pd.DataFrame(cash_series, columns=["现金"])
    Output["时间序列"]["负债"] = debt_series
    Output["时间序列"]["证券"] = account_value_series - (cash_series - debt_series)
    Output["时间序列"]["账户价值"] = account_value_series
    AccountEarnings = account_value_series.diff()
    AccountEarnings.iloc[0] = account_value_series.iloc[0] - init_cash
    # 现金流调整
    CashDelta = cash_record.loc[:, ["时间点", "现金流"]].groupby(by=["时间点"]).sum().get("现金流", pd.Series(dtype=float))
    CashDelta = CashDelta[CashDelta!=0]
    if CashDelta.shape[0]>0:
        Output["时间序列"]["累计资金投入"] = init_cash + CashDelta.reindex(index=Output["时间序列"].index).fillna(0).cumsum()
    else:
        Output["时间序列"]["累计资金投入"] = init_cash
    AccountEarnings[CashDelta.index] -= CashDelta
    Output["时间序列"]["收益"] = AccountEarnings
    PreAccountValue = np.r_[init_cash, account_value_series.values[:-1]]
    AccountReturn = AccountEarnings / np.abs(PreAccountValue)
    AccountReturn[AccountEarnings==0] = 0.0
    Output["时间序列"]["收益率"] = AccountReturn
    AccountReturn[np.isinf(AccountReturn)] = np.nan
    Output["时间序列"]["累计收益率"] = AccountReturn.cumsum()
    Output["时间序列"]["净值"] = (AccountReturn + 1).cumprod()
    if CashDelta.shape[0]>0:
        Output["时间序列"]["考虑资金投入的累计收益率"] = Output["时间序列"]["账户价值"] / Output["时间序列"]["累计资金投入"] - 1
        Output["时间序列"]["考虑资金投入的净值"] = Output["时间序列"]["考虑资金投入的累计收益率"] + 1
    # 负债调整
    debt_record = debt_record[debt_record["融资"]!=0]
    if debt_record.shape[0]>0:
        DebtDelta = debt_record.loc[:, ["时间点", "融资"]].groupby(by=["时间点"]).sum()["融资"]
        PreUnleveredValue = pd.Series(np.r_[init_cash, (account_value_series.values + debt_series.values)[:-1]], index=AccountEarnings.index)
        PreUnleveredValue[DebtDelta.index] += DebtDelta.clip(0, np.inf)
        UnleveredReturn = AccountEarnings / np.abs(PreUnleveredValue)
        UnleveredReturn[AccountEarnings==0] = 0.0
        Output["时间序列"]["无杠杆收益率"] = UnleveredReturn
        UnleveredReturn[np.isinf(UnleveredReturn)] = np.nan
        Output["时间序列"]["无杠杆累计收益率"] = UnleveredReturn.cumsum()
        Output["时间序列"]["无杠杆净值"] = (1+UnleveredReturn).cumprod()
    # 以日期为索引的序列
    if date_index.shape[0]==Output["时间序列"].shape[0]:# 判断回测序列为日级别, 直接将时间点索引更改为日期索引
        Output["日期序列"] = Output.pop("时间序列")
        Output["日期序列"].index = date_index.index
    else:
        Output["日期序列"] = Output["时间序列"].iloc[date_index.values].copy()
        Output["日期序列"].index = date_index.index
        AccountValueSeries = Output["日期序列"]["账户价值"]
        DebtSeries = Output["日期序列"]["负债"]
        AccountEarnings = AccountValueSeries.diff()
        AccountEarnings.iloc[0] = AccountValueSeries.iloc[0] - init_cash
        # 现金流调整
        cash_record = cash_record.copy()
        cash_record["时间点"]= [iDateTime.date() for iDateTime in cash_record["时间点"]]
        CashDelta = cash_record.loc[:, ["时间点", "现金流"]].groupby(by=["时间点"]).sum()["现金流"]
        CashDelta = CashDelta[CashDelta!=0]
        if CashDelta.shape[0]>0:
            Output["日期序列"]["累计资金投入"] = init_cash + CashDelta.reindex(index=Output["日期序列"].index).fillna(0).cumsum()
        else:
            Output["日期序列"]["累计资金投入"] = init_cash
        AccountEarnings[CashDelta.index] -= CashDelta
        Output["日期序列"]["收益"] = AccountEarnings
        PreAccountValue = np.append(np.array(init_cash), AccountValueSeries.values[:-1])
        AccountReturn = AccountEarnings / np.abs(PreAccountValue)
        AccountReturn[AccountEarnings==0] = 0.0
        Output["日期序列"]["收益率"] = AccountReturn
        AccountReturn[np.isinf(AccountReturn)] = np.nan
        Output["日期序列"]["累计收益率"] = AccountReturn.cumsum()
        Output["日期序列"]["净值"] = (AccountReturn+1).cumprod()
        if CashDelta.shape[0]>0:
            Output["日期序列"]["考虑资金投入的累计收益率"] = Output["日期序列"]["账户价值"] / Output["日期序列"]["累计资金投入"] - 1
            Output["日期序列"]["考虑资金投入的净值"] = Output["日期序列"]["考虑资金投入的累计收益率"] + 1
        # 负债调整
        if debt_record.shape[0]>0:
            debt_record = debt_record.copy()
            debt_record["时间点"] = [iDateTime.date() for iDateTime in debt_record["时间点"]]
            DebtDelta = debt_record.loc[:, ["时间点", "融资"]].groupby(by=["时间点"]).sum()["融资"]
            PreUnleveredValue = pd.Series(np.append(np.array(init_cash), (AccountValueSeries.values+DebtSeries.values)[:-1]), index=AccountEarnings.index)
            PreUnleveredValue[DebtDelta.index] += DebtDelta.clip(0, np.inf)
            UnleveredReturn = AccountEarnings / np.abs(PreUnleveredValue)
            UnleveredReturn[AccountEarnings==0] = 0.0
            Output["日期序列"]["无杠杆收益率"] = UnleveredReturn
            UnleveredReturn[np.isinf(UnleveredReturn)] = np.nan
            Output["日期序列"]["无杠杆累计收益率"] = UnleveredReturn.cumsum()
            Output["日期序列"]["无杠杆净值"] = (1 + UnleveredReturn).cumprod()
    # 统计数据
    TargetCols = ["净值"] + ["考虑资金投入的净值"] * (CashDelta.shape[0]>0) + ["无杠杆净值"] * (debt_record.shape[0]>0)
    Output["统计数据"] = summaryStrategy(Output["日期序列"][TargetCols].values, list(Output["日期序列"].index), init_wealth=[1] * len(TargetCols), risk_free_rate=risk_free_rate)
    Output["统计数据"].columns = ["绝对表现"] + ["考虑资金投入的表现"] * (CashDelta.shape[0]>0) + ["无杠杆表现"] * (debt_record.shape[0]>0)
    return Output


# 账户基类, 本身只能存放现金
class Account(BaseModule):
    """账户"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        RiskFreeRate = Float(0.0, arg_type="Float", label="无风险利率", order=-1)
        InitCash = Float(1e8, arg_type="Double", label="初始资金", order=0, low=0.0, high=np.inf, single_step=0.00001, decimals=5)
        DebtLimit = Float(0.0, arg_type="Double", label="负债上限", order=1, low=0.0, high=np.inf, single_step=0.00001, decimals=5)
    def __init__(self, name="Account", sys_args={}, config_file=None, **kwargs):
        self._Cash = None# 剩余现金, >=0,  array(shape=(nDT+1,))
        self._FrozenCash = 0# 当前被冻结的现金, >=0, float
        self._Debt = None# 负债, >=0, array(shape=(nDT+1,))
        self._CashRecord = None# 现金流记录, 现金流入为正, 现金流出为负, DataFrame(columns=["时间点", "现金流", "备注"])
        self._DebtRecord = None# 融资记录, 增加负债为正, 减少负债为负, DataFrame(columns=["时间点", "融资", "备注"])
        self._TradingRecord = None# 交易记录, DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])
        self._Output = None# 缓存的输出结果
        return super().__init__(name=name, sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        nDT = len(dts)
        self._Cash, self._Debt = np.zeros(nDT+1), np.zeros(nDT+1)
        self._Cash[0] = self._QSArgs.InitCash
        self._FrozenCash = 0.0
        self._CashRecord = pd.DataFrame(columns=["时间点", "现金流", "备注"])
        self._DebtRecord = pd.DataFrame(columns=["时间点", "融资", "备注"])
        self._TradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])
        return super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
    def __QS_move__(self, idt, **kwargs):# 先于策略运行
        if self._iDT==idt: return self._TradingRecord
        iIndex = self._Model.DateTimeIndex
        self._Cash[iIndex+1] = self._Cash[iIndex]
        self._Debt[iIndex+1] = self._Debt[iIndex]
        return self._TradingRecord
    def __QS_after_move__(self, idt, **kwargs):# 晚于策略运行
        self._iDT = idt
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        CashSeries = self.getCashSeries()
        DebtSeries = self.getDebtSeries()
        AccountValueSeries = self.getAccountValueSeries()
        self._Output = genAccountOutput(self._QSArgs.InitCash, CashSeries, DebtSeries, AccountValueSeries, self._CashRecord, self._DebtRecord, self._Model.DateIndexSeries, risk_free_rate=self._QSArgs.RiskFreeRate)
        self._Output["现金流记录"] = self._CashRecord
        self._Output["融资记录"] = self._DebtRecord
        self._Output["交易记录"] = self._TradingRecord
        return super().__QS_end__()
    # 当前账户的剩余现金
    @property
    def Cash(self):
        return self._Cash[self._Model.DateTimeIndex+1]
    # 当前账户的负债
    @property
    def Debt(self):
        return self._Debt[self._Model.DateTimeIndex+1]
    # 当前账户可提取的现金, = Cash - FronzenCash + 负债上限 - Debt
    @property
    def AvailableCash(self):
        return self._Cash[self._Model.DateTimeIndex+1] - self._FrozenCash + max(self._QSArgs.DebtLimit - self._Debt[self._Model.DateTimeIndex+1], 0)
    # 当前账户价值, = Cash - Debt
    @property
    def AccountValue(self):
        return self._Cash[self._Model.DateTimeIndex+1] - self._Debt[self._Model.DateTimeIndex+1]
    # 截止到当前的现金流记录
    @property
    def CashRecord(self):
        return self._CashRecord
    # 截止到当前的负债记录
    @property
    def DebtRecord(self):
        return self._DebtRecord
    # 截止到当前的交易记录
    @property
    def TradingRecord(self):
        return self._TradingRecord
    # 剩余现金的历史序列, 以时间点为索引
    def getCashSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.Series(self._Cash[1:self._Model.DateTimeIndex+2], index=self._Model.DateTimeSeries)
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取债务的历史序列, 以时间点为索引
    def getDebtSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.Series(self._Debt[1:self._Model.DateTimeIndex+2], index=self._Model.DateTimeSeries)
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取账户价值的历史序列, 以时间点为索引
    def getAccountValueSeries(self, dts=None, start_dt=None, end_dt=None):
        return self.getCashSeries(dts=dts, start_dt=start_dt, end_dt=end_dt) - self.getDebtSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 更新账户现金负债
    def _QS_updateCashDebt(self, cash_changed):
        iIndex = self._Model.DateTimeIndex + 1
        if cash_changed>0:
            if self._Debt[iIndex]>0:
                DebtDec = min(cash_changed, self._Debt[iIndex])
                self._Debt[iIndex] -= DebtDec
                self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, -DebtDec, "")
            else: DebtDec = 0
            self._Cash[iIndex] += cash_changed - DebtDec
        elif cash_changed<0:
            if -cash_changed>self._Cash[iIndex]:
                DebtInc = min(- cash_changed - self._Cash[iIndex], self._QSArgs.DebtLimit-self._Debt[iIndex])
                self._Debt[iIndex] += DebtInc
                self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, DebtInc, "")
                self._Cash[iIndex] = 0
            else:
                self._Cash[iIndex] += cash_changed
        return 0
    # 抽取现金
    def fetchCash(self, target_cash, remark=""):
        iIndex = self._Model.DateTimeIndex + 1
        Cash = min(target_cash, self.AvailableCash)
        DebtDelta =  - min(0, self._Cash[iIndex] - Cash)
        self._Debt[iIndex] += DebtDelta
        self._Cash[iIndex] -= min(Cash, self._Cash[iIndex])
        self._CashRecord.loc[self._CashRecord.shape[0]] = (self._Model.DateTime, -Cash, remark)
        if DebtDelta>0: self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, DebtDelta, remark)
        return Cash
    # 增加现金
    def addCash(self, target_cash, remark=""):
        iIndex = self._Model.DateTimeIndex + 1
        DebtDelta = - min(target_cash, self._Debt[iIndex])
        self._Cash[iIndex] += target_cash + DebtDelta
        self._Debt[iIndex] += DebtDelta
        self._CashRecord.loc[self._CashRecord.shape[0]] = (self._Model.DateTime, target_cash, remark)
        if DebtDelta<0: self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, DebtDelta, remark)
        return 0

class _Benchmark(QSArgs):
    """基准"""
    FactorTable = Instance(FactorTable, arg_type="FactorTable", label="因子表", order=0)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    #BenchmarkID = Enum(None, arg_type="SingleOption", label="基准ID", order=2)
    RebalanceDTs = List(dt.datetime, arg_type="DateTimeList", label="再平衡时点", order=3)
    RiskFreeRate = Float(0.0, arg_type="Float", label="无风险利率", order=4)
    def __QS_initArgs__(self):
        self.add_trait("PriceFactor", Enum(None, arg_type="SingleOption", label="价格因子", order=1, option_range=[None]))
        self.add_trait("BenchmarkID", Enum(None, arg_type="SingleOption", label="基准ID", order=2, option_range=[None]))
    
    @property
    def ObservedArgs(self):
        return super().ObservedArgs + ("因子表", "价格因子")

    @on_trait_change("FactorTable")
    def _on_FactorTable_changed(self, obj, name, old, new):
        if self.FactorTable is not None:
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self.FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1, option_range=DefaultNumFactorList))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            IDs = self.FactorTable.getID(ifactor_name=self.PriceFactor)
            self.add_trait("BenchmarkID", Enum(*IDs, arg_type="SingleOption", label="基准ID", order=2, option_range=IDs))
        else:
            self.add_trait("PriceFactor", Enum(None, arg_type="SingleOption", label="价格因子", order=1, option_range=[None]))
            self.add_trait("BenchmarkID", Enum(None, arg_type="SingleOption", label="基准ID", order=2, option_range=[None]))
    
    @on_trait_change("PriceFactor")
    def _on_PriceFactor_changed(self, obj, name, old, new):
        if self.FactorTable is not None:
            IDs = self.FactorTable.getID(ifactor_name=self.PriceFactor)
            self.add_trait("BenchmarkID", Enum(*IDs, arg_type="SingleOption", label="基准ID", order=2, option_range=IDs))
        else:
            self.add_trait("BenchmarkID", Enum(None, arg_type="SingleOption", label="基准ID", order=2, option_range=[None]))

# 策略基类
class Strategy(BaseModule):
    """策略基类"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        Benchmark = Instance(_Benchmark, arg_type="ArgObject", label="比较基准", order=0)
        def __QS_initArgs__(self):
            self.Benchmark = _Benchmark()
        
    def __init__(self, name, accounts=[], fts=[], sys_args={}, config_file=None, **kwargs):
        self.Accounts = accounts# 策略所用到的账户
        self.FactorTables = fts# 策略所用到的因子表
        self.ModelArgs = {}# 模型参数，即用户自定义参数
        self.UserData = {}# 用户数据存放
        self._AllSignals = {}# 存储所有生成的信号, {时点:信号}
        return super().__init__(name=name, sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_start__(self, mdl, dts, **kwargs):
        if self._isStarted: return ()
        self.UserData = {}
        self._AllSignals = {}
        Rslt = ()
        for iAccount in self.Accounts: Rslt += iAccount.__QS_start__(mdl=mdl, dts=dts, **kwargs)
        Rslt += super().__QS_start__(mdl=mdl, dts=dts, **kwargs)
        self.init()
        return Rslt+tuple(self.FactorTables)
    def __QS_move__(self, idt, **kwargs):
        if self._iDT==idt: return 0
        self._iDT = idt
        iTradingRecord = {iAccount.Name:iAccount.__QS_move__(idt, **kwargs) for iAccount in self.Accounts}
        Signal = self.genSignal(idt, iTradingRecord)
        self._AllSignals[idt] = Signal
        self.trade(idt, iTradingRecord, Signal)
        for iAccount in self.Accounts: iAccount.__QS_after_move__(idt, **kwargs)
        return 0
    def __QS_end__(self):
        if not self._isStarted: return 0
        for iAccount in self.Accounts: iAccount.__QS_end__()
        self._Output = self._output()
        return super().__QS_end__()
    # 返回策略在时点 idt 生成的信号
    def getSignal(self, idt): 
        return self._AllSignals.get(idt, None)
    # 可选实现
    def init(self):
        return 0
    # 可选实现, trading_record: {账户名: 交易记录, 比如: DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])}
    def genSignal(self, idt, trading_record):
        return None
    # 可选实现, trading_record: {账户名: 交易记录, 比如: DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])}
    def trade(self, idt, trading_record, signal):
        return 0
    def output(self, recalculate=False):
        if recalculate: self._Output = self._output()
        return self._Output
    def _output(self):
        Output = {}
        for i, iAccount in enumerate(self.Accounts):
            iOutput = iAccount.output()
            if iOutput: Output[str(i)+"-"+iAccount.Name] = iOutput
        AccountValueSeries, CashSeries, DebtSeries, InitCash, DebtRecord, CashRecord = 0, 0, 0, 0, None, None
        for iAccount in self.Accounts:
            AccountValueSeries += iAccount.getAccountValueSeries()
            CashSeries += iAccount.getCashSeries()
            DebtSeries += iAccount.getDebtSeries()
            InitCash += iAccount._QSArgs.InitCash
            DebtRecord = iAccount.DebtRecord.append(DebtRecord)
            CashRecord = iAccount.CashRecord.append(CashRecord)
        DebtRecord = DebtRecord[DebtRecord["融资"]!=0]
        StrategyOutput = genAccountOutput(InitCash, CashSeries, DebtSeries, AccountValueSeries, CashRecord, DebtRecord, self._Model.DateIndexSeries, risk_free_rate=self._QSArgs.Benchmark.RiskFreeRate)
        if self._QSArgs.Benchmark.FactorTable is not None:# 设置了基准
            BenchmarkPrice = self._QSArgs.Benchmark.FactorTable.readData(factor_names=[self._QSArgs.Benchmark.PriceFactor], dts=AccountValueSeries.index.tolist(), ids=[self._QSArgs.Benchmark.BenchmarkID]).iloc[0,:,0]
            BenchmarkOutput = pd.DataFrame(calcYieldSeq(wealth_seq=BenchmarkPrice.values), index=BenchmarkPrice.index, columns=["基准收益率"])
            BenchmarkOutput["基准累计收益率"] = BenchmarkOutput["基准收益率"].cumsum()
            BenchmarkOutput["基准净值"] = BenchmarkPrice / BenchmarkPrice.iloc[0]
            if DebtRecord.shape[0]>0:
                LYield = (StrategyOutput["日期序列"]["无杠杆收益率"].values if "时间序列" not in StrategyOutput else StrategyOutput["时间序列"]["无杠杆收益率"].values)
            else:
                LYield = (StrategyOutput["日期序列"]["收益率"].values if "时间序列" not in StrategyOutput else StrategyOutput["时间序列"]["收益率"].values)
            if not self._QSArgs.Benchmark.RebalanceDTs: RebalanceIndex = None
            else:
                RebalanceIndex = pd.Series(np.arange(BenchmarkOutput["基准收益率"].shape[0]), index=BenchmarkOutput["基准收益率"].index, dtype=int)
                RebalanceIndex = sorted(RebalanceIndex.loc[RebalanceIndex.index.intersection(self._QSArgs.Benchmark.RebalanceDTs)].values)
            BenchmarkOutput["相对收益率"] = calcLSYield(long_yield=LYield, short_yield=BenchmarkOutput["基准收益率"].values, rebalance_index=RebalanceIndex)
            BenchmarkOutput["相对累计收益率"] = BenchmarkOutput["相对收益率"].cumsum()
            BenchmarkOutput["相对净值"] = (1 + BenchmarkOutput["相对收益率"]).cumprod()
            if "时间序列" in StrategyOutput:
                StrategyOutput["时间序列"] = pd.merge(StrategyOutput["时间序列"], BenchmarkOutput, left_index=True, right_index=True)
                BenchmarkOutput = BenchmarkOutput.iloc[self._Model.DateIndexSeries.values]
                BenchmarkOutput["基准收益率"] = BenchmarkOutput["基准净值"].values / np.r_[1, BenchmarkOutput["基准净值"].iloc[:-1].values] - 1
                BenchmarkOutput["基准累计收益率"] = BenchmarkOutput["基准收益率"].cumsum()
                BenchmarkOutput["相对收益率"] = BenchmarkOutput["相对净值"].values / np.r_[1, BenchmarkOutput["相对净值"].iloc[:-1].values] - 1
                BenchmarkOutput["相对累计收益率"] = BenchmarkOutput["相对收益率"].cumsum()
            BenchmarkOutput.index = StrategyOutput["日期序列"].index
            StrategyOutput["日期序列"] = pd.merge(StrategyOutput["日期序列"], BenchmarkOutput, left_index=True, right_index=True)
            BenchmarkStatistics = summaryStrategy(BenchmarkOutput[["基准净值", "相对净值"]].values, list(BenchmarkOutput.index), init_wealth=[1, 1], risk_free_rate=self._QSArgs.Benchmark.RiskFreeRate)
            BenchmarkStatistics.columns = ["基准表现", "相对表现"]
            StrategyOutput["统计数据"] = pd.merge(StrategyOutput["统计数据"], BenchmarkStatistics, left_index=True, right_index=True)
        Output["Strategy"] = StrategyOutput
        return Output
    def genMatplotlibFig(self, file_path=None):
        StrategyOutput = self._Output["Strategy"]
        hasCapitalInvest = ("考虑资金投入的表现" in StrategyOutput["统计数据"])
        hasBenchmark = ("相对表现" in StrategyOutput["统计数据"])
        nRow, nCol = 1, 2+hasCapitalInvest+hasBenchmark
        Fig = Figure(figsize=(min(40, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(0, StrategyOutput["日期序列"].shape[0])
        xTicks = np.arange(0, StrategyOutput["日期序列"].shape[0], int(StrategyOutput["日期序列"].shape[0]/10))
        xTickLabels = [StrategyOutput["日期序列"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        iAxes = Fig.add_subplot(nRow, nCol, 1)
        iAxes.plot(xData, StrategyOutput["日期序列"]["账户价值"].values, label="账户价值", color="indianred", lw=2.5)
        iRAxes = iAxes.twinx()
        iRAxes.bar(xData, StrategyOutput["日期序列"]["收益"].values, label="账户收益", color="steelblue")
        iRAxes.legend(loc="upper right")
        iAxes.set_xticks(xTicks)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc="upper left")
        iAxes.set_title("账户表现")
        iAxes = Fig.add_subplot(nRow, nCol, 2)
        iRAxes = iAxes.twinx()
        iRAxes.yaxis.set_major_formatter(yMajorFormatter)
        if "无杠杆净值" in StrategyOutput["日期序列"]:
            iAxes.plot(xData, StrategyOutput["日期序列"]["无杠杆净值"].values, label="无杠杆净值", color="indianred", lw=2.5)
            iRAxes.bar(xData, StrategyOutput["日期序列"]["无杠杆收益率"].values, label="无杠杆收益率", color="steelblue")
        else:
            iAxes.plot(xData, StrategyOutput["日期序列"]["净值"].values, label="净值", color="indianred", lw=2.5)
            iRAxes.bar(xData, StrategyOutput["日期序列"]["收益率"].values, label="收益率", color="steelblue")
        if hasBenchmark: iAxes.plot(xData, StrategyOutput["日期序列"]["基准净值"].values, label="基准净值", color="forestgreen", lw=2.5)
        iRAxes.legend(loc="upper right")
        iAxes.set_xticks(xTicks)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc="upper left")
        iAxes.set_title("净值表现")
        if hasCapitalInvest:
            iAxes = Fig.add_subplot(nRow, nCol, 3)
            iAxes.plot(xData, StrategyOutput["日期序列"]["累计资金投入"].values, label="累计资金投入", color="indianred", lw=2.5)
            iRAxes = iAxes.twinx()
            iRAxes.yaxis.set_major_formatter(yMajorFormatter)
            iRAxes.plot(xData, StrategyOutput["日期序列"]["考虑资金投入的累计收益率"].values, label="考虑资金投入的累计收益率", color="steelblue", lw=2.5)
            iRAxes.legend(loc="upper right")
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iAxes.set_title("考虑资金投入的表现")
        if hasBenchmark:
            iAxes = Fig.add_subplot(nRow, nCol, 3+hasCapitalInvest)
            iAxes.plot(xData, StrategyOutput["日期序列"]["相对净值"].values, label="相对净值", color="indianred", lw=2.5)
            iRAxes = iAxes.twinx()
            iRAxes.yaxis.set_major_formatter(yMajorFormatter)
            iRAxes.bar(xData, StrategyOutput["日期序列"]["相对收益率"].values, label="相对收益率", color="steelblue")
            iRAxes.legend(loc="upper right")
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iAxes.set_title("相对表现")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        HTML = formatStrategySummary(self._Output["Strategy"]["统计数据"]).to_html()
        Pos = HTML.find(">")
        HTML = HTML[:Pos]+' align="center"'+HTML[Pos:]
        Fig = self.genMatplotlibFig()
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer)
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML

# 多策略对比
class MultiStrategy(BaseModule):
    """多策略对比"""
    Modules = List(Strategy, arg_type="List", label="对比策略", order=0)
    BenchmarkStrategies = ListInt(arg_type="List", label="基准策略", order=1)
    RebalanceDTs = List(dt.datetime, arg_type="DateTimeList", label="再平衡时点", order=2)
    def __init__(self, name="多策略对比", sys_args={}, **kwargs):
        super().__init__(name=name, sys_args=sys_args, **kwargs)
        self._QS_isMulti = True
    def output(self, recalculate=False):
        if (not recalculate)  and self._Output: return self._Output
        if len(self.Modules)==0:
            self._Output = {}
            return self._Output
        self._Output = {"日期序列": {"现金": {}, "负债": {}, "证券": {}, "账户价值":{}, "累计资金投入": {},
                                                       "收益率": {}, "无杠杆收益率":{}, "相对收益率": {},
                                                       "累计收益率": {}, "考虑资金投入的累计收益率": {}, "无杠杆累计收益率": {}, "相对累计收益率": {},
                                                       "净值": {}, "考虑资金投入的净值": {}, "无杠杆净值":{}, "相对净值": {}},
                                  "统计数据": {"绝对表现": {}, "考虑资金投入的表现": {}, "无杠杆表现": {}, "基准表现": {}, "相对表现": {}}}
        SeriesDefault = {"无杠杆收益率": "收益率", "无杠杆累计收益率": "累计收益率", "无杠杆净值": "净值", "考虑资金投入的累计收益率": "累计收益率", "考虑资金投入的净值": "净值"}
        SeriesAnyExist = {jKey: False for jKey in self._Output["日期序列"]}
        NonLeverageAnyExist = False
        CapitalInvestAnyExist = False
        StrategyNames = []
        for i, iModule in enumerate(self.Modules):
            iOutput = iModule.output(recalculate=recalculate)
            iName = str(i)+"-"+iModule.Name
            StrategyNames.append(iName)
            self._Output[iName] = iOutput
            if (i==0) and ("时间序列" in iOutput):
                self._Output["时间序列"] = deepcopy(self._Output["日期序列"])
            for jKey in self._Output["日期序列"]:
                if jKey in iOutput["Strategy"]["日期序列"]:
                    self._Output["日期序列"][jKey][iName] = iOutput["Strategy"]["日期序列"][jKey]
                    SeriesAnyExist[jKey] = True
                elif jKey in SeriesDefault:
                    self._Output["日期序列"][jKey][iName] = iOutput["Strategy"]["日期序列"][SeriesDefault[jKey]]
                if "时间序列" in iOutput:
                    if jKey in iOutput["Strategy"]["时间序列"]:
                        self._Output["时间序列"][jKey][iName] = iOutput["Strategy"]["时间序列"][jKey]
                        SeriesAnyExist[jKey] = True
                    elif jKey in SeriesDefault:
                        self._Output["时间序列"][jKey][iName] = iOutput["Strategy"]["时间序列"][SeriesDefault[jKey]]
            self._Output["统计数据"]["绝对表现"][iName] = iOutput["Strategy"]["统计数据"]["绝对表现"]
            if "考虑资金投入的表现" in iOutput["Strategy"]["统计数据"]:
                self._Output["统计数据"]["考虑资金投入的表现"][iName] = iOutput["Strategy"]["统计数据"]["考虑资金投入的表现"]
                CapitalInvestAnyExist = True
            else:
                self._Output["统计数据"]["考虑资金投入的表现"][iName] = iOutput["Strategy"]["统计数据"]["绝对表现"]
            if "无杠杆表现" in iOutput["Strategy"]["统计数据"]:
                self._Output["统计数据"]["无杠杆表现"][iName] = iOutput["Strategy"]["统计数据"]["无杠杆表现"]
                NonLeverageAnyExist = True
            else:
                self._Output["统计数据"]["无杠杆表现"][iName] = iOutput["Strategy"]["统计数据"]["绝对表现"]
            if "相对表现" in iOutput["Strategy"]["统计数据"]:
                self._Output["统计数据"]["基准表现"][iName] = iOutput["Strategy"]["统计数据"]["基准表现"]
                self._Output["统计数据"]["相对表现"][iName] = iOutput["Strategy"]["统计数据"]["相对表现"]
        for jKey in list(self._Output["日期序列"]):
            if SeriesAnyExist.get(jKey, True):
                self._Output["日期序列"][jKey] = pd.DataFrame(self._Output["日期序列"][jKey]).loc[:, StrategyNames]
                if "时间序列" in self._Output: self._Output["时间序列"][jKey] = pd.DataFrame(self._Output["时间序列"][jKey]).loc[:, StrategyNames]
            else:
                self._Output["日期序列"].pop(jKey)
                if "时间序列" in self._Output: self._Output["时间序列"].pop(jKey)
        self._Output["统计数据"]["绝对表现"] = pd.DataFrame(self._Output["统计数据"]["绝对表现"]).loc[:, StrategyNames]
        if CapitalInvestAnyExist:
            self._Output["统计数据"]["考虑资金投入的表现"] = pd.DataFrame(self._Output["统计数据"]["考虑资金投入的表现"]).loc[:, StrategyNames]
        else:
            self._Output["统计数据"].pop("考虑资金投入的表现")
        if NonLeverageAnyExist:
            self._Output["统计数据"]["无杠杆表现"] = pd.DataFrame(self._Output["统计数据"]["无杠杆表现"]).loc[:, StrategyNames]
        else:
            self._Output["统计数据"].pop("无杠杆表现")
        if not self._Output["统计数据"]["相对表现"]:
            self._Output["统计数据"].pop("相对表现")
            self._Output["统计数据"].pop("基准表现")
        else:
            self._Output["统计数据"]["相对表现"] = pd.DataFrame(self._Output["统计数据"]["相对表现"]).loc[:, StrategyNames]
            self._Output["统计数据"]["基准表现"] = pd.DataFrame(self._Output["统计数据"]["基准表现"]).loc[:, StrategyNames]
        if len(self.BenchmarkStrategies)>0:
            self._Output["相对表现"] = self._genRelativeOutput(self._Output)
        return self._Output
    def _genRelativeOutput(self, output):
        RelativeOutput = {"日期序列": {"收益率": {}, "累计收益率": {}, "净值": {}}, "统计数据": {}}
        Return = output["日期序列"]["收益率"]
        if not self.RebalanceDTs: RebalanceIndex = None
        else:
            RebalanceIndex = pd.Series(np.arange(Return.shape[0]), index=Return.index, dtype=int)
            RebalanceIndex = sorted(RebalanceIndex.loc[RebalanceIndex.index.intersection(self.RebalanceDTs)].values)
        for jStrategyIdx in self.BenchmarkStrategies:
            jBenchmarkName = str(jStrategyIdx)+"-"+self.Modules[jStrategyIdx].Name
            jStrategyNames = []
            RelativeOutput["日期序列"]["收益率"][jBenchmarkName] = {}
            for i, iModule in enumerate(self.Modules):
                if i==jStrategyIdx: continue
                iName = str(i)+"-"+iModule.Name
                jStrategyNames.append(iName)
                RelativeOutput["日期序列"]["收益率"][jBenchmarkName][iName] = calcLSYield(long_yield=Return[iName].values, short_yield=Return[jBenchmarkName].values, rebalance_index=RebalanceIndex)
            RelativeOutput["日期序列"]["收益率"][jBenchmarkName] = pd.DataFrame(RelativeOutput["日期序列"]["收益率"][jBenchmarkName], index=Return.index).loc[:, jStrategyNames]
            RelativeOutput["日期序列"]["累计收益率"][jBenchmarkName] = RelativeOutput["日期序列"]["收益率"][jBenchmarkName].cumsum()
            RelativeOutput["日期序列"]["净值"][jBenchmarkName] = (1 + RelativeOutput["日期序列"]["收益率"][jBenchmarkName]).cumprod()
            RelativeOutput["统计数据"][jBenchmarkName] = summaryStrategy(RelativeOutput["日期序列"]["净值"][jBenchmarkName].values, list(RelativeOutput["日期序列"]["净值"][jBenchmarkName].index), init_wealth=[1]*RelativeOutput["日期序列"]["净值"][jBenchmarkName].shape[1], risk_free_rate=0)
            RelativeOutput["统计数据"][jBenchmarkName].columns = RelativeOutput["日期序列"]["净值"][jBenchmarkName].columns
        return RelativeOutput
    def _formatStatistics(self, stats):
        FormattedStats = pd.DataFrame(index=stats.index, columns=stats.columns, dtype="O")
        DateFormatFun = np.vectorize(lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT")
        IntFormatFun = np.vectorize(lambda x: ("%d" % (x, )))
        FloatFormatFun = np.vectorize(lambda x: ("%.2f" % (x, )))
        PercentageFormatFun = np.vectorize(lambda x: ("%.2f%%" % (x*100, )))
        FormattedStats.iloc[:2] = DateFormatFun(stats.iloc[:2, :].values)
        FormattedStats.iloc[2] = IntFormatFun(stats.iloc[2, :].values)
        FormattedStats.iloc[3:6] = PercentageFormatFun(stats.iloc[3:6, :].values)
        FormattedStats.iloc[6:8] = FloatFormatFun(stats.iloc[6:8, :].values)
        FormattedStats.iloc[8:10] = PercentageFormatFun(stats.iloc[8:10, :].values)
        FormattedStats.iloc[10:] = DateFormatFun(stats.iloc[10:, :].values)
        return FormattedStats
    def genMatplotlibFig(self, file_path=None):
        StrategyOutput = self._Output
        hasCapitalInvest = ("考虑资金投入的表现" in StrategyOutput["统计数据"])
        hasBenchmark = ("相对表现" in StrategyOutput["统计数据"])
        nRow, nCol = 1, 2+hasCapitalInvest+hasBenchmark
        nBenchmarkStrategy = len(self.BenchmarkStrategies)
        if nBenchmarkStrategy>0:
            nRow, nCol = 1+nBenchmarkStrategy//3+(nBenchmarkStrategy%3!=0), 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(0, StrategyOutput["日期序列"]["账户价值"].shape[0])
        xTicks = np.arange(0, StrategyOutput["日期序列"]["账户价值"].shape[0], int(StrategyOutput["日期序列"]["账户价值"].shape[0]/10))
        xTickLabels = [StrategyOutput["日期序列"]["账户价值"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        iAxes = Fig.add_subplot(nRow, nCol, 1)
        for i, iCol in enumerate(StrategyOutput["日期序列"]["账户价值"].columns):
            iAxes.plot(xData, StrategyOutput["日期序列"]["账户价值"].values[:, i], label=iCol+"-账户价值", lw=2.5)
        iAxes.set_xticks(xTicks)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc="upper left")
        iAxes.set_title("账户表现")
        iAxes = Fig.add_subplot(nRow, nCol, 2)
        if "无杠杆净值" in StrategyOutput["日期序列"]:
            for i, iCol in enumerate(StrategyOutput["日期序列"]["无杠杆净值"].columns):
                iAxes.plot(xData, StrategyOutput["日期序列"]["无杠杆净值"].values[:, i], label=iCol+"-无杠杆净值", lw=2.5)
        else:
            for i, iCol in enumerate(StrategyOutput["日期序列"]["净值"].columns):
                iAxes.plot(xData, StrategyOutput["日期序列"]["净值"].values[:, i], label=iCol+"-净值", lw=2.5)
        iAxes.set_xticks(xTicks)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc="upper left")
        iAxes.set_title("净值表现")
        if hasCapitalInvest:
            iAxes = Fig.add_subplot(nRow, nCol, 3)
            for i, iCol in enumerate(StrategyOutput["日期序列"]["考虑资金投入的累计收益率"].columns):
                iAxes.plot(xData, StrategyOutput["日期序列"]["考虑资金投入的累计收益率"].values[:, i], label=iCol+"-考虑资金投入的累计收益率", lw=2.5)
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iAxes.set_title("考虑资金投入的表现")
        if hasBenchmark:
            iAxes = Fig.add_subplot(nRow, nCol, 3+hasCapitalInvest)
            for i, iCol in enumerate(StrategyOutput["日期序列"]["相对净值"].columns):
                iAxes.plot(xData, StrategyOutput["日期序列"]["相对净值"].values[:, i], label=iCol+"-相对净值", lw=2.5)
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iAxes.set_title("相对表现")
        if nBenchmarkStrategy>0:
            for j, jStrategyIdx in enumerate(self.BenchmarkStrategies):
                jBenchmarkName = str(jStrategyIdx)+"-"+self.Modules[jStrategyIdx].Name
                iAxes = Fig.add_subplot(nRow, nCol, 4+j)
                jNV = StrategyOutput["相对表现"]["日期序列"]["净值"][jBenchmarkName]
                for i in range(jNV.shape[1]):
                    iAxes.plot(jNV.index, jNV.iloc[:, i].values, label=str(jNV.columns[i]), lw=2.5)
                iAxes.legend(loc="best")
                iAxes.set_title("相对 %s 表现" % jBenchmarkName)
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        HTML = ""
        Output = self.output()
        for iKey in ["绝对表现", "考虑资金投入的表现", "无杠杆表现", "基准表现", "相对表现"]:
            if iKey in Output["统计数据"]:
                iHTML = self._formatStatistics(Output["统计数据"][iKey]).to_html()
                iPos = iHTML.find(">")
                HTML += iKey+": "+iHTML[:iPos]+' align="center"'+iHTML[iPos:]
        if len(self.BenchmarkStrategies)>0:
            for j, jStrategyIdx in enumerate(self.BenchmarkStrategies):
                jBenchmarkName = str(jStrategyIdx)+"-"+self.Modules[jStrategyIdx].Name
                iHTML = self._formatStatistics(Output["相对表现"]["统计数据"][jBenchmarkName]).to_html()
                iPos = iHTML.find(">")
                HTML += ("相对 %s 表现" % jBenchmarkName)+": "+iHTML[:iPos]+' align="center"'+iHTML[iPos:]
        Fig = self.genMatplotlibFig()
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer)
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML