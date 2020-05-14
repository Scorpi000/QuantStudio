# -*- coding: utf-8 -*-
import os
import shutil
import base64
from io import BytesIO
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Float, Str, Instance, Dict, on_trait_change
from traitsui.api import Item, Group, View
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.BackTest.BackTestModel import BaseModule
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import summaryStrategy, calcYieldSeq, calcLSYield
from QuantStudio.FactorDataBase.FactorDB import FactorTable
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage

_QS_MinPositionNum = 1e-8# 会被忽略掉的最小持仓数量
_QS_MinCash = 1e-8# 会被忽略掉的最小现金量

def cutDateTime(df, dts=None, start_dt=None, end_dt=None):
    if dts is not None: df = df.loc[dts]
    if start_dt is not None: df = df[df.index>=start_dt]
    if end_dt is not None: df = df[df.index<=end_dt]
    return df
def genAccountOutput(init_cash, cash_series, debt_series, account_value_series, cash_record, debt_record, date_index):
    Output = {}
    # 以时间点为索引的序列
    Output["时间序列"] = pd.DataFrame(cash_series, columns=["现金"])
    Output["时间序列"]["负债"] = debt_series
    Output["时间序列"]["证券"] = account_value_series - (cash_series - debt_series)
    Output["时间序列"]["账户价值"] = account_value_series
    AccountEarnings = account_value_series.diff()
    AccountEarnings.iloc[0] = account_value_series.iloc[0] - init_cash
    # 现金流调整
    CashDelta = cash_record.loc[:, ["时间点", "现金流"]].groupby(by=["时间点"]).sum()["现金流"]
    AccountEarnings[CashDelta.index] -= CashDelta
    Output["时间序列"]["收益"] = AccountEarnings
    PreAccountValue = np.r_[init_cash, account_value_series.values[:-1]]
    AccountReturn = AccountEarnings / np.abs(PreAccountValue)
    AccountReturn[AccountEarnings==0] = 0.0
    Output["时间序列"]["收益率"] = AccountReturn
    AccountReturn[np.isinf(AccountReturn)] = np.nan
    Output["时间序列"]["累计收益率"] = AccountReturn.cumsum()
    Output["时间序列"]["净值"] = (AccountReturn + 1).cumprod()
    # 负债调整
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
        AccountEarnings.iloc[0] = AccountValueSeries.iloc[0]-init_cash
        # 现金流调整
        cash_record = cash_record.copy()
        cash_record["时间点"]= [iDateTime.date() for iDateTime in cash_record["时间点"]]
        CashDelta = cash_record.loc[:, ["时间点", "现金流"]].groupby(by=["时间点"]).sum()["现金流"]
        AccountEarnings[CashDelta.index] -= CashDelta
        Output["日期序列"]["收益"] = AccountEarnings
        PreAccountValue = np.append(np.array(init_cash), AccountValueSeries.values[:-1])
        AccountReturn = AccountEarnings / np.abs(PreAccountValue)
        AccountReturn[AccountEarnings==0] = 0.0
        Output["日期序列"]["收益率"] = AccountReturn
        AccountReturn[np.isinf(AccountReturn)] = np.nan
        Output["日期序列"]["累计收益率"] = AccountReturn.cumsum()
        Output["日期序列"]["净值"] = (AccountReturn+1).cumprod()
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
    Output["统计数据"] = summaryStrategy(Output["日期序列"][["净值", "无杠杆净值"]].values, list(Output["日期序列"].index), init_wealth=[1, 1])
    Output["统计数据"].columns = ["账户价值", "无杠杆价值"]
    return Output


# 账户基类, 本身只能存放现金
class Account(BaseModule):
    """账户"""
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
        self._Cash[0] = self.InitCash
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
        self._Output = genAccountOutput(self.InitCash, CashSeries, DebtSeries, AccountValueSeries, self._CashRecord, self._DebtRecord, self._Model.DateIndexSeries)
        self._Output["现金流记录"] = self._CashRecord
        self._Output["融资记录"] = self._DebtRecord
        self._Output["交易记录"] = self._TradingRecord
        self._Output = self.output(recalculate=True)
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
        return self._Cash[self._Model.DateTimeIndex+1] - self._FrozenCash + max(self.DebtLimit - self._Debt[self._Model.DateTimeIndex+1], 0)
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
                DebtInc = min(- cash_changed - self._Cash[iIndex], self.DebtLimit-self._Debt[iIndex])
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

class _Benchmark(__QS_Object__):
    """基准"""
    FactorTable = Instance(FactorTable, arg_type="FactorTable", label="因子表", order=0)
    #PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    #BenchmarkID = Enum(None, arg_type="SingleOption", label="基准ID", order=2)
    RebalanceDTs = List(dt.datetime, arg_type="DateTimeList", label="再平衡时点", order=3)
    def __QS_initArgs__(self):
        self.add_trait("PriceFactor", Enum(None, arg_type="SingleOption", label="价格因子", order=1))
        self.add_trait("BenchmarkID", Enum(None, arg_type="SingleOption", label="基准ID", order=2))
    @on_trait_change("FactorTable")
    def _on_FactorTable_changed(self, obj, name, old, new):
        if self.FactorTable is not None:
            DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(self.FactorTable.getFactorMetaData(key="DataType")))
            self.add_trait("PriceFactor", Enum(*DefaultNumFactorList, arg_type="SingleOption", label="价格因子", order=1))
            self.PriceFactor = searchNameInStrList(DefaultNumFactorList, ['价','Price','price'])
            self.add_trait("BenchmarkID", Enum(*self.FactorTable.getID(ifactor_name=self.PriceFactor), arg_type="SingleOption", label="基准ID", order=2))
        else:
            self.add_trait("PriceFactor", Enum(None, arg_type="SingleOption", label="价格因子", order=1))
            self.add_trait("BenchmarkID", Enum(None, arg_type="SingleOption", label="基准ID", order=2))
    @on_trait_change("PriceFactor")
    def _on_PriceFactor_changed(self, obj, name, old, new):
        if self.FactorTable is not None:
            self.add_trait("BenchmarkID", Enum(*self.FactorTable.getID(ifactor_name=self.PriceFactor), arg_type="SingleOption", label="基准ID", order=2))
        else:
            self.add_trait("BenchmarkID", Enum(None, arg_type="SingleOption", label="基准ID", order=2))
# 策略基类
class Strategy(BaseModule):
    """策略基类"""
    Accounts = List(Account)# 策略所用到的账户
    FactorTables = List(FactorTable)# 策略所用到的因子表
    Benchmark = Instance(_Benchmark, arg_type="ArgObject", label="比较基准", order=0)
    def __init__(self, name, accounts=[], fts=[], sys_args={}, config_file=None, **kwargs):
        self.Accounts = accounts# 策略所用到的账户
        self.FactorTables = fts# 策略所用到的因子表
        self.ModelArgs = {}# 模型参数，即用户自定义参数
        self.UserData = {}# 用户数据存放
        self._AllSignals = {}# 存储所有生成的信号, {时点:信号}
        return super().__init__(name=name, sys_args=sys_args, config_file=config_file, **kwargs)
    def __QS_initArgs__(self):
        self.Benchmark = _Benchmark()
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
        return super().__QS_end__()
    def getViewItems(self, context_name=""):
        Prefix = (context_name+"." if context_name else "")
        Groups, Context = [], {}
        for j, jAccount in enumerate(self.Accounts):
            jItems, jContext = jAccount.getViewItems(context_name=context_name+"_Account"+str(j))
            Groups.append(Group(*jItems, label=str(j)+"-"+jAccount.Name))
            Context.update(jContext)
        return ([Group(*Groups, orientation='horizontal', layout='tabbed', springy=True)], Context)
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
        if not recalculate: return self._Output
        for i, iAccount in enumerate(self.Accounts):
            iOutput = iAccount.output(recalculate=True)
            if iOutput: self._Output[str(i)+"-"+iAccount.Name] = iOutput
        AccountValueSeries, CashSeries, DebtSeries, InitCash, DebtRecord, CashRecord = 0, 0, 0, 0, None, None
        for iAccount in self.Accounts:
            AccountValueSeries += iAccount.getAccountValueSeries()
            CashSeries += iAccount.getCashSeries()
            DebtSeries += iAccount.getDebtSeries()
            InitCash += iAccount.InitCash
            DebtRecord = iAccount.DebtRecord.append(DebtRecord)
            CashRecord = iAccount.CashRecord.append(CashRecord)
        StrategyOutput = genAccountOutput(InitCash, CashSeries, DebtSeries, AccountValueSeries, CashRecord, DebtRecord, self._Model.DateIndexSeries)
        StrategyOutput["统计数据"].columns = ["策略表现", "无杠杆表现"]
        if self.Benchmark.FactorTable is not None:# 设置了基准
            BenchmarkPrice = self.Benchmark.FactorTable.readData(factor_names=[self.Benchmark.PriceFactor], dts=AccountValueSeries.index.tolist(), ids=[self.Benchmark.BenchmarkID]).iloc[0,:,0]
            BenchmarkOutput = pd.DataFrame(calcYieldSeq(wealth_seq=BenchmarkPrice.values), index=BenchmarkPrice.index, columns=["基准收益率"])
            BenchmarkOutput["基准累计收益率"] = BenchmarkOutput["基准收益率"].cumsum()
            BenchmarkOutput["基准净值"] = BenchmarkPrice / BenchmarkPrice.iloc[0]
            LYield = (StrategyOutput["日期序列"]["无杠杆收益率"].values if "时间序列" not in StrategyOutput else StrategyOutput["时间序列"]["无杠杆收益率"].values)
            if not self.Benchmark.RebalanceDTs: RebalanceIndex = None
            else:
                RebalanceIndex = pd.Series(np.arange(BenchmarkOutput["基准收益率"].shape[0]), index=BenchmarkOutput["基准收益率"].index, dtype=int)
                RebalanceIndex = RebalanceIndex.loc[RebalanceIndex.index.intersection(self.Benchmark.RebalanceDTs)].values.tolist()
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
            BenchmarkStatistics = summaryStrategy(BenchmarkOutput[["基准净值", "相对净值"]].values, list(BenchmarkOutput.index), init_wealth=[1, 1])
            BenchmarkStatistics.columns = ["基准", "相对表现"]
            StrategyOutput["统计数据"] = pd.merge(StrategyOutput["统计数据"], BenchmarkStatistics, left_index=True, right_index=True)
        self._Output["Strategy"] = StrategyOutput
        return self._Output
    def _formatStatistics(self):
        Stats = self._Output["Strategy"]["统计数据"]
        FormattedStats = pd.DataFrame(index=Stats.index, columns=Stats.columns, dtype="O")
        DateFormatFun = np.vectorize(lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT")
        IntFormatFun = np.vectorize(lambda x: ("%d" % (x, )))
        FloatFormatFun = np.vectorize(lambda x: ("%.2f" % (x, )))
        PercentageFormatFun = np.vectorize(lambda x: ("%.2f%%" % (x*100, )))
        FormattedStats.iloc[:2] = DateFormatFun(Stats.iloc[:2, :].values)
        FormattedStats.iloc[2] = IntFormatFun(Stats.iloc[2, :].values)
        FormattedStats.iloc[3:6] = PercentageFormatFun(Stats.iloc[3:6, :].values)
        FormattedStats.iloc[6] = FloatFormatFun(Stats.iloc[6, :].values)
        FormattedStats.iloc[7:9] = PercentageFormatFun(Stats.iloc[7:9, :].values)
        FormattedStats.iloc[9:] = DateFormatFun(Stats.iloc[9:, :].values)
        return FormattedStats
    def genMatplotlibFig(self, file_path=None):
        StrategyOutput = self._Output["Strategy"]
        hasBenchmark = ("基准" in StrategyOutput["统计数据"])
        if hasBenchmark: nRow, nCol = 1, 3
        else: nRow, nCol = 1, 2
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
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
        iAxes.legend(loc='upper left')
        iAxes.set_title("策略表现")
        iAxes = Fig.add_subplot(nRow, nCol, 2)
        iAxes.plot(xData, StrategyOutput["日期序列"]["无杠杆净值"].values, label="无杠杆净值", color="indianred", lw=2.5)
        if hasBenchmark: iAxes.plot(xData, StrategyOutput["日期序列"]["基准净值"].values, label="基准净值", color="forestgreen", lw=2.5)
        iRAxes = iAxes.twinx()
        iRAxes.yaxis.set_major_formatter(yMajorFormatter)
        iRAxes.bar(xData, StrategyOutput["日期序列"]["无杠杆收益率"].values, label="无杠杆收益率", color="steelblue")
        iRAxes.legend(loc="upper right")
        iAxes.set_xticks(xTicks)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc='upper left')
        iAxes.set_title("策略无杠杆表现")
        if hasBenchmark:
            iAxes = Fig.add_subplot(nRow, nCol, 3)
            iAxes.plot(xData, StrategyOutput["日期序列"]["相对净值"].values, label="相对净值", color="indianred", lw=2.5)
            iRAxes = iAxes.twinx()
            iRAxes.yaxis.set_major_formatter(yMajorFormatter)
            iRAxes.bar(xData, StrategyOutput["日期序列"]["相对收益率"].values, label="相对收益率", color="steelblue")
            iRAxes.legend(loc="upper right")
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc='upper left')
            iAxes.set_title("策略相对表现")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        HTML = self._formatStatistics().to_html()
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