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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.HistoryTest.HistoryTestModel import BaseModule
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import summaryStrategy, calcYieldSeq, calcLSYield
from QuantStudio.FactorDataBase.FactorDB import FactorTable
from QuantStudio.HistoryTest.SectionTest.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage

def cutDateTime(df, dts=None, start_dt=None, end_dt=None):
    if dts is not None: df = df.ix[dts]
    if start_dt is not None: df = df[df.index>=start_dt]
    if end_dt is not None: df = df[df.index<=end_dt]
    return df
def genAccountOutput(init_cash, cash_series, debt_series, account_value_series, debt_record, date_index):
    Output = {}
    # 以时间点为索引的序列
    Output["时间序列"] = pd.DataFrame(cash_series, columns=["现金"])
    Output["时间序列"]["负债"] = debt_series
    Output["时间序列"]["证券"] = account_value_series - (cash_series - debt_series)
    Output["时间序列"]["账户价值"] = account_value_series
    AccountEarnings = account_value_series.diff()
    AccountEarnings.iloc[0] = account_value_series.iloc[0] - init_cash
    Output["时间序列"]["收益"] = AccountEarnings
    PreAccountValue = np.r_[init_cash, account_value_series.values[:-1]]
    AccountReturn = AccountEarnings / np.abs(PreAccountValue)
    AccountReturn[AccountEarnings==0] = 0.0
    Output["时间序列"]["收益率"] = AccountReturn
    AccountReturn[np.isinf(AccountReturn)] = np.nan
    Output["时间序列"]["累计收益率"] = AccountReturn.cumsum()
    Output["时间序列"]["净值"] = (AccountReturn + 1).cumprod()
    DebtDelta = debt_record.groupby(by=["时间点"]).sum()["融资"]
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
        Output["日期序列"]["收益"] = AccountEarnings
        PreAccountValue = np.append(np.array(init_cash), AccountValueSeries.values[:-1])
        AccountReturn = AccountEarnings/np.abs(PreAccountValue)
        AccountReturn[AccountEarnings==0] = 0.0
        Output["日期序列"]["收益率"] = AccountReturn
        AccountReturn[np.isinf(AccountReturn)] = np.nan
        Output["日期序列"]["累计收益率"] = AccountReturn.cumsum()
        Output["日期序列"]["净值"] = (AccountReturn+1).cumprod()
        debt_record = debt_record.copy()
        debt_record["时间点"] = [iDateTime.date() for iDateTime in debt_record["时间点"]]
        DebtDelta = debt_record.groupby(by=["时间点"]).sum()["融资"]
        PreUnleveredValue = pd.Series(np.append(np.array(init_cash), (AccountValueSeries.values+DebtSeries.values)[:-1]), index=AccountEarnings.index)
        PreUnleveredValue[DebtDelta.index] += DebtDelta.clip(0, np.inf)
        UnleveredReturn = AccountEarnings/np.abs(PreUnleveredValue)
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
    InitCash = Float(1e9, arg_type="Double", label="初始资金", order=0, low=0.0, high=np.inf, single_step=0.00001, decimals=5)
    DeltLimit = Float(0.0, arg_type="Double", label="负债上限", order=1, low=0.0, high=np.inf, single_step=0.00001, decimals=5)
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(name="Account", sys_args=sys_args, **kwargs)
        self._Cash = np.array([])# 剩余现金, 等于时间点长度+1, >=0
        self._Debt = np.array([])# 负债, 等于时间点长度+1, >=0
        self._CashRecord = pd.DataFrame(columns=["时间点", "现金流", "备注"])# 现金流记录, 现金流入为正, 现金流出为负
        self._DebtRecord = pd.DataFrame(columns=["时间点", "融资", "备注"])# 融资记录, 增加负债为正, 减少负债为负
        self._TradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])# 交易记录
        self._Output = None# 缓存的输出结果
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        nDT = len(dts)
        self._Cash, self._Debt = np.zeros(nDT+1), np.zeros(nDT+1)
        self._Cash[0] = self.InitCash
        self._CashRecord = pd.DataFrame(columns=["时间点", "现金流"])
        self._DebtRecord = pd.DataFrame(columns=["时间点", "融资"])
        self._TradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])
        return super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
    def __QS_move__(self, idt, *args, **kwargs):# 先于策略运行
        iIndex = self._Model.DateTimeIndex
        self._Cash[iIndex+1] = self._Cash[iIndex]
        self._Debt[iIndex+1] = self._Debt[iIndex]
        return 0
    def __QS_after_move__(self, idt, *args, **kwargs):# 晚于策略运行
        return 0
    def __QS_end__(self):
        self._Output = {}
        return super().__QS_end__()
    def output(self):
        if self._Output: return self._Output
        CashSeries = self.getCashSeries()
        DebtSeries = self.getDebtSeries()
        AccountValueSeries = self.getAccountValueSeries()
        self._Output = genAccountOutput(self.InitCash, CashSeries, DebtSeries, AccountValueSeries, self._DebtRecord, self._Model.DateIndexSeries)
        self._Output["现金流记录"] = self._CashRecord
        self._Output["融资记录"] = self._DebtRecord
        self._Output["交易记录"] = self._TradingRecord
        return self._Output
    # 当前账户的剩余现金
    @property
    def Cash(self):
        return self._Cash[self._Model.DateTimeIndex+1]
    # 当前账户的负债
    @property
    def Debt(self):
        return self._Debt[self._Model.DateTimeIndex+1]
    # 当前账户可提取的现金, = Cash + 负债上限 - Debt
    @property
    def AvailableCash(self):
        return self._Cash[self._Model.DateTimeIndex+1] + max((self.DeltLimit - self._Debt[self._Model.DateTimeIndex+1], 0))
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
    # 抽取现金
    def fetchCash(self, target_cash, remark=""):
        iIndex = self._Model.DateTimeIndex
        Cash = min((target_cash, self.AvailableCash))
        DebtDelta = Cash - min((Cash, self._Cash[iIndex]))
        self._Debt[iIndex] += DebtDelta
        self._Cash[iIndex] -= min((Cash, self._Cash[iIndex]))
        self._CashRecord.loc[self._CashRecord.shape[0]] = (self._Model.DateTime, -Cash, remark)
        if DebtDelta>0:
            self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, DebtDelta, remark)
        return Cash
    # 增加现金
    def addCash(self, target_cash, remark=""):
        iIndex = self._Model.DateTimeIndex
        self._Cash[iIndex] += target_cash - min((target_cash, self._Debt[iIndex]))
        DebtDelta = -min((target_cash, self._Debt[iIndex]))
        self._Debt[iIndex] += DebtDelta
        self._CashRecord.loc[self._CashRecord.shape[0]] = (self._Model.DateTime, target_cash, remark)
        if DebtDelta<0:
            self._DebtRecord.loc[self._DebtRecord.shape[0]] = (self._Model.DateTime, DebtDelta, remark)
        return 0

class _Benchmark(__QS_Object__):
    """基准"""
    FactorTable = Instance(FactorTable, arg_type="FactorTable", label="因子表", order=0)
    PriceFactor = Enum(None, arg_type="SingleOption", label="价格因子", order=1)
    BenchmarkID = Enum(None, arg_type="SingleOption", label="基准ID", order=2)
    RebalanceDTs = List(dt.datetime, arg_type="DateTimeList", label="再平衡时点", order=3)
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
        self.add_trait("BenchmarkID", Enum(*self.FactorTable.getID(ifactor_name=self.PriceFactor), arg_type="SingleOption", label="基准ID", order=2))
# 策略基类
class Strategy(BaseModule):
    """策略基类"""
    Accounts = List(Account)# 策略所用到的账户
    FactorTables = List(FactorTable)# 策略所用到的因子表
    Benchmark = Instance(_Benchmark, arg_type="ArgObject", label="比较基准", order=0)
    def __init__(self, name, sys_args={}, **kwargs):
        super().__init__(name=name, sys_args=sys_args, **kwargs)
        self.ModelArgs = {}# 模型参数，即用户自定义参数
        self.UserData = {}# 用户数据存放
    def __QS_initArgs__(self):
        self.Benchmark = _Benchmark()
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        self.UserData = {}
        Rslt = ()
        for iAccount in self.Accounts: Rslt += iAccount.__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        Rslt += super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        Rslt += self.init()
        return Rslt+tuple(self.FactorTables)
    def __QS_move__(self, idt, *args, **kwargs):
        iTradingRecord = {iAccount.Name:iAccount.__QS_move__(idt) for iAccount in self.Accounts}
        Signal = self.genSignal(idt, iTradingRecord)
        self.trade(idt, iTradingRecord, Signal)
        for iAccount in self.Accounts: iAccount.__QS_after_move__(idt)
        return 0
    def __QS_end__(self):
        for iAccount in self.Accounts: iAccount.__QS_end__()
        return 0
    def getViewItems(self, context_name=""):
        Prefix = (context_name+"." if context_name else "")
        Groups, Context = [], {}
        for j, jAccount in enumerate(self.Accounts):
            jItems, jContext = jAccount.getViewItems(context_name=context_name+"_Account"+str(j))
            Groups.append(Group(*jItems, label=str(j)+"-"+jAccount.Name))
            Context.update(jContext)
        return ([Group(*Groups, orientation='horizontal', layout='tabbed', springy=True)], Context)
    # 可选实现
    def init(self):
        return ()
    # 可选实现, trading_record: {账户名: 交易记录, 比如: DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])}
    def genSignal(self, idt, trading_record):
        return None
    # 可选实现, trading_record: {账户名: 交易记录, 比如: DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])}
    def trade(self, idt, trading_record, signal):
        return 0
    def output(self):
        if self._Output: return self._Output
        for i, iAccount in enumerate(self.Accounts):
            iOutput = iAccount.output()
            if iOutput: self._Output[str(i)+"-"+iAccount.Name] = iOutput
        AccountValueSeries, CashSeries, DebtSeries, InitCash, DebtRecord = 0, 0, 0, 0, None
        for iAccount in self.Accounts:
            AccountValueSeries += iAccount.getAccountValueSeries()
            CashSeries += iAccount.getCashSeries()
            DebtSeries += iAccount.getDebtSeries()
            InitCash += iAccount.InitCash
            DebtRecord = iAccount.DebtRecord.append(DebtRecord)
        StrategyOutput = genAccountOutput(InitCash, CashSeries, DebtSeries, AccountValueSeries, DebtRecord, self._Model.DateIndexSeries)
        StrategyOutput["统计数据"].columns = ["策略表现", "无杠杆表现"]
        if self.Benchmark.FactorTable is not None:# 设置了基准
            BenchmarkPrice = self.Benchmark.FactorTable.readData(factor_names=[self.Benchmark.PriceFactor], dts=AccountValueSeries.index.tolist(), ids=[self.Benchmark.BenchmarkID]).iloc[0,:,0]
            BenchmarkOutput = pd.DataFrame(calcYieldSeq(wealth_seq=BenchmarkPrice.values), index=BenchmarkPrice.index, columns=["基准收益率"])
            BenchmarkOutput["基准累计收益率"] = BenchmarkOutput["基准收益率"].cumsum()
            BenchmarkOutput["基准净值"] = BenchmarkPrice / BenchmarkPrice.iloc[0]
            LYield = (StrategyOutput["日期序列"]["无杠杆收益率"].values if "时间序列" not in StrategyOutput else StrategyOutput["时间序列"]["无杠杆收益率"].values)
            BenchmarkOutput["相对收益率"] = calcLSYield(long_yield=LYield, short_yield=BenchmarkOutput["基准收益率"].values)# 再平衡时点的设置, TODO
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
    def genExcelReport(self, xl_book, sheet_name):
        xlBook = xw.Book(save_path)
        NewSheet = xlBook.sheets.add(name="占位表")
        for i, iModule in enumerate(self._Modules):
            iModule.__QS_genExcelReport__(xlBook)
        xlBook.app.display_alerts = False
        if xlBook.sheets.count>1: xlBook.sheets["占位表"].delete()
        xlBook.app.display_alerts = True
        xlBook.save()
        xlBook.app.quit()
        return 0
    def _formatStatistics(self):
        Stats = self._Output["Strategy"]["统计数据"]
        FormattedStats = pd.DataFrame(index=Stats.index, columns=Stats.columns, dtype="O")
        DateFormatFun = np.vectorize(lambda x: x.strftime("%Y-%m-%d"))
        IntFormatFun = np.vectorize(lambda x: ("%d" % (x, )))
        FloatFormatFun = np.vectorize(lambda x: ("%.2f" % (x, )))
        PercentageFormatFun = np.vectorize(lambda x: ("%.2f%%" % (x*100, )))
        FormattedStats.iloc[:2] = DateFormatFun(Stats.iloc[:2, :].values)
        FormattedStats.iloc[2] = IntFormatFun(Stats.iloc[:2, :].values)
        FormattedStats.iloc[3:6] = PercentageFormatFun(Stats.iloc[3:6, :].values)
        FormattedStats.iloc[6] = FloatFormatFun(Stats.iloc[6, :].values)
        FormattedStats.iloc[7:9] = PercentageFormatFun(Stats.iloc[7:9, :].values)
        FormattedStats.iloc[9:] = DateFormatFun(Stats.iloc[9:, :].values)
        return FormattedStats
    def genMatplotlibFig(self):
        nRow, nCol = 3, 3
        Fig = plt.figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        AxesGrid = gridspec.GridSpec(nRow, nCol)
        xData = np.arange(1, self._Output["统计数据"].shape[0]-1)
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"].index[:-2]]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        self._plotStatistics(plt.subplot(AxesGrid[0, 0]), xData, xTickLabels, self._Output["统计数据"]["年化超额收益率"].iloc[:-2], PercentageFormatter, self._Output["统计数据"]["胜率"].iloc[:-2], PercentageFormatter)
        self._plotStatistics(plt.subplot(AxesGrid[0, 1]), xData, xTickLabels, self._Output["统计数据"]["信息比率"].iloc[:-2], PercentageFormatter, None)
        self._plotStatistics(plt.subplot(AxesGrid[0, 2]), xData, xTickLabels, self._Output["统计数据"]["超额最大回撤率"].iloc[:-2], PercentageFormatter, None)
        self._plotStatistics(plt.subplot(AxesGrid[1, 0]), xData, xTickLabels, self._Output["统计数据"]["年化收益率"].iloc[:-2], PercentageFormatter, pd.Series(self._Output["统计数据"].loc["市场", "年化收益率"], index=self._Output["统计数据"].index[:-2], name="市场"), PercentageFormatter, False)
        self._plotStatistics(plt.subplot(AxesGrid[1, 1]), xData, xTickLabels, self._Output["统计数据"]["Sharpe比率"].iloc[:-2], FloatFormatter, pd.Series(self._Output["统计数据"].loc["市场", "Sharpe比率"], index=self._Output["统计数据"].index[:-2], name="市场"), FloatFormatter, False)
        self._plotStatistics(plt.subplot(AxesGrid[1, 2]), xData, xTickLabels, self._Output["统计数据"]["平均换手率"].iloc[:-2], PercentageFormatter, None)
        Axes = plt.subplot(AxesGrid[2, 0])
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, 0].values, label=str(self._Output["净值"].iloc[:, 0].name), color="r", alpha=0.6, lw=3)
        Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, -3].values, label=str(self._Output["净值"].iloc[:, -3].name), color="b", alpha=0.6, lw=3)
        Axes.plot(self._Output["净值"].index, self._Output["净值"]["市场"].values, label="市场", color="g", alpha=0.6, lw=3)
        Axes.legend(loc='best')
        Axes = plt.subplot(AxesGrid[2, 1])
        xData = np.arange(0, self._Output["净值"].shape[0])
        xTicks = np.arange(0, self._Output["净值"].shape[0], int(self._Output["净值"].shape[0]/8))
        xTickLabels = [self._Output["净值"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        Axes.plot(xData, self._Output["净值"]["L-S"].values, label="多空净值", color="r", alpha=0.6, lw=3)
        Axes.legend(loc='upper left')
        RAxes = Axes.twinx()
        RAxes.yaxis.set_major_formatter(PercentageFormatter)
        RAxes.bar(xData, self._Output["收益率"]["L-S"].values, label="多空收益率", color="b")
        RAxes.legend(loc="upper right")
        Axes.set_xticks(xTicks)
        Axes.set_xticklabels(xTickLabels)
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    def _repr_html_(self):
        HTML = self._formatStatistics().to_html()
        Pos = HTML.find(">")
        HTML = HTML[:Pos]+' align="center"'+HTML[Pos:]
        Fig = self.genMatplotlibFig()
        # figure 保存为二进制文件
        Buffer = BytesIO()
        plt.savefig(Buffer)
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML