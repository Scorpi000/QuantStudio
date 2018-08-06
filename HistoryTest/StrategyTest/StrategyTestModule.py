# -*- coding: utf-8 -*-
import os
import shutil

import numpy as np
import pandas as pd
from traits.api import ListStr, Enum, List, Int, Float, Str, Instance, Dict, on_trait_change
from traitsui.api import Item, Group, View

from QuantStudio import __QS_Error__
from QuantStudio.HistoryTest.HistoryTestModel import BaseModule
from QuantStudio.Tools import DateTimeFun
from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.StrategyTestFun import summaryStrategy

def cutDateTime(df, dts=None, start_dt=None, end_dt=None):
    if dts is not None:
        df = df.ix[dts]
    if start_dt is not None:
        df = df[df.index>=start_dt]
    if end_dt is not None:
        df = df[df.index<=end_dt]
    return df
def genAccountOutput(init_cash, cash_series, debt_series, account_value_series, debt_record, date_index):
    Output = {}
    # 以时间点为索引的序列
    Output["时间序列"] = pd.DataFrame(cash_series, columns=["现金"])
    Output["时间序列"]["负债"] = debt_series
    Output["时间序列"]["证券"] = account_value_series - (cash_series - debt_series)
    Output["时间序列"]["账户价值"] = account_value_series
    AccountEarnings = account_value_series.diff()
    AccountEarnings.iloc[0] = account_value_series.iloc[0]-init_cash
    Output["时间序列"]["收益"] = AccountEarnings
    PreAccountValue = np.append(np.array(init_cash), account_value_series.values[:-1])
    AccountReturn = AccountEarnings/np.abs(PreAccountValue)
    AccountReturn[AccountEarnings==0] = 0.0
    Output["时间序列"]["收益率"] = AccountReturn
    AccountReturn[np.isinf(AccountReturn)] = np.nan
    Output["时间序列"]["累计收益率"] = AccountReturn.cumsum()
    Output["时间序列"]["净值"] = (AccountReturn+1).cumprod()
    DebtDelta = debt_record.groupby(by=["时间点"]).sum()["融资"]
    PreUnleveredValue = pd.Series(np.append(np.array(init_cash), (account_value_series.values+debt_series.values)[:-1]), index=AccountEarnings.index)
    PreUnleveredValue[DebtDelta.index] += DebtDelta.clip(0, np.inf)
    UnleveredReturn = AccountEarnings/np.abs(PreUnleveredValue)
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
        self._Cash = np.zeros(dts.shape[0]+1)
        self._Cash[0] = self.InitCash
        self._Debt = np.zeros(dts.shape[0]+1)
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
        if self._Output:
            return self._Output
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

# 策略基类
class Strategy(BaseModule):
    """策略基类"""
    Accounts = List(Account)# 策略所用到的账户
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(name="Strategy", sys_args=sys_args, **kwargs)
        self.ModelArgs = {}# 模型参数，即用户自定义参数
        self.UserData = {}# 用户数据存放
    @property
    def Model(self):
        return self._Model
    def __QS_start__(self, mdl, dts=None, dates=None, times=None):
        self.UserData = {}
        Rslt = ()
        for iAccount in self.Accounts: Rslt += iAccount.__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        Rslt += super().__QS_start__(mdl=mdl, dts=dts, dates=dates, times=times)
        Rslt += self.init()
        return Rslt
    def __QS_move__(self, idt, *args, **kwargs):
        iTradingRecord = {iAccount.Name:iAccount.__QS_move__(idt) for iAccount in self.Accounts}
        Signal = self.genSignal(idt, iTradingRecord)
        self.trade(idt, iTradingRecord, Signal)
        for iAccount in self.Accounts:
            iAccount.__QS_after_move__(idt)
        return 0
    def __QS_end__(self):
        for iAccount in self.Accounts: iAccount.__QS_end__()
        return 0
    # 可选实现
    def init(self):
        return ()
    # 可选实现
    def genSignal(self, idt, trading_record):
        return None
    # 可选实现
    def trade(self, idt, trading_record, signal):
        return 0
    # 生成策略结果, 可选实现
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
        self._Output[self.Name] = genAccountOutput(InitCash, CashSeries, DebtSeries, AccountValueSeries, DebtRecord, self._Model.DateIndexSeries)
        return self._Output
    # 可选实现
    def genExcelReport(self, xl_book, sheet_name):
        xlBook = xw.Book(save_path)
        NewSheet = xlBook.sheets.add(name="占位表")
        for i,iModule in enumerate(self._Modules):
            iModule.__QS_genExcelReport__(xlBook)
        xlBook.app.display_alerts = False
        if xlBook.sheets.count>1:
            xlBook.sheets["占位表"].delete()
        xlBook.app.display_alerts = True
        xlBook.save()
        xlBook.app.quit()
        return 0
    def getViewItems(self, context_name=""):
        Prefix = (context_name+"." if context_name else "")
        Groups, Context = [], {}
        for j, jAccount in enumerate(self.Accounts):
            jItems, jContext = jAccount.getViewItems(context_name=context_name+"_Account"+str(j))
            Groups.append(Group(*jItems, label=str(j)+"-"+jAccount.Name))
            Context.update(jContext)
        return ([Group(*Groups, orientation='horizontal', layout='tabbed', springy=True)], Context)
# 策略报告基类, TODO: 完善各种统计指标
class Report(BaseModule):
    """策略报告基类"""
    def __init__(self, name, qs_env, sys_args={}):
        super().__init__(name, qs_env, sys_args)
        self.__QS_Type__ = "BaseReport"
        self._Output = None
    # 产生基准对冲参数
    def _genBenchmarkArgs(self, args=None):
        if self.QSEnv.DefaultTable is None:
            return QSArgs()
        DefaultFT = (self.QSEnv.getTable(args["基准数据源"]) if (args is not None) and (args["基准数据源"] in self.QSEnv.TableNames) else self.QSEnv.DefaultTable)
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(DefaultFT.getFactorMetaData(key="DataType")))
        if args is None:
            SysArgs = {"基准数据源":DefaultFT.Name,
                       "基准价格":searchNameInStrList(DefaultNumFactorList, ["价","Price","price"]),
                       "基准 ID":"无",
                       "再平衡时点":[]}
            ArgInfo = {}
            ArgInfo['基准数据源'] = {'type':'SingleOption','range':list(self.QSEnv.TableNames),'refresh':True,'order':0}
            ArgInfo['基准价格'] = {'type':'SingleOption','range':DefaultNumFactorList,'order':1}
            ArgInfo['基准 ID'] = {'type':'SingleOption','range':list(DefaultFT.getID())+['无'],'order':2}
            ArgInfo['再平衡时点'] = {'type':'DateList','order':3}
            return QSArgs(SysArgs, ArgInfo, self._onBenchmarkArgChanged)
        args._QS_MonitorChange = False
        args['基准数据源'] = DefaultFT.Name
        args.ArgInfo["基准数据源"]["range"] = list(self.QSEnv.TableNames)
        if args['基准价格'] not in DefaultNumFactorList:
            args['基准价格'] = searchNameInStrList(DefaultNumFactorList, ["价","Price","price"])
        args.ArgInfo["基准价格"]["range"] = DefaultNumFactorList
        if args['基准 ID'] not in DefaultFT.IDs:
            args['基准 ID'] = '无'
        args.ArgInfo["基准 ID"]["range"] = DefaultFT.IDs+['无']
        args._QS_MonitorChange = True
        return args
    def _onBenchmarkArgChanged(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="基准数据源"):# 数据源发生了变化
            Args["基准数据源"] = Value
            self._genBenchmarkArgs(args=Args)
            return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    # 生成日历统计参数及其初始值
    def _genCalendarArgs(self, args=None):
        if args is None:
            SysArgs = {"年度统计":False,
                       "月度统计":False,
                       "星期统计":False,
                       "日度统计":False,
                       "日度窗口":[0,0]}
            ArgInfo = {}
            ArgInfo['年度统计'] = {'type':'Bool','order':0}
            ArgInfo['月度统计'] = {'type':'Bool','order':1}
            ArgInfo["星期统计"] = {'type':'Bool','order':2}
            ArgInfo['日度统计'] = {'type':'Bool','order':3}
            ArgInfo['日度窗口'] = {'type':'ArgList','subarg_info':{"type":"Integer","min":0,"max":np.inf,"single_step":1},'order':4}
            return QSArgs(SysArgs, ArgInfo, None)
        return args
    # 生成滚动分析参数及其初始值
    def _genRollingAnalysisArgs(self, args=None):
        if args is None:
            SysArgs = {"滚动分析":False,
                       "最小窗口":252}
            ArgInfo = {}
            ArgInfo['滚动分析'] = {'type':'Bool','order':0}
            ArgInfo['最小窗口'] = {'Integer':'Integer','min':1,"max":np.inf,"single_step":1,'order':1}
            return QSArgs(SysArgs, ArgInfo, None)
        return args
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if args is None:
            SysArgs = {"基准对冲":self._genBenchmarkArgs(None),
                       "日历分析":self._genCalendarArgs(None),
                       "滚动分析":self._genRollingAnalysisArgs(None),
                       "统计日期序列":[]}
            ArgInfo = {}
            ArgInfo['基准对冲'] = {'type':'ArgSet','order':0}
            ArgInfo['日历分析'] = {'type':'ArgSet','order':1}
            ArgInfo['滚动分析'] = {'type':'ArgSet','order':2}
            ArgInfo['统计日期序列'] = {'type':'DateList','order':3,'visible':False}
            return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
        args._QS_MonitorChange = False
        args["基准对冲"] = self._genBenchmarkArgs(args.get("基准对冲"))
        args._QS_MonitorChange = True
        return args
    def output(self):
        if self._Output is not None:
            return self._Output
        AccountValueSeries, CashSeries, DebtSeries, InitCash, DebtRecord = 0, 0, 0, 0, None
        for iAccount in self.QSEnv.STM.Accounts.values():
            AccountValueSeries += iAccount.getAccountValueSeries()
            CashSeries += iAccount.getCashSeries()
            DebtSeries += iAccount.getDebtSeries()
            InitCash += iAccount.InitCash
            DebtRecord = iAccount.DebtRecord.append(DebtRecord)
        self._Output = genAccountOutput(InitCash, CashSeries, DebtSeries, AccountValueSeries, DebtRecord, self.QSEnv.STM.DateIndexSeries)
        return self._Output