# coding=utf-8
"""ETF账户(TODO)"""
import os
import shelve
from copy import deepcopy

import pandas as pd
import numpy as np

from QuantStudio.Tools.AuxiliaryFun import getFactorList,searchNameInStrList
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio.Tools import DateTimeFun
from QuantStudio import QSError, QSArgs, QSObject
from QuantStudio.HistoryTest.StrategyTest.StrategyTestModule import Account

def _getDefaultNontradableIDFilter(data_source=None,nonbuyable=True,qs_env=None):
    if data_source is None:
        return None
    with shelve.open(qs_env.SysArgs['LibPath']+os.sep+'IDFilter') as LibFile:
        if nonbuyable:
            DefaultNontradableIDFilter = LibFile.get('默认限买条件')
        else:
            DefaultNontradableIDFilter = LibFile.get('默认限卖条件')
    if DefaultNontradableIDFilter is not None:
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(DefaultNontradableIDFilter, data_source.FactorNames)
        if CompiledIDFilterStr is not None:
            return DefaultNontradableIDFilter
    AllFactorNames = set(data_source.FactorNames)
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

class ETFAccount(Account):
    """ETF账户"""
    def __init__(self, name, qs_env):
        super().__init__(name, qs_env)
        self.__QS_Type__ = "ETFAccount"
        # 继承自 Account 的属性
        #self._Cash = []# 剩余现金, 等于时间点长度+1, >=0
        #self._Debt = []# 负债, 等于时间点长度+1, >=0
        #self._CashRecord = pd.DataFrame(columns=["时间点", "现金流"])# 现金流记录, pd.DataFrame(columns=["日期", "时间点", "现金流"]), 现金流入为正, 现金流出为负
        #self._DebtRecord = pd.DataFrame(columns=["时间点", "融资"])# 融资记录, pd.DataFrame(columns=["日期", "时间点", "融资"]), 增加负债为正, 减少负债为负
        #self._TradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])# 交易记录
        self._Position = []# 仓位, [pd.Series(股票数, index=[ID])], 和时间戳等长
        self._PositionAmount = []# 当前的持仓金额, [pd.Series(金额, index=[ID])], 和时间戳等长
        self._OrderQueue = []# 当前接收到的订单队列, [pd.DataFrame(columns=["ID", "数量", "目标价"])], 账户每个时点从队列头部取出订单处理, 并在尾部添加空订单
        self._PRRecord = pd.DataFrame(columns=["时间点", "ID", "申赎类型", "数量"])# 申购赎回记录, 申赎类型: 申购, 赎回
        return
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(None, **kwargs)
        if self.QSEnv.DSs.isEmpty():
            return SysArgs
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        PriceFactor = searchNameInStrList(DefaultNumFactorList,['价','Price','price'])
        if (args is None) or ("数据源" not in args):
            nSysArgs = len(SysArgs)
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"最新价":PriceFactor,
                            "交易延迟":0,
                            "成交价":PriceFactor,
                            "买入限制":self._genLimitArgs(None, DefaultDS.Name, True),
                            "卖出限制":self._genLimitArgs(None, DefaultDS.Name, False),
                            "数据源":DefaultDS.Name})
            SysArgs.ArgInfo["最新价"] = {"type":"SingleOption","order":nSysArgs,"range":DefaultNumFactorList}
            SysArgs.ArgInfo["交易延迟"] = {"type":"Integer","order":nSysArgs+1,"min":0,"max":np.inf,"single_step":1}
            SysArgs.ArgInfo["成交价"] = {"type":"SingleOption","order":nSysArgs+2,"range":DefaultNumFactorList}
            SysArgs.ArgInfo["买入限制"] = {"type":"ArgSet","order":nSysArgs+3}
            SysArgs.ArgInfo["卖出限制"] = {"type":"ArgSet","order":nSysArgs+4}
            SysArgs.ArgInfo["数据源"] = {"type":"SingleOption","range":list(self.QSEnv.DSs.keys()),"order":nSysArgs+5,"refresh":True,"visible":False}
            SysArgs._QS_MonitorChange = True
            return SysArgs
        args._QS_MonitorChange = False
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        args["数据源"] = DefaultDS.Name
        if args["成交价"] not in DefaultNumFactorList:
            args["成交价"] = PriceFactor
        args.ArgInfo["成交价"]["range"] = DefaultNumFactorList
        if args["最新价"] not in DefaultNumFactorList:
            args["最新价"] = PriceFactor
        args.ArgInfo["最新价"]["range"] = DefaultNumFactorList
        args["买入限制"] = self._genLimitArgs(args["买入限制"], args["数据源"], nonbuyable=True)
        args["卖出限制"] = self._genLimitArgs(args["卖出限制"], args["数据源"], nonbuyable=False)
        args._QS_MonitorChange = True
        return args
    def __QS_onSysArgChanged__(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="数据源"):# 数据源发生了变化
            Args["数据源"] = Value
            self.__QS_genSysArgs__(args=Args, **kwargs)
            return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def _genLimitArgs(self, args, ds_name, nonbuyable=True):
        DefaultDS = self.QSEnv.DSs[ds_name]
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        PriceFactor = searchNameInStrList(DefaultNumFactorList,['价','Price','price'])
        if args is None:
            Args = {"禁止条件":_getDefaultNontradableIDFilter(DefaultDS,nonbuyable,self.QSEnv),
                    "交易费率":0.003,
                    "成交额":"不限制",
                    "最小单位":0,
                    "单位价格":PriceFactor}
            ArgInfo = {"禁止条件":{"type":"IDFilter","order":0,"factor_list":DefaultDS.FactorNames},
                       "交易费率":{"type":"Double","order":1,"min":0,"max":1,"single_step":0.0001,"decimals":4},
                       "成交额":{"type":"SingleOption","order":2,"refresh":True,"range":["不限制"]+DefaultNumFactorList},
                       "最小单位":{"type":"Integer","order":3,"min":0,"max":np.inf,"single_step":1},
                       "单位价格":{"type":"SingleOption","order":4,"range":DefaultNumFactorList}}
            return QSArgs(Args, ArgInfo, self._onLimitSysArgChanged)
        args._QS_MonitorChange = False
        if not set(args.ArgInfo["禁止条件"]["factor_list"]).issubset(set(DefaultDS.FactorNames)):
            args["禁止条件"] = _getDefaultNontradableIDFilter(DefaultDS,nonbuyable,self.QSEnv)
        args.ArgInfo["禁止条件"]["factor_list"] = DefaultDS.FactorNames
        if args["成交额"] not in DefaultNumFactorList:
            args["成交额"] = "不限制"
            args.ArgInfo.pop("成交额限比", None)
        args.ArgInfo["成交额"]["range"] = ["不限制"]+DefaultNumFactorList
        if args["单位价格"] not in DefaultNumFactorList:
            args["单位价格"] = PriceFactor
        args.ArgInfo["单位价格"]["range"] = DefaultNumFactorList
        args._QS_MonitorChange = True
        return args
    def _onLimitSysArgChanged(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="成交额"):
            Args["成交额"] = Value
            if Value=="不限制":
                Args.pop("成交额限比")
                Args.ArgInfo.pop("成交额限比")
            else:
                Args["成交额限比"] = 0.1
                Args.ArgInfo["成交额限比"] = {"type":"Double","order":1.5,"min":0,"max":np.inf,"single_step":0.01}
            return True
        return super(QSObject, self).__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_start__(self):
        super().__QS_start__()
        self._Position = [pd.Series([],dtype="float")]
        self._PositionAmount = [pd.Series([],dtype="float")]
        self._OrderQueue = [pd.DataFrame(columns=["ID", "数量", "目标价"]) for i in range(max((self.SysArgs["交易延迟"]+1,1)))]
        self._DS = self.QSEnv.DSs[self.SysArgs["数据源"]]
        #self._NonbuyableID = None# 用于缓存禁止买入证券列表
        #self._NonsellableID = None# 用于缓存禁止卖出证券列表
        #self._LastPrice = None# 用于缓存当前的最新价
        return 0
    def __QS_move__(self, idt, timestamp, *args, **kwargs):
        super().__QS_move__(idt, timestamp, *args, **kwargs)
        # 更新当前的账户信息
        self._Position.append(self._Position[-1])
        # 撮合成交
        Orders = self._OrderQueue.pop(0)
        TradingRecord = self._matchMarketOrder(Orders)
        TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)),
                                     columns=self._TradingRecord.columns)
        self._OrderQueue.append(pd.DataFrame(columns=["ID", "数量", "目标价"]))
        # 更新当前的账户信息
        Position = self._Position[-1]
        self._PositionAmount.append(Position*self.LastPrice[Position.index])
        self._TradingRecord = self._TradingRecord.append(TradingRecord)
        return TradingRecord
    def __QS_end__(self):
        self._DS = None
        self._OrderQueue = []
        self._Output = None# 缓存的输出结果
        return 0
    # 当前账户价值
    @property
    def AccountValue(self):
        return super().AccountValue + self._PositionAmount[-1].sum()
    # 当前账户的持仓
    @property
    def Position(self):
        return self._Position[-1].copy()
    # 当前账户的持仓金额
    @property
    def PositionAmount(self):
        return self._PositionAmount[-1].copy()
    # 当前账户的投资组合
    @property
    def Portfolio(self):
        return self.PositionAmount/self.AccountValue
    # 本账户支持交易的证券 ID
    @property
    def IDs(self):
        return self.QSEnv.DSs[self.SysArgs["数据源"]].IDs
    # 本账户当前不能执行买入的证券 ID
    @property
    def NonbuyableIDs(self):
        return self.getNonbuyableID(self.QSEnv.STM.Date)
    # 本账户当前不能执行卖出的证券 ID
    @property
    def NonsellableIDs(self):
        return self.getNonsellableID(self.QSEnv.STM.Date)
    # 本账户是否可以卖空
    @property
    def isShortAllowed(self):
        return False
    # 当前最新价
    @property
    def LastPrice(self):
        return self._DS.getFactorData(ifactor_name=self.SysArgs["最新价"], dates=[self.QSEnv.STM.Date]).iloc[0]
    # 最小买入数量, 0 表示无限分割
    @property
    def MinBuyNum(self):
        UnitPrice = self._DS.getFactorData(ifactor_name=[self.SysArgs["买入限制"]["单位价格"]], dates=[CurDate]).iloc[0]
        return self.SysArgs["买入限制"]["最小单位"]*UnitPrice/self.LastPrice
    # 单位买入价格
    @property
    def BuyUnitPrice(self):
        return self.LastPrice
    # 单位买入交易费
    @property
    def BuyUnitFee(self):
        return self.LastPrice*self.SysArgs["买入限制"]["交易费率"]
    # 最小卖出数量, 0 表示无限分割
    @property
    def MinSellNum(self):
        UnitPrice = self._DS.getFactorData(ifactor_name=[self.SysArgs["卖出限制"]["单位价格"]], dates=[CurDate]).iloc[0]
        return self.SysArgs["卖出限制"]["最小单位"]*UnitPrice/self.LastPrice
    # 单位卖出价格
    @property
    def SellUnitPrice(self):
        return self.LastPrice
    # 单位卖出交易费
    @property
    def SellUnitFee(self):
        return self.LastPrice*self.SysArgs["卖出限制"]["交易费率"]
    # 提取证券
    def fetchEquity(self, target_id, num):
        Position = self._Position[-1].get(target_id, 0)
        EquityNum = min((num, Position))
        if EquityNum<=0:
            return 0
        self._Position[-1][target_id] -= Position - EquityNum
        self._PositionAmount[-1][target_id] = (1-EquityNum/Position)*self._PositionAmount[-1][target_id]
        self._EquitySOA.loc[self._EquitySOA.shape[0]] = (self.QSEnv.STM.Date, self.QSEnv.STM.Timestamp, target_id, -EquityNum)
        return EquityNum
    # 增加证券
    def addEquity(self, target_id, num):
        if num<=0:
            return 0
        self._Position[-1][target_id] = self._Position[-1].get(target_id, 0) + num
        LastPrice = self.LastPrice.get(target_id)
        self._PositionAmount[-1][target_id] = self._PositionAmount[-1].get(target_id, 0) + LastPrice*num
        self._EquitySOA.loc[self._EquitySOA.shape[0]] = (self.QSEnv.STM.Date, self.QSEnv.STM.Timestamp, target_id, num)
        return 0
    # 申购 ETF, portfolio: Series(数量, [ID]), TODO
    def purchaseETF(self, target_id, num, portfolio):
        pass
    # 赎回 ETF, 返回: Series(数量, [ID]), TODO
    def redeemETF(self, target_id, num):
        pass
    # 当前的申购赎回清单, TODO
    @property
    def PCF(self):
        pass
    # 当前的参考资产净值, TODO
    @property
    def IOPV(self):
        pass
    # 预估现金差额, TODO
    @property
    def EstCashBalance(self):
        pass
    # 获取持仓的历史序列, 以时间戳为索引
    def getPositionSeries(self, start_tms=None, end_tms=None):
        TMSSeries = self.QSEnv.STM.TimestampSeries
        StartIndex = (1+TMSSeries.index(start_tms) if start_tms is not None else 1)
        EndIndex = (1+TMSSeries.index(end_tms) if end_tms is not None else len(TMSSeries))
        return pd.DataFrame(self._Position[StartIndex:EndIndex+1], index=TMSSeries[StartIndex-1:EndIndex])
    # 获取持仓证券的金额历史序列, 以时间戳为索引, 返回: pd.DataFrame(持仓金额, index=[时间戳], columns=[ID])
    def getPositionAmountSeries(self, start_tms=None, end_tms=None):
        TMSSeries = self.QSEnv.STM.TimestampSeries
        StartIndex = (1+TMSSeries.index(start_tms) if start_tms is not None else 1)
        EndIndex = (1+TMSSeries.index(end_tms) if end_tms is not None else len(TMSSeries))
        return pd.DataFrame(self._PositionAmount[StartIndex:EndIndex+1], index=TMSSeries[StartIndex-1:EndIndex])
    # 计算持仓投资组合, 返回: pd.DataFrame(权重, index=[日期], columns=[ID])
    def getPortfolioSeries(self, start_tms=None, end_tms=None):
        PositionAmountSeries = self.getPositionAmountSeries(start_tms=start_tms, end_tms=end_tms)
        AccountValue = self.getAccountValueSeries(start_tms=start_tms, end_tms=end_tms)
        AccountValue[AccountValue==0] += 0.0001
        return (PositionAmount.T/AccountValue).T
    # 获取账户价值的历史序列, 以时间戳为索引
    def getAccountValueSeries(self, start_tms=None, end_tms=None):
        CashSeries = self.getCashSeries(start_tms=start_tms, end_tms=end_tms)
        PositionAmountSeries = self.getPositionAmountSeries(start_tms=start_tms, end_tms=end_tms).sum(axis=1)
        DebtSeries = self.getDebtSeries(start_tms=start_tms, end_tms=end_tms)
        return CashSeries+PositionAmountSeries-DebtSeries
    # 获得仓位分布的历史序列, 以日期为索引, 返回: pd.DataFrame(股票数, index=[日期], columns=[ID])
    def getPositionDateSeries(self, start_date=None, end_date=None):
        Dates = DateTimeFun.cutDate(self.QSEnv.STM.DateSeries, start_date, end_date)
        EndIndex = self.QSEnv.STM.getDateEndIndex(Dates)
        Rslt = pd.DataFrame(self._Position[1:])
        Rslt = Rslt.iloc[EndIndex]
        Rslt.index = Dates
        return Rslt
    # 获得持仓金额分布的历史序列, 以日期为索引, 返回: pd.DataFrame(持仓金额, index=[日期], columns=[ID])
    def getPositionAmountDateSeries(self, start_date=None, end_date=None):
        Dates = DateTimeFun.cutDate(self.QSEnv.STM.DateSeries, start_date, end_date)
        EndIndex = self.QSEnv.STM.getDateEndIndex(Dates)
        Rslt = pd.DataFrame(self._PositionAmount[1:])
        Rslt = Rslt.iloc[EndIndex]
        Rslt.index = Dates
        return Rslt
    # 获得持仓投资组合的历史序列, 以日期为索引, 返回: pd.DataFrame(持仓金额, index=[日期], columns=[ID])
    def getPortfolioDateSeries(self, start_date=None, end_date=None):
        Dates = DateTimeFun.cutDate(self.QSEnv.STM.DateSeries, start_date, end_date)
        EndIndex = self.QSEnv.STM.getDateEndIndex(Dates)
        Rslt = self.getPortfolioSeries()
        Rslt = Rslt.iloc[EndIndex]
        Rslt.index = Dates
        return Rslt
    # 获取指定日期序列的账户价值
    def getAccountValueDateSeries(self, start_date=None, end_date=None):
        Dates = DateTimeFun.cutDate(self.QSEnv.STM.DateSeries, start_date, end_date)
        EndIndex = self.QSEnv.STM.getDateEndIndex(Dates)
        Rslt = self.getAccountValueSeries()
        Rslt = Rslt.iloc[EndIndex]
        Rslt.index = Dates
        return Rslt
    # 获取不能买入的证券 ID
    def getNonbuyableID(self, date):
        if self.SysArgs["买入限制"]['禁止条件'] is None:
            return []
        return self.QSEnv.DSs[self.SysArgs["数据源"]].getID(date, is_filtered=True, id_filter_str=self.SysArgs["买入限制"]['禁止条件'])
    # 获取不能卖出的证券ID
    def getNonsellableID(self, date):
        if self.SysArgs["卖出限制"]['禁止条件'] is None:
            return []
        return self.QSEnv.DSs[self.SysArgs["数据源"]].getID(date, is_filtered=True, id_filter_str=self.SysArgs["卖出限制"]['禁止条件'])
    # 执行给定数量的证券委托单, target_id: 目标证券 ID, num: 待买卖的数量, target_price: nan 表示市价单, 
    # combined_order: 组合订单, DataFrame(index=[ID],columns=[数量, 目标价])
    # 基本的下单函数, 必须实现, 目前只实现了市价单
    def order(self, target_id=None, num=0, target_price=np.nan, combined_order=None):
        Orders = self._OrderQueue[-1]
        if target_id is not None:
            Orders.loc[Orders.shape[0]] = (target_id, num, target_price)
        if combined_order is not None:
            combined_order = combined_order.reset_index()
            combined_order.columns = ["ID", "数量", "目标价"]
            combined_order.index = np.arange(Orders.shape[0], Orders.shape[0]+combined_order.shape[0])
            Orders = Orders.append(combined_order)
        self._OrderQueue[-1] = Orders
        return 0
    # 执行给定金额的证券委托单, target_id: 目标证券 ID, amount: 待买卖的金额, target_price: nan 表示市价单, 
    # combined_order: 组合订单, DataFrame(index=[ID],columns=[金额, 目标价])
    # 非基本的下单函数, 以最新价将金额转换成数量, 目前只实现了市价单
    def orderAmount(self, target_id=None, amount=0, target_price=np.nan, combined_order=None):
        Orders = self._OrderQueue[-1]
        LastPrice = self.LastPrice
        if target_id is not None:
            iLastPrice = LastPrice.get(target_id)
            if (iLastPrice is not None) and (iLastPrice>0):
                Orders.loc[Orders.shape[0]] = (target_id, amount/iLastPrice, target_price)
        if combined_order is not None:
            combined_order["金额"] = combined_order["金额"]/LastPrice[combined_order.index]
            combined_order = combined_order.reset_index()
            combined_order.columns = ["ID", "数量", "目标价"]
            combined_order.index = np.arange(Orders.shape[0], Orders.shape[0]+combined_order.shape[0])
            Orders = Orders.append(combined_order)
        self._OrderQueue[-1] = Orders
        return 0
    # 撮合成交市价单, 订单执行方式是先卖后买, 可用现金或股票不足时实际成交数量将少于目标数量
    def _matchMarketOrder(self, orders):
        Orders = orders[pd.isnull(orders["目标价"])]
        if Orders.shape[0]==0:
            return []
        Orders = Orders.groupby(by=["ID"]).sum()["数量"]
        Orders = Orders[Orders!=0]
        # 提取当前信息
        CurDate = self.QSEnv.STM.Date
        CurTimestamp = self.QSEnv.STM.Timestamp
        AvailableCash = self.AvailableCash
        CashChanged = 0.0
        Position = self.Position
        Position = Position.append(pd.Series(0.0,index=Orders.index.difference(Position.index)))
        TradePrice = self._DS.getFactorData(ifactor_name=self.SysArgs["成交价"], dates=[CurDate]).iloc[0]
        TradingRecord = []
        # 先执行卖出交易
        SellOrders = -Orders[Orders<0]
        if SellOrders.shape[0]>0:
            SellPrice = TradePrice[SellOrders.index]
            SellLimit = pd.Series(np.zeros(SellOrders.shape[0])+np.inf,index=SellOrders.index)
            SellLimit[pd.isnull(SellPrice)] = 0.0# 成交价缺失的不能卖出
            if self.SysArgs["卖出限制"]["禁止条件"] is not None:
                SellLimit[self._DS.getIDMask(CurDate, self.SysArgs["卖出限制"]["禁止条件"])[SellOrders.index]] = 0.0# 满足禁止条件的不能卖出
            if self.SysArgs["卖出限制"]["成交额"]!="不限制":# 成交额缺失的不能卖出
                MaxAmount = self._DS.getFactorData(ifactor_name=self.SysArgs["卖出限制"]["成交额"], 
                                                   ids=list(SellOrders.index), dates=[CurDate]).iloc[0]
                MaxAmount = MaxAmount * self.SysArgs["卖出限制"]["成交额限比"]
                SellLimit = np.minimum(SellLimit, MaxAmount)
            SellAmounts = np.minimum(np.minimum(SellLimit, Position[SellOrders.index]*SellPrice), SellOrders*SellPrice)
            if self.SysArgs["卖出限制"]['最小单位']!=0.0:
                TradeUnitPrice = self._DS.getFactorData(ifactor_name=[self.SysArgs["卖出限制"]["单位价格"]], 
                                                        ids=list(SellOrders.index), dates=[CurDate]).iloc[0]
                SellAmounts = (SellAmounts/(self.SysArgs["卖出限制"]['最小单位']*TradeUnitPrice)).astype("int")*self.SysArgs["卖出限制"]['最小单位']*TradeUnitPrice
            SellFees = SellAmounts*self.SysArgs["卖出限制"]["交易费率"]
            SellNums = SellAmounts/SellPrice
            SellNums = SellNums[SellNums>0]
            Position[SellNums.index] -= SellNums
            CashChanged += SellAmounts.sum() - SellFees.sum()
            AvailableCash += CashChanged
            TradingRecord.extend(list(zip([CurDate]*SellNums.shape[0], [CurTimestamp]*SellNums.shape[0], 
                                          SellNums.index, -SellNums, SellPrice[SellNums.index], SellFees[SellNums.index],
                                          (SellAmounts-SellFees)[SellNums.index], ["close"]*SellNums.shape[0])))
        # 再执行买入交易
        BuyOrders = Orders[Orders>0]
        if BuyOrders.shape[0]>0:
            BuyPrice = TradePrice[BuyOrders.index]
            BuyLimit = pd.Series(np.zeros(BuyOrders.shape[0])+np.inf, index=BuyOrders.index)
            BuyLimit[pd.isnull(BuyPrice)] = 0.0# 成交价缺失的不能买入
            if self.SysArgs["买入限制"]["禁止条件"] is not None:
                BuyLimit[self._DS.getIDMask(CurDate, self.SysArgs["买入限制"]["禁止条件"])[BuyOrders.index]] = 0.0# 满足禁止条件的不能卖出
            if self.SysArgs["买入限制"]["成交额"]!="不限制":# 成交额缺失的不能买入
                MaxAmount = self._DS.getFactorData(ifactor_name=self.SysArgs["买入限制"]["成交额"], 
                                                   ids=list(BuyOrders.index), dates=[CurDate]).iloc[0]
                MaxAmount = MaxAmount * self.SysArgs["买入限制"]["成交额限比"]
                BuyLimit = np.minimum(BuyLimit, MaxAmount)
            BuyAmounts = np.minimum(BuyLimit, BuyOrders*BuyPrice)
            TotalBuyAmounts = BuyAmounts.sum()
            if TotalBuyAmounts>0:
                BuyAmounts = min((TotalBuyAmounts*(1+self.SysArgs["买入限制"]["交易费率"]), max((0, AvailableCash))))*BuyAmounts/TotalBuyAmounts/(1+self.SysArgs["买入限制"]["交易费率"])
                if self.SysArgs["买入限制"]['最小单位']!=0.0:
                    TradeUnitPrice = self._DS.getFactorData(ifactor_name=[self.SysArgs["买入限制"]["单位价格"]], 
                                                            ids=list(BuyOrders.index), dates=[CurDate]).iloc[0]
                    BuyAmounts = (BuyAmounts/(self.SysArgs["买入限制"]['最小单位']*TradeUnitPrice)).astype("int")*self.SysArgs["买入限制"]['最小单位']*TradeUnitPrice
                BuyFees = BuyAmounts*self.SysArgs["买入限制"]["交易费率"]
                BuyNums = BuyAmounts/BuyPrice
                BuyNums = BuyNums[BuyNums>0]
                Position[BuyNums.index] += BuyNums
                CashChanged -= BuyAmounts.sum() + BuyFees.sum()
                TradingRecord.extend(list(zip([CurDate]*BuyNums.shape[0], [CurTimestamp]*BuyNums.shape[0], 
                                              BuyNums.index, BuyNums, BuyPrice[BuyNums.index], BuyFees[BuyNums.index],
                                              -(BuyAmounts+BuyFees)[BuyNums.index], ["open"]*BuyNums.shape[0])))
        # 更新账户信息
        self._Debt[-1] += max((- CashChanged - self.Cash, 0))
        self._Cash[-1] -= min((- CashChanged, self.Cash))
        Position = Position[Position.abs()>1e-6]
        self._Position[-1] = Position
        return TradingRecord
# 给定金额的证券委托单, target_id: 目标证券 ID, amount: 待买卖的金额, target_price: None 表示市价单, 
# combined_order: 组合订单, DataFrame(index=[ID],columns=[金额, 目标价]), 如果不为 None, 则忽略前面三个参数
# 返回 (Error, 实际成交金额)
#def orderByAmount(target_id=None, amount=0, target_price=None, combined_order=None):
    #pass
# 给定目标比例的证券委托单, target_id: 目标证券 ID, target_pct: 目标比例, target_price: None 表示市价单, 
# combined_order: 组合订单, DataFrame(index=[ID],columns=[金额, 目标价]), 如果不为 None, 则忽略前面三个参数
# 返回 (Error, 实际成交金额)
#def orderByTargetPct(target_id, target_pct, target_price=None):
    #pass