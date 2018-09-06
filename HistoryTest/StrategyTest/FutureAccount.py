# coding=utf-8
"""期货账户(TODO)"""
import os
from copy import deepcopy

import pandas as pd
import numpy as np
from traits.api import Enum, Either, List, ListStr, Int, Float, Str, Bool, Dict, Instance, on_trait_change

from QuantStudio.Tools.AuxiliaryFun import getFactorList, searchNameInStrList
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio.Tools import DateTimeFun
from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.HistoryTest.StrategyTest.StrategyTestModule import Account, cutDateTime
from QuantStudio.HistoryTest.StrategyTest.StockAccount import _TradeLimit, _BarFactorMap, _TickFactorMap

# 基于 Bar 数据的股票账户
# 行情因子表: 开盘价(非必需), 最高价(非必需), 最低价(非必需), 最新价, 成交量(非必需). 最新价用于记录账户价值变化; 成交价: 用于模拟市价单的成交.
# 复权因子表: 复权因子或者每股送转(日期索引为股权登记日), 每股派息(税后, 日期索引为股权登记日), 派息日(日期索引为股权登记日), 红股上市日(日期索引为股权登记日)
# 市价单根据当前时段的成交价和成交量的情况成交; 限价单根据最高价, 最低价和成交量的情况成交, 假定成交量在最高价和最低价之间的分布为均匀分布, 如果没有指定最高价和最低价, 则最高价和最低价相等且等于成交价
# TODO: 加入交易状态, 涨跌停信息
class TimeBarAccount(Account):
    """基于 Bar 数据的期货账户"""
    Delay = Bool(True, arg_type="Bool", label="交易延迟", order=2)
    TargetIDs = ListStr(arg_type="IDList", label="目标ID", order=3)
    Multiplier = Float(300, arg_type="Double", label="合约乘数", order=4)
    InitMarginRate = Float(0.15, arg_type="Double", label="初始保证金率", order=5)
    MaintenanceMarginRate = Float(0.08, arg_type="Double", label="维持保证金率", order=5)
    BuyLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="买入限制", order=6)
    SellLimit = Instance(_TradeLimit, allow_none=False, arg_type="ArgObject", label="卖出限制", order=7)
    MarketFactorMap = Instance(_BarFactorMap, arg_type="ArgObject", label="行情因子", order=8)
    def __init__(self, market_ft, sys_args={}, **kwargs):
        # 继承自 Account 的属性
        #self._Cash = np.array([])# 剩余现金, 等于时间点长度+1, >=0
        #self._Debt = np.array([])# 负债, 等于时间点长度+1, >=0
        #self._CashRecord = pd.DataFrame(columns=["时间点", "现金流"])# 现金流记录, pd.DataFrame(columns=["日期", "时间点", "现金流"]), 现金流入为正, 现金流出为负
        #self._DebtRecord = pd.DataFrame(columns=["时间点", "融资"])# 融资记录, pd.DataFrame(columns=["日期", "时间点", "融资"]), 增加负债为正, 减少负债为负
        #self._TradingRecord = pd.DataFrame(columns=["时间点", "ID", "买卖数量", "价格", "交易费", "现金收支", "类型"])# 交易记录
        self._IDs = []# 本账户支持交易的证券 ID, []
        self._Position = np.array([])# 仓位, array(index=dts+1, columns=self._IDs)
        self._PositionAmount = np.array([])# 持仓金额, array(index=dts+1, columns=self._IDs)
        self._NominalAmount = np.array([])# 持仓名义金额, array(index=dts+1, columns=self._IDs)
        self._Orders = pd.DataFrame(columns=["ID", "数量", "目标价"])# 当前接收到的订单
        self._LastPrice = np.array([])# 最新价, array(len(self._IDs))
        self._PositionCostPrice = np.array([])# 当前持仓的成本价, 由于计算保证金的变化, array(len(self._IDs))
        self._MarketFT = market_ft# 行情因子表对象
        super().__init__(sys_args=sys_args, **kwargs)
        self.Name = "FutureAccount"
    def __QS_initArgs__(self):
        self.MarketFactorMap = _BarFactorMap(self._MarketFT)
        self.AdjustFactorMap = _AdjustFactorMap(self._AdjustFT)
        self.BuyLimit = _TradeLimit(direction="Buy")
        self.SellLimit = _TradeLimit(direction="Sell")
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(args, **kwargs)
        if self.QSEnv.DefaultTable is None:
            return SysArgs
        DefaultTable = (self.QSEnv.getTable(args["因子表"]) if (args is not None) and (args.get("因子表") in self.QSEnv.TableNames) else self.QSEnv.DefaultTable)
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(DefaultTable.getFactorMetaData(key="DataType")))
        PriceFactor = searchNameInStrList(DefaultNumFactorList,['价','Price','price'])
        if (args is None) or ("因子表" not in args):
            nSysArgs = len(SysArgs)
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"最新价":PriceFactor,
                            "交易延迟":1,
                            "成交价":PriceFactor,
                            "合约乘数":1.0,
                            "保证金率":0.15,
                            "买入限制":self._genLimitArgs(None, DefaultTable.Name, True),
                            "卖出限制":self._genLimitArgs(None, DefaultTable.Name, False),
                            "因子表":DefaultTable.Name})
            SysArgs.ArgInfo["最新价"] = {"type":"SingleOption","order":nSysArgs,"range":DefaultNumFactorList}
            SysArgs.ArgInfo["交易延迟"] = {"type":"Integer","order":nSysArgs+1,"min":0,"max":np.inf,"single_step":1}
            SysArgs.ArgInfo["成交价"] = {"type":"SingleOption","order":nSysArgs+2,"range":DefaultNumFactorList}
            SysArgs.ArgInfo["合约乘数"] = {"type":"Double","order":nSysArgs+3,"min":0,"max":np.inf,"single_step":1,"decimals":0}
            SysArgs.ArgInfo["保证金率"] = {"type":"Double","order":nSysArgs+4,"min":0,"max":1,"single_step":0.0001,"decimals":4}
            SysArgs.ArgInfo["买入限制"] = {"type":"ArgSet","order":nSysArgs+5}
            SysArgs.ArgInfo["卖出限制"] = {"type":"ArgSet","order":nSysArgs+6}
            SysArgs.ArgInfo["因子表"] = {"type":"SingleOption","order":nSysArgs+7,"range":list(self.QSEnv.TableNames),"refresh":True,"visible":False}
            SysArgs._QS_MonitorChange = True
            return SysArgs
        args._QS_MonitorChange = False
        args.ArgInfo["因子表"]["range"] = list(self.QSEnv.TableNames)
        args["因子表"] = DefaultTable.Name
        if args["成交价"] not in DefaultNumFactorList:
            args["成交价"] = PriceFactor
        args.ArgInfo["成交价"]["range"] = DefaultNumFactorList
        if args["最新价"] not in DefaultNumFactorList:
            args["最新价"] = PriceFactor
        args.ArgInfo["最新价"]["range"] = DefaultNumFactorList
        args["买入限制"] = self._genLimitArgs(args["买入限制"], args["因子表"], nonbuyable=True)
        args["卖出限制"] = self._genLimitArgs(args["卖出限制"], args["因子表"], nonbuyable=False)
        args._QS_MonitorChange = True
        return args
    def _genLimitArgs(self, args, table_name, nonbuyable=True):
        DefaultTable = self.QSEnv.getTable(table_name)
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(dict(DefaultTable.getFactorMetaData(key="DataType")))
        if args is None:
            Args = {"禁止条件":_getDefaultNontradableIDFilter(DefaultTable,nonbuyable,self.QSEnv),
                    "交易费率":0.003,
                    "成交量":"不限制",
                    "最小单位":0}
            ArgInfo = {"禁止条件":{"type":"IDFilter","order":0,"factor_list":DefaultTable.FactorNames},
                       "交易费率":{"type":"Double","order":1,"min":0,"max":1,"single_step":0.0001,"decimals":4},
                       "成交量":{"type":"SingleOption","order":2,"refresh":True,"range":["不限制"]+DefaultNumFactorList},
                       "最小单位":{"type":"Integer","order":3,"min":0,"max":np.inf,"single_step":1}}
            return QSArgs(Args, ArgInfo, self._onLimitSysArgChanged)
        args._QS_MonitorChange = False
        if not set(args.ArgInfo["禁止条件"]["factor_list"]).issubset(set(DefaultTable.FactorNames)):
            args["禁止条件"] = _getDefaultNontradableIDFilter(DefaultTable,nonbuyable,self.QSEnv)
        args.ArgInfo["禁止条件"]["factor_list"] = DefaultTable.FactorNames
        args.ArgInfo["成交量"]["range"] = ["不限制"]+DefaultNumFactorList
        args._QS_MonitorChange = True
        return args
    def _onLimitSysArgChanged(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="成交量"):
            Args["成交量"] = Value
            if Value=="不限制":
                Args.pop("成交量限比")
                Args.ArgInfo.pop("成交量限比")
            else:
                Args["成交量限比"] = 0.1
                Args.ArgInfo["成交量限比"] = {"type":"Double","order":1.5,"min":0,"max":np.inf,"single_step":0.01}
            return True
        return super(QSObject, self).__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_start__(self, dts=None, dates=None, times=None):
        super().__QS_start__(dts=dts, dates=dates, times=times)
        self._IDs = list(self.QSEnv.getTable(self.SysArgs["因子表"]).getID(ifactor_name=self._SysArgs["成交价"]))
        nDT = dts.shape[0]
        nID = len(self._IDs)
        #self._Cash = np.zeros(nDT+1)
        #self._Debt = np.zeros(nDT+1)
        self._Position = np.zeros((nDT+1, nID))
        self._PositionAmount = np.zeros((nDT+1, nID))
        self._NominalAmount = np.zeros((nDT+1, nID))
        self._OrderQueue = [pd.DataFrame(columns=["ID", "数量", "目标价"]) for i in range(max((1, self.SysArgs["交易延迟"])))]
        self._PositionCostPrice = np.full(nID, np.nan)# 当前持仓的成本价, 由于计算保证金的变化
        self._MarketFT = self.QSEnv.getTable(self.SysArgs["因子表"])
        self._iTradingRecord = []# 暂存的交易记录
        return 0
    def __QS_move__(self, idt, *args, **kwargs):
        super().__QS_move__(idt, *args, **kwargs)
        # 更新当前的账户信息
        iIndex = self.QSEnv.STM.DateTimeIndex
        self._Position[iIndex+1] = self._Position[iIndex]
        self._PositionAmount[iIndex+1] = self._PositionAmount[iIndex]
        self._NominalAmount[iIndex+1] = self._NominalAmount[iIndex]
        self._LastPrice = self._MarketFT.readData(dts=[self.QSEnv.STM.DateTime], ids=self._IDs, factor_names=[self.SysArgs["最新价"]]).iloc[0, 0].values
        # 撮合成交
        Orders = self._OrderQueue.pop(0)
        TradingRecord = self._matchMarketOrder(Orders)
        TradingRecord.extend(self._iTradingRecord)
        TradingRecord = pd.DataFrame(TradingRecord, index=np.arange(self._TradingRecord.shape[0], self._TradingRecord.shape[0]+len(TradingRecord)),
                                                              columns=self._TradingRecord.columns)
        self._TradingRecord = self._TradingRecord.append(TradingRecord)
        self._OrderQueue.append(pd.DataFrame(columns=["ID", "数量", "目标价"]))
        # 更新当前的账户信息
        Position = self._Position[iIndex+1]
        LastPrice = self._LastPrice.copy()
        NaMask = np.isnan(LastPrice)
        LastPrice[NaMask] = self._PositionCostPrice[NaMask]
        self._NominalAmount[iIndex+1] = Position*LastPrice*self.SysArgs["合约乘数"]
        self._PositionAmount[iIndex+1] += (LastPrice - self._PositionCostPrice)*Position*self.SysArgs["合约乘数"]
        self._PositionCostPrice = LastPrice
        return TradingRecord
    def __QS_after_move__(self, idt, *args, **kwargs):
        super().__QS_after_move__(self, idt, *args, **kwargs)
        if self.SysArgs["交易延迟"]==0:
            Orders = self._OrderQueue[0]
            self._iTradingRecord = self._matchMarketOrder(Orders)
            self._OrderQueue[0] = pd.DataFrame(columns=["ID", "数量", "目标价"])
            # 更新当前的账户信息
            iIndex = self.QSEnv.STM.DateTimeIndex
            Position = self._Position[iIndex+1]
            LastPrice = self._LastPrice.copy()
            NaMask = np.isnan(LastPrice)
            LastPrice[NaMask] = self._PositionCostPrice[NaMask]
            self._NominalAmount[iIndex+1] = Position*LastPrice*self.SysArgs["合约乘数"]
            self._PositionAmount[iIndex+1] += (LastPrice - self._PositionCostPrice)*Position*self.SysArgs["合约乘数"]
            self._PositionCostPrice = LastPrice
        return 0
    def __QS_end__(self):
        self._MarketFT = None
        self._OrderQueue = []
        self._PositionCostPrice = None
        return 0
    # 当前账户价值
    @property
    def AccountValue(self):
        return super().AccountValue + np.nansum(self._PositionAmount[self._Model.DateTimeIndex+1])
    # 当前账户的持仓
    @property
    def Position(self):
        return pd.Series(self._Position[self._Model.DateTimeIndex+1], index=self._IDs)
    # 当前账户的持仓金额, 即保证金
    @property
    def PositionAmount(self):
        return pd.Series(self._PositionAmount[self._Model.DateTimeIndex+1], index=self._IDs)
    # 当前账户的持仓名义金额
    @property
    def NominalAmount(self):
        return pd.Series(self._NominalAmount[self.QSEnv.STM.DateTimeIndex+1], index=self._IDs)
    # 本账户支持交易的证券 ID
    @property
    def IDs(self):
        return self._IDs
    # 当前账户中还未成交的订单, DataFrame(index=[int], columns=["ID", "数量", "目标价"])
    @property
    def Orders(self):
        return self._Orders
    # 当前最新价
    @property
    def LastPrice(self):
        return pd.Series(self._LastPrice, index=self._IDs)
    # 获取持仓的历史序列, 以时间点为索引, 返回: pd.DataFrame(持仓, index=[时间点], columns=[ID])
    def getPositionSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.DataFrame(self._Position[1:self.QSEnv.STM.DateTimeIndex+2], index=self.QSEnv.STM.DateTimeSeries, columns=self._IDs)
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取持仓证券的金额历史序列, 以时间点为索引, 返回: pd.DataFrame(持仓金额, index=[时间点], columns=[ID])
    def getPositionAmountSeries(self, dts=None, start_dt=None, end_dt=None):
        Data = pd.DataFrame(self._PositionAmount[1:self.QSEnv.STM.DateTimeIndex+2], index=self.QSEnv.STM.DateTimeSeries, columns=self._IDs)
        return cutDateTime(Data, dts=dts, start_dt=start_dt, end_dt=end_dt)
    # 获取账户价值的历史序列, 以时间点为索引
    def getAccountValueSeries(self, dts=None, start_dt=None, end_dt=None):
        CashSeries = self.getCashSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        PositionAmountSeries = self.getPositionAmountSeries(dts=dts, start_dt=start_dt, end_dt=end_dt).sum(axis=1)
        DebtSeries = self.getDebtSeries(dts=dts, start_dt=start_dt, end_dt=end_dt)
        return CashSeries + PositionAmountSeries - DebtSeries
    # 获取不能买入的证券 ID
    def getNonbuyableID(self, dt):
        if self.SysArgs["买入限制"]['禁止条件'] is None:
            return []
        return self.QSEnv.getTable(self.SysArgs["因子表"]).getID(dt, id_filter_str=self.SysArgs["买入限制"]['禁止条件'])
    # 获取不能卖出的证券ID
    def getNonsellableID(self, dt):
        if self.SysArgs["卖出限制"]['禁止条件'] is None:
            return []
        return self.QSEnv.getTable(self.SysArgs["因子表"]).getID(dt, id_filter_str=self.SysArgs["卖出限制"]['禁止条件'])
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
        UnitMargin = self.UnitMargin
        if target_id is not None:
            iUnitMargin = UnitMargin.get(target_id)
            if (iUnitMargin is not None) and (iUnitMargin>0):
                Orders.loc[Orders.shape[0]] = (target_id, amount/iUnitMargin, target_price)
        if combined_order is not None:
            combined_order["金额"] = combined_order["金额"]/UnitMargin[combined_order.index]
            combined_order = combined_order.reset_index()
            combined_order.columns = ["ID", "数量", "目标价"]
            combined_order.index = np.arange(Orders.shape[0], Orders.shape[0]+combined_order.shape[0])
            Orders = Orders.append(combined_order)
        self._OrderQueue[-1] = Orders
        return 0
    # 撮合成交市价单, 订单执行方式是先卖后买, 可用现金不足时实际成交数量将少于目标数量
    def _matchMarketOrder(self, orders):
        Orders = orders[pd.isnull(orders["目标价"])]
        if Orders.shape[0]==0:
            return []
        Orders = Orders.groupby(by=["ID"]).sum()["数量"]
        Orders = Orders[Orders!=0]
        # 提取当前信息
        CurDateTime = self.QSEnv.STM.DateTime
        iIndex = self.QSEnv.STM.DateTimeIndex
        AvailableCash = self.AvailableCash
        CashChanged = 0.0
        Position = self.Position
        TradePrice = self._MarketFT.readData(dts=[CurDateTime], ids=self._IDs, factor_names=[self.SysArgs["成交价"]]).iloc[0, 0]
        PositionAmount = self._PositionAmount[iIndex+1] + (TradePrice - self._PositionCostPrice)*Position*self.SysArgs["合约乘数"]
        TradingRecord = []
        # 先执行平仓交易
        CloseOrders = Orders[Position[Orders.index]*Orders<0]
        if CloseOrders.shape[0]>0:
            ClosePrice = TradePrice[CloseOrders.index]
            CloseLongOrders = CloseOrders[CloseOrders<0]# 平多委托单
            CloseShortOrders = CloseOrders[CloseOrders>0]# 平空委托单
            CloseLongLimit = pd.Series(np.zeros(CloseLongOrders.shape[0])+np.inf,index=CloseLongOrders.index)
            CloseShortLimit = pd.Series(np.zeros(CloseShortOrders.shape[0])+np.inf,index=CloseShortOrders.index)
            CloseLongLimit[pd.isnull(ClosePrice[CloseLongOrders.index])] = 0.0# 成交价缺失的不能交易
            CloseShortLimit[pd.isnull(ClosePrice[CloseShortOrders.index])] = 0.0# 成交价缺失的不能交易
            if self.SysArgs["卖出限制"]["禁止条件"] is not None:
                CloseLongLimit[self._MarketFT.getIDMask(CurDateTime, self.SysArgs["卖出限制"]["禁止条件"])[CloseLongOrders.index]] = 0.0# 满足禁止条件的不能交易
            if self.SysArgs["买入限制"]["禁止条件"] is not None:
                CloseShortLimit[self._MarketFT.getIDMask(CurDateTime, self.SysArgs["买入限制"]["禁止条件"])[CloseShortOrders.index]] = 0.0# 满足禁止条件的不能交易
            if self.SysArgs["卖出限制"]["成交量"]!="不限制":# 成交量缺失的不能交易
                MaxNum = self._MarketFT.readData(dts=[CurDateTime], factor_names=[self.SysArgs["卖出限制"]["成交量"]], ids=list(CloseLongOrders.index)).iloc[0, 0] * self.SysArgs["卖出限制"]["成交量限比"]
                CloseLongLimit = np.minimum(CloseLongLimit, MaxNum)
            if self.SysArgs["买入限制"]["成交量"]!="不限制":# 成交量缺失的不能交易
                MaxNum = self._MarketFT.readData(dts=[CurDateTime], factor_names=[self.SysArgs["买入限制"]["成交量"]], ids=list(CloseShortOrders.index)).iloc[0, 0] * self.SysArgs["买入限制"]["成交量限比"]
                CloseShortLimit = np.minimum(CloseShortLimit, MaxNum)
            CloseLongNum = np.minimum(np.minimum(CloseLongLimit, Position[CloseLongOrders.index]), CloseLongOrders.abs())
            CloseShortNum = np.minimum(np.minimum(CloseShortLimit, Position[CloseShortOrders.index].abs()), CloseShortOrders)
            if self.SysArgs["卖出限制"]['最小单位']!=0:
                CloseLongNum = (CloseLongNum/self.SysArgs["卖出限制"]['最小单位']).astype("int")*self.SysArgs["卖出限制"]['最小单位']
            if self.SysArgs["买入限制"]["最小单位"]!=0:
                CloseShortNum = (CloseShortNum/self.SysArgs["买入限制"]['最小单位']).astype("int")*self.SysArgs["买入限制"]['最小单位']
            CloseLongNum = CloseLongNum[CloseLongNum>0]
            CloseShortNum = CloseShortNum[CloseShortNum>0]
            CloseLongFees = CloseLongNum*ClosePrice[CloseLongNum.index]*self.SysArgs["合约乘数"]*self.SysArgs["卖出限制"]["交易费率"]
            CloseShortFees = CloseShortNum*ClosePrice[CloseShortNum.index]*self.SysArgs["合约乘数"]*self.SysArgs["买入限制"]["交易费率"]
            CloseLongAmount = CloseLongNum/Position[CloseLongNum.index]*PositionAmount[CloseLongNum.index]
            CloseShortAmount = -CloseShortNum/Position[CloseShortNum.index]*PositionAmount[CloseShortNum.index]
            PositionAmount[CloseLongNum.index] -= CloseLongAmount
            PositionAmount[CloseShortNum.index] -= CloseShortAmount
            Position[CloseLongNum.index] -= CloseLongNum
            Position[CloseShortNum.index] += CloseShortNum
            Orders[CloseLongNum.index] += CloseLongNum
            Orders[CloseShortNum.index] -= CloseShortNum
            CashChanged += CloseLongAmount.sum() + CloseShortAmount.sum() - CloseLongFees.sum() - CloseShortFees.sum()
            AvailableCash += CashChanged
            TradingRecord.extend(list(zip([CurDateTime]*CloseLongNum.shape[0], CloseLongNum.index, -CloseLongNum, 
                                          ClosePrice[CloseLongNum.index], CloseLongFees, CloseLongAmount-CloseLongFees, 
                                          ["close"]*CloseLongNum.shape[0])))
            TradingRecord.extend(list(zip([CurDateTime]*CloseShortNum.shape[0], CloseShortNum.index, CloseShortNum, 
                                          ClosePrice[CloseShortNum.index], CloseShortFees, CloseShortAmount-CloseShortFees, 
                                          ["close"]*CloseShortNum.shape[0])))
        # 再执行开仓交易
        Orders = Orders[Orders!=0]
        OpenOrders = Orders[Position[Orders.index]*Orders>=0]
        if OpenOrders.shape[0]>0:
            OpenPrice = TradePrice[OpenOrders.index]
            OpenLongOrders = OpenOrders[OpenOrders>0]# 开多委托单
            OpenShortOrders = OpenOrders[OpenOrders<0]# 开空委托单
            OpenLongLimit = pd.Series(np.zeros(OpenLongOrders.shape[0])+np.inf,index=OpenLongOrders.index)
            OpenShortLimit = pd.Series(np.zeros(OpenShortOrders.shape[0])+np.inf,index=OpenShortOrders.index)
            OpenLongLimit[pd.isnull(OpenPrice[OpenLongOrders.index])] = 0.0# 成交价缺失的不能交易
            OpenShortLimit[pd.isnull(OpenPrice[OpenShortOrders.index])] = 0.0# 成交价缺失的不能交易
            if self.SysArgs["买入限制"]["禁止条件"] is not None:
                OpenLongLimit[self._MarketFT.getIDMask(CurDateTime, self.SysArgs["买入限制"]["禁止条件"])[OpenLongOrders.index]] = 0.0# 满足禁止条件的不能交易
            if self.SysArgs["卖出限制"]["禁止条件"] is not None:
                OpenShortLimit[self._MarketFT.getIDMask(CurDateTime, self.SysArgs["卖出限制"]["禁止条件"])[OpenShortOrders.index]] = 0.0# 满足禁止条件的不能交易
            if self.SysArgs["买入限制"]["成交量"]!="不限制":# 成交量缺失的不能交易
                MaxNum = self._MarketFT.readData(dts=[CurDateTime], factor_names=[self.SysArgs["买入限制"]["成交量"]], ids=list(OpenLongOrders.index)).iloc[0, 0] * self.SysArgs["买入限制"]["成交量限比"]
                OpenLongLimit = np.minimum(OpenLongLimit, MaxNum)
            if self.SysArgs["卖出限制"]["成交量"]!="不限制":# 成交量缺失的不能交易
                MaxNum = self._MarketFT.readData(dts=[CurDateTime], factor_names=[self.SysArgs["卖出限制"]["成交量"]], ids=list(OpenLongOrders.index)).iloc[0, 0] * self.SysArgs["卖出限制"]["成交量限比"]
                OpenShortLimit = np.minimum(OpenShortLimit, MaxNum)
            OpenLongNum = np.minimum(OpenLongLimit, OpenLongOrders)
            OpenShortNum = np.minimum(OpenShortLimit, OpenShortOrders.abs())
            OpenLongCash = OpenLongNum*OpenPrice[OpenLongNum.index]*self.SysArgs["合约乘数"]*(self.SysArgs["保证金率"]+self.SysArgs["买入限制"]["交易费率"])
            OpenShortCash = OpenShortNum*OpenPrice[OpenShortNum.index]*self.SysArgs["合约乘数"]*(self.SysArgs["保证金率"]+self.SysArgs["卖出限制"]["交易费率"])
            TotalCash = OpenLongCash.sum()+OpenShortCash.sum()
            OpenLongCash = OpenLongCash/TotalCash*min((max((AvailableCash,0)),TotalCash))
            OpenShortCash = OpenShortCash/TotalCash*min((max((AvailableCash,0)),TotalCash))
            if self.SysArgs["买入限制"]['最小单位']!=0.0:
                OpenLongNum = (OpenLongCash/(self.SysArgs["买入限制"]['最小单位']*OpenPrice[OpenLongCash.index]*self.SysArgs["合约乘数"]*(self.SysArgs["保证金率"]+self.SysArgs["买入限制"]["交易费率"]))).astype("int")*self.SysArgs["开多限制"]['最小单位']
            if self.SysArgs["卖出限制"]['最小单位']!=0.0:
                OpenShortNum = (OpenShortCash/(self.SysArgs["卖出限制"]['最小单位']*OpenPrice[OpenShortCash.index]*self.SysArgs["合约乘数"]*(self.SysArgs["保证金率"]+self.SysArgs["卖出限制"]["交易费率"]))).astype("int")*self.SysArgs["开空限制"]['最小单位']
            OpenLongNum = OpenLongNum[OpenLongNum>0]
            OpenShortNum = OpenShortNum[OpenShortNum>0]
            Position[OpenLongNum.index] += OpenLongNum
            Position[OpenShortNum.index] -= OpenShortNum
            OpenLongMargin = OpenLongNum*OpenPrice[OpenLongNum.index]*self.SysArgs["合约乘数"]*self.SysArgs["保证金率"]
            OpenShortMargin = OpenShortNum*OpenPrice[OpenShortNum.index]*self.SysArgs["合约乘数"]*self.SysArgs["保证金率"]
            PositionAmount[OpenLongNum.index] += OpenLongMargin
            PositionAmount[OpenShortNum.index] += OpenShortMargin
            OpenLongFees = OpenLongNum*OpenPrice[OpenLongNum.index]*self.SysArgs["合约乘数"]*self.SysArgs["买入限制"]["交易费率"]
            OpenShortFees = OpenShortNum*OpenPrice[OpenShortNum.index]*self.SysArgs["合约乘数"]*self.SysArgs["卖出限制"]["交易费率"]
            CashChanged -= OpenLongMargin.sum() + OpenShortMargin.sum() + OpenLongFees.sum() + OpenShortFees.sum()
            TradingRecord.extend(list(zip([CurDateTime]*OpenLongNum.shape[0], OpenLongNum.index, OpenLongNum, 
                                          OpenPrice[OpenLongNum.index], OpenLongFees, -OpenLongMargin-OpenLongFees, 
                                          ["open"]*OpenLongNum.shape[0])))
            TradingRecord.extend(list(zip([CurDateTime]*OpenShortNum.shape[0], OpenShortNum.index, -OpenShortNum, 
                                          OpenPrice[OpenShortNum.index], OpenShortFees, -OpenShortMargin-OpenShortFees, 
                                          ["open"]*OpenShortNum.shape[0])))
        # 更新账户信息
        if TradingRecord:
            DebtDelta = max((-CashChanged - self.Cash, 0))
            self._Debt[iIndex+1] += DebtDelta
            self._Cash[iIndex+1] -= min((-CashChanged, self.Cash))
            if DebtDelta>0:
                self._DebtRecord.loc[self._DebtRecord.shape[0]] = (CurDateTime, DebtDelta)
            Position[Position.abs()<1e-6] = 0
            self._Position[iIndex+1] = Position.values
        self._PositionAmount[iIndex+1] = PositionAmount.values
        PositionCostPrice = TradePrice.values
        NaMask = pd.isnull(PositionCostPrice)
        PositionCostPrice[NaMask] = self._PositionCostPrice[NaMask]
        self._PositionCostPrice = PositionCostPrice
        return TradingRecord