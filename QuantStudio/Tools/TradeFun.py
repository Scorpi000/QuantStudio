# coding=utf-8
"""交易相关函数"""
import numpy as np

# 模拟撮合成交, 使用订单簿数据
# quantity: 委托单的数量, >0 表示买入, <0 表示卖出;
# limit_price: 限价, nan 表示市价单
# order_book: 订单簿, array(shape=(n, 2)), 第一列表示价格, 第二列表示数量
# last_price: 最新价(前一成交价), 如果为 nan 按照以下原则撮合:
# * 最高买入申报价格与最低卖出申报价格相同，以该价格为成交价格；
# * 买入申报价格高于即时揭示的最低卖出申报价格的，以即时揭示的最低卖出申报价格为成交价格；
# * 卖出申报价格低于即时揭示的最高买入申报价格的，以即时揭示的最高买入申报价格为成交价格。
# 如果 last_price 不为 nan 则撮合成交价等于买入价(bp)、卖出价(sp)和前一成交价(cp)三者中居中的一个价格。:
# * 当 bp≥sp≥cp，则最新成交价=sp；
# * 当 bp≥cp≥sp，则最新成交价=cp；
# * 当 cp≥bp≥sp，则最新成交价=bp。
# 返回: array(shape=(n, 2)), 第一列表示成交价, 第二列表示对应成交价下的成交量
def matchOrderByOrderBook(quantity, limit_price=np.nan, order_book=np.full(shape=(0,2), fill_value=np.nan), last_price=np.nan):
    Direction = np.sign(quantity)# 买卖方向, +1 买, -1 卖
    if not np.isnan(limit_price): order_book = order_book[order_book[:, 0] * Direction <= limit_price * Direction, :]
    else: limit_price = Direction * np.inf
    order_book = order_book[np.argsort(Direction * order_book[:, 0]), :]
    MatchVol = (order_book[:, 1] - np.clip(np.cumsum(order_book[:, 1]) - abs(quantity), 0, np.inf)).clip(0, np.inf)
    Mask = (MatchVol>0)
    MatchVol, MatchPrice = MatchVol[Mask], order_book[Mask, 0]
    if not np.isnan(last_price):
        if Direction>0: MatchPrice = np.minimum(limit_price, np.maximum(last_price, MatchPrice))
        else: MatchPrice = np.maximum(limit_price, np.minimum(last_price, MatchPrice))
    return np.c_[MatchPrice, MatchVol]

# 模拟撮合成交, 使用 Tick 数据, 模拟成交方式等同于订单簿只有买一和卖一的情况
# quantity: 委托单的数量, >0 表示买入, <0 表示卖出;
# limit_price: 限价, nan 表示市价单
# bid_price: 最高买入申报价格
# bid_size: 最高买入申报数量, 为 0 表示无买盘
# ask_price: 最低卖出申报价格
# ask_size: 最低卖出申报数量, 为 0 表示无卖盘
# last_price: 最新价(前一成交价),
# 返回: array(shape=(1, 2)), 第一列表示成交价, 第二列表示对应成交价下的成交量
def matchOrderByTickData(quantity, limit_price=np.nan, bid_price=np.nan, bid_size=0, ask_price=np.nan, ask_size=0, last_price=np.nan):
    Direction = np.sign(quantity)# 买卖方向, +1 买, -1 卖
    if np.isnan(limit_price): limit_price = Direction * np.inf
    if Direction>0: bp, sp, Size = limit_price, ask_price, ask_size
    else: bp, sp, Size = bid_price, limit_price, bid_size
    if not (bp>=sp): return np.array(((np.nan, 0), ))
    if np.isnan(last_price):
        if Direction>0: return np.array(((ask_price, min(ask_size, quantity)), ))
        else: return np.array(((bid_price, min(bid_size, -quantity)), ))
    if last_price<=sp: return np.array(((sp, min(Size, abs(quantity))), ))
    elif last_price>=bp: return np.array(((bp, min(Size, abs(quantity))), ))
    else: return np.array(((last_price, min(Size, abs(quantity))), ))

# 模拟撮合成交, 使用 Bar 数据, 等同于订单簿的价格从最低价到最高价以最小变动单位为间隔取值, 成交量均匀分布
# quantity: 委托单的数量, >0 表示买入, <0 表示卖出;
# limit_price: 限价, nan 表示市价单
# bar_price: array([open, high, low, close]), 开高低收
# bar_vol: 成交量
# min_chg: 最小变动单位, 0 表示假设连续价格
# last_price: 最新价(前一成交价)
# 返回: array(shape=(n, 2)), 第一列表示成交价, 第二列表示对应成交价下的成交量
def matchOrderByBarData(quantity, limit_price=np.nan, bar_price=np.full(shape=(4,), fill_value=np.nan), bar_vol=0, min_chg=0, last_price=np.nan):
    Direction = np.sign(quantity)# 买卖方向, +1 买, -1 卖
    if np.isnan(limit_price): limit_price = Direction * np.inf
    if np.isnan(bar_price[1]) or np.isnan(bar_price[2]): return np.array(((np.nan, 0), ))
    if min_chg==0:
        if Direction>0:
            MatchPrice = (bar_price[2] + min(limit_price, bar_price[1])) / 2
            MatchVol = bar_vol * min(1, max(0, 2 * (MatchPrice - Low) / (bar_price[1] - bar_price[2])))
        else:
            MatchPrice = (max(limit_price, bar_price[2]) + bar_price[1]) / 2
            MatchVol = bar_vol * min(1, max(0, 2 * (High - MatchPrice) / (bar_price[1] - bar_price[2])))
        return np.array(((MatchPrice, MatchVol), ))
    BookPrice = np.arange(bar_price[2], bar_price[1]+min_chg/2, min_chg)
    BookVol = np.full(shape=BookPrice.shape, fill_value=int(bar_vol / BookPrice.shape[0]))
    BookVol[int(BookVol.shape[0]/2)] += bar_vol - np.sum(BookVol)
    return matchOrderByOrderBook(quantity, limit_price, order_book=np.c_[BookPrice, BookVol], last_price=last_price)

# 给定交易行为的策略回测(非自融资策略)
# num_units: 每个时点的交易数量, DataFrame(index=[DateTime], columns=[ID]), nDT: 时点数, nID: ID 数
# simulator: 交易模拟器, callable, simulator(idt, iid, quantity), 返回: array(shape=(n, 2)), 第一列是成交价, 第二列是成交数量
# price: 最新价, array(shape=(nDT, nID))
# fee: 手续费率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# long_margin: 多头保证金率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# short_margin: 空头保证金率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# close_old_first: 是否优先平老仓, bool
# 返回: 交易记录
def testActionStrategy(num_units, simulator, price, fee=0.0, long_margin=1.0, short_margin=1.0, close_old_first=True):
    def _normalize(data, nrow, ncol):
        if np.isscalar(data): return np.full(shape=(nrow, ncol), fill_value=data)
        elif data.ndim==1: return np.reshape(data, (1, ncol)).repeat(nrow, axis=0)
        else: return data
    def _matchCloseNum(index, holdings, close_nums, close_price):
        Rslt = []
        for i, iCloseNum in enumerate(close_nums):
            if not holdings: break
            while iCloseNum>0:
                if not holdings: break
                if holdings[0]<=iCloseNum:
                    iCloseNum += holdings[0]
                    Rslt.append((index.pop(0), holdings.pop(0), close_price[i]))
                else:
                    Rslt.append((index[0], iCloseNum, close_price[i]))
                    holdings[0] += iCloseNum
                    iCloseNum = 0
            close_nums[i] = iCloseNum
        return (np.array(Rslt), close_nums, index, holdings)
    fee = _normalize(fee, num_units.shape[0], num_units.shape[1])
    long_margin = _normalize(long_margin, num_units.shape[0], num_units.shape[1])
    short_margin = _normalize(short_margin, num_units.shape[0], num_units.shape[1])
    TradeRecord = pd.DataFrame(columns=["证券代码","数量","合约乘数","开仓时点","开仓价格","开仓手续费","开仓滑点","开仓保证金","平仓时点","平仓价格","平仓手续费","平仓滑点","平仓盈亏","最新价","浮动盈亏"])
    for i, iDT in enumerate(num_units.index):
        for j, jID in enumerate(num_units.columns):
            ijNum = num_units.iloc[i, j]
            if ijNum==0: continue
            ijMatchRslt = simulator(iDT, jID, ijNum)
            ijMatchRslt = ijMatchRslt[((~np.isnan(ijMatchRslt[:, 0])) & (ijMatchRslt[:, 1]!=0)), :]
            if ijMatchRslt.shape[0]==0: continue
            ijHoldingMask = ((TradeRecord["证券代码"]==jID) & pd.isnull(TradeRecord["平仓时点"]))
            ijHoldingRecord = TradeRecord[ijHoldingMask].sort_values(by=["开仓时点"], ascending=close_old_first)
            ijTotalPosition = ijHoldingRecord["数量"].sum()
            ijDirection = np.sign(ijNum)
            if (ijTotalPosition!=0) and (ijDirection!=np.sign(ijTotalPosition)):# 有平仓行为
                ijCloseRslt, ijMatchRslt[:, 1], ijPostion = _matchCloseNum(ijHoldingRecord["数量"].tolist(), ijMatchRslt[:, 1], ijMatchRslt[:, 0])
                ijMatchRslt = ijMatchRslt[ijMatchRslt[:, 1]!=0, :]
                ijCloseRecord = pd.DataFrame(index=TradeRecord.shape[0]+np.arange(0, ijCloseRslt.shape[0]), 
                                            columns=["证券代码","数量","合约乘数","开仓时点","开仓价格","开仓手续费","开仓滑点","开仓保证金","平仓时点","平仓价格","平仓手续费","平仓滑点","平仓盈亏","最新价","浮动盈亏"])
                ijCloseRecord["证券代码"] = jID
                ijCloseRecord["数量"] = ijCloseRslt[:, 1] * ijDirection
                ijOpenRecord["合约乘数"] = 1
                ijOpenRecord["开仓时点"] = iDT
                ijOpenRecord["开仓价格"] = iRslt[:, 0]
                ijOpenRecord["开仓手续费"] = iRslt[:, 0] * iRslt[:, 1] * ijOpenRecord["合约乘数"] * fee[i, j]
                ijOpenRecord["开仓滑点"] = (iRslt[:, 0] - price[i, j]) * ijDirection
                ijOpenRecord["开仓保证金"] = (iRslt[:, 0] * iRslt[:, 1]) * (max(0, ijDirection) * long_margin[i, j] + max(0, -ijDirection) * short_margin[i, j])
                
            if (ijTotalPosition==0) or (ijDirection==np.sign(ijTotalPosition)):# 开仓行为
                ijOpenRecord = pd.DataFrame(index=TradeRecord.shape[0]+np.arange(0, ijRslt.shape[0]), 
                                            columns=["证券代码","数量","合约乘数","开仓时点","开仓价格","开仓手续费","开仓滑点","开仓保证金","平仓时点","平仓价格","平仓手续费","平仓滑点","平仓盈亏","最新价","浮动盈亏"])
                ijOpenRecord["证券代码"] = jID
                ijOpenRecord["数量"] = iRslt[:, 1] * ijDirection
                ijOpenRecord["合约乘数"] = 1
                ijOpenRecord["开仓时点"] = iDT
                ijOpenRecord["开仓价格"] = iRslt[:, 0]
                ijOpenRecord["开仓手续费"] = iRslt[:, 0] * iRslt[:, 1] * ijOpenRecord["合约乘数"] * fee[i, j]
                ijOpenRecord["开仓滑点"] = (iRslt[:, 0] - price[i, j]) * ijDirection
                ijOpenRecord["开仓保证金"] = (iRslt[:, 0] * iRslt[:, 1]) * (max(0, ijDirection) * long_margin[i, j] + max(0, -ijDirection) * short_margin[i, j])

# 生成交易记录
# holding_record: 持仓记录, pd.DataFrame(columns=["ID","数量","合约乘数","开仓时点","开仓价格","开仓滑点"])
# trade_result: 交易结果, pd.DataFrame(columns=["ID","成交价","数量"])
def genTradeRecord(holding_record, trade_result, close_old_first=True):
    def _matchCloseNum(index, holdings, close_nums, close_price):
        Rslt = []
        for i, iCloseNum in enumerate(close_nums):
            if not holdings: break
            while iCloseNum>0:
                if not holdings: break
                if holdings[0]<=iCloseNum:
                    iCloseNum += holdings[0]
                    Rslt.append((index.pop(0), holdings.pop(0), close_price[i]))
                else:
                    Rslt.append((index[0], iCloseNum, close_price[i]))
                    holdings[0] += iCloseNum
                    iCloseNum = 0
            close_nums[i] = iCloseNum
        return (Rslt, close_nums, index, holdings)
    CloseRecord = pd.DataFrame(columns=["ID","数量","合约乘数","开仓时点","开仓价格","开仓滑点","平仓时点","平仓价格","平仓滑点","平仓盈亏"])
    OpenRecord = pd.DataFrame(columns=["ID","数量","合约乘数","开仓时点","开仓价格","开仓滑点"])
    NewHoldingRecord = pd.DataFrame(columns=["ID","数量","合约乘数","开仓时点","开仓价格","开仓滑点"])
    trade_result = trade_result[pd.notnull(trade_result["ID"]) & pd.notnull(trade_result["成交价"]) & (trade_result["数量"].abs()>0)]
    if trade_result.shape[0]==0: return (CloseRecord, holding_record)
    for i, iID in enumerate(trade_result["ID"].unique()):
        iTradeResult = trade_result[trade_result["ID"]==iID]
        iHoldingRecord = holding_record[holding_record["ID"]==iID]
        iTotalPosition = iHoldingRecord["数量"].sum()
        if (ijTotalPosition!=0) and (ijDirection!=np.sign(ijTotalPosition)):# 有平仓行为