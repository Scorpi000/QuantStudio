# -*- coding: utf-8 -*-
import base64
from io import BytesIO
import datetime as dt
from typing import Optional, Literal, List, Any, Tuple

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured
from pydantic import Field
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import DTLocalContext, DTInitData
from QuantStudio.Factor.Factor import Factor, FactorInitData, FactorContext, DataFactor, FactorLocalContext
from QuantStudio.Factor.FactorOperation import PanelOperation, PanelOperator
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.Tools.StrategyTestFun import summaryStrategy, calcYieldSeq, calcLSYield, formatStrategySummary


def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x * 100, )

_QS_MinPositionNum = 1e-8# 会被忽略掉的最小持仓数量
_QS_MinCash = 1e-8# 会被忽略掉的最小现金量

class CalcSimpleAccount(PanelOperator):
    """简单账户计算算子"""

    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=3, title="入参数", frozen=True)

    def __init__(self, init_cash:float=1e6, signal_type:Literal["买卖数量", "目标权重"]="买卖数量", short_allowed:bool=False, start_dt:Optional[dt.datetime]=None, args:dict = {}, config_file:Optional[str] = None, **kwargs):
        """初始化简单账户计算算子

        Args:
            init_cash: 初始资金
            signal_type: 信号类型
            short_allowed: 是否允许卖空
            start_dt: 净值开始日, 如果为 None, 表示从计算的第一个时点开始
        """
        Arity = args.get("Arity", None) or 3
        Args = {"Name": "calcSimpleAccount"} | args | {"DTMode": "单时点", "OutputMode": "全截面", "DataType": "object", "iInitFactor": 0}
        Args["ModelArgs"] = {"init_cash": init_cash, "short_allowed": short_allowed, "signal_type": signal_type} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [None])[0]] * Arity
        Args["LookBack"] = [1, 0, 0] + [0] * max(0, Arity - 3)
        Args["LookBackMode"] = ["扩张窗口"] + ["滚动窗口"] * max(0, Arity - 1)
        Args["StartDT"] = [start_dt] + [None] * max(0, Arity - 1)
        Args["CompoundType"] = [("Cash", float), ("Position", float), ("Amount", float), ("Turnover", float)]
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    # 更新交易限制条件
    def _getTradeLimit(self, position_num, last_price, buy_price, buy_limit, buy_amt_limit, sell_price, sell_limit, sell_amt_limit):
        SellVolLimit, BuyVolLimit = np.full(shape=last_price.shape, fill_value=np.inf), np.full(shape=last_price.shape, fill_value=np.inf)
        BuyVolLimit[~ (buy_price > 0)] = 0.0# 买入成交价缺失的不能买入
        SellVolLimit[~ (sell_price > 0)] = 0.0# 卖出成交价缺失的不能卖出
        if buy_limit is not None:# 满足买入禁止条件的不能买入
            BuyVolLimit[buy_limit == 1] = 0.0
        if sell_limit is not None:# 满足卖出禁止条件的不能卖出
            SellVolLimit[sell_limit == 1] = 0.0
        if buy_amt_limit is not None:# 指定了买入成交额, 成交额满足限制要求
            BuyVolLimit = np.clip(BuyVolLimit, a_min=None, a_max=buy_amt_limit / buy_price)
        if sell_amt_limit is not None:# 指定了卖出成交额, 成交额满足限制要求
            SellVolLimit = np.clip(SellVolLimit, a_min=None, a_max=sell_amt_limit / sell_price)
        if not self._QSArgs.ModelArgs["short_allowed"]:
            SellVolLimit = np.clip(SellVolLimit, a_min=None, a_max=np.clip(position_num, a_min=0.0, a_max=None))
        return BuyVolLimit, SellVolLimit

    # 撮合成交市价单
    # 以成交价完成成交, 满足交易限制要求
    # 未成交的市价单自动撤销
    def _matchMarketOrder(self, available_cash, position_num, ids, order_num, buy_price, buy_vol_limit, buy_fee, sell_price, sell_vol_limit, sell_fee):
        order_num = np.clip(order_num, a_max=buy_vol_limit, a_min=-sell_vol_limit)# 过滤限制条件
        # 先执行卖出交易
        SellAmounts = np.abs(np.clip(sell_price * order_num, a_min=None, a_max=0))
        Fees = SellAmounts * sell_fee# 卖出交易费
        CashChanged = SellAmounts - Fees
        Mask = (SellAmounts > 0)
        SellNums = np.clip(order_num, a_min=None, a_max=0)
        SellTradingRecord = {
            "ID": ids[Mask],
            "Num": SellNums[Mask],
            "Price": sell_price[Mask],
            "Fee": Fees[Mask],
            "CashChanged": CashChanged[Mask],
            "Direction": ["sell"] * np.sum(Mask)
        }
        # 再执行买入交易
        BuyAmounts = np.clip(buy_price * order_num, a_min=0, a_max=None)
        CashAcquired = BuyAmounts * (1 + buy_fee)
        TotalCashAcquired = np.nansum(CashAcquired)
        if TotalCashAcquired > 0:
            AvailableCash = available_cash + np.nansum(CashChanged)
            CashAllocated = min(AvailableCash, TotalCashAcquired) * CashAcquired / TotalCashAcquired
            BuyAmounts = CashAllocated / (1 + buy_fee)
            Fees = BuyAmounts * buy_fee
            BuyNums = BuyAmounts / buy_price
            Mask = (BuyAmounts > 0)
            BuyTradingRecord = {
                "ID": ids[Mask],
                "Num": BuyNums[Mask],
                "Price": buy_price[Mask],
                "Fee": Fees[Mask],
                "CashChanged": - CashAllocated[Mask],
                "Direction": ["buy"] * np.sum(Mask)
            }
        else:
            CashAllocated = BuyNums = np.zeros(shape=BuyAmounts.shape)
        # 更新持仓数量和现金
        PositionNum = position_num.copy()
        TotalAmount = np.nansum(buy_price * np.clip(PositionNum, a_min=None, a_max=0) + sell_price * np.clip(PositionNum, a_min=0, a_max=None))
        Turnover = (np.nansum(SellAmounts) + np.nansum(BuyAmounts)) / (TotalAmount + available_cash)
        PositionNum = PositionNum + BuyNums + SellNums
        Cash = available_cash + np.nansum(CashChanged) - np.nansum(CashAllocated)
        return Cash, PositionNum, Turnover, BuyTradingRecord, SellTradingRecord

    # 目标权重信号转买卖数量
    def _TargetWeightSignal2OrderNum(self, signal:np.ndarray, price:np.ndarray, cash:float, position_amt:np.ndarray) -> np.ndarray:
        AccountValue = abs(cash + np.nansum(position_amt))
        NaMask = pd.isnull(signal)
        if np.all(NaMask): return np.full_like(signal, np.nan)
        return (signal * AccountValue - position_amt) / price

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        ModelArgs = f._QSArgs.ModelArgs
        LastAccount = x[0].astype(self._QSArgs.CompoundType)[0]
        Cash, PositionNum = LastAccount["Cash"][0], LastAccount["Position"]
        LastPrice, Signal = x[1][0], x[2][0]
        if not np.any(np.abs(Signal) > 0):# 没有交易信号
            Rslt = np.array([np.full(shape=PositionNum.shape, fill_value=Cash), PositionNum, PositionNum * LastPrice, np.zeros_like(PositionNum)]).T
            return unstructured_to_structured(Rslt, dtype=np.dtype(self._QSArgs.CompoundType)).astype("O")
        if self._QSArgs.ModelArgs["signal_type"]=="买卖数量":
            OrderNum = Signal
        elif self._QSArgs.ModelArgs["signal_type"]=="目标权重":
            OrderNum = self._TargetWeightSignal2OrderNum(Signal, LastPrice, Cash, PositionNum * LastPrice)
        else:
            raise __QS_Error__(f"不支持的信号类型: {self._QSArgs.ModelArgs['signal_type']}")
        _, x = ((x[3][0], x[4:]) if ModelArgs["target_price"] else (None, x[3:]))
        BuyPrice, x = (x[0][0], x[1:]) if ModelArgs["buy_price"] else (LastPrice, x)
        BuyLimit, x = (x[0][0], x[1:]) if ModelArgs["buy_limit"] else (None, x)
        BuyFee, x = (x[0][0], x[1:]) if ModelArgs["buy_fee"] else (None, x)
        BuyAmtLimit, x = (x[0][0], x[1:]) if ModelArgs["buy_amt_limit"] else (None, x)
        SellPrice, x = (x[0][0], x[1:]) if ModelArgs["sell_price"] else (LastPrice, x)
        SellLimit, x = (x[0][0], x[1:]) if ModelArgs["sell_limit"] else (None, x)
        SellFee, x = (x[0][0], x[1:]) if ModelArgs["sell_fee"] else (None, x)
        SellAmtLimit, x = (x[0][0], x[1:]) if ModelArgs["sell_amt_limit"] else (None, x)
        # 撮合成交
        BuyVolLimit, SellVolLimit = self._getTradeLimit(PositionNum, LastPrice, BuyPrice, BuyLimit, BuyAmtLimit, SellPrice, SellLimit, SellAmtLimit)
        Cash, PositionNum, Turnover, BuyTradingRecord, SellTradingRecord = self._matchMarketOrder(Cash, PositionNum, np.array(iid), OrderNum, BuyPrice, BuyVolLimit, BuyFee, SellPrice, SellVolLimit, SellFee)
        PositionNum[np.abs(PositionNum) < _QS_MinPositionNum] = 0
        if abs(Cash) < _QS_MinCash: Cash = 0
        BuyTradingRecord, SellTradingRecord = pd.DataFrame(BuyTradingRecord), pd.DataFrame(SellTradingRecord)
        SellTradingRecord["trading_dt"] = BuyTradingRecord["trading_dt"] = idt[-1]
        f.UserData["TradingRecord"] = pd.concat([f.UserData.get("TradingRecord", pd.DataFrame()), SellTradingRecord, BuyTradingRecord])
        Rslt = np.array([np.full_like(PositionNum, Cash), PositionNum, PositionNum * LastPrice, np.full_like(PositionNum, Turnover)]).T
        return unstructured_to_structured(Rslt, dtype=np.dtype(self._QSArgs.CompoundType)).astype("O")

    def __call__(self, last_price: Factor, signal: Factor, target_price: Optional[Factor]=None, init_account: Optional[Factor]=None, 
        buy_price: Optional[Factor]=None, buy_limit: Optional[Factor]=None, buy_fee: float | Factor=0, buy_amt_limit: Optional[Factor]=None,
        sell_price: Optional[Factor]=None, sell_limit: Optional[Factor]=None, sell_fee: float | Factor=0, sell_amt_limit: Optional[Factor]=None,
        factor_args:dict={}, **kwargs
    ) -> PanelOperation:
        """将算子作用在若干个因子对象上以产生简单账户因子

        Args:
            last_price: 最新价因子
            signal: 信号因子, 默认是买卖数量
            target_price: 目标价因子, None 表示信号将转换成市价单
            init_account: 初始账户因子, 复合因子, [("Cash", float), ("Position", float), ("Amount", float), ("Turnover", float)], None 表示使用算子创建时提供的初始资金创建该因子
            buy_price: 买入成交价因子
            buy_limit: 禁止买入条件因子, 该因子值等于 1 的 ID 禁止买入
            buy_fee: 买入交易费率因子
            buy_amt_limit: 买入成交额限制因子, 该期买入额不能超过该因子值
            sell_price: 卖出成交价因子
            sell_limit: 禁止卖出条件因子, 该因子值等于 1 的 ID 禁止卖出
            sell_fee: 卖出交易费率因子
            sell_amt_limit: 卖出成交额限制因子, 该期卖出额不能超过该因子值
            factor_args: 创建 IC 因子时传递个它的参数集
            kwargs: 创建 IC 因子时传递给它的其他入参

        Returns:
            简单账户因子
        """
        if init_account is None: init_account = DataFactor(data=(self._QSArgs.ModelArgs["init_cash"], 0, 0, 0), args={"Name": "InitAccount"})
        Factors = [init_account, last_price, signal]
        if target_price is not None: Factors.append(target_price)
        if buy_price is not None: Factors.append(buy_price)
        if buy_limit is not None: Factors.append(buy_limit)
        if buy_fee is not None: Factors.append(buy_fee)
        if buy_amt_limit is not None: Factors.append(buy_amt_limit)
        if sell_price is not None: Factors.append(sell_price)
        if sell_limit is not None: Factors.append(sell_limit)
        if sell_fee is not None: Factors.append(sell_fee)
        if sell_amt_limit is not None: Factors.append(sell_amt_limit)
        ModelArgs = {
            "target_price": (target_price is not None), 
            "buy_price": (buy_price is not None), 
            "buy_limit": (buy_limit is not None),
            "buy_fee": (buy_fee is not None),
            "buy_amt_limit": (buy_amt_limit is not None),
            "sell_price": (sell_price is not None),
            "sell_limit": (sell_limit is not None),
            "sell_fee": (sell_fee is not None),
            "sell_amt_limit": (sell_amt_limit is not None)
        }
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | ModelArgs
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


def genAccountOutput(init_cash, cash_series, debt_series, account_value_series, cash_record, debt_record, risk_free_rate=0.0):
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
    if CashDelta.shape[0] > 0:
        Output["时间序列"]["累计资金投入"] = init_cash + CashDelta.reindex(index=Output["时间序列"].index).fillna(0).cumsum()
        AccountEarnings[CashDelta.index] -= CashDelta
    else:
        Output["时间序列"]["累计资金投入"] = init_cash
    Output["时间序列"]["收益"] = AccountEarnings
    PreAccountValue = np.r_[init_cash, account_value_series.values[:-1]]
    AccountReturn = AccountEarnings / np.abs(PreAccountValue)
    AccountReturn[AccountEarnings==0] = 0.0
    Output["时间序列"]["收益率"] = AccountReturn
    AccountReturn[np.isinf(AccountReturn)] = np.nan
    Output["时间序列"]["累计收益率"] = AccountReturn.cumsum()
    Output["时间序列"]["净值"] = (AccountReturn + 1).cumprod()
    if CashDelta.shape[0] > 0:
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
    # 统计数据
    TargetCols = ["净值"] + ["考虑资金投入的净值"] * (CashDelta.shape[0]>0) + ["无杠杆净值"] * (debt_record.shape[0]>0)
    Output["统计数据"] = summaryStrategy(Output["时间序列"][TargetCols].values, list(Output["时间序列"].index), init_wealth=[1] * len(TargetCols), risk_free_rate=risk_free_rate)
    Output["统计数据"].columns = ["绝对表现"] + ["考虑资金投入的表现"] * (CashDelta.shape[0]>0) + ["无杠杆表现"] * (debt_record.shape[0]>0)
    return Output


class AccountReport(BTNode):
    """账户报告"""
    
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="账户报告", frozen=True, title="名称")
        InitCash: Optional[float] = Field(default=None, frozen=True, title="初始资金")
        RiskFreeRate: float = Field(default=0, frozen=True, title="无风险利率")
        RebalanceDTs: Optional[List[dt.datetime]] = Field(default=None, title="再平衡时点", frozen=True)
    
    def __init__(self, account: Factor, bmk_nv:Optional[Factor]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=[account] + ([bmk_nv] if bmk_nv else []), args=args, config_file=config_file, **kwargs)
    
    def genMatplotlibFig(self, output:dict, file_path: Optional[str]=None) -> Figure:
        hasCapitalInvest = ("考虑资金投入的表现" in output["统计数据"])
        hasBenchmark = ("相对表现" in output["统计数据"])
        nRow, nCol = 1, 2+hasCapitalInvest+hasBenchmark
        Fig = Figure(figsize=(min(40, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(0, output["时间序列"].shape[0])
        xTicks = np.arange(0, output["时间序列"].shape[0], int(output["时间序列"].shape[0]/10))
        xTickLabels = [output["时间序列"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        iAxes = Fig.add_subplot(nRow, nCol, 1)
        iAxes.plot(xData, output["时间序列"]["账户价值"].values, label="账户价值", color="indianred", lw=2.5)
        iRAxes = iAxes.twinx()
        iRAxes.bar(xData, output["时间序列"]["收益"].values, label="账户收益", color="steelblue")
        iRAxes.legend(loc="upper right")
        iAxes.set_xticks(xTicks)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc="upper left")
        iAxes.set_title("账户表现")
        iAxes = Fig.add_subplot(nRow, nCol, 2)
        iRAxes = iAxes.twinx()
        iRAxes.yaxis.set_major_formatter(yMajorFormatter)
        if "无杠杆净值" in output["时间序列"]:
            iAxes.plot(xData, output["时间序列"]["无杠杆净值"].values, label="无杠杆净值", color="indianred", lw=2.5)
            iRAxes.bar(xData, output["时间序列"]["无杠杆收益率"].values, label="无杠杆收益率", color="steelblue")
        else:
            iAxes.plot(xData, output["时间序列"]["净值"].values, label="净值", color="indianred", lw=2.5)
            iRAxes.bar(xData, output["时间序列"]["收益率"].values, label="收益率", color="steelblue")
        if hasBenchmark: iAxes.plot(xData, output["时间序列"]["基准净值"].values, label="基准净值", color="forestgreen", lw=2.5)
        iRAxes.legend(loc="upper right")
        iAxes.set_xticks(xTicks)
        iAxes.set_xticklabels(xTickLabels)
        iAxes.legend(loc="upper left")
        iAxes.set_title("净值表现")
        if hasCapitalInvest:
            iAxes = Fig.add_subplot(nRow, nCol, 3)
            iAxes.plot(xData, output["时间序列"]["累计资金投入"].values, label="累计资金投入", color="indianred", lw=2.5)
            iRAxes = iAxes.twinx()
            iRAxes.yaxis.set_major_formatter(yMajorFormatter)
            iRAxes.plot(xData, output["时间序列"]["考虑资金投入的累计收益率"].values, label="考虑资金投入的累计收益率", color="steelblue", lw=2.5)
            iRAxes.legend(loc="upper right")
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iAxes.set_title("考虑资金投入的表现")
        if hasBenchmark:
            iAxes = Fig.add_subplot(nRow, nCol, 3+hasCapitalInvest)
            iAxes.plot(xData, output["时间序列"]["相对净值"].values, label="相对净值", color="indianred", lw=2.5)
            iRAxes = iAxes.twinx()
            iRAxes.yaxis.set_major_formatter(yMajorFormatter)
            iRAxes.bar(xData, output["时间序列"]["相对收益率"].values, label="相对收益率", color="steelblue")
            iRAxes.legend(loc="upper right")
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iAxes.set_title("相对表现")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    
    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if isinstance(getattr(self.Deps[0], "Operator", None), CalcSimpleAccount):
            ModelArgs = self.Deps[0].Operator._QSArgs.ModelArgs
            HTML += f"<li>信号类型: {ModelArgs['signal_type']}</li>"
            HTML += f"<li>允许卖空: {ModelArgs['short_allowed']}</li>"
        if self._QSArgs.InitCash is not None:
            HTML += f"<li>初始资金: {self._QSArgs.InitCash}</li>"
        HTML += f"<li>无风险利率: {self._QSArgs.RiskFreeRate}</li>"
        HTML += "</ul>"
        HTML = formatStrategySummary(output["统计数据"]).to_html()
        Pos = HTML.find(">")
        HTML = HTML[:Pos]+' align="center"'+HTML[Pos:]
        Fig = self.genMatplotlibFig(output)
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer)
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML

    def init_compute(self, path: List[str], init_data: DTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=None) for i, iInitData in enumerate(InitData)]
    
    def forward_compute(self, path: List[str], fwd_data: DTLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], DTLocalContext]:
        return [FactorLocalContext(DTs=fwd_data.DTs, IDs=context.NodeState[iDep.QSID]["section_ids"], PIDs=context.PIDList) for iDep in self.Deps], DTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[DTLocalContext]=None) -> dict:
        Account = bwd_data_list[0].dropna(how="all", axis=0)
        CashSeries, PositionNum, PositionAmt, Turnover = Account.map(lambda x: x[0] if pd.notnull(x) else np.nan), Account.map(lambda x: x[1] if pd.notnull(x) else np.nan), Account.map(lambda x: x[2] if pd.notnull(x) else np.nan), Account.map(lambda x: x[3] if pd.notnull(x) else np.nan)
        CashSeries = CashSeries.iloc[:, 0]
        AccountValueSeries = CashSeries + PositionAmt.sum(axis=1)
        InitCash = (self._QSArgs.InitCash if self._QSArgs.InitCash is not None else AccountValueSeries.iloc[0])
        DebtSeries = pd.Series(0, index=CashSeries.index)
        CashRecord, DebtRecord = pd.DataFrame(columns=["时间点", "现金流", "备注"]), pd.DataFrame(columns=["时间点", "融资", "备注"])
        Output = genAccountOutput(InitCash, CashSeries, DebtSeries, AccountValueSeries, CashRecord, DebtRecord, risk_free_rate=self._QSArgs.RiskFreeRate)
        Output["换手率"] = Turnover
        Output["持仓数量"] = PositionNum
        if len(bwd_data_list) > 1:# 设置了基准
            BmkNV = bwd_data_list[1].iloc[:, 0]
            BenchmarkOutput = pd.DataFrame(calcYieldSeq(wealth_seq=BmkNV.values), index=BmkNV.index, columns=["基准收益率"])
            BenchmarkOutput["基准累计收益率"] = BenchmarkOutput["基准收益率"].cumsum()
            BenchmarkOutput["基准净值"] = BmkNV / BmkNV.iloc[0]
            if DebtRecord.shape[0]>0:
                LYield = Output["时间序列"]["无杠杆收益率"].values
            else:
                LYield = Output["时间序列"]["收益率"].values
            if not self._QSArgs.RebalanceDTs: RebalanceIndex = None
            else:
                RebalanceIndex = pd.Series(np.arange(BenchmarkOutput["基准收益率"].shape[0]), index=BenchmarkOutput["基准收益率"].index, dtype=int)
                RebalanceIndex = sorted(RebalanceIndex.loc[RebalanceIndex.index.intersection(self._QSArgs.RebalanceDTs)].values)
            BenchmarkOutput["相对收益率"] = calcLSYield(long_yield=LYield, short_yield=BenchmarkOutput["基准收益率"].values, rebalance_index=RebalanceIndex)
            BenchmarkOutput["相对累计收益率"] = BenchmarkOutput["相对收益率"].cumsum()
            BenchmarkOutput["相对净值"] = (1 + BenchmarkOutput["相对收益率"]).cumprod()
            Output["时间序列"] = pd.merge(Output["时间序列"], BenchmarkOutput, left_index=True, right_index=True)
            BenchmarkOutput["基准收益率"] = BenchmarkOutput["基准净值"].values / np.r_[1, BenchmarkOutput["基准净值"].iloc[:-1].values] - 1
            BenchmarkOutput["基准累计收益率"] = BenchmarkOutput["基准收益率"].cumsum()
            BenchmarkOutput["相对收益率"] = BenchmarkOutput["相对净值"].values / np.r_[1, BenchmarkOutput["相对净值"].iloc[:-1].values] - 1
            BenchmarkOutput["相对累计收益率"] = BenchmarkOutput["相对收益率"].cumsum()
            BenchmarkStatistics = summaryStrategy(BenchmarkOutput[["基准净值", "相对净值"]].values, list(BenchmarkOutput.index), init_wealth=[1, 1], risk_free_rate=self._QSArgs.RiskFreeRate)
            BenchmarkStatistics.columns = ["基准表现", "相对表现"]
            Output["统计数据"] = pd.merge(Output["统计数据"], BenchmarkStatistics, left_index=True, right_index=True)
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
