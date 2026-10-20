# -*- coding: utf-8 -*-
"""择时策略：等权目标组合，以及按标的独立出入金的固定本金账户。"""
import datetime as dt
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import DTLocalContext
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.Factor.Factor import DataFactor, Factor, FactorInitData, FactorLocalContext
from QuantStudio.Factor.FactorOperation import FactorInputModel, OperatorModelArgs, PanelOperation, PanelOperator, SectionOperation, SectionOperator
from QuantStudio.BackTest.Strategy.Strategy import (
    MakeAccount as _BaseMakeAccount, AccountStats as _BaseAccountStats,
    AccountReport as _BaseAccountReport, _QS_MinCash, _QS_MinPositionNum,
)
from QuantStudio.Tools.StrategyTestFun import summaryStrategy, formatStrategySummary


class CalcTimingPortfolio(SectionOperator):
    """仅做多的等资金择时组合算子。

    每个交易时点将总资金等分给所有 mask=True 的标的。信号为 1 的标的
    使用其分配资金做多，0（中性）、-1（看空）或缺失信号对应的资金保留
    为现金。mask=False 或缺失的标的目标权重为 0，已有持仓需要清仓。
    """

    class __QS_InputClass__(FactorInputModel):
        """择时组合的公开因子输入。"""

        signal: Factor = Field(title="择时信号")
        mask: Optional[Factor] = Field(default=None, title="标的池")

    __QS_CallLayout__ = {
        "signal": {"grouped": False, "keyword_only": False},
        "mask": {"grouped": False, "keyword_only": False},
    }

    def __init__(self, descriptor_ids: Optional[List[str]] = None, args: dict = {},
                 config_file: Optional[str] = None, **kwargs):
        """初始化择时组合算子。

        Args:
            descriptor_ids: 依赖因子的完整标的截面，None 表示使用输出截面。
            args: 算子参数集。
            config_file: 配置文件地址。
            **kwargs: 传递给父类的其他参数。
        """
        Args = {"Name": "calcTimingPortfolio"} | args | {"DTMode": "多时点", "DataType": "double"}
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * 2
        super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str],
                  x: List[np.ndarray], args: dict) -> np.ndarray:
        """按完整标的池计算目标权重，并对齐输出截面。

        Args:
            f: 当前组合因子。
            idt: 计算时点序列。
            iid: 输出标的序列。
            x: 择时信号和标的池的二维数组。
            args: 模型参数。

        Returns:
            目标权重数组；全池空仓时输出全零，非交易时点输出 NaN。
        """
        SectionIDs = self._QSArgs.DescriptorSection[0] or iid
        Mask = x[1] == 1
        Count = Mask.sum(axis=1, keepdims=True)
        Portfolio = pd.DataFrame(
            ((x[0] == 1) & Mask) / np.maximum(Count, 1), index=idt, columns=SectionIDs
        )
        if f._QSArgs.CalcDTRuler:
            Portfolio.loc[~Portfolio.index.isin(f._QSArgs.CalcDTRuler)] = np.nan
        return Portfolio.reindex(columns=iid, fill_value=0.0).values

    def __call__(self, signal: Factor, mask: Optional[Factor] = None,
                 factor_args: dict = {}, **kwargs) -> SectionOperation:
        """构造仅做多的择时目标权重因子。

        Args:
            signal: 方向信号因子，1 看多、0 中性、-1 看空；缺失值按空仓处理。
            mask: 标的池因子，True 或 1 表示入选，None 表示全部标的入选。
            factor_args: 组合因子参数，可用 CalcDTRuler 指定交易时点。
            **kwargs: 传递给组合因子的其他参数。

        Returns:
            目标权重因子，可作为 MakeAccount 的目标权重信号，或传给
            CalcPortfolioNV 计算组合净值。未投资部分保留为现金。
        """
        if mask is None:
            mask = DataFactor(data=True)
        return super().__call__(signal, mask, factor_args=factor_args, **kwargs)


class _TimingAccountArgs(OperatorModelArgs):
    """独立子账户的本金与交易方向约束。"""

    init_cash: float = Field(default=1e6, gt=0, allow_inf_nan=False, title="每个标的的初始资金")
    short_allowed: Literal[False] = Field(default=False, title="允许卖空")


class MakeTimingAccount(_BaseMakeAccount):
    """为每个入池标的注入固定本金并独立交易的仅做多择时账户。

    Cash 是各标的自己的现金，不在列之间共享。入池时注入 init_cash，
    看多时投入子账户可用现金，中性、看空或缺失信号时清仓。出池后待持仓
    全部卖出才提走剩余现金；再次入池重新注入本金。连续看多时不再平衡。
    """

    class __QS_InputClass__(FactorInputModel):
        """独立子账户的公开输入。"""

        last_price: Factor = Field(title="最新价")
        signal: Factor = Field(title="择时方向")
        mask: Optional[Factor] = Field(default=None, title="标的池")
        buy_price: Optional[Factor] = Field(default=None, title="买入价")
        buy_limit: Optional[Factor] = Field(default=None, title="买入限制")
        buy_fee: float | Factor = Field(default=0, title="买入费率")
        buy_amt_limit: Optional[Factor] = Field(default=None, title="买入金额限制")
        sell_price: Optional[Factor] = Field(default=None, title="卖出价")
        sell_limit: Optional[Factor] = Field(default=None, title="卖出限制")
        sell_fee: float | Factor = Field(default=0, title="卖出费率")
        sell_amt_limit: Optional[Factor] = Field(default=None, title="卖出金额限制")

    __QS_CallLayout__ = {name: {"keyword_only": True}
                        for name in __QS_InputClass__.model_fields if name not in ("last_price", "signal")}

    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        ModelArgs: _TimingAccountArgs = Field(default_factory=_TimingAccountArgs, title="模型参数")

    def __init__(self, init_cash: float = 1e6, start_dt: Optional[dt.datetime] = None,
                 args: dict = {}, config_file: Optional[str] = None, **kwargs):
        """初始化独立择时账户。

        Args:
            init_cash: 每次入池注入每个标的的固定本金，必须为有限正数。
            start_dt: 账户起始时点，None 表示从计算的第一个时点开始。
            args: 算子参数集。
            config_file: 配置文件地址。
            **kwargs: 传递给父类的其他参数。
        """
        Args = {"Name": "makeTimingAccount"} | args
        Args["ModelArgs"] = Args.get("ModelArgs", {}) | {"init_cash": init_cash, "short_allowed": False}
        if not np.isfinite(init_cash) or init_cash <= 0:
            raise __QS_Error__("每个标的的初始资金 init_cash 必须是有限正数")
        Args.update({"DTMode": "单时点", "DataType": "object", "iInitFactor": 0,
                     "LookBack": [1] + [0] * 11, "StartDT": [start_dt] + [None] * 11})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [None])[0]] * 12
        Args["CompoundType"] = [(name, "double") for name in (
            "Cash", "Position", "Amount", "Signal", "TradeNum", "TradePrice", "Fee",
            "Active", "Deposit", "Withdrawal", "LastPrice",
        )]
        PanelOperator.__init__(self, args=Args, config_file=config_file, **kwargs)

    def __call__(self, last_price: Factor, signal: Factor, *, mask: Optional[Factor] = None,
                 buy_price: Optional[Factor] = None, buy_limit: Optional[Factor] = None,
                 buy_fee: float | Factor = 0, buy_amt_limit: Optional[Factor] = None,
                 sell_price: Optional[Factor] = None, sell_limit: Optional[Factor] = None,
                 sell_fee: float | Factor = 0, sell_amt_limit: Optional[Factor] = None,
                 factor_args: dict = {}, **kwargs) -> PanelOperation:
        """构造按标的记录现金、交易和出入金的复合账户因子。

        Args:
            last_price: 最新估值价，缺失时沿用最近有效估值价，缺失报价不成交。
            signal: 方向信号，1 看多，0、-1 或缺失值空仓。
            mask: 标的池因子，True 或 1 表示入池，None 表示全部入池。
            buy_price: 买入报价，None 使用最新价。
            buy_limit: 等于 1 时禁止买入。
            buy_fee: 买入费率。
            buy_amt_limit: 每个时点每个标的的买入金额上限。
            sell_price: 卖出报价，None 使用最新价。
            sell_limit: 等于 1 时禁止卖出。
            sell_fee: 卖出费率。
            sell_amt_limit: 每个时点每个标的的卖出金额上限。
            factor_args: 账户因子参数。
            **kwargs: 传递给账户因子的其他参数。

        Returns:
            复合账户因子。Deposit、Withdrawal 均为正金额，净现金流为二者之差；
            Active 表示资金仍在子账户内，LastPrice 保存最近有效估值价。
        """
        init_account = DataFactor((0, 0, 0, np.nan, 0, np.nan, 0, 0, 0, 0, np.nan))
        factors = [init_account, last_price, signal, DataFactor(True) if mask is None else mask,
                   last_price if buy_price is None else buy_price,
                   DataFactor(False) if buy_limit is None else buy_limit, buy_fee,
                   DataFactor(np.inf) if buy_amt_limit is None else buy_amt_limit,
                   last_price if sell_price is None else sell_price,
                   DataFactor(False) if sell_limit is None else sell_limit, sell_fee,
                   DataFactor(np.inf) if sell_amt_limit is None else sell_amt_limit]
        return PanelOperator.__call__(self, *factors, factor_args=factor_args, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str],
                  x: List[np.ndarray], args: dict) -> np.ndarray:
        """在各子账户内独立撮合，并在入池和出池结算时记录现金流。

        Args:
            f: 当前账户因子。
            idt: 当前计算窗口。
            iid: 标的截面。
            x: 前期账户、价格、信号、标的池及交易条件。
            args: 模型参数。

        Returns:
            当前时点的复合子账户状态。
        """
        previous = x[0].astype(self._QSArgs.CompoundType)[-2]
        price, signal, mask, buy_price, buy_limit, buy_fee, buy_amount, sell_price, sell_limit, sell_fee, sell_amount = [
            np.asarray(data[-1], dtype=float) for data in x[1:]
        ]
        valuation = np.where(np.isfinite(price) & (price > 0), price, previous["LastPrice"])
        rows = []
        for i in range(len(iid)):
            cash, position = previous["Cash"][i], previous["Position"][i]
            active, in_pool = previous["Active"][i] == 1, mask[i] == 1
            deposit = withdrawal = 0.0
            if in_pool and not active:
                cash, active, deposit = args["init_cash"], True, args["init_cash"]
            order = 0.0
            if active:
                if in_pool and signal[i] == 1:
                    if np.isfinite(buy_price[i]) and buy_price[i] > 0:
                        order = max(0.0, cash) / (buy_price[i] * (1 + buy_fee[i]))
                else:
                    order = -position
            values = [np.array([value]) for value in (
                position, price[i], buy_price[i], buy_limit[i], buy_amount[i],
                sell_price[i], sell_limit[i], sell_amount[i],
            )]
            buy_vol, sell_vol = self._getTradeLimit(*values)
            buy_vol[~np.isfinite(values[2])] = 0
            sell_vol[~np.isfinite(values[5])] = 0
            cash, positions, trades, prices, fees = self._matchMarketOrder(
                cash, values[0], np.array([order]), values[2], buy_vol, np.array([buy_fee[i]]),
                values[5], sell_vol, np.array([sell_fee[i]]),
            )
            position = positions[0]
            if abs(position) < _QS_MinPositionNum:
                position = 0.0
            if active and not in_pool and position == 0:
                withdrawal, cash, active = cash, 0.0, False
            if abs(cash) < _QS_MinCash:
                cash = 0.0
            amount = position * valuation[i] if position != 0 else 0.0
            rows.append((cash, position, amount, signal[i], trades[0], prices[0], fees[0],
                         float(active), deposit, withdrawal, valuation[i]))
        return unstructured_to_structured(np.asarray(rows), dtype=np.dtype(self._QSArgs.CompoundType)).astype("O")


def _trade_details(account: pd.DataFrame, fields: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从实际成交重建已平仓交易与未平仓交易。

    Args:
        account: 账户数据，用于取得时间和标的标签。
        fields: 复合账户各字段的二维数组。

    Returns:
        已平仓明细和未平仓明细；收益均扣除实际交易费用。
    """
    columns = ["ID", "开仓时点", "平仓时点", "买入金额", "卖出金额", "交易费", "盈亏金额", "收益率", "持仓天数"]
    closed, opened = [], []
    for j, identifier in enumerate(account.columns):
        start, cost, proceeds, fees, buy_fees = None, 0.0, 0.0, 0.0, 0.0
        for i, timestamp in enumerate(account.index):
            quantity, price = fields["TradeNum"][i, j], fields["TradePrice"][i, j]
            if quantity > 0:
                if start is None:
                    start = timestamp
                cost += quantity * price
                buy_fees += fields["Fee"][i, j]
            elif quantity < 0 and start is not None:
                proceeds -= quantity * price
            if quantity != 0 and start is not None:
                fees += fields["Fee"][i, j]
            if start is not None and fields["Position"][i, j] == 0:
                profit = proceeds - cost - fees
                closed.append([identifier, start, timestamp, cost, proceeds, fees, profit,
                               profit / (cost + buy_fees) if cost else np.nan, (timestamp - start).total_seconds() / 86400])
                start, cost, proceeds, fees, buy_fees = None, 0.0, 0.0, 0.0, 0.0
        if start is not None:
            profit = proceeds + fields["Amount"][-1, j] - cost - fees
            opened.append([identifier, start, pd.NaT, cost, proceeds, fees, profit,
                           profit / (cost + buy_fees) if cost else np.nan,
                           (account.index[-1] - start).total_seconds() / 86400])
    return pd.DataFrame(closed, columns=columns), pd.DataFrame(opened, columns=columns)


def _timing_statistics(trades: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    """按已平仓交易汇总总账户和各标的择时指标。

    Args:
        trades: 扣除交易费用的已平仓交易明细。
        ids: 需要统计的全部标的。

    Returns:
        包含胜率、赔率、盈亏比和持仓时长等指标的统计表。
    """
    result = {}
    for name, data in [("总计", trades)] + [(identifier, trades[trades["ID"] == identifier]) for identifier in ids]:
        profit = data["盈亏金额"].astype(float)
        wins, losses = profit[profit > _QS_MinCash], profit[profit < -_QS_MinCash]
        ratio = wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan
        result[name] = pd.Series({
            "交易次数": len(data), "盈利次数": len(wins), "亏损次数": len(losses),
            "持平次数": len(data) - len(wins) - len(losses),
            "胜率": len(wins) / len(data) if len(data) else np.nan,
            "赔率": ratio, "盈亏比": ratio,
            "平均盈利金额": wins.mean(), "平均亏损金额": -losses.mean(),
            "盈利因子": wins.sum() / -losses.sum() if len(losses) else np.nan,
            "总盈亏": profit.sum(), "平均收益率": data["收益率"].mean(),
            "平均持仓天数": data["持仓天数"].mean(),
        })
    return pd.DataFrame(result)


class TimingAccountStats(_BaseAccountStats):
    """独立择时账户统计，汇总出入金、资金流调整净值及实际择时交易。"""

    def init_compute(self, path, init_data, context):
        """请求从账户起始时点开始的数据，以完整识别跨统计区间的交易。

        Args:
            path: 节点调用路径。
            init_data: 统计区间。
            context: 因子计算上下文。

        Returns:
            账户完整历史与可选基准的初始化请求。
        """
        requests = list(BTNode.init_compute(self, path, init_data, context))
        start = self.Deps[0].Operator.Args.StartDT[0] or context.DTRuler[0]
        requests[0] = FactorInitData(DTRange=(start, requests[0].DTRange[1]), SectionIDs=self._QSArgs.AccountSection)
        if self._HasBmk:
            requests[1] = FactorInitData(DTRange=requests[1].DTRange,
                                        SectionIDs=[self._QSArgs.BmkID] if self._QSArgs.BmkID else None)
        return requests

    def forward_compute(self, path, fwd_data, context):
        """获取账户完整历史，同时保留报告实际请求的统计时点。

        Args:
            path: 节点调用路径。
            fwd_data: 报告请求的统计时点。
            context: 因子计算上下文。

        Returns:
            依赖数据请求和统计本地上下文。
        """
        start = self.Deps[0].Operator.Args.StartDT[0] or context.DTRuler[0]
        ids = self._QSArgs.AccountSection or self.Deps[0].Args.SectionIDs or context.SectionIDs
        dates = [timestamp for timestamp in context.DTRuler if start <= timestamp <= fwd_data.DTs[-1]]
        requests = [FactorLocalContext(DTs=dates, IDs=ids, SectionIDs=ids, PIDs=context.PIDList)]
        if self._HasBmk:
            benchmark_ids = [self._QSArgs.BmkID] if self._QSArgs.BmkID else self.Deps[1].Args.SectionIDs or context.SectionIDs
            requests.append(FactorLocalContext(DTs=fwd_data.DTs, IDs=benchmark_ids,
                                               SectionIDs=benchmark_ids, PIDs=context.PIDList))
        return requests, DTLocalContext(DTs=fwd_data.DTs)

    def backward_compute(self, path, bwd_data_list, context, local_context=None) -> dict:
        """生成固定本金择时账户统计。

        Args:
            path: 节点调用路径。
            bwd_data_list: 账户及可选基准净值。
            context: 因子计算上下文。
            local_context: 当前统计时点。

        Returns:
            账户时间序列、资金流水、成交明细、择时明细及统计数据。
        """
        full = bwd_data_list[0]
        account = full.reindex(index=local_context.DTs)
        names = [name for name, _ in self.Deps[0].Operator.Args.CompoundType]
        structured = account.values.astype(np.dtype([(name, float) for name in names]))
        fields = {name: structured[name] for name in names}
        cash, amount = fields["Cash"].sum(axis=1), fields["Amount"].sum(axis=1)
        value = cash + amount
        deposit, withdrawal = fields["Deposit"].sum(axis=1), fields["Withdrawal"].sum(axis=1)
        before = full.loc[full.index < account.index[0]]
        initial = (sum(record[0] + record[2] for record in before.iloc[-1]) if len(before) else 0.0)
        previous = np.r_[initial, value[:-1]]
        earnings = value - previous - deposit + withdrawal
        capital = previous + deposit
        returns = np.divide(earnings, capital, out=np.zeros_like(earnings), where=capital > _QS_MinCash)
        nv = np.cumprod(1 + returns)
        timeseries = pd.DataFrame({
            "现金": cash, "负债": 0.0, "证券": amount, "账户价值": value,
            "入金": deposit, "出金": withdrawal, "现金流": deposit - withdrawal,
            "累计入金": deposit.cumsum(), "累计出金": withdrawal.cumsum(),
            "累计资金投入": initial + (deposit - withdrawal).cumsum(),
            "收益": earnings, "收益率": returns, "累计收益率": returns.cumsum(), "净值": nv,
        }, index=account.index)
        records = []
        for i, timestamp in enumerate(account.index):
            for j, identifier in enumerate(account.columns):
                for field, sign, reason in (("Deposit", 1, "入池入金"), ("Withdrawal", -1, "出池清仓出金")):
                    if fields[field][i, j] != 0:
                        records.append((timestamp, identifier, sign * fields[field][i, j], reason))
        cash_record = pd.DataFrame(records, columns=["时间点", "ID", "现金流", "备注"])
        trades = []
        for i, j in zip(*np.nonzero(fields["TradeNum"])):
            trades.append((account.index[i], account.columns[j], fields["TradeNum"][i, j],
                           fields["TradePrice"][i, j], fields["Fee"][i, j]))
        full_structured = full.values.astype(np.dtype([(name, float) for name in names]))
        closed, opened = _trade_details(full, {name: full_structured[name] for name in names})
        closed = closed[closed["平仓时点"].isin(account.index)]
        statistics = summaryStrategy(nv, list(account.index), init_wealth=[1], risk_free_rate=self._QSArgs.RiskFreeRate)
        statistics.columns = ["绝对表现"]
        output = {"初始资金": initial, "时间序列": timeseries, "资金流水": cash_record,
                  "现金记录": cash_record, "统计数据": statistics,
                  "交易信号": pd.DataFrame(fields["Signal"], index=account.index, columns=account.columns),
                  "持仓数量": pd.DataFrame(fields["Position"], index=account.index, columns=account.columns),
                  "子账户现金": pd.DataFrame(fields["Cash"], index=account.index, columns=account.columns),
                  "子账户权益": pd.DataFrame(fields["Cash"] + fields["Amount"], index=account.index, columns=account.columns),
                  "交易记录": pd.DataFrame(trades, columns=["交易时点", "ID", "交易量", "成交价", "交易费"]),
                  "择时交易": closed, "未平仓交易": opened,
                  "择时统计": _timing_statistics(closed, account.columns.tolist())}
        if self._HasBmk:
            benchmark = bwd_data_list[1].reindex(index=account.index).iloc[:, 0].astype(float)
            benchmark = benchmark / benchmark.iloc[0]
            timeseries["基准净值"] = benchmark
            timeseries["相对净值"] = nv / benchmark
        return output


class TimingAccountReport(_BaseAccountReport):
    """展示择时账户收益、出入金流水和按实际交易计算的择时指标。"""

    def backward_compute(self, path, bwd_data_list, context, local_context=None) -> dict:
        """将择时统计输出为 HTML 报告。

        Args:
            path: 节点调用路径。
            bwd_data_list: 择时账户统计。
            context: 计算上下文。
            local_context: 报告本地上下文。

        Returns:
            增加 HTML 报告的账户统计字典。
        """
        output = dict(bwd_data_list[0])
        html = "<h3>账户绩效</h3>" + formatStrategySummary(output["统计数据"]).to_html()
        for title in ("择时统计", "资金流水", "择时交易", "未平仓交易"):
            html += f"<h3>{title}</h3>" + output[title].to_html()
        output[self._QSArgs.ReportKey] = html
        return output
