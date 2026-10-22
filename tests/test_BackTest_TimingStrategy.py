# -*- coding: utf-8 -*-
"""择时策略测试：验证组合权重与独立子账户，可直接运行生成完整 HTML 报告。

实现逻辑：使用固定随机种子生成三个模拟标的的开收盘行情，以昨日的
5/20 日收盘均线关系生成方向信号，在当日开盘价成交、收盘价估值。
每个标的入池注入 10 万元，独立交易并复投，出池清仓后提走余额。
通过 MakeTimingAccount、TimingAccountStats、TimingAccountReport 和真实引擎
执行回测，核对资金守恒及已实现和未实现盈亏，再输出报告、图表和 CSV。

使用方法：在项目根目录运行 ``python tests/test_BackTest_TimingStrategy.py``。
可用 ``--output-dir 路径`` 指定目录，默认 ``tests/data/timing_demo``。
通过 ``python -m unittest discover -s tests -p 'test_BackTest_TimingStrategy.py'``
运行自动测试时，报告写入临时目录并在测试完成后清理。
生成的 ``择时策略报告.html`` 内嵌所有图片，无需网络；目录内同时保留
账户时间序列、资金流水、逐笔成交、择时统计、输入行情和摘要 JSON。
模拟数据仅用于展示框架行为，不能代表任何真实证券的历史表现。
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from QuantStudio.BackTest.Strategy.api import TimingStrategy
from QuantStudio.BackTest.Strategy.AllocationStrategy import CalcPortfolioNV
from QuantStudio.BackTest.Strategy.Strategy import MakeAccount
from QuantStudio.BackTest.Strategy.TimingStrategy import (
    MakeTimingAccount, TimingAccountReport, TimingAccountStats,
)
from QuantStudio.Core import setDefaultLogLevel
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Node import DTInitData, DTLocalContext
from QuantStudio.Factor import FactorOperator as fo
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorInitData, FactorLocalContext


class TestTimingStrategy(unittest.TestCase):
    """通过计算引擎验证真实因子与账户输出。"""

    def setUp(self):
        self.ids = ["A", "B", "C", "D"]
        self.dates = [dt.datetime(2026, 1, 1) + dt.timedelta(days=i) for i in range(7)]
        self.signal_data = np.array([
            [1, 1, 1, 1],
            [1, 0, -1, 1],
            [0, 1, 1, 1],
            [-1, -1, np.nan, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ])
        self.signal = self._factor(self.signal_data)
        self.mask = self._factor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [np.nan, 1, 0, 0],
        ])
        self.expected = np.array([
            [1/3, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/4, 1/4, 1/4, 1/4],
            [0, 1, 0, 0],
        ])

    def _factor(self, values):
        """将确定性测试数据包装为因子。

        Args:
            values: 时点与标的组成的二维数据。

        Returns:
            数据因子。
        """
        return DataFactor(pd.DataFrame(values, index=self.dates, columns=self.ids))

    def _run(self, factor, ids=None):
        """使用完整标尺运行因子，预留一个账户初始化时点。

        Args:
            factor: 待运行因子。
            ids: 输出标的，None 使用全部标的。

        Returns:
            引擎计算结果。
        """
        ids = self.ids if ids is None else ids
        with FactorContext(DTRuler=self.dates, SectionIDs=self.ids) as context:
            with Engine() as engine:
                result, = engine.run(
                    [factor], context,
                    fwd_data_list=[FactorLocalContext(DTs=self.dates[1:], IDs=ids)],
                    init_data_list=[FactorInitData(DTRange=(self.dates[1], self.dates[-1]))],
                )
        return result

    def test_equal_budget_and_empty_pool(self):
        """分母包含池内中性和看空标的，空池与缺失信号输出零仓位。"""
        portfolio = TimingStrategy.CalcTimingPortfolio()(self.signal, mask=self.mask)
        np.testing.assert_allclose(self._run(portfolio).values, self.expected)

    def test_default_mask_and_boolean_mask(self):
        """省略标的池时等分整个截面，也接受布尔池因子。"""
        operator = TimingStrategy.CalcTimingPortfolio()
        portfolio = operator(self.signal)
        np.testing.assert_allclose(self._run(portfolio).values,
                                   (self.signal_data[1:] == 1) / 4)
        mask = self._factor(np.ones((7, 4), dtype=bool))
        np.testing.assert_allclose(self._run(operator(self.signal, mask)).values,
                                   self._run(portfolio).values)

    def test_descriptor_section_and_binding(self):
        """输出部分标的时仍按完整池分配，部分绑定保持公开输入语义。"""
        operator = TimingStrategy.CalcTimingPortfolio(descriptor_ids=self.ids)
        portfolio = operator.bind(mask=self.mask)(self.signal)
        np.testing.assert_allclose(self._run(portfolio, ids=["B", "A"]).values,
                                   self.expected[:, [1, 0]])

    def test_trade_dates_keep_nontrading_rows_missing(self):
        """指定交易时点时仅在交易时点给出目标权重。"""
        portfolio = TimingStrategy.CalcTimingPortfolio()(
            self.signal, self.mask, factor_args={"CalcDTRuler": self.dates[1::2]}
        )
        expected = self.expected.copy()
        expected[1::2] = np.nan
        np.testing.assert_allclose(self._run(portfolio).values, expected, equal_nan=True)

    def test_account_liquidates_and_matches_vector_backtest(self):
        """信号转中性、看空或移出池时清仓，并与向量化回测净值一致。"""
        price_data = 100 * np.power([1.1, 1.2, 1.3, 1.4], np.arange(7).reshape(-1, 1))
        price = self._factor(price_data)
        portfolio = TimingStrategy.CalcTimingPortfolio()(self.signal, self.mask)
        account = MakeAccount(init_cash=1200, short_allowed=False, start_dt=self.dates[1])(
            last_price=price, signal=portfolio
        )
        result = self._run(account)
        amount = result.map(lambda value: value[2]).to_numpy(dtype=float)
        cash = result.iloc[:, 0].map(lambda value: value[0]).to_numpy(dtype=float)
        value = cash + amount.sum(axis=1)
        np.testing.assert_allclose(value, [1200, 1240, 1364, 1364, 1364, 1705])
        np.testing.assert_allclose(amount / value[:, None], self.expected, atol=1e-12)
        np.testing.assert_allclose(cash, value * (1 - self.expected.sum(axis=1)), atol=1e-10)
        vector = CalcPortfolioNV(descriptor_ids=self.ids, start_dt=self.dates[1])(
            portfolio, price=price, init_nv=1200, portfolio_name_list=["择时"]
        )
        np.testing.assert_allclose(self._run(vector, ids=["择时"]).iloc[:, 0].to_numpy(dtype=float), value)


def build_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成固定种子的模拟行情，以及包含出池和再入池的标的池。

    Returns:
        开盘价、收盘价和标的池，包含 40 个预热时点和 260 个回测时点。
    """
    rng = np.random.default_rng(20261020)
    dates = pd.bdate_range("2025-01-02", periods=300)
    ids = ["DEMO_A", "DEMO_B", "DEMO_C"]
    t = np.arange(len(dates))[:, None]
    changes = .0002 + .0045 * np.sin(t / np.array([12, 18, 9]) + [0, 1.5, 3])
    changes = changes + rng.normal(0, .011, size=(len(dates), 3))
    close = pd.DataFrame([100, 80, 120] * np.exp(np.cumsum(changes, axis=0)), index=dates, columns=ids)
    previous = close.shift(1).fillna(pd.Series([100, 80, 120], index=ids))
    opening = previous * np.exp(rng.normal(0, .003, size=close.shape))
    mask = pd.DataFrame(False, index=dates, columns=ids)
    mask.iloc[40:, 0] = True
    mask.iloc[145:175, 0] = False
    mask.iloc[65:200, 1] = True
    mask.iloc[105:, 2] = True
    return opening, close, mask


def run_backtest(opening: pd.DataFrame, close: pd.DataFrame, mask: pd.DataFrame) -> dict:
    """使用正式择时账户和报告节点执行均线策略。

    Args:
        opening: 当日开盘成交价。
        close: 当日收盘估值价。
        mask: 当日开盘前已知的模拟标的池。

    Returns:
        含正式报告 HTML、资金流水和交易统计的回测结果。
    """
    dates = list(close.index.to_pydatetime())
    test_dates = dates[40:]
    price = DataFactor(close, args={"Name": "模拟收盘价"})
    open_price = DataFactor(opening, args={"Name": "模拟开盘价"})
    ma_short = fo.RollingMean(window=5, min_periods=5)(price)
    ma_long = fo.RollingMean(window=20, min_periods=20)(price)
    direction = fo.Lag(lag_period=1)(fo.Where()(1, ma_short >= ma_long, -1))
    account = MakeTimingAccount(init_cash=100_000, start_dt=test_dates[0])(
        last_price=price, signal=direction, mask=DataFactor(mask),
        buy_price=open_price, sell_price=open_price, buy_fee=.0003, sell_fee=.0003,
        factor_args={"Name": "5/20日均线独立择时账户"},
    )
    stats = TimingAccountStats(account, args={"AccountSection": list(close.columns)})
    report = TimingAccountReport(stats)
    with FactorContext(DTRuler=dates, SectionIDs=list(close.columns)) as context:
        with Engine() as engine:
            output, = engine.run(
                [report], context,
                fwd_data_list=[DTLocalContext(DTs=test_dates)],
                init_data_list=[DTInitData(DTRange=(test_dates[0], test_dates[-1]))],
            )
    return output


def validate_output(output: dict) -> dict:
    """核对资金守恒、资金流水及交易盈亏，生成可读摘要。

    Args:
        output: 真实回测引擎输出。

    Returns:
        回测参数、关键指标与核对结果。
    """
    series = output["时间序列"]
    profit = series["账户价值"] + series["累计出金"] - series["累计入金"]
    np.testing.assert_allclose(profit, series["收益"].cumsum(), atol=1e-7)
    np.testing.assert_allclose(output["资金流水"]["现金流"].sum(), series["现金流"].sum(), atol=1e-7)
    np.testing.assert_allclose(
        output["择时交易"]["盈亏金额"].sum() + output["未平仓交易"]["盈亏金额"].sum(),
        series["收益"].sum(), atol=1e-7,
    )
    stats = output["择时统计"]["总计"]
    return {
        "数据来源": "固定随机种子 20261020 生成的模拟行情，非真实证券历史",
        "开始日期": series.index[0].strftime("%Y-%m-%d"),
        "结束日期": series.index[-1].strftime("%Y-%m-%d"), "回测时点数": len(series),
        "每标的入池本金": 100_000, "单边费率": .0003,
        "净值收益率": float(series["净值"].iloc[-1] - 1),
        "最大回撤": float((1 - series["净值"] / series["净值"].cummax().clip(lower=1)).max()),
        "期末权益": float(series["账户价值"].iloc[-1]),
        "累计入金": float(series["入金"].sum()), "累计出金": float(series["出金"].sum()),
        "累计净收益": float(profit.iloc[-1]),
        "已平仓交易数": int(stats["交易次数"]), "胜率": float(stats["胜率"]),
        "赔率": float(stats["赔率"]), "未平仓交易数": len(output["未平仓交易"]),
        "交易费用": float(output["交易记录"]["交易费"].sum()),
        "核对结果": "逐日资金守恒、净资金流水、已实现和未实现盈亏三项核对通过",
    }


def make_figures(output: dict, close: pd.DataFrame, mask: pd.DataFrame, directory: Path) -> list[Path]:
    """生成账户概览和带实际成交标记的逐标的行情图。

    Args:
        output: 账户回测输出。
        close: 模拟收盘价。
        mask: 模拟标的池。
        directory: 图片输出目录。

    Returns:
        两张可单独分享的 PNG 图片路径。
    """
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.labelcolor": "#526477", "text.color": "#183449"})
    series = output["时间序列"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    nv = series["净值"]
    axes[0, 0].plot(nv.index, nv, color="#087f8c", linewidth=2)
    axes[0, 0].axhline(1, color="#cbd5df", linewidth=1)
    axes[0, 0].set_title("资金流调整后的账户净值")
    drawdown = nv / nv.cummax().clip(lower=1) - 1
    axes[0, 1].fill_between(drawdown.index, drawdown, 0, color="#d56b67", alpha=.45)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].set_title("净值回撤")
    axes[1, 0].bar(series.index, series["入金"] / 10_000, width=3, label="入金", color="#087f8c")
    axes[1, 0].bar(series.index, -series["出金"] / 10_000, width=3, label="出金", color="#d56b67")
    axes[1, 0].set_title("标的进出池的资金流水（万元）")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(series.index, series["账户价值"] / 10_000, label="账户权益", color="#183449")
    axes[1, 1].plot(series.index, series["累计资金投入"] / 10_000, label="累计净投入", color="#8096a4", linestyle="--")
    axes[1, 1].plot(series.index, series["收益"].cumsum() / 10_000, label="累计净收益", color="#087f8c")
    axes[1, 1].set_title("权益与资金核对（万元）")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    overview = directory / "账户概览.png"
    fig.savefig(overview, dpi=150, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(len(close.columns), 1, figsize=(14, 10), layout="constrained")
    for ax, identifier in zip(axes, close.columns):
        prices = close[identifier].reindex(series.index)
        ax.plot(prices.index, prices, label="收盘价", color="#183449", linewidth=1.2)
        for window, color in ((5, "#087f8c"), (20, "#d89c43")):
            ma = close[identifier].rolling(window).mean().reindex(series.index)
            ax.plot(ma.index, ma, label=f"{window}日均线", color=color, linewidth=1)
        trades = output["交易记录"].query("ID == @identifier")
        for positive, marker, color, label in ((True, "^", "#c65f5f", "实际买入"), (False, "v", "#258b6a", "实际卖出")):
            data = trades[(trades["交易量"] > 0) == positive]
            ax.scatter(data["交易时点"], data["成交价"], marker=marker, color=color, s=28, label=label, zorder=4)
        excluded = ~mask[identifier].reindex(series.index)
        ax.fill_between(prices.index, 0, 1, where=excluded, transform=ax.get_xaxis_transform(),
                        color="#b8c4cd", alpha=.3, label="不在标的池")
        ax.set_title(identifier + " · 模拟标的")
        ax.legend(loc="upper left", ncol=6, fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=.15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    trade_path = directory / "标的行情与成交.png"
    fig.savefig(trade_path, dpi=150, facecolor="white")
    plt.close(fig)
    return [overview, trade_path]


def save_report(output: dict, summary: dict, images: list[Path], directory: Path) -> Path:
    """将正式报告、图表、参数和核对结果组装为离线 HTML。

    Args:
        output: TimingAccountReport 生成的原始报告与明细。
        summary: 核对后的回测摘要。
        images: 账户与逐标的图表。
        directory: 报告输出目录。

    Returns:
        最终 HTML 报告路径。
    """
    cards = [("净值收益率", f'{summary["净值收益率"]:.2%}'),
             ("最大回撤", f'{summary["最大回撤"]:.2%}'),
             ("已平仓胜率", f'{summary["胜率"]:.2%}'),
             ("赔率", f'{summary["赔率"]:.2f}'),
             ("累计净收益", f'{summary["累计净收益"]:,.2f} 元'),
             ("已平仓交易", str(summary["已平仓交易数"]) + " 笔")]
    card_html = "".join(f'<div class="card"><span>{title}</span><strong>{value}</strong></div>' for title, value in cards)
    figures = "".join('<img alt="回测图表" src="data:image/png;base64,' +
                      base64.b64encode(path.read_bytes()).decode("ascii") + '">' for path in images)
    series_html = output["时间序列"].to_html(float_format=lambda value: f"{value:,.4f}")
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>均线择时策略回测报告 · 模拟行情</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:#183449;font:15px/1.8 "Microsoft YaHei",sans-serif}}
header{{background:#183449;color:white;padding:38px max(5vw,24px)}}header p{{color:#c9dce3;margin:6px 0}}
header h1{{margin:8px 0;font-size:30px}}.tag{{color:#8ce2d8;font-size:13px;letter-spacing:2px}}
main{{max-width:1280px;margin:24px auto;padding:0 20px}}section{{background:white;border:1px solid #e0e7ed;border-radius:12px;padding:26px;margin-bottom:22px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}}.card{{background:white;padding:20px;border-radius:10px;border:1px solid #e0e7ed}}
.card span{{display:block;color:#65798c;font-size:13px}}.card strong{{display:block;font-size:27px;margin-top:4px}}
img{{display:block;width:100%;height:auto}}h2{{font-size:21px;margin-top:0}}h3{{margin-top:32px;font-size:18px}}
.tables{{overflow-x:auto}}table{{border-collapse:collapse;min-width:700px;width:100%;font-size:13px;margin:16px 0}}
th,td{{padding:9px 12px;white-space:nowrap;border:1px solid #e4ebf1;text-align:right}}th{{background:#edf4f7}}tr:nth-child(even) td{{background:#f8fafb}}
.note{{background:#edf7f5;border-left:4px solid #087f8c;padding:16px 20px}}.muted{{color:#65798c}}details summary{{cursor:pointer;font-weight:bold}}
@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:16px}}.card strong{{font-size:22px}}}}
@media print{{body{{background:white}}section{{break-inside:avoid}}header{{padding:20px}}.card{{padding:10px}}}}
</style></head><body><header><div class="tag">模拟行情 / 独立子账户 / 固定本金</div>
<h1>5 / 20 日均线择时策略</h1><p>{summary["开始日期"]} 至 {summary["结束日期"]} · {summary["回测时点数"]} 个模拟交易日 · 3 个模拟标的</p>
<p>通过 QuantStudio 正式计算引擎和 TimingAccountReport 节点生成</p></header><main>
<div class="cards">{card_html}</div>
<section><h2>本次如何回测</h2><p class="note">使用固定种子生成的模拟行情。工作日序列未按任何真实交易所节假日调整；这些结果用于展示账户与报告功能，不能代表真实证券历史表现。</p>
<p>以前一日收盘的 5 日与 20 日均线比较生成信号，短均线不低于长均线时看多，否则空仓。
信号滞后一天，当日开盘成交、收盘估值。每个标的入池注入 100,000 元，盈亏独立复投。
买入、卖出费率均为 0.03%，成交数量允许小数，不额外模拟滑点、印花税、停牌和涨跌停。</p>
<p>A 全程入池，期间退出一次再入池；B 延迟入池并提前退出；C 延迟入池后保留至期末。
未平仓交易仅展示浮动盈亏，不计入胜率与赔率。赔率是平均盈利金额 / 平均亏损金额的绝对值，均已扣费。</p></section>
<section><h2>净值、现金流与实际成交</h2>{figures}</section>
<section><h2>资金核对</h2><p>累计入金 <b>{summary["累计入金"]:,.2f}</b> 元，累计出金 <b>{summary["累计出金"]:,.2f}</b> 元，
期末账户权益 <b>{summary["期末权益"]:,.2f}</b> 元。</p>
<p>期末权益 + 累计出金 − 累计入金 = <b>{summary["累计净收益"]:,.2f}</b> 元；
实际交易费用合计 <b>{summary["交易费用"]:,.2f}</b> 元。</p>
<p class="note">{summary["核对结果"]}。净值按资金流调整后的日收益连乘，不能用期末权益直接除以累计入金。</p></section>
<section class="tables"><h2>正式报告输出</h2>{output["Report"]}</section>
<section class="tables"><details><summary>查看完整账户时间序列</summary>{series_html}</details></section>
<section><h2>复现与原始数据</h2><p>运行仓库内 <code>python tests/test_BackTest_TimingStrategy.py</code> 可复现本报告。
模拟数据随机种子为 20261020。报告同目录保存全部 CSV 和摘要 JSON；本 HTML 已内嵌图片，可离线打开。</p>
<p class="muted">回测路径：MakeTimingAccount → TimingAccountStats → TimingAccountReport → Engine。</p></section>
</main></body></html>'''
    report_path = directory / "择时策略报告.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


class TestTimingStrategyDemo(unittest.TestCase):
    """使用固定行情验证资金流、成交时序及完整离线报告。"""

    ReportDirectory: Path | None = None

    @classmethod
    def setUpClass(cls):
        """执行一次完整回测，并核对资金守恒与交易盈亏。"""
        cls.opening, cls.close, cls.mask = build_inputs()
        cls.output = run_backtest(cls.opening, cls.close, cls.mask)
        cls.summary = validate_output(cls.output)

    def test_account_cash_flows_and_trade_outcomes(self):
        """进出池产生独立资金流水，已平仓和未平仓交易分别统计。"""
        series = self.output["时间序列"]
        flows = self.output["资金流水"]
        self.assertEqual(len(series), 260)
        self.assertEqual((flows["现金流"] > 0).sum(), 4)
        self.assertEqual((flows["现金流"] < 0).sum(), 2)
        self.assertAlmostEqual(series["入金"].sum(), 400_000)
        stats = self.output["择时统计"]["总计"]
        self.assertEqual(stats["交易次数"], 17)
        self.assertEqual(stats["盈利次数"], 9)
        self.assertEqual(len(self.output["未平仓交易"]), 1)
        self.assertAlmostEqual(stats["胜率"], 9 / 17)

    def test_lagged_signals_and_opening_price_execution(self):
        """信号仅使用昨日收盘信息，实际成交使用当日开盘价并扣费。"""
        short = self.close.rolling(5, min_periods=5).mean()
        long = self.close.rolling(20, min_periods=20).mean()
        expected = pd.DataFrame(np.where(short >= long, 1, -1),
                                index=self.close.index, columns=self.close.columns).shift(1)
        signals = self.output["交易信号"]
        np.testing.assert_array_equal(signals.to_numpy(), expected.loc[signals.index].to_numpy())
        for _, trade in self.output["交易记录"].iterrows():
            with self.subTest(timestamp=trade["交易时点"], identifier=trade["ID"]):
                self.assertAlmostEqual(trade["成交价"], self.opening.loc[trade["交易时点"], trade["ID"]])
                self.assertAlmostEqual(trade["交易费"], abs(trade["交易量"]) * trade["成交价"] * .0003)
        self.assertTrue((self.output["持仓数量"] >= 0).all().all())

    def test_report_and_data_exports(self):
        """报告内嵌完整图表，并输出可核对的 CSV 与 JSON 原始结果。"""
        with tempfile.TemporaryDirectory(prefix="qs_timing_report_") as temporary:
            directory = self.ReportDirectory or Path(temporary)
            directory.mkdir(parents=True, exist_ok=True)
            with plt.rc_context():
                images = make_figures(self.output, self.close, self.mask, directory)
            report = save_report(self.output, self.summary, images, directory)
            for name in ("时间序列", "资金流水", "交易记录", "择时交易", "未平仓交易", "择时统计", "子账户权益", "交易信号"):
                self.output[name].to_csv(directory / f"{name}.csv", encoding="utf-8-sig")
            for name, frame in (("模拟开盘价", self.opening), ("模拟收盘价", self.close), ("模拟标的池", self.mask)):
                frame.to_csv(directory / f"{name}.csv", encoding="utf-8-sig")
            summary_path = directory / "回测摘要.json"
            summary_path.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            html = report.read_text(encoding="utf-8")
            self.assertEqual(html.count("data:image/png;base64,"), 3)
            self.assertIn("模拟行情", html)
            self.assertIn("tests/test_BackTest_TimingStrategy.py", html)
            for name in ("择时统计", "资金流水", "账户净值"):
                self.assertIn(name, html)
            self.assertNotIn("<h3>择时交易</h3>", html)
            self.assertNotIn("<h3>未平仓交易</h3>", html)
            pd.testing.assert_frame_equal(pd.read_csv(directory / "模拟收盘价.csv", index_col=0, parse_dates=True),
                                          self.close, check_freq=False)
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), self.summary)
            if self.ReportDirectory is not None:
                print(json.dumps(self.summary, ensure_ascii=False, indent=2))
                print(f"报告：{report}")


def main() -> None:
    """运行测试并将报告保存在指定目录，支持 unittest 的筛选和详细输出参数。"""
    parser = argparse.ArgumentParser(description="运行独立择时策略测试并生成报告", add_help=False)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tests/data/timing_demo")
    args, test_args = parser.parse_known_args()
    TestTimingStrategyDemo.ReportDirectory = args.output_dir.resolve()
    setDefaultLogLevel(logging.WARNING)
    unittest.main(argv=[__file__, *test_args])


if __name__ == "__main__":
    main()
