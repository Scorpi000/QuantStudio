# -*- coding: utf-8 -*-
"""验证固定本金择时账户的独立交易、现金流与交易统计。"""
import base64
import datetime as dt
import unittest

import numpy as np
import pandas as pd

from QuantStudio.BackTest.Strategy import TimingStrategy as ts
from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Node import DTLocalContext, DTInitData
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorInitData, FactorLocalContext


class TestTimingAccount(unittest.TestCase):
    """使用可手工核算的价格和方向信号验证账户与统计节点。"""

    def setUp(self):
        self.ids = ["A", "B"]
        self.dates = [dt.datetime(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
        self.price = self._factor([[10, 20], [10, 20], [12, 20], [14, 18],
                                   [10, 16], [11, 20], [11, 22], [20, 22]])
        self.signal = self._factor([[0, 0], [1, 1], [1, 1], [0, 1],
                                    [1, -1], [1, 1], [0, 0], [1, 0]])
        self.mask = self._factor([[0, 0], [1, 0], [1, 1], [1, 1],
                                  [1, 1], [0, 1], [0, 0], [1, 0]])

    def _factor(self, values):
        """将二维数据包装为确定性因子。

        Args:
            values: 时间和标的组成的二维数据。

        Returns:
            数据因子。
        """
        return DataFactor(pd.DataFrame(values, index=self.dates, columns=self.ids))

    def _account(self, **kwargs):
        """使用统一的初始本金创建待测试账户。

        Args:
            **kwargs: 交易条件。

        Returns:
            独立择时账户因子。
        """
        return ts.MakeTimingAccount(init_cash=1000, start_dt=self.dates[1])(
            last_price=self.price, signal=self.signal, mask=self.mask, **kwargs
        )

    def _run(self, node, statistics=False, start=1):
        """通过真实引擎计算账户或统计节点。

        Args:
            node: 账户因子或统计节点。
            statistics: 是否使用统计节点的时间上下文。
            start: 输出起始时点在标尺中的位置。

        Returns:
            节点输出。
        """
        dates = self.dates[start:]
        local = DTLocalContext(DTs=dates) if statistics else FactorLocalContext(DTs=dates, IDs=self.ids)
        initial = DTInitData(DTRange=(dates[0], dates[-1])) if statistics else FactorInitData(DTRange=(dates[0], dates[-1]))
        with FactorContext(DTRuler=self.dates, SectionIDs=self.ids) as context:
            with Engine() as engine:
                result, = engine.run([node], context, fwd_data_list=[local], init_data_list=[initial])
        return result

    def _fields(self, account):
        """把账户复合元组转换为具名数组。

        Args:
            account: 待计算的账户因子。

        Returns:
            包含全部复合字段的结构化数组。
        """
        result = self._run(account)
        return result.to_numpy().astype(np.dtype([(name, float) for name, _ in account.Operator.Args.CompoundType]))

    def test_independent_cash_compounds_without_rebalancing(self):
        """盈利留在原子账户，其他标的入池不改变已有持仓。"""
        fields = self._fields(self._account())
        np.testing.assert_allclose(fields["Position"], [[100, 0], [100, 50], [0, 50],
                                                        [140, 0], [0, 40], [0, 0], [50, 0]])
        np.testing.assert_allclose(fields["Cash"], [[0, 0], [0, 0], [1400, 0],
                                                    [0, 800], [0, 0], [0, 0], [0, 0]], atol=1e-10)
        np.testing.assert_allclose(fields["Deposit"].sum(axis=1), [1000, 1000, 0, 0, 0, 0, 1000])
        np.testing.assert_allclose(fields["Withdrawal"].sum(axis=1), [0, 0, 0, 0, 1540, 880, 0])

    def test_cashflows_and_completed_trade_statistics(self):
        """资金流不计入收益，四笔平仓交易与一笔未平仓交易分别统计。"""
        output = self._run(ts.TimingAccountStats(self._account()), statistics=True)
        series = output["时间序列"]
        self.assertAlmostEqual(series["收益"].sum(), 420)
        np.testing.assert_allclose(series["收益率"], [0, .1, 100/2200, -100/2300, 140/2200, .1, 0])
        self.assertAlmostEqual(output["资金流水"]["现金流"].sum(), 580)
        stats = output["择时统计"]["总计"]
        self.assertEqual(stats["交易次数"], 4)
        self.assertEqual(stats["盈利次数"], 3)
        self.assertAlmostEqual(stats["胜率"], .75)
        self.assertAlmostEqual(stats["赔率"], 31/30)
        self.assertAlmostEqual(stats["总盈亏"], 420)
        self.assertEqual(len(output["未平仓交易"]), 1)
        self.assertEqual(output["择时统计"]["A"]["交易次数"], 2)

    def test_flow_only_flat_account_and_simultaneous_flow(self):
        """中性账户只出入金不产生收益，空交易样本和报告可正常输出。"""
        signal = DataFactor(0)
        account = ts.MakeTimingAccount(init_cash=1000, start_dt=self.dates[1])(
            last_price=self.price, signal=signal, mask=self.mask
        )
        output = self._run(ts.TimingAccountStats(account), statistics=True)
        np.testing.assert_allclose(output["时间序列"]["净值"], 1)
        self.assertEqual(output["择时统计"].loc["交易次数", "总计"], 0)
        self.assertTrue(np.isnan(output["择时统计"].loc["胜率", "总计"]))
        self.assertEqual(len(output["资金流水"]), 5)
        report = self._run(ts.TimingAccountReport(ts.TimingAccountStats(account)), statistics=True)
        self.assertIn("择时统计", report["Report"])

    def test_simultaneous_equal_deposit_and_withdrawal(self):
        """相抵的入金和出金分别记账，净流为零且不影响收益。"""
        mask = self._factor([[0, 0], [1, 0], [0, 1], [0, 1],
                             [0, 1], [0, 1], [0, 1], [0, 1]])
        account = ts.MakeTimingAccount(init_cash=1000, start_dt=self.dates[1])(
            last_price=self.price, signal=DataFactor(0), mask=mask
        )
        output = self._run(ts.TimingAccountStats(account), statistics=True)
        series = output["时间序列"].loc[self.dates[2]]
        self.assertEqual(series["入金"], 1000)
        self.assertEqual(series["出金"], 1000)
        self.assertEqual(series["现金流"], 0)
        self.assertEqual(series["收益率"], 0)
        self.assertEqual(len(output["资金流水"].query("时间点 == @self.dates[2]")), 2)

    def test_sell_restriction_defers_withdrawal(self):
        """出池时不能卖出的标的继续持有，清仓后才出金。"""
        limit = np.zeros((8, 2))
        limit[5, 0] = 1
        fields = self._fields(self._account(sell_limit=self._factor(limit)))
        self.assertEqual(fields["Position"][4, 0], 140)
        self.assertEqual(fields["Withdrawal"][4, 0], 0)
        self.assertEqual(fields["Active"][4, 0], 1)
        self.assertEqual(fields["Withdrawal"][5, 0], 1540)
        self.assertEqual(fields["Deposit"][6, 0], 1000)

    def test_partial_sell_is_one_round_trip(self):
        """分批成交不会把一次完整择时交易拆成多笔胜负样本。"""
        limits = np.full((8, 2), np.inf)
        limits[3, 0] = 700
        output = self._run(ts.TimingAccountStats(self._account(sell_amt_limit=self._factor(limits))), statistics=True)
        self.assertEqual(output["择时统计"].loc["交易次数", "A"], 1)
        self.assertEqual(len(output["择时交易"].query("ID == 'A'")), 1)

    def test_fees_are_paid_by_each_subaccount(self):
        """买入资金包含费用，往返收益扣除买卖费用。"""
        output = self._run(ts.TimingAccountStats(self._account(buy_fee=.01, sell_fee=.02)), statistics=True)
        trade = output["择时交易"].query("ID == 'A'").iloc[0]
        expected = (1000 / 1.01) * 1.4 * .98 - 1000
        self.assertAlmostEqual(trade["盈亏金额"], expected)
        self.assertAlmostEqual(trade["收益率"], expected / 1000)
        self.assertGreater(trade["交易费"], 0)

    def test_missing_price_preserves_valuation(self):
        """缺失报价不成交且沿用有效估值，恢复报价后正常交易。"""
        prices = np.array([[10, 20], [10, 20], [np.nan, 20], [14, 18],
                           [10, 16], [11, 20], [11, 22], [20, 22]])
        account = ts.MakeTimingAccount(init_cash=1000, start_dt=self.dates[1])(
            last_price=self._factor(prices), signal=self.signal, mask=self.mask
        )
        fields = self._fields(account)
        self.assertEqual(fields["Amount"][1, 0], 1000)
        self.assertEqual(fields["Position"][1, 0], 100)
        self.assertEqual(fields["TradeNum"][1, 0], 0)

    def test_late_statistics_keep_original_open_cost(self):
        """统计区间开始前的开仓成本仍用于区间内平仓交易的盈亏。"""
        output = self._run(ts.TimingAccountStats(self._account()), statistics=True, start=3)
        self.assertEqual(output["择时统计"].loc["交易次数", "总计"], 4)
        self.assertAlmostEqual(output["择时统计"].loc["总盈亏", "总计"], 420)
        self.assertAlmostEqual(output["初始资金"], 2200)

    def test_binding_and_invalid_capital(self):
        """支持预绑定标的池，并拒绝无效本金。"""
        operator = ts.MakeTimingAccount(init_cash=1000, start_dt=self.dates[1])
        account = operator.bind(mask=self.mask)(last_price=self.price, signal=self.signal)
        np.testing.assert_allclose(self._fields(account)["Deposit"].sum(), 3000)
        for cash in (0, -1, np.nan, np.inf):
            with self.subTest(cash=cash), self.assertRaises(__QS_Error__):
                ts.MakeTimingAccount(init_cash=cash)

    def test_report_defaults_to_totals_and_keeps_raw_details(self):
        """默认报告仅展示总计与图像，原始交易和逐标的资金流水仍可读取。"""
        node = ts.TimingAccountReport(ts.TimingAccountStats(self._account()))
        self.assertFalse(node.Args.ShowDetails)
        output = self._run(node, statistics=True)
        html = output["Report"]
        self.assertEqual(html.count("<table"), 2)
        self.assertIn("<th>总计</th>", html)
        self.assertNotIn("<th>A</th>", html)
        self.assertNotIn("<th>B</th>", html)
        self.assertNotIn("<h3>分标的净值与交易标记</h3>", html)
        self.assertEqual(html.count("data:image/png;base64,"), 1)
        self.assertIn("账户净值与资金流水", html)
        for title in ("资金流水", "择时交易", "未平仓交易"):
            self.assertNotIn(f"<h3>{title}</h3>", html)
        image = html.split("data:image/png;base64,", 1)[1].split('"', 1)[0]
        self.assertTrue(base64.b64decode(image).startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(output["择时统计"].columns.tolist(), ["总计", "A", "B"])
        self.assertIn("ID", output["资金流水"].columns)
        self.assertEqual(len(output["择时交易"]), 4)
        self.assertEqual(len(output["未平仓交易"]), 1)

    def test_report_details_and_custom_report_key(self):
        """开启明细后展示各标的统计，仍不添加交易明细表并支持自定义报告键。"""
        node = ts.TimingAccountReport(ts.TimingAccountStats(self._account()),
                                      args={"ShowDetails": True, "ReportKey": "TimingHTML"})
        output = self._run(node, statistics=True)
        html = output["TimingHTML"]
        self.assertNotIn("Report", output)
        self.assertEqual(html.count("<table"), 2)
        self.assertIn("<th>A</th>", html)
        self.assertIn("<th>B</th>", html)
        self.assertIn("<h3>分标的净值与交易标记</h3>", html)
        self.assertEqual(html.count("data:image/png;base64,"), 3)
        self.assertIn('<img alt="A 独立净值与交易标记"', html)
        self.assertIn('<img alt="B 独立净值与交易标记"', html)
        self.assertNotIn("<h3>择时交易</h3>", html)
        self.assertNotIn("<h3>未平仓交易</h3>", html)

    def test_report_charts_use_account_nv_and_aggregated_cashflow(self):
        """图像使用资金流调整净值，资金柱只按时点汇总且可叠加基准。"""
        output = self._run(ts.TimingAccountStats(self._account(), bmk_nv=DataFactor(1)), statistics=True)
        figure = ts.TimingAccountReport.genMatplotlibFig(output)
        nv_axis, flow_axis = figure.axes
        self.assertEqual(len(nv_axis.lines), 2)
        np.testing.assert_allclose(nv_axis.lines[0].get_ydata(), output["时间序列"]["净值"])
        np.testing.assert_allclose(nv_axis.lines[1].get_ydata(), 1)
        np.testing.assert_allclose([bar.get_height() for bar in flow_axis.patches],
                                   [1000, 1000, 0, 0, -1540, -880, 1000])

    def test_cashflow_chart_nets_simultaneous_deposit_and_withdrawal(self):
        """同一时点等额出入金在资金图中相抵，资金流不影响账户净值。"""
        mask = self._factor([[0, 0], [1, 0], [0, 1], [0, 1],
                             [0, 1], [0, 1], [0, 1], [0, 1]])
        account = ts.MakeTimingAccount(init_cash=1000, start_dt=self.dates[1])(
            last_price=self.price, signal=DataFactor(0), mask=mask)
        output = self._run(ts.TimingAccountStats(account), statistics=True)
        figure = ts.TimingAccountReport.genMatplotlibFig(output)
        self.assertEqual(figure.axes[1].patches[1].get_height(), 0)
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), 1)

    def test_subaccount_nv_adjusts_flows_and_preserves_reentry_history(self):
        """各标的净值独立扣除出入金，出池后持平且再入池不重置。"""
        output = self._run(ts.TimingAccountStats(self._account()), statistics=True)
        expected = [[1, 1], [1.2, 1], [1.4, .9], [1.4, .8],
                    [1.54, .8], [1.54, .88], [1.54, .88]]
        np.testing.assert_allclose(output["子账户净值"], expected)
        late = self._run(ts.TimingAccountStats(self._account()), statistics=True, start=3)
        np.testing.assert_allclose(late["子账户净值"], np.array(expected)[2:] / [1.2, 1])

    def test_detail_chart_marks_actual_fill_dates_and_prices(self):
        """每个买卖标记使用实际成交时点和价格，并与对应标的净值匹配。"""
        output = self._run(ts.TimingAccountStats(self._account()), statistics=True)
        figure = ts.TimingAccountReport.genDetailMatplotlibFig(output, "A")
        nv_axis, price_axis = figure.axes
        np.testing.assert_allclose(nv_axis.lines[0].get_ydata(), output["子账户净值"]["A"])
        np.testing.assert_allclose(price_axis.lines[0].get_ydata(), [10, 12, 14, 10, 11, 11, 20])
        np.testing.assert_allclose(price_axis.collections[0].get_offsets(), [[0, 10], [3, 10], [6, 20]])
        np.testing.assert_allclose(price_axis.collections[1].get_offsets(), [[2, 14], [4, 11]])

    def test_detail_chart_has_no_markers_for_blocked_orders(self):
        """有看多信号但禁止买入时净值持平，图像不标注未成交信号。"""
        output = self._run(ts.TimingAccountStats(self._account(buy_limit=DataFactor(1))), statistics=True)
        figure = ts.TimingAccountReport.genDetailMatplotlibFig(output, "A")
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), 1)
        self.assertEqual(len(figure.axes[1].collections), 0)


if __name__ == "__main__":
    unittest.main()
