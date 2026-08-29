# -*- coding: utf-8 -*-
"""测试 BackTest.Strategy 模块中的算子和回测流程"""
import time
import unittest
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Core.CalcEngine import Engine
import QuantStudio.Factor.FactorOperator as fo
from QuantStudio.BackTest.Strategy.Strategy import MakeAccount
from QuantStudio.BackTest.Strategy.AllocationStrategy import CalcPortfolioNV, CalcMaskPortfolio, MergeTopDownSignal
from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime, getNaturalDay


class TestMergeTopDownSignal(unittest.TestCase):
    """测试 MergeTopDownSignal 算子"""

    @classmethod
    def setUpClass(cls):
        """设置测试数据"""
        # 时间序列
        cls.DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(days=i) for i in range(5)]
        # 顶层信号截面
        cls.TopIDs = ["TopA", "TopB"]
        # 底层信号截面
        cls.DownIDs_A = ["A1", "A2", "A3"]
        cls.DownIDs_B = ["B1", "B2"]
        cls.AllDownIDs = cls.DownIDs_A + cls.DownIDs_B
        # 设置随机种子
        np.random.seed(42)

    def _create_data_factors(self, top_data, down_data_a, down_data_b):
        """创建测试用的 DataFactor

        Args:
            top_data: 顶层信号数据 (n_dt, n_top_ids)
            down_data_a: TopA 对应的底层信号数据 (n_dt, n_down_ids_a)
            down_data_b: TopB 对应的底层信号数据 (n_dt, n_down_ids_b)

        Returns:
            tuple: (top_signal, down_signal_a, down_signal_b)
        """
        top_signal = DataFactor(
            data=pd.DataFrame(top_data, index=self.DTs, columns=self.TopIDs),
            args={"Name": "TopSignal"}
        )
        down_signal_a = DataFactor(
            data=pd.DataFrame(down_data_a, index=self.DTs, columns=self.DownIDs_A),
            args={"Name": "DownSignalA"}
        )
        down_signal_b = DataFactor(
            data=pd.DataFrame(down_data_b, index=self.DTs, columns=self.DownIDs_B),
            args={"Name": "DownSignalB"}
        )
        return top_signal, down_signal_a, down_signal_b

    def test_basic_merge(self):
        """测试基本的信号合并功能"""
        # 顶层信号
        top_data = np.array([
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.6, 0.4],
        ])
        # 底层信号 - 每个顶层截面对应的证券权重
        down_data_a = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.6, 0.2, 0.2],
            [0.3, 0.5, 0.2],
            [0.5, 0.3, 0.2],
        ])
        down_data_b = np.array([
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.6, 0.4],
        ])

        top_signal, down_signal_a, down_signal_b = self._create_data_factors(
            top_data, down_data_a, down_data_b
        )

        # 定义映射关系
        top_id_to_down_section = {
            "TopA": self.DownIDs_A,
            "TopB": self.DownIDs_B,
        }

        # 创建算子
        operator = MergeTopDownSignal(
            top_id_to_down_section=top_id_to_down_section,
            top_ids=self.TopIDs,
        )

        # 调用算子生成因子
        result_factor = operator(
            top_signal=top_signal,
            top_id_to_down_signal={
                "TopA": down_signal_a,
                "TopB": down_signal_b,
            },
            factor_args={"Name": "MergedSignal"},
        )

        # 运行计算引擎
        with FeatherFactorCache(args={"DTRuler": self.DTs, "StartMode": "new", "CacheDir": None}) as Cache:
            with FactorContext(
                DTRuler=self.DTs,
                SectionIDs=self.AllDownIDs,
                DataCache=Cache,
            ) as Context:
                with Engine() as ExecEngine:
                    fwd_data = FactorLocalContext(DTs=self.DTs, IDs=self.AllDownIDs)
                    Rslt = ExecEngine.run(
                        [result_factor], Context,
                        fwd_data_list=[fwd_data]
                    )

        result = Rslt[0].values

        # 验证结果形状
        self.assertEqual(result.shape, (len(self.DTs), len(self.AllDownIDs)))

        # 手动计算期望结果
        # 对于每个时点 t 和证券 s:
        # result[t, s] = top_data[t, "TopA"] * down_data_a[t, s_idx] (如果 s 属于 A)
        #               + top_data[t, "TopB"] * down_data_b[t, s_idx] (如果 s 属于 B)
        expected = np.zeros((len(self.DTs), len(self.AllDownIDs)))
        for t in range(len(self.DTs)):
            for i, s in enumerate(self.DownIDs_A):
                idx = self.AllDownIDs.index(s)
                expected[t, idx] = top_data[t, 0] * down_data_a[t, i]
            for i, s in enumerate(self.DownIDs_B):
                idx = self.AllDownIDs.index(s)
                expected[t, idx] = top_data[t, 1] * down_data_b[t, i]

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_partial_top_ids(self):
        """测试只使用部分顶层 ID 的情况

        当只使用 TopA 时, B1/B2 列未被任何底层信号覆盖, 保持 NaN
        """
        top_data = np.array([
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.6, 0.4],
        ])
        down_data_a = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.6, 0.2, 0.2],
            [0.3, 0.5, 0.2],
            [0.5, 0.3, 0.2],
        ])
        down_data_b = np.array([
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.6, 0.4],
        ])

        top_signal, down_signal_a, down_signal_b = self._create_data_factors(
            top_data, down_data_a, down_data_b
        )

        # 定义映射，但只使用 TopA
        top_id_to_down_section = {
            "TopA": self.DownIDs_A,
            "TopB": self.DownIDs_B,
        }

        operator = MergeTopDownSignal(
            top_id_to_down_section=top_id_to_down_section,
            top_ids=["TopA"],  # 只使用 TopA
        )

        result_factor = operator(
            top_signal=top_signal,
            top_id_to_down_signal={
                "TopA": down_signal_a,
            },
            factor_args={"Name": "MergedSignalPartial"},
        )

        with FeatherFactorCache(args={"DTRuler": self.DTs, "StartMode": "new", "CacheDir": None}) as Cache:
            with FactorContext(
                DTRuler=self.DTs,
                SectionIDs=self.AllDownIDs,
                DataCache=Cache,
            ) as Context:
                with Engine() as ExecEngine:
                    fwd_data = FactorLocalContext(DTs=self.DTs, IDs=self.AllDownIDs)
                    Rslt = ExecEngine.run(
                        [result_factor], Context,
                        fwd_data_list=[fwd_data]
                    )

        result = Rslt[0].values

        # A 列: 只有 TopA 的贡献
        # B 列: 未被覆盖, 保持 NaN
        expected = np.full((len(self.DTs), len(self.AllDownIDs)), np.nan)
        for t in range(len(self.DTs)):
            for i, s in enumerate(self.DownIDs_A):
                idx = self.AllDownIDs.index(s)
                expected[t, idx] = top_data[t, 0] * down_data_a[t, i]

        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_overlapping_down_sections(self):
        """测试底层截面有重叠的情况"""
        # TopA 和 TopB 共享部分底层证券
        shared_ids = ["S1", "S2"]
        top_data = np.array([
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.6, 0.4],
        ])
        # 底层信号：A 包含 S1, S2, A3；B 包含 S1, S2, B3
        down_data_a = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.6, 0.2, 0.2],
            [0.3, 0.5, 0.2],
            [0.5, 0.3, 0.2],
        ])
        down_data_b = np.array([
            [0.6, 0.4, 0.0],
            [0.7, 0.3, 0.0],
            [0.5, 0.5, 0.0],
            [0.8, 0.2, 0.0],
            [0.6, 0.4, 0.0],
        ])

        down_ids_a = ["S1", "S2", "A3"]
        down_ids_b = ["S1", "S2", "B3"]
        all_ids = ["S1", "S2", "A3", "B3"]

        top_signal = DataFactor(
            data=pd.DataFrame(top_data, index=self.DTs, columns=self.TopIDs),
            args={"Name": "TopSignal"}
        )
        down_signal_a = DataFactor(
            data=pd.DataFrame(down_data_a, index=self.DTs, columns=down_ids_a),
            args={"Name": "DownSignalA"}
        )
        down_signal_b = DataFactor(
            data=pd.DataFrame(down_data_b, index=self.DTs, columns=down_ids_b),
            args={"Name": "DownSignalB"}
        )

        top_id_to_down_section = {
            "TopA": down_ids_a,
            "TopB": down_ids_b,
        }

        operator = MergeTopDownSignal(
            top_id_to_down_section=top_id_to_down_section,
            top_ids=self.TopIDs,
        )

        result_factor = operator(
            top_signal=top_signal,
            top_id_to_down_signal={
                "TopA": down_signal_a,
                "TopB": down_signal_b,
            },
            factor_args={"Name": "MergedSignalOverlap"},
        )

        with FeatherFactorCache(args={"DTRuler": self.DTs, "StartMode": "new", "CacheDir": None}) as Cache:
            with FactorContext(
                DTRuler=self.DTs,
                SectionIDs=all_ids,
                DataCache=Cache,
            ) as Context:
                with Engine() as ExecEngine:
                    fwd_data = FactorLocalContext(DTs=self.DTs, IDs=all_ids)
                    Rslt = ExecEngine.run(
                        [result_factor], Context,
                        fwd_data_list=[fwd_data]
                    )

        result = Rslt[0].values

        # 手动计算：S1 和 S2 是重叠的，应该累加
        expected = np.zeros((len(self.DTs), len(all_ids)))
        for t in range(len(self.DTs)):
            # TopA 的贡献
            for i, s in enumerate(down_ids_a):
                idx = all_ids.index(s)
                expected[t, idx] += top_data[t, 0] * down_data_a[t, i]
            # TopB 的贡献
            for i, s in enumerate(down_ids_b):
                idx = all_ids.index(s)
                expected[t, idx] += top_data[t, 1] * down_data_b[t, i]

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestBacktestConsistency(unittest.TestCase):
    """测试不同回测方法（迭代式 vs 向量化）的结果一致性"""

    @classmethod
    def setUpClass(cls):
        """设置测试数据"""
        cls.StartDT = dt.datetime(2024, 1, 1)
        cls.EndDT = dt.datetime(2025, 12, 31)
        cls.TestStartDT = dt.datetime(2025, 1, 31)
        cls.TestEndDT = cls.EndDT
        cls.DTRuler = getNaturalDay(start_date=cls.StartDT, end_date=cls.EndDT)
        cls.TestDTs = getNaturalDay(start_date=cls.TestStartDT, end_date=cls.TestEndDT)
        cls.SectionIDs = [f"{str(i).zfill(6)}.SZ" for i in range(1, 101)]
        # 再平衡时点序列（月末）
        cls.BalanceDTs = getMonthLastDateTime(cls.DTRuler)

    def _create_mock_factors(self):
        """创建 MOCK 数据因子

        Returns:
            tuple: (Mask, Industry, Price, ExpectedReturn, Factor2, Weight)
        """
        np.random.seed(0)
        Mask = DataFactor(
            data=pd.DataFrame(
                np.random.choice([0, 1], size=(len(self.DTRuler), len(self.SectionIDs)), p=[0.1, 0.9]).astype(bool),
                index=self.DTRuler, columns=self.SectionIDs
            ),
            args={"Name": "Mask"}
        )
        Industry = DataFactor(
            data=pd.Series(
                np.random.choice(["Fin", "TMT", "Ind"], size=(len(self.SectionIDs),)),
                index=self.SectionIDs, dtype=pd.StringDtype(storage="python")
            ),
            args={"Name": "Industry", "DataType": "string"}
        )
        Price = DataFactor(
            data=pd.DataFrame(
                np.random.rand(len(self.DTRuler), len(self.SectionIDs)) * 10,
                index=self.DTRuler, columns=self.SectionIDs
            ),
            args={"Name": "Price"}
        )
        ExpectedReturn = DataFactor(
            data=pd.DataFrame(
                np.random.randn(len(self.DTRuler), len(self.SectionIDs)),
                index=self.DTRuler, columns=self.SectionIDs
            ),
            args={"Name": "ExpectedReturn"}
        )
        Factor2 = DataFactor(
            data=pd.DataFrame(
                np.random.randn(len(self.DTRuler), len(self.SectionIDs)),
                index=self.DTRuler, columns=self.SectionIDs
            ),
            args={"Name": "Factor2"}
        )
        Weight = DataFactor(
            data=pd.DataFrame(
                np.random.rand(len(self.DTRuler), len(self.SectionIDs)),
                index=self.DTRuler, columns=self.SectionIDs
            ),
            args={"Name": "Weight"}
        )
        return Mask, Industry, Price, ExpectedReturn, Factor2, Weight

    def _build_portfolio(self, ExpectedReturn, Mask, Industry, Weight):
        """构造投资组合信号

        Args:
            ExpectedReturn: 预期收益因子
            Mask: 掩码因子
            Industry: 行业因子
            Weight: 权重因子

        Returns:
            Portfolio: 投资组合因子
        """
        ExpectedRank = fo.SectionRank(ascending=True, uniformization=True)(
            ExpectedReturn, mask=Mask, cat_data=Industry
        )
        Portfolio = CalcMaskPortfolio(descriptor_ids=self.SectionIDs)(
            mask=(ExpectedRank >= 0.7),
            weight=None,
            cat_data=Industry,
            cat_weight=Weight,
            factor_args={"CalcDTRuler": self.BalanceDTs, "Name": "Portfolio"}
        )
        return Portfolio

    def _run_backtest(self, node_list, fwd_data_list):
        """运行单次回测并返回结果和耗时

        Args:
            node_list: 节点列表
            fwd_data_list: 前向数据列表

        Returns:
            tuple: (results, elapsed_seconds)
        """
        with FeatherFactorCache(args={"DTRuler": self.DTRuler, "StartMode": "new", "CacheDir": None}) as Cache:
            with FactorContext(DTRuler=self.DTRuler, SectionIDs=self.SectionIDs, DataCache=Cache) as Context:
                with Engine() as ExecEngine:
                    start = time.perf_counter()
                    Rslt = ExecEngine.run(
                        node_list, Context,
                        fwd_data_list=fwd_data_list
                    )
                    elapsed = time.perf_counter() - start
        return Rslt, elapsed

    def test_backtest_comparison(self):
        """测试迭代式回测与向量化回测的结果一致性"""
        Mask, Industry, Price, ExpectedReturn, Factor2, Weight = self._create_mock_factors()
        Portfolio = self._build_portfolio(ExpectedReturn, Mask, Industry, Weight)

        # 迭代式回测
        Account = MakeAccount(signal_type="目标权重", init_cash=1e6, short_allowed=False, start_dt=self.TestDTs[0])(
            last_price=Price, signal=Portfolio,
            factor_args={"Name": "迭代式回测"}
        )
        # 向量化回测
        PortfolioNV1 = CalcPortfolioNV(
            descriptor_ids=self.SectionIDs, start_dt=self.TestDTs[0], calc_type="numpy"
        )(Portfolio, price=Price, init_nv=1e6, portfolio_name_list=["Portfolio"],
          factor_args={"Name": "向量化回测-numpy"})
        PortfolioNV2 = CalcPortfolioNV(
            descriptor_ids=self.SectionIDs, start_dt=self.TestDTs[0], calc_type="pandas"
        )(Portfolio, price=Price, init_nv=1e6, portfolio_name_list=["Portfolio"],
          factor_args={"Name": "向量化回测-pandas"})

        NodeList = [Account, PortfolioNV1, PortfolioNV2]
        FwdDataList = [
            FactorLocalContext(DTs=self.TestDTs, IDs=self.SectionIDs, SectionIDs=self.SectionIDs),
            FactorLocalContext(DTs=self.TestDTs, IDs=["Portfolio"], SectionIDs=["Portfolio"]),
            FactorLocalContext(DTs=self.TestDTs, IDs=["Portfolio"], SectionIDs=["Portfolio"]),
        ]
        
        Rslt, _ = self._run_backtest(NodeList, FwdDataList)

        # 计算归一化净值
        NV = {}
        Cash = Rslt[0].iloc[:, 0].apply(lambda x: x[0])
        iNV = Cash + Rslt[0].map(lambda x: x[2]).sum(axis=1)
        iNV = iNV / iNV.iloc[0]
        NV["迭代式回测"] = iNV
        NV["向量化回测-numpy"] = Rslt[1].iloc[:, 0] / Rslt[1].iloc[0, 0]
        NV["向量化回测-pandas"] = Rslt[2].iloc[:, 0] / Rslt[2].iloc[0, 0]
        NV = pd.DataFrame(NV)

        # 验证：迭代式回测与向量化回测（numpy）误差应很小
        max_err_iter = (NV["迭代式回测"] - NV["向量化回测-numpy"]).abs().max()
        self.assertLess(max_err_iter, 1e-10, f"迭代式回测与向量化回测-numpy 最大误差: {max_err_iter}")
        # 验证：向量化回测 pandas 与 numpy 误差应很小
        max_err_vec = (NV["向量化回测-pandas"] - NV["向量化回测-numpy"]).abs().max()
        self.assertLess(max_err_vec, 1e-10, f"向量化回测-pandas 与向量化回测-numpy 最大误差: {max_err_vec}")

    def test_backtest_speed(self):
        """对比不同回测方法的计算速度"""
        Mask, Industry, Price, ExpectedReturn, Factor2, Weight = self._create_mock_factors()
        Portfolio = self._build_portfolio(ExpectedReturn, Mask, Industry, Weight)
        n_runs = 3  # 重复次数，取平均值

        # 迭代式回测
        Account = MakeAccount(signal_type="目标权重", init_cash=1e6, short_allowed=False, start_dt=self.TestDTs[0])(
            last_price=Price, signal=Portfolio,
            factor_args={"Name": "迭代式回测"}
        )
        times_iter = []
        for _ in range(n_runs):
            _, elapsed = self._run_backtest(
                [Account],
                [FactorLocalContext(DTs=self.TestDTs, IDs=self.SectionIDs, SectionIDs=self.SectionIDs)]
            )
            times_iter.append(elapsed)

        # 向量化回测 - numpy
        PortfolioNV_numpy = CalcPortfolioNV(
            descriptor_ids=self.SectionIDs, start_dt=self.TestDTs[0], calc_type="numpy"
        )(Portfolio, price=Price, init_nv=1e6, portfolio_name_list=["Portfolio"],
          factor_args={"Name": "向量化回测-numpy"})
        times_numpy = []
        for _ in range(n_runs):
            _, elapsed = self._run_backtest(
                [PortfolioNV_numpy],
                [FactorLocalContext(DTs=self.TestDTs, IDs=["Portfolio"], SectionIDs=["Portfolio"])]
            )
            times_numpy.append(elapsed)

        # 向量化回测 - pandas
        PortfolioNV_pandas = CalcPortfolioNV(
            descriptor_ids=self.SectionIDs, start_dt=self.TestDTs[0], calc_type="pandas"
        )(Portfolio, price=Price, init_nv=1e6, portfolio_name_list=["Portfolio"],
          factor_args={"Name": "向量化回测-pandas"})
        times_pandas = []
        for _ in range(n_runs):
            _, elapsed = self._run_backtest(
                [PortfolioNV_pandas],
                [FactorLocalContext(DTs=self.TestDTs, IDs=["Portfolio"], SectionIDs=["Portfolio"])]
            )
            times_pandas.append(elapsed)

        avg_iter = sum(times_iter) / n_runs
        avg_numpy = sum(times_numpy) / n_runs
        avg_pandas = sum(times_pandas) / n_runs

        # 打印速度对比结果
        print(f"\n{'='*60}")
        print(f"回测速度对比（{n_runs} 次平均，{len(self.TestDTs)} 个交易日，{len(self.SectionIDs)} 只证券）")
        print(f"{'='*60}")
        print(f"  迭代式回测:       {avg_iter:.4f} 秒")
        print(f"  向量化回测-numpy: {avg_numpy:.4f} 秒")
        print(f"  向量化回测-pandas:{avg_pandas:.4f} 秒")
        print(f"{'='*60}")
        if avg_numpy > 0:
            print(f"  迭代式 / numpy 加速比:  {avg_iter / avg_numpy:.2f}x")
        if avg_pandas > 0:
            print(f"  迭代式 / pandas 加速比: {avg_iter / avg_pandas:.2f}x")
        if avg_numpy > 0:
            print(f"  pandas / numpy 比值:    {avg_pandas / avg_numpy:.2f}x")
        print(f"{'='*60}")

        # 基本断言：所有方法都应该能在合理时间内完成
        self.assertLess(avg_iter, 60, "迭代式回测耗时超过 60 秒")
        self.assertLess(avg_numpy, 60, "向量化回测-numpy 耗时超过 60 秒")
        self.assertLess(avg_pandas, 60, "向量化回测-pandas 耗时超过 60 秒")


if __name__ == "__main__":
    unittest.main()
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestMergeTopDownSignal("test_partial_top_ids"))
    # Suite.addTest(TestBacktestConsistency("test_backtest_comparison"))
    # Suite.addTest(TestBacktestConsistency("test_backtest_speed"))
    # Runner = unittest.TextTestRunner()
    # Runner.run(Suite)