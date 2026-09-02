# -*- coding: utf-8 -*-
"""BTStorer 回测结果持久化测试.

测试覆盖:
    - HDF5BTResultDB: 写入、读取、路径层级、metadata 查询
    - BTStorer: 单节点存储、split 模式
"""

import os
import shutil
import unittest
import tempfile

import numpy as np
import pandas as pd

from QuantStudio.BackTest.BTResultDB import BTResultDB, _HDF5BTResultDB, HDF5BTResultDB
from QuantStudio.BackTest.BTStorer import BTStorer


class Test_HDF5BTResultDB(unittest.TestCase):
    """_HDF5BTResultDB 基本功能测试."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._file_path = os.path.join(self._tmp_dir, "test_btresult.h5")
        self.db = _HDF5BTResultDB(args={"FilePath": self._file_path})

    def tearDown(self):
        if os.path.isfile(self._file_path):
            os.remove(self._file_path)
        os.rmdir(self._tmp_dir)

    def _make_sample_result(self):
        """构造一个典型的回测结果 dict."""
        return {
            "IC": pd.DataFrame(
                {"factor1": [0.05, 0.03, -0.02], "factor2": [0.01, -0.01, 0.04]},
                index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            ),
            "统计数据": pd.DataFrame(
                {"平均值": [0.02, 0.013], "标准差": [0.035, 0.025]},
                index=["factor1", "factor2"],
            ),
            "初始资金": 1000000.0,
            "Report": "<html>test report</html>",
        }

    def test_write_and_read(self):
        """写入后应能完整读回."""
        result = self._make_sample_result()
        self.db.writeResult(result, "test_group")

        loaded = self.db.readResult("test_group")
        self.assertIsNotNone(loaded)
        self.assertIn("IC", loaded)
        self.assertIn("统计数据", loaded)
        self.assertIn("初始资金", loaded)
        self.assertIn("Report", loaded)
        pd.testing.assert_frame_equal(loaded["IC"], result["IC"])
        pd.testing.assert_frame_equal(loaded["统计数据"], result["统计数据"])
        self.assertEqual(loaded["初始资金"], 1000000.0)
        self.assertEqual(loaded["Report"], "<html>test report</html>")

    def test_read_nonexistent(self):
        """读取不存在的 group 应返回 None."""
        self.assertIsNone(self.db.readResult("nonexistent"))

    def test_write_overwrites_group(self):
        """重复写入同一 group 应覆盖."""
        result1 = {"初始资金": 100.0}
        result2 = {"初始资金": 200.0}
        self.db.writeResult(result1, "grp")
        self.db.writeResult(result2, "grp")
        loaded = self.db.readResult("grp")
        self.assertEqual(loaded["初始资金"], 200.0)

    def test_path_hierarchy(self):
        """group_name 支持路径层级."""
        result = self._make_sample_result()
        self.db.writeResult(result, "A股/IC/沪深300")
        self.db.writeResult(result, "ETF/IC")

        loaded1 = self.db.readResult("A股/IC/沪深300")
        loaded2 = self.db.readResult("ETF/IC")
        self.assertIsNotNone(loaded1)
        self.assertIsNotNone(loaded2)
        self.assertIn("IC", loaded1)
        self.assertIn("IC", loaded2)

    def test_multiple_groups_independent(self):
        """不同 group 互不影响."""
        self.db.writeResult({"初始资金": 100.0}, "grp1")
        self.db.writeResult({"初始资金": 200.0}, "grp2")

        self.assertEqual(self.db.readResult("grp1")["初始资金"], 100.0)
        self.assertEqual(self.db.readResult("grp2")["初始资金"], 200.0)

    def test_metadata_write_and_list(self):
        """写入 metadata 后应能通过 listResults 查询."""
        result = self._make_sample_result()
        self.db.writeResult(result, "A股_IC", metadata={"资产": "A股", "策略": "IC"})
        self.db.writeResult(result, "ETF_IC", metadata={"资产": "ETF", "策略": "IC"})
        self.db.writeResult(result, "A股_账户", metadata={"资产": "A股", "策略": "账户"})

        # 查所有
        all_results = self.db.listResults()
        self.assertEqual(len(all_results), 3)

        # 按资产筛选
        a_stock = self.db.listResults(metadata={"资产": "A股"})
        self.assertEqual(sorted(a_stock), ["A股_IC", "A股_账户"])

        # 按策略筛选
        ic_results = self.db.listResults(metadata={"策略": "IC"})
        self.assertEqual(sorted(ic_results), ["A股_IC", "ETF_IC"])

        # 组合筛选
        combined = self.db.listResults(metadata={"资产": "A股", "策略": "IC"})
        self.assertEqual(combined, ["A股_IC"])

    def test_metadata_with_path_hierarchy(self):
        """路径层级 + metadata 组合使用."""
        result = self._make_sample_result()
        self.db.writeResult(result, "A股/IC/沪深300", metadata={"资产": "A股", "池子": "沪深300"})
        self.db.writeResult(result, "A股/IC/中证500", metadata={"资产": "A股", "池子": "中证500"})
        self.db.writeResult(result, "ETF/IC", metadata={"资产": "ETF"})

        hs300 = self.db.listResults(metadata={"池子": "沪深300"})
        self.assertEqual(hs300, ["A股/IC/沪深300"])

        a_stock = self.db.listResults(metadata={"资产": "A股"})
        self.assertEqual(sorted(a_stock), ["A股/IC/中证500", "A股/IC/沪深300"])

    def test_metadata_overwrite(self):
        """重复写入同一 group 应更新 metadata."""
        self.db.writeResult({"初始资金": 100.0}, "grp", metadata={"版本": "v1"})
        self.db.writeResult({"初始资金": 200.0}, "grp", metadata={"版本": "v2"})

        v1 = self.db.listResults(metadata={"版本": "v1"})
        v2 = self.db.listResults(metadata={"版本": "v2"})
        self.assertEqual(v1, [])
        self.assertEqual(v2, ["grp"])

    def test_group_names_property(self):
        """ResultNames 应返回所有结果组."""
        self.db.writeResult({"初始资金": 100.0}, "grp1", metadata={"a": 1})
        self.db.writeResult({"初始资金": 200.0}, "grp2", metadata={"b": 2})
        self.assertEqual(sorted(self.db.ResultNames), ["grp1", "grp2"])

    def test_read_metadata_all(self):
        """readMetaData(key=None) 应返回所有元信息."""
        self.db.writeResult({"初始资金": 100.0}, "grp", metadata={"资产": "A股", "策略": "IC"})
        meta = self.db.readMetaData("grp")
        self.assertEqual(meta["资产"], "A股")
        self.assertEqual(meta["策略"], "IC")

    def test_read_metadata_key(self):
        """readMetaData(key=...) 应返回指定键的值."""
        self.db.writeResult({"初始资金": 100.0}, "grp", metadata={"资产": "A股", "策略": "IC"})
        self.assertEqual(self.db.readMetaData("grp", key="资产"), "A股")
        self.assertEqual(self.db.readMetaData("grp", key="策略"), "IC")
        self.assertIsNone(self.db.readMetaData("grp", key="不存在"))

    def test_read_metadata_nonexistent_group(self):
        """读取不存在 group 的 metadata 应返回 None."""
        self.assertIsNone(self.db.readMetaData("nonexistent"))
        self.assertIsNone(self.db.readMetaData("nonexistent", key="k"))

    def test_set_metadata_key_value(self):
        """setMetaData 用 key/value 设置单个标签."""
        self.db.writeResult({"初始资金": 100.0}, "grp", metadata={"资产": "A股"})
        self.db.setMetaData("grp", key="策略", value="IC")

        meta = self.db.readMetaData("grp")
        self.assertEqual(meta["资产"], "A股")
        self.assertEqual(meta["策略"], "IC")

    def test_set_metadata_dict(self):
        """setMetaData 用 metadata dict 批量设置标签."""
        self.db.writeResult({"初始资金": 100.0}, "grp")
        self.db.setMetaData("grp", metadata={"资产": "A股", "策略": "IC", "池子": "沪深300"})

        meta = self.db.readMetaData("grp")
        self.assertEqual(meta["资产"], "A股")
        self.assertEqual(meta["策略"], "IC")
        self.assertEqual(meta["池子"], "沪深300")

    def test_set_metadata_overwrite(self):
        """setMetaData 应覆盖已有同名键."""
        self.db.writeResult({"初始资金": 100.0}, "grp", metadata={"版本": "v1"})
        self.db.setMetaData("grp", key="版本", value="v2")

        self.assertEqual(self.db.readMetaData("grp", key="版本"), "v2")
        # listResults 也应反映更新后的 metadata
        self.assertEqual(self.db.listResults(metadata={"版本": "v1"}), [])
        self.assertEqual(self.db.listResults(metadata={"版本": "v2"}), ["grp"])

    def test_set_metadata_nonexistent_group(self):
        """对不存在的 group 设置 metadata 应静默忽略."""
        self.db.setMetaData("nonexistent", key="k", value="v")  # 不应报错


class TestBTStorer(unittest.TestCase):
    """BTStorer 节点测试."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._results_dir = os.path.join(self._tmp_dir, "results")
        os.makedirs(self._results_dir, exist_ok=True)
        self.db = HDF5BTResultDB(args={"MainDir": self._results_dir})

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _make_bt_node(self, name, result):
        """创建一个返回固定结果的 BTNode 用于测试."""
        from pydantic import Field
        from QuantStudio.BackTest.BackTestModel import BTNode

        class _StubBTNode(BTNode):
            class __QS_ArgClass__(BTNode.__QS_ArgClass__):
                Name: str = Field(default="StubBTNode", frozen=True)

            def __init__(self, name, result, **kwargs):
                super().__init__(args={"Name": name}, **kwargs)
                self._result = result

            def backward_compute(self, path, bwd_data_list, context, local_context=None):
                return self._result

        return _StubBTNode(name, result)

    def test_single_dep_no_split(self):
        """单个依赖: 直接写入, 不拆分."""
        result = {"IC": pd.DataFrame({"a": [1, 2]}), "初始资金": 100.0}
        node = self._make_bt_node("IC测试", result)
        storer = BTStorer(deps=[node], args={"TargetDB": self.db})

        # 模拟 backward_compute
        storer.backward_compute(
            path=[storer.QSID],
            bwd_data_list=[result],
            context=type("Ctx", (), {"PID": "0"})(),
        )

        loaded = self.db.readResult("IC测试")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["初始资金"], 100.0)
        pd.testing.assert_frame_equal(loaded["IC"], result["IC"])

    def test_custom_group_name(self):
        """指定 GroupName 时应使用自定义名称."""
        result = {"初始资金": 50.0}
        node = self._make_bt_node("IC测试", result)
        storer = BTStorer(deps=[node], args={"TargetDB": self.db, "GroupName": "自定义组名"})

        storer.backward_compute(
            path=[storer.QSID],
            bwd_data_list=[result],
            context=type("Ctx", (), {"PID": "0"})(),
        )

        loaded = self.db.readResult("自定义组名")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["初始资金"], 50.0)

    def test_split_mode_multiple_deps(self):
        """多个依赖: 自动拆分为子 Storer, 各自写入."""
        result1 = {"IC": pd.DataFrame({"a": [1]})}
        result2 = {"账户": pd.DataFrame({"b": [2]})}
        node1 = self._make_bt_node("IC节点", result1)
        node2 = self._make_bt_node("账户节点", result2)

        storer = BTStorer(deps=[node1, node2], args={"TargetDB": self.db})
        self.assertTrue(storer._Splited)
        self.assertEqual(len(storer.Deps), 2)

        # 子 Storer 各自执行 backward_compute
        ctx = type("Ctx", (), {"PID": "0"})()
        storer.Deps[0].backward_compute(
            path=[storer.Deps[0].QSID],
            bwd_data_list=[result1],
            context=ctx,
        )
        storer.Deps[1].backward_compute(
            path=[storer.Deps[1].QSID],
            bwd_data_list=[result2],
            context=ctx,
        )

        loaded1 = self.db.readResult("IC节点")
        loaded2 = self.db.readResult("账户节点")
        self.assertIsNotNone(loaded1)
        self.assertIsNotNone(loaded2)
        pd.testing.assert_frame_equal(loaded1["IC"], result1["IC"])
        pd.testing.assert_frame_equal(loaded2["账户"], result2["账户"])

    def test_storer_with_metadata(self):
        """BTStorer 应透传 metadata 给 ResultDB."""
        result = {"初始资金": 100.0}
        node = self._make_bt_node("IC测试", result)
        storer = BTStorer(
            deps=[node],
            args={"TargetDB": self.db, "Metadata": {"资产": "A股", "策略": "IC"}},
        )

        storer.backward_compute(
            path=[storer.QSID],
            bwd_data_list=[result],
            context=type("Ctx", (), {"PID": "0"})(),
        )

        matches = self.db.listResults(metadata={"资产": "A股"})
        self.assertEqual(matches, ["IC测试"])


class TestHDF5BTResultDB(unittest.TestCase):
    """HDF5BTResultDB 目录模式测试."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._results_dir = os.path.join(self._tmp_dir, "results")
        os.makedirs(self._results_dir, exist_ok=True)
        self.db = HDF5BTResultDB(args={"MainDir": self._results_dir})

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _make_sample_result(self):
        return {
            "IC": pd.DataFrame(
                {"factor1": [0.05, 0.03], "factor2": [0.01, -0.01]},
                index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
            ),
            "统计数据": pd.DataFrame({"平均值": [0.02, 0.01]}, index=["factor1", "factor2"]),
            "初始资金": 1000000.0,
            "Report": "<html>test</html>",
        }

    def test_write_and_read(self):
        """写入后应能完整读回."""
        result = self._make_sample_result()
        self.db.writeResult(result, "test_group")
        loaded = self.db.readResult("test_group")
        self.assertIsNotNone(loaded)
        self.assertIn("IC", loaded)
        self.assertEqual(loaded["初始资金"], 1000000.0)
        pd.testing.assert_frame_equal(loaded["IC"], result["IC"])

    def test_path_hierarchy_creates_directories(self):
        """路径层级 group_name 应自动创建目录结构."""
        result = {"初始资金": 100.0}
        self.db.writeResult(result, "A股/IC/沪深300")
        self.db.writeResult(result, "A股/IC/中证500")
        self.db.writeResult(result, "ETF/IC")

        # 验证文件结构
        self.assertTrue(os.path.isfile(os.path.join(self._results_dir, "A股", "IC", "沪深300.h5")))
        self.assertTrue(os.path.isfile(os.path.join(self._results_dir, "A股", "IC", "中证500.h5")))
        self.assertTrue(os.path.isfile(os.path.join(self._results_dir, "ETF", "IC.h5")))

    def test_read_path_hierarchy(self):
        """按路径层级读取."""
        self.db.writeResult({"初始资金": 100.0}, "A股/IC/沪深300")
        self.db.writeResult({"初始资金": 200.0}, "ETF/IC")

        self.assertEqual(self.db.readResult("A股/IC/沪深300")["初始资金"], 100.0)
        self.assertEqual(self.db.readResult("ETF/IC")["初始资金"], 200.0)

    def test_read_nonexistent(self):
        """读取不存在的结果应返回 None."""
        self.assertIsNone(self.db.readResult("nonexistent"))

    def test_write_overwrites(self):
        """重复写入同一 group 应覆盖."""
        self.db.writeResult({"初始资金": 100.0}, "grp")
        self.db.writeResult({"初始资金": 200.0}, "grp")
        self.assertEqual(self.db.readResult("grp")["初始资金"], 200.0)

    def test_list_results_all(self):
        """listResults() 应返回所有有 metadata 的结果."""
        self.db.writeResult({"初始资金": 1.0}, "A股/IC/沪深300", metadata={"资产": "A股"})
        self.db.writeResult({"初始资金": 2.0}, "A股/IC/中证500", metadata={"资产": "A股"})
        self.db.writeResult({"初始资金": 3.0}, "ETF/IC", metadata={"资产": "ETF"})

        all_results = self.db.listResults()
        self.assertEqual(sorted(all_results), ["A股/IC/中证500", "A股/IC/沪深300", "ETF/IC"])

    def test_list_results_by_metadata(self):
        """listResults(metadata=...) 应按标签筛选."""
        self.db.writeResult({"初始资金": 1.0}, "A股/IC/沪深300", metadata={"资产": "A股", "池子": "沪深300"})
        self.db.writeResult({"初始资金": 2.0}, "A股/IC/中证500", metadata={"资产": "A股", "池子": "中证500"})
        self.db.writeResult({"初始资金": 3.0}, "ETF/IC", metadata={"资产": "ETF"})

        a_stock = self.db.listResults(metadata={"资产": "A股"})
        self.assertEqual(sorted(a_stock), ["A股/IC/中证500", "A股/IC/沪深300"])

        hs300 = self.db.listResults(metadata={"池子": "沪深300"})
        self.assertEqual(hs300, ["A股/IC/沪深300"])

    def test_list_results_empty_dir(self):
        """空目录应返回空列表."""
        self.assertEqual(self.db.listResults(), [])

    def test_metadata_read_write(self):
        """metadata 独立读写."""
        self.db.writeResult({"初始资金": 100.0}, "grp", metadata={"资产": "A股"})
        self.assertEqual(self.db.readMetaData("grp"), {"资产": "A股"})
        self.assertEqual(self.db.readMetaData("grp", key="资产"), "A股")

        self.db.setMetaData("grp", key="策略", value="IC")
        meta = self.db.readMetaData("grp")
        self.assertEqual(meta["资产"], "A股")
        self.assertEqual(meta["策略"], "IC")

    def test_metadata_nonexistent_group(self):
        """不存在的 group 应返回 None / 静默忽略."""
        self.assertIsNone(self.db.readMetaData("nonexistent"))
        self.db.setMetaData("nonexistent", key="k", value="v")  # 不应报错

    def test_no_metadata_not_listed(self):
        """没有 metadata 的结果不应出现在 listResults 中."""
        self.db.writeResult({"初始资金": 1.0}, "with_meta", metadata={"a": 1})
        self.db.writeResult({"初始资金": 2.0}, "without_meta")

        results = self.db.listResults()
        self.assertIn("with_meta", results)
        self.assertNotIn("without_meta", results)

    def test_group_names_property(self):
        """ResultNames 应返回所有结果组."""
        self.db.writeResult({"初始资金": 1.0}, "grp1", metadata={"a": 1})
        self.db.writeResult({"初始资金": 2.0}, "grp2", metadata={"b": 2})
        self.assertEqual(sorted(self.db.ResultNames), ["grp1", "grp2"])


if __name__ == "__main__":
    unittest.main()
