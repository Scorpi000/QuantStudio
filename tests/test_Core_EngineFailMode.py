# -*- coding: utf-8 -*-
"""Engine FailMode 容错机制测试.

测试覆盖:
    - Engine: FailMode="skip" 下 init/prepare/compute 各阶段的错误隔离
    - StackEngine: forward/backward 阶段的容错
    - TreeEngine: Thread 并发模式下的容错和失败传播
    - ParallelEngine: 多进程模式下的容错
"""

import os
import unittest
from typing import List, Any

from pydantic import Field

from QuantStudio.Core.CalcEngine import Engine, StackEngine
from QuantStudio.Core.TreeEngine import TreeEngine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Core.Node import Node, Context


class _TestNode(Node):
    """测试用正常节点, 返回预设结果."""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="_TestNode", frozen=True, title="名称")

    def __init__(self, name="_TestNode", deps=None, result=None, qs_id=None, **kwargs):
        if deps is None:
            deps = []
        super().__init__(deps=deps, args={"Name": name}, qs_id=qs_id, **kwargs)
        self._result = result if result is not None else f"result_{self.Name}"

    def init_compute(self, path, init_data, context):
        context.PrepareNodeDict[self.QSID] = (self.QSID, f"prepare_{self.Name}")
        return [init_data] * len(self.Deps)

    def prepare_compute(self, prepare_data, context):
        pass

    def backward_compute(self, path, bwd_data_list, context, local_context=None):
        return self._result


class _FailingNode(Node):
    """测试用失败节点, 可在 init/prepare/backward 阶段抛出异常."""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="_FailingNode", frozen=True, title="名称")

    def __init__(self, name="_FailingNode", deps=None, qs_id=None, fail_in="backward", **kwargs):
        if deps is None:
            deps = []
        super().__init__(deps=deps, args={"Name": name}, qs_id=qs_id, **kwargs)
        self._fail_in = fail_in

    def init_compute(self, path, init_data, context):
        if self._fail_in == "init":
            raise RuntimeError(f"init 失败: {self.Name}")
        context.PrepareNodeDict[self.QSID] = (self.QSID, f"prepare_{self.Name}")
        return [init_data] * len(self.Deps)

    def prepare_compute(self, prepare_data, context):
        if self._fail_in == "prepare":
            raise RuntimeError(f"prepare 失败: {self.Name}")

    def backward_compute(self, path, bwd_data_list, context, local_context=None):
        if self._fail_in == "backward":
            raise RuntimeError(f"compute 失败: {self.Name}")
        return f"result_{self.Name}"


class TestEngineFailMode(unittest.TestCase):
    """Engine FailMode 容错测试."""

    def test_fail_mode_raise_raises(self):
        """默认模式 (raise) 下, 失败节点应导致整个计算图报错."""
        engine = Engine()
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad")
        good = _TestNode(name="good", qs_id="good", result="ok")

        with self.assertRaises(RuntimeError):
            engine.run([bad, good], context)

    def test_fail_mode_skip_single_node(self):
        """skip 模式下, 单个失败节点结果为 None, 其他节点正常."""
        engine = Engine(args={"FailMode": "skip"})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    def test_fail_mode_skip_downstream_skipped(self):
        """skip 模式下, 依赖失败节点的下游应被标记为 SKIPPED."""
        engine = Engine(args={"FailMode": "skip"})
        context = Context()
        leaf = _FailingNode(name="leaf", qs_id="leaf")
        root = _TestNode(name="root", deps=[leaf], qs_id="root", result="should_not_reach")

        rslt = engine.run([root], context)

        self.assertIsNone(rslt[0])
        self.assertIn("leaf", context.NodeErrors)
        self.assertEqual(context.NodeState.get("root", {}).get("__status__"), "SKIPPED")

    def test_fail_mode_skip_init_failure(self):
        """skip 模式下, init 阶段失败的容错."""
        engine = Engine(args={"FailMode": "skip"})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad", fail_in="init")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    def test_fail_mode_skip_prepare_failure(self):
        """skip 模式下, prepare 阶段失败的容错."""
        engine = Engine(args={"FailMode": "skip"})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad", fail_in="prepare")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    def test_fail_mode_skip_diamond_deps(self):
        """skip 模式下, 菱形依赖场景: 共享依赖失败, 两个下游都被跳过."""
        engine = Engine(args={"FailMode": "skip"})
        context = Context()
        leaf = _FailingNode(name="leaf", qs_id="leaf")
        left = _TestNode(name="left", deps=[leaf], qs_id="left", result=1)
        right = _TestNode(name="right", deps=[leaf], qs_id="right", result=2)
        root = _TestNode(name="root", deps=[left, right], qs_id="root", result=3)

        rslt = engine.run([root], context)

        self.assertIsNone(rslt[0])
        self.assertIn("leaf", context.NodeErrors)

    def test_fail_mode_skip_error_message(self):
        """skip 模式下, NodeErrors 记录正确的异常信息."""
        engine = Engine(args={"FailMode": "skip"})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad")

        engine.run([bad], context)

        self.assertIn("bad", context.NodeErrors)
        self.assertIn("compute 失败", str(context.NodeErrors["bad"]))

    def test_fail_mode_default_is_raise(self):
        """FailMode 默认应为 raise."""
        engine = Engine()
        self.assertEqual(engine._QSArgs.FailMode, "raise")


class TestStackEngineFailMode(unittest.TestCase):
    """StackEngine FailMode 容错测试."""

    def test_fail_mode_skip_single_node(self):
        """StackEngine skip 模式下, 失败节点结果为 None, 其他节点正常."""
        engine = StackEngine(args={"FailMode": "skip"})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    def test_fail_mode_skip_with_deps(self):
        """StackEngine skip 模式下, 依赖失败节点的下游被跳过."""
        engine = StackEngine(args={"FailMode": "skip"})
        context = Context()
        leaf = _FailingNode(name="leaf", qs_id="leaf")
        root = _TestNode(name="root", deps=[leaf], qs_id="root", result="should_not")

        rslt = engine.run([root], context)

        self.assertIsNone(rslt[0])
        self.assertIn("leaf", context.NodeErrors)

    def test_fail_mode_skip_init_failure(self):
        """StackEngine skip 模式下, init 阶段失败的容错."""
        engine = StackEngine(args={"FailMode": "skip"})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad", fail_in="init")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")

    def test_fail_mode_raise_raises(self):
        """StackEngine 默认模式下失败节点应导致报错."""
        engine = StackEngine()
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad")

        with self.assertRaises(RuntimeError):
            engine.run([bad], context)


class TestTreeEngineFailMode(unittest.TestCase):
    """TreeEngine FailMode 容错测试."""

    def test_fail_mode_skip_single_node(self):
        """TreeEngine (Thread) skip 模式下, 失败节点结果为 None, 其他节点正常."""
        engine = TreeEngine(args={"FailMode": "skip", "CalcConcurrentMode": "Thread", "CalcConcurrentNum": 2})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    def test_fail_mode_skip_downstream_skipped(self):
        """TreeEngine skip 模式下, 依赖失败节点的下游被跳过."""
        engine = TreeEngine(args={"FailMode": "skip", "CalcConcurrentMode": "Thread", "CalcConcurrentNum": 2})
        context = Context()
        leaf = _FailingNode(name="leaf", qs_id="leaf")
        root = _TestNode(name="root", deps=[leaf], qs_id="root", result="no")

        rslt = engine.run([root], context)

        self.assertIsNone(rslt[0])
        self.assertIn("leaf", context.NodeErrors)

    def test_fail_mode_skip_multi_branch(self):
        """TreeEngine skip 模式下, 多分支依赖: 失败分支被跳过, 成功分支正常."""
        engine = TreeEngine(args={"FailMode": "skip", "CalcConcurrentMode": "Thread", "CalcConcurrentNum": 2})
        context = Context()
        bad_leaf = _FailingNode(name="bad_leaf", qs_id="bad_leaf")
        good_leaf = _TestNode(name="good_leaf", qs_id="good_leaf", result=10)
        root = _TestNode(name="root", deps=[bad_leaf, good_leaf], qs_id="root", result="merged")

        rslt = engine.run([root], context)

        # root 依赖 bad_leaf (失败), 所以 root 也被跳过
        self.assertIsNone(rslt[0])
        self.assertIn("bad_leaf", context.NodeErrors)

    def test_fail_mode_skip_init_failure(self):
        """TreeEngine skip 模式下, init 阶段失败的容错."""
        engine = TreeEngine(args={"FailMode": "skip", "CalcConcurrentMode": "Thread", "CalcConcurrentNum": 2})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad", fail_in="init")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    def test_fail_mode_skip_prepare_failure(self):
        """TreeEngine skip 模式下, prepare 阶段失败的容错."""
        engine = TreeEngine(args={"FailMode": "skip", "CalcConcurrentMode": "Thread", "CalcConcurrentNum": 2})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad", fail_in="prepare")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    def test_fail_mode_raise_raises(self):
        """TreeEngine 默认模式下失败节点应导致报错."""
        engine = TreeEngine(args={"CalcConcurrentMode": "Thread", "CalcConcurrentNum": 2})
        context = Context()
        bad = _FailingNode(name="bad", qs_id="bad")

        with self.assertRaises(RuntimeError):
            engine.run([bad], context)

    def test_fail_mode_default_is_raise(self):
        """TreeEngine FailMode 默认应为 raise."""
        engine = TreeEngine()
        self.assertEqual(engine._QSArgs.FailMode, "raise")


class TestParallelEngineFailMode(unittest.TestCase):
    """ParallelEngine FailMode 容错测试.

    注意: ParallelEngine 使用多进程, 在 Windows 上可能因 fork+Queue 兼容性问题导致卡死.
    如遇此情况, 可通过设置环境变量 SKIP_PARALLEL_TESTS=1 跳过.
    """

    def _make_context(self, n_pids=2):
        """创建带多进程支持的 Context."""
        from multiprocess import SimpleQueue
        ctx = Context(PIDList=[str(i) for i in range(n_pids)])
        ctx.Sub2MainQueue = SimpleQueue()
        return ctx

    @unittest.skipIf(os.environ.get("SKIP_PARALLEL_TESTS"), "跳过多进程测试")
    def test_fail_mode_skip_single_node(self):
        """ParallelEngine skip 模式下, 失败节点结果为 None, 其他节点正常."""
        engine = ParallelEngine(args={"FailMode": "skip"})
        context = self._make_context()
        bad = _FailingNode(name="bad", qs_id="bad")
        good = _TestNode(name="good", qs_id="good", result="ok")

        rslt = engine.run([bad, good], context)

        self.assertIsNone(rslt[0])
        self.assertEqual(rslt[1], "ok")
        self.assertIn("bad", context.NodeErrors)

    @unittest.skipIf(os.environ.get("SKIP_PARALLEL_TESTS"), "跳过多进程测试")
    def test_fail_mode_skip_multiple_nodes(self):
        """ParallelEngine skip 模式下, 多个节点混合成功和失败."""
        engine = ParallelEngine(args={"FailMode": "skip"})
        context = self._make_context()
        n1 = _TestNode(name="n1", qs_id="n1", result=1)
        bad = _FailingNode(name="bad", qs_id="bad")
        n3 = _TestNode(name="n3", qs_id="n3", result=3)

        rslt = engine.run([n1, bad, n3], context)

        self.assertEqual(rslt[0], 1)
        self.assertIsNone(rslt[1])
        self.assertEqual(rslt[2], 3)
        self.assertIn("bad", context.NodeErrors)

    @unittest.skipIf(os.environ.get("SKIP_PARALLEL_TESTS"), "跳过多进程测试")
    def test_fail_mode_raise_raises(self):
        """ParallelEngine 默认模式下失败节点应导致报错."""
        engine = ParallelEngine()
        context = self._make_context()
        bad = _FailingNode(name="bad", qs_id="bad")

        with self.assertRaises(RuntimeError):
            engine.run([bad], context)

    def test_fail_mode_default_is_raise(self):
        """ParallelEngine FailMode 默认应为 raise."""
        engine = ParallelEngine()
        self.assertEqual(engine._QSArgs.FailMode, "raise")


if __name__ == "__main__":
    unittest.main()
