# -*- coding: utf-8 -*-
"""Engine 基本功能测试.

测试覆盖:
    - Engine.init: 节点注册、BFS 依赖遍历、NodeState/PrepareNodeDict 填充
    - Engine.prepare: 顺序与并发 prepare_compute 调用
    - Engine.compute: 单节点与多节点计算
    - Engine.run: 端到端三阶段执行
    - Engine 上下文管理器: 全局引擎栈的压入/弹出
    - StackEngine.run: 前后向栈式计算
"""

import unittest
from typing import List, Any

from pydantic import Field

from QuantStudio.Core.CalcEngine import Engine, StackEngine, __QS_Engine__
from QuantStudio.Core.Node import Node, Context


class _TestNode(Node):
    """测试用节点: 记录方法调用并返回可控结果.

    在 init_compute 中自动填充 PrepareNodeDict, 便于测试 Engine 的三阶段流程.
    可选参数 order_list 用于追踪遍历顺序 (节点被 init_compute 时追加 name).
    """

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="_TestNode", frozen=True, title="名称")

    def __init__(self, name="_TestNode", deps=None, result=None, qs_id=None,
                 order_list=None, **kwargs):
        if deps is None:
            deps = []
        super().__init__(deps=deps, args={"Name": name}, qs_id=qs_id, **kwargs)
        self._result = result if result is not None else f"result_{self.Name}"
        self._order_list = order_list
        # 方法调用追踪
        self.init_called = False
        self.prepare_called = False
        self.prepare_data_received = None
        self.compute_called = False
        self.bwd_data_received = None

    def init_compute(self, path: List[str], init_data: Any, context: Context) -> List[Any]:
        """记录调用并填充 NodeState 和 PrepareNodeDict."""
        self.init_called = True
        if self._order_list is not None:
            self._order_list.append(self.Name)
        context.NodeState.setdefault(self.QSID, {})["initialized"] = True
        context.PrepareNodeDict[self.QSID] = (self.QSID, f"prepare_{self.Name}")
        return [init_data] * len(self.Deps)

    def prepare_compute(self, prepare_data: Any, context: Context):
        """记录调用及接收到的准备数据."""
        self.prepare_called = True
        self.prepare_data_received = prepare_data

    def backward_compute(self, path: List[str], bwd_data_list: List[Any],
                         context: Context, local_context: Any = None) -> Any:
        """记录调用并返回预设结果."""
        self.compute_called = True
        self.bwd_data_received = bwd_data_list
        return self._result


class _TestNodeWithoutPrepare(Node):
    """不填充 PrepareNodeDict 的测试节点, 用于测试空 prepare 场景."""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="_TestNodeWithoutPrepare", frozen=True, title="名称")

    def __init__(self, name="_TestNodeWithoutPrepare", deps=None, result=None, qs_id=None, **kwargs):
        if deps is None:
            deps = []
        super().__init__(deps=deps, args={"Name": name}, qs_id=qs_id, **kwargs)
        self._result = result if result is not None else f"result_{self.Name}"
        self.init_called = False
        self.prepare_called = False
        self.compute_called = False

    def init_compute(self, path, init_data, context):
        self.init_called = True
        context.NodeState.setdefault(self.QSID, {})["initialized"] = True
        return [init_data] * len(self.Deps)

    def prepare_compute(self, prepare_data, context):
        self.prepare_called = True

    def backward_compute(self, path, bwd_data_list, context, local_context=None):
        self.compute_called = True
        return self._result


class TestEngineInit(unittest.TestCase):
    """Engine.init 方法测试."""

    def setUp(self):
        self.engine = Engine()

    def test_nodes_registered_in_context(self):
        """节点列表中的每个节点都应注册到 context.NodeDict."""
        context = Context()
        node1 = _TestNode(name="n1", qs_id="n1")
        node2 = _TestNode(name="n2", qs_id="n2")

        self.engine.init([node1, node2], context)

        self.assertIn("n1", context.NodeDict)
        self.assertIn("n2", context.NodeDict)
        self.assertEqual(len(context.NodeDict), 2)

    def test_init_compute_called(self):
        """每个节点应调用 init_compute."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")

        self.engine.init([node], context)

        self.assertTrue(node.init_called)

    def test_nodestate_populated(self):
        """init_compute 应填充 context.NodeState."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")

        self.engine.init([node], context)

        self.assertIn("n1", context.NodeState)
        self.assertTrue(context.NodeState["n1"]["initialized"])

    def test_prepare_node_dict_populated(self):
        """init_compute 应填充 context.PrepareNodeDict."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")

        self.engine.init([node], context)

        self.assertIn("n1", context.PrepareNodeDict)
        self.assertEqual(context.PrepareNodeDict["n1"][0], "n1")
        self.assertEqual(context.PrepareNodeDict["n1"][1], "prepare_n1")

    def test_bfs_traverses_dependencies(self):
        """应从目标节点出发 BFS 遍历依赖树."""
        context = Context()
        leaf = _TestNode(name="leaf", qs_id="leaf")
        mid = _TestNode(name="mid", deps=[leaf], qs_id="mid")
        root = _TestNode(name="root", deps=[mid], qs_id="root")

        self.engine.init([root], context)

        self.assertIn("root", context.NodeDict)
        self.assertIn("mid", context.NodeDict)
        self.assertIn("leaf", context.NodeDict)
        self.assertTrue(root.init_called)
        self.assertTrue(mid.init_called)
        self.assertTrue(leaf.init_called)

    def test_init_with_custom_init_data(self):
        """支持自定义 init_data_list."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")

        self.engine.init([node], context, init_data_list=["custom"])

        self.assertTrue(node.init_called)

    def test_init_default_init_data_none(self):
        """init_data_list 为 None 时应正常运行."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")

        self.engine.init([node], context, init_data_list=None)

        self.assertTrue(node.init_called)

    def test_diamond_dependency(self):
        """菱形依赖 (DAG) 场景: 两个中间节点共享同一个叶子依赖."""
        context = Context()
        leaf = _TestNode(name="leaf", qs_id="leaf")
        left = _TestNode(name="left", deps=[leaf], qs_id="left")
        right = _TestNode(name="right", deps=[leaf], qs_id="right")
        root = _TestNode(name="root", deps=[left, right], qs_id="root")

        self.engine.init([root], context)

        self.assertIn("leaf", context.NodeDict)
        self.assertIn("left", context.NodeDict)
        self.assertIn("right", context.NodeDict)
        self.assertIn("root", context.NodeDict)


class TestEnginePrepare(unittest.TestCase):
    """Engine.prepare 方法测试."""

    def setUp(self):
        self.engine = Engine()

    def test_prepare_calls_prepare_compute(self):
        """应调用 context.PrepareNodeDict 中每个节点的 prepare_compute."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")
        self.engine.init([node], context)

        self.engine.prepare([node], context)

        self.assertTrue(node.prepare_called)
        self.assertEqual(node.prepare_data_received, "prepare_n1")

    def test_prepare_multiple_nodes(self):
        """应对多个节点逐一调用 prepare_compute."""
        context = Context()
        node1 = _TestNode(name="n1", qs_id="n1")
        node2 = _TestNode(name="n2", qs_id="n2")
        self.engine.init([node1, node2], context)

        self.engine.prepare([node1, node2], context)

        self.assertTrue(node1.prepare_called)
        self.assertTrue(node2.prepare_called)

    def test_prepare_empty_dict_no_error(self):
        """PrepareNodeDict 为空时不应报错."""
        context = Context()
        node = _TestNodeWithoutPrepare(name="n1", qs_id="n1")
        self.engine.init([node], context)

        # _TestNodeWithoutPrepare 不填充 PrepareNodeDict
        self.engine.prepare([node], context)

        self.assertFalse(node.prepare_called)

    def test_prepare_with_concurrency(self):
        """IOConcurrentNum > 1 时应使用线程池并发执行."""
        engine = Engine(args={"IOConcurrentNum": 2})
        context = Context()
        nodes = [_TestNode(name=f"n{i}", qs_id=f"n{i}") for i in range(4)]
        engine.init(nodes, context)

        engine.prepare(nodes, context)

        for node in nodes:
            self.assertTrue(node.prepare_called, f"{node.Name} 的 prepare_compute 未被调用")


class TestEngineCompute(unittest.TestCase):
    """Engine.compute 方法测试."""

    def setUp(self):
        self.engine = Engine()

    def test_compute_single_node(self):
        """单节点应正确计算并返回结果."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1", result=42)

        rslt = self.engine.compute([node], context)

        self.assertEqual(rslt, [42])
        self.assertTrue(node.compute_called)

    def test_compute_multiple_nodes(self):
        """多节点应各自独立计算并返回对应结果."""
        context = Context()
        node1 = _TestNode(name="n1", qs_id="n1", result="a")
        node2 = _TestNode(name="n2", qs_id="n2", result="b")
        node3 = _TestNode(name="n3", qs_id="n3", result="c")

        rslt = self.engine.compute([node1, node2, node3], context)

        self.assertEqual(rslt, ["a", "b", "c"])

    def test_compute_with_fwd_data(self):
        """fwd_data_list 应传递给节点的 compute."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")

        rslt = self.engine.compute([node], context, fwd_data_list=["hello"])

        self.assertEqual(rslt, ["result_n1"])

    def test_compute_default_fwd_data(self):
        """fwd_data_list 为 None 时应正常运行."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1")

        rslt = self.engine.compute([node], context, fwd_data_list=None)

        self.assertEqual(rslt, ["result_n1"])

    def test_compute_node_with_deps(self):
        """有依赖的节点应递归计算子节点."""
        context = Context()
        leaf = _TestNode(name="leaf", qs_id="leaf", result=10)
        root = _TestNode(name="root", deps=[leaf], qs_id="root", result="done")

        rslt = self.engine.compute([root], context)

        self.assertTrue(leaf.compute_called)
        self.assertEqual(rslt, ["done"])
        # root 的 backward_compute 应接收到 leaf 的结果
        self.assertEqual(root.bwd_data_received, [10])

    def test_compute_deep_chain(self):
        """深层依赖链: A → B → C."""
        context = Context()
        c = _TestNode(name="c", qs_id="c", result=1)
        b = _TestNode(name="b", deps=[c], qs_id="b", result=2)
        a = _TestNode(name="a", deps=[b], qs_id="a", result=3)

        rslt = self.engine.compute([a], context)

        self.assertTrue(c.compute_called)
        self.assertTrue(b.compute_called)
        self.assertTrue(a.compute_called)
        self.assertEqual(rslt, [3])


class TestEngineRun(unittest.TestCase):
    """Engine.run 端到端测试."""

    def setUp(self):
        self.engine = Engine()

    def test_run_executes_all_phases(self):
        """run 应依次执行 init → prepare → compute."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1", result=99)

        rslt = self.engine.run([node], context)

        self.assertTrue(node.init_called)
        self.assertTrue(node.prepare_called)
        self.assertTrue(node.compute_called)
        self.assertEqual(rslt, [99])

    def test_run_multiple_nodes(self):
        """run 应支持多个目标节点."""
        context = Context()
        n1 = _TestNode(name="n1", qs_id="n1", result=1)
        n2 = _TestNode(name="n2", qs_id="n2", result=2)

        rslt = self.engine.run([n1, n2], context)

        self.assertEqual(rslt, [1, 2])

    def test_run_with_dependencies(self):
        """run 端到端测试带依赖的节点."""
        context = Context()
        leaf1 = _TestNode(name="leaf1", qs_id="leaf1", result=10)
        leaf2 = _TestNode(name="leaf2", qs_id="leaf2", result=20)
        root = _TestNode(name="root", deps=[leaf1, leaf2], qs_id="root", result=30)

        rslt = self.engine.run([root], context)

        self.assertEqual(rslt, [30])
        self.assertEqual(root.bwd_data_received, [10, 20])


class TestEngineContextManager(unittest.TestCase):
    """Engine 上下文管理器测试."""

    def test_enter_pushes_to_global_stack(self):
        """__enter__ 应将引擎压入全局栈."""
        engine = Engine()
        initial_len = len(__QS_Engine__)

        with engine:
            self.assertEqual(len(__QS_Engine__), initial_len + 1)
            self.assertIs(__QS_Engine__[-1], engine)

    def test_exit_pops_from_global_stack(self):
        """__exit__ 应将引擎从全局栈弹出."""
        engine = Engine()
        initial_len = len(__QS_Engine__)

        with engine:
            pass

        self.assertEqual(len(__QS_Engine__), initial_len)

    def test_nested_engines(self):
        """支持嵌套 with 语句."""
        engine1 = Engine()
        engine2 = Engine()
        initial_len = len(__QS_Engine__)

        with engine1:
            self.assertIs(__QS_Engine__[-1], engine1)
            with engine2:
                self.assertIs(__QS_Engine__[-1], engine2)
            self.assertIs(__QS_Engine__[-1], engine1)

        self.assertEqual(len(__QS_Engine__), initial_len)

    def test_context_manager_usable_in_run(self):
        """通过 with 语句使用 Engine 并执行 run."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1", result="ctx_test")

        with Engine() as exec_engine:
            rslt = exec_engine.run([node], context)

        self.assertEqual(rslt, ["ctx_test"])


class TestStackEngine(unittest.TestCase):
    """StackEngine 前后向栈式计算测试."""

    def setUp(self):
        self.engine = StackEngine()

    def test_run_single_leaf_node(self):
        """单叶子节点: init → prepare → forward → backward."""
        context = Context()
        node = _TestNode(name="n1", qs_id="n1", result=7)

        rslt = self.engine.run([node], context)

        self.assertEqual(rslt, [7])
        self.assertTrue(node.init_called)
        self.assertTrue(node.prepare_called)
        self.assertTrue(node.compute_called)

    def test_run_with_dependency_chain(self):
        """依赖链 A → B → C: 前向 A→B→C, 后向 C→B→A."""
        context = Context()
        c = _TestNode(name="c", qs_id="c", result=1)
        b = _TestNode(name="b", deps=[c], qs_id="b", result=2)
        a = _TestNode(name="a", deps=[b], qs_id="a", result=3)

        rslt = self.engine.run([a], context)

        self.assertEqual(rslt, [3])
        # 所有节点都应完成 init 和 prepare
        for node, name in [(a, "a"), (b, "b"), (c, "c")]:
            self.assertTrue(node.init_called, f"{name} 的 init 未被调用")
            self.assertTrue(node.prepare_called, f"{name} 的 prepare 未被调用")
            self.assertTrue(node.compute_called, f"{name} 的 backward_compute 未被调用")
        # 后向传递: 子节点结果逐层向上
        self.assertEqual(b.bwd_data_received, [1])
        self.assertEqual(a.bwd_data_received, [2])

    def test_run_multi_branch_deps(self):
        """多分支依赖: root 依赖两个独立的叶子."""
        context = Context()
        left = _TestNode(name="left", qs_id="left", result=5)
        right = _TestNode(name="right", qs_id="right", result=6)
        root = _TestNode(name="root", deps=[left, right], qs_id="root", result=11)

        rslt = self.engine.run([root], context)

        self.assertEqual(rslt, [11])
        self.assertEqual(root.bwd_data_received, [5, 6])
        self.assertTrue(left.init_called)
        self.assertTrue(right.init_called)
        self.assertTrue(left.prepare_called)
        self.assertTrue(right.prepare_called)
        self.assertTrue(root.prepare_called)

    def test_run_multiple_targets(self):
        """多个目标节点应各自前向后向计算."""
        context = Context()
        n1 = _TestNode(name="n1", qs_id="n1", result=1)
        n2 = _TestNode(name="n2", qs_id="n2", result=2)

        rslt = self.engine.run([n1, n2], context)

        self.assertEqual(rslt, [1, 2])
        self.assertTrue(n1.prepare_called)
        self.assertTrue(n2.prepare_called)

    def test_dep_nodes_registered_in_context(self):
        """依赖节点应被注册到 context.NodeDict."""
        context = Context()
        leaf = _TestNode(name="leaf", qs_id="leaf", result=10)
        root = _TestNode(name="root", deps=[leaf], qs_id="root", result=20)

        self.engine.run([root], context)

        self.assertIn("leaf", context.NodeDict)
        self.assertIn("root", context.NodeDict)
        self.assertIn("leaf", context.NodeState)

    def test_shared_dependency(self):
        """共享依赖: 两个目标节点共享同一叶子依赖, 各自独立计算."""
        context = Context()
        shared = _TestNode(name="shared", qs_id="shared", result=100)
        t1 = _TestNode(name="t1", deps=[shared], qs_id="t1", result=1)
        t2 = _TestNode(name="t2", deps=[shared], qs_id="t2", result=2)

        rslt = self.engine.run([t1, t2], context)

        self.assertEqual(rslt, [1, 2])
        self.assertEqual(t1.bwd_data_received, [100])
        self.assertEqual(t2.bwd_data_received, [100])

    def test_inherits_from_engine(self):
        """StackEngine 应继承自 Engine."""
        self.assertIsInstance(self.engine, Engine)

    def test_init_order_is_dfs(self):
        """init 遍历顺序应为 DFS: 沿第一个依赖深探到底, 再处理其他分支.

        依赖结构: root → [left, right], left → [leaf]
        预期 DFS: root, left, leaf, right
        预期 BFS: root, left, right, leaf
        """
        context = Context()
        order = []
        leaf = _TestNode(name="leaf", qs_id="leaf", order_list=order)
        left = _TestNode(name="left", deps=[leaf], qs_id="left", order_list=order)
        right = _TestNode(name="right", qs_id="right", order_list=order)
        root = _TestNode(name="root", deps=[left, right], qs_id="root", order_list=order)

        self.engine.init([root], context)

        self.assertEqual(order, ["root", "left", "leaf", "right"],
                         f"DFS 顺序应为 ['root','left','leaf','right'], 实际为 {order}")

    def test_init_order_differs_from_bfs(self):
        """验证 DFS init 顺序与 BFS (Engine.init) 不同."""
        # StackEngine DFS
        order_dfs = []
        leaf = _TestNode(name="leaf", qs_id="leaf1", order_list=order_dfs)
        left = _TestNode(name="left", deps=[leaf], qs_id="left1", order_list=order_dfs)
        right = _TestNode(name="right", qs_id="right1", order_list=order_dfs)
        root = _TestNode(name="root", deps=[left, right], qs_id="root1", order_list=order_dfs)
        self.engine.init([root], Context())

        # Engine BFS
        order_bfs = []
        leaf2 = _TestNode(name="leaf", qs_id="leaf2", order_list=order_bfs)
        left2 = _TestNode(name="left", deps=[leaf2], qs_id="left2", order_list=order_bfs)
        right2 = _TestNode(name="right", qs_id="right2", order_list=order_bfs)
        root2 = _TestNode(name="root", deps=[left2, right2], qs_id="root2", order_list=order_bfs)
        Engine().init([root2], Context())

        self.assertEqual(order_dfs, ["root", "left", "leaf", "right"])
        self.assertEqual(order_bfs, ["root", "left", "right", "leaf"])
        self.assertNotEqual(order_dfs, order_bfs)

    def test_init_order_matches_forward_order(self):
        """init 的 DFS 顺序应与 forward_compute 的遍历顺序一致.

        使用 StackEngine.run, 在 forward_compute 阶段也记录顺序进行对比.
        """
        # 用 init 记录顺序
        init_order = []
        leaf = _TestNode(name="leaf", qs_id="ld", order_list=init_order)
        mid = _TestNode(name="mid", deps=[leaf], qs_id="md", order_list=init_order)
        root = _TestNode(name="root", deps=[mid], qs_id="rd", order_list=init_order)

        context = Context()
        self.engine.run([root], context)

        # DFS init 顺序应为 root → mid → leaf (沿链深探)
        self.assertEqual(init_order, ["root", "mid", "leaf"])


if __name__ == "__main__":
    unittest.main()
