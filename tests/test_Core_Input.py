# -*- coding: utf-8 -*-
"""依赖模型、节点模板、执行身份和多引擎的验收测试。"""
import unittest
import pickle

from pydantic import Field, ValidationError, model_validator

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.api import Node, Context, Engine, DependencyModel, Binding, InputError, NodeTemplate, ExecutionEffect
from QuantStudio.Core.CalcEngine import StackEngine
from QuantStudio.Core.TreeEngine import TreeEngine


class Leaf(Node):
    """返回声明参数的无依赖节点。"""

    class __QS_DepClass__(DependencyModel):
        pass

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Value: int = Field(title="节点值")

    def backward_compute(self, path, inputs, context, local_context=None):
        return self.Args.Value


class SumNode(Node):
    """验证单值、列表和可选槽位的结果对应关系。"""

    class __QS_DepClass__(DependencyModel):
        head: Node = Field(title="首项")
        tail: tuple[Node, ...] = Field(min_length=1, max_length=3, title="后续项")
        mask: Node | None = Field(default=None, title="可选掩码")

    def backward_compute(self, path, inputs, context, local_context=None):
        return inputs.head + sum(inputs.tail) if inputs.mask is None or inputs.mask else 0


class TestInput(unittest.TestCase):
    def setUp(self):
        self.one = Leaf(args={"Value": 1})
        self.two = Leaf(args={"Value": 2})

    def test_partial_template_and_complete_construction(self):
        template = SumNode.template(deps={"head": self.one})
        self.assertIsInstance(template, NodeTemplate)
        self.assertNotIsInstance(template, Node)
        self.assertEqual(template.missing_dependencies, ("tail",))
        self.assertEqual(template.dependency_info()["head"]["title"], "首项")
        with self.assertRaises(InputError):
            template.build()
        complete = template.bind(tail=(self.two,))
        node = complete.build()
        self.assertIs(node.Dependencies.head, self.one)
        self.assertEqual(template.missing_dependencies, ("tail",))
        self.assertEqual(Engine().run([node], Context()), [3])

    def test_partial_parameters_do_not_initialize_node(self):
        template = Leaf.template()
        self.assertTrue(template.validate())
        with self.assertRaises(InputError):
            template.build()
        leaf = template.with_args(Value=7).build()
        self.assertEqual(Engine().run([leaf], Context()), [7])

    def test_missing_null_and_empty_collection_are_distinct(self):
        template = SumNode.template(deps={"head": self.one})
        with self.assertRaises(InputError):
            template.bind(head=None)
        short = template.bind(tail=())
        self.assertNotIn("tail", short.missing_dependencies)
        self.assertEqual(short.validate()[0]["path"], ("deps", "tail"))
        self.assertFalse(template.bind(tail=(self.two,), mask=None).validate())

    def test_type_unknown_input_and_upper_bound(self):
        template = SumNode.template()
        for values in ({"unknown": self.one}, {"head": template}, {"tail": (self.one,) * 4}):
            with self.subTest(values=values), self.assertRaises(InputError):
                template.bind(**values)

    def test_immutable_models_and_binding_replacement(self):
        template = SumNode.template(deps={"head": self.one, "tail": (self.two,)})
        updated = template.bind(head=self.two).unbind("tail")
        self.assertIs(template.Dependencies["head"], self.one)
        self.assertIs(updated.Dependencies["head"], self.two)
        node = template.build()
        with self.assertRaises(ValidationError):
            node.Dependencies.head = self.two
        node.Deps.clear()
        self.assertEqual(len(tuple(node.iter_dependency_edges())), 2)

    def test_edge_paths_and_named_results(self):
        node = SumNode(deps={"head": self.one, "tail": (self.one, self.two)})
        self.assertEqual([edge.path for edge in node.iter_dependency_edges()], [("head",), ("tail", 0), ("tail", 1)])
        values = node.dependency_values([1, 1, 2])
        self.assertEqual(values.tail, (1, 2))
        self.assertIsNone(values.mask)
        with self.assertRaises(ValueError):
            node.dependency_values([1])

    def test_all_engines_preserve_repeated_dependency_slots(self):
        node = SumNode(deps={"head": self.one, "tail": (self.one, self.two)})
        engines = (Engine(), StackEngine(),
                   TreeEngine(args={"CalcConcurrentMode": "Thread", "CalcConcurrentNum": 2}),
                   TreeEngine(args={"CalcConcurrentMode": "Process", "CalcConcurrentNum": 2}))
        for engine in engines:
            with self.subTest(engine=type(engine).__name__, args=engine.Args):
                self.assertEqual(engine.run([node], Context()), [4])

    def test_identical_qsid_has_independent_runtime_state(self):
        other = Leaf(args={"Value": 1})
        self.assertEqual(self.one.QSID, other.QSID)
        self.assertNotEqual(self.one.InstanceID, other.InstanceID)
        context = Context()
        Engine().run([self.one, other], context)
        self.assertEqual(set(context.NodeDict), {self.one.InstanceID, other.InstanceID})
        first_run = context.RunID
        Engine().run([self.two], context)
        self.assertNotEqual(context.RunID, first_run)
        self.assertEqual(set(context.NodeDict), {self.two.InstanceID})

    def test_execution_key_includes_request_and_data_version(self):
        first = self.one.execution_key({"ids": ("A",)}, "v1")
        self.assertNotEqual(first, self.one.execution_key({"ids": ("B",)}, "v1"))
        self.assertNotEqual(first, self.one.execution_key({"ids": ("A",)}, "v2"))

    def test_template_and_named_values_pickle(self):
        template = SumNode.template(deps={"head": self.one, "tail": (self.two,)})
        restored = pickle.loads(pickle.dumps(template))
        self.assertEqual(restored.build().QSID, template.build().QSID)

    def test_engine_rejects_template_before_initialization(self):
        with self.assertRaises(TypeError):
            Engine().run([SumNode.template()], Context())

    def test_complete_model_validator(self):
        class Dependencies(DependencyModel):
            left: Node
            right: Node

            @model_validator(mode="after")
            def distinct(self):
                if self.left is self.right:
                    raise ValueError("两个槽位要求不同实例")
                return self

        binding = Binding(Dependencies, {"left": self.one})
        self.assertTrue(binding.validate())
        with self.assertRaises(InputError):
            binding.bind(right=self.one).build()
        self.assertFalse(binding.bind(right=self.two).validate())

    def test_execution_events_are_application_independent(self):
        events = []
        context = Context(EventHandler=events.append)
        Engine().run([self.one], context)
        self.assertEqual([event["status"] for event in events if event["stage"] == "compute"], ["running", "done"])
        self.assertEqual(events[-1]["run_id"], context.RunID)


if __name__ == "__main__":
    unittest.main()
