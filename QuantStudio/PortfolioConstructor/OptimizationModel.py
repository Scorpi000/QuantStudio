# -*- coding: utf-8 -*-
"""优化模型的组件配置、图输入提取和当期问题构造。"""
import copy
import inspect
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, SerializeAsAny

from QuantStudio.Core import __QS_Args__, __QS_Error__
from QuantStudio.Core.Node import Node
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Risk.RiskTable import RiskTable, FactorRT
from .BasePC import OptimizationObjective, Constraint, PortfolioConstructor
from .CVXPC import CVXPC


class _SourceRef(__QS_Args__):
    """模型中的输入位置；节点本身只保存在算子的输入绑定及图依赖中。"""

    kind: Literal["factor", "risk"]
    index: int = Field(ge=0)
    field: str = "Cov"


class _ComponentSpec(__QS_Args__):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    component: type
    args: SerializeAsAny[__QS_Args__]
    inputs: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, component, args=None, inputs=None, **kwargs):
        values = dict(args or {})
        unknown = values.keys() - component.__QS_ArgClass__.model_fields.keys()
        if unknown:
            raise ValueError(f"{component.__name__} 未声明参数: {sorted(unknown)}")
        args = component.__QS_ArgClass__(**values)
        signature = inspect.signature(component.__init__)
        names = {name for name, param in signature.parameters.items()
                 if name not in ("self", "mask", "args", "config_file")
                 and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)}
        inputs = dict(inputs or {})
        if inputs.keys() - names:
            raise ValueError(f"{component.__name__} 未声明数据输入: {sorted(inputs.keys() - names)}")
        missing = {name for name in names if signature.parameters[name].default is inspect.Parameter.empty} - inputs.keys()
        if missing:
            raise ValueError(f"{component.__name__} 缺少数据输入: {sorted(missing)}")
        super().__init__(component=component, args=args, inputs=inputs, **kwargs)

    def build(self, mask, ids, resolve, logger=None):
        """解析当期输入并实例化原有目标或约束，统一注入 mask。"""
        values = {name: resolve(value) for name, value in self.inputs.items()}
        exposure = values.get("factor_data")
        factor_cov = values.get("factor_cov")
        if isinstance(exposure, pd.DataFrame) and isinstance(factor_cov, pd.DataFrame):
            values["factor_data"] = exposure.reindex(columns=factor_cov.index)
            values["factor_cov"] = factor_cov.reindex(columns=factor_cov.index)
        values = {name: _align_input(name, value, ids) for name, value in values.items()}
        exposure = values.get("factor_data")
        if exposure is not None and np.ndim(exposure) == 1:
            values["factor_data"] = np.asarray(exposure).reshape(len(ids), 1)
        return self.component(mask=mask, args=self.args.to_dict(repr=False), logger=logger, **values)


class ObjectiveSpec(_ComponentSpec):
    """优化目标类型、该类型的原生参数及数据绑定。"""

    component: type[OptimizationObjective]


class ConstraintSpec(_ComponentSpec):
    """约束类型、该类型的原生参数（含 DropPriority）及数据绑定。"""

    component: type[Constraint]


class OptimizationModel(__QS_Args__):
    """可组合模型。目标和约束共享股票池；后端可由 PortfolioConstructor 子类扩展。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    objective: ObjectiveSpec
    constraints: tuple[ConstraintSpec, ...] = ()
    backend: type[PortfolioConstructor] = CVXPC

    def __init__(self, objective, constraints=(), backend=CVXPC, **kwargs):
        super().__init__(objective=objective, constraints=tuple(constraints), backend=backend, **kwargs)
        if hasattr(backend, "validate_model"):
            backend.validate_model(self.objective.component, tuple(spec.component for spec in self.constraints))

    def compile(self):
        """提取去重后的因子和风险节点，返回不含节点的模型及两组输入绑定。"""
        factors, risks = [], []
        fields = {"cov": "Cov", "factor_cov": "FactorCov", "factor_data": "FactorData",
                  "specific_risk": "SpecificRisk"}

        def extract(value, name):
            if isinstance(value, (Factor, RiskTable)):
                nodes = factors if isinstance(value, Factor) else risks
                index = next((i for i, node in enumerate(nodes) if node is value), len(nodes))
                if index == len(nodes):
                    nodes.append(value)
                if isinstance(value, Factor):
                    return _SourceRef(kind="factor", index=index)
                if name not in fields or (name != "cov" and not isinstance(value, FactorRT)):
                    raise ValueError(f"风险表不能提供输入 {name}")
                return _SourceRef(kind="risk", index=index, field=fields[name])
            if isinstance(value, Node):
                raise TypeError("优化模型的数据依赖必须是 Factor 或 RiskTable")
            if isinstance(value, (tuple, list)):
                return tuple(extract(item, name) for item in value)
            if isinstance(value, dict):
                return {key: extract(item, name) for key, item in value.items()}
            return copy.deepcopy(value)

        def compile_spec(spec):
            return type(spec)(spec.component, args=spec.args,
                              inputs={name: extract(value, name) for name, value in spec.inputs.items()})

        model = OptimizationModel(compile_spec(self.objective),
                                  tuple(compile_spec(spec) for spec in self.constraints), self.backend)
        return model, tuple(factors), tuple(risks)

    def risk_fields(self, count):
        """按风险依赖汇总所需字段，同一风险表只调度一次。"""
        fields = [set() for _ in range(count)]

        def visit(value):
            if isinstance(value, _SourceRef) and value.kind == "risk":
                fields[value.index].add(value.field)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)

        for spec in (self.objective, *self.constraints):
            visit(spec.inputs)
        return tuple(tuple(sorted(items)) for items in fields)

    def solve(self, mask, ids, resolve=lambda value: value, optim_option=None, logger=None):
        """构造并求解一个截面，也可在逐期账户策略中用当期数组直接调用。"""
        objective = self.objective.build(mask, ids, resolve, logger)
        constraints = [spec.build(mask, ids, resolve, logger) for spec in self.constraints]
        return self.backend(mask=mask, objective=objective, constraints=constraints,
                            args={"OptimOption": optim_option or {}}, logger=logger).solve()


def resolve_sources(value, factors, risks, idt):
    """从图计算结果中取得模型组件需要的当期数据。"""
    if isinstance(value, _SourceRef):
        if value.kind == "factor":
            return factors[value.index]
        data = risks[value.index]
        data = data[value.field] if isinstance(data, dict) else data
        if value.field == "FactorData":  # 因子 × 日期 × 证券
            return data.loc[:, idt, :]
        return data.loc[idt]  # Cov / FactorCov: 日期 × 矩阵
    if isinstance(value, (tuple, list)):
        return tuple(resolve_sources(item, factors, risks, idt) for item in value)
    if isinstance(value, dict):
        return {key: resolve_sources(item, factors, risks, idt) for key, item in value.items()}
    return value


def _align_input(name, value, ids):
    """将带标签输入对齐到证券截面，数值数组按当前截面顺序解释。"""
    if value is None:
        return None
    if isinstance(value, Node):
        raise TypeError("图依赖尚未解析为当期数据")
    vectors = {"expected_return", "p0", "bmk", "budget", "specific_risk", "up_limit", "down_limit", "amt"}
    if isinstance(value, pd.Series):
        value = value.reindex(ids).to_numpy()
    elif isinstance(value, pd.DataFrame):
        if not value.index.is_unique or not value.columns.is_unique:
            raise ValueError(f"{name} 的标签重复")
        if name == "cov":
            value = value.reindex(index=ids, columns=ids).to_numpy(dtype=float)
        elif name == "factor_cov":
            value = value.reindex(columns=value.index).to_numpy(dtype=float)
        else:
            value = value.reindex(index=ids).to_numpy()
            if name in vectors:
                value = value.reshape(-1)
    elif name == "factor_data" and isinstance(value, (tuple, list)) and any(isinstance(item, np.ndarray) for item in value):
        value = np.column_stack(value)
    if name in vectors:
        value = np.full(len(ids), value) if np.ndim(value) == 0 else np.asarray(value)
        if value.shape != (len(ids),):
            raise ValueError(f"{name} 必须是长度为 {len(ids)} 的证券向量")
    if name in ("cov", "factor_cov"):
        value = np.asarray(value, dtype=float)
        shape = (len(ids), len(ids)) if name == "cov" else (value.shape[0], value.shape[0]) if value.ndim == 2 else ()
        if value.ndim != 2 or value.shape != shape or not np.isfinite(value).all() or not np.allclose(value, value.T):
            raise ValueError(f"{name} 必须是覆盖完整截面、有限且对称的协方差矩阵")
    if name == "wealth" and np.ndim(value):
        if not np.all(value == value[0]):
            raise ValueError("wealth 因子在同一期必须是相同的账户总金额")
        value = float(value[0])
    return value
