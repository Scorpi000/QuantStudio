# -*- coding: utf-8 -*-
"""可直接注册到计算图的优化目标、约束、问题与独立求解节点。"""
from datetime import datetime
from threading import RLock
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Identity import ExecutionEffect
from QuantStudio.Core.Input import DependencyModel
from QuantStudio.Core.Node import Node
from QuantStudio.Core.SectionExecution import init_section, section_plan, cached_section, publish_section
from QuantStudio.Factor.Factor import DataFactor, Factor, FactorInitData, FactorLocalContext
from QuantStudio.Risk.RiskTable import RiskTable, FactorRT
from . import BasePC as native
from .Numerical import solve_period
from .Types import PeriodComponentData, PeriodProblem, PortfolioResult, SolveResult


class PortfolioNode(Node):
    """所有依赖共享完整截面的组合节点。"""

    __QS_ExecutionEffect__ = ExecutionEffect.PURE
    __QS_ParallelBroadcast__ = True

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        SectionIDs: tuple[str, ...] | None = Field(default=None, frozen=True, title="完整证券截面")

    def section_ids(self, data, context):
        """取得稳定的完整证券顺序。"""
        ids = self.Args.SectionIDs or getattr(data, "SectionIDs", None) or context.SectionIDs
        ids = tuple(ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("优化需要非空且不重复的完整证券截面")
        return ids

    def init_compute(self, path, init_data, context):
        ids = self.section_ids(init_data, context)
        dt_range = init_section(self, init_data, ids, context)
        return super().init_compute(path, FactorInitData(DTRange=dt_range, SectionIDs=list(ids)), context)

    def merge_result(self, result_list, context):
        """各进程已取得完整结构化结果，根输出保留一份。"""
        return result_list[0]

    def forward_compute(self, path, fwd_data, context):
        ids = self.section_ids(fwd_data, context)
        local = FactorLocalContext(DTs=fwd_data.DTs, IDs=list(ids), SectionIDs=list(ids), PIDs=context.PIDList)
        requests = []
        for edge in self.iter_dependency_edges():
            request = local.model_copy(deep=True)
            if isinstance(edge.node, FactorRT):
                request.ExtraData["Fields"] = ["FactorCov", "FactorData", "SpecificRisk"]
            requests.append(request)
        return requests, local


class _ComponentNode(PortfolioNode):
    """读取组件的外生依赖，账户输入留到决策时绑定。"""

    ComponentClass = None

    class __QS_ArgClass__(PortfolioNode.__QS_ArgClass__):
        HoldingSource: Literal["不使用", "外部因子", "账户状态"] = Field(default="不使用", title="初始权重来源")
        WealthSource: Literal["不使用", "外部因子", "账户状态"] = Field(default="不使用", title="净资产来源")

    def init_compute(self, path, init_data, context):
        for source, name in ((self.Args.HoldingSource, "p0"), (self.Args.WealthSource, "wealth")):
            dependency = getattr(self.Dependencies, name, None)
            if source == "账户状态" and dependency is not None:
                raise ValueError(f"{name} 不能同时绑定账户状态和外部因子")
            if (source == "外部因子") != (dependency is not None):
                raise ValueError(f"{name} 的来源声明与外部输入不匹配")
        return super().init_compute(path, init_data, context)

    def backward_compute(self, path, inputs, context, local_context=None):
        ids = tuple(local_context.IDs)
        args = {name: getattr(self.Args, name) for name in self.ComponentClass.__QS_ArgClass__.model_fields}
        result = {}
        for date in local_context.DTs:
            values = {}
            for name in type(self.Dependencies).model_fields:
                dependency, value = getattr(self.Dependencies, name), inputs[name]
                if dependency is None:
                    continue
                if name == "risk_table":
                    if isinstance(dependency, FactorRT):
                        values.update(factor_cov=value["FactorCov"].loc[date],
                                      factor_data=value["FactorData"].loc[:, date, :],
                                      specific_risk=value["SpecificRisk"].loc[date])
                    else:
                        values["cov"] = value.loc[date]
                elif isinstance(dependency, tuple):
                    values[name] = tuple(item.loc[date].reindex(ids).to_numpy() for item in value)
                else:
                    values[name] = value.loc[date].reindex(ids).to_numpy()
            result[date] = PeriodComponentData(self.ComponentClass, args, values, ids, self.InstanceID,
                                               self.Args.HoldingSource, self.Args.WealthSource)
        return result


class OptimizationObjectiveNode(_ComponentNode):
    """目标节点公共类型。"""


class _RiskInputs(DependencyModel):
    risk_table: RiskTable | None = Field(default=None, title="风险表")


class MeanVarianceObjectiveNode(OptimizationObjectiveNode):
    """均值方差目标，支持收益、风险、基准和真实持仓费用惩罚。"""

    ComponentClass = native.MeanVarianceObjective

    class __QS_DepClass__(_RiskInputs):
        expected_return: Factor | None = Field(default=None, title="预期收益")
        bmk: Factor | None = Field(default=None, title="基准权重")
        p0: Factor | None = Field(default=None, title="外部初始权重")

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.MeanVarianceObjective.__QS_ArgClass__):
        pass


class RiskBudgetObjectiveNode(OptimizationObjectiveNode):
    """风险预算目标；后端不支持附加外部约束。"""

    ComponentClass = native.RiskBudgetObjective

    class __QS_DepClass__(_RiskInputs):
        budget: Factor | None = Field(default=None, title="风险预算")

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.RiskBudgetObjective.__QS_ArgClass__):
        pass


class MaxDiversificationObjectiveNode(OptimizationObjectiveNode):
    """最大分散化目标。"""

    ComponentClass = native.MaxDiversificationObjective
    __QS_DepClass__ = _RiskInputs

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.MaxDiversificationObjective.__QS_ArgClass__):
        pass


class ConstraintNode(_ComponentNode):
    """约束节点公共类型。"""


class _BenchmarkInputs(DependencyModel):
    bmk: Factor | None = Field(default=None, title="基准权重")


class BudgetConstraintNode(ConstraintNode):
    """总证券权重约束。"""

    ComponentClass = native.BudgetConstraint
    __QS_DepClass__ = _BenchmarkInputs

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.BudgetConstraint.__QS_ArgClass__):
        pass


class WeightConstraintNode(ConstraintNode):
    """单证券权重约束。"""

    ComponentClass = native.WeightConstraint

    class __QS_DepClass__(_BenchmarkInputs):
        up_limit: Factor = Field(default_factory=lambda: DataFactor(1.), title="权重上限")
        down_limit: Factor = Field(default_factory=lambda: DataFactor(0.), title="权重下限")

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.WeightConstraint.__QS_ArgClass__):
        pass


class FactorExposeConstraintNode(ConstraintNode):
    """一个或多个数值/分类因子的组合暴露约束。"""

    ComponentClass = native.FactorExposeConstraint

    class __QS_DepClass__(_BenchmarkInputs):
        factor_data: tuple[Factor, ...] = Field(min_length=1, title="暴露因子")

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.FactorExposeConstraint.__QS_ArgClass__):
        pass


class VolatilityConstraintNode(ConstraintNode):
    """绝对或相对基准的波动率约束。"""

    ComponentClass = native.VolatilityConstraint

    class __QS_DepClass__(_RiskInputs, _BenchmarkInputs):
        pass

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.VolatilityConstraint.__QS_ArgClass__):
        pass


class ExpectedReturnConstraintNode(ConstraintNode):
    """组合预期收益约束。"""

    ComponentClass = native.ExpectedReturnConstraint

    class __QS_DepClass__(_BenchmarkInputs):
        expected_return: Factor = Field(title="预期收益")

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.ExpectedReturnConstraint.__QS_ArgClass__):
        pass


class TurnoverConstraintNode(ConstraintNode):
    """以外部权重或账户实际权重为起点的换手/成交额约束。"""

    ComponentClass = native.TurnoverConstraint

    class __QS_DepClass__(DependencyModel):
        p0: Factor | None = Field(default=None, title="外部初始权重")
        wealth: Factor | None = Field(default=None, title="外部净资产")
        amt: Factor | None = Field(default=None, title="成交金额")

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.TurnoverConstraint.__QS_ArgClass__):
        pass


class NonZeroNumConstraintNode(ConstraintNode):
    """证券数量约束，需要支持整数变量的求解器。"""

    ComponentClass = native.NonZeroNumConstraint
    __QS_DepClass__ = _BenchmarkInputs

    class __QS_ArgClass__(_ComponentNode.__QS_ArgClass__, native.NonZeroNumConstraint.__QS_ArgClass__):
        pass


class PortfolioProblem(PortfolioNode):
    """组装可独立求解或在账户决策时绑定状态的当期问题。"""

    class __QS_DepClass__(DependencyModel):
        objective: OptimizationObjectiveNode = Field(title="优化目标")
        constraints: tuple[ConstraintNode, ...] = Field(default=(), title="约束集合")
        mask: Factor = Field(default_factory=lambda: DataFactor(True), title="可选证券")

    def backward_compute(self, path, inputs, context, local_context=None):
        ids = tuple(local_context.IDs)
        result = {}
        for date in local_context.DTs:
            objective = inputs.objective[date]
            constraints = tuple(data[date] for data in inputs.constraints)
            if any(item.ids != ids for item in (objective, *constraints)):
                raise ValueError("目标、约束与问题的完整截面不同")
            values = inputs.mask.loc[date].to_numpy()
            mask = pd.notna(values) & (values != 0)
            result[date] = PeriodProblem(date, ids, mask, objective, constraints)
        return result


class PortfolioExecutionNode(PortfolioNode):
    """共享整段请求结果，保证多个投影不会重复求解或推进账户。"""

    class __QS_ArgClass__(PortfolioNode.__QS_ArgClass__):
        OptimOption: dict = Field(default_factory=dict, frozen=True, title="求解选项")
        Tolerance: float = Field(default=1e-5, gt=0, allow_inf_nan=False, title="约束残差容差")
        AcceptInaccurate: bool = Field(default=False, title="接受不精确最优解")
        RebalanceDTs: tuple[datetime, ...] | None = Field(default=None, frozen=True, title="调仓日")
        OnFailure: Literal["raise", "nan"] = Field(default="raise", title="求解失败处理")

    def compute(self, path, fwd_data, context):
        state = context.NodeState.setdefault(self.InstanceID, {})
        # 锁仅在正式计算时创建，不进入节点参数或准备阶段的进程序列化。
        lock = state.setdefault("portfolio_lock", RLock())
        with lock:
            return super().compute(path, fwd_data, context)

    def forward_compute(self, path, fwd_data, context):
        ids = self.section_ids(fwd_data, context)
        plan = section_plan(self, ids, context, ordered=self.ExecutionEffect == ExecutionEffect.STATEFUL)
        full = FactorLocalContext(DTs=plan["dates"], IDs=list(ids), SectionIDs=list(ids))
        requests, local = super().forward_compute(path, full, context)
        local.ExtraData["section_plan"] = plan
        return ([] if cached_section(self, plan, context) is not None else requests), local

    def cached_result(self, context, local_context):
        """读取本次请求的已完成结果。"""
        return cached_section(self, local_context.ExtraData["section_plan"], context)

    def save_result(self, result, context, local_context):
        """仅在完整计算成功后发布结果。"""
        return publish_section(self, local_context.ExtraData["section_plan"], result, self.merge_period_results, context)

    def merge_period_results(self, parts):
        """按日期合并独立优化的权重与诊断。"""
        weights = pd.concat([part.weights for part in parts]).sort_index()
        diagnostics = {}
        for part in parts:
            if diagnostics.keys() & part.diagnostics.keys():
                raise ValueError("并行日期分片重复，不能重复求解同一期问题")
            diagnostics.update(part.diagnostics)
        return PortfolioResult(weights, dict(sorted(diagnostics.items())))


class CVXPortfolio(PortfolioExecutionNode):
    """各日期独立求解，使用 PortfolioWeights 继续连接因子。"""

    class __QS_DepClass__(DependencyModel):
        problem: PortfolioProblem = Field(title="优化问题")

    def backward_compute(self, path, inputs, context, local_context=None):
        cached = self.cached_result(context, local_context)
        if cached is not None:
            return cached
        dates = local_context.ExtraData["section_plan"]["assigned"]
        weights = pd.DataFrame(np.nan, index=dates, columns=local_context.IDs)
        diagnostics = {}
        for date in dates:
            if self.Args.RebalanceDTs is not None and date not in self.Args.RebalanceDTs:
                diagnostics[date] = SolveResult("skipped", None)
                continue
            result = solve_period(inputs.problem[date], optim_option=self.Args.OptimOption,
                                  tolerance=self.Args.Tolerance, accept_inaccurate=self.Args.AcceptInaccurate,
                                  logger=self.Logger)
            diagnostics[date] = result
            if result.weights is None:
                if self.Args.OnFailure == "raise":
                    raise __QS_Error__(f"投资组合在 {date} 求解失败: {result.info}")
                self.Logger.warning(f"投资组合在 {date} 求解失败: {result.info}")
            else:
                weights.loc[date] = result.weights
            context.emit(self, "solve", "done", datetime=date.isoformat(), solve_status=result.status)
        return self.save_result(PortfolioResult(weights, diagnostics), context, local_context)


PORTFOLIO_COMPONENTS = [
    MeanVarianceObjectiveNode, RiskBudgetObjectiveNode, MaxDiversificationObjectiveNode,
    BudgetConstraintNode, WeightConstraintNode, FactorExposeConstraintNode, VolatilityConstraintNode,
    ExpectedReturnConstraintNode, TurnoverConstraintNode, NonZeroNumConstraintNode, PortfolioProblem, CVXPortfolio,
]
