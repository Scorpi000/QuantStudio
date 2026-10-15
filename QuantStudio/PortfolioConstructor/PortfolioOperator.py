# -*- coding: utf-8 -*-
"""通过截面算子生成投资组合目标权重因子。"""
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field, create_model

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Input import DependencyModel
from QuantStudio.Factor.Factor import DataFactor, Factor, FactorLocalContext, makeFactorRunningKey
from QuantStudio.Factor.FactorOperation import (
    FactorInputModel, OperatorModelArgs, SectionOperator, SectionOperation,
)
from QuantStudio.Risk.RiskTable import RiskTable
from .BasePC import MeanVarianceObjective, BudgetConstraint, WeightConstraint
from .OptimizationModel import OptimizationModel, ObjectiveSpec, ConstraintSpec, resolve_sources


class MeanVariancePortfolioArgs(OperatorModelArgs):
    """逐期独立优化的模型参数；不包含实际持仓或成交状态。"""

    cov: Optional[pd.DataFrame] = Field(default=None, title="固定协方差矩阵",
                                       description="行列均为证券 ID；动态风险数据由 risk_table 依赖提供")
    expected_return_coef: float = Field(default=0.0, allow_inf_nan=False, title="收益项系数")
    risk_aversion_coef: float = Field(default=1.0, ge=0, allow_inf_nan=False, title="风险厌恶系数")
    benchmark: bool = Field(default=False, title="相对基准")
    budget: float = Field(default=1.0, allow_inf_nan=False, title="总权重")
    optim_option: dict = Field(default_factory=dict, title="求解器选项")
    on_failure: Literal["raise", "nan"] = Field(default="raise", title="求解失败处理",
                                              description="raise 抛出异常；nan 记录警告并返回整期缺失值")


class OptimizedPortfolio(SectionOperation):
    """目标权重因子，风险依赖始终取得完整优化截面的协方差。"""

    def init_compute(self, path, init_data, context):
        requests = list(super().init_compute(path, init_data, context))
        if requests and self._ExtraDeps:
            section_ids = init_data.SectionIDs or self.Args.SectionIDs or context.SectionIDs
            for index in range(len(self._Descriptors), len(requests)):
                requests[index] = requests[index].model_copy(update={"SectionIDs": section_ids})
        return requests

    def forward_compute(self, path, fwd_data, context):
        # 截面和数据版本共同决定缓存范围；不能只使用因子的 QSID。
        running_key = makeFactorRunningKey(self.QSID, fwd_data.SectionIDs or self.Args.SectionIDs,
                                          context=context)
        dt_range = context.NodeState.get(self.InstanceID, {}).get(running_key, {}).get("dt_range")
        if context.DataCache and self.Args.CacheEnabled and dt_range is not None:
            if context.DataCache.getDTRange(running_key, dt_range) is None:
                return [], fwd_data
        requests, local_context = super().forward_compute(path, fwd_data, context)
        requests = list(requests)
        if requests and self._ExtraDeps:
            section_ids = fwd_data.SectionIDs or self.Args.SectionIDs or context.SectionIDs
            model = getattr(self.Operator.Args.ModelArgs, "model", None)
            fields = model.risk_fields(len(self._ExtraDeps)) if model is not None else ((),) * len(self._ExtraDeps)
            for index, requested_fields in enumerate(fields, start=len(self._Descriptors)):
                requests[index] = FactorLocalContext(
                    DTs=local_context.ExtraData["TotalCalcDTs"], IDs=section_ids,
                    SectionIDs=section_ids, PIDs=context.PIDList,
                    ExtraData={"Fields": list(requested_fields)},
                )
        return requests, local_context


class MeanVariancePortfolio(OptimizedPortfolio):
    """保留均值方差便捷接口产生的权重因子类型。"""


class OptimizedPortfolioArgs(OperatorModelArgs):
    model: OptimizationModel = Field(title="优化模型")
    optim_option: dict = Field(default_factory=dict, title="求解器选项")
    on_failure: Literal["raise", "nan"] = Field(default="raise", title="求解失败处理")


class _PortfolioDependencies(DependencyModel):
    mask: Factor
    factors: tuple[Factor, ...] = ()
    extra_deps: tuple[RiskTable, ...] = ()


class MakeOptimizedPortfolio(SectionOperator):
    """从可组合的 OptimizationModel 生成权重因子。

    模型中的 Factor 和 RiskTable 自动提取为显式图依赖。标量、Series、DataFrame
    和 numpy 数组可作为固定输入；有标签的数据按截面对齐，数组按截面顺序解释。
    """

    class __QS_InputClass__(FactorInputModel):
        mask: Factor = Field(default_factory=lambda: DataFactor(True), title="可选证券")
        factors: tuple[Factor, ...] = Field(default=(), title="模型因子")
        risk_tables: tuple[RiskTable, ...] = Field(default=(), title="模型风险表")

    __QS_CallLayout__ = {name: {"keyword_only": True, "grouped": True}
                         for name in ("factors", "risk_tables")}

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: OptimizedPortfolioArgs = Field(frozen=True, title="模型参数")
        DTMode: Literal["单时点"] = Field(default="单时点", frozen=True, title="运算时点")
        DataType: Literal["double"] = Field(default="double", frozen=True, title="数据类型")

    def __init__(self, model: OptimizationModel, optim_option=None, on_failure="raise",
                 args=None, config_file=None, **kwargs):
        compiled, factors, risks = model.compile()
        args = dict(args or {})
        options = {"model": compiled, "optim_option": optim_option or {}, "on_failure": on_failure}
        model_args = dict(args.pop("ModelArgs", {}))
        if "model" in model_args:
            raise ValueError("请通过 MakeOptimizedPortfolio(model=...) 指定模型")
        options.update(model_args)
        super().__init__(args={"Name": "makeOptimizedPortfolio", **args, "ModelArgs": options},
                         config_file=config_file, **kwargs)
        self._InputClass = create_model(
            "PortfolioInputs", __base__=self.__QS_InputClass__,
            factors=(tuple[Factor, ...], Field(default_factory=lambda: factors, min_length=len(factors), max_length=len(factors) or None)),
            risk_tables=(tuple[RiskTable, ...], Field(default_factory=lambda: risks, min_length=len(risks), max_length=len(risks) or None)),
        )
        self._InputClass.__QS_DynamicInputs__ = True

    def _QS_bindInputs(self, operands=(), named=None, initial=None):
        # 风险表是节点集合，不应被因子基类的集合常量归一化包装成 DataFactor。
        named = dict(named or {})
        risks = named.pop("risk_tables", None)
        if risks is not None and initial and "risk_tables" in initial:
            raise __QS_Error__("已绑定操作数 risk_tables 不能在调用中覆盖")
        binding = super()._QS_bindInputs(operands, named, initial)
        if risks is not None:
            binding = binding.bind(risk_tables=tuple(risks))
        return binding

    def _QS_updateArgs(self, args, updates, operand_count=None):
        if "model" in dict(updates.get("ModelArgs", {})):
            raise ValueError("更换模型请重新创建 MakeOptimizedPortfolio，以同步图输入绑定")
        return super()._QS_updateArgs(args, updates, operand_count)

    def _QS_validate(self, *descriptors, **kwargs):
        # 风险输入不属于数值描述子，DescriptorSection 只配置 mask 与模型因子。
        operator = self if getattr(self, "_OperandCount", None) == len(descriptors) else self.new(_qs_operand_count=len(descriptors))
        if len(operator.Args.DescriptorSection) != len(descriptors) or any(
                section is not None for section in operator.Args.DescriptorSection):
            raise __QS_Error__("优化模型的全部因子必须使用同一个完整证券截面")
        return operator

    def __call__(self, mask: Factor | bool = True, *, factors=(), risk_tables=(), factor_args=None, **kwargs):
        """构建权重节点；模型数据默认来自模型绑定，也可通过 bind 替换同形输入组。"""
        for name, values in (("factors", factors), ("risk_tables", risk_tables)):
            if len(values) != len(self.InputClass.model_fields[name].default_factory()):
                raise ValueError(f"{name} 输入组长度必须与模型定义相同")
        operator = self._QS_validate(mask, *factors)
        deps = _PortfolioDependencies(mask=mask, factors=tuple(factors), extra_deps=tuple(risk_tables))
        return OptimizedPortfolio.from_dependencies(
            deps, args={"Name": operator.Name, **(factor_args or {}), "Operator": operator}, **kwargs)

    def calculate(self, f, idt, iid, x, args):
        count = len(f._Descriptors) - 1
        resolve = lambda value: resolve_sources(value, x[1:count + 1], x[count + 1:], idt)
        return _solve_model(args.model, idt, iid, x[0], resolve, args.optim_option, args.on_failure, self.Logger)


def _solve_model(model, idt, ids, mask_data, resolve, optim_option, on_failure, logger):
    """通用与便捷算子共用的当期求解、空股票池和失败处理。"""
    def failure(message):
        message = f"投资组合在 {idt} 求解失败: {message}"
        if on_failure == "raise":
            raise __QS_Error__(message)
        logger.warning(message)
        return np.full(len(ids), np.nan)

    mask = pd.notna(mask_data) & (mask_data != 0)
    if not np.any(mask):
        budgets = [spec for spec in model.constraints if issubclass(spec.component, BudgetConstraint)]
        if budgets and all(spec.args.UpLimit == spec.args.DownLimit == 0 for spec in budgets) and all(
                issubclass(spec.component, (BudgetConstraint, WeightConstraint)) for spec in model.constraints):
            return np.zeros(len(ids))
        return failure("可选证券为空")
    try:
        weights, info = model.solve(mask, ids, resolve, optim_option, logger)
    except (ValueError, TypeError, __QS_Error__) as error:
        return failure(error)
    if weights is None or info.get("status") != 1 or not np.isfinite(weights[mask]).all():
        return failure(info)
    if info.get("ReleasedConstraint"):
        logger.warning(f"投资组合在 {idt} 释放约束: {info['ReleasedConstraint']}")
    return np.where(mask, weights, 0.)


class MakeMeanVariancePortfolio(SectionOperator):
    """均值方差优化算子，生成可计算、缓存及回测的目标权重因子。

    每期最大化 expected_return_coef * mu'w - risk_aversion_coef / 2 * w'Cov*w，
    benchmark=True 时以 w-bmk 计算收益和风险；预算约束始终作用于绝对权重。
    cov 为带证券标签的固定协方差矩阵，或调用时通过 risk_table 提供动态协方差，
    两者只能指定一个。各期独立求解，不自动使用上一期持仓。
    """

    class __QS_InputClass__(FactorInputModel):
        expected_return: Factor = Field(default_factory=lambda: DataFactor(0.0), title="预期收益")
        mask: Factor = Field(default_factory=lambda: DataFactor(True), title="可选证券")
        bmk: Factor = Field(default_factory=lambda: DataFactor(0.0), title="基准权重")
        up_limit: Factor = Field(default_factory=lambda: DataFactor(1.0), title="单证券权重上限")
        down_limit: Factor = Field(default_factory=lambda: DataFactor(0.0), title="单证券权重下限")

    __QS_CallLayout__ = {name: {"keyword_only": True}
                         for name in ("mask", "bmk", "up_limit", "down_limit")}

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: MeanVariancePortfolioArgs = Field(default_factory=MeanVariancePortfolioArgs,
                                                     frozen=True, title="模型参数")
        DTMode: Literal["单时点"] = Field(default="单时点", frozen=True, title="运算时点")
        DataType: Literal["double"] = Field(default="double", frozen=True, title="数据类型")

    def __init__(self, cov=None, expected_return_coef=0.0, risk_aversion_coef=1.0,
                 benchmark=False, budget=1.0, optim_option=None, on_failure="raise",
                 args=None, config_file=None, **kwargs):
        model_args = dict(cov=cov, expected_return_coef=expected_return_coef,
                          risk_aversion_coef=risk_aversion_coef, benchmark=benchmark,
                          budget=budget, optim_option=optim_option or {}, on_failure=on_failure)
        args = dict(args or {})
        model_args.update(dict(args.pop("ModelArgs", {})))
        super().__init__(args={"Name": "makeMeanVariancePortfolio", **args,
                               "ModelArgs": model_args}, config_file=config_file, **kwargs)

    def __call__(self, expected_return: Factor | float = 0.0, *,
                 mask: Factor | bool = True, bmk: Factor | float = 0.0,
                 up_limit: Factor | float = 1.0, down_limit: Factor | float = 0.0,
                 risk_table: Optional[RiskTable] = None, factor_args=None, **kwargs):
        """绑定当期输入并返回目标权重因子。

        Args:
            expected_return: 预期收益因子或常量，默认零收益对应最小方差组合。
            mask: 可选证券因子；缺失或零表示不可选，成功求解后这些证券权重为零。
            bmk: 基准权重因子，benchmark=True 时使用。
            up_limit: 单证券权重上限因子或常量。
            down_limit: 单证券权重下限因子或常量。
            risk_table: 图中的风险表节点，输出按日期和证券标记的协方差 Panel。
            factor_args: 因子参数，如 Name、SectionIDs、CalcDTRuler、CacheEnabled。

        Returns:
            MeanVariancePortfolio；非调仓日为 NaN。证券查询子集不会改变优化截面，
            优化截面由 section_ids、因子 SectionIDs 或 FactorContext.SectionIDs 指定。
        """
        if risk_table is not None and not isinstance(risk_table, RiskTable):
            raise TypeError("risk_table 必须是 RiskTable 节点")
        model = self.Args.ModelArgs
        if risk_table is not None and model.cov is not None:
            raise __QS_Error__("cov 和 risk_table 只能指定一个")
        if model.risk_aversion_coef and risk_table is None and model.cov is None:
            raise __QS_Error__("均值方差优化需要 cov 或 risk_table")
        descriptors = [expected_return, mask, bmk, up_limit, down_limit]
        operator = self._QS_validate(*descriptors)
        return MeanVariancePortfolio(
            descriptors=descriptors, extra_deps=[] if risk_table is None else [risk_table],
            args={"Operator": operator, **(factor_args or {})}, **kwargs,
        )

    def calculate(self, f, idt, iid, x, args):
        cov = x[5].loc[idt] if args.risk_aversion_coef and len(x) > 5 else args.cov
        model = OptimizationModel(
            ObjectiveSpec(MeanVarianceObjective,
                          args={"Benchmark": args.benchmark, "ExpectedReturnCoef": args.expected_return_coef,
                                "RiskAversionCoef": args.risk_aversion_coef},
                          inputs={"expected_return": x[0], "bmk": x[2] if args.benchmark else None,
                                  "cov": cov if args.risk_aversion_coef else None}),
            [ConstraintSpec(BudgetConstraint, args={"UpLimit": args.budget, "DownLimit": args.budget}),
             ConstraintSpec(WeightConstraint, inputs={"up_limit": x[3], "down_limit": x[4]})],
        )
        return _solve_model(model, idt, iid, x[1], lambda value: value,
                            args.optim_option, args.on_failure, self.Logger)
