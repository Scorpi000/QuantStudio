# -*- coding: utf-8 -*-
"""根据价格因子的历史收益率与当期可用资产掩码估计滚动协方差。"""
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Identity import ExecutionEffect
from QuantStudio.Core.Input import DependencyModel
from QuantStudio.Core.Node import Node
from QuantStudio.Core.SectionExecution import init_section, section_plan, cached_section, publish_section
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import DataFactor, Factor, FactorContext, FactorInitData, FactorLocalContext
from .RiskTable import RiskTable


class RollingCovarianceRiskTable(RiskTable):
    """基于简单收益率样本协方差的派生风险表。

    每个输出时点使用截至当期的最近 Window 期收益率，需要 Window + 1 个
    价格时点。mask 可选，未提供时全部资产可用；提供后仅当期 mask 等于 1 的
    资产参与估计，历史 mask 不影响价格样本。
    参与资产使用共同有效的收益率样本，样本数小于 MinPeriods 时整个有效资产
    子矩阵为 NaN。非正、缺失或非有限价格不填充，其他资产的行列始终为 NaN。
    输出未年化，样本协方差的自由度为 1。并行时按输出日期分工，始终使用完整截面。
    """

    __QS_ExecutionEffect__ = ExecutionEffect.PURE

    class __QS_DepClass__(DependencyModel):
        price: Factor = Field(title="资产价格")
        mask: Factor = Field(default_factory=lambda: DataFactor(True), title="可用资产掩码",
                             description="可选，不连接时默认全部资产可用")

    class __QS_ArgClass__(RiskTable.__QS_ArgClass__):
        Name: str = Field(default="RollingCovarianceRiskTable", frozen=True, title="名称")
        SourceRDB: Literal[None] = Field(default=None, frozen=True, exclude=True, repr=False)
        Window: int = Field(default=60, ge=2, frozen=True, title="收益率窗口期数")
        MinPeriods: Annotated[int, Field(ge=2)] | None = Field(default=None, frozen=True, title="最少有效样本数",
                                                            description="None 表示需要完整 Window 期共同有效收益率")
        SectionIDs: tuple[str, ...] | None = Field(default=None, frozen=True, title="完整证券截面")

        @model_validator(mode="after")
        def _validate_window(self):
            if self.MinPeriods is not None and self.MinPeriods > self.Window:
                raise ValueError("最少有效样本数不能超过收益率窗口期数")
            if self.SectionIDs is not None and (not self.SectionIDs or len(set(self.SectionIDs)) != len(self.SectionIDs)):
                raise ValueError("完整证券截面必须非空且不能重复")
            return self

    def __init__(self, price: Factor, mask: Factor | None = None, args=None, config_file=None, **kwargs):
        """初始化价格因子驱动的风险表，可选用 mask 限制当期资产。

        Args:
            price: 日期 × 证券的价格因子，建议使用复权价格。
            mask: 可选的日期 × 证券掩码因子；None 表示全部资产可用，提供后仅
                数值 1 或 True 表示当期可用。
            args: 窗口、最少样本数和完整证券截面等对象参数。
            config_file: 对象参数配置文件。
            **kwargs: 日志等通用对象选项。
        """
        if "rdb" in kwargs:
            raise TypeError("滚动协方差风险表依赖因子，不接受风险数据库")
        self._QS_PrepareIgnoredArgs = ()
        deps = {"price": price}
        if mask is not None:
            deps["mask"] = mask
        Node.__init__(self, deps=deps, args=args or {}, config_file=config_file, **kwargs)

    @classmethod
    def from_dependencies(cls, deps, args=None, **kwargs):
        """从具名依赖模型构造，支持模板、克隆和序列化还原。"""
        dependencies = cls.__QS_DepClass__.model_validate(deps)
        values = args.to_dict(repr=False) if hasattr(args, "to_dict") else args
        return cls(dependencies.price, dependencies.mask, args=values, **kwargs)

    def getID(self, idt=None):
        """返回完整证券截面，保留当期不可用但需要输出 NaN 的资产。"""
        return list(self.Args.SectionIDs or self.Dependencies.price.getID(idt=idt) or self.Dependencies.mask.getID(idt=idt))

    def getDateTime(self, start_dt=None, end_dt=None):
        """从价格因子的固有日期中取得可请求时点，不读取价格数据。"""
        return [date for date in self.Dependencies.price.getDateTime()
                if (start_dt is None or date >= start_dt) and (end_dt is None or date <= end_dt)]

    def _section_ids(self, data, context):
        """确定完整估计截面，与输出请求的证券子集分开。"""
        return list(self.Args.SectionIDs or data.SectionIDs or context.SectionIDs)

    def _price_dates(self, start, end, context):
        """在运行标尺内回溯 Window 个时点，历史不足时截到标尺起点。"""
        start_index = np.searchsorted(context.DTRuler, start, side="left")
        end_index = np.searchsorted(context.DTRuler, end, side="right")
        return context.DTRuler[max(0, start_index - self.Args.Window):end_index]

    def init_compute(self, path, init_data, context):
        """为价格依赖扩展历史请求，mask 仅请求实际输出区间。"""
        requests = Node.init_compute(self, path, init_data, context)
        if not requests:
            return []
        ruler = pd.Index(context.DTRuler)
        if ruler.empty or not ruler.is_unique or not ruler.is_monotonic_increasing:
            raise ValueError("计算时点标尺必须非空、递增且不能重复")
        ids = self._section_ids(init_data, context)
        if len(set(ids)) != len(ids):
            raise ValueError("完整证券截面不能重复")
        init_section(self, init_data, ids, context)
        dates = self._price_dates(*init_data.DTRange, context)
        if not dates:
            raise ValueError("请求区间不在计算时点标尺内")
        return [FactorInitData(DTRange=(dates[0], init_data.DTRange[1]), SectionIDs=ids),
                FactorInitData(DTRange=init_data.DTRange, SectionIDs=ids)]

    def forward_compute(self, path, fwd_data, context):
        """读取连续历史标尺上的价格，即使输出日期是稀疏调仓日。"""
        dates, ids = list(fwd_data.DTs), self._section_ids(fwd_data, context)
        if len(set(dates)) != len(dates):
            raise ValueError("请求时点不能重复")
        if not pd.Index(dates).isin(context.DTRuler).all():
            raise ValueError("请求时点必须位于计算时点标尺内")
        plan = section_plan(self, ids, context)
        history = self._price_dates(plan["dates"][0], plan["dates"][-1], context)
        requests = [FactorLocalContext(DTs=history, IDs=ids, SectionIDs=ids, PIDs=context.PIDList),
                    FactorLocalContext(DTs=plan["dates"], IDs=ids, SectionIDs=ids, PIDs=context.PIDList)]
        local = fwd_data.model_copy(update={"ExtraData": {**fwd_data.ExtraData, "risk_dates": history,
                                                       "risk_ids": ids, "section_plan": plan}})
        return ([] if cached_section(self, plan, context) is not None else requests), local

    def backward_compute(self, path, inputs, context, local_context=None):
        """估计当期有效资产的协方差，并保留请求顺序与无效资产的 NaN。"""
        plan = local_context.ExtraData["section_plan"]
        cached = cached_section(self, plan, context)
        if cached is not None:
            return self._project(cached, local_context)
        history, ids = local_context.ExtraData["risk_dates"], local_context.ExtraData["risk_ids"]
        prices = inputs.price.reindex(index=history, columns=ids).astype(float)
        prices = prices.where(np.isfinite(prices) & (prices > 0))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            returns = prices.div(prices.shift(1)).sub(1.)
        returns = returns.where(np.isfinite(returns))
        mask = inputs.mask.reindex(index=plan["dates"], columns=ids).eq(1).fillna(False)
        matrices = {}
        minimum = self.Args.MinPeriods if self.Args.MinPeriods is not None else self.Args.Window
        for date in plan["assigned"]:
            covariance = pd.DataFrame(np.nan, index=ids, columns=ids)
            active = mask.columns[mask.loc[date]].tolist()
            if active:
                end = returns.index.get_loc(date) + 1
                sample = returns.iloc[max(0, end - self.Args.Window):end][active].dropna(how="any")
                if len(sample) >= minimum:
                    covariance.loc[active, active] = sample.cov(ddof=1)
            matrices[date] = covariance
        part = Panel(matrices, items=plan["assigned"], major_axis=ids, minor_axis=ids)

        def merge(parts):
            matrices = {date: value.loc[date] for value in parts for date in value.items}
            return Panel(matrices, items=plan["dates"], major_axis=ids, minor_axis=ids)

        full = publish_section(self, plan, part, merge, context)
        return self._project(full, local_context)

    @staticmethod
    def _project(data, request):
        """在完整协方差交换完成后，投影请求的日期和证券。"""
        return Panel({date: data.loc[date].reindex(index=request.IDs, columns=request.IDs) for date in request.DTs},
                     items=request.DTs, major_axis=request.IDs, minor_axis=request.IDs)

    def readCov(self, dts, ids=None, *, section_ids=None, dt_ruler=None):
        """通过顺序引擎读取滚动协方差。

        Args:
            dts: 请求的输出日期，可以是稀疏调仓日。
            ids: 输出证券顺序，None 使用完整证券截面。
            section_ids: 完整估计截面，默认使用对象配置或价格因子的证券。
            dt_ruler: 包含预热历史的完整计算标尺；默认尝试价格因子的固有日期。

        Returns:
            日期 × 证券 × 证券的 Panel，未启用资产的行列和样本不足的矩阵为 NaN。
        """
        dates = list(dts)
        section = list(self.Args.SectionIDs or section_ids or self.getID() or ids or [])
        ids = section if ids is None else list(ids)
        if not dates:
            return Panel(items=[], major_axis=ids, minor_axis=ids)
        ruler = list(dt_ruler) if dt_ruler is not None else self.getDateTime()
        ruler = ruler or sorted(set(dates))
        with FactorContext(DTRuler=ruler, SectionIDs=section) as context, Engine() as engine:
            return engine.run([self], context, init_data_list=[FactorInitData(DTRange=(min(dates), max(dates)), SectionIDs=section)],
                              fwd_data_list=[FactorLocalContext(DTs=dates, IDs=ids, SectionIDs=section)])[0]
