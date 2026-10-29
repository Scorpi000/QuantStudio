# -*- coding: utf-8 -*-
"""将共享优化结果投影为可继续计算的因子或诊断节点。"""
from copy import deepcopy
from QuantStudio.Core.Input import DependencyModel
from QuantStudio.Factor.Factor import Factor
from pydantic import Field

from .Nodes import PortfolioExecutionNode, PortfolioNode


class PortfolioWeights(Factor):
    """标准权重因子，始终请求来源的完整证券截面。"""

    ResultField = "weights"

    class __QS_DepClass__(DependencyModel):
        source: PortfolioExecutionNode = Field(title="组合结果")

    def init_compute(self, path, init_data, context):
        source = self.Dependencies.source
        request = init_data.model_copy(update={"SectionIDs": list(source.section_ids(init_data, context))})
        return super().init_compute(path, request, context)

    def forward_compute(self, path, fwd_data, context):
        source = self.Dependencies.source
        ids = list(source.section_ids(fwd_data, context))
        return [fwd_data.model_copy(update={"IDs": ids, "SectionIDs": ids})], fwd_data

    def backward_compute(self, path, inputs, context, local_context=None):
        data = getattr(inputs.source, self.ResultField)
        return data.reindex(index=local_context.DTs, columns=local_context.IDs).copy(deep=True)


class PortfolioDiagnostics(PortfolioNode):
    """求解诊断，与权重出口复用同一次求解。"""

    class __QS_DepClass__(DependencyModel):
        source: PortfolioExecutionNode = Field(title="组合结果")

    def backward_compute(self, path, inputs, context, local_context=None):
        return deepcopy({date: value for date, value in inputs.source.diagnostics.items() if date in local_context.DTs})
