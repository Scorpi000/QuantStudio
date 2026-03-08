# coding=utf-8
import datetime as dt
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import QSArgs
from QuantStudio.Core.Node import Node, LocalContext
from QuantStudio.Factor.Factor import FactorContext, FactorLocalContext, FactorInitData


class BTLocalContext(LocalContext):
    DTs: List[dt.datetime]


class BTInitData(QSArgs):
    DTRange: Tuple[dt.datetime, dt.datetime]


class BTNode(Node):
    """回测计算节点"""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="BTNode", frozen=True, title="名称")
        GenReport: bool = Field(default=False, frozen=True, title="生成报告")
    
    def genReport(self, output: dict) -> str:
        raise NotImplementedError

    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[FactorInitData]:
        NodeState = context.NodeState.setdefault(self.QSID, {})
        # 处理时点
        DTRange = NodeState.get("dt_range", None)
        if DTRange is None:
            NodeState["dt_range"] = init_data.DTRange
        else:
            NodeState["dt_range"] = (min(DTRange[0], init_data.DTRange[0]), max(DTRange[1], init_data.DTRange[1]))
        # 默认
        if self.QSID in path: return []
        InitData = [FactorInitData(DTRange=NodeState["dt_range"], SectionIDs=iDep._QSArgs.SectionIDs) for iDep in self.Deps]
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: BTLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], BTLocalContext]:
        return [FactorLocalContext(IDs=context.getID(iDep.QSID, pids=None), DTs=fwd_data.DTs, PIDs=context.PIDList) for iDep in self.Deps], BTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> dict:
        return {}
    
    def merge_result(self, result_list: List[dict], context: FactorContext):
        return result_list[0]
