# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import QSArgs
from QuantStudio.Core.Node import Node, LocalContext
from QuantStudio.Core.Factor import Factor, FactorContext, FactorLocalContext, FactorInitData


class BTLocalContext(LocalContext):
    DTs: List[dt.datetime]


class BTInitData(QSArgs):
    DTRange: Tuple[dt.datetime, dt.datetime]


class BTOutputNode(Node):
    """BTOutput"""
    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="BTOutput", frozen=True, title="名称")
    
    def __init__(self, deps:List[Factor]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)

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


class BTReportNode(Node):
    """BTReport"""
    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="BTReport", frozen=True, title="名称")    
    
    def __init__(self, output_node: BTOutputNode, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=[output_node], args=args, config_file=config_file, **kwargs)
    
    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[BTInitData]:
        NodeState = context.NodeState.setdefault(self.QSID, {})
        # 处理时点
        DTRange = NodeState.get("dt_range", None)
        if DTRange is None:
            NodeState["dt_range"] = init_data.DTRange
        else:
            NodeState["dt_range"] = (min(DTRange[0], init_data.DTRange[0]), max(DTRange[1], init_data.DTRange[1]))
        # 默认
        if self.QSID in path: return []
        InitData = [BTInitData(DTRange=NodeState["dt_range"]) for iDep in self.Deps]
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: BTLocalContext, context: FactorContext) -> Tuple[List[BTLocalContext], BTLocalContext]:
        return [BTLocalContext(DTs=fwd_data.DTs) for iDep in self.Deps], BTLocalContext(DTs=fwd_data.DTs)    
    
    def backward_compute(self, path: List[str], bwd_data_list: List[dict], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> str:
        return ""

