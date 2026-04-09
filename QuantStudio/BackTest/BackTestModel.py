# coding=utf-8
from typing import List, Tuple, Optional, Any

from pydantic import Field

from QuantStudio.Core.Node import Context, Node, DTLocalContext, DTInitData


class BTNode(Node):
    """回测计算节点"""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="BTNode", frozen=True, title="名称")
        GenReport: bool = Field(default=False, frozen=True, title="生成报告")
    
    def genReport(self, output: dict) -> str:
        raise NotImplementedError

    def init_compute(self, path: List[str], init_data: DTInitData, context: Context) -> List[DTInitData]:
        NodeState = context.NodeState.setdefault(self.QSID, {})
        # 处理时点
        DTRange = NodeState.get("dt_range", None)
        if DTRange is None:
            NodeState["dt_range"] = init_data.DTRange
        else:
            NodeState["dt_range"] = (min(DTRange[0], init_data.DTRange[0]), max(DTRange[1], init_data.DTRange[1]))
        # 默认
        if self.QSID in path[:-1]: return []
        InitData = [DTInitData(DTRange=NodeState["dt_range"])] * len(self.Deps)
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: DTLocalContext, context: Context) -> Tuple[List[DTLocalContext], DTLocalContext]:
        return [DTLocalContext(DTs=fwd_data.DTs)] * len(self.Deps), DTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Optional[DTLocalContext]=None) -> dict:
        return {}
    
    def merge_result(self, result_list: List[dict], context: Context):
        return result_list[0]


class BTReport(Node):
    """回测报告节点"""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="BTReport", frozen=True, title="名称")
    
    def __init__(self, bt_node_list:List[BTNode], args:dict = {}, config_file:Optional[str] = None, **kwargs):
        return super().__init__(deps=bt_node_list, args=args, config_file=config_file, **kwargs)

    def init_compute(self, path: List[str], init_data: DTInitData, context: Context) -> List[DTInitData]:
        NodeState = context.NodeState.setdefault(self.QSID, {})
        # 处理时点
        DTRange = NodeState.get("dt_range", None)
        if DTRange is None:
            NodeState["dt_range"] = init_data.DTRange
        else:
            NodeState["dt_range"] = (min(DTRange[0], init_data.DTRange[0]), max(DTRange[1], init_data.DTRange[1]))
        # 默认
        if self.QSID in path[:-1]: return []
        return [DTInitData(DTRange=NodeState["dt_range"])] * len(self.Deps)
    
    def forward_compute(self, path: List[str], fwd_data: DTLocalContext, context: Context) -> Tuple[List[DTLocalContext], DTLocalContext]:
        return [DTLocalContext(DTs=fwd_data.DTs)] * len(self.Deps), DTLocalContext(DTs=fwd_data.DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[dict], context: Context, local_context: Optional[DTLocalContext]=None) -> dict:
        HTML = ''
        SepStr = '<HR style="FILTER: alpha(opacity=100,finishopacity=0,style=3)" width="90%" color=#987cb9 SIZE=5><div align="center" style="font-size:1.17em"><strong>{Module}</strong></div>'
        Output = {}
        for i, iOutput in enumerate(bwd_data_list):
            Output[str(i)+"-"+self.Deps[i].Name] = iOutput
            if "Report" in iOutput:
                iHTML = iOutput["Report"]
            else:
                iHTML = self.Deps[i].genReport(output=iOutput)
            HTML += SepStr.format(Module=str(i)+". "+self.Deps[i].Name) + iHTML
        Output["Report"] = HTML
        return Output
    
    @staticmethod
    def genOutputReport(output_list:List[dict], name_list:Optional[List[str]]=None) -> str:
        if not name_list: name_list = [""] * len(output_list)
        HTML = ""
        SepStr = '<HR style="FILTER: alpha(opacity=100,finishopacity=0,style=3)" width="90%" color=#987cb9 SIZE=5><div align="center" style="font-size:1.17em"><strong>{Module}</strong></div>'
        for i, iOutput in enumerate(output_list):
            if "Report" in iOutput:
                iHTML = iOutput["Report"]
            else:
                iHTML = "暂无报告"
            HTML += SepStr.format(Module=str(i)+". "+name_list[i]) + "\n" + iHTML + "\n"
        return HTML

    def merge_result(self, result_list: List[dict], context: Context):
        return result_list[0]