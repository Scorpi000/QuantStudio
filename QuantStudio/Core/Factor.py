import datetime as dt
from collections import OrderedDict
from typing import List, Optional, Any

import pandas as pd

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.QSObject import Panel


class FactorContext(Context):
    # node_state: {节点ID: {"start_dt", "section_ids"}}
    dt_ruler: List[dt.datetime]

# 因子
# 因子可看做一个 DataFrame(index=[时间点], columns=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class Factor(Node):
    """因子"""
    class __QS_ArgClass__(Node.__QS_ArgClass__):
        pass

    def __init__(self, descriptors: List["Factor"] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
        return super().__init__(deps=descriptors, args=args, config_file=config_file, **kwargs)
    
    @property
    def FactorDB(self):
        return None

    @property
    def Descriptors(self):
        return self.Deps

    # 获取 ID 序列
    def getID(self, idt=None, **kwargs):
        return []

    # 获取时间点序列
    def getDateTime(self, iid=None, start_dt=None, end_dt=None, **kwargs):
        return []
    
    def readData(self, ids, dts):
        raise NotImplementedError
    
    # init_data: {"start_dt", "section_ids"}
    def init_compute(self, path: List[str], init_data: Any, context: Context) -> List[Any]:
        FactorState = context.NodeState.setdefault(self.QSID, {})
        FactorState["start_dt"] = min(init_data["start_dt"], FactorState.get("start_dt", pd.NaT))
        if "section_ids" not in FactorState:
            FactorState["section_ids"] = init_data["section_ids"]
        elif init_data["section_ids"] != FactorState["section_ids"]:
            raise __QS_Error__(f"因子 {self._QSArgs.name}({self.QSID}) 指定了不同的截面!")
        if self.QSID in path: return []
        return [init_data] * len(self.Deps)


class CompoundFactor(Factor):
    """复合因子"""
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        pass

    def __init__(self, descriptors: List["Factor"] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
        super().__init__(deps=descriptors, args=args, config_file=config_file, **kwargs)
        self._Descriptors = OrderedDict((iFactor._QSArgs.Name, iFactor) for iFactor in self.Deps)
        if len(self._Descriptors)<len(self.Deps):
            raise __QS_Error__(f"因子有重名: {[iFactor._QSArgs.Name for iFactor in self.Deps]}")
    
    @property
    def FactorNames(self):
        return list(self._Descriptors.keys())
    
    # 返回因子对象
    def getFactor(self, factor_name, args={}):
        return self._Descriptors[factor_name]

    def __getitem__(self, factor_name):
        return self.getFactor(factor_name)
    
    def readData(self, ids, dts, factor_names=None):
        if not factor_names: factor_names = self.FactorNames
        Data = {iFactor: self.getFactor(iFactor).readData(ids=ids, dts=dts) for iFactor in factor_names}
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)


class DataFactor(Factor):
    pass