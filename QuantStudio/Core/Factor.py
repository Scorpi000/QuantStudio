# -*- coding: utf-8 -*-
import datetime as dt
from collections import OrderedDict
from typing import List, Optional, Any, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import Field, BaseModel

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback


class FactorContext(Context):
    # node_state: {节点ID: {"start_dt", "section_ids"}}
    dt_ruler: List[dt.datetime]

class FactorLocalContext(BaseModel):
    dts: List[dt.datetime]
    ids: List[str]

# 因子
# 因子可看做一个 DataFrame(index=[时间点], columns=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
class Factor(Node):
    """因子"""
    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="Factor", frozen=True, title="名称")

    def __init__(self, descriptors: List["Factor"] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
        return super().__init__(deps=descriptors, args=args, config_file=config_file, **kwargs)
    
    @property
    def FactorDB(self):
        return None

    @property
    def Descriptors(self):
        return self.Deps

    def getMetaData(self, key=None):
        if not key: return {}
        else: return None

    # 获取 ID 序列
    def getID(self, idt=None, **kwargs):
        return []

    # 获取时间点序列
    def getDateTime(self, iid=None, start_dt=None, end_dt=None, **kwargs):
        return []
    
    def readData(self, ids, dts, **kwargs):
        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, tuple): key += (slice(None),) * (2 - len(key))
        else: key = (key, slice(None))
        if len(key)>2: raise IndexError("QuantStudio.Core.Factor: Too many indexers")
        DTs, IDs = key
        if DTs==slice(None): DTs = None
        elif isinstance(DTs, dt.datetime): DTs = [DTs]
        if IDs==slice(None): IDs = None
        elif isinstance(IDs, str): IDs = [IDs]
        Data = self.readData(IDs, DTs)
        return Data.loc[key]
    
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

    def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
        return [fwd_data] * len(self.Deps), fwd_data

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Any=None) -> Any:
        pass


# 复合因子
# 复合因子可看做一个 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
class CompoundFactor(Factor):
    """复合因子"""

    def __init__(self, descriptors: List["Factor"] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
        super().__init__(descriptors=descriptors, args=args, config_file=config_file, **kwargs)
        self._Descriptors = OrderedDict((iFactor._QSArgs.Name, iFactor) for iFactor in self.Deps)
        if len(self._Descriptors)<len(self.Deps):
            raise __QS_Error__(f"因子有重名: {[iFactor._QSArgs.Name for iFactor in self.Deps]}")
    
    @property
    def FactorNames(self):
        return list(self._Descriptors.keys())
    
    # 返回因子对象
    def getFactor(self, factor_name_or_list: str | list[str], args={}):
        if isinstance(factor_name_or_list, str):
            if not args:
                return self._Descriptors[factor_name_or_list]
            else:
                return self._Descriptors[factor_name_or_list].new(args=args)
        else:
            Descriptors = [self._Descriptors[iFactorName] for iFactorName in factor_name_or_list]
            Args = self._QSArgs.to_dict(repr=False) | args
            return CompoundFactor(descriptors=Descriptors, args=Args)
    
    def getFactorMetaData(self, factor_name, key=None):
        return self._Descriptors[factor_name].getMetaData(key=key)
    
    # 获取 ID 序列
    def getID(self, idt=None, factor_name=None, **kwargs):
        if not factor_name: factor_name = self.FactorNames[0]
        return self.getFactor(factor_name_or_list=factor_name).getID(idt=idt, **kwargs)

    # 获取时间点序列
    def getDateTime(self, iid=None, start_dt=None, end_dt=None, factor_name=None, **kwargs):
        if not factor_name: factor_name = self.FactorNames[0]
        return self.getFactor(factor_name_or_list=factor_name).getDatetime(iid=iid, start_dt=start_dt, end_dt=end_dt, **kwargs)
    
    def readData(self, ids, dts, factor_names=None, **kwargs):
        if not factor_names: factor_names = self.FactorNames
        Data = {iFactor: self.getFactor(factor_name_or_list=iFactor).readData(ids=ids, dts=dts) for iFactor in factor_names}
        return Panel(Data, items=factor_names, major_axis=dts, minor_axis=ids)

    def __getitem__(self, key):
        if isinstance(key, tuple): key += (slice(None),) * (2 - len(key))
        else: key = (key, slice(None))
        if len(key)>2: raise IndexError("QuantStudio.Core.Factor: Too many indexers")
        DTs, IDs = key
        if DTs==slice(None): DTs = None
        elif isinstance(DTs, dt.datetime): DTs = [DTs]
        if IDs==slice(None): IDs = None
        elif isinstance(IDs, str): IDs = [IDs]
        Data = self.readData(IDs, DTs)
        return Data.loc[(slice(None), ) + key]

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        return Panel(bwd_data_list, items=self.FactorNames, major_axis=local_context.dts, minor_axis=local_context.ids)


class CompoundExtractedFactor(Factor):
    def __init__(self, compound_factor: CompoundFactor, args: dict = {}, config_file: Optional[str] = None, **kwargs):
        super().__init__(descriptors=[compound_factor], args=args, config_file=config_file, **kwargs)
        self._CompoundFactor = compound_factor
        if self._QSArgs.Name not in self._CompoundFactor.FactorNames:
            raise __QS_Error__(f"复合因子{compound_factor._QSArgs}中不存在因子{self._QSArgs.Name}")
    
    def getID(self, idt=None, **kwargs):
        return self._CompoundFactor.getID(idt=idt, factor_name=self._QSArgs.Name, **kwargs)
    
    def getDateTime(self, iid=None, start_dt=None, end_dt=None, **kwargs):
        return self._CompoundFactor.getDateTime(iid=iid, start_dt=start_dt, end_dt=end_dt, factor_name=self._QSArgs.Name, **kwargs)
    
    def readData(self, ids, dts, **kwargs):
        return self._CompoundFactor.readData(ids=ids, dts=dts, factor_names=[self._QSArgs.Name], **kwargs).loc[self._QSArgs.Name]
    
    def getMetaData(self, key=None):
        return self._CompoundFactor.getFactorMetaData(factor_name=self._QSArgs.Name, key=key)


# 直接赋予数据产生的因子
# data: DataFrame(index=[时点], columns=[ID])
class DataFactor(Factor):
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Name: str = Field(default="DataFactor", frozen=True, title="名称")
        DataType: Literal["double", "string", "object"] = Field(default="double", frozen=True, title="数据类型")
        LookBack: int = Field(default=0, title="回溯天数", frozen=True)

    def __init__(self, data, args: dict={}, config_file: Optional[str] = None, **kwargs):
        if isinstance(data, pd.Series):
            if data.index.is_all_dates:
                self._DataContent = "DateTime"
            else:
                self._DataContent = "ID"
            if "DataType" not in args:
                try:
                    data = data.astype(float)
                except:
                    args["DataType"] = "object"
                else:
                    args["DataType"] = "double"
        elif isinstance(data, pd.DataFrame):
            self._DataContent = "Factor"
            if "DataType" not in args:
                try:
                    data = data.astype(float)
                except:
                    args["DataType"] = "object"
                else:
                    args["DataType"] = "double"
        else:
            self._DataContent = "Value"
            if "DataType" not in args:
                if isinstance(data, str):
                    args["DataType"] = "string"
                else:
                    try:
                        data = float(data)
                    except:
                        args["DataType"] = "object"
                    else:
                        args["DataType"] = "double"
        self._Data = data
        return super().__init__(descriptors=[], args=args, config_file=config_file, **kwargs)

    def getMetaData(self, key=None):
        DataType = self._QSArgs.DataType
        if key is None:
            return {"DataType": DataType}
        elif key == "DataType":
            return DataType
        return None

    def getID(self, idt=None):
        if self._DataContent == "Factor":
            return self._Data.columns.tolist()
        elif self._DataContent == "ID":
            return self._Data.index.tolist()
        else:
            return []

    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        if self._DataContent in ("DateTime", "Factor"):
            return self._Data.index.tolist()
        else:
            return []

    def readData(self, ids, dts, **kwargs):
        if self._DataContent == "Value":
            return pd.DataFrame([(self._Data,) * len(ids)] * len(dts), index=dts, columns=ids)
        elif self._DataContent == "ID":
            Data = pd.DataFrame(self._Data.values.reshape((1, self._Data.shape[0])).repeat(len(dts), axis=0), index=dts, columns=self._Data.index)
        elif self._DataContent == "DateTime":
            Data = pd.DataFrame(self._Data.values.reshape((self._Data.shape[0], 1)).repeat(len(ids), axis=1), index=self._Data.index, columns=ids)
        else:
            Data = self._Data
        if Data.columns.intersection(ids).shape[0] == 0:
            return pd.DataFrame(index=dts, columns=ids, dtype=("O" if self._QSArgs.DataType != "double" else float))
        if self._QSArgs.LookBack == 0:
            return Data.reindex(index=dts, columns=ids)
        else:
            return fillNaByLookback(Data.reindex(index=sorted(Data.index.union(dts)), columns=ids), lookback=self._QSArgs.LookBack * 24.0 * 3600).loc[dts, :]

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        return self.readData(ids=local_context.ids, dts=local_context.dts)

if __name__=="__main__":
    np.random.seed(0)
    nDT, nID = 5, 3
    DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
    IDs = [str(i).zfill(6)+".SZ" for i in range(1, nID+1)]
    Data = pd.DataFrame(
        np.random.randn(nDT, nID), 
        index=DTs,
        columns=IDs
    )
    F = DataFactor(data=Data, args={"Name": "test_factor"})
    print(F.Args)
    print(F.readData(ids=IDs, dts=DTs))

    print("===")