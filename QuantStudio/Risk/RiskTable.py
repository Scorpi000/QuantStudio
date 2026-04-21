# coding=utf-8
import datetime as dt
from typing import Optional, List, Union, Any

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Core.Node import Node
from QuantStudio.Factor.Factor import FactorInitData, FactorContext, FactorLocalContext
from QuantStudio.Risk.RiskDB import RiskDB, FactorRDB
from QuantStudio.Tools.DateTimeFun import cutDateTime
from QuantStudio.Tools.DataTypeFun import dict2id


class RiskTable(Node):
    """风险表"""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="风险表", frozen=True) 
    
    def __init__(self, rdb: FactorRDB, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """实例化风险表

        Args:
            rdb: 风险表所属的风险库
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于内部打印日志, 如果没有指定则使用默认的 __QS_Logger__ 对象
        """
        self._RiskDB = rdb
        self._QS_PrepareIgnoredArgs: tuple = tuple()# 决定 PrepareID 不需要的参数集，如果为空，表示所有参数都需要
        return super().__init__(deps=[], args=args, config_file=config_file, **kwargs)
    
    @property
    def RiskDB(self) -> RiskDB:
        """风险表所属的风险库"""
        return self._RiskDB
    
    @property
    def PrepareID(self) -> str:
        if not self._QS_PrepareIgnoredArgs: return self.QSID
        if (not getattr(self, "_QS_ID", None)) or (not getattr(self, "_QS_PrepareID", None)):
            DumpedMdl = self.model_dump()
            DumpedMdl["__qsargs__"] = {iArg: iVal for iArg, iVal in DumpedMdl["__qsargs__"].items() if iArg not in self._QS_PrepareIgnoredArgs}
            self._QS_PrepareID = dict2id(DumpedMdl)
        return self._QS_PrepareID

    def getMetaData(self, key:Optional[str]=None) -> Union[Any, pd.Series]:
        """获取风险表的元信息, 元信息由若干个键值对组成

        Args:
            key: 元信息键, None 表示获取所有的元信息

        Returns:
            如果 key 非 None 则返回该 key 对应的元信息
            如果 key=None, 则返回 Series(index=[所有的 key])
        """
        if key is None: return pd.Series()
        return None
    
    def getDateTime(self, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        """获取时点序列

        Args:
            start_dt: 起始日, 非 None 表示截取 start_dt 之后的时点
            end_dt: 结束日, 非 None 表示截取 end_dt 之前的时点

        Returns:
            时点序列, 若为空 list, 表示该风险表没有固定的时点序列或者无法获取
        """
        return []
    
    def getID(self, idt:Optional[dt.datetime]=None) -> List[str]:
        """获取 ID 序列

        Args:
            idt: 给定的时点, 非 None 表示获取该时点的 ID 序列, None 表示获取所有的 ID 序列

        Returns:
            ID 序列, 若为空 list, 表示该风险表没有固定的 ID 序列或者无法获取
        """
        if idt is None: idt = self.getDateTime()[-1]
        Cov = self.readCov(dts=[idt]).iloc[0]
        return Cov.index.tolist()
    
    def readCov(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> Panel:
        """读取风险矩阵

        Args:
            dts: 时点序列
            ids: ID 序列, None 表示读取所有的 ID

        Returns:
            Panel(item=dts, major_axis=ids, minor_axis=ids)
        """
        raise NotImplementedError
    
    def __getitem__(self, key):
        if isinstance(key, tuple): key += (slice(None),) * (2 - len(key))
        else: key = (key, slice(None))
        if len(key)>2: raise IndexError("QuantStudio.Risk.RiskTable.RiskTable: Too many indexers")
        DTs, IDs = key
        if DTs==slice(None): DTs = self.getDateTime()
        elif isinstance(DTs, dt.datetime): DTs = [DTs]
        if IDs==slice(None): IDs = None
        elif isinstance(IDs, str): IDs = [IDs]
        Data = self.readCov(DTs, ids=IDs)
        return Data.loc[key[0], key[1], key[1]]

    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[Any]:
        PrepareData = {
            "DTRange": init_data.DTRange,
            "SectionIDs": init_data.SectionIDs,
            "Args": self._QSArgs.to_dict(repr=False)
        }
        _, PrepareData = context.PrepareNodeDict.setdefault(self.PrepareID, (self.QSID, PrepareData))
        PrepareData["DTRange"] = (min(init_data.DTRange[0], PrepareData["DTRange"][0]), max(init_data.DTRange[1], PrepareData["DTRange"][1]))
        return super().init_compute(path=path, init_data=init_data, context=context)
    
    def forward_compute(self, path, fwd_data, context):
        return super().forward_compute(path, fwd_data, context)

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Panel:
        return self.readCov(dts=local_context.DTs, ids=local_context.IDs)


class FactorRT(RiskTable):
    """多因子风险表"""

    @property
    def FactorNames(self) -> List[str]:
        """因子名称列表"""
        return []
    
    def getID(self, idt:Optional[dt.datetime]=None) -> List[str]:
        if idt is None: idt = self.getDateTime()
        if not idt: return super().getID(idt=None)
        else: idt = idt[-1]
        SpecificRisk = self.readSpecificRisk(idt=idt)
        return SpecificRisk.index.tolist()
    
    def getFactorReturnDateTime(self, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        """获取因子收益的时点序列

        Args:
            start_dt: 起始日, 非 None 表示截取 start_dt 之后的时点
            end_dt: 结束日, 非 None 表示截取 end_dt 之前的时点

        Returns:
            时点序列, 若为空 list, 表示该风险表没有固定的因子收益时点序列或者无法获取
        """
        FactorReturn = self.readFactorReturn()
        if FactorReturn is not None: return cutDateTime(FactorReturn.index, start_dt=start_dt, end_dt=end_dt)
        return []
    
    def getSpecificReturnDateTime(self, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        """获取特异性收益的时点序列

        Args:
            start_dt: 起始日, 非 None 表示截取 start_dt 之后的时点
            end_dt: 结束日, 非 None 表示截取 end_dt 之前的时点

        Returns:
            时点序列, 若为空 list, 表示该风险表没有固定的特异性收益时点序列或者无法获取
        """
        SpecificReturn = self.readSpecificReturn()
        if SpecificReturn is not None: return cutDateTime(SpecificReturn.index, start_dt=start_dt, end_dt=end_dt)
        return []
    
    def readCov(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> Panel:
        FactorCov = self.readFactorCov(dts=dts)
        FactorData = self.readFactorData(dts=dts, ids=ids)
        SpecificRisk = self.readSpecificRisk(dts=dts, ids=ids)
        Data = {}
        for iDT in FactorCov:
            if ids is None:
                iIDs = SpecificRisk.loc[iDT].index
                iFactorData = FactorData.loc[:, iDT].reindex(index=iIDs)
            else:
                iIDs = ids
                iFactorData = FactorData.loc[:, iDT]
            iCov = np.dot(np.dot(iFactorData.values, FactorCov[iDT].values), iFactorData.values.T) + np.diag(SpecificRisk.loc[iDT].values**2)
            Data[iDT] = pd.DataFrame(iCov, index=iIDs, columns=iIDs)
        return Panel(Data, items=dts)
    
    def readFactorCov(self, dts:List[dt.datetime]):
        """读取因子风险矩阵

        Args:
            dts: 时点序列

        Returns:
            Panel(item=dts, major_axis=[因子], minor_axis=[因子])
        """
        return Panel(items=dts)
    
    def readSpecificRisk(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> pd.DataFrame:
        """读取特异性风险

        Args:
            dts: 时点序列
            ids: 证券 ID 序列, None 表示获取表里所有的 ID

        Returns:
            DataFrame(index=dts, columns=ids)
        """
        return pd.DataFrame(index=dts, columns=ids)
    
    def readFactorData(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> Panel:
        """读取因子截面数据

        Args:
            dts: 时点序列
            ids: 证券 ID 序列, None 表示获取表里所有的 ID

        Returns:
            Panel(items=[因子], major_axis=dts, minor_axis=ids)
        """
        return Panel(major_axis=dts, minor_axis=ids)
    
    def readFactorReturn(self, dts:List[dt.datetime]) -> pd.DataFrame:
        """读取因子收益率

        Args:
            dts: 时点序列

        Returns:
            DataFrame(index=dts, columns=[因子])
        """
        return pd.DataFrame(index=dts)
    
    def readSpecificReturn(self, dts:List[dt.datetime], ids:Optional[List[str]]=None) -> pd.DataFrame:
        """读取特异性收益率

        Args:
            dts: 时点序列
            ids: 证券 ID 序列, None 表示获取表里所有的 ID

        Returns:
            DataFrame(index=dts, columns=ids)
        """
        return pd.DataFrame(index=dts, columns=ids)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Panel | dict:
        if not local_context.ExtraData.get("Fields", []):
            return super().backward_compute(path=path, bwd_data_list=bwd_data_list, context=context, local_context=local_context)
        Rslt = {}
        for iField in local_context.ExtraData["Fields"]:
            if iField == "FactorCov": Rslt["FactorCov"] = self.readFactorCov(dts=local_context.DTs)
            elif iField == "FactorData": Rslt["FactorData"] = self.readFactorData(dts=local_context.DTs, ids=local_context.IDs)
            elif iField == "SpecificRisk": Rslt["SpecificRisk"] = self.readSpecificRisk(dts=local_context.DTs, ids=local_context.IDs)
            if iField == "Cov": Rslt["Cov"] = self.readCov(dts=local_context.DTs, ids=local_context.IDs)
            if iField == "FactorReturn": Rslt["FactorReturn"] = self.readFactorReturn(dts=local_context.DTs)
            if iField == "SpecificReturn": Rslt["SpecificReturn"] = self.readSpecificReturn(dts=local_context.DTs, ids=local_context.IDs)
        return Rslt
