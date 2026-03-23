# -*- coding: utf-8 -*-
import datetime as dt
from typing import List, Optional, Any, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import Node, Context, DTLocalContext, DTInitData, __QS_Context__
from QuantStudio.Core.CalcEngine import __QS_Engine__, Engine
from QuantStudio.Factor.FactorCache import FactorCache
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.AuxiliaryFun import partitionListMovingSampling, partitionList


class FactorContext(Context):
    """因子节点运算时全局上下文对象"""

    # NodeDict: {节点ID: Factor}
    # NodeState: {节点ID: {"start_dt", "section_ids"}}
    # PID: str = Field(default="0", title="运行ID", description="当前的运行 ID, 默认为 '0'")
    # PIDList: List[str] = Field(default=["0"], title="所有运行ID")
    # Event: dict = Field(default={}, title="", description="{节点ID: (Sub2MainQueue, Event)}, 用于多进程同步的 Event 数据")
    DTRuler: List[dt.datetime] = Field(title="时点标尺", description="当前运行计算时点标尺", frozen=True)
    DefaultSectionIDs: List[str] = Field(title="默认截面", description="当前运行需要计算的默认截面 ID", frozen=True)
    FactorDataCache: Optional[FactorCache] = Field(default=None, title="因子缓存", frozen=True)

    def model_post_init(self, context: Any, /) -> None:
        self._DefaultPIDIDs = self.splitID(self.DefaultSectionIDs)

    # 并发运行后返回需要同步的内容
    def getUpdateData(self) -> dict:
        UpdateData = super().getUpdateData()
        if self.FactorDataCache: UpdateData["cache"] = self.FactorDataCache.getUpdateData()
        return UpdateData

    # 并发运行后更新同步内容
    def updateContext(self, update_data: dict):
        if self.FactorDataCache: self.FactorDataCache.updateCache(update_data.pop("cache", {}))
        return super().updateContext(update_data)

    def getDateTime(self, dt_range):
        StartIdx, EndIdx = np.searchsorted(self.DTRuler, dt_range[0], side="left"), np.searchsorted(self.DTRuler, dt_range[1], side="right")
        return self.DTRuler[StartIdx:EndIdx]

    def __setattr__(self, name, value):
        if name == "PIDList":
            self._DefaultPIDIDs = self.splitID(self.DefaultSectionIDs)
        return super().__setattr__(name, value)

    @property
    def DefaultPIDIDs(self):
        return self._DefaultPIDIDs

    # 划分 ID
    def splitID(self, ids):
        nPrcs = len(self.PIDList)
        if nPrcs == 0: return {}
        elif nPrcs == 1: return {self.PIDList[0]: ids}
        if self.SplitType == "连续切分":
            SubIDs = partitionList(ids, nPrcs)
        elif self.SplitType == "间隔切分":
            SubIDs = partitionListMovingSampling(ids, nPrcs)
        else:
            raise __QS_Error__(f"不支持的 ID 切分方式: {self.SplitType}")
        return {iPID: SubIDs[i] for i, iPID in enumerate(self.PIDList)}

    def getID(self, factor_id, pids=None):
        if pids is not None:
            PIDIDs = self.NodeState[factor_id]["pid_ids"]
            return sorted(sum((PIDIDs[iPID] for iPID in pids), []))
        else:
            return self.NodeState[factor_id]["section_ids"]


class FactorLocalContext(DTLocalContext):
    """因子节点运算时局部上下文对象"""

    IDs: List[str] = Field(title="ID 序列")
    PIDs: Optional[List[str]] = Field(default=None)
    
    # 并发运行时切分自身成 n 份
    def split(self, n: int, context: FactorContext, **kwargs):
        PIDIDs = context.splitID(self.IDs)
        Args = self.model_dump()
        return [self.__class__(**(Args | {"IDs": PIDIDs[iPID]})) for iPID in context.PIDList]


class FactorInitData(DTInitData):
    """因子节点初始化数据对象"""
    
    SectionIDs: Optional[List[str]] = Field(default=None, title="截面ID")
    SubFactorName: Optional[str] = Field(default=None, title="因子名称", description="传递给因子表用于准备原始数据的因子名称")


class Factor(Node):
    """
    因子对象
    因子可看做 DataFrame(index=[时点], columns=[ID])
    时点数据类型是 datetime, ID 的数据类型是 str
    """
    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="Factor", frozen=True, title="名称")
        Meta: dict = Field(default={}, title="元信息", frozen=False, exclude=True)
        SectionIDs: Optional[List[str]] = Field(default=None, title="截面ID", frozen=True)
        CalcDTRuler: Optional[List[dt.datetime]] = Field(default=None, title="计算时点标尺", frozen=True)
        CacheEnabled: bool = Field(default=True, frozen=True, title="启用缓存")

    def __init__(self, ft: Optional["FactorTable"]=None, descriptors: List["Factor"] = [], extra_deps: List[Node] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
        """初始化因子对象

        Args:
            ft: 因子所属的因子表对象, 如果为 None 表示因子不属于任何因子表
            descriptors: 因子依赖的因子(描述子)列表
            extra_deps: 因子依赖的其他计算节点列表
            args: 指定的对象参数集
            config_file: 配置文件路径
        """
        if ft and descriptors:
            raise __QS_Error__("因子表和描述子列表不能都存在!")
        self._FactorTable = ft
        self._Descriptors = descriptors
        self._ExtraDeps = extra_deps
        kwargs.pop("deps", [])
        if ft:
            return super().__init__(deps=[ft] + extra_deps, args=args, config_file=config_file, **kwargs)
        else:
            return super().__init__(deps=descriptors + extra_deps, args=args, config_file=config_file, **kwargs)
    
    def new(self, args:dict={}, **kwargs) -> "Factor":
        kwargs = {"ft": self._FactorTable, "descriptors": self._Descriptors, "extra_deps": self._ExtraDeps} | kwargs
        return super().new(args=args, **kwargs)
    
    @property
    def FactorTable(self) -> "FactorTable":
        """因子所属的因子表, None 表示因子不属于任何因子表"""
        return self._FactorTable

    @property
    def Descriptors(self) -> List["Factor"]:
        """因子所依赖的因子列表, 返回 [] 表示因子不依赖任何因子"""
        if self._FactorTable:
            return []
        else:
            return self._Descriptors.copy()

    def getMetaData(self, key:Optional[str]=None) -> Union[Any, pd.Series]:
        """获取因子的元信息, 元信息由若干个键值对组成

        Args:
            key: 元信息键, None 表示获取所有的元信息

        Returns:
            如果 key 非 None 则返回该 key 对应的元信息
            如果 key=None, 则返回 Series(index=[所有的 key])
        """
        if key:
            if key in self._QSArgs.Meta: return self._QSArgs.Meta[key]
            elif self._FactorTable: return self._FactorTable.getFactorMetaData(factor_names=[self._QSArgs.Name], key=key).loc[self._QSArgs.Name]
            else: return None
        else:
            Meta = pd.Series(self._QSArgs.Meta)
            if self._FactorTable:
                Meta = Meta.combine_first(self._FactorTable.getFactorMetaData(factor_names=[self._QSArgs.Name], key=None).loc[self._QSArgs.Name])
            return Meta

    def getID(self, idt:Optional[dt.datetime]=None, **kwargs) -> List[str]:
        """获取 ID 序列

        Args:
            idt: 给定的时点, 非 None 表示获取该时点的 ID 序列, None 表示获取所有的 ID 序列

        Returns:
            ID 序列, 若为空 list, 表示该因子没有固定的 ID 序列或者无法获取
        """
        if self._QSArgs.SectionIDs is not None: return self._QSArgs.SectionIDs
        for iContext in reversed(__QS_Context__):
            if hasattr(iContext, "SectionIDs") and (iContext.SectionIDs is not None): return iContext.SectionIDs
        if self._FactorTable is not None:
            return self._FactorTable.getID(ifactor_name=self._QSArgs.Name, idt=idt, **kwargs)
        return []

    def getDateTime(self, iid:Optional[str]=None, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None, **kwargs) -> List[dt.datetime]:
        """获取时点序列

        Args:
            iid: 给定的 ID, 非 None 表示获取该 ID 的时点序列, None 表示获取所有的时点序列
            start_dt: 起始日, 非 None 表示截取 start_dt 之后的时点
            end_dt: 结束日, 非 None 表示截取 end_dt 之前的时点

        Returns:
            时点序列, 若为空 list, 表示该因子没有固定的时点序列或者无法获取
        """
        if self._FactorTable is not None:
            return self._FactorTable.getDateTime(ifactor_name=self._QSArgs.Name, iid=iid, start_dt=start_dt, end_dt=end_dt, **kwargs)
        return []
    
    def readData(self, ids:List[str], dts:List[dt.datetime], **kwargs) -> pd.DataFrame:
        """读取因子数据

        Args:
            ids: ID 序列
            dts: 时点序列
            kwargs: 可传入的参数有
                dt_ruler: 时点标尺序列, 如果没有传入则为 dts
                section_ids: 截面 ID 序列，如果没有传入则为因子参数集中指定的 SectionIDs, 如果参数集中未指定则为 ids

        Returns:
            DataFrame(index=dts, columns=ids)
        """
        if (not __QS_Context__) and (self._FactorTable is not None): return self._FactorTable.readData(factor_names=[self._QSArgs.Name], ids=ids, dts=dts, **kwargs).iloc[0]
        SectionIDs = kwargs.get("section_ids", self._QSArgs.SectionIDs)
        if not SectionIDs: SectionIDs = ids
        if not __QS_Context__: Context = FactorContext(DTRuler=kwargs.get("dt_ruler", dts), DefaultSectionIDs=SectionIDs)
        else: Context = __QS_Context__[-1]
        if not __QS_Engine__: ExecEngine = Engine()
        else: ExecEngine = __QS_Engine__[-1]
        LocalContext = FactorLocalContext(DTs=dts, IDs=ids)
        InitData = FactorInitData(DTRange=(dts[0], dts[-1]), SectionIDs=SectionIDs)
        Rslt = ExecEngine.run([self], Context, fwd_data_list=[LocalContext], init_data_list=[InitData])
        return Rslt[0]

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
    
    def _QS_getCalcDTs(self, dts:list[dt.datetime], mask:bool=False):
        CalcDTs = self._QSArgs.CalcDTRuler
        if CalcDTs:
            StartIdx, EndIdx = np.searchsorted(CalcDTs, dts[0], side="left"), np.searchsorted(CalcDTs, dts[-1], side="right")
            CalcDTs = CalcDTs[StartIdx:EndIdx]
            if mask:
                return np.isin(dts, CalcDTs)
            else:
                return CalcDTs
        else:
            return None
    
    # 准备缓存数据
    def _prepareCacheData(self, context: FactorContext):
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return 0
        DTRange = context.FactorDataCache.getDTRange(key=self.QSID, dt_range=DTRange)
        if DTRange is None: return 0
        DTs = context.getDateTime(DTRange)
        if not DTs: return 0
        if self._FactorTable:
            RawKey = self._FactorTable.PrepareID
        else:
            RawKey = None
        PIDIDs = context.NodeState[self.QSID]["pid_ids"]
        iSectionIDs = PIDIDs[context.PID]
        CalcDTs = self._QS_getCalcDTs(DTs, mask=False)
        if (CalcDTs is not None) and (not CalcDTs): 
            StdData = pd.DataFrame(index=DTs, columns=iSectionIDs)
        else:
            if RawKey is None:
                RawData = None
            else:
                RawData = context.FactorDataCache.readRawData(key=RawKey + "-" + self._QSArgs.Name, target_fields=None, pids=[context.PID])
            if RawData:
                if len(RawData) == 1: RawData = RawData["RawData"]
                StdData = self._FactorTable.__QS_calcData__(RawData, factor_names=[self._QSArgs.Name], ids=iSectionIDs, dts=CalcDTs or DTs).iloc[0]
            elif self._FactorTable:
                RawData = self._FactorTable.__QS_prepareRawData__(factor_names=[self._QSArgs.Name], ids=iSectionIDs, dts=CalcDTs or DTs)
                if RawData is not None:
                    self._QS_Logger.warning(f"因子 {self._QSArgs.Name} (QSID: {self.QSID}) 的原始数据缓存丢失!")
                StdData = self._FactorTable.__QS_calcData__(raw_data=RawData, factor_names=[self._QSArgs.Name], ids=iSectionIDs, dts=CalcDTs or DTs).iloc[0]
            else:
                return 0
            if CalcDTs: StdData = StdData.reindex(index=DTs)
        DataType = self.getMetaData(key="DataType")
        if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps], "FactorTable": None if not self._FactorTable else self._FactorTable.Name}
        else: Meta = {}
        context.FactorDataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=context.PID, if_exists="append", data_type=DataType, meta=Meta)
        context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=DTRange)
        return 0

    # NodeState: {"dt_range", "section_ids", "pid_ids"}
    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        FactorState = context.NodeState.setdefault(self.QSID, {})
        # 处理时点
        DTRange = FactorState.get("dt_range", None)
        if DTRange is None:
            FactorState["dt_range"] = init_data.DTRange
        else:
            FactorState["dt_range"] = (min(DTRange[0], init_data.DTRange[0]), max(DTRange[1], init_data.DTRange[1]))
        DTRuler = context.DTRuler
        if FactorState["dt_range"][0] < DTRuler[0]:
            raise __QS_Error__(f"对于因子 {self.Name}(QSID: {self.QSID}), 起始时点为 {FactorState["dt_range"][0]}, 时点标尺长度不足, 起始时点为 {DTRuler[0]}")
        if FactorState["dt_range"][1] > DTRuler[-1]:
            raise __QS_Error__(f"对于因子 {self.Name}(QSID: {self.QSID}), 结束时点为 {FactorState["dt_range"][1]}, 时点标尺长度不足, 结束时点为 {DTRuler[-1]}")
        # 处理截面ID
        if init_data.SectionIDs is not None:
            InitSectionIDs = init_data.SectionIDs
        elif self._QSArgs.SectionIDs is not None:
            InitSectionIDs = self._QSArgs.SectionIDs
        else:
            InitSectionIDs = context.DefaultSectionIDs
        if "section_ids" in FactorState: SectionIDs = FactorState["section_ids"]
        elif self._QSArgs.SectionIDs: SectionIDs = self._QSArgs.SectionIDs
        else: SectionIDs = InitSectionIDs
        if InitSectionIDs != SectionIDs:
            raise __QS_Error__(f"因子 {self._QSArgs.Name}({self.QSID}) 指定了不同的截面!")
        if "section_ids" not in FactorState:
            FactorState["section_ids"] = SectionIDs
            if SectionIDs == context.DefaultSectionIDs:
                FactorState["pid_ids"] = context.DefaultPIDIDs
            else:
                FactorState["pid_ids"] = context.splitID(SectionIDs)
        # 默认
        DefaultInitData = super().init_compute(path=path, init_data=init_data, context=context)
        if self.QSID in path: return []
        if self._FactorTable:
            return [FactorInitData(DTRange=FactorState["dt_range"], SectionIDs=SectionIDs, SubFactorName=self._QSArgs.Name)] + DefaultInitData[1:]
        else:
            return [FactorInitData(DTRange=FactorState["dt_range"], SectionIDs=SectionIDs)] * len(self._Descriptors) + DefaultInitData[len(self._Descriptors):]

    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        if self._FactorTable:
            return [], fwd_data
        else:
            return super().forward_compute(path=path, fwd_data=fwd_data, context=context)

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        if context.FactorDataCache and self._QSArgs.CacheEnabled:
            self._prepareCacheData(context=context)
            StdData = context.FactorDataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self.getMetaData(key="DataType"))
            return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)
        elif self._FactorTable:
            RawData = self._FactorTable.__QS_prepareRawData__(factor_names=[self._QSArgs.Name], ids=local_context.IDs, dts=local_context.DTs)
            return self._FactorTable.__QS_calcData__(raw_data=RawData, factor_names=[self._QSArgs.Name], ids=local_context.IDs, dts=local_context.DTs).iloc[0]
        else:
            raise NotImplementedError
    
    def merge_result(self, result_list: List[Any], context: FactorContext) -> Any:
        Data = pd.concat(result_list, join="outer", axis=1, ignore_index=False)
        return Data.sort_index(axis=1)
    
    # -----------------------------重载运算符-------------------------------------
    def __add__(self, other):
        from QuantStudio.Factor.BasicOperator import add
        return add(self, other)
    
    def __radd__(self, other):
        from QuantStudio.Factor.BasicOperator import add
        return add(other, self)
    
    def __sub__(self, other):
        from QuantStudio.Factor.BasicOperator import sub
        return sub(self, other)
    
    def __rsub__(self, other):
        from QuantStudio.Factor.BasicOperator import sub
        return sub(other, self)
    
    def __mul__(self, other):
        from QuantStudio.Factor.BasicOperator import mul
        return mul(self, other)
    
    def __rmul__(self, other):
        from QuantStudio.Factor.BasicOperator import mul
        return mul(other, self)
    
    def __pow__(self, other):
        from QuantStudio.Factor.BasicOperator import qs_pow
        return qs_pow(self, other)
    
    def __rpow__(self, other):
        from QuantStudio.Factor.BasicOperator import qs_pow
        return qs_pow(other, self)
    
    def __truediv__(self, other):
        from QuantStudio.Factor.BasicOperator import div
        return div(self, other)
    
    def __rtruediv__(self, other):
        from QuantStudio.Factor.BasicOperator import div
        return div(other, self)
    
    def __floordiv__(self, other):
        from QuantStudio.Factor.BasicOperator import floordiv
        return floordiv(self, other)
    
    def __rfloordiv__(self, other):
        from QuantStudio.Factor.BasicOperator import floordiv
        return floordiv(other, self)
    
    def __mod__(self, other):
        from QuantStudio.Factor.BasicOperator import mod
        return mod(self, other)
        
    def __rmod__(self, other):
        from QuantStudio.Factor.BasicOperator import mod
        return mod(other, self)
    
    def __and__(self, other):
        from QuantStudio.Factor.BasicOperator import qs_and
        return qs_and(self, other)
        
    def __rand__(self, other):
        from QuantStudio.Factor.BasicOperator import qs_and
        return qs_and(other, self)
    
    def __or__(self, other):
        from QuantStudio.Factor.BasicOperator import qs_or
        return qs_or(self, other)        
    
    def __ror__(self, other):
        from QuantStudio.Factor.BasicOperator import qs_or
        return qs_or(other, self)
    
    def __xor__(self, other):
        from QuantStudio.Factor.BasicOperator import xor
        return xor(self, other)
        
    def __rxor__(self, other):
        from QuantStudio.Factor.BasicOperator import xor
        return xor(other, self)
    
    def __lt__(self, other):
        from QuantStudio.Factor.BasicOperator import lt
        return lt(self, other)
    
    def __le__(self, other):
        from QuantStudio.Factor.BasicOperator import le
        return le(self, other)
    
    def __eq__(self, other):
        from QuantStudio.Factor.BasicOperator import eq
        return eq(self, other)
    
    def __ne__(self, other):
        from QuantStudio.Factor.BasicOperator import neq
        return neq(self, other)
    
    def __gt__(self, other):
        from QuantStudio.Factor.BasicOperator import gt
        return gt(self, other)
    
    def __ge__(self, other):
        from QuantStudio.Factor.BasicOperator import ge
        return ge(self, other)
    
    def __neg__(self):
        from QuantStudio.Factor.BasicOperator import neg
        return neg(self)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        from QuantStudio.Factor.BasicOperator import qs_abs
        return qs_abs(self)
    
    def __invert__(self):
        from QuantStudio.Factor.BasicOperator import qs_not
        return qs_not(self)


class DataFactor(Factor):
    """数据因子: 直接赋予数据产生的因子"""
    
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Name: str = Field(default="DataFactor", frozen=True, title="名称")
        DataType: Literal["double", "string", "object"] = Field(default="double", frozen=True, title="数据类型")
        LookBack: int = Field(default=0, title="回溯天数", frozen=True)

    def __init__(self, data:Union[Any, pd.DataFrame, pd.Series], args: dict={}, config_file: Optional[str] = None, **kwargs):
        """初始化数据因子对象

        Args:
            data: 构造因子的数据
            args: 指定的对象参数集
            config_file: 配置文件路径
        """
        args = args.copy()
        if "DataType" not in args:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                try:
                    data = data.astype(float)
                except:
                    args["DataType"] = "object"
                else:
                    args["DataType"] = "double"
            else:
                args.setdefault("Name", str(data))
                if isinstance(data, str):
                    args["DataType"] = "string"
                else:
                    try:
                        data = float(data)
                    except:
                        args["DataType"] = "object"
                    else:
                        args["DataType"] = "double"
        elif args["DataType"]=="double":
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data = data.astype(float)
            else:
                args.setdefault("Name", str(data))
                data = float(data)
        elif args["DataType"]=="string":
            if not isinstance(data, (pd.Series, pd.DataFrame)):
                args.setdefault("Name", str(data))
                data = str(data)
        super().__init__(ft=None, descriptors=[], args=args, config_file=config_file, **kwargs)
        SectionIDs = self._QSArgs.SectionIDs
        if isinstance(data, pd.Series):
            if pd.api.types.is_datetime64_any_dtype(data.index):
                CalcMask = self._QS_getCalcDTs(data.index, mask=True)
                if SectionIDs is None:
                    self._DataContent = "DateTime"
                else:
                    self._DataContent = "Factor"
                    data = pd.DataFrame(np.repeat(np.reshape(data.values, (-1, 1)), len(SectionIDs), axis=1), index=data.index, columns=SectionIDs)
                if CalcMask is not None: data = data[CalcMask]
            else:
                CalcDTs = self._QSArgs.CalcDTRuler
                if CalcDTs is None:
                    self._DataContent = "ID"
                    if SectionIDs is not None: data = data.reindex(index=SectionIDs)
                else:
                    self._DataContent = "Factor"
                    data = pd.DataFrame(np.repeat(np.reshape(data.values, (1, -1)), len(CalcDTs), axis=0), index=CalcDTs, columns=data.index)
                    if SectionIDs is not None: data = data.reindex(columns=SectionIDs)
        elif isinstance(data, pd.DataFrame):
            self._DataContent = "Factor"
            if SectionIDs is not None: data = data.reindex(columns=SectionIDs)
            CalcMask = self._QS_getCalcDTs(data.index, mask=True)
            if CalcMask is not None: data = data[CalcMask]
        else:
            CalcDTs = self._QSArgs.CalcDTRuler
            if (SectionIDs is not None) and (CalcDTs is not None):
                self._DataContent = "Factor"
                data = pd.DataFrame([(data,)*len(SectionIDs)]*len(CalcDTs), index=CalcDTs, columns=SectionIDs)
            elif SectionIDs is not None:
                self._DataContent = "ID"
                data = pd.Series([data] * len(SectionIDs), index=SectionIDs)
            elif CalcDTs is not None:
                self._DataContent = "DateTime"
                data = pd.Series([data] * len(CalcDTs), index=CalcDTs)
            else:
                self._DataContent = "Value"
        self._Data = data        
    
    def new(self, args:dict={}, **kwargs) -> "DataFactor":
        kwargs = {"data": self._Data} | kwargs
        return super().new(args=args, **kwargs)
    
    def getMetaData(self, key:Optional[str]=None) -> Union[Any, pd.Series]:
        DataType = self._QSArgs.DataType
        if key is None:
            return {"DataType": DataType}
        elif key == "DataType":
            return DataType
        return None

    def getID(self, idt:Optional[dt.datetime]=None) -> List[str]:
        if self._DataContent == "Factor":
            return self._Data.columns.tolist()
        elif self._DataContent == "ID":
            return self._Data.index.tolist()
        else:
            return []

    def getDateTime(self, iid:Optional[str]=None, start_dt:Optional[dt.datetime]=None, end_dt:Optional[dt.datetime]=None) -> List[dt.datetime]:
        if self._DataContent in ("DateTime", "Factor"):
            return self._Data.index.tolist()
        else:
            return []

    def readData(self, ids:List[str], dts:List[dt.datetime], **kwargs) -> pd.DataFrame:
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
    
    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        return []

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        return self.readData(ids=local_context.IDs, dts=local_context.DTs)

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