# -*- coding: utf-8 -*-
"""因子运算"""
import os
import datetime as dt
from functools import partial
from typing import Optional, Literal, List, Any, Tuple, Callable, Union
from multiprocessing import Queue, Event

import dill
import pandas as pd
import numpy as np
from pydantic import Field

from QuantStudio.Core import __QS_Error__, __QS_Object__
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import Factor, DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame
from QuantStudio.Tools.AuxiliaryFun import partitionList


class FactorOperator(__QS_Object__):
    """因子算子"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        OperatorType: Literal["Point", "Time", "Section", "Panel"] = Field(title="算子类型", frozen=True)
        Name: str = Field(default="FactorOperator", title="名称", frozen=True)
        ModelArgs: dict = Field(default={}, title="模型参数", frozen=True)
        Arity: Optional[int] = Field(default=None, ge=1, title="入参数", frozen=True)
        DataType: Literal["double", "string", "object"] = Field(default="double",title="数据类型", frozen=True)
        Description: str = Field(default="", title="描述信息", frozen=False, exclude=True)
        Meta: dict = Field(default={}, title="元信息", frozen=False, exclude=True)
        InputFormat: Literal["numpy", "pandas"] = Field(default="numpy", title="输入格式", frozen=True)
        ExpandDescriptors: list[int] = Field(default=[], title="展开描述子", frozen=True)
        DescriptorCompoundType: list = Field(default=[], title="描述子复合类型", frozen=True)
        MultiMapping: bool = Field(default=False, title="多重映射", frozen=True)
        CompoundType: list = Field(default=[], title="复合类型", frozen=True)
        
        def __init__(self, /, **data: Any) -> None:
            if ("DataType" not in data) and (data.get("CompoundType", []) or data.get("MultiMapping", False)):
                data["DataType"] = "object"
            return super().__init__(**data)

    def __getstate__(self):
        if "calculate" in self.__dict__:
            state = self.__dict__.copy()
            # Remove the unpicklable entries.
            state["calculate"] = dill.dumps(self.calculate)
            return state
        else:
            return super().__getstate__()
    
    def __setstate__(self, state):
        if "calculate" in state: state["calculate"] = dill.loads(state["calculate"])
        self.__dict__.update(state)

    @property
    def Name(self) -> str:
        """算子名称"""
        return self._QSArgs.Name
    
    def model_dump(self):
        DumpedModel = super().model_dump()
        DumpedModel["__func__"] = self.calculate
        return DumpedModel
    
    def new(self, args={}, **kwargs):
        NewOperator = super().new(args=args, **kwargs)
        if getattr(self.calculate, "__self__", None) != self:
            NewOperator.calculate = self.calculate
        return NewOperator
    
    def _QS_validate(self, *x, **kwargs):
        Arity = len(x)
        if self._QSArgs.Arity is None:
            return self.new(args={"Arity": Arity}, **kwargs)
        elif Arity != self._QSArgs.Arity:
            raise __QS_Error__(f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 和指定的入参数 {self._QSArgs.Arity} 不符!")
        else:
            return self

    def _QS_adjOutputPandas(self, df, cols, dts, ids):
        if isinstance(df, pd.DataFrame):
            if isinstance(df.index, pd.MultiIndex):
                if self._QSArgs.MultiMapping:
                    TmpData, Cols, df = {}, df.columns, df.groupby(axis=0, level=[0, 1], as_index=True)
                    for iCol in Cols:
                        TmpData[iCol] = df[iCol].apply(lambda s: s.tolist())
                    df, TmpData = pd.DataFrame(TmpData).loc[:, Cols], None
                elif df.index.duplicated().any():
                    raise __QS_Error__(
                        f"算子 '{self.Name}' 的数据无法保证唯一性, 可以尝试将 '多重映射' 参数取值调整为 True")
                df = df.reindex(columns=cols).apply(lambda s: tuple(s), axis=1).unstack()
            return df.reindex(index=dts, columns=ids)
        elif isinstance(df, pd.Series) and isinstance(df.index, pd.MultiIndex):
            if self._QSArgs.MultiMapping:
                df = df.groupby(axis=0, level=[0, 1], as_index=True).apply(lambda s: s.tolist())
            elif df.index.duplicated().any():
                raise __QS_Error__(
                    f"算子 '{self.Name}' 的数据无法保证唯一性, 可以尝试将 '多重映射' 参数取值调整为 True")
            return df.unstack().reindex(index=dts, columns=ids)
        raise __QS_Error__(f"不支持的返回格式: {df}")

    def _QS_partitionSectionIDs(self, section_ids):
        SectionIdx = []  # [([ID], [idx])]
        for i, iIDs in enumerate(section_ids):
            for jIDs, jIdx in SectionIdx:
                if iIDs == jIDs:
                    jIdx.append(i)
                    break
            else:
                SectionIdx.append((iIDs, [i]))
        return SectionIdx

    def _QS_Compound2Frame(self, descriptor_data, compound_type_list):
        if not any(compound_type_list): return descriptor_data
        Data = []
        for i in range(descriptor_data.shape[1]):
            iCompoundType = compound_type_list[i]
            if not iCompoundType:
                Data.append(descriptor_data.iloc[:, i:i + 1])
            else:
                iData = descriptor_data.iloc[:, i].values
                DefaultData = np.array([None], dtype="O")
                DefaultData[0] = (None,) * len(iCompoundType)
                DefaultData = DefaultData.repeat(iData.shape[0], axis=0)
                iData = np.where(pd.notnull(iData), iData, DefaultData)
                iDataType = np.dtype([(iCol, float if iType == "double" else "O") for iCol, iType in iCompoundType])
                iData = iData.astype(iDataType)
                iData = pd.DataFrame(
                    {iName: pd.Series(iData[iName], index=descriptor_data.index) for iName, iDType in iCompoundType})
                Data.append(iData)
        return pd.concat(Data, axis=1, keys=descriptor_data.columns.tolist())

    def calculate(self, f: Factor, idt: dt.datetime | List[dt.datetime], iid: str | List[str], x: list, args: dict):
        """算子的运算逻辑实现

        Args:
            f: 该算子所属的因子对象
            idt: 当前待计算的时点
            iid: 当前待计算的 ID
            x: 描述子当期的数据
            args: 计算需要的附加参数, 来自于算子和因子对象的 ModelArgs, {参数名: 参数值}
        
        Returns:
            在时点 idt, ID 为 iid 的因子值
        """
        raise NotImplementedError

    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None, section_ids=None):
        raise NotImplementedError

    def __call__(self, *x: Factor, factor_args: dict = {}, **kwargs) -> Factor:
        """将算子作用在若干个因子对象上以产生新的因子

        Args:
            x: 作用于的因子对象
            factor_args: 创建新因子时传递个它的参数集
            kwargs: 创建新因子时传递给它的其他入参
        
        Returns:
            算子作用后产生的新因子对象
        """
        raise NotImplementedError


class PointOperator(FactorOperator):
    """单点算子, 运算只依赖于被依赖因子在单个时点和单个ID的因子值"""
    
    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Point"] = Field(default="Point", title="算子类型", frozen=True)
        Name: str = Field(default="PointOperator", title="名称", frozen=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        IDMode: Literal["单ID", "多ID"] = Field(default="单ID", title="运算ID", frozen=True)

    def calculate(self, f: Factor, idt: dt.datetime | List[dt.datetime], iid: str | List[str], x: list, args: dict):
        """算子的运算逻辑实现

        Args:
            f: 该算子所属的因子对象
            idt: 当前待计算的时点, 如果 DTMode 为多时点, 则该值为时点序列 list[datetime]
            iid: 当前待计算的 ID, 如果 IDMode 为多ID, 则该值为 ID 序列 list[str], 注意并发时 iid 并不一定是全截面
            x: 描述子当期的数据, [单个描述子值 or array]
                * 如果 DTMode 为单时点, IDMode 为单ID, 那么 x 元素为单个描述子值, 同时方法需返回单个元素
                * 如果 DTMode 为单时点, IDMode 为多ID, 那么 x 元素为 array(shape=(len(iid), )), 同时方法需返回 array(shape=(len(iid), ))
                * 如果 DTMode 为多时点, IDMode 为单ID, 那么 x 元素为 array(shape=(len(idt), )), 同时方法需返回 array(shape=(len(idt), ))
                * 如果 DTMode 为多时点, IDMode 为多ID, 那么 x 元素为 array(shape=(len(idt), len(iid))), 同时方法需返回 array(shape=(len(idt), len(iid)))
            args: 计算需要附加的模型参数, 来自于算子和因子对象的 ModelArgs, {参数名: 参数值}
        
        Returns:
            在时点 idt, ID 为 iid 的因子值
        """
        raise NotImplementedError

    def __call__(self, *x: Factor, factor_args: dict = {}, **kwargs) -> "PointOperation":
        Operator = self._QS_validate(*x, **kwargs.pop("operator_kwargs", {}))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return PointOperation(descriptors=Descriptors, args={"Operator": Operator, **factor_args}, **kwargs)

    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, ModelArgs):
        if self._QSArgs.DataType == 'double':
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else:
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        if (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '多ID'):
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is not None:
                descriptor_data = [iData[CalcMask] for iData in descriptor_data]
                dts = np.array(dts, dtype="O")[CalcMask].tolist()
                iStdData = self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
                StdData[CalcMask, :] = iStdData
            else:
                return self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '单ID'):
            CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
            if CalcDTs: CalcDTs = set(CalcDTs)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                for j, jID in enumerate(ids):
                    StdData[i, j] = self.calculate(factor, iDT, jID, [iData[i, j] for iData in descriptor_data], ModelArgs)
        elif (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '单ID'):
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is None:
                for j, jID in enumerate(ids):
                    StdData[:, j] = self.calculate(factor, dts, jID, [iData[:, j] for iData in descriptor_data],
                                                   ModelArgs)
            else:
                dts = np.array(dts, dtype="O")[CalcMask].tolist()
                for j, jID in enumerate(ids):
                    StdData[CalcMask, j] = self.calculate(factor, dts, jID,
                                                          [iData[CalcMask, j] for iData in descriptor_data], ModelArgs)
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '多ID'):
            CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
            if CalcDTs: CalcDTs = set(CalcDTs)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                StdData[i, :] = self.calculate(factor, iDT, ids, [iData[i, :] for iData in descriptor_data], ModelArgs)
        return StdData

    def _calcDataPandas(self, factor, ids, dts, descriptor_data, ModelArgs):
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is not None:
            descriptor_data = Panel(
                {f"d{i}": descriptor_data[i][CalcMask] for i in range(len(descriptor_data))}).to_frame(
                filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        else:
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).to_frame(
                filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        descriptor_data = self._QS_Compound2Frame(descriptor_data, self._QSArgs.DescriptorCompoundType)
        if self._QSArgs.ExpandDescriptors:
            descriptor_data, iOtherData = descriptor_data.iloc[:, self._QSArgs.ExpandDescriptors], descriptor_data.loc[:, descriptor_data.columns.difference(descriptor_data.columns[self._QSArgs.ExpandDescriptors])]
            descriptor_data = expandListElementDataFrame(descriptor_data, expand_index=True)
            descriptor_data = descriptor_data.set_index(descriptor_data.columns[:2].tolist())
            if not iOtherData.empty:
                descriptor_data.index, iOtherData.index = descriptor_data.index.rename(
                    ("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                descriptor_data = pd.merge(descriptor_data, iOtherData, how="left", left_index=True, right_index=True)
            descriptor_data = descriptor_data.sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        if self._QSArgs.CompoundType:
            CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
        else:
            CompoundCols = None
        if (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '多ID'):
            DTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
            StdData = self.calculate(factor, DTs, ids, descriptor_data, ModelArgs)
            return self._QS_adjOutputPandas(StdData, CompoundCols, DTs, ids).reindex(index=dts)
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '单ID'):
            if self._QSArgs.DataType == 'double':
                StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
            else:
                StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
            CalcDTs = (set(np.array(dts, dtype="O")[CalcMask]) if CalcMask is not None else None)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                for j, jID in enumerate(ids):
                    iStdData = self.calculate(factor, iDT, jID, descriptor_data.loc[iDT].loc[jID], ModelArgs)
                    if isinstance(iStdData, pd.DataFrame):
                        iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                    elif isinstance(iStdData, pd.Series):
                        iStdData = tuple(iStdData.reindex(index=CompoundCols))
                    StdData[i, j] = iStdData
            return pd.DataFrame(StdData, index=dts, columns=ids)
        elif (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '单ID'):
            descriptor_data = descriptor_data.swaplevel(axis=0)
            CalcDTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
            StdData = []
            for j, jID in enumerate(ids):
                iStdData = self.calculate(factor, CalcDTs, jID, descriptor_data.loc[jID], ModelArgs)
                if isinstance(iStdData, pd.DataFrame):
                    iStdData["_QS_ID"] = jID
                elif isinstance(iStdData, pd.Series):
                    iStdData = iStdData.to_frame("_QS_Factor")
                    iStdData["_QS_ID"] = jID
                else:
                    raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                StdData.append(iStdData)
            StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_ID"], append=True)
            if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
            return self._QS_adjOutputPandas(StdData, CompoundCols, CalcDTs, ids).reindex(index=dts)
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '多ID'):
            CalcDTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
            StdData = []
            for i, iDT in enumerate(CalcDTs):
                iStdData = self.calculate(factor, iDT, ids, descriptor_data.loc[iDT], ModelArgs)
                if isinstance(iStdData, pd.DataFrame):
                    iStdData["_QS_DT"] = iDT
                elif isinstance(iStdData, pd.Series):
                    iStdData = iStdData.to_frame("_QS_Factor")
                    iStdData["_QS_DT"] = iDT
                else:
                    raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                StdData.append(iStdData)
            StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_DT"], append=True)
            StdData = StdData.swaplevel(axis=0)
            if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
            return self._QS_adjOutputPandas(StdData, CompoundCols, CalcDTs, ids).reindex(index=dts)

    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None, section_ids=None):
        ModelArgs = dict(self._QSArgs.ModelArgs)
        ModelArgs.update(factor._QSArgs.ModelArgs)
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(factor, ids, dts, descriptor_data, ModelArgs)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, ModelArgs)


class TimeOperator(FactorOperator):
    """时序算子, 运算依赖于被依赖因子在过去若干个时点以及单个ID的因子值"""
    
    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Time"] = Field(default="Time", title="算子类型", frozen=True)
        Name: str = Field(default="TimeOperator", title="名称", frozen=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        IDMode: Literal["单ID", "多ID"] = Field(default="单ID", title="运算ID", frozen=True)
        LookBack: List[int] = Field(default=[], title="回溯期数", frozen=True, description="描述子向前回溯的时点数(不包括当前时点)")
        LookBackMode: List[Literal["滚动窗口", "扩张窗口"]] = Field(default=[], title="回溯模式", description="描述子的回溯模式", frozen=True)
        StartDT: List[Optional[dt.datetime]] = Field(default=[], title="起始时点", frozen=True, description="扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数")
        iInitFactor: int = Field(default=-1, title="起始因子", ge=-1, frozen=True)
        
        def __init__(self, /, **data):
            Arity = data.get("Arity", 0)
            if Arity is None: Arity = 0
            if not data.get("LookBack", []): data["LookBack"] = [0] * Arity
            if not data.get("LookBackMode", []): data["LookBackMode"] = ["滚动窗口"] * Arity
            if not data.get("StartDT", []): data["StartDT"] = [None] * Arity
            return super().__init__(**data)
        
        def model_post_init(self, context: Any, /) -> None:
            if self.Arity is not None:
                if self.Arity != len(self.LookBack):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBack({self.__pydantic_fields__['LookBack'].title}): {self.LookBack} 的长度不一致!")
                if self.Arity != len(self.LookBackMode):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBackMode({self.__pydantic_fields__['LookBackMode'].title}): {self.LookBackMode} 的长度不一致!")
                if self.Arity != len(self.StartDT):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 StartDT({self.__pydantic_fields__['StartDT'].title}): {self.StartDT} 的长度不一致!")
                if self.iInitFactor >= self.Arity:
                    raise __QS_Error__(f"算子{self.Name}的 iInitFactor({self.__pydantic_fields__['iInitFactor'].title}): {self.iInitFactor} 超出了 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity}!")
            return super().model_post_init(context)
    
    def calculate(self, f: Factor, idt: dt.datetime | List[dt.datetime], iid: str | List[str], x: list, args: dict):
        """算子的运算逻辑实现

        Args:
            f: 该算子所属的因子对象
            idt: 当前待计算的时点, 如果 DTMode 为多时点，则该值为时点序列 list[datetime]
            iid: 当前待计算的 ID, 如果 IDMode 为多ID, 则该值为 ID 序列 list[str], 注意并发时 iid 并不一定是全截面
            x: 描述子当期的数据, [array]
                * 如果 DTMode 为单时点, IDMode 为单ID, 那么 x 的第 i 个元素为 array(shape=(LookBack[i]+1, )), 同时方法需返回单个元素
                * 如果 DTMode 为单时点, IDMode 为多ID, 那么 x 的第 i 个元素为 array(shape=(LookBack[i]+1, len(iid))), 同时方法需返回 array(shape=(len(iid), ))
                * 如果 DTMode 为多时点, IDMode 为单ID, 那么 x 的第 i 个元素为 array(shape=(LookBack[i]+len(idt), )), 同时方法需返回返回 array(shape=(nDate,))
                * 如果 DTMode 为多时点, IDMode 为多ID, 那么 x 的第 i 个元素为 array(shape=(LookBack[i]+len(idt), len(iid))), 同时方法需返回 array(shape=(len(idt), len(iid)))
            args: 计算需要附加的模型参数, 来自于算子和因子对象的 ModelArgs, {参数名: 参数值}
        
        Returns:
            在时点 idt, ID 为 iid 的因子值
        """
        raise NotImplementedError

    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> "TimeOperation":
        Operator = self._QS_validate(*x, **kwargs.pop("operator_kwargs", {}))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return TimeOperation(descriptors=Descriptors, args={"Operator": Operator, **factor_args}, **kwargs)
    
    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData):
        if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
            CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
            if CalcDTs: CalcDTs = set(CalcDTs)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                for j, jID in enumerate(ids):
                    x = []
                    for k, kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i, j])
                    StdData[iStartIdx+i, j] = self.calculate(factor, iDTs, jID, x, ModelArgs)
        elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
            CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
            if CalcDTs: CalcDTs = set(CalcDTs)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                x = []
                for k,kDescriptorData in enumerate(descriptor_data):
                    kStartInd, kLen = StartIndAndLen[k]
                    x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                StdData[iStartIdx+i, :] = self.calculate(factor, iDTs, ids, x, ModelArgs)
        elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
            for j, jID in enumerate(ids):
                StdData[iStartIdx:, j] = self.calculate(factor, DTRuler, jID, [kDescriptorData[:, j] for kDescriptorData in descriptor_data], ModelArgs)
            StdData = StdData[iStartIdx:, :]
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask, :] = None
            return StdData
        else:
            StdData = self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)[iStartIdx:, :]
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask, :] = None
            return StdData
        return StdData[iStartIdx:, :]
    
    def _calcDataPandas(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData):
        StdData = pd.DataFrame(StdData, columns=ids, index=DTRuler[-StdData.shape[0]:])
        descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).loc[:, DTRuler].to_frame(filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        descriptor_data = self._QS_Compound2Frame(descriptor_data, self._QSArgs.DescriptorCompoundType)
        if self._QSArgs.ExpandDescriptors:
            descriptor_data, iOtherData = descriptor_data.iloc[:, self._QSArgs.ExpandDescriptors], descriptor_data.loc[:, descriptor_data.columns.difference(descriptor_data.columns[self._QSArgs.ExpandDescriptors])]
            descriptor_data = expandListElementDataFrame(descriptor_data, expand_index=True)
            descriptor_data = descriptor_data.set_index(descriptor_data.columns[:2].tolist())
            if not iOtherData.empty:
                descriptor_data.index, iOtherData.index = descriptor_data.index.rename(("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                descriptor_data = pd.merge(descriptor_data, iOtherData, how="left", left_index=True, right_index=True)
            descriptor_data = descriptor_data.sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        if self._QSArgs.CompoundType:
            CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
        else:
            CompoundCols = None
        if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
            StdData = StdData.values
            descriptor_data = descriptor_data.swaplevel(axis=0)
            CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
            if CalcDTs: CalcDTs = set(CalcDTs)
            for j, jID in enumerate(ids):
                jDescriptorData = descriptor_data.loc[jID]
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                    iStdData = self.calculate(factor, iDTs, jID, jDescriptorData.loc[iDTs], ModelArgs)
                    if isinstance(iStdData, pd.DataFrame):
                        iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                    elif isinstance(iStdData, pd.Series):
                        iStdData = tuple(iStdData.reindex(index=CompoundCols))
                    StdData[iStartIdx + i, j] = iStdData
            return pd.DataFrame(StdData[iStartIdx:, :], index=dts, columns=ids)
        elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
            CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
            if CalcDTs: CalcDTs = set(CalcDTs)
            StdData = []
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                iStdData = self.calculate(factor, iDTs, ids, descriptor_data.loc[iDTs], ModelArgs)
                if isinstance(iStdData, pd.DataFrame):
                    iStdData["_QS_DT"] = iDT
                elif isinstance(iStdData, pd.Series):
                    iStdData = iStdData.to_frame("_QS_Factor")
                    iStdData["_QS_DT"] = iDT
                else:
                    raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                StdData.append(iStdData)
            StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_DT"], append=True)
            StdData = StdData.swaplevel(axis=0)
            if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
            if CalcDTs is not None:
                return self._QS_adjOutputPandas(StdData, CompoundCols, sorted(CalcDTs), ids).reindex(index=dts)
            else:
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
        elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
            descriptor_data = descriptor_data.swaplevel(axis=0)
            StdData = []
            for j, jID in enumerate(ids):
                iStdData = self.calculate(factor, DTRuler, jID, descriptor_data.loc[jID], ModelArgs)
                if isinstance(iStdData, pd.DataFrame):
                    iStdData["_QS_ID"] = jID
                elif isinstance(iStdData, pd.Series):
                    iStdData = iStdData.to_frame("_QS_Factor")
                    iStdData["_QS_ID"] = jID
                else:
                    raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                StdData.append(iStdData)
            StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_ID"], append=True)
            if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
            StdData = self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask] = None
            return StdData
        else:
            StdData = self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)
            StdData = self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask] = None
            return StdData
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None, section_ids=None):
        if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        if dt_ruler is None: dt_ruler = dts
        StartIdx, EndIdx = np.searchsorted(dt_ruler, dts[0], side="left"), np.searchsorted(dt_ruler, dts[-1], side="right")
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1# StartIndAndLen: [(开始位置, 数据长度)], MaxLookBack: 最大回溯期, MaxLen: 最大数据长度
        for i in range(len(descriptor_data)):
            iLookBack = self._QSArgs.LookBack[i]
            if (self._QSArgs.LookBackMode[i]=="滚动窗口") or (self._QSArgs.StartDT[i] is None):
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                iLookBack = max(0, StartIdx - np.searchsorted(dt_ruler, self._QSArgs.StartDT[i], side="left") + iLookBack)
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        iStartIdx = 0
        if self._QSArgs.iInitFactor>=0:# 自身回溯
            StdData = np.r_[descriptor_data[self._QSArgs.iInitFactor], StdData]
            iStartIdx = descriptor_data[self._QSArgs.iInitFactor].shape[0]
            descriptor_data[self._QSArgs.iInitFactor] = StdData
        if StartIdx >= MaxLookBack: DTRuler = dt_ruler[StartIdx-MaxLookBack:EndIdx]
        else: DTRuler = [None] * (MaxLookBack - StartIdx) + dt_ruler[:EndIdx]
        ModelArgs = self._QSArgs.ModelArgs
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData)


class SectionOperator(FactorOperator):
    """截面算子, 运算依赖于被依赖因子在过去若干个时点以及整个截面的因子值"""

    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Section"] = Field(default="Section", title="算子类型", frozen=True)
        Name: str = Field(default="SectionOperator", title="名称", frozen=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        OutputMode: Literal["全截面", "单ID"] = Field(default="全截面", title="输出形式", frozen=True)
        DescriptorSection: List[Optional[List[str]]] = Field(default=[], title="描述子截面", frozen=True, description="None 表示该描述子和当前因子的截面一致")
        
        def __init__(self, /, **data):
            Arity = data.get("Arity", 0)
            if Arity is None: Arity = 0
            if not data.get("DescriptorSection", []): data["DescriptorSection"] = [None] * Arity
            return super().__init__(**data)
        
        def model_post_init(self, context: Any, /) -> None:
            if (self.Arity is not None) and (self.Arity != len(self.DescriptorSection)):
                raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 DescriptorSection({self.__pydantic_fields__['DescriptorSection'].title}): {self.DescriptorSection} 的长度不一致!")
            return super().model_post_init(context)
    
    def calculate(self, f: Factor, idt: dt.datetime | List[dt.datetime], iid: str | List[str], x: list, args: dict):
        """算子的运算逻辑实现

        Args:
            f: 该算子所属的因子对象
            idt: 当前待计算的时点, 如果 DTMode 为多时点, 则该值为时点序列 list[datetime]
            iid: 当前待计算的 ID, 如果 OutputMode 为全截面, 则该值为 ID 序列 list[str], 该序列在并发时也是全体截面 ID
            x: 描述子当期的数据, [array]
                * 如果 DTMode 为单时点, 那么 x 元素为 array(shape=(len(iid), )), 如果输出形式为全截面返回 array(shape=(len(iid), )), 否则返回单个值
                * 如果 DTMode 为多时点, 那么 x 元素为 array(shape=(len(idt), len(iid))), 如果输出形式为全截面返回 array(shape=(len(idt), len(iid))), 否则返回 array(shape=(len(idt), ))
            args: 计算需要附加的模型参数, 来自于算子和因子对象的 ModelArgs, {参数名: 参数值}
        
        Returns:
            在时点 idt, ID 为 iid 的因子值
        """
        raise NotImplementedError

    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> "SectionOperation":
        Operator = self._QS_validate(*x, **kwargs.pop("operator_kwargs", {}))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return SectionOperation(descriptors=Descriptors, args={"Operator": Operator, **factor_args}, **kwargs)
        
    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, SectionIDs, ModelArgs):
        if self._QSArgs.DataType=="double": StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=np.nan, dtype="float")
        else: StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=None, dtype="O")
        if self._QSArgs.OutputMode=="全截面":
            if self._QSArgs.DTMode=="单时点":
                CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
                if CalcDTs: CalcDTs = set(CalcDTs)
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    StdData[i, :] = self.calculate(factor, iDT, SectionIDs, [kDescriptorData[i] for kDescriptorData in descriptor_data], ModelArgs)
            else:
                CalcMask = factor._QS_getCalcDTs(dts, mask=True)
                if CalcMask is not None:
                    descriptor_data = [iData[CalcMask] for iData in descriptor_data]
                    dts = np.array(dts, dtype="O")[CalcMask].tolist()
                    iStdData = self.calculate(factor, dts, SectionIDs, descriptor_data, ModelArgs)
                    StdData[CalcMask, :] = iStdData
                else:
                    StdData = self.calculate(factor, dts, SectionIDs, descriptor_data, ModelArgs)
        else:
            if self._QSArgs.DTMode=="单时点":
                CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
                if CalcDTs: CalcDTs = set(CalcDTs)
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    x = [kDescriptorData[i] for kDescriptorData in descriptor_data]
                    for j, jID in enumerate(SectionIDs):
                        StdData[i, j] = self.calculate(factor, iDT, jID, x, ModelArgs)
            else:
                CalcMask = factor._QS_getCalcDTs(dts, mask=True)
                if CalcMask is not None:
                    descriptor_data = [iData[CalcMask] for iData in descriptor_data]
                    dts = np.array(dts, dtype="O")[CalcMask].tolist()
                    for j, jID in enumerate(SectionIDs):
                        StdData[CalcMask, j] = self.calculate(factor, dts, jID, descriptor_data, ModelArgs)
                else:
                    for j, jID in enumerate(SectionIDs):
                        StdData[:, j] = self.calculate(factor, dts, jID, descriptor_data, ModelArgs)
        return pd.DataFrame(StdData, columns=SectionIDs).reindex(columns=ids).values
    
    def _calcDataPandas(self, factor, ids, dts, descriptor_data, SectionIDs, ModelArgs):
        SectionIdx = self._QS_partitionSectionIDs(factor._QSArgs.DescriptorSection)
        DescriptorData = []
        DescriptorCompoundType = ([None]*len(descriptor_data) if not self._QSArgs.DescriptorCompoundType else self._QSArgs.DescriptorCompoundType)
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        for iSectionIDs, iIdx in SectionIdx:
            if CalcMask is not None:
                iDescriptorData = Panel({f"d{i}": descriptor_data[i][CalcMask] for i in range(len(descriptor_data)) if i in iIdx}).to_frame(filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
            else:
                iDescriptorData = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data)) if i in iIdx}).to_frame(filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
            iDescriptorData = self._QS_Compound2Frame(iDescriptorData, [DescriptorCompoundType[i] for i in range(len(descriptor_data)) if i in iIdx])
            iExpandDescriptors = sorted((f"d{i}" for i in set(self._QSArgs.ExpandDescriptors).intersection(iIdx)), key=lambda x: int(x[1:]))
            if iExpandDescriptors:
                iDescriptorData, iOtherData = iDescriptorData.loc[:, iExpandDescriptors], iDescriptorData.loc[:, iDescriptorData.columns.difference(iExpandDescriptors)]
                iDescriptorData = expandListElementDataFrame(iDescriptorData, expand_index=True)
                iDescriptorData = iDescriptorData.set_index(iDescriptorData.columns[:2].tolist())
                if not iOtherData.empty:
                    iDescriptorData.index, iOtherData.index = iDescriptorData.index.rename(("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                    iDescriptorData = pd.merge(iDescriptorData, iOtherData, how="left", left_index=True, right_index=True)
            iDescriptorData = iDescriptorData.sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
            DescriptorData.append(iDescriptorData)
        descriptor_data, DescriptorData = DescriptorData, None
        if self._QSArgs.CompoundType:
            CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
        else:
            CompoundCols = None
        if self._QSArgs.OutputMode=="全截面":
            CalcDTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
            if self._QSArgs.DTMode=="单时点":
                StdData = []
                for i, iDT in enumerate(CalcDTs):
                    iStdData = self.calculate(factor, iDT, SectionIDs, [iData.loc[iDT] for iData in descriptor_data], ModelArgs)
                    if isinstance(iStdData, pd.DataFrame):
                        iStdData["_QS_DT"] = iDT
                    elif isinstance(iStdData, pd.Series):
                        iStdData = iStdData.to_frame("_QS_Factor")
                        iStdData["_QS_DT"] = iDT
                    else:
                        raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                    StdData.append(iStdData)
                StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_DT"], append=True)
                StdData = StdData.swaplevel(axis=0)
                if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
                return self._QS_adjOutputPandas(StdData, CompoundCols, CalcDTs, ids).reindex(index=dts)
            else:
                StdData = self.calculate(factor, CalcDTs, SectionIDs, descriptor_data, ModelArgs)
                return self._QS_adjOutputPandas(StdData, CompoundCols, CalcDTs, ids).reindex(index=dts)
        else:
            if self._QSArgs.DTMode=="单时点":
                if self._QSArgs.DataType == "double": StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=np.nan, dtype="float")
                else: StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=None, dtype="O")
                CalcDTs = (set(np.array(dts, dtype="O")[CalcMask]) if CalcMask is not None else None)
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    iDescriptorData = [iData.loc[iDT] for iData in descriptor_data]
                    for j, jID in enumerate(SectionIDs):
                        iStdData = self.calculate(factor, iDT, jID, iDescriptorData, ModelArgs)
                        if isinstance(iStdData, pd.DataFrame):
                            iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                        elif isinstance(iStdData, pd.Series):
                            iStdData = tuple(iStdData.reindex(index=CompoundCols))
                        StdData[i, j] = iStdData
                return pd.DataFrame(StdData, index=dts, columns=SectionIDs).reindex(columns=ids)
            else:
                CalcDTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
                StdData = []
                for j, jID in enumerate(SectionIDs):
                    iStdData = self.calculate(factor, CalcDTs, jID, descriptor_data, ModelArgs)
                    if isinstance(iStdData, pd.DataFrame):
                        iStdData["_QS_ID"] = jID
                    elif isinstance(iStdData, pd.Series):
                        iStdData = iStdData.to_frame("_QS_Factor")
                        iStdData["_QS_ID"] = jID
                    else:
                        raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                    StdData.append(iStdData)
                StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_ID"], append=True)
                if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
                return self._QS_adjOutputPandas(StdData, CompoundCols, CalcDTs, ids).reindex(index=dts)
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None, section_ids=None):
        ModelArgs = self._QSArgs.ModelArgs
        if section_ids is None: section_ids = ids
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(factor, ids, dts, descriptor_data, section_ids, ModelArgs)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, section_ids, ModelArgs)


class PanelOperator(FactorOperator):
    """面板算子, 运算依赖于被依赖因子在单个时点和整个截面的因子值"""

    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Panel"] = Field(default="Panel", title="算子类型", frozen=True)
        Name: str = Field(default="PanelOperator", title="名称", frozen=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        OutputMode: Literal["全截面", "单ID"] = Field(default="全截面", title="输出形式", frozen=True)
        DescriptorSection: List[Optional[List[str]]] = Field(default=[], title="描述子截面", frozen=True, description="None 表示该描述子和当前因子的截面一致")
        LookBack: List[int] = Field(default=[], title="回溯期数", frozen=True, description="描述子向前回溯的时点数(不包括当前时点)")
        LookBackMode: List[Literal["滚动窗口", "扩张窗口"]] = Field(default=[], title="回溯模式", description="描述子的回溯模式", frozen=True)
        StartDT: List[Optional[dt.datetime]] = Field(default=[], title="起始时点", frozen=True, description="扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数")
        iInitFactor: int = Field(default=-1, title="起始因子", ge=-1, frozen=True)
        
        def __init__(self, /, **data):
            Arity = data.get("Arity", 0)
            if Arity is None: Arity = 0
            if not data.get("DescriptorSection", []): data["DescriptorSection"] = [None] * Arity
            if not data.get("LookBack", []): data["LookBack"] = [0] * Arity
            if not data.get("LookBackMode", []): data["LookBackMode"] = ["滚动窗口"] * Arity
            if not data.get("StartDT", []): data["StartDT"] = [None] * Arity
            return super().__init__(**data)
         
        def model_post_init(self, context: Any, /) -> None:
            if self.Arity is not None:
                if self.Arity != len(self.DescriptorSection):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 DescriptorSection({self.__pydantic_fields__['DescriptorSection'].title}): {self.DescriptorSection} 的长度不一致!")            
                if self.Arity != len(self.LookBack):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBack({self.__pydantic_fields__['LookBack'].title}): {self.LookBack} 的长度不一致!")
                if self.Arity != len(self.LookBackMode):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBackMode({self.__pydantic_fields__['LookBackMode'].title}): {self.LookBackMode} 的长度不一致!")
                if self.Arity != len(self.StartDT):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 StartDT({self.__pydantic_fields__['StartDT'].title}): {self.StartDT} 的长度不一致!")
                if self.iInitFactor >= self.Arity:
                    raise __QS_Error__(f"算子{self.Name}的 iInitFactor({self.__pydantic_fields__['iInitFactor'].title}): {self.iInitFactor} 超出了 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity}!")
            return super().model_post_init(context)
    
    def calculate(self, f: Factor, idt: dt.datetime | List[dt.datetime], iid: str | List[str], x: list, args: dict):
        """算子的运算逻辑实现

        Args:
            f: 该算子所属的因子对象
            idt: 当前待计算的时点, 如果 DTMode 为多时点, 则该值为时点序列 list[datetime]
            iid: 当前待计算的 ID, 如果 OutputMode 为全截面, 则该值为 ID 序列 list[str], 该序列在并发时也是全体截面 ID
            x: 描述子当期的数据, [array]
                * 如果 DTMode 为单时点, 那么 x 的第 i 个元素为 array(shape=(LookBack[i]+1, len(iid))), 如果输出形式为全截面返回 array(shape=(len(iid), )), 否则返回单个值
                * 如果 DTMode 为多时点, 那么 x 的第 i 个元素为 array(shape=(LookBack[i]+len(idt), len(iid))), 如果输出形式为全截面返回 array(shape=(len(idt), len(iid))), 否则返回 array(shape=(len(idt), ))
            args: 计算需要附加的模型参数, 来自于算子和因子对象的 ModelArgs, {参数名: 参数值}
        
        Returns:
            在时点 idt, ID 为 iid 的因子值
        """
        raise NotImplementedError

    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> "PanelOperation":
        Operator = self._QS_validate(*x, **kwargs.pop("operator_kwargs", {}))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return PanelOperation(descriptors=Descriptors, args={"Operator": Operator, **factor_args}, **kwargs)
    
    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData):
        if self._QSArgs.OutputMode=='全截面':
            if self._QSArgs.DTMode=='单时点':
                CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
                if CalcDTs: CalcDTs = set(CalcDTs)
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                    x = []
                    for k, kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                    StdData[iStartIdx+i, :] = self.calculate(factor, iDTs, SectionIDs, x, ModelArgs)
            else:
                StdData = self.calculate(factor, DTRuler, SectionIDs, descriptor_data, ModelArgs)
                CalcMask = factor._QS_getCalcDTs(dts, mask=True)
                if CalcMask is not None:
                    StdData[~CalcMask, :] = None
                return pd.DataFrame(StdData, columns=SectionIDs).reindex(columns=ids).values
        else:
            if self._QSArgs.DTMode=='单时点':
                CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
                if CalcDTs: CalcDTs = set(CalcDTs)
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                    x = []
                    for k, kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                    for j, jID in enumerate(SectionIDs):
                        StdData[iStartIdx+i, j] = self.calculate(factor, iDTs, jID, x, ModelArgs)
            else:
                for j, jID in enumerate(SectionIDs):
                    StdData[iStartIdx:, j] = self.calculate(factor, DTRuler, jID, descriptor_data, ModelArgs)
                StdData = StdData[iStartIdx:, :]
                CalcMask = factor._QS_getCalcDTs(dts, mask=True)
                if CalcMask is not None:
                    StdData[~CalcMask, :] = None
                return pd.DataFrame(StdData, columns=SectionIDs).reindex(columns=ids).values
        return pd.DataFrame(StdData[iStartIdx:, :], columns=SectionIDs).reindex(columns=ids).values
    
    def _calcDataPandas(self, factor, ids, dts, descriptor_data, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData):
        StdData = pd.DataFrame(StdData, columns=SectionIDs, index=DTRuler[-StdData.shape[0]:])
        SectionIdx = self._QS_partitionSectionIDs(factor._QSArgs.DescriptorSection)
        DescriptorData = []
        DescriptorCompoundType = ([None]*len(descriptor_data) if not self._QSArgs.DescriptorCompoundType else self._QSArgs.DescriptorCompoundType)
        for iSectionIDs, iIdx in SectionIdx:
            iDescriptorData = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data)) if i in iIdx}).loc[:, DTRuler].to_frame(filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
            iDescriptorData = self._QS_Compound2Frame(iDescriptorData, [DescriptorCompoundType[i] for i in range(len(descriptor_data)) if i in iIdx])
            iExpandDescriptors = sorted((f"d{i}" for i in set(self._QSArgs.ExpandDescriptors).intersection(iIdx)), key=lambda x: int(x[1:]))
            if iExpandDescriptors:
                iDescriptorData, iOtherData = iDescriptorData.loc[:, iExpandDescriptors], iDescriptorData.loc[:, iDescriptorData.columns.difference(iExpandDescriptors)]
                iDescriptorData = expandListElementDataFrame(iDescriptorData, expand_index=True)
                iDescriptorData = iDescriptorData.set_index(iDescriptorData.columns[:2].tolist())
                if not iOtherData.empty:
                    iDescriptorData.index, iOtherData.index = iDescriptorData.index.rename(("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                    iDescriptorData = pd.merge(iDescriptorData, iOtherData, how="left", left_index=True, right_index=True)
            iDescriptorData = iDescriptorData.sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
            DescriptorData.append(iDescriptorData)
        descriptor_data, DescriptorData = DescriptorData, None
        if self._QSArgs.CompoundType:
            CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
        else:
            CompoundCols = None
        if self._QSArgs.OutputMode=='全截面':
            if self._QSArgs.DTMode=='单时点':
                CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
                if CalcDTs: CalcDTs = set(CalcDTs)
                StdData = []
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
                    iStdData = self.calculate(factor, iDTs, SectionIDs, [iData.loc[iDTs] for iData in descriptor_data], ModelArgs)
                    if isinstance(iStdData, pd.DataFrame):
                        iStdData["_QS_DT"] = iDT
                    elif isinstance(iStdData, pd.Series):
                        iStdData = iStdData.to_frame("_QS_Factor")
                        iStdData["_QS_DT"] = iDT
                    else:
                        raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                    StdData.append(iStdData)
                StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_DT"], append=True)
                StdData = StdData.swaplevel(axis=0)
                if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
                if CalcDTs is not None:
                    return self._QS_adjOutputPandas(StdData, CompoundCols, sorted(CalcDTs), ids).reindex(index=dts)
                else:
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            else:
                StdData = self.calculate(factor, DTRuler, SectionIDs, descriptor_data, ModelArgs)
                StdData = self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
                CalcMask = factor._QS_getCalcDTs(dts, mask=True)
                if CalcMask is not None:
                    StdData[~CalcMask] = None
                return StdData
        else:
            if self._QSArgs.DTMode=='单时点':
                CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
                if CalcDTs: CalcDTs = set(CalcDTs)
                StdData = StdData.values
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
                    for j, jID in enumerate(SectionIDs):
                        iStdData = self.calculate(factor, iDTs, jID, [iData.loc[iDTs] for iData in descriptor_data], ModelArgs)
                        if isinstance(iStdData, pd.DataFrame):
                            iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                        elif isinstance(iStdData, pd.Series):
                            iStdData = tuple(iStdData.reindex(index=CompoundCols))
                        StdData[iStartIdx + i, j] = iStdData
                return pd.DataFrame(StdData[iStartIdx:, :], index=dts, columns=SectionIDs).reindex(columns=ids)
            else:
                descriptor_data = descriptor_data.swaplevel(axis=0)
                StdData = []
                for j, jID in enumerate(SectionIDs):
                    iStdData = self.calculate(factor, DTRuler, jID, descriptor_data, ModelArgs)
                    if isinstance(iStdData, pd.DataFrame):
                        iStdData["_QS_ID"] = jID
                    elif isinstance(iStdData, pd.Series):
                        iStdData = iStdData.to_frame("_QS_Factor")
                        iStdData["_QS_ID"] = jID
                    else:
                        raise __QS_Error__(f"不支持的返回格式: {iStdData}")
                    StdData.append(iStdData)
                StdData = pd.concat(StdData, axis=0, ignore_index=False).set_index(["_QS_ID"], append=True)
                if StdData.shape[1] == 1: StdData = StdData.iloc[:, 0]
                StdData = self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
                CalcMask = factor._QS_getCalcDTs(dts, mask=True)
                if CalcMask is not None:
                    StdData[~CalcMask] = None
                return StdData
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None, section_ids=None):
        if dt_ruler is None: dt_ruler = dts
        if section_ids is None: section_ids = ids
        if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(section_ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(section_ids)), fill_value=None, dtype='O')
        StartIdx, EndIdx = np.searchsorted(dt_ruler, dts[0], side="left"), np.searchsorted(dt_ruler, dts[-1], side="right")
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1# StartIndAndLen: [(开始位置, 数据长度)], MaxLookBack: 最大回溯期, MaxLen: 最大数据长度
        for i in range(len(descriptor_data)):
            iLookBack = self._QSArgs.LookBack[i]
            if (self._QSArgs.LookBackMode[i]=="滚动窗口") or (self._QSArgs.StartDT[i] is None):
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                iLookBack = max(0, StartIdx - np.searchsorted(dt_ruler, self._QSArgs.StartDT[i], side="left") + iLookBack)
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        iStartIdx = 0
        if self._QSArgs.iInitFactor>=0:# 自身回溯
            StdData = np.r_[descriptor_data[self._QSArgs.iInitFactor], StdData]
            iStartIdx = descriptor_data[self._QSArgs.iInitFactor].shape[0]
            descriptor_data[self._QSArgs.iInitFactor] = StdData
        if StartIdx >= MaxLookBack: DTRuler = dt_ruler[StartIdx-MaxLookBack:EndIdx]
        else: DTRuler = [None] * (MaxLookBack - StartIdx) + dt_ruler[:EndIdx]
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(factor, ids, dts, descriptor_data, DTRuler, section_ids, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, DTRuler, section_ids, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData)


FactorOperatorFunc = Callable[[Factor, dt.datetime | List[dt.datetime], str | List[str], list, dict], Any]

def makeFactorOperator(func: FactorOperatorFunc, operator_type: Literal['Point', 'Time', 'Section', 'Panel'], args:dict={}, **kwargs) -> FactorOperator:
    """算子工厂函数, 给定一个函数创建一个因子算子对象
    
    Args:
        func: 定义了算子运算逻辑的函数
        operator_type: 算子类型
        args: 创建算子对象时传递给它的参数集
        kwargs: 创建算子对象时传递给它的其他入参
    
    Returns:
        创建的因子算子对象
    """
    if not callable(func): raise __QS_Error__("func 必须是可调用对象!")
    if "Name" not in args: args = args | {"Name": func.__qualname__}
    if operator_type == "Point":
        FactorOperator = PointOperator(args=args, config_file=None, **kwargs)
    elif operator_type == "Time":
        FactorOperator = TimeOperator(args=args, config_file=None, **kwargs)
    elif operator_type == "Section":
        FactorOperator = SectionOperator(args=args, config_file=None, **kwargs)
    elif operator_type == "Panel":
        FactorOperator = PanelOperator(args=args, config_file=None, **kwargs)
    else:
        raise __QS_Error__(f"错误的因子算子类型: '{operator_type}', 必须为 'Point', 'Time', 'Section' 或者 'Panel'")
    FactorOperator.calculate = func
    return FactorOperator

def FactorOperatorized(operator_type: Literal['Point', 'Time', 'Section', 'Panel'], args:dict={}, **kwargs) -> Callable[[FactorOperatorFunc], FactorOperator]:
    """将函数转换成因子算子对象的装饰器
    
    Args:
        operator_type: 算子类型
        args: 创建算子对象时传递给它的参数集
        kwargs: 创建算子对象时传递给它的其他入参
    
    Returns:
        算子装饰器
    """
    return partial(makeFactorOperator, operator_type=operator_type, args=args, **kwargs)


class DerivativeFactor(Factor):
    """衍生因子"""

    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Operator: FactorOperator = Field(title="算子", frozen=True)
        ModelArgs: dict = Field(default={}, title="参数", frozen=True)
        Meta: dict = Field(default={}, title="元信息", frozen=False, exclude=True)
        CalcDTRuler: list[dt.datetime] = Field(default=[], title="计算时点标尺", frozen=True)

    def __init__(self, descriptors: List[Factor], args: dict={}, config_file: Optional[str]=None, **kwargs):
        self.UserData = {}
        if descriptors: kwargs.setdefault("logger", descriptors[0]._QS_Logger)
        Operator = args["Operator"]._QS_validate(*descriptors, **kwargs.pop("operator_kwargs", {}))
        args = {"Name": Operator._QSArgs.Name} | args | {"Operator": Operator}
        super().__init__(descriptors=descriptors, args=args, config_file=config_file, **kwargs)
        self._Operator = self._QSArgs.Operator
        self._QS_checkConsistency()

    # 检查因子定义的相容性
    def _QS_checkConsistency(self):
        pass
    
    # 获取描述子的截面ID
    def _QS_getDescriptorSectionIDs(self, i:int, context:FactorContext) -> List[str]:
        SectionIDs = context.NodeState[self.QSID]["section_ids"]
        # iDescriptor = self.Descriptors[i]
        if hasattr(self._Operator._QSArgs, "DescriptorSection") and (self._Operator._QSArgs.DescriptorSection[i] is not None):
            return self._Operator._QSArgs.DescriptorSection[i]
        else:
            return SectionIDs
        # elif iDescriptor._QSArgs.SectionIDs:
        #     return iDescriptor._QSArgs.SectionIDs
        # else:
        #     return context.DefaultSectionIDs

    @property
    def Operator(self) -> FactorOperator:
        """因子算子"""
        return self._Operator

    def getMetaData(self, key:Optional[str]=None) -> Union[Any, pd.Series]:
        DataType = self._Operator._QSArgs.DataType
        if key is None:
            return pd.Series({"DataType": DataType, **self._QSArgs.Meta})
        elif key == "DataType":
            return DataType
        else:
            return self._QSArgs.Meta.get(key, None)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        Cached = (context.FactorDataCache and self._QSArgs.CacheEnabled)
        if bwd_data_list:
            iSectionIDs = context.getID(self.QSID, [context.PID] if Cached else (local_context.PIDs or [context.PID]))
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not iSectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=iSectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                if self._Operator._QSArgs.InputFormat == "numpy":
                    StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=[iBwdData.values for iBwdData in bwd_data_list], dt_ruler=context.DTRuler, section_ids=iSectionIDs)
                    try:
                        StdData = pd.DataFrame(StdData, index=CalcDTs, columns=iSectionIDs)
                    except:
                        print("DEBUG")
                else:
                    StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=bwd_data_list, dt_ruler=context.DTRuler, section_ids=iSectionIDs)
            if Cached and (not StdData.empty):
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.FactorDataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids={context.PID: iSectionIDs}, pid=context.PID, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(CalcDTs[0], CalcDTs[-1]))
        if Cached:
            StdData = context.FactorDataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__("走到了不该走到的地方!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)


class PointOperation(DerivativeFactor):
    """基于单点算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: PointOperator = Field(title="算子", frozen=True)

    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        Cached = (context.FactorDataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs)
        return [FactorLocalContext(IDs=context.getID(self.QSID, ([context.PID] if Cached else (fwd_data.PIDs or [context.PID]))), DTs=CalcDTs, PIDs=fwd_data.PIDs)] * len(self._Descriptors), FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": CalcDTs})


class TimeOperation(DerivativeFactor):
    """基于时序算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: TimeOperator = Field(title="算子", frozen=True)
    
    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {})
        StartDT, EndDT = FactorState["dt_range"]
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
            if (self._Operator._QSArgs.LookBackMode[i]=="滚动窗口") or (self._Operator._QSArgs.StartDT[i] is None):# 滚动窗口模式或者未指定起始时点, 从当前位置回溯 LookBack[i] 期
                iStartIdx = StartIdx - self._Operator._QSArgs.LookBack[i]
            else:# 指定了起始时点, 以起始时点 StartDT[i] 的位置为准
                iStartIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left") - self._Operator._QSArgs.LookBack[i]
            if iStartIdx < 0: self._QS_Logger.warning("注意: 对于因子 '%s'(QSID: %s) 的描述子 '%s'(QSID: %s), 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, self.QSID, iDescriptor.Name, iDescriptor.QSID))
            iStartIdx = max(0, iStartIdx)
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
            else:
                iEndDT = EndDT
            InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"DTRange": (DTRuler[iStartIdx], iEndDT)}))
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        Cached = (context.FactorDataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        DTRuler = context.DTRuler
        StartIdx, EndIdx = DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1])
        iSectionIDs = context.getID(self.QSID, ([context.PID] if Cached else (fwd_data.PIDs or [context.PID])))
        FwdData = []
        for i in range(len(self._Descriptors)):
            if (self._Operator._QSArgs.LookBackMode[i]=="滚动窗口") or (self._Operator._QSArgs.StartDT[i] is None):
                iStartIdx, iEndIdx = StartIdx - self._Operator._QSArgs.LookBack[i], EndIdx
            else:
                iStartIdx, iEndIdx = np.searchsorted(DTRuler, max(self._Operator._QSArgs.StartDT[i], DTRuler[0]), side="left") - self._Operator._QSArgs.LookBack[i], EndIdx
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndIdx = StartIdx - 1
            iDTs = DTRuler[max(iStartIdx, 0):iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=iSectionIDs, DTs=iDTs, PIDs=fwd_data.PIDs))
        return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": CalcDTs})


class SectionOperation(DerivativeFactor):
    """基于截面算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: SectionOperator = Field(title="算子", frozen=True)
    
    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        for i, iDescriptor in enumerate(self._Descriptors):
            iSectionIDs = self._QS_getDescriptorSectionIDs(i, context=context)
            if iSectionIDs != InitData[i].SectionIDs:
                InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"SectionIDs": iSectionIDs}))
        if (len(context.PIDList) > 1) and (self.QSID not in context.Event):
            if os.name == "nt":
                context.Event[self.QSID] = (context.ExtraData["mp_manager"].Queue(), context.ExtraData["mp_manager"].Event())
            else:
                context.Event[self.QSID] = (Queue(), Event())
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        if context.FactorDataCache and self._QSArgs.CacheEnabled:
            DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        if context.FactorDataCache and self._QSArgs.CacheEnabled and (len(context.PIDList) > 1):
            PID = context.PID
            DTPartition = partitionList(CalcDTs, len(context.PIDList))
            iCalcDTs = DTPartition[context.PIDList.index(PID)]
        else:
            iCalcDTs = CalcDTs
        return [FactorLocalContext(IDs=self._QS_getDescriptorSectionIDs(i, context), DTs=CalcDTs, PIDs=context.PIDList) for i, iDescriptor in enumerate(self.Descriptors)], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": iCalcDTs, "TotalCalcDTs": CalcDTs})
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        if bwd_data_list:
            iSectionIDs = context.getID(self.QSID, pids=None)
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not iSectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=iSectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                if self._Operator._QSArgs.InputFormat == "numpy":
                    DescriptorData = [iBwdData.reindex(index=CalcDTs).values for iBwdData in bwd_data_list]
                    StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=iSectionIDs)
                    StdData = pd.DataFrame(StdData, index=CalcDTs, columns=iSectionIDs)
                else:
                    DescriptorData = [iBwdData.reindex(CalcDTs) for iBwdData in bwd_data_list]
                    StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=bwd_data_list, dt_ruler=context.DTRuler, section_ids=iSectionIDs)
            if context.FactorDataCache and self._QSArgs.CacheEnabled and (not StdData.empty):
                PIDIDs = context.NodeState[self.QSID]["pid_ids"]
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.FactorDataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(CalcDTs[0], CalcDTs[-1]))
        if len(context.PIDList) > 1:
            Sub2MainQueue, PIDEvent = context.Event[self.QSID]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
            if "TotalCalcDTs" in local_context.ExtraData:
                TotalCalcDTs = local_context.ExtraData["TotalCalcDTs"]
                context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(TotalCalcDTs[0], TotalCalcDTs[-1]))
        if context.FactorDataCache and self._QSArgs.CacheEnabled:
            StdData = context.FactorDataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__("走到了不该走到的地方!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)


class PanelOperation(DerivativeFactor):
    """基于面板算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: PanelOperator = Field(title="算子", frozen=True)
    
    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {})
        StartDT, EndDT = FactorState["dt_range"]
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
            if (self._Operator._QSArgs.LookBackMode[i]=="滚动窗口") or (self._Operator._QSArgs.StartDT[i] is None):# 未指定起始时点, 从当前位置回溯 LookBack[i] 期
                iStartIdx = StartIdx - self._Operator._QSArgs.LookBack[i]
            else:# 指定了起始时点, 以起始时点 StartDT[i] 的位置为准
                iStartIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left") - self._Operator._QSArgs.LookBack[i]
            if iStartIdx < 0: self._QS_Logger.warning("注意: 对于因子 '%s'(QSID: %s) 的描述子 '%s'(QSID: %s), 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, self.QSID, iDescriptor.Name, iDescriptor.QSID))
            iStartIdx = max(0, iStartIdx)
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
            else:
                iEndDT = EndDT
            InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"DTRange": (DTRuler[iStartIdx], iEndDT), "SectionIDs": self._QS_getDescriptorSectionIDs(i, context=context)}))
        if (len(context.PIDList) > 1) and (self.QSID not in context.Event):
            if os.name == "nt":
                context.Event[self.QSID] = (context.ExtraData["mp_manager"].Queue(), context.ExtraData["mp_manager"].Event())
            else:
                context.Event[self.QSID] = (Queue(), Event())
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        if context.FactorDataCache and self._QSArgs.CacheEnabled:
            DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        if context.FactorDataCache and self._QSArgs.CacheEnabled and (len(context.PIDList) > 1):
            PID = context.PID
            i = self._Operator._QSArgs.iInitFactor
            if (i >= 0) and (self._Operator._QSArgs.LookBackMode[i]=="扩张窗口") and (self._Operator._QSArgs.StartDT[i] is not None):
                ResponsibleCalcDTs = (CalcDTs if context.PIDList.index(PID)==0 else [])
            else:
                DTPartition = partitionList(CalcDTs, len(context.PIDList))
                ResponsibleCalcDTs = DTPartition[context.PIDList.index(PID)]
        else:
            ResponsibleCalcDTs = CalcDTs
        DTRuler = context.DTRuler
        StartIdx, EndIdx = DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1])
        if ResponsibleCalcDTs: ResponsibleStartIdx, ResponsibleEndIdx = DTRuler.index(ResponsibleCalcDTs[0]), DTRuler.index(ResponsibleCalcDTs[-1])
        else: ResponsibleStartIdx = ResponsibleEndIdx = StartIdx
        FwdData, DescriptorDTs = [], []
        for i, iDescriptor in enumerate(self._Descriptors):
            if (self._Operator._QSArgs.LookBackMode[i]=="滚动窗口") or (self._Operator._QSArgs.StartDT[i] is None):
                iStartIdx, iEndIdx = StartIdx - self._Operator._QSArgs.LookBack[i], EndIdx
                iResponsibleStartIdx, iResponsibleEndIdx = ResponsibleStartIdx - self._Operator._QSArgs.LookBack[i], ResponsibleEndIdx
            else:
                iStartIdx, iEndIdx = np.searchsorted(DTRuler, max(self._Operator._QSArgs.StartDT[i], DTRuler[0]), side="left") - self._Operator._QSArgs.LookBack[i], EndIdx
                iResponsibleStartIdx, iResponsibleEndIdx = iStartIdx, iEndIdx
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndIdx = StartIdx - 1
                iResponsibleEndIdx = ResponsibleStartIdx - 1
            iDTs = DTRuler[max(iStartIdx, 0):iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=self._QS_getDescriptorSectionIDs(i, context), DTs=iDTs, PIDs=context.PIDList))
            DescriptorDTs.append(DTRuler[max(iResponsibleStartIdx, 0):iResponsibleEndIdx+1])
        return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": ResponsibleCalcDTs, "TotalCalcDTs": CalcDTs, "DescriptorDTs": DescriptorDTs})
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        if bwd_data_list:
            iSectionIDs = context.getID(self.QSID, pids=None)
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not iSectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=iSectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                DescriptorDTs = local_context.ExtraData["DescriptorDTs"]
                if self._Operator._QSArgs.InputFormat == "numpy":
                    DescriptorData = [iBwdData.reindex(index=DescriptorDTs[i]).values for i, iBwdData in enumerate(bwd_data_list)]
                    StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=iSectionIDs)
                    StdData = pd.DataFrame(StdData, index=CalcDTs, columns=iSectionIDs)
                else:
                    DescriptorData = [iBwdData.reindex(index=DescriptorDTs[i]) for i, iBwdData in enumerate(bwd_data_list)]
                    StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=bwd_data_list, dt_ruler=context.DTRuler, section_ids=iSectionIDs)
            if context.FactorDataCache and self._QSArgs.CacheEnabled and (not StdData.empty):
                PIDIDs = context.NodeState[self.QSID]["pid_ids"]
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.FactorDataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(CalcDTs[0], CalcDTs[-1]))
        if len(context.PIDList) > 1:
            Sub2MainQueue, PIDEvent = context.Event[self.QSID]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
            if "TotalCalcDTs" in local_context.ExtraData:
                TotalCalcDTs = local_context.ExtraData["TotalCalcDTs"]
                context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(TotalCalcDTs[0], TotalCalcDTs[-1]))
        if context.FactorDataCache and self._QSArgs.CacheEnabled:
            StdData = context.FactorDataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__("走到了不该走到的地方!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)
