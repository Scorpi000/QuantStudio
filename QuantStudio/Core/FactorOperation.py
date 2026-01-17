# -*- coding: utf-8 -*-
"""因子运算"""
import gc
import datetime as dt
from functools import partial
from typing import Optional, Literal, List, Any, Tuple
from multiprocessing import Queue, Event

import pandas as pd
import numpy as np
from pydantic import Field

from QuantStudio.Core import __QS_Error__, __QS_Object__
from QuantStudio.Core.Factor import Factor, DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame
from QuantStudio.Tools.AuxiliaryFun import partitionList


class FactorOperator(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        OperatorType: Literal["Point", "Time", "Section", "Panel"] = Field(title="算子类型", frozen=True)
        Name: str = Field(default="FactorOperator", title="名称", frozen=True)
        ModelArgs: dict = Field(default={}, title="参数", frozen=True)
        Arity: int = Field(default=1, ge=1, label="入参数", frozen=True)
        DataType: Literal["double", "string", "object"] = Field(default="double",title="数据类型", frozen=True)
        Description: str = Field(default="", label="描述信息", frozen=False, exclude=True)
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

    def __init__(self, args={}, config_file=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._QS_CachedOperators = {}

    @property
    def Name(self):
        return self._QSArgs.Name

    def _QS_checkArity(self, *x):
        Arity = len(x)
        if Arity != self._QSArgs.Arity:
            return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 和指定的入参数 {self._QSArgs.Arity} 不符!")
        return (True, None)

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

    def _QS_getCalcDTs(self, factor, dts, mask=False):
        CalcDTs = factor._QSArgs.CalcDTRuler
        if CalcDTs:
            StartIdx, EndIdx = np.searchsorted(CalcDTs, dts[0], side="left"), np.searchsorted(CalcDTs, dts[-1],
                                                                                              side="right")
            CalcDTs = CalcDTs[StartIdx:EndIdx]
            if mask:
                return np.isin(dts, CalcDTs)
            else:
                return set(CalcDTs)
        else:
            return None

    def calculate(self, f: Factor, idt: dt.datetime | list[dt.datetime], iid: str | list[str], x: list, args: dict):
        """
        算子的运算逻辑实现
        :param f: 该算子所属的因子, 因子对象
        :param idt: 当前待计算的时点, 如果运算时点为多时点，则该值为 [时点]
        :param iid: 当前待计算的 ID, 如果运算ID为多ID，则该值为 [ID]
        :param x: 描述子当期的数据
        :param args: 计算需要的附加参数, {参数名: 参数值}
        :return: 在时点 idt, ID 为 iid 的因子值
        """
        raise NotImplementedError

    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None, section_ids=None):
        raise NotImplementedError

    def __call__(self, *x, factor_args: dict = {}, **kwargs):
        raise NotImplementedError


# 单点算子
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算时点为多时点，则该值为[时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [单个描述子值 or array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么 x 元素为单个描述子值, 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nID, )), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID,))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么 x 元素为 array(shape=(nDT, )), 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
class PointOperator(FactorOperator):
    """单点算子"""
    
    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Point"] = Field(default="Point", title="算子类型", frozen=True)
        Name: str = Field(default="PointOperator", title="名称", frozen=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        IDMode: Literal["单ID", "多ID"] = Field(default="单ID", title="运算ID", frozen=True)

    def __call__(self, *x, factor_args: dict = {}, **kwargs):
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return PointOperation(descriptors=Descriptors, args={"Operator": self, **factor_args}, **kwargs)

    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, ModelArgs):
        if self._QSArgs.DataType == 'double':
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else:
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        if (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '多ID'):
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
            if CalcMask is not None:
                descriptor_data = [iData[CalcMask] for iData in descriptor_data]
                dts = np.array(dts, dtype="O")[CalcMask].tolist()
                iStdData = self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
                StdData[CalcMask, :] = iStdData
            else:
                return self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '单ID'):
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                for j, jID in enumerate(ids):
                    StdData[i, j] = self.calculate(factor, iDT, jID, [iData[i, j] for iData in descriptor_data],
                                                   ModelArgs)
        elif (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '单ID'):
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
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
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                StdData[i, :] = self.calculate(factor, iDT, ids, [iData[i, :] for iData in descriptor_data], ModelArgs)
        return StdData

    def _calcDataPandas(self, factor, ids, dts, descriptor_data, ModelArgs):
        CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
        if CalcMask is not None:
            descriptor_data = Panel(
                {f"d{i}": descriptor_data[i][CalcMask] for i in range(len(descriptor_data))}).to_frame(
                filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        else:
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).to_frame(
                filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        descriptor_data = self._QS_Compound2Frame(descriptor_data, self._QSArgs.DescriptorCompoundType)
        if self._QSArgs.ExpandDescriptors:
            descriptor_data, iOtherData = descriptor_data.iloc[:, self._QSArgs.ExpandDescriptors], descriptor_data.loc[
                                                                                                   :,
                                                                                                   descriptor_data.columns.difference(
                                                                                                       descriptor_data.columns[
                                                                                                           self._QSArgs.ExpandDescriptors])]
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


# 时序算子
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数, )), 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数+nDT, )), 返回 array(shape=(nDate,))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数+nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
class TimeOperator(FactorOperator):
    """时序算子"""
    
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
            Arity = data.get("Arity", self.__pydantic_fields__["Arity"].default)
            if "LookBack" not in data: data["LookBack"] = [0] * Arity
            if "LookBackMode" not in data: data["LookBackMode"] = ["滚动窗口"] * Arity
            if "StartDT" not in data: data["StartDT"] = [None] * Arity
            return super().__init__(**data)
        
        def model_post_init(self, context: Any, /) -> None:
            if self.Arity != len(self.LookBack):
                raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBack({self.__pydantic_fields__['LookBack'].title}): {self.LookBack} 的长度不一致!")
            if self.Arity != len(self.LookBackMode):
                raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBackMode({self.__pydantic_fields__['LookBackMode'].title}): {self.LookBackMode} 的长度不一致!")
            if self.Arity != len(self.StartDT):
                raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 StartDT({self.__pydantic_fields__['StartDT'].title}): {self.StartDT} 的长度不一致!")
            if self.iInitFactor >= self.Arity:
                raise __QS_Error__(f"算子{self.Name}的 iInitFactor({self.__pydantic_fields__['iInitFactor'].title}): {self.iInitFactor} 超出了 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity}!")
            return super().model_post_init(context)
    
    def __call__(self, *x, factor_args:dict={}, **kwargs):
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return TimeOperation(descriptors=Descriptors, args={"Operator": self, **factor_args}, **kwargs)
    
    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData):
        if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask, :] = None
            return StdData
        else:
            StdData = self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)[iStartIdx:, :]
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
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
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask] = None
            return StdData
        else:
            StdData = self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)
            StdData = self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
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
            iLookBack = factor._QSArgs.LookBack[i]
            if (factor._QSArgs.LookBackMode[i]=="滚动窗口") or (factor._QSArgs.StartDT[i] is None):
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                iLookBack = max(0, StartIdx - np.searchsorted(dt_ruler, factor._QSArgs.StartDT[i], side="left"))
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        iStartIdx = 0
        if factor._QSArgs.iInitFactor>=0:# 自身回溯
            StdData = np.r_[descriptor_data[factor._QSArgs.iInitFactor], StdData]
            iStartIdx = descriptor_data[factor._QSArgs.iInitFactor].shape[0]
            descriptor_data[factor._QSArgs.iInitFactor] = StdData
        if StartIdx >= MaxLookBack: DTRuler = dt_ruler[StartIdx-MaxLookBack:EndIdx]
        else: DTRuler = [None] * (MaxLookBack - StartIdx) + dt_ruler[:EndIdx]
        ModelArgs = dict(self._QSArgs.ModelArgs)
        ModelArgs.update(factor._QSArgs.ModelArgs)
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData)




# 截面算子
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(nID, )), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class SectionOperator(FactorOperator):
    """截面算子"""
    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Section"] = Field(default="Section", title="算子类型", frozen=True)
        Name: str = Field(default="SectionOperator", title="名称", frozen=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        OutputMode: Literal["全截面", "单ID"] = Field(default="全截面", title="输出形式", frozen=True)
        DescriptorSection: List[Optional[List[str]]] = Field(default=[], title="描述子截面", frozen=True)
        
        def __init__(self, /, **data):
            Arity = data.get("Arity", self.__pydantic_fields__["Arity"].default)
            if "DescriptorSection" not in data: data["DescriptorSection"] = [None] * Arity
            return super().__init__(**data)
        
        def model_post_init(self, context: Any, /) -> None:
            if self.Arity != len(self.DescriptorSection):
                raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 DescriptorSection({self.__pydantic_fields__['DescriptorSection'].title}): {self.DescriptorSection} 的长度不一致!")
            return super().model_post_init(context)
    
    def __call__(self, *x, factor_args:dict={}, **kwargs):
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return SectionOperation(descriptors=Descriptors, args={"Operator": self, **factor_args}, **kwargs)
        
    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, SectionIDs, ModelArgs):
        if self._QSArgs.DataType=="double": StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=np.nan, dtype="float")
        else: StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=None, dtype="O")
        if self._QSArgs.OutputMode=="全截面":
            if self._QSArgs.DTMode=="单时点":
                CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    StdData[i, :] = self.calculate(factor, iDT, SectionIDs, [kDescriptorData[i] for kDescriptorData in descriptor_data], ModelArgs)
            else:
                CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
                if CalcMask is not None:
                    descriptor_data = [iData[CalcMask] for iData in descriptor_data]
                    dts = np.array(dts, dtype="O")[CalcMask].tolist()
                    iStdData = self.calculate(factor, dts, SectionIDs, descriptor_data, ModelArgs)
                    StdData[CalcMask, :] = iStdData
                else:
                    StdData = self.calculate(factor, dts, SectionIDs, descriptor_data, ModelArgs)
        else:
            if self._QSArgs.DTMode=="单时点":
                CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
                for i, iDT in enumerate(dts):
                    if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                    x = [kDescriptorData[i] for kDescriptorData in descriptor_data]
                    for j, jID in enumerate(SectionIDs):
                        StdData[i, j] = self.calculate(factor, iDT, jID, x, ModelArgs)
            else:
                CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
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
        CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
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
        ModelArgs = dict(self._QSArgs.ModelArgs)
        ModelArgs.update(factor._QSArgs.ModelArgs)
        if section_ids is None: section_ids = ids
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(factor, ids, dts, descriptor_data, section_ids, ModelArgs)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, section_ids, ModelArgs)


# 面板算子
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多日期，则该值为 [回溯期数]+[时点]
# iid: 当前待计算的 ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(回溯期数, nID)), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(回溯期数+nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class PanelOperator(FactorOperator):
    """面板算子"""
    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Panel"] = Field(default="Panel", title="算子类型", frozen=True)
        Name: str = Field(default="PanelOperator", title="名称", frozen=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        OutputMode: Literal["全截面", "单ID"] = Field(default="全截面", title="输出形式", frozen=True)
        DescriptorSection: List[Optional[List[str]]] = Field(default=[], title="描述子截面", frozen=True)
        LookBack: List[int] = Field(default=[], title="回溯期数", frozen=True, description="描述子向前回溯的时点数(不包括当前时点)")
        LookBackMode: List[Literal["滚动窗口", "扩张窗口"]] = Field(default=[], title="回溯模式", description="描述子的回溯模式", frozen=True)
        StartDT: List[Optional[dt.datetime]] = Field(default=[], title="起始时点", frozen=True, description="扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数")
        iInitFactor: int = Field(default=-1, title="起始因子", ge=-1, frozen=True)
        
        def __init__(self, /, **data):
            Arity = data.get("Arity", self.__pydantic_fields__["Arity"].default)
            if "DescriptorSection" not in data: data["DescriptorSection"] = [None] * Arity
            if "LookBack" not in data: data["LookBack"] = [0] * Arity
            if "LookBackMode" not in data: data["LookBackMode"] = ["滚动窗口"] * Arity
            if "StartDT" not in data: data["StartDT"] = [None] * Arity
            return super().__init__(**data)
         
        def model_post_init(self, context: Any, /) -> None:
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
    
    def __call__(self, *x, factor_args:dict={}, **kwargs):
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(data=iFactor, logger=self._QS_Logger)) for i, iFactor in enumerate(x)]
        return PanelOperation(descriptors=Descriptors, args={"Operator": self, **factor_args}, **kwargs)
    
    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData):
        if self._QSArgs.OutputMode=='全截面':
            if self._QSArgs.DTMode=='单时点':
                CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
                CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
                if CalcMask is not None:
                    StdData[~CalcMask, :] = None
                return pd.DataFrame(StdData, columns=SectionIDs).reindex(columns=ids).values
        else:
            if self._QSArgs.DTMode=='单时点':
                CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
                CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
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
                CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
                CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
                if CalcMask is not None:
                    StdData[~CalcMask] = None
                return StdData
        else:
            if self._QSArgs.DTMode=='单时点':
                CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
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
                CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
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
            iLookBack = factor._QSArgs.LookBack[i]
            if (factor._QSArgs.LookBackMode[i]=="滚动窗口") or (factor._QSArgs.StartDT[i] is None):
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                iLookBack = max(0, StartIdx - np.searchsorted(dt_ruler, factor._QSArgs.StartDT[i], side="left"))
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        iStartIdx = 0
        if factor._QSArgs.iInitFactor>=0:# 自身回溯
            StdData = np.r_[descriptor_data[factor._QSArgs.iInitFactor], StdData]
            iStartIdx = descriptor_data[factor._QSArgs.iInitFactor].shape[0]
            descriptor_data[factor._QSArgs.iInitFactor] = StdData
        if StartIdx >= MaxLookBack: DTRuler = dt_ruler[StartIdx-MaxLookBack:EndIdx]
        else: DTRuler = [None] * (MaxLookBack - StartIdx) + dt_ruler[:EndIdx]
        ModelArgs = dict(self._QSArgs.ModelArgs)
        ModelArgs.update(factor._QSArgs.ModelArgs)
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(factor, ids, dts, descriptor_data, DTRuler, section_ids, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, DTRuler, section_ids, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData)


# 算子工厂函数
# operator_type: 算子类型, 可选: 'Point', 'Time', 'Section', 'Panel'
# sys_args: 算子参数
def makeFactorOperator(func, operator_type, args={}, **kwargs):
    if not callable(func): raise __QS_Error__("func 必须是可调用对象!")
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


# 将函数转换成因子定义的装饰器
def FactorOperatorized(operator_type, args={}, **kwargs):
    return partial(makeFactorOperator, operator_type=operator_type, args=args, **kwargs)


class DerivativeFactor(Factor):
    """衍生因子"""

    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Operator: FactorOperator = Field(title="算子", frozen=True)
        ModelArgs: dict = Field(default={}, title="参数", frozen=True)
        Meta: dict = Field(default={}, title="元信息", frozen=False, exclude=True)
        CalcDTRuler: list[dt.datetime] = Field(default=[], title="计算时点标尺", frozen=True)

    def __init__(self, descriptors, args={}, config_file=None, **kwargs):
        self.UserData = {}
        if descriptors: kwargs.setdefault("logger", descriptors[0]._QS_Logger)
        super().__init__(descriptors=descriptors, args=args, config_file=config_file, **kwargs)
        self._Operator = self._QSArgs.Operator
        self._QS_checkConsistency()

    # 检查因子定义的相容性
    def _QS_checkConsistency(self):
        pass

    @property
    def Operator(self):
        return self._Operator

    def getMetaData(self, key=None):
        DataType = self._Operator._QSArgs.DataType
        if key is None:
            return {"DataType": DataType, **self._QSArgs.Meta}
        elif key == "DataType":
            return DataType
        else:
            return self._QSArgs.get(key, None)
        return None
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        DTs, IDs = local_context.DTs, local_context.IDs
        iSectionIDs = context.getID(self.QSID, [context.PID])
        if (not iSectionIDs) or (not DTs):
            StdData = pd.DataFrame(index=DTs, columns=iSectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
        else:
            if self._Operator._QSArgs.InputFormat == "numpy":
                StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=DTs, descriptor_data=[iBwdData.values for iBwdData in bwd_data_list], dt_ruler=context.DTRuler, section_ids=iSectionIDs)
                StdData = pd.DataFrame(StdData, index=DTs, columns=iSectionIDs)
            else:
                StdData = self._Operator.calcData(factor=self, ids=iSectionIDs, dts=DTs, descriptor_data=bwd_data_list, dt_ruler=context.DTRuler, section_ids=iSectionIDs)
        context.FactorDataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids={context.PID: iSectionIDs}, pid=context.PID, if_exists="append")
        context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(DTs[0], DTs[1]))
        return StdData.reindex(index=DTs, columns=StdData.columns.intersection(IDs)).sort_index(axis=1)
    
class PointOperation(DerivativeFactor):
    """单点运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: PointOperator = Field(title="算子", frozen=True)

    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTs = context.getDateTime(DTRange)
        if not DTs: return [], FactorLocalContext(DTs=[], IDs=IDs)
        return [FactorLocalContext(IDs=context.getID(self.QSID, [context.PID]), DTs=DTs)] * len(self.Deps), FactorLocalContext(IDs=fwd_data.IDs, DTs=DTs)


class TimeOperation(DerivativeFactor):
    """时序运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: TimeOperator = Field(title="算子", frozen=True)
    
    def init_compute(self, path: List[str], init_data: Any, context: FactorContext) -> List[Any]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {})
        StartDT, EndDT = FactorState["dt_range"]
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self.Deps):
            if self._Operator._QSArgs.StartDT[i] is None:# 未指定起始时点, 从当前位置回溯 LookBack[i] 期
                iStartIdx = StartIdx - self._Operator._QSArgs.LookBack[i]
            else:# 指定了起始时点, 以起始时点 StartDT[i] 的位置为准
                iStartIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
            if iStartIdx < 0: self._QS_Logger.warning("注意: 对于因子 '%s'(QSID: %s) 的描述子 '%s'(QSID: %s), 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, self.QSID, iDescriptor.Name, iDescriptor.QSID))
            iStartIdx = max(0, iStartIdx)
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
            else:
                iEndDT = EndDT
            InitData[i]["dt_range"] = (DTRuler[iStartIdx], iEndDT)
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTs = context.getDateTime(DTRange)
        if not DTs: return [], FactorLocalContext(DTs=[], IDs=IDs)
        DTRuler = context.DTRuler
        StartIdx, EndIdx = DTRuler.index(DTs[0]), DTRuler.index(DTs[-1])
        iSectionIDs = context.getID(self.QSID, [context.PID])
        FwdData = []
        for i, iDescriptor in enumerate(self.Deps):
            if (self._Operator._QSArgs.LookBackMode[i]=="滚动窗口") or (self._Operator._QSArgs.StartDT[i] is None):
                iStartIdx, iEndIdx = StartIdx - self._Operator._QSArgs.LookBack[i], EndIdx
            else:
                iStartIdx, iEndIdx = np.searchsorted(DTRuler, max(self._Operator._QSArgs.StartDT[i], DTRuler[0]), side="left"), EndIdx
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndIdx = StartIdx - 1
            iDTs = DTRuler[max(iStartIdx, 0):iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=iSectionIDs, DTs=iDTs))
        return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=DTs)


class SectionOperation(DerivativeFactor):
    """截面运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: SectionOperator = Field(title="算子", frozen=True)
    
    def init_compute(self, path: List[str], init_data: Any, context: FactorContext) -> List[Any]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        for i, iDescriptor in enumerate(self.Deps):
            if self._Operator._QSArgs.DescriptorSection[i] is not None:
                InitData[i]["section_ids"] = self._Operator._QSArgs.DescriptorSection[i]
        if (len(context.PIDList) > 1) and (self.QSID not in context.Event):
            context._Event[self.QSID] = (Queue(), Event())
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTs = context.getDateTime(DTRange)
        if not DTs: return [], FactorLocalContext(DTs=[], IDs=IDs)
        PID = context.PID
        DTPartition = partitionList(DTs, len(context.PIDList))
        iDTs = DTPartition[context.PIDList.index(PID)]
        if not iDTs:# 该进程未分配到计算任务
            return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        return [FactorLocalContext(IDs=context.getID(iDescriptor.QSID, pids=None), DTs=iDTs) for iDescriptor in self.Deps], FactorLocalContext(IDs=fwd_data.IDs, DTs=iDTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        DTs, IDs = local_context.DTs, local_context.IDs
        SectionIDs = context.getID(self.QSID, pids=None)
        if (not SectionIDs) or (not DTs):
            StdData = pd.DataFrame(index=DTs, columns=SectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
        else:
            if self._Operator._QSArgs.InputFormat == "numpy":
                StdData = self._Operator.calcData(factor=self, ids=SectionIDs, dts=DTs, descriptor_data=[iBwdData.values for iBwdData in bwd_data_list], dt_ruler=context.DTRuler, section_ids=SectionIDs)
                StdData = pd.DataFrame(StdData, index=DTs, columns=SectionIDs)
            else:
                StdData = self._Operator.calcData(factor=self, ids=SectionIDs, dts=DTs, descriptor_data=bwd_data_list, dt_ruler=context.DTRuler, section_ids=SectionIDs)
        PIDIDs = context.NodeState[self.QSID]["pid_ids"]
        context.FactorDataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append")
        context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(DTs[0], DTs[1]))
        if len(context.PIDList) > 1:
            Sub2MainQueue, PIDEvent = context.Event[self.QSID]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        return StdData.reindex(index=DTs, columns=IDs).sort_index(axis=1)


class PanelOperation(DerivativeFactor):
    """面板运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Operator: PanelOperator = Field(title="算子", frozen=True)
    
    def init_compute(self, path: List[str], init_data: Any, context: FactorContext) -> List[Any]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {})
        StartDT, EndDT = FactorState["dt_range"]
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self.Deps):
            if self._Operator._QSArgs.StartDT[i] is None:# 未指定起始时点, 从当前位置回溯 LookBack[i] 期
                iStartIdx = StartIdx - self._Operator._QSArgs.LookBack[i]
            else:# 指定了起始时点, 以起始时点 StartDT[i] 的位置为准
                iStartIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
            if iStartIdx < 0: self._QS_Logger.warning("注意: 对于因子 '%s'(QSID: %s) 的描述子 '%s'(QSID: %s), 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, self.QSID, iDescriptor.Name, iDescriptor.QSID))
            iStartIdx = max(0, iStartIdx)
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
            else:
                iEndDT = EndDT
            InitData[i]["dt_range"] = (DTRuler[iStartIdx], iEndDT)
            if self._Operator._QSArgs.DescriptorSection[i] is not None:
                InitData[i]["section_ids"] = self._Operator._QSArgs.DescriptorSection[i]
        if (len(context.PIDList) > 1) and (self.QSID not in context.Event):
            context._Event[self.QSID] = (Queue(), Event())
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTRange = context.FactorDataCache.getDTRange(self.QSID, DTRange)
        if DTRange is None: return [], FactorLocalContext(DTs=[], IDs=fwd_data.IDs)
        DTs = context.getDateTime(DTRange)
        if not DTs: return [], FactorLocalContext(DTs=[], IDs=IDs)
        DTRuler = context.DTRuler
        StartIdx, EndIdx = DTRuler.index(DTs[0]), DTRuler.index(DTs[-1])
        FwdData = []
        for i, iDescriptor in enumerate(self.Deps):
            if (self._Operator._QSArgs.LookBackMode[i]=="滚动窗口") or (self._Operator._QSArgs.StartDT[i] is None):
                iStartIdx, iEndIdx = StartIdx - self._Operator._QSArgs.LookBack[i], EndIdx
            else:
                iStartIdx, iEndIdx = np.searchsorted(DTRuler, max(self._Operator._QSArgs.StartDT[i], DTRuler[0]), side="left"), EndIdx
            if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndIdx = StartIdx - 1
            iDTs = DTRuler[max(iStartIdx, 0):iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=context.getID(iDescriptor.QSID, pids=None), DTs=iDTs))
        return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=DTs)
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        DTs, IDs = local_context.DTs, local_context.IDs
        SectionIDs = context.getID(self.QSID, pids=None)
        if (not SectionIDs) or (not DTs):
            StdData = pd.DataFrame(index=DTs, columns=SectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
        else:
            if self._Operator._QSArgs.InputFormat == "numpy":
                StdData = self._Operator.calcData(factor=self, ids=SectionIDs, dts=DTs, descriptor_data=[iBwdData.values for iBwdData in bwd_data_list], dt_ruler=context.DTRuler, section_ids=SectionIDs)
                StdData = pd.DataFrame(StdData, index=DTs, columns=SectionIDs)
            else:
                StdData = self._Operator.calcData(factor=self, ids=SectionIDs, dts=DTs, descriptor_data=bwd_data_list, dt_ruler=context.DTRuler, section_ids=SectionIDs)
        PIDIDs = context.NodeState[self.QSID]["pid_ids"]
        context.FactorDataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append")
        context.FactorDataCache.updateDTRange(key=self.QSID, dt_range=(DTs[0], DTs[1]))
        if len(context.PIDList) > 1:
            Sub2MainQueue, PIDEvent = context.Event[self.QSID]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        return StdData.reindex(index=DTs, columns=IDs).sort_index(axis=1)
    
    
    
if __name__ == "__main__":
    import datetime as dt

    from QuantStudio.Core.Factor import DataFactor, Factorize

    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Factor1 = DataFactor(name="Factor1", data=1)
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))

    # 表达式方式
    Factor3 = Factorize(Factor1 + Factor2, factor_name="Factor3")


    # 工厂函数方式
    def test_point(f, idt, iid, x, args):
        return x[0] + x[1]


    test_point = makeFactorOperator(test_point, "Point", sys_args={"入参数": 2, "运算时点": "多时点", "运算ID": "多ID"})
    Factor4 = test_point(Factor1, Factor2, factor_name="Factor4")


    # 装饰器方式
    @FactorOperatorized(operator_type="Time", sys_args={"入参数": 1, "运算ID": "多ID", "回溯期数": [3 - 1]})
    def test_time(f, idt, iid, x, args):
        return np.nansum(x[0], axis=0)


    Factor5 = test_time(Factor1, args={"回溯期数": [2 - 1]}, factor_name="Factor5",
                        factor_args={"描述信息": "我是 Factor5!"})
    print(Factor5.getMetaData(key="Description"))


    # 直接实例化方式, 不推荐
    def test_section(f, idt, iid, x, args):
        return np.argsort(np.argsort(x[0]))


    Factor6 = SectionOperation(name="Factor6", descriptors=[Factor2],
                               sys_args={"算子": test_section, "描述子截面": [IDs], "运算时点": "单时点"})


    def test_panel(f, idt, iid, x, args):
        return np.argsort(np.argsort(x[0][0]))


    Factor7 = PanelOperation(name="Factor7", descriptors=[Factor2], sys_args={
        "算子": makeFactorOperator(test_panel, "Panel", sys_args={"运算时点": "单时点", "回溯期数": [1 - 1]}),
        "描述子截面": [IDs]})

    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    print(Factor6.readData(ids=IDs, dts=DTs))
    print(Factor7.readData(ids=IDs, dts=DTs))

    print("===")