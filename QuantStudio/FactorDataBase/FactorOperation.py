# -*- coding: utf-8 -*-
"""因子运算"""
import gc
import datetime as dt
from functools import partial
from typing import Optional
from multiprocessing import Queue, Event

import pandas as pd
import numpy as np
from traits.api import Dict, Enum, List, ListInt, Int, Instance, Str, Range

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.FactorDataBase.FactorDB import Factor, DataFactor
from QuantStudio.Tools.AuxiliaryFun import partitionList
from QuantStudio.Tools.QSObjects import Panel
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame

class FactorOperator(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        OperatorType = Enum("Point", "Time", "Section", "Panel", arg_type="SingleOption", label="算子类型", order=0, option_range=["Point", "Time", "Section", "Panel"], mutable=False)
        Name = Str("FactorOperator", label="名称", order=1, arg_type="String")
        ModelArgs = Dict(arg_type="Dict", label="参数", order=2, mutable=False)
        Arity = Range(value=1, low=1, high=None, label="入参数", order=3, arg_type="Integer", mutable=False)
        MaxArity = Range(value=0, low=-1, high=None, label="最大入参数", order=4, arg_type="Integer", mutable=False)
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=5, option_range=["double", "string", "object"], mutable=False)
        Description = Str("", label="描述信息", order=6, arg_type="String")
        Meta = Dict(arg_type="Dict", label="元信息", order=7)
        InputFormat = Enum("numpy", "pandas", label="输入格式", order=8, arg_type="SingleOption", option_range=["numpy", "pandas"], mutable=False)
        ExpandDescriptors = ListInt(arg_type="MultiOption", label="展开描述子", order=9, mutable=False)
        DescriptorCompoundType = List(arg_type="List", label="描述子复合类型", order=10, mutable=False)
        MultiMapping = Enum(False, True, arg_type="Bool", label="多重映射", order=11, mutable=False)
        CompoundType = List(arg_type="List", label="复合类型", order=12, mutable=False)
        
        def __QS_initArgValue__(self, args={}):
            if args.get("复合类型", []) or args.get("多重映射", False): args["数据类型"] = "object"
            Arity = args.get("入参数", self.Arity)
            MaxArity = args.get("最大入参数", self.MaxArity)
            if (MaxArity>0) and (Arity>MaxArity):
                raise __QS_Error__(f"最小入参数必须小于等于最大入参数!")
            return super().__QS_initArgValue__(args=args)
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        self._QS_CachedOperators = {}
    
    @property
    def Name(self):
        return self._QSArgs.Name
    
    def _QS_checkArity(self, *x):
        Arity = len(x)
        if self._QSArgs.MaxArity==0:
            if Arity!=self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 和指定的入参数 {self._QSArgs.Arity} 不符!")
        elif self._QSArgs.MaxArity<0:
            if Arity < self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 小于最小入参数 {self._QSArgs.Arity}!")
        else:
            if Arity > self._QSArgs.MaxArity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 大于最大入参数 {self._QSArgs.MaxArity}!")
            elif Arity < self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 小于最小入参数 {self._QSArgs.Arity}!")
        return (True, None)
    
    def _QS_makeOperator(self, *x, args:dict={}, cached_id=None):
        isOK, Msg = self._QS_checkArity(*x)
        if not isOK: raise __QS_Error__(Msg)
        if not args: return self
        elif cached_id is not None:
            if cached_id not in self._QS_CachedOperators:
                self._QS_CachedOperators[cached_id] = self._QS_makeOperator(*x, args=args, cached_id=None)
            return self._QS_CachedOperators[cached_id]
        else:
            Args, args = self._QSArgs.to_dict(), args.copy()
            Args["参数"].update(args.pop("参数", {}))
            Args["元信息"].update(args.pop("元信息", {}))
            Args.update(args)
            Operator = self.__class__(sys_args=Args, logger=self._QS_Logger)
            if getattr(self.calculate, "__self__", None) is not self: Operator.calculate = self.calculate
            return Operator
    
    def _QS_adjOutputPandas(self, df, cols, dts, ids):
        if isinstance(df, pd.DataFrame):
            if isinstance(df.index, pd.MultiIndex):
                if self._QSArgs.MultiMapping:
                    TmpData, Cols, df = {}, df.columns, df.groupby(axis=0, level=[0, 1], as_index=True)
                    for iCol in Cols:
                        TmpData[iCol] = df[iCol].apply(lambda s: s.tolist())
                    df, TmpData = pd.DataFrame(TmpData).loc[:, Cols], None
                elif df.index.duplicated().any():
                    raise __QS_Error__(f"算子 '{self.Name}' 的数据无法保证唯一性, 可以尝试将 '多重映射' 参数取值调整为 True")
                df = df.reindex(columns=cols).apply(lambda s: tuple(s), axis=1).unstack()
            return df.reindex(index=dts, columns=ids)
        elif isinstance(df, pd.Series) and isinstance(df.index, pd.MultiIndex):
            if self._QSArgs.MultiMapping:
                df = df.groupby(axis=0, level=[0, 1], as_index=True).apply(lambda s: s.tolist())
            elif df.index.duplicated().any():
                raise __QS_Error__(f"算子 '{self.Name}' 的数据无法保证唯一性, 可以尝试将 '多重映射' 参数取值调整为 True")
            return df.unstack().reindex(index=dts, columns=ids)
        raise __QS_Error__(f"不支持的返回格式: {df}")
    
    def _QS_partitionSectionIDs(self, section_ids):
        SectionIdx = []# [([ID], [idx])]
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
                Data.append(descriptor_data.iloc[:, i:i+1])
            else:
                iData = descriptor_data.iloc[:, i].values
                DefaultData = np.array([None], dtype="O")
                DefaultData[0] = (None,) * len(iCompoundType)
                DefaultData = DefaultData.repeat(iData.shape[0], axis=0)
                iData = np.where(pd.notnull(iData), iData, DefaultData)
                iDataType = np.dtype([(iCol, float if iType=="double" else "O") for iCol, iType in iCompoundType])
                iData = iData.astype(iDataType)
                iData = pd.DataFrame({iName: pd.Series(iData[iName], index=descriptor_data.index) for iName, iDType in iCompoundType})
                Data.append(iData)
        return pd.concat(Data, axis=1, keys=descriptor_data.columns.tolist())
    
    def _QS_getCalcDTs(self, factor, dts, mask=False):
        CalcDTs = factor._QSArgs.CalcDTRuler
        if CalcDTs:
            StartIdx, EndIdx = np.searchsorted(CalcDTs, dts[0], side="left"), np.searchsorted(CalcDTs, dts[-1], side="right")
            CalcDTs = CalcDTs[StartIdx:EndIdx]
            if mask:
                return np.isin(dts, CalcDTs)
            else:
                return set(CalcDTs)
        else:
            return None
    
    def calculate(self, f, idt, iid, x, args):
        raise NotImplementedError
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None, section_ids=None):
        raise NotImplementedError
    
    def __call__(self, *x, args:dict={}, factor_name:Optional[str]=None, factor_args:dict={}, **kwargs):
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
        OperatorType = Enum("Point", arg_type="SingleOption", label="算子类型", order=0, option_range=["Point"], mutable=False)
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=13, option_range=["单时点", "多时点"], mutable=False)
        IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=14, option_range=["单ID", "多ID"], mutable=False)
    
    def __call__(self, *x, args:dict={}, factor_name:Optional[str]=None, factor_args:dict={}, **kwargs):
        Operator = self._QS_makeOperator(*x, args=args, cached_id=kwargs.pop("cached_id", None))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(name=f"D{i}", data=iFactor)) for i, iFactor in enumerate(x)]
        return PointOperation(name=(Operator._QSArgs.Name if not factor_name else factor_name), descriptors=Descriptors, sys_args={"算子": Operator, **factor_args}, **kwargs)
    
    def _calcDataNumpy(self, factor, ids, dts, descriptor_data, ModelArgs):
        if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        if (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='多ID'):
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
            if CalcMask is not None:
                descriptor_data = [iData[CalcMask] for iData in descriptor_data]
                dts = np.array(dts, dtype="O")[CalcMask].tolist()
                iStdData = self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
                StdData[CalcMask, :] = iStdData
            else:
                return self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
        elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                for j, jID in enumerate(ids):
                    StdData[i, j] = self.calculate(factor, iDT, jID, [iData[i, j] for iData in descriptor_data], ModelArgs)
        elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
            CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
            if CalcMask is None:
                for j, jID in enumerate(ids):
                    StdData[:, j] = self.calculate(factor, dts, jID, [iData[:, j] for iData in descriptor_data], ModelArgs)
            else:
                dts = np.array(dts, dtype="O")[CalcMask].tolist()
                for j, jID in enumerate(ids):
                    StdData[CalcMask, j] = self.calculate(factor, dts, jID, [iData[CalcMask, j] for iData in descriptor_data], ModelArgs)
        elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
            CalcDTs = self._QS_getCalcDTs(factor, dts, mask=False)
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                StdData[i, :] = self.calculate(factor, iDT, ids, [iData[i, :] for iData in descriptor_data], ModelArgs)
        return StdData
    
    def _calcDataPandas(self, factor, ids, dts, descriptor_data, ModelArgs):
        CalcMask = self._QS_getCalcDTs(factor, dts, mask=True)
        if CalcMask is not None:
            descriptor_data = Panel({f"d{i}": descriptor_data[i][CalcMask] for i in range(len(descriptor_data))}).to_frame(filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))                                
        else:
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).to_frame(filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
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
        if (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='多ID'):
            DTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
            StdData = self.calculate(factor, DTs, ids, descriptor_data, ModelArgs)
            return self._QS_adjOutputPandas(StdData, CompoundCols, DTs, ids).reindex(index=dts)
        elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
            if self._QSArgs.DataType == 'double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
            else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
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
        elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
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
            if StdData.shape[1]==1: StdData = StdData.iloc[:, 0]
            return self._QS_adjOutputPandas(StdData, CompoundCols, CalcDTs, ids).reindex(index=dts)
        elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
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
        OperatorType = Enum("Time", arg_type="SingleOption", label="算子类型", order=0, option_range=["Time"], mutable=False)
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=13, option_range=["单时点", "多时点"], mutable=False)
        IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=14, option_range=["单ID", "多ID"], mutable=False)
        LookBack = ListInt(arg_type="ArgList", label="回溯期数", order=15, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=16, mutable=False)# 描述子的回溯模式
        StartDT = List(arg_type="ArgList", label="起始时点", order=17, mutable=False)# 扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数
        iInitFactor = Int(-1, arg_type="Integer", label="起始因子", order=18)
        
        def __QS_initArgValue__(self, args={}):
            if "回溯期数" in args:
                args = args.copy()
                if "回溯模式" not in args:
                    args["回溯模式"] = ["滚动窗口"] * len(args["回溯期数"])
                if "起始时点" not in args:
                    args["起始时点"] = [None] * len(args["回溯期数"])
            return super().__QS_initArgValue__(args=args)
    
    def __call__(self, *x, args:dict={}, factor_name:Optional[str]=None, factor_args:dict={}, **kwargs):
        Operator = self._QS_makeOperator(*x, args=args, cached_id=kwargs.pop("cached_id", None))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(name=f"D{i}", data=iFactor)) for i, iFactor in enumerate(x)]
        return TimeOperation(name=(self._QSArgs.Name if not factor_name else factor_name), descriptors=Descriptors, sys_args={"算子": Operator, **factor_args}, **kwargs)
    
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
        OperatorType = Enum("Section", arg_type="SingleOption", label="算子类型", order=0, option_range=["Section"], mutable=False)
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=13, option_range=["单时点", "多时点"], mutable=False)
        OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=14, option_range=["全截面", "单ID"], mutable=False)
        DescriptorSection = List(arg_type="List", label="描述子截面", order=15, mutable=False)
    
    def __call__(self, *x, args:dict={}, factor_name:Optional[str]=None, factor_args:dict={}, **kwargs):
        Operator = self._QS_makeOperator(*x, args=args, cached_id=kwargs.pop("cached_id", None))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(name=f"D{i}", data=iFactor)) for i, iFactor in enumerate(x)]
        return SectionOperation(name=(self._QSArgs.Name if not factor_name else factor_name), descriptors=Descriptors, sys_args={"算子": Operator, **factor_args}, **kwargs)
        
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
        OperatorType = Enum("Panel", arg_type="SingleOption", label="算子类型", order=0, option_range=["Panel"], mutable=False)
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=13, option_range=["单时点", "多时点"], mutable=False)
        OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=14, option_range=["全截面", "单ID"], mutable=False)
        DescriptorSection = List(arg_type="List", label="描述子截面", order=15, mutable=False)
        LookBack = List(arg_type="ArgList", label="回溯期数", order=16, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=17, mutable=False)
        StartDT = List(arg_type="ArgList", label="起始时点", order=18, mutable=False)# 扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数
        iInitFactor = Int(-1, arg_type="DataFrame", label="起始因子", order=19)
        
        def __QS_initArgValue__(self, args={}):
            if "回溯期数" in args:
                args = args.copy()
                if "回溯模式" not in args:
                    args["回溯模式"] = ["滚动窗口"] * len(args["回溯期数"])
                if "起始时点" not in args:
                    args["起始时点"] = [None] * len(args["回溯期数"])
            return super().__QS_initArgValue__(args=args)
    
    def __call__(self, *x, args:dict={}, factor_name:Optional[str]=None, factor_args:dict={}, **kwargs):
        Operator = self._QS_makeOperator(*x, args=args, cached_id=kwargs.pop("cached_id", None))
        Descriptors = [(iFactor if isinstance(iFactor, Factor) else DataFactor(name=f"D{i}", data=iFactor)) for i, iFactor in enumerate(x)]
        return PanelOperation(name=(self._QSArgs.Name if not factor_name else factor_name), descriptors=Descriptors, sys_args={"算子": Operator, **factor_args}, **kwargs)
    
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
                StdData = self.calculate(factor, DTRuler, SectionIDs, descriptor_data, ModelArgs)[iStartIdx:, :]
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
def makeFactorOperator(func, operator_type, sys_args={}, **kwargs):
    if not callable(func): raise __QS_Error__("func 必须是可调用对象!")
    if operator_type == "Point":
        FactorOperator = PointOperator(sys_args=sys_args, config_file=None, **kwargs)
    elif operator_type == "Time":
        FactorOperator = TimeOperator(sys_args=sys_args, config_file=None, **kwargs)
    elif operator_type == "Section":
        FactorOperator = SectionOperator(sys_args=sys_args, config_file=None, **kwargs)
    elif operator_type == "Panel":
        FactorOperator = PanelOperator(sys_args=sys_args, config_file=None, **kwargs)
    else:
        raise __QS_Error__(f"错误的因子算子类型: '{operator_type}', 必须为 'Point', 'Time', 'Section' 或者 'Panel'")
    FactorOperator.calculate = func
    return FactorOperator

# 将函数转换成因子定义的装饰器
def FactorOperatorized(operator_type, sys_args={}):
    return partial(makeFactorOperator, operator_type=operator_type, sys_args=sys_args)

class DerivativeFactor(Factor):
    """衍生因子"""
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Operator = Instance(FactorOperator, arg_type="QSObject", label="算子", order=0, mutable=False)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
        Meta = Dict(arg_type="Dict", label="元信息", order=2)
        CalcDTRuler = List(dt.datetime, arg_type="DateTimeList", label="计算时点标尺", order=3, mutable=False)
    
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        self._Descriptors = descriptors
        self.UserData = {}
        if descriptors: kwargs.setdefault("logger", descriptors[0]._QS_Logger)
        super().__init__(name=name, ft=None, sys_args=sys_args, config_file=None, **kwargs)
        self._Operator = self._QSArgs.Operator
        self._QS_checkConsistency()
    
    # 检查因子定义的相容性
    def _QS_checkConsistency(self):
        pass
    
    @property
    def Descriptors(self):
        return self._Descriptors
    
    @property
    def Operator(self):
        return self._Operator
    
    def getMetaData(self, key=None, args={}):
        DataType = args.get("数据类型", self._Operator._QSArgs.DataType)
        if key is None: return pd.Series({"DataType": DataType, **self._QSArgs.Meta})
        elif key=="DataType": return DataType
        else: return self._QSArgs.get(key, None)
        return None


class PointOperation(DerivativeFactor):
    """单点运算"""
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Point", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, PointOperator): raise __QS_Error__(f"类型为 PointOperation 的衍生因子 {name} 的算子类型必须为 PointOperator, 但传入的算子类型为 {Operator.__class__}")
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **sys_args}, **kwargs)
        
    def readData(self, ids, dts, **kwargs):
        Context = self.BatchContext
        if Context is not None:
            return Context.readData(factors=[self], ids=ids, dts=dts, **kwargs).iloc[0]
        if self._Operator._QSArgs.InputFormat=="numpy":
            StdData = self._Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs).values for iDescriptor in self._Descriptors])
            return pd.DataFrame(StdData, index=dts, columns=ids)
        else:
            StdData = self._Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs) for iDescriptor in self._Descriptors])
            return StdData
    
    def __QSBC_initOperation__(self, context, dt_range, section_ids):
        super().__QSBC_initOperation__(context, dt_range, section_ids)
        for i, iDescriptor in enumerate(self._Descriptors):
            iDescriptor.__QSBC_initOperation__(context, dt_range, section_ids)
    
    def __QSBC_prepareCacheData__(self, dt_range):
        Context = self.BatchContext
        PID = Context._iPID
        DTs = Context.getDateTime(dt_range)
        if not DTs: return 0
        IDs = Context.getID(self._QSID, [PID])
        if IDs:
            if self._Operator._QSArgs.InputFormat=="numpy":
                StdData = self._Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor.__QSBC_getData__(DTs, pids=[PID]).values for iDescriptor in self._Descriptors])
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = self._Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor.__QSBC_getData__(DTs, pids=[PID]) for iDescriptor in self._Descriptors])
        else:
            for iDescriptor in self._Descriptors:
                iDescriptor.__QSBC_getData__(DTs, pids=[PID])
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if self._Operator._QSArgs.DataType=="double" else "O"))
        Context._Cache.writeFactorData(key=self._QSID, target_field="StdData", factor_data=StdData, pid_ids={PID: IDs}, pid=PID, if_exists="append")
        Context.updateDTRange(factor_id=self._QSID, dt_range=dt_range)
        Context._Cache.writeFactorData(key=self._QSID, target_field="DTRange", factor_data=Context._CachedDTRange[self._QSID], pid_ids=None, pid=PID, if_exists="replace")
        return 0

class TimeOperation(DerivativeFactor):
    """时间序列运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        LookBack = List(arg_type="ArgList", label="回溯期数", order=4, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=5, mutable=False)# 描述子的回溯模式
        StartDT = List(arg_type="ArgList", label="起始时点", order=6, mutable=False)# 扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数
        iInitFactor = Int(-1, arg_type="Integer", label="起始因子", order=7)
    
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Time", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, TimeOperator): raise __QS_Error__(f"类型为 TimeOperation 的衍生因子 {name} 的算子类型必须为 TimeOperator, 但传入的算子类型为 {Operator.__class__}")
        FactorArgs = {
            "回溯期数": sys_args.pop("回溯期数", Operator._QSArgs.LookBack),
            "回溯模式": sys_args.pop("回溯模式", Operator._QSArgs.LookBackMode),
            "起始时点": sys_args.pop("起始时点", Operator._QSArgs.StartDT),
            "起始因子": sys_args.pop("起始因子", Operator._QSArgs.iInitFactor)
        }
        if not FactorArgs["回溯模式"]: FactorArgs["回溯模式"] = ["滚动窗口"] * len(FactorArgs["回溯期数"])
        if not FactorArgs["起始时点"]: FactorArgs["起始时点"] = [None] * len(FactorArgs["回溯期数"])
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **FactorArgs, **sys_args}, **kwargs)
    
    def _QS_checkConsistency(self):
        Operator = self._Operator
        nDescriptor = len(self._Descriptors)
        LookBack = (self._QSArgs.LookBack if self._QSArgs.LookBack else Operator._QSArgs.LookBack)
        if len(LookBack) < nDescriptor: raise  __QS_Error__("时序运算因子 '%s'(QSID: %s) 的参数 '回溯期数' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        LookBackMode = (self._QSArgs.LookBackMode if self._QSArgs.LookBackMode else Operator._QSArgs.LookBackMode)
        if len(LookBackMode) < nDescriptor: raise  __QS_Error__("时序运算因子 '%s'(QSID: %s) 的参数 '回溯模式' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        StartDT = (self._QSArgs.StartDT if self._QSArgs.StartDT else Operator._QSArgs.StartDT)
        if len(StartDT) < nDescriptor: raise  __QS_Error__("时序运算因子 '%s'(QSID: %s) 的参数 '起始时点' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        return super()._QS_checkConsistency()
        
    def readData(self, ids, dts, **kwargs):
        Context = self.BatchContext
        if Context is not None: return Context.readData(factors=[self], ids=ids, dts=dts, **kwargs).iloc[0]
        Operator = self._Operator
        if "dt_ruler" in kwargs:
            DTRuler = kwargs.get("dt_ruler", dts)
            StartIdx, EndIdx = np.searchsorted(DTRuler, dts[0], side="left"), np.searchsorted(DTRuler, dts[-1], side="right")
        else:
            DTRuler = dts
            StartIdx, EndIdx = 0, len(dts)
        if StartIdx >= EndIdx: return pd.DataFrame(index=dts, columns=ids)
        nID = len(ids)
        DescriptorData = []
        for i, iDescriptor in enumerate(self._Descriptors):
            if (self._QSArgs.LookBackMode[i]=="滚动窗口") or (self._QSArgs.StartDT[i] is None):
                iLookBack = self._QSArgs.LookBack[i]
            else:
                if self._QSArgs.StartDT[i] < DTRuler[0]:
                    self._QS_Logger.warning(f"因子 '{self.Name}'(QSID: {self._QSID}) 的第 {i} 个描述子 '{iDescriptor.Name}'(QSID: {iDescriptor._QSID}) 的起始时点 {self._QSArgs.StartDT[i]} 小于时点标尺的起始时点 {DTRuler[0]}, 其起始时点将重置为时点标尺的起始时点")
                elif self._QSArgs.StartDT[i] > dts[0]:
                    self._QS_Logger.warning(f"因子 '{self.Name}'(QSID: {self._QSID}) 的第 {i} 个描述子 '{iDescriptor.Name}'(QSID: {iDescriptor._QSID}) 的起始时点 {self._QSArgs.StartDT[i]} 大于待计算时点序列的起始时点 {dts[0]}, 其起始时点将重置为待计算时点序列的起始时点")
                iLookBack = max(0, StartIdx - np.searchsorted(DTRuler, self._QSArgs.StartDT[i], side="left"))
            if i!=self._QSArgs.iInitFactor:
                iDTs = DTRuler[max(StartIdx-iLookBack, 0):EndIdx]
            else:
                iDTs = DTRuler[max(StartIdx-self._QSArgs.LookBack[i], 0):StartIdx]
            if Operator._QSArgs.InputFormat=="numpy":
                if iDTs: iDescriptorData = iDescriptor.readData(ids=ids, dts=iDTs, **kwargs).values
                else: iDescriptorData = np.full((0, nID), np.nan)
                if StartIdx < self._QSArgs.LookBack[i]:
                    iLookBackData = np.full((self._QSArgs.LookBack[i] - StartIdx, nID), np.nan)
                    iDescriptorData = np.r_[iLookBackData, iDescriptorData]
            else:
                if iDTs: iDescriptorData = iDescriptor.readData(ids=ids, dts=iDTs, **kwargs)
                else: iDescriptorData = pd.DataFrame(columns=ids)
            DescriptorData.append(iDescriptorData)
        StdData = Operator.calcData(factor=self, ids=ids, dts=DTRuler[StartIdx:EndIdx], descriptor_data=DescriptorData, dt_ruler=DTRuler)
        if Operator._QSArgs.InputFormat=="numpy":
            return pd.DataFrame(StdData, index=DTRuler[StartIdx:EndIdx], columns=ids).reindex(index=dts)
        else:
            return StdData.reindex(index=dts)
    
    def __QSBC_initOperation__(self, context, dt_range, section_ids):
        super().__QSBC_initOperation__(context, dt_range, section_ids)
        StartDT, EndDT = context._DTRange[self.QSID]
        DTRuler = list(context._QSArgs.DTRuler)
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
            if self._QSArgs.StartDT[i] is None:# 未指定起始时点, 从当前位置回溯 LookBack[i] 期
                iStartIdx = StartIdx - self._QSArgs.LookBack[i]
            else:# 指定了起始时点, 以起始时点 StartDT[i] 的位置为准
                iStartIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
            if iStartIdx < 0: self._QS_Logger.warning("注意: 对于因子 '%s'(QSID: %s) 的描述子 '%s'(QSID: %s), 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, self.QSID, iDescriptor.Name, iDescriptor.QSID))
            iStartIdx = max(0, iStartIdx)
            if i==self._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
            else:
                iEndDT = EndDT
            iDescriptor.__QSBC_initOperation__(context, (DTRuler[iStartIdx], iEndDT), section_ids)
    
    def __QSBC_prepareCacheData__(self, dt_range):
        Context = self.BatchContext
        DTs = Context.getDateTime(dt_range)
        if not DTs: return 0
        PID = Context._iPID
        IDs = Context.getID(self._QSID, [PID])
        DTRuler = list(Context._QSArgs.DTRuler)
        StartIdx, EndIdx = DTRuler.index(DTs[0]), DTRuler.index(DTs[-1])
        Operator = self._Operator
        DescriptorData = []
        for i, iDescriptor in enumerate(self._Descriptors):
            if (self._QSArgs.LookBackMode[i]=="滚动窗口") or (self._QSArgs.StartDT[i] is None):
                iStartIdx, iEndIdx = StartIdx - self._QSArgs.LookBack[i], EndIdx
            else:
                iStartIdx, iEndIdx = np.searchsorted(DTRuler, max(self._QSArgs.StartDT[i], DTRuler[0]), side="left"), EndIdx
            if i==self._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndIdx = StartIdx - 1
            iDTs = DTRuler[max(iStartIdx, 0):iEndIdx+1]
            if Operator._QSArgs.InputFormat=="numpy":
                if iDTs:
                    iDescriptorData = iDescriptor.__QSBC_getData__(iDTs, pids=[PID]).values
                    if iStartIdx<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartIdx), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
                else:
                    iIDs = Context.getID(iDescriptor._QSID, [PID])
                    iDescriptorData = np.full(shape=(abs(min(iStartIdx, 0)), len(iIDs)), fill_value=np.nan)
            else:
                iDescriptorData = iDescriptor.__QSBC_getData__(iDTs, pids=[PID])
            DescriptorData.append(iDescriptorData)
        if not IDs:
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        elif Operator._QSArgs.InputFormat=="numpy":
            StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
            StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        else:
            StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
        Context._Cache.writeFactorData(key=self._QSID, target_field="StdData", factor_data=StdData, pid_ids={PID: IDs}, pid=PID, if_exists="append")
        Context.updateDTRange(factor_id=self._QSID, dt_range=dt_range)
        Context._Cache.writeFactorData(key=self._QSID, target_field="DTRange", factor_data=Context._CachedDTRange[self._QSID], pid_ids=None, pid=PID, if_exists="replace")
        return 0

class SectionOperation(DerivativeFactor):
    """截面运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DescriptorSection = List(arg_type="List", label="描述子截面", order=4, mutable=False)
    
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Section", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, SectionOperator): raise __QS_Error__(f"类型为 SectionOperation 的衍生因子 {name} 的算子类型必须为 SectionOperator, 但传入的算子类型为 {Operator.__class__}")
        FactorArgs = {
            "描述子截面": sys_args.pop("描述子截面", Operator._QSArgs.DescriptorSection)
        }
        if not FactorArgs["描述子截面"]: FactorArgs["描述子截面"] = [None] * len(descriptors)
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **FactorArgs, **sys_args}, **kwargs)
    
    def _QS_checkConsistency(self):
        Operator = self._Operator
        nDescriptor = len(self._Descriptors)
        DescriptorSection = (self._QSArgs.DescriptorSection if self._QSArgs.DescriptorSection else Operator._QSArgs.DescriptorSection)
        if len(DescriptorSection) < nDescriptor: raise  __QS_Error__("截面运算因子 '%s'(QSID: %s) 的参数 '描述子截面' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        return super()._QS_checkConsistency()
    
    def readData(self, ids, dts, **kwargs):
        Context = self.BatchContext
        if Context is not None: return Context.readData(factors=[self], ids=ids, dts=dts, **kwargs).iloc[0]
        Operator = self._Operator
        SectionIDs = kwargs.pop("section_ids", ids)
        DescriptorData = []
        if Operator._QSArgs.InputFormat == "numpy":
            for i, iDescriptor in enumerate(self._Descriptors):
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                DescriptorData.append(iDescriptor.readData(ids=iSectionIDs, dts=dts, **kwargs).values)
            StdData = Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=DescriptorData, section_ids=SectionIDs)
            return pd.DataFrame(StdData, index=dts, columns=ids)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                DescriptorData.append(iDescriptor.readData(ids=iSectionIDs, dts=dts, **kwargs))            
            StdData = Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=DescriptorData, section_ids=SectionIDs)
            return StdData
    
    def __QSBC_initOperation__(self, context, dt_range, section_ids):
        super().__QSBC_initOperation__(context, dt_range, section_ids)
        for i, iDescriptor in enumerate(self._Descriptors):
            if self._QSArgs.DescriptorSection[i] is None:
                iDescriptor.__QSBC_initOperation__(context, dt_range, section_ids)
            else:
                iDescriptor.__QSBC_initOperation__(context, dt_range, self._QSArgs.DescriptorSection[i])
        if (context._QSArgs.CalcConcurrentNum > 0) and (self.QSID not in context._Event):
            context._Event[self.QSID] = (Queue(), Event())
        
    def __QSBC_prepareCacheData__(self, dt_range):
        Context = self.BatchContext
        PID = Context._iPID
        DTs = Context.getDateTime(dt_range)
        if not DTs: return 0
        IDs = Context.getID(self._QSID, pids=None)
        DTPartition = partitionList(DTs, len(Context._PIDs))
        DTs = DTPartition[Context._PIDs.index(PID)]
        Operator = self._Operator
        if not DTs:# 该进程未分配到计算任务
            iDTs = [Context._DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor.__QSBC_getData__(iDTs, pids=None)
            StdData = pd.DataFrame(columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        elif IDs:
            if Operator._QSArgs.InputFormat == "numpy":
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor.__QSBC_getData__(DTs, pids=None).values for iDescriptor in self._Descriptors])
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor.__QSBC_getData__(DTs, pids=None) for iDescriptor in self._Descriptors])
        else:
            for iDescriptor in self._Descriptors:
                iDescriptor.__QSBC_getData__(DTs, pids=None)
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        PID_IDs = Context.getPIDID(self._QSID)
        Context._Cache.writeFactorData(key=self._QSID, target_field="StdData", factor_data=StdData, pid_ids=PID_IDs, pid=None, if_exists="append")
        Context.updateDTRange(factor_id=self._QSID, dt_range=dt_range)
        Context._Cache.writeFactorData(key=self._QSID, target_field="DTRange", factor_data=Context._CachedDTRange[self._QSID], pid_ids=PID_IDs, pid=None, if_exists="replace")
        StdData = None# 释放数据
        gc.collect()
        if Context._QSArgs.CalcConcurrentNum>0:
            Sub2MainQueue, PIDEvent = Context._Event[self._QSID]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        return 0

class PanelOperation(DerivativeFactor):
    """面板运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DescriptorSection = List(arg_type="List", label="描述子截面", order=3.5, mutable=False)
        LookBack = List(arg_type="ArgList", label="回溯期数", order=4, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=5, mutable=False)# 描述子的回溯模式
        StartDT = List(arg_type="ArgList", label="起始时点", order=6, mutable=False)# 扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数
        iInitFactor = Int(-1, arg_type="Integer", label="起始因子", order=7)
    
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Panel", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, PanelOperator): raise __QS_Error__(f"类型为 PanelOperation 的衍生因子 {name} 的算子类型必须为 PanelOperator, 但传入的算子类型为 {Operator.__class__}")
        FactorArgs = {
            "描述子截面": sys_args.pop("描述子截面", Operator._QSArgs.DescriptorSection),
            "回溯期数": sys_args.pop("回溯期数", Operator._QSArgs.LookBack),
            "回溯模式": sys_args.pop("回溯模式", Operator._QSArgs.LookBackMode),
            "起始时点": sys_args.pop("起始时点", Operator._QSArgs.StartDT),
            "起始因子": sys_args.pop("起始因子", Operator._QSArgs.iInitFactor)
        }
        if not FactorArgs["回溯模式"]: FactorArgs["回溯模式"] = ["滚动窗口"] * len(FactorArgs["回溯期数"])
        if not FactorArgs["起始时点"]: FactorArgs["起始时点"] = [None] * len(FactorArgs["回溯期数"])
        if not FactorArgs["描述子截面"]: FactorArgs["描述子截面"] = [None] * len(descriptors)
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **FactorArgs, **sys_args}, **kwargs)
    
    def _QS_checkConsistency(self):
        Operator = self._Operator
        nDescriptor = len(self._Descriptors)
        LookBack = (self._QSArgs.LookBack if self._QSArgs.LookBack else Operator._QSArgs.LookBack)
        if len(LookBack) < nDescriptor: raise  __QS_Error__("面板运算因子 '%s'(QSID: %s) 的参数 '回溯期数' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        LookBackMode = (self._QSArgs.LookBackMode if self._QSArgs.LookBackMode else Operator._QSArgs.LookBackMode)
        if len(LookBackMode) < nDescriptor: raise  __QS_Error__("面板运算因子 '%s'(QSID: %s) 的参数 '回溯模式' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        StartDT = (self._QSArgs.StartDT if self._QSArgs.StartDT else Operator._QSArgs.StartDT)
        if len(StartDT) < nDescriptor: raise  __QS_Error__("面板运算因子 '%s'(QSID: %s) 的参数 '起始时点' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        DescriptorSection = (self._QSArgs.DescriptorSection if self._QSArgs.DescriptorSection else Operator._QSArgs.DescriptorSection)
        if len(DescriptorSection) < nDescriptor: raise  __QS_Error__("面板运算因子 '%s'(QSID: %s) 的参数 '描述子截面' 序列长度小于描述子个数!" % (self.Name, self.QSID))
        return super()._QS_checkConsistency()
    
    def readData(self, ids, dts, **kwargs):
        Context = self.BatchContext
        if Context is not None: return Context.readData(factors=[self], ids=ids, dts=dts, **kwargs).iloc[0]
        Operator = self._Operator
        if "dt_ruler" in kwargs:
            DTRuler = kwargs.get("dt_ruler", dts)
            StartIdx, EndIdx = np.searchsorted(DTRuler, dts[0], side="left"), np.searchsorted(DTRuler, dts[-1], side="right")
        else:
            DTRuler = dts
            StartIdx, EndIdx = 0, len(dts)
        SectionIDs = kwargs.pop("section_ids", ids)
        if StartIdx >= EndIdx: return pd.DataFrame(index=dts, columns=ids)
        DescriptorData = []
        for i, iDescriptor in enumerate(self._Descriptors):
            if (self._QSArgs.LookBackMode[i]=="滚动窗口") or (self._QSArgs.StartDT[i] is None):
                iLookBack = self._QSArgs.LookBack[i]
            else:
                if self._QSArgs.StartDT[i] < DTRuler[0]:
                    self._QS_Logger.warning(f"因子 '{self.Name}'(QSID: {self._QSID}) 的第 {i} 个描述子 '{iDescriptor.Name}'(QSID: {iDescriptor._QSID}) 的起始时点 {self._QSArgs.StartDT[i]} 小于时点标尺的起始时点 {DTRuler[0]}, 其起始时点将重置为时点标尺的起始时点")
                elif self._QSArgs.StartDT[i] > dts[0]:
                    self._QS_Logger.warning(f"因子 '{self.Name}'(QSID: {self._QSID}) 的第 {i} 个描述子 '{iDescriptor.Name}'(QSID: {iDescriptor._QSID}) 的起始时点 {self._QSArgs.StartDT[i]} 大于待计算时点序列的起始时点 {dts[0]}, 其起始时点将重置为待计算时点序列的起始时点")
                iLookBack = max(0, StartIdx - np.searchsorted(DTRuler, self._QSArgs.StartDT[i], side="left"))
            if i!=self._QSArgs.iInitFactor:
                iDTs = DTRuler[max(StartIdx-iLookBack, 0):EndIdx]
            else:
                iDTs = DTRuler[max(StartIdx-self._QSArgs.LookBack[i], 0):StartIdx]
            iSectionIDs = self._QSArgs.DescriptorSection[i]
            if iSectionIDs is None: iSectionIDs = SectionIDs
            if Operator._QSArgs.InputFormat=="numpy":
                if iDTs: iDescriptorData = iDescriptor.readData(ids=iSectionIDs, dts=iDTs, **kwargs).values
                else: iDescriptorData = np.full((0, len(iSectionIDs)), np.nan)
                if StartIdx < self._QSArgs.LookBack[i]:
                    iLookBackData = np.full((self._QSArgs.LookBack[i] - StartIdx, len(iSectionIDs)), np.nan)
                    iDescriptorData = np.r_[iLookBackData, iDescriptorData]
            else:
                if iDTs: iDescriptorData = iDescriptor.readData(ids=iSectionIDs, dts=iDTs, **kwargs)
                else: iDescriptorData = pd.DataFrame(columns=iSectionIDs)
            DescriptorData.append(iDescriptorData)
        StdData = Operator.calcData(factor=self, ids=ids, dts=DTRuler[StartIdx:EndIdx], descriptor_data=DescriptorData, dt_ruler=DTRuler, section_ids=SectionIDs)
        if Operator._QSArgs.InputFormat=="numpy":
            return pd.DataFrame(StdData, index=DTRuler[StartIdx:EndIdx], columns=ids).reindex(index=dts)
        else:
            return StdData.reindex(index=dts, columns=ids)
    
    def __QSBC_initOperation__(self, context, dt_range, section_ids):
        super().__QSBC_initOperation__(context, dt_range, section_ids)
        StartDT, EndDT = context._DTRange[self.QSID]
        DTRuler = list(context._QSArgs.DTRuler)
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
            if self._QSArgs.StartDT[i] is None:# 未指定起始时点, 从当前位置回溯 LookBack[i] 期
                iStartIdx = StartIdx - self._QSArgs.LookBack[i]
            else:# 指定了起始时点, 以起始时点 StartDT[i] 的位置为准
                iStartIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
            if iStartIdx < 0: self._QS_Logger.warning("注意: 对于因子 '%s'(QSID: %s) 的描述子 '%s'(QSID: %s), 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, self.QSID, iDescriptor.Name, iDescriptor.QSID))
            iStartIdx = max(0, iStartIdx)
            if i==self._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
            else:
                iEndDT = EndDT
            if self._QSArgs.DescriptorSection[i] is None:
                iDescriptor.__QSBC_initOperation__(context, (DTRuler[iStartIdx], iEndDT), section_ids)
            else:
                iDescriptor.__QSBC_initOperation__(context, (DTRuler[iStartIdx], iEndDT), self._QSArgs.DescriptorSection[i])
        if (context._QSArgs.CalcConcurrentNum>0) and (self.QSID not in context._Event):
            context._Event[self.QSID] = (Queue(), Event())
    
    def __QSBC_prepareCacheData__(self, dt_range):
        Context = self.BatchContext
        DTs = Context.getDateTime(dt_range)
        if not DTs: return 0
        PID = Context._iPID
        Operator = self._Operator
        DTRuler = list(Context._QSArgs.DTRuler)
        if (self._QSArgs.iInitFactor>=0) and (self._QSArgs.LookBackMode[self._QSArgs.iInitFactor]=="扩张窗口"):
            DTPartition = [DTs]+[[]]*(len(Context._PIDs)-1)
        else:
            DTPartition = partitionList(DTs, len(Context._PIDs))
        DTs = DTPartition[Context._PIDs.index(PID)]
        IDs = Context.getID(self._QSID, pids=None)
        if not DTs:# 该切片未分配到计算任务
            iDTs = [Context._DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor.__QSBC_getData__(iDTs, pids=None)
            StdData = pd.DataFrame(columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        else:
            StartIdx, EndIdx = DTRuler.index(DTs[0]), DTRuler.index(DTs[-1])
            DescriptorData = []
            for i, iDescriptor in enumerate(self._Descriptors):
                if self._QSArgs.StartDT[i] is None:# 没有指定起始时点
                    iStartIdx, iEndIdx = StartIdx - self._QSArgs.LookBack[i], EndIdx
                else:
                    iStartIdx, iEndIdx = np.searchsorted(DTRuler, max(self._QSArgs.StartDT[i], DTRuler[0]), side="left"), EndIdx
                if i==self._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                    iEndIdx = StartIdx - 1
                iDTs = DTRuler[max(iStartIdx, 0):iEndIdx+1]
                if Operator._QSArgs.InputFormat=="numpy":
                    if iDTs:
                        iDescriptorData = iDescriptor.__QSBC_getData__(iDTs, pids=None).values
                        if iStartIdx<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartIdx), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
                    else:
                        iIDs = Context.getID(iDescriptor._QSID, None)
                        iDescriptorData = np.full(shape=(abs(min(iStartIdx, 0)), len(iIDs)), fill_value=np.nan)
                else:
                    iDescriptorData = iDescriptor.__QSBC_getData__(iDTs, pids=None)
                DescriptorData.append(iDescriptorData)
            if not IDs:
                StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
            elif Operator._QSArgs.InputFormat=="numpy":
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
            DescriptorData, iDescriptorData = None, None
        PID_IDs = Context.getPIDID(self._QSID)
        Context._Cache.writeFactorData(key=self._QSID, target_field="StdData", factor_data=StdData, pid_ids=PID_IDs, pid=None, if_exists="append")
        Context.updateDTRange(factor_id=self._QSID, dt_range=dt_range)
        Context._Cache.writeFactorData(key=self._QSID, target_field="DTRange", factor_data=Context._CachedDTRange[self._QSID], pid_ids=PID_IDs, pid=None, if_exists="replace")
        StdData = None# 释放数据
        gc.collect()
        if Context._QSArgs.CalcConcurrentNum>0:
            Sub2MainQueue, PIDEvent = Context._Event[self._QSID]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        return 0


if __name__=="__main__":
    import datetime as dt
    
    from QuantStudio.FactorDataBase.FactorDB import DataFactor, Factorize
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
    @FactorOperatorized(operator_type="Time", sys_args={"入参数": 1, "运算ID": "多ID", "回溯期数": [3-1]})
    def test_time(f, idt, iid, x, args):
        return np.nansum(x[0], axis=0)
    Factor5 = test_time(Factor1, args={"回溯期数": [2-1]}, factor_name="Factor5", factor_args={"描述信息": "我是 Factor5!"})
    print(Factor5.getMetaData(key="Description"))
    
    # 直接实例化方式, 不推荐
    def test_section(f, idt, iid, x, args):
        return np.argsort(np.argsort(x[0]))
    Factor6 = SectionOperation(name="Factor6", descriptors=[Factor2], sys_args={"算子": test_section, "描述子截面": [IDs], "运算时点": "单时点"})
    
    def test_panel(f, idt, iid, x, args):
        return np.argsort(np.argsort(x[0][0]))    
    Factor7 = PanelOperation(name="Factor7", descriptors=[Factor2], sys_args={"算子": makeFactorOperator(test_panel, "Panel", sys_args={"运算时点": "单时点", "回溯期数": [1-1]}), "描述子截面": [IDs]})
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    print(Factor6.readData(ids=IDs, dts=DTs))
    print(Factor7.readData(ids=IDs, dts=DTs))

    print("===")