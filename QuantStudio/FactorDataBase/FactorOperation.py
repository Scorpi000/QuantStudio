# -*- coding: utf-8 -*-
"""因子运算"""
import gc
from functools import partial
from typing import Optional
from multiprocessing import Queue, Event

import pandas as pd
import numpy as np
from traits.api import Dict, Enum, List, ListInt, Int, Instance, Str, Either, Range

from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.Tools.AuxiliaryFun import partitionList, partitionListMovingSampling
from QuantStudio.Tools.QSObjects import Panel
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame

class FactorOperator(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        OperatorType = Enum("Point", "Time", "Section", "Panel", arg_type="SingleOption", label="算子类型", order=0, option_range=["Point", "Time", "Section", "Panel"], mutable=False)
        Name = Str("FactorOperator", label="名称", order=1, arg_type="String", mutable=False)
        Arity = Range(value=1, low=1, high=None, label="入参数", order=1, arg_type="Integer", mutable=False)
        MaxArity = Range(value=0, low=0, high=None, label="最大入参数", order=1, arg_type="Integer", mutable=False)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1, mutable=False)
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=2, option_range=["double", "string", "object"], mutable=False)
        Expression = Str("", arg_type="String", label="表达式", order=3)
        Description = Str("", label="描述信息", order=4, arg_type="String")
        Unit = Str("", label="量纲", order=5, arg_type="String", mutable=False)
        Meta = Dict(arg_type="Dict", label="元信息", order=1)
        InputFormat = Enum("numpy", "pandas", label="输入格式", order=5, arg_type="SingleOption", option_range=["numpy", "pandas"], mutable=False)
        ExpandDescriptors = ListInt(arg_type="MultiOption", label="展开描述子", order=6, mutable=False)
        DescriptorCompoundType = List(arg_type="List", label="描述子复合类型", order=7, mutable=False)
        MultiMapping = Enum(False, True, arg_type="Bool", label="多重映射", order=8, mutable=False)
        CompoundType = List(arg_type="List", label="复合类型", order=9, mutable=False)
        
        def __QS_initArgValue__(self, args={}):
            if args.get("复合类型", []) or args.get("多重映射", False): args["数据类型"] = "object"
            Arity = args.get("入参数", self.Arity)
            MaxArity = args.get("最大入参数", self.MaxArity)
            if (MaxArity!=0) and (Arity>MaxArity):
                raise __QS_Error__(f"最小入参数必须小于等于最大入参数!")
            return super().__QS_initArgValue__(args=args)
    
    def _QS_checkArity(self, *x):
        Arity = len(x)
        if self._QSArgs.MaxArity==0:
            if Arity!=self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 和指定的入参数 {self._QSArgs.Arity} 不符!")
        else:
            if Arity > self._QSArgs.MaxArity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 大于最大入参数 {self._QSArgs.MaxArity}!")
            elif Arity < self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 小于最小入参数 {self._QSArgs.Arity}!")
        return (True, None)
    
    def _QS_makeOperator(self, *x, args:dict={}):
        isOK, Msg = self._QS_checkArity(*x)
        if not isOK: raise __QS_Error__(Msg)
        if not args: return self
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
    
    def calculate(self, f, idt, iid, x, args):
        raise NotImplementedError
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None):
        raise NotImplementedError
    
    def __call__(self, *x, factor_name:Optional[str]=None, args:dict={}, factor_args:dict={}):
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
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=8, option_range=["单ID", "多ID"], mutable=False)
    
    def __call__(self, *x, factor_name:Optional[str]=None, args:dict={}, factor_args:dict={}):
        Operator = self._QS_makeOperator(*x, args=args)
        return PointOperation(name=(Operator._QSArgs.Name if not factor_name else factor_name), descriptors=x, sys_args={"算子": Operator, **factor_args})
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None):
        ModelArgs = self._QSArgs.ModelArgs.copy()
        ModelArgs.update(factor.Args.ModelArgs)
        if self._QSArgs.InputFormat == "numpy":
            if (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='多ID'):
                return self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
            if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
            else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
            if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
                for i, iDT in enumerate(dts):
                    for j, jID in enumerate(ids):
                        StdData[i, j] = self.calculate(factor, iDT, jID, [iData[i, j] for iData in descriptor_data], ModelArgs)
            elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
                for j, jID in enumerate(ids):
                    StdData[:, j] = self.calculate(factor, dts, jID, [iData[:, j] for iData in descriptor_data], ModelArgs)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
                for i, iDT in enumerate(dts):
                    StdData[i, :] = self.calculate(factor, iDT, ids, [iData[i, :] for iData in descriptor_data], ModelArgs)
            return StdData
        else:
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).to_frame(filter_observations=False).sort_index(axis=1)
            descriptor_data = self._QS_Compound2Frame(descriptor_data, self._QSArgs.DescriptorCompoundType)
            if self._QSArgs.ExpandDescriptors:
                descriptor_data, iOtherData = descriptor_data.iloc[:, self._QSArgs.ExpandDescriptors], descriptor_data.loc[:, descriptor_data.columns.difference(descriptor_data.columns[self._QSArgs.ExpandDescriptors])]
                descriptor_data = expandListElementDataFrame(descriptor_data, expand_index=True)
                descriptor_data = descriptor_data.set_index(descriptor_data.columns[:2].tolist())
                if not iOtherData.empty:
                    descriptor_data.index, iOtherData.index = descriptor_data.index.rename(("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                    descriptor_data = pd.merge(descriptor_data, iOtherData, how="left", left_index=True, right_index=True)
                descriptor_data = descriptor_data.sort_index(axis=1)
            if self._QSArgs.CompoundType:
                CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
            else:
                CompoundCols = None
            if (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='多ID'):
                StdData = self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
                if self._QSArgs.DataType == 'double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
                else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
                for i, iDT in enumerate(dts):
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
                StdData = []
                for j, jID in enumerate(ids):
                    iStdData = self.calculate(factor, dts, jID, descriptor_data.loc[jID], ModelArgs)
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
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
                StdData = []
                for i, iDT in enumerate(dts):
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
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
    
    
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
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=8, option_range=["单ID", "多ID"], mutable=False)
        LookBack = List(arg_type="ArgList", label="回溯期数", order=9, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=10, mutable=False)# 描述子的回溯模式
        StartDT = List(arg_type="ArgList", label="起始时点", order=10.5, mutable=False)# 扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数
        iLookBack = Int(0, arg_type="Integer", label="自身回溯期数", order=11, mutable=False)
        iLookBackMode = Enum("滚动窗口", "扩张窗口", arg_type="SingleOption", label="自身回溯模式", order=12, option_range=["滚动窗口", "扩张窗口"], mutable=False)
        iInitData = Instance(pd.DataFrame, arg_type="DataFrame", label="自身初始值", order=13)
        
        def __QS_initArgValue__(self, args={}):
            if "回溯期数" in args:
                args = args.copy()
                if "回溯模式" not in args:
                    args["回溯模式"] = ["滚动窗口"] * len(args["回溯期数"])
                if "起始时点" not in args:
                    args["起始时点"] = [None] * len(args["回溯期数"])
            return super().__QS_initArgValue__(args=args)
    
    def _QS_makeOperator(self, *x, args:dict={}):
        args = args.copy()
        LookBack = args.get("回溯期数", self._QSArgs.LookBack)
        if not LookBack: args["回溯期数"] = [0] * len(x)
        elif len(LookBack)<len(x): raise  __QS_Error__("时序算子 '%s'(QSID: %s) 的参数 '回溯期数' 序列长度小于描述子个数!" % (self._QSArgs.Name, self.QSID))
        LookBackMode = args.get("回溯模式", self._QSArgs.LookBackMode)
        if not LookBackMode: args["回溯模式"] = ["滚动窗口"] * len(x)
        elif len(LookBackMode)<len(x): raise  __QS_Error__("时序算子 '%s'(QSID: %s) 的参数 '回溯模式' 序列长度小于描述子个数!" % (self._QSArgs.Name, self.QSID))
        StartDT = args.get("起始时点", self._QSArgs.StartDT)
        if not StartDT: args["起始时点"] = [None] * len(x)
        elif len(StartDT)<len(x): raise  __QS_Error__("时序算子 '%s'(QSID: %s) 的参数 '起始时点' 序列长度小于描述子个数!" % (self._QSArgs.Name, self.QSID))
        return super()._QS_makeOperator(*x, args=args)
    
    def __call__(self, *x, factor_name:Optional[str]=None, args:dict={}, factor_args:dict={}):
        Operator = self._QS_makeOperator(*x, args=args)
        return TimeOperation(name=(self._QSArgs.Name if not factor_name else factor_name), descriptors=x, sys_args={"算子": Operator, **factor_args})
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None):
        if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        StartInd, EndInd = dt_ruler.index(dts[0]), dt_ruler.index(dts[-1])
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
        for i in range(len(descriptor_data)):
            iLookBack = self._QSArgs.LookBack[i]
            if (self._QSArgs.LookBackMode[i]=="滚动窗口") and (self._QSArgs.StartDT[i] is None):
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                if self._QSArgs.StartDT[i] is not None:
                    iLookBack = max(0, StartInd - dt_ruler.index(self._QSArgs.StartDT[i]))
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        iStartInd = 0
        if (self._QSArgs.iLookBackMode=="扩张窗口") or (self._QSArgs.iLookBack!=0):
            if self._QSArgs.iInitData is not None:
                iInitData = self._QSArgs.iInitData.loc[self._QSArgs.iInitData.index<dts[0], :]
                if iInitData.shape[0]>0:
                    iInitData = iInitData.reindex(columns=ids).values.astype(StdData.dtype)
                    iStartInd = min(self._QSArgs.iLookBack, iInitData.shape[0])
                    StdData = np.r_[iInitData[-iStartInd:], StdData]
            if self._QSArgs.iLookBackMode=="扩张窗口":
                StartIndAndLen.insert(0, (iStartInd-1, np.inf))
                MaxLen = np.inf
            else:
                StartIndAndLen.insert(0, (iStartInd-1, self._QSArgs.iLookBack))
                MaxLen = max(MaxLen, self._QSArgs.iLookBack+1)
            MaxLookBack = max(MaxLookBack, self._QSArgs.iLookBack)
            descriptor_data.insert(0, StdData)
        
        if StartInd>=MaxLookBack: DTRuler = dt_ruler[StartInd-MaxLookBack:EndInd+1]
        else: DTRuler = [None]*(MaxLookBack-StartInd) + dt_ruler[:EndInd+1]
        ModelArgs = self._QSArgs.ModelArgs
        ModelArgs.update(factor.Args.ModelArgs)
        if self._QSArgs.InputFormat == "numpy":
            if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
                for i, iDT in enumerate(dts):
                    iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                    for j, jID in enumerate(ids):
                        x = []
                        for k, kDescriptorData in enumerate(descriptor_data):
                            kStartInd, kLen = StartIndAndLen[k]
                            x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i, j])
                        StdData[iStartInd+i, j] = self.calculate(factor, iDTs, jID, x, ModelArgs)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
                for i, iDT in enumerate(dts):
                    iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                    x = []
                    for k,kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                    StdData[iStartInd+i, :] = self.calculate(factor, iDTs, ids, x, ModelArgs)
            elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
                for j, jID in enumerate(ids):
                    StdData[iStartInd:, j] = self.calculate(factor, DTRuler, jID, [kDescriptorData[:, j] for kDescriptorData in descriptor_data], ModelArgs)
            else:
                return self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)
            return StdData[iStartInd:, :]
        else:
            StdData = pd.DataFrame(StdData, columns=ids, index=DTRuler[-StdData.shape[0]:])
            if (self._QSArgs.iLookBackMode=="扩张窗口") or (self._QSArgs.iLookBack!=0):
                descriptor_data[0] = StdData
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).loc[:, DTRuler].to_frame(filter_observations=False).sort_index(axis=1)
            descriptor_data = self._QS_Compound2Frame(descriptor_data, self._QSArgs.DescriptorCompoundType)
            if self._QSArgs.ExpandDescriptors:
                descriptor_data, iOtherData = descriptor_data.iloc[:, self._QSArgs.ExpandDescriptors], descriptor_data.loc[:, descriptor_data.columns.difference(descriptor_data.columns[self._QSArgs.ExpandDescriptors])]
                descriptor_data = expandListElementDataFrame(descriptor_data, expand_index=True)
                descriptor_data = descriptor_data.set_index(descriptor_data.columns[:2].tolist())
                if not iOtherData.empty:
                    descriptor_data.index, iOtherData.index = descriptor_data.index.rename(("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                    descriptor_data = pd.merge(descriptor_data, iOtherData, how="left", left_index=True, right_index=True)
                descriptor_data = descriptor_data.sort_index(axis=1)
            if self._QSArgs.CompoundType:
                CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
            else:
                CompoundCols = None
            if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
                StdData = StdData.values
                descriptor_data = descriptor_data.swaplevel(axis=0)
                for j, jID in enumerate(ids):
                    jDescriptorData = descriptor_data.loc[jID]
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                        iStdData = self.calculate(factor, iDTs, jID, jDescriptorData.loc[iDTs], ModelArgs)
                        if isinstance(iStdData, pd.DataFrame):
                            iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                        elif isinstance(iStdData, pd.Series):
                            iStdData = tuple(iStdData.reindex(index=CompoundCols))
                        StdData[iStartInd + i, j] = iStdData
                return pd.DataFrame(StdData[iStartInd:, :], index=dts, columns=ids)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
                StdData = []
                for i, iDT in enumerate(dts):
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
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            else:
                StdData = self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)


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
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=8, option_range=["全截面", "单ID"], mutable=False)
    
    def __call__(self, *x, factor_name:Optional[str]=None, args:dict={}, factor_args:dict={}):
        Operator = self._QS_makeOperator(*x, args=args)
        return SectionOperation(name=(self._QSArgs.Name if not factor_name else factor_name), descriptors=x, sys_args={"算子": Operator, **factor_args})
        
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None):
        ModelArgs = self._QSArgs.ModelArgs.copy()
        ModelArgs.update(factor._QSArgs.ModelArgs)
        if self._QSArgs.InputFormat == "numpy":
            if self._QSArgs.DataType=="double": StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype="float")
            else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype="O")
            if self._QSArgs.OutputMode=="全截面":
                if self._QSArgs.DTMode=="单时点":
                    for i, iDT in enumerate(dts):
                        StdData[i, :] = self.calculate(factor, iDT, ids, [kDescriptorData[i] for kDescriptorData in descriptor_data], ModelArgs)
                else:
                    StdData = self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
            else:
                if self._QSArgs.DTMode=="单时点":
                    for i, iDT in enumerate(dts):
                        x = [kDescriptorData[i] for kDescriptorData in descriptor_data]
                        for j, jID in enumerate(ids):
                            StdData[i, j] = self.calculate(factor, iDT, jID, x, ModelArgs)
                else:
                    for j, jID in enumerate(ids):
                        StdData[:, j] = self.calculate(factor, dts, jID, descriptor_data, ModelArgs)
            return StdData
        else:
            SectionIdx = self._QS_partitionSectionIDs(self._QSArgs.DescriptorSection)
            DescriptorData = []
            for iSectionIDs, iIdx in SectionIdx:
                iDescriptorData = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data)) if i in iIdx}).to_frame(filter_observations=False).sort_index(axis=1)
                iDescriptorData = self._QS_Compound2Frame(iDescriptorData, [self._QSArgs.DescriptorCompoundType[i] for i in range(len(descriptor_data)) if i in iIdx])
                iExpandDescriptors = sorted(f"d{i}" for i in set(self._QSArgs.ExpandDescriptors).intersection(iIdx))
                if iExpandDescriptors:
                    iDescriptorData, iOtherData = iDescriptorData.loc[:, iExpandDescriptors], iDescriptorData.loc[:, iDescriptorData.columns.difference(iExpandDescriptors)]
                    iDescriptorData = expandListElementDataFrame(iDescriptorData, expand_index=True)
                    iDescriptorData = iDescriptorData.set_index(iDescriptorData.columns[:2].tolist())
                    if not iOtherData.empty:
                        iDescriptorData.index, iOtherData.index = iDescriptorData.index.rename(("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                        iDescriptorData = pd.merge(iDescriptorData, iOtherData, how="left", left_index=True, right_index=True)
                iDescriptorData = iDescriptorData.sort_index(axis=1)
                DescriptorData.append(iDescriptorData)
            descriptor_data, DescriptorData = DescriptorData, None
            if self._QSArgs.CompoundType:
                CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
            else:
                CompoundCols = None
            if self._QSArgs.OutputMode=="全截面":
                if self._QSArgs.DTMode=="单时点":
                    StdData = []
                    for i, iDT in enumerate(dts):
                        iStdData = self.calculate(factor, iDT, ids, [iData.loc[iDT] for iData in descriptor_data], ModelArgs)
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
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
                else:
                    StdData = self.calculate(factor, dts, ids, descriptor_data, ModelArgs)
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            else:
                if self._QSArgs.DTMode=="单时点":
                    if self._QSArgs.DataType == "double": StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype="float")
                    else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype="O")
                    for i, iDT in enumerate(dts):
                        iDescriptorData = [iData.loc[iDT] for iData in descriptor_data]
                        for j, jID in enumerate(ids):
                            iStdData = self.calculate(factor, iDT, jID, iDescriptorData, ModelArgs)
                            if isinstance(iStdData, pd.DataFrame):
                                iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                            elif isinstance(iStdData, pd.Series):
                                iStdData = tuple(iStdData.reindex(index=CompoundCols))
                            StdData[i, j] = iStdData
                    return pd.DataFrame(StdData, index=dts, columns=ids)
                else:
                    StdData = []
                    for j, jID in enumerate(ids):
                        iStdData = self.calculate(factor, dts, jID, descriptor_data, ModelArgs)
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
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)


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
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=8, option_range=["全截面", "单ID"], mutable=False)
        LookBack = List(arg_type="ArgList", label="回溯期数", order=9, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=10, mutable=False)
        StartDT = List(arg_type="ArgList", label="起始时点", order=10.5, mutable=False)# 扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数
        iLookBack = Int(0, arg_type="Integer", label="自身回溯期数", order=11, mutable=False)
        iLookBackMode = Enum("滚动窗口", "扩张窗口", arg_type="SingleOption", label="自身回溯模式", order=12, option_range=["滚动窗口", "扩张窗口"], mutable=False)
        iInitData = Instance(pd.DataFrame, arg_type="DataFrame", label="自身初始值", order=13)
        
        def __QS_initArgValue__(self, args={}):
            if "回溯期数" in args:
                args = args.copy()
                if "回溯模式" not in args:
                    args["回溯模式"] = ["滚动窗口"] * len(args["回溯期数"])
                if "起始时点" not in args:
                    args["起始时点"] = [None] * len(args["回溯期数"])
            return super().__QS_initArgValue__(args=args)
    
    def _QS_makeOperator(self, *x, args:dict={}):
        args = args.copy()
        LookBack = args.get("回溯期数", self._QSArgs.LookBack)
        if not LookBack: args["回溯期数"] = [0] * len(x)
        elif len(LookBack)<len(x): raise  __QS_Error__("面板算子 '%s'(QSID: %s) 的参数 '回溯期数' 序列长度小于描述子个数!" % (self._QSArgs.Name, self.QSID))
        LookBackMode = args.get("回溯模式", self._QSArgs.LookBackMode)
        if not LookBackMode: args["回溯模式"] = ["滚动窗口"] * len(x)
        elif len(LookBackMode)<len(x): raise  __QS_Error__("面板算子 '%s'(QSID: %s) 的参数 '回溯模式' 序列长度小于描述子个数!" % (self._QSArgs.Name, self.QSID))
        StartDT = args.get("起始时点", self._QSArgs.StartDT)
        if not StartDT: args["起始时点"] = [None] * len(x)
        elif len(StartDT)<len(x): raise  __QS_Error__("面板算子 '%s'(QSID: %s) 的参数 '起始时点' 序列长度小于描述子个数!" % (self._QSArgs.Name, self.QSID))        
        return super()._QS_makeOperator(*x, args=args)
    
    def __call__(self, *x, factor_name:Optional[str]=None, args:dict={}, factor_args:dict={}):
        Operator = self._QS_makeOperator(*x, args=args)
        return PanelOperation(name=(self._QSArgs.Name if not factor_name else factor_name), descriptors=x, sys_args={"算子": Operator, **factor_args})
    
    def calcData(self, factor, ids, dts, descriptor_data, dt_ruler=None):
        if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        StartInd, EndInd = dt_ruler.index(dts[0]), dt_ruler.index(dts[-1])
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
        for i, iDescriptor in enumerate(factor._Descriptors):
            iLookBack = self._QSArgs.LookBack[i]
            if (self._QSArgs.LookBackMode[i]=="滚动窗口") and (self._QSArgs.StartDT[i] is None):
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
                if self._QSArgs.StartDT[i] is not None:
                    iLookBack = max(0, StartInd - dt_ruler.index(self._QSArgs.StartDT[i]))
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        iStartInd = 0
        if (self._QSArgs.iLookBackMode=="扩张窗口") or (self._QSArgs.iLookBack!=0):
            if self._QSArgs.iInitData is not None:
                iInitData = self._QSArgs.iInitData.loc[self._QSArgs.iInitData.index<dts[0], :]
                if iInitData.shape[0]>0:
                    iInitData = iInitData.reindex(columns=ids).values.astype(StdData.dtype)
                    iStartInd = min(self._QSArgs.iLookBack, iInitData.shape[0])
                    StdData = np.r_[iInitData[-iStartInd:], StdData]
            if self._QSArgs.iLookBackMode=="扩张窗口":# 自身为扩张窗口模式
                StartIndAndLen.insert(0, (iStartInd-1, np.inf))
                MaxLen = np.inf
            else:# 自身为滚动窗口模式
                StartIndAndLen.insert(0, (iStartInd-1, self._QSArgs.iLookBack))
                MaxLen = max(MaxLen, self._QSArgs.iLookBack+1)
            descriptor_data.insert(0, StdData)
            MaxLookBack = max(MaxLookBack, self._QSArgs.iLookBack)
        if StartInd>=MaxLookBack: DTRuler = dt_ruler[StartInd-MaxLookBack:EndInd+1]
        else: DTRuler = [None]*(MaxLookBack-StartInd) + dt_ruler[:EndInd+1]
        ModelArgs = self._QSArgs.ModelArgs.copy()
        ModelArgs.update(factor._QSArgs.ModelArgs)
        if self._QSArgs.InputFormat == "numpy":
            if self._QSArgs.OutputMode=='全截面':
                if self._QSArgs.DTMode=='单时点':
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                        x = []
                        for k, kDescriptorData in enumerate(descriptor_data):
                            kStartInd, kLen = StartIndAndLen[k]
                            x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                        StdData[iStartInd+i, :] = self.calculate(factor, iDTs, ids, x, ModelArgs)
                else:
                    return self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)
            else:
                if self._QSArgs.DTMode=='单时点':
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                        x = []
                        for k, kDescriptorData in enumerate(descriptor_data):
                            kStartInd, kLen = StartIndAndLen[k]
                            x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                        for j, jID in enumerate(ids):
                            StdData[iStartInd+i, j] = self.calculate(factor, iDTs, jID, x, ModelArgs)
                else:
                    for j, jID in enumerate(ids):
                        StdData[iStartInd:, j] = self.calculate(factor, DTRuler, jID, descriptor_data, ModelArgs)
            return StdData[iStartInd:, :]
        else:
            StdData = pd.DataFrame(StdData, columns=ids, index=DTRuler[-StdData.shape[0]:])
            if (self._QSArgs.iLookBackMode=="扩张窗口") or (self._QSArgs.iLookBack!=0):
                descriptor_data[0] = StdData
            SectionIdx = self._QS_partitionSectionIDs(factor._QSArgs.DescriptorSection)
            DescriptorData = []
            for iSectionIDs, iIdx in SectionIdx:
                iDescriptorData = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data)) if i in iIdx}).loc[:, DTRuler].to_frame(filter_observations=False).sort_index(axis=1)
                iDescriptorData = self._QS_Compound2Frame(iDescriptorData, [self._QSArgs.DescriptorCompoundType[i] for i in range(len(descriptor_data)) if i in iIdx])
                iExpandDescriptors = sorted(f"d{i}" for i in set(self._QSArgs.ExpandDescriptors).intersection(iIdx))
                if iExpandDescriptors:
                    iDescriptorData, iOtherData = iDescriptorData.loc[:, iExpandDescriptors], iDescriptorData.loc[:, iDescriptorData.columns.difference(iExpandDescriptors)]
                    iDescriptorData = expandListElementDataFrame(iDescriptorData, expand_index=True)
                    iDescriptorData = iDescriptorData.set_index(iDescriptorData.columns[:2].tolist())
                    if not iOtherData.empty:
                        iDescriptorData.index, iOtherData.index = iDescriptorData.index.rename(("DT", "ID")), iOtherData.index.rename(("DT", "ID"))
                        iDescriptorData = pd.merge(iDescriptorData, iOtherData, how="left", left_index=True, right_index=True)
                iDescriptorData = iDescriptorData.sort_index(axis=1)
                DescriptorData.append(iDescriptorData)
            descriptor_data, DescriptorData = DescriptorData, None
            if self._QSArgs.CompoundType:
                CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
            else:
                CompoundCols = None
            if self._QSArgs.OutputMode=='全截面':
                if self._QSArgs.DTMode=='单时点':
                    StdData = []
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
                        iStdData = self.calculate(factor, iDTs, ids, [iData.loc[iDTs] for iData in descriptor_data], ModelArgs)
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
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
                else:
                    StdData = self.calculate(factor, DTRuler, ids, descriptor_data, ModelArgs)
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            else:
                if self._QSArgs.DTMode=='单时点':
                    StdData = StdData.values
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
                        for j, jID in enumerate(ids):
                            iStdData = self.calculate(factor, iDTs, jID, [iData.loc[iDTs] for iData in descriptor_data], ModelArgs)
                            if isinstance(iStdData, pd.DataFrame):
                                iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                            elif isinstance(iStdData, pd.Series):
                                iStdData = tuple(iStdData.reindex(index=CompoundCols))
                            StdData[iStartInd + i, j] = iStdData
                    return pd.DataFrame(StdData[iStartInd:, :], index=dts, columns=ids)
                else:
                    descriptor_data = descriptor_data.swaplevel(axis=0)
                    StdData = []
                    for j, jID in enumerate(ids):
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
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)


# 算子工厂函数
# operator_type: 算子类型, 可选: 'Point', 'Time', 'Section', 'Panel'
# sys_args: 算子参数
def makeFactorOperator(operator_type, func, sys_args={}, **kwargs):
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
    return partial(makeFactorOperator, operator_type, sys_args=sys_args)

class DerivativeFactor(Factor):
    """衍生因子"""
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Operator = Instance(FactorOperator, arg_type="QSObject", label="算子", order=0, mutable=False)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
        Meta = Dict(arg_type="Dict", label="元信息", order=2)
    
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        self._Descriptors = descriptors
        self.UserData = {}
        if descriptors: kwargs.setdefault("logger", descriptors[0]._QS_Logger)
        super().__init__(name=name, ft=None, sys_args=sys_args, config_file=None, **kwargs)
        self._Operator = self._QSArgs.Operator
    
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
    
    def start(self, dts, **kwargs):
        for iDescriptor in self._Descriptors: iDescriptor.start(dts=dts, **kwargs)
        return 0
    
    def end(self):
        for iDescriptor in self._Descriptors: iDescriptor.end()
        return 0
    

class PointOperation(DerivativeFactor):
    """单点运算"""
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Point", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, PointOperator): raise __QS_Error__(f"类型为 PointOperation 的衍生因子 {name} 的算子类型必须为 PointOperator, 但传入的算子类型为 {Operator.__class__}")
        FactorArgs = {
            "参数": sys_args.pop("参数", {}),
            "描述信息": sys_args.pop("描述信息", ""),
        }
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **FactorArgs}, **kwargs)
        
    def readData(self, ids, dts, **kwargs):
        if self._Operator._QSArgs.InputFormat=="numpy":
            StdData = self._Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs).values for iDescriptor in self._Descriptors])
            return pd.DataFrame(StdData, index=dts, columns=ids)
        else:
            StdData = self._Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs) for iDescriptor in self._Descriptors])
            return StdData
    
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        super()._QS_initOperation(start_dt, dt_dict, prepare_ids, id_dict)
        for i, iDescriptor in enumerate(self._Descriptors):
            iDescriptor._QS_initOperation(dt_dict[self.QSID], dt_dict, prepare_ids, id_dict)
    
    def __QS_prepareCacheData__(self, ids=None):
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = list(self._OperationMode.DTRuler[StartInd:EndInd+1])
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None:
            IDs = list(self._OperationMode._PID_IDs[PID])
        else:
            IDs = partitionListMovingSampling(IDs, len(self._OperationMode._PIDs))[self._OperationMode._PIDs.index(PID)]
        if IDs:
            if self._QSArgs.InputFormat=="numpy":
                StdData = self._Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=[PID]).values for iDescriptor in self._Descriptors])
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = self._Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=[PID]) for iDescriptor in self._Descriptors])
        else:
            for iDescriptor in self._Descriptors:
                iDescriptor._QS_getData(DTs, pids=[PID])
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if self._Operator._QSArgs.DataType=="double" else "O"))
        self._OperationMode._Cache.writeFactorData(self.Name+str(self._OperationMode._FactorID[self.Name]), StdData, pid=PID, pid_ids={PID: IDs})
        self._isCacheDataOK = True
        return StdData

class TimeOperation(DerivativeFactor):
    """时间序列运算"""
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Time", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, TimeOperator): raise __QS_Error__(f"类型为 TimeOperation 的衍生因子 {name} 的算子类型必须为 TimeOperator, 但传入的算子类型为 {Operator.__class__}")
        FactorArgs = {
            "参数": sys_args.pop("参数", {}),
            "描述信息": sys_args.pop("描述信息", ""),
        }
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **FactorArgs}, **kwargs)

    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        super()._QS_initOperation(start_dt, dt_dict, prepare_ids, id_dict)
        StartDT = dt_dict[self.QSID]
        DTRuler = list(self._OperationMode.DTRuler)
        StartInd = DTRuler.index(StartDT)
        if (self._Operator._QSArgs.iLookBackMode=="扩张窗口") and (self._Operator._QSArgs.iInitData is not None) and (self._Operator._QSArgs.iInitData.shape[0]>0):
            if self._Operator._QSArgs.iInitData.index[-1] not in DTRuler: self._QS_Logger.warning("注意: 因子 '%s'(QSID: %s) 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, self.QSID))
            else: StartInd = min(StartInd, DTRuler.index(self._Operator._QSArgs.iInitData.index[-1]) + 1)
        for i, iDescriptor in enumerate(self._Descriptors):
            if self._Operator._QSArgs.StartDT[i] is None:# 未指定起始时点
                iStartInd = StartInd - self._Operator._QSArgs.LookBack[i]
                if iStartInd<0: self._QS_Logger.warning("注意: 对于因子 '%s'(QSID: %s) 的描述子 '%s'(QSID: %s), 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, self.QSID, iDescriptor.Name, iDescriptor.QSID))
                iStartDT = DTRuler[max(0, iStartInd)]
            else:
                iStartDT = self._Operator._QSArgs.StartDT[i]
                if iStartDT<DTRuler[0]: self._QS_Logger.warning(f"注意: 对于因子 '{self.Name}'(QSID: {self.QSID}) 的第 {i} 个描述子起始时点 {iStartDT} 小于时点标尺的起始时点 {DTRuler[0]}, 时点标尺长度不足!")
                iStartDT = max(iStartDT, DTRuler[0])
            iDescriptor._QS_initOperation(iStartDT, dt_dict, prepare_ids, id_dict)
    
    def readData(self, ids, dts, **kwargs):
        DTRuler = kwargs.get("dt_ruler", dts)
        Operator = self._Operator
        StartInd = (DTRuler.index(dts[0]) if dts[0] in DTRuler else 0)
        if (Operator._QSArgs.iLookBackMode=="扩张窗口") and (Operator._QSArgs.iInitData is not None) and (Operator._QSArgs.iInitData.shape[0]>0):
            if Operator._QSArgs.iInitData.index[-1] not in DTRuler: self._QS_Logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, ))
            else: StartInd = min(StartInd, DTRuler.index(Operator._QSArgs.iInitData.index[-1]) + 1)
        EndInd = (DTRuler.index(dts[-1]) if dts[-1] in DTRuler else len(DTRuler)-1)
        if StartInd>EndInd: return pd.DataFrame(index=dts, columns=ids)
        nID = len(ids)
        DescriptorData = []
        if Operator._QSArgs.InputFormat == "numpy":
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-Operator._QSArgs.LookBack[i], 0):EndInd+1]
                if iDTs: iDescriptorData = iDescriptor.readData(ids=ids, dts=iDTs, **kwargs).values
                else: iDescriptorData = np.full((0, nID), np.nan)
                if StartInd<Operator._QSArgs.LookBack[i]:
                    iLookBackData = np.full((Operator._QSArgs.LookBack[i]-StartInd, nID), np.nan)
                    iDescriptorData = np.r_[iLookBackData, iDescriptorData]
                DescriptorData.append(iDescriptorData)
            StdData = Operator.calcData(factor=self, ids=ids, dts=DTRuler[StartInd:EndInd+1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return pd.DataFrame(StdData, index=DTRuler[StartInd:EndInd+1], columns=ids).reindex(index=dts)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-Operator._QSArgs.LookBack[i], 0):EndInd+1]
                if iDTs: iDescriptorData = iDescriptor.readData(ids=ids, dts=iDTs, **kwargs)
                else: iDescriptorData = pd.DataFrame(columns=ids)
                DescriptorData.append(iDescriptorData)
            StdData = Operator.calcData(factor=self, ids=ids, dts=DTRuler[StartInd:EndInd+1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return StdData.reindex(index=dts)
    
    def __QS_prepareCacheData__(self, ids=None):
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        DTRuler = list(self._OperationMode.DTRuler)
        StartInd, EndInd = DTRuler.index(StartDT), DTRuler.index(EndDT)
        DTs = DTRuler[StartInd:EndInd+1]
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None: IDs = list(self._OperationMode._PID_IDs[PID])
        else:
            IDs = partitionListMovingSampling(IDs, len(self._OperationMode._PIDs))[self._OperationMode._PIDs.index(PID)]
        Operator = self._Operator
        if IDs:
            DescriptorData = []
            if Operator._QSArgs.InputFormat=="numpy":
                for i, iDescriptor in enumerate(self._Descriptors):
                    if Operator._QSArgs.StartDT[i] is None:
                        iStartInd = StartInd - Operator._QSArgs.LookBack[i]
                        iDTs = DTRuler[max(0, iStartInd):StartInd] + DTs
                    else:
                        iStartDT = max(Operator._QSArgs.StartDT[i], DTRuler[0])
                        iStartInd = DTRuler.index(iStartDT)
                        iDTs = DTRuler[iStartInd:DTRuler.index(DTs[-1])]
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=[PID]).values
                    if iStartInd<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
                    DescriptorData.append(iDescriptorData)
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                for i, iDescriptor in enumerate(self._Descriptors):
                    if Operator._QSArgs.StartDT[i] is None:
                        iStartInd = StartInd - Operator._QSArgs.LookBack[i]
                        iDTs = DTRuler[max(0, iStartInd):StartInd] + DTs
                    else:
                        iStartDT = max(Operator._QSArgs.StartDT[i], DTRuler[0])
                        iStartInd = DTRuler.index(iStartDT)
                        iDTs = DTRuler[iStartInd:DTRuler.index(DTs[-1])]
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=[PID])
                    DescriptorData.append(iDescriptorData)
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                if Operator._QSArgs.StartDT[i] is None:
                    iStartInd = StartInd - Operator._QSArgs.LookBack[i]
                    iDTs = DTRuler[max(0, iStartInd):StartInd] + DTs
                else:
                    iStartDT = max(Operator._QSArgs.StartDT[i], DTRuler[0])
                    iStartInd = DTRuler.index(iStartDT)
                    iDTs = DTRuler[iStartInd:DTRuler.index(DTs[-1])]
                iDescriptor._QS_getData(iDTs, pids=[PID])
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        self._OperationMode._Cache.writeFactorData(self.Name+str(self._OperationMode._FactorID[self.Name]), StdData, pid=PID, pid_ids={PID: IDs})
        self._isCacheDataOK = True
        return StdData

class SectionOperation(DerivativeFactor):
    """截面运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DescriptorSection = List(arg_type="List", label="描述子截面", order=9, mutable=False)
        
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.DescriptorSection = [None] * len(self._Owner._Descriptors)
    
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Section", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, SectionOperator): raise __QS_Error__(f"类型为 SectionOperation 的衍生因子 {name} 的算子类型必须为 SectionOperator, 但传入的算子类型为 {Operator.__class__}")
        FactorArgs = {
            "参数": sys_args.pop("参数", {}),
            "描述信息": sys_args.pop("描述信息", ""),
            "描述子截面": sys_args.pop("描述子截面", [None] * len(descriptors)),
        }
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **FactorArgs}, **kwargs)
            
    def readData(self, ids, dts, **kwargs):
        Operator = self._Operator
        SectionIDs = kwargs.pop("section_ids", ids)
        DescriptorData = []
        if Operator._QSArgs.InputFormat == "numpy":
            for i, iDescriptor in enumerate(self._Descriptors):
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                DescriptorData.append(iDescriptor.readData(ids=iSectionIDs, dts=dts, **kwargs).values)
            StdData = Operator.calcData(factor=self, ids=SectionIDs, dts=dts, descriptor_data=DescriptorData)
            return pd.DataFrame(StdData, index=dts, columns=SectionIDs).reindex(columns=ids)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                DescriptorData.append(iDescriptor.readData(ids=iSectionIDs, dts=dts, **kwargs))            
            StdData = Operator.calcData(factor=self, ids=SectionIDs, dts=dts, descriptor_data=DescriptorData)
            return StdData.reindex(columns=ids)
    
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        OldStartDT = dt_dict.get(self.QSID, None)
        if (OldStartDT is None) or (start_dt<OldStartDT):
            dt_dict[self.QSID] = start_dt
            StartInd, EndInd = self._OperationMode.DTRuler.index(dt_dict[self.QSID]), self._OperationMode.DTRuler.index(self._OperationMode.DateTimes[-1])
            DTs = self._OperationMode.DTRuler[StartInd:EndInd+1]
            DTPartition = partitionList(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID:DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
        PrepareIDs = id_dict.setdefault(self.QSID, prepare_ids)
        if prepare_ids != PrepareIDs: raise __QS_Error__("因子 '%s'(QSID: %s) 指定了不同的截面!" % (self.Name, self.QSID))
        for i, iDescriptor in enumerate(self._Descriptors):
            if self._QSArgs.DescriptorSection[i] is None:
                iDescriptor._QS_initOperation(start_dt, dt_dict, prepare_ids, id_dict)
            else:
                iDescriptor._QS_initOperation(start_dt, dt_dict, self._QSArgs.DescriptorSection[i], id_dict)
        if (self._OperationMode.SubProcessNum>0) and (self.QSID not in self._OperationMode._Event):
            self._OperationMode._Event[self.QSID] = (Queue(), Event())
        
    def __QS_prepareCacheData__(self, ids=None):
        Operator = self._Operator
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None: IDs = list(self._OperationMode.IDs)
        if len(DTs)==0:# 该进程未分配到计算任务
            iDTs = [self._OperationMode.DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor._QS_getData(iDTs, pids=None)
            StdData = pd.DataFrame(columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        elif IDs:
            if Operator._QSArgs.InputFormat == "numpy":
                StdData = Operator.calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=None).values for iDescriptor in self._Descriptors])
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = Operator.calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=None) for iDescriptor in self._Descriptors])
        else:
            for iDescriptor in self._Descriptors:
                iDescriptor._QS_getData(DTs, pids=None)
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        if self._OperationMode._FactorPrepareIDs[self.Name] is None:
            PID_IDs = self._OperationMode._PID_IDs
        else:
            PID_IDs = {self._OperationMode._PIDs[i]: iSubIDs for i, iSubIDs in enumerate(partitionListMovingSampling(IDs, len(self._OperationMode._PIDs)))}
        self._OperationMode._Cache.writeFactorData(self.Name + str(self._OperationMode._FactorID[self.Name]), StdData, pid=None, pid_ids=PID_IDs)
        StdData = None# 释放数据
        gc.collect()
        if self._OperationMode.SubProcessNum>0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData

class PanelOperation(DerivativeFactor):
    """面板运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DescriptorSection = List(arg_type="List", label="描述子截面", order=9, mutable=False)
        
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.DescriptorSection = [None] * len(self._Owner._Descriptors)
    
    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        sys_args = sys_args.copy()
        Operator = sys_args.pop("算子", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator): Operator = makeFactorOperator(operator_type="Panel", func=Operator, args=sys_args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, PanelOperator): raise __QS_Error__(f"类型为 PanelOperation 的衍生因子 {name} 的算子类型必须为 PanelOperator, 但传入的算子类型为 {Operator.__class__}")
        FactorArgs = {
            "参数": sys_args.pop("参数", {}),
            "描述信息": sys_args.pop("描述信息", ""),
            "描述子截面": sys_args.pop("描述子截面", [None] * len(descriptors)),
        }
        return super().__init__(name=name, descriptors=descriptors, sys_args={"算子": Operator, **FactorArgs}, **kwargs)
        
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        Operator = self._Operator
        if len(self._Descriptors)>len(Operator._QSArgs.LookBack): raise  __QS_Error__("面板运算因子 '%s'(QSID: %s) 的参数'回溯期数'序列长度小于描述子个数!" % (self.Name, self.QSID))
        OldStartDT = dt_dict.get(self.QSID, None)
        DTRuler = self._OperationMode.DTRuler
        if (OldStartDT is None) or (start_dt<OldStartDT):
            StartDT = dt_dict[self.QSID] = start_dt
            StartInd, EndInd = DTRuler.index(StartDT), DTRuler.index(self._OperationMode.DateTimes[-1])
            if (Operator._QSArgs.iLookBackMode=="扩张窗口") and (Operator._QSArgs.iInitData is not None) and (Operator._QSArgs.iInitData.shape[0]>0):
                if Operator._QSArgs.iInitData.index[-1] not in self._OperationMode.DTRuler: self._QS_Logger.warning("注意: 因子 '%s'(QSID: %s) 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, self.QSID))
                else: StartInd = min(StartInd, self._OperationMode.DTRuler.index(Operator._QSArgs.iInitData.index[-1]) + 1)
            DTs = DTRuler[StartInd:EndInd+1]
            if Operator._QSArgs.iLookBackMode=="扩张窗口":
                DTPartition = [DTs]+[[]]*(len(self._OperationMode._PIDs)-1)
            else:
                DTPartition = partitionList(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID:DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
        else:
            StartInd = DTRuler.index(OldStartDT)
        PrepareIDs = id_dict.setdefault(self.QSID, prepare_ids)
        if prepare_ids != PrepareIDs: raise __QS_Error__("因子 '%s'(QSID: %s) 指定了不同的截面!" % (self.Name, self.QSID))
        for i, iDescriptor in enumerate(self._Descriptors):
            if Operator._QSArgs.StartDT[i] is None:# 未指定起始时点
                iStartInd = StartInd - Operator._QSArgs.LookBack[i]
                if iStartInd<0: self._QS_Logger.warning(f"注意: 对于因子 '{self.Name}'(QSID: {self.QSID}) 的第 {i} 个描述子 '{iDescriptor.Name}'(QSID: {iDescriptor.QSID}), 时点标尺长度不足!")
                iStartDT = DTRuler[max(0, iStartInd)]
            else:
                iStartDT = Operator._QSArgs.StartDT[i]
                if iStartDT<DTRuler[0]: self._QS_Logger.warning(f"注意: 对于因子 '{self.Name}'(QSID: {self.QSID}) 的第 {i} 个描述子 '{iDescriptor.Name}'(QSID: {iDescriptor.QSID}), 指定的起始时点 {iStartDT} 小于时点标尺的起始时点 {DTRuler[0]}, 时点标尺长度不足!")
                iStartDT = max(iStartDT, DTRuler[0])
            if self._QSArgs.DescriptorSection[i] is None:
                iDescriptor._QS_initOperation(iStartDT, dt_dict, prepare_ids, id_dict)
            else:
                iDescriptor._QS_initOperation(iStartDT, dt_dict, self._QSArgs.DescriptorSection[i], id_dict)
        if (self._OperationMode.SubProcessNum>0) and (self.QSID not in self._OperationMode._Event):
            self._OperationMode._Event[self.QSID] = (Queue(), Event())
    
    def readData(self, ids, dts, **kwargs):
        Operator = self._Operator
        DTRuler = kwargs.get("dt_ruler", dts)
        SectionIDs = kwargs.pop("section_ids", ids)
        StartInd = (DTRuler.index(dts[0]) if dts[0] in DTRuler else 0)
        if (Operator._QSArgs.iLookBackMode=="扩张窗口") and (Operator._QSArgs.iInitData is not None) and (Operator._QSArgs.iInitData.shape[0]>0):
            if Operator._QSArgs.iInitData.index[-1] not in DTRuler: self._QS_Logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, ))
            else: StartInd = min(StartInd, DTRuler.index(Operator._QSArgs.iInitData.index[-1]) + 1)
        EndInd = (DTRuler.index(dts[-1]) if dts[-1] in DTRuler else len(DTRuler)-1)
        if StartInd>EndInd: return pd.DataFrame(index=dts, columns=ids)
        DescriptorData = []
        if Operator._QSArgs.InputFormat == "numpy":
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-Operator._QSArgs.LookBack[i], 0):EndInd+1]
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                iIDNum = len(iSectionIDs)
                if iDTs:
                    iDescriptorData = iDescriptor.readData(ids=iSectionIDs, dts=iDTs, **kwargs).values
                else:
                    iDescriptorData = np.full((0, iIDNum), np.nan)
                if StartInd<Operator._QSArgs.LookBack[i]:
                    iLookBackData = np.full((Operator._QSArgs.LookBack[i]-StartInd, iIDNum), np.nan)
                    iDescriptorData = np.r_[iLookBackData, iDescriptorData]
                DescriptorData.append(iDescriptorData)
            StdData = Operator.calcData(factor=self, ids=SectionIDs, dts=DTRuler[StartInd:EndInd+1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return pd.DataFrame(StdData, index=DTRuler[StartInd:EndInd+1], columns=SectionIDs).reindex(index=dts, columns=ids)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-Operator._QSArgs.LookBack[i], 0):EndInd+1]
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                if iDTs:
                    iDescriptorData = iDescriptor.readData(ids=iSectionIDs, dts=iDTs, **kwargs)
                else:
                    iDescriptorData = pd.DataFrame(columns=iSectionIDs)
                DescriptorData.append(iDescriptorData)
            StdData = Operator.calcData(factor=self, ids=SectionIDs, dts=DTRuler[StartInd:EndInd + 1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return StdData.reindex(index=dts, columns=ids)
        
    def __QS_prepareCacheData__(self, ids=None):
        Operator = self._Operator
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        DTRuler = list(self._OperationMode.DTRuler)
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None: IDs = list(self._OperationMode.IDs)
        if len(DTs)==0:# 该进程未分配到计算任务
            iDTs = [self._OperationMode.DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor._QS_getData(iDTs, pids=None)
            StdData = pd.DataFrame(columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        elif IDs:
            if Operator._QSArgs.InputFormat == "numpy":
                DescriptorData, StartInd = [], DTRuler.index(DTs[0])
                for i, iDescriptor in enumerate(self._Descriptors):
                    if Operator._QSArgs.StartDT[i] is None:
                        iStartInd = StartInd - Operator._QSArgs.LookBack[i]
                        iDTs = DTRuler[max(0, iStartInd):StartInd] + DTs
                    else:
                        iStartDT = max(Operator._QSArgs.StartDT[i], DTRuler[0])
                        iStartInd = DTRuler.index(iStartDT)
                        iDTs = DTRuler[iStartInd:DTRuler.index(DTs[-1])]
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=None).values
                    if iStartInd<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]                    
                    DescriptorData.append(iDescriptorData)
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
                DescriptorData, iDescriptorData, StdData = None, None, pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                DescriptorData, StartInd = [], DTRuler.index(DTs[0])
                for i, iDescriptor in enumerate(self._Descriptors):
                    if Operator._QSArgs.StartDT[i] is None:
                        iStartInd = StartInd - Operator._QSArgs.LookBack[i]
                        iDTs = DTRuler[max(0, iStartInd):StartInd] + DTs
                    else:
                        iStartDT = max(Operator._QSArgs.StartDT[i], DTRuler[0])
                        iStartInd = DTRuler.index(iStartDT)
                        iDTs = DTRuler[iStartInd:DTRuler.index(DTs[-1])]
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=None)
                    DescriptorData.append(iDescriptorData)
                StdData = Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=DTRuler)
                DescriptorData, iDescriptorData = None, None
        else:
            DescriptorData, StartInd = [], DTRuler.index(DTs[0])
            for i, iDescriptor in enumerate(self._Descriptors):
                if Operator._QSArgs.StartDT[i] is None:
                    iStartInd = StartInd - Operator._QSArgs.LookBack[i]
                    iDTs = DTRuler[max(0, iStartInd):StartInd] + DTs
                else:
                    iStartDT = max(Operator._QSArgs.StartDT[i], DTRuler[0])
                    iStartInd = DTRuler.index(iStartDT)
                    iDTs = DTRuler[iStartInd:DTRuler.index(DTs[-1])]
                iDescriptor._QS_getData(iDTs, pids=None)
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if Operator._QSArgs.DataType=="double" else "O"))
        if self._OperationMode._FactorPrepareIDs[self.Name] is None:
            PID_IDs = self._OperationMode._PID_IDs
        else:
            PID_IDs = {self._OperationMode._PIDs[i]: iSubIDs for i, iSubIDs in enumerate(partitionListMovingSampling(IDs, len(self._OperationMode._PIDs)))}
        self._OperationMode._Cache.writeFactorData(self.Name + str(self._OperationMode._FactorID[self.Name]), StdData, pid=None, pid_ids=PID_IDs)
        StdData = None# 释放数据
        gc.collect()
        if self._OperationMode.SubProcessNum>0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData



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
    test_point = makeFactorOperator("Point", test_point, sys_args={"入参数": 2, "运算时点": "多时点", "运算ID": "多ID"})    
    Factor4 = test_point(Factor1, Factor2, factor_name="Factor4")
    
    # 装饰器方式
    @FactorOperatorized(operator_type="Time", sys_args={"入参数": 1, "运算ID": "多ID", "回溯期数": [3-1]})
    def test_time(f, idt, iid, x, args):
        return np.nansum(x[0], axis=0)
    Factor5 = test_time(Factor1, factor_name="Factor5", args={"回溯期数": [2-1]}, factor_args={"描述信息": "我是 Factor5!"})
    print(Factor5.getMetaData(key="Description"))
    
    # 直接实例化方式, 不推荐
    def test_section(f, idt, iid, x, args):
        return np.argsort(np.argsort(x[0]))
    Factor6 = SectionOperation(name="Factor6", descriptors=[Factor2], sys_args={"算子": test_section, "描述子截面": [IDs], "运算时点": "单时点"})
    
    def test_panel(f, idt, iid, x, args):
        return np.argsort(np.argsort(x[0][0]))    
    Factor7 = PanelOperation(name="Factor7", descriptors=[Factor2], sys_args={"算子": makeFactorOperator("Panel", test_panel, sys_args={"运算时点": "单时点", "回溯期数": [1-1]}), "描述子截面": [IDs]})
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    print(Factor6.readData(ids=IDs, dts=DTs))
    print(Factor7.readData(ids=IDs, dts=DTs))

    print("===")