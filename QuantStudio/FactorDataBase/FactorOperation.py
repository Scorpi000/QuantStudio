# -*- coding: utf-8 -*-
"""因子运算"""
import os
import gc
from multiprocessing import Queue, Event

import pandas as pd
import numpy as np
import sympy
from traits.api import Callable, Dict, Enum, List, ListInt, Int, Instance, Str, Either

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.Tools.AuxiliaryFun import partitionList, partitionListMovingSampling
from QuantStudio.Tools.QSObjects import Panel
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame

def _DefaultOperator(f, idt, iid, x, args):
    return np.nan

class DerivativeFactor(Factor):
    """衍生因子"""
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        Operator = Callable(default_value=_DefaultOperator, arg_type="Function", label="算子", order=0, mutable=False)
        ModelArgs = Dict(arg_type="Dict", label="参数", order=1)
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=2, option_range=["double", "string", "object"], mutable=False)
        Expression = Either(Str(""), Instance(sympy.Expr), Instance(sympy.logic.boolalg.Boolean), arg_type="String", label="表达式", order=3)
        CompoundType = List(arg_type="List", label="复合类型", order=4, mutable=False)
        InputFormat = Enum("numpy", "pandas", label="输入格式", order=5, arg_type="SingleOption", option_range=["numpy", "pandas"], mutable=False)
        # ExpandDescriptors = ListInt(arg_type="MultiOption", label="展开描述子", order=6)
        Description = Str("", label="描述信息", order=7, arg_type="String")

        def __init__(self, owner=None, sys_args={}, config_file=None, **kwargs):
            super().__init__(owner=owner, sys_args=sys_args, config_file=config_file, **kwargs)
            self._QS_Frozen = False
            if self.CompoundType: self.DataType = "object"
            self._QS_Frozen = True

        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.add_trait("ExpandDescriptors", ListInt([], arg_type="MultiOption", label="展开描述子", order=6, option_range=list(range(len(self._Owner._Descriptors))), mutable=False))

    def __init__(self, name="", descriptors=[], sys_args={}, **kwargs):
        self._Descriptors = descriptors
        self.UserData = {}
        if descriptors: kwargs.setdefault("logger", descriptors[0]._QS_Logger)
        return super().__init__(name=name, ft=None, sys_args=sys_args, config_file=None, **kwargs)
    
    @property
    def Descriptors(self):
        return self._Descriptors
    def getMetaData(self, key=None, args={}):
        DataType = args.get("数据类型", self._QSArgs.DataType)
        Description = args.get("描述信息", self._QSArgs.Description)
        if key is None: return pd.Series({"DataType": DataType, "Description": Description})
        elif key=="DataType": return DataType
        elif key=="Description": return Description
        return None
    def start(self, dts, **kwargs):
        for iDescriptor in self._Descriptors: iDescriptor.start(dts=dts, **kwargs)
        return 0
    def end(self):
        for iDescriptor in self._Descriptors: iDescriptor.end()
        return 0

    def expression(self, penetrated=False):
        if self._QSArgs.Expression:
            if isinstance(self._QSArgs.Expression, str):
                if penetrated:
                    Vars = {f"_d{i+1}": iDescriptor.expression(penetrated=True) for i, iDescriptor in enumerate(self.Descriptors)}
                else:
                    Vars = {f"_d{i+1}": sympy.Symbol(iDescriptor.Name if iDescriptor.Name else f"_d{i+1}") for i, iDescriptor in enumerate(self.Descriptors)}
                return sympy.sympify(self._QSArgs.Expression, locals=Vars)
            else:
                Expr = self._QSArgs.Expression
                for i, iDescriptor in enumerate(self.Descriptors):
                    if penetrated:
                        iVar = iDescriptor.expression(penetrated=True)
                    else:
                        iVar = sympy.Symbol(iDescriptor.Name if iDescriptor.Name else f"_d{i+1}")
                    Expr = Expr.subs(sympy.Symbol(f"_d{i+1}"), iVar)
                return Expr
        Fun = sympy.Function(self._QSArgs.Operator.__name__)
        if penetrated:
            Vars = [iDescriptor.expression(penetrated=True) for iDescriptor in self.Descriptors]
        else:
            Vars = [sympy.Symbol(iDescriptor.Name if iDescriptor.Name else f"_d{i+1}") for i, iDescriptor in enumerate(self.Descriptors)]
        return Fun(*Vars)
    
    def _QS_adjOutputPandas(self, df, cols, dts, ids):
        if isinstance(df, pd.DataFrame):
            if isinstance(df.index, pd.MultiIndex):
                if df.index.duplicated().any():
                    TmpData, Cols, df = {}, df.columns, df.groupby(axis=0, level=[0, 1], as_index=True)
                    for iCol in Cols:
                        TmpData[iCol] = df[iCol].apply(lambda s: s.tolist())
                    df, TmpData = pd.DataFrame(TmpData).loc[:, Cols], None
                df = df.reindex(columns=cols).apply(lambda s: tuple(s), axis=1).unstack()
            return df.reindex(index=dts, columns=ids)
        elif isinstance(df, pd.Series): 
            return df.unstack().reindex(index=dts, columns=ids)
        else:
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



# 单点运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算时点为多时点，则该值为[时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [单个描述子值 or array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么 x 元素为单个描述子值, 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nID, )), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID,))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么 x 元素为 array(shape=(nDT, )), 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
class PointOperation(DerivativeFactor):
    """单点运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=8, option_range=["单ID", "多ID"], mutable=False)
    def readData(self, ids, dts, **kwargs):
        if self._QSArgs.InputFormat=="numpy":
            StdData = self._calcData(ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs).values for iDescriptor in self._Descriptors])
            return pd.DataFrame(StdData, index=dts, columns=ids)
        else:
            StdData = self._calcData(ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs) for iDescriptor in self._Descriptors])
            return StdData
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        super()._QS_initOperation(start_dt, dt_dict, prepare_ids, id_dict)
        for i, iDescriptor in enumerate(self._Descriptors):
            iDescriptor._QS_initOperation(dt_dict[self.Name], dt_dict, prepare_ids, id_dict)
    def _calcData(self, ids, dts, descriptor_data):
        Operator, ModelArgs = self._QSArgs.Operator, self._QSArgs.ModelArgs
        if self._QSArgs.InputFormat == "numpy":
            if (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='多ID'):
                return Operator(self, dts, ids, descriptor_data, ModelArgs)
            if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
            else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
            if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
                for i, iDT in enumerate(dts):
                    for j, jID in enumerate(ids):
                        StdData[i, j] = Operator(self, iDT, jID, [iData[i, j] for iData in descriptor_data], ModelArgs)
            elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
                for j, jID in enumerate(ids):
                    StdData[:, j] = Operator(self, dts, jID, [iData[:, j] for iData in descriptor_data], ModelArgs)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
                for i, iDT in enumerate(dts):
                    StdData[i, :] = Operator(self, iDT, ids, [iData[i, :] for iData in descriptor_data], ModelArgs)
            return StdData
        else:
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(self._Descriptors))}).to_frame(filter_observations=False).sort_index(axis=1)
            if self._QSArgs.ExpandDescriptors:
                descriptor_data, iOtherData = descriptor_data.iloc[:, self._QSArgs.ExpandDescriptors], descriptor_data.loc[:, descriptor_data.columns.difference(descriptor_data.columns[self._QSArgs.ExpandDescriptors])]
                descriptor_data = expandListElementDataFrame(descriptor_data, expand_index=True)
                descriptor_data = descriptor_data.set_index(descriptor_data.columns[:2].tolist())
                if not iOtherData.empty:
                    descriptor_data = pd.merge(descriptor_data, iOtherData, how="left", left_index=True, right_index=True)
                descriptor_data = descriptor_data.sort_index(axis=1)
            if self._QSArgs.CompoundType:
                CompoundCols = [iCol[0] for iCol in self._QSArgs.CompoundType]
            else:
                CompoundCols = None
            if (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='多ID'):
                StdData = Operator(self, dts, ids, descriptor_data, ModelArgs)
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
                if self._QSArgs.DataType == 'double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
                else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
                for i, iDT in enumerate(dts):
                    for j, jID in enumerate(ids):
                        iStdData = Operator(self, iDT, jID, descriptor_data.loc[iDT].loc[jID], ModelArgs)
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
                    iStdData = Operator(self, dts, jID, descriptor_data.loc[jID], ModelArgs)
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
                    iStdData = Operator(self, iDT, ids, descriptor_data.loc[iDT], ModelArgs)
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
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=[PID]).values for iDescriptor in self._Descriptors])
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=[PID]) for iDescriptor in self._Descriptors])
        else:
            for iDescriptor in self._Descriptors:
                iDescriptor._QS_getData(DTs, pids=[PID])
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if self._QSArgs.DataType=="double" else "O"))
        with self._OperationMode._PID_Lock[PID]:
            with pd.HDFStore(self._OperationMode._CacheDataDir+os.sep+PID+os.sep+self.Name+str(self._OperationMode._FactorID[self.Name])+self._OperationMode._FileSuffix, mode="a") as CacheFile:
                CacheFile["StdData"] = StdData
                CacheFile["_QS_IDs"] = pd.Series(IDs)
        self._isCacheDataOK = True
        return StdData

# 时间序列运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数, )), 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数+nDT, )), 返回 array(shape=(nDate,))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数+nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
class TimeOperation(DerivativeFactor):
    """时间序列运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=8, option_range=["单ID", "多ID"], mutable=False)
        LookBack = List(arg_type="ArgList", label="回溯期数", order=9, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=10, mutable=False)# 描述子的回溯模式
        iLookBack = Int(0, arg_type="Integer", label="自身回溯期数", order=11, mutable=False)
        iLookBackMode = Enum("滚动窗口", "扩张窗口", arg_type="SingleOption", label="自身回溯模式", order=12, option_range=["滚动窗口", "扩张窗口"], mutable=False)
        iInitData = Instance(pd.DataFrame, arg_type="DataFrame", label="自身初始值", order=13)
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.LookBack = [0]*len(self._Owner._Descriptors)
            self.LookBackMode = ["滚动窗口"]*len(self._Owner._Descriptors)
    
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        super()._QS_initOperation(start_dt, dt_dict, prepare_ids, id_dict)
        if len(self._Descriptors)>len(self._QSArgs.LookBack): raise  __QS_Error__("时间序列运算因子 : '%s' 的参数'回溯期数'序列长度小于描述子个数!" % self.Name)
        StartDT = dt_dict[self.Name]
        StartInd = self._OperationMode.DTRuler.index(StartDT)
        if (self._QSArgs.iLookBackMode=="扩张窗口") and (self._QSArgs.iInitData is not None) and (self._QSArgs.iInitData.shape[0]>0):
            if self._QSArgs.iInitData.index[-1] not in self._OperationMode.DTRuler: self._QS_Logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, ))
            else: StartInd = min(StartInd, self._OperationMode.DTRuler.index(self._QSArgs.iInitData.index[-1]) + 1)
        for i, iDescriptor in enumerate(self._Descriptors):
            iStartInd = StartInd - self._QSArgs.LookBack[i]
            if iStartInd<0: self._QS_Logger.warning("注意: 对于因子 '%s' 的描述子 '%s', 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, iDescriptor.Name))
            iStartDT = self._OperationMode.DTRuler[max(0, iStartInd)]
            iDescriptor._QS_initOperation(iStartDT, dt_dict, prepare_ids, id_dict)
    def readData(self, ids, dts, **kwargs):
        DTRuler = kwargs.get("dt_ruler", dts)
        StartInd = (DTRuler.index(dts[0]) if dts[0] in DTRuler else 0)
        if (self._QSArgs.iLookBackMode=="扩张窗口") and (self._QSArgs.iInitData is not None) and (self._QSArgs.iInitData.shape[0]>0):
            if self._QSArgs.iInitData.index[-1] not in DTRuler: self._QS_Logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, ))
            else: StartInd = min(StartInd, DTRuler.index(self._QSArgs.iInitData.index[-1]) + 1)
        EndInd = (DTRuler.index(dts[-1]) if dts[-1] in DTRuler else len(DTRuler)-1)
        if StartInd>EndInd: return pd.DataFrame(index=dts, columns=ids)
        nID = len(ids)
        DescriptorData = []
        if self._QSArgs.InputFormat == "numpy":
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-self._QSArgs.LookBack[i], 0):EndInd+1]
                if iDTs: iDescriptorData = iDescriptor.readData(ids=ids, dts=iDTs, **kwargs).values
                else: iDescriptorData = np.full((0, nID), np.nan)
                if StartInd<self._QSArgs.LookBack[i]:
                    iLookBackData = np.full((self._QSArgs.LookBack[i]-StartInd, nID), np.nan)
                    iDescriptorData = np.r_[iLookBackData, iDescriptorData]
                DescriptorData.append(iDescriptorData)
            StdData = self._calcData(ids=ids, dts=DTRuler[StartInd:EndInd+1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return pd.DataFrame(StdData, index=DTRuler[StartInd:EndInd+1], columns=ids).reindex(index=dts)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-self._QSArgs.LookBack[i], 0):EndInd+1]
                if iDTs: iDescriptorData = iDescriptor.readData(ids=ids, dts=iDTs, **kwargs)
                else: iDescriptorData = pd.DataFrame(columns=ids)
                DescriptorData.append(iDescriptorData)
            StdData = self._calcData(ids=ids, dts=DTRuler[StartInd:EndInd+1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return StdData.reindex(index=dts)
    def _calcData(self, ids, dts, descriptor_data, dt_ruler):
        if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
        for i in range(len(self._Descriptors)):
            iLookBack = self._QSArgs.LookBack[i]
            if self._QSArgs.LookBackMode[i]=="滚动窗口":
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
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
        StartInd, EndInd = dt_ruler.index(dts[0]), dt_ruler.index(dts[-1])
        if StartInd>=MaxLookBack: DTRuler = dt_ruler[StartInd-MaxLookBack:EndInd+1]
        else: DTRuler = [None]*(MaxLookBack-StartInd) + dt_ruler[:EndInd+1]
        Operator, ModelArgs = self._QSArgs.Operator, self._QSArgs.ModelArgs
        if self._QSArgs.InputFormat == "numpy":
            if (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='单ID'):
                for i, iDT in enumerate(dts):
                    iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                    for j, jID in enumerate(ids):
                        x = []
                        for k, kDescriptorData in enumerate(descriptor_data):
                            kStartInd, kLen = StartIndAndLen[k]
                            x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i, j])
                        StdData[iStartInd+i, j] = Operator(self, iDTs, jID, x, ModelArgs)
            elif (self._QSArgs.DTMode=='单时点') and (self._QSArgs.IDMode=='多ID'):
                for i, iDT in enumerate(dts):
                    iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                    x = []
                    for k,kDescriptorData in enumerate(descriptor_data):
                        kStartInd, kLen = StartIndAndLen[k]
                        x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                    StdData[iStartInd+i, :] = Operator(self, iDTs, ids, x, ModelArgs)
            elif (self._QSArgs.DTMode=='多时点') and (self._QSArgs.IDMode=='单ID'):
                for j, jID in enumerate(ids):
                    StdData[iStartInd:, j] = Operator(self, DTRuler, jID, [kDescriptorData[:, j] for kDescriptorData in descriptor_data], ModelArgs)
            else:
                return Operator(self, DTRuler, ids, descriptor_data, ModelArgs)
            return StdData[iStartInd:, :]
        else:
            StdData = pd.DataFrame(StdData, columns=ids, index=DTRuler[-StdData.shape[0]:])
            if (self._QSArgs.iLookBackMode=="扩张窗口") or (self._QSArgs.iLookBack!=0):
                descriptor_data[0] = StdData
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).loc[:, DTRuler].to_frame(filter_observations=False).sort_index(axis=1)
            if self._QSArgs.ExpandDescriptors:
                descriptor_data, iOtherData = descriptor_data.iloc[:, self._QSArgs.ExpandDescriptors], descriptor_data.loc[:, descriptor_data.columns.difference(descriptor_data.columns[self._QSArgs.ExpandDescriptors])]
                descriptor_data = expandListElementDataFrame(descriptor_data, expand_index=True)
                descriptor_data = descriptor_data.set_index(descriptor_data.columns[:2].tolist())
                if not iOtherData.empty:
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
                        iStdData = Operator(self, iDTs, jID, jDescriptorData.loc[iDTs], ModelArgs)
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
                    iStdData = Operator(self, iDTs, ids, descriptor_data.loc[iDTs], ModelArgs)
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
                    iStdData = Operator(self, DTRuler, jID, descriptor_data.loc[jID], ModelArgs)
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
                StdData = Operator(self, DTRuler, ids, descriptor_data, ModelArgs)
                return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
    
    def __QS_prepareCacheData__(self, ids=None):
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = list(self._OperationMode.DTRuler[StartInd:EndInd+1])
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None: IDs = list(self._OperationMode._PID_IDs[PID])
        else:
            IDs = partitionListMovingSampling(IDs, len(self._OperationMode._PIDs))[self._OperationMode._PIDs.index(PID)]
        if IDs:
            DescriptorData = []
            if self._QSArgs.InputFormat=="numpy":
                for i, iDescriptor in enumerate(self._Descriptors):
                    iStartInd = StartInd - self._QSArgs.LookBack[i]
                    iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=[PID]).values
                    if iStartInd<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
                    DescriptorData.append(iDescriptorData)
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=self._OperationMode.DTRuler)
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                for i, iDescriptor in enumerate(self._Descriptors):
                    iStartInd = StartInd - self._QSArgs.LookBack[i]
                    iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=[PID])
                    DescriptorData.append(iDescriptorData)
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=self._OperationMode.DTRuler)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iStartInd = StartInd - self._QSArgs.LookBack[i]
                iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                iDescriptor._QS_getData(iDTs, pids=[PID])
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if self._QSArgs.DataType=="double" else "O"))
        with self._OperationMode._PID_Lock[PID]:
            with pd.HDFStore(self._OperationMode._CacheDataDir+os.sep+PID+os.sep+self.Name+str(self._OperationMode._FactorID[self.Name])+self._OperationMode._FileSuffix) as CacheFile:
                CacheFile["StdData"] = StdData
                CacheFile["_QS_IDs"] = pd.Series(IDs)
        self._isCacheDataOK = True
        return StdData

# 截面运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(nID, )), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class SectionOperation(DerivativeFactor):
    """截面运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=8, option_range=["全截面", "单ID"], mutable=False)
        DescriptorSection = List(arg_type="List", label="描述子截面", order=9, mutable=False)
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            self.DescriptorSection = [None]*len(self._Owner._Descriptors)
    
    def readData(self, ids, dts, **kwargs):
        SectionIDs = kwargs.pop("section_ids", ids)
        DescriptorData = []
        if self._QSArgs.InputFormat == "numpy":
            for i, iDescriptor in enumerate(self._Descriptors):
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                DescriptorData.append(iDescriptor.readData(ids=iSectionIDs, dts=dts, **kwargs).values)
            StdData = self._calcData(ids=SectionIDs, dts=dts, descriptor_data=DescriptorData)
            return pd.DataFrame(StdData, index=dts, columns=SectionIDs).reindex(columns=ids)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                DescriptorData.append(iDescriptor.readData(ids=iSectionIDs, dts=dts, **kwargs))            
            StdData = self._calcData(ids=SectionIDs, dts=dts, descriptor_data=DescriptorData)
            return StdData.reindex(columns=ids)
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        OldStartDT = dt_dict.get(self.Name, None)
        if (OldStartDT is None) or (start_dt<OldStartDT):
            dt_dict[self.Name] = start_dt
            StartInd, EndInd = self._OperationMode.DTRuler.index(dt_dict[self.Name]), self._OperationMode.DTRuler.index(self._OperationMode.DateTimes[-1])
            DTs = self._OperationMode.DTRuler[StartInd:EndInd+1]
            DTPartition = partitionList(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID:DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
        PrepareIDs = id_dict.setdefault(self.Name, prepare_ids)
        if prepare_ids != PrepareIDs: raise __QS_Error__("因子 %s 指定了不同的截面!" % self.Name)
        for i, iDescriptor in enumerate(self._Descriptors):
            if self._QSArgs.DescriptorSection[i] is None:
                iDescriptor._QS_initOperation(start_dt, dt_dict, prepare_ids, id_dict)
            else:
                iDescriptor._QS_initOperation(start_dt, dt_dict, self._QSArgs.DescriptorSection[i], id_dict)
        if (self._OperationMode.SubProcessNum>0) and (self.Name not in self._OperationMode._Event):
            self._OperationMode._Event[self.Name] = (Queue(), Event())
    def _calcData(self, ids, dts, descriptor_data):
        Operator, ModelArgs = self._QSArgs.Operator, self._QSArgs.ModelArgs
        if self._QSArgs.InputFormat == "numpy":
            if self._QSArgs.DataType=="double": StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype="float")
            else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype="O")
            if self._QSArgs.OutputMode=="全截面":
                if self._QSArgs.DTMode=="单时点":
                    for i, iDT in enumerate(dts):
                        StdData[i, :] = Operator(self, iDT, ids, [kDescriptorData[i] for kDescriptorData in descriptor_data], ModelArgs)
                else:
                    StdData = Operator(self, dts, ids, descriptor_data, ModelArgs)
            else:
                if self._QSArgs.DTMode=="单时点":
                    for i, iDT in enumerate(dts):
                        x = [kDescriptorData[i] for kDescriptorData in descriptor_data]
                        for j, jID in enumerate(ids):
                            StdData[i, j] = Operator(self, iDT, jID, x, ModelArgs)
                else:
                    for j, jID in enumerate(ids):
                        StdData[:, j] = Operator(self, dts, jID, descriptor_data, ModelArgs)
            return StdData
        else:
            SectionIdx = self._QS_partitionSectionIDs(self._QSArgs.DescriptorSection)
            DescriptorData = []
            for iSectionIDs, iIdx in SectionIdx:
                iDescriptorData = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data)) if i in iIdx}).to_frame(filter_observations=False).sort_index(axis=1)
                iExpandDescriptors = sorted(f"d{i}" for i in set(self._QSArgs.ExpandDescriptors).intersection(iIdx))
                if iExpandDescriptors:
                    iDescriptorData, iOtherData = iDescriptorData.loc[:, iExpandDescriptors], iDescriptorData.loc[:, iDescriptorData.columns.difference(iExpandDescriptors)]
                    iDescriptorData = expandListElementDataFrame(iDescriptorData, expand_index=True)
                    iDescriptorData = iDescriptorData.set_index(iDescriptorData.columns[:2].tolist())
                    if not iOtherData.empty:
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
                        iStdData = Operator(self, iDT, ids, [iData.loc[iDT] for iData in descriptor_data], ModelArgs)
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
                    StdData = Operator(self, dts, ids, descriptor_data, ModelArgs)
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            else:
                if self._QSArgs.DTMode=="单时点":
                    if self._QSArgs.DataType == "double": StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype="float")
                    else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype="O")
                    for i, iDT in enumerate(dts):
                        iDescriptorData = [iData.loc[iDT] for iData in descriptor_data]
                        for j, jID in enumerate(ids):
                            iStdData = Operator(self, iDT, jID, iDescriptorData, ModelArgs)
                            if isinstance(iStdData, pd.DataFrame):
                                iStdData = tuple(iStdData.reindex(columns=CompoundCols).T.values.tolist())
                            elif isinstance(iStdData, pd.Series):
                                iStdData = tuple(iStdData.reindex(index=CompoundCols))
                            StdData[i, j] = iStdData
                    return pd.DataFrame(StdData, index=dts, columns=ids)
                else:
                    StdData = []
                    for j, jID in enumerate(ids):
                        iStdData = Operator(self, dts, jID, descriptor_data, ModelArgs)
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

    def __QS_prepareCacheData__(self, ids=None):
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None: IDs = list(self._OperationMode.IDs)
        if len(DTs)==0:# 该进程未分配到计算任务
            iDTs = [self._OperationMode.DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor._QS_getData(iDTs, pids=None)
            StdData = pd.DataFrame(columns=IDs, dtype=("float" if self._QSArgs.DataType=="double" else "O"))
        elif IDs:
            if self._QSArgs.InputFormat == "numpy":
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=None).values for iDescriptor in self._Descriptors])
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=[iDescriptor._QS_getData(DTs, pids=None) for iDescriptor in self._Descriptors])
        else:
            for iDescriptor in self._Descriptors:
                iDescriptor._QS_getData(DTs, pids=None)
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if self._QSArgs.DataType=="double" else "O"))
        if self._OperationMode._FactorPrepareIDs[self.Name] is None:
            PID_IDs = self._OperationMode._PID_IDs
        else:
            PID_IDs = {self._OperationMode._PIDs[i]: iSubIDs for i, iSubIDs in enumerate(partitionListMovingSampling(IDs, len(self._OperationMode._PIDs)))}
        for iPID, iIDs in PID_IDs.items():
            with self._OperationMode._PID_Lock[iPID]:
                with pd.HDFStore(self._OperationMode._CacheDataDir+os.sep+iPID+os.sep+self.Name+str(self._OperationMode._FactorID[self.Name])+self._OperationMode._FileSuffix) as CacheFile:
                    if "StdData" in CacheFile:
                        CacheFile["StdData"] = pd.concat([CacheFile["StdData"], StdData.reindex(columns=iIDs)]).sort_index()
                    else:
                        CacheFile["StdData"] = StdData.reindex(columns=iIDs)
                    CacheFile["_QS_IDs"] = pd.Series(iIDs)
        StdData = None# 释放数据
        gc.collect()
        if self._OperationMode.SubProcessNum>0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData

# 面板运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多日期，则该值为 [回溯期数]+[时点]
# iid: 当前待计算的 ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(回溯期数, nID)), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(回溯期数+nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class PanelOperation(DerivativeFactor):
    """面板运算"""
    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=7, option_range=["单时点", "多时点"], mutable=False)
        OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=8, option_range=["全截面", "单ID"], mutable=False)
        LookBack = List(arg_type="ArgList", label="回溯期数", order=9, mutable=False)# 描述子向前回溯的时点数(不包括当前时点)
        LookBackMode = List(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=10, mutable=False)
        iLookBack = Int(0, arg_type="Integer", label="自身回溯期数", order=11, mutable=False)
        iLookBackMode = Enum("滚动窗口", "扩张窗口", arg_type="SingleOption", label="自身回溯模式", order=12, option_range=["滚动窗口", "扩张窗口"], mutable=False)
        iInitData = Instance(pd.DataFrame, arg_type="DataFrame", label="自身初始值", order=13)
        DescriptorSection = List(arg_type="List", label="描述子截面", order=14, mutable=False)
        def __QS_initArgs__(self, args={}):
            super().__QS_initArgs__(args=args)
            nDescriptor = len(self._Owner._Descriptors)
            self.LookBack = [0]*nDescriptor
            self.LookBackMode = ["滚动窗口"]*nDescriptor
            self.DescriptorSection = [None]*nDescriptor
    
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        if len(self._Descriptors)>len(self._QSArgs.LookBack): raise  __QS_Error__("面板运算因子 : '%s' 的参数'回溯期数'序列长度小于描述子个数!" % self.Name)
        OldStartDT = dt_dict.get(self.Name, None)
        DTRuler = self._OperationMode.DTRuler
        if (OldStartDT is None) or (start_dt<OldStartDT):
            StartDT = dt_dict[self.Name] = start_dt
            StartInd, EndInd = DTRuler.index(StartDT), DTRuler.index(self._OperationMode.DateTimes[-1])
            if (self._QSArgs.iLookBackMode=="扩张窗口") and (self._QSArgs.iInitData is not None) and (self._QSArgs.iInitData.shape[0]>0):
                if self._QSArgs.iInitData.index[-1] not in self._OperationMode.DTRuler: self._QS_Logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, ))
                else: StartInd = min(StartInd, self._OperationMode.DTRuler.index(self._QSArgs.iInitData.index[-1]) + 1)
            DTs = DTRuler[StartInd:EndInd+1]
            if self._QSArgs.iLookBackMode=="扩张窗口":
                DTPartition = [DTs]+[[]]*(len(self._OperationMode._PIDs)-1)
            else:
                DTPartition = partitionList(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID:DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
        else:
            StartInd = DTRuler.index(OldStartDT)
        PrepareIDs = id_dict.setdefault(self.Name, prepare_ids)
        if prepare_ids != PrepareIDs: raise __QS_Error__("因子 %s 指定了不同的截面!" % self.Name)
        for i, iDescriptor in enumerate(self._Descriptors):
            iStartInd = StartInd - self._QSArgs.LookBack[i]
            if iStartInd<0: self._QS_Logger.warning("注意: 对于因子 '%s' 的描述子 '%s', 时点标尺长度不足!" % (self.Name, iDescriptor.Name))
            iStartDT = DTRuler[max(0, iStartInd)]
            if self._QSArgs.DescriptorSection[i] is None:
                iDescriptor._QS_initOperation(iStartDT, dt_dict, prepare_ids, id_dict)
            else:
                iDescriptor._QS_initOperation(iStartDT, dt_dict, self._QSArgs.DescriptorSection[i], id_dict)
        if (self._OperationMode.SubProcessNum>0) and (self.Name not in self._OperationMode._Event):
            self._OperationMode._Event[self.Name] = (Queue(), Event())
    def readData(self, ids, dts, **kwargs):
        DTRuler = kwargs.get("dt_ruler", dts)
        SectionIDs = kwargs.pop("section_ids", ids)
        StartInd = (DTRuler.index(dts[0]) if dts[0] in DTRuler else 0)
        if (self._QSArgs.iLookBackMode=="扩张窗口") and (self._QSArgs.iInitData is not None) and (self._QSArgs.iInitData.shape[0]>0):
            if self._QSArgs.iInitData.index[-1] not in DTRuler: self._QS_Logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name, ))
            else: StartInd = min(StartInd, DTRuler.index(self._QSArgs.iInitData.index[-1]) + 1)
        EndInd = (DTRuler.index(dts[-1]) if dts[-1] in DTRuler else len(DTRuler)-1)
        if StartInd>EndInd: return pd.DataFrame(index=dts, columns=ids)
        DescriptorData = []
        if self._QSArgs.InputFormat == "numpy":
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-self._QSArgs.LookBack[i], 0):EndInd+1]
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                iIDNum = len(iSectionIDs)
                if iDTs:
                    iDescriptorData = iDescriptor.readData(ids=iSectionIDs, dts=iDTs, **kwargs).values
                else:
                    iDescriptorData = np.full((0, iIDNum), np.nan)
                if StartInd<self._QSArgs.LookBack[i]:
                    iLookBackData = np.full((self._QSArgs.LookBack[i]-StartInd, iIDNum), np.nan)
                    iDescriptorData = np.r_[iLookBackData, iDescriptorData]
                DescriptorData.append(iDescriptorData)
            StdData = self._calcData(ids=SectionIDs, dts=DTRuler[StartInd:EndInd+1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return pd.DataFrame(StdData, index=DTRuler[StartInd:EndInd+1], columns=SectionIDs).reindex(index=dts, columns=ids)
        else:
            for i, iDescriptor in enumerate(self._Descriptors):
                iDTs = DTRuler[max(StartInd-self._QSArgs.LookBack[i], 0):EndInd+1]
                iSectionIDs = self._QSArgs.DescriptorSection[i]
                if iSectionIDs is None: iSectionIDs = SectionIDs
                if iDTs:
                    iDescriptorData = iDescriptor.readData(ids=iSectionIDs, dts=iDTs, **kwargs)
                else:
                    iDescriptorData = pd.DataFrame(columns=iSectionIDs)
                DescriptorData.append(iDescriptorData)
            StdData = self._calcData(ids=SectionIDs, dts=DTRuler[StartInd:EndInd + 1], descriptor_data=DescriptorData, dt_ruler=DTRuler)
            return StdData.reindex(index=dts, columns=ids)
    
    def _calcData(self, ids, dts, descriptor_data, dt_ruler):
        if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else: StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
        for i, iDescriptor in enumerate(self._Descriptors):
            iLookBack = self._QSArgs.LookBack[i]
            if self._QSArgs.LookBackMode[i]=="滚动窗口":
                StartIndAndLen.append((iLookBack, iLookBack+1))
                MaxLen = max(MaxLen, iLookBack+1)
            else:
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
        StartInd, EndInd = dt_ruler.index(dts[0]), dt_ruler.index(dts[-1])
        if StartInd>=MaxLookBack: DTRuler = dt_ruler[StartInd-MaxLookBack:EndInd+1]
        else: DTRuler = [None]*(MaxLookBack-StartInd) + dt_ruler[:EndInd+1]
        Operator, ModelArgs = self._QSArgs.Operator, self._QSArgs.ModelArgs
        if self._QSArgs.InputFormat == "numpy":
            if self._QSArgs.OutputMode=='全截面':
                if self._QSArgs.DTMode=='单时点':
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                        x = []
                        for k, kDescriptorData in enumerate(descriptor_data):
                            kStartInd, kLen = StartIndAndLen[k]
                            x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                        StdData[iStartInd+i, :] = Operator(self, iDTs, ids, x, ModelArgs)
                else:
                    return Operator(self, DTRuler, ids, descriptor_data, ModelArgs)
            else:
                if self._QSArgs.DTMode=='单时点':
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
                        x = []
                        for k, kDescriptorData in enumerate(descriptor_data):
                            kStartInd, kLen = StartIndAndLen[k]
                            x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
                        for j, jID in enumerate(ids):
                            StdData[iStartInd+i, j] = Operator(self, iDTs, jID, x, ModelArgs)
                else:
                    for j, jID in enumerate(ids):
                        StdData[iStartInd:, j] = Operator(self, DTRuler, jID, descriptor_data, ModelArgs)
            return StdData[iStartInd:, :]
        else:
            StdData = pd.DataFrame(StdData, columns=ids, index=DTRuler[-StdData.shape[0]:])
            if (self._QSArgs.iLookBackMode=="扩张窗口") or (self._QSArgs.iLookBack!=0):
                descriptor_data[0] = StdData
            SectionIdx = self._QS_partitionSectionIDs(self._QSArgs.DescriptorSection)
            DescriptorData = []
            for iSectionIDs, iIdx in SectionIdx:
                iDescriptorData = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data)) if i in iIdx}).loc[:, DTRuler].to_frame(filter_observations=False).sort_index(axis=1)
                iExpandDescriptors = sorted(f"d{i}" for i in set(self._QSArgs.ExpandDescriptors).intersection(iIdx))
                if iExpandDescriptors:
                    iDescriptorData, iOtherData = iDescriptorData.loc[:, iExpandDescriptors], iDescriptorData.loc[:, iDescriptorData.columns.difference(iExpandDescriptors)]
                    iDescriptorData = expandListElementDataFrame(iDescriptorData, expand_index=True)
                    iDescriptorData = iDescriptorData.set_index(iDescriptorData.columns[:2].tolist())
                    if not iOtherData.empty:
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
                        iStdData = Operator(self, iDTs, ids, [iData.loc[iDTs] for iData in descriptor_data], ModelArgs)
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
                    StdData = Operator(self, DTRuler, ids, descriptor_data, ModelArgs)
                    return self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            else:
                if self._QSArgs.DTMode=='单时点':
                    StdData = StdData.values
                    for i, iDT in enumerate(dts):
                        iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
                        for j, jID in enumerate(ids):
                            iStdData = Operator(self, iDTs, jID, [iData.loc[iDTs] for iData in descriptor_data], ModelArgs)
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
                        iStdData = Operator(self, DTRuler, jID, descriptor_data, ModelArgs)
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
    
    def __QS_prepareCacheData__(self, ids=None):
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None: IDs = list(self._OperationMode.IDs)
        if len(DTs)==0:# 该进程未分配到计算任务
            iDTs = [self._OperationMode.DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor._QS_getData(iDTs, pids=None)
            StdData = pd.DataFrame(columns=IDs, dtype=("float" if self._QSArgs.DataType=="double" else "O"))
        elif IDs:
            if self._QSArgs.InputFormat == "numpy":
                DescriptorData, StartInd = [], self._OperationMode.DTRuler.index(DTs[0])
                for i, iDescriptor in enumerate(self._Descriptors):
                    iStartInd = StartInd - self._QSArgs.LookBack[i]
                    iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=None).values
                    if iStartInd<0: iDescriptorData = np.r_[np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
                    DescriptorData.append(iDescriptorData)
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=self._OperationMode.DTRuler)
                DescriptorData, iDescriptorData, StdData = None, None, pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                DescriptorData, StartInd = [], self._OperationMode.DTRuler.index(DTs[0])
                for i, iDescriptor in enumerate(self._Descriptors):
                    iStartInd = StartInd - self._QSArgs.LookBack[i]
                    iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                    iDescriptorData = iDescriptor._QS_getData(iDTs, pids=None)
                    DescriptorData.append(iDescriptorData)
                StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData, dt_ruler=self._OperationMode.DTRuler)
                DescriptorData, iDescriptorData = None, None
        else:
            DescriptorData, StartInd = [], self._OperationMode.DTRuler.index(DTs[0])
            for i, iDescriptor in enumerate(self._Descriptors):
                iStartInd = StartInd - self._QSArgs.LookBack[i]
                iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                iDescriptor._QS_getData(iDTs, pids=None)
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if self._QSArgs.DataType=="double" else "O"))
        if self._OperationMode._FactorPrepareIDs[self.Name] is None:
            PID_IDs = self._OperationMode._PID_IDs
        else:
            PID_IDs = {self._OperationMode._PIDs[i]: iSubIDs for i, iSubIDs in enumerate(partitionListMovingSampling(IDs, len(self._OperationMode._PIDs)))}
        for iPID, iIDs in PID_IDs.items():
            with self._OperationMode._PID_Lock[iPID]:
                with pd.HDFStore(self._OperationMode._CacheDataDir+os.sep+iPID+os.sep+self.Name+str(self._OperationMode._FactorID[self.Name])+self._OperationMode._FileSuffix) as CacheFile:
                    if "StdData" in CacheFile:
                        CacheFile["StdData"] = pd.concat([CacheFile["StdData"], StdData.reindex(columns=iIDs)]).sort_index()
                    else:
                        CacheFile["StdData"] = StdData.reindex(columns=iIDs)
                    CacheFile["_QS_IDs"] = pd.Series(iIDs)
        StdData = None# 释放数据
        gc.collect()
        if self._OperationMode.SubProcessNum>0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData

# 将算子转换成因子定义的装饰器
# operation_type: 因子运算类型, 可选: 'PointOperation', 'TimeOperation', 'SectionOperation', 'PanelOperation'
# sys_args: 因子定义参数将
def FactorOperation(operation_type, sys_args={}):
    def Decorator(func):
        def defFactor(f="", x=[], args={}):
            Args = sys_args.copy()
            Args.update(args)
            Args["算子"] = func
            if operation_type=="PointOperation":
                return PointOperation(name=f, descriptors=x, sys_args=Args)
            elif operation_type=="TimeOperation":
                return TimeOperation(name=f, descriptors=x, sys_args=Args)
            elif operation_type=="SectionOperation":
                return SectionOperation(name=f, descriptors=x, sys_args=Args)
            elif operation_type=="PanelOperation":
                return PanelOperation(name=f, descriptors=x, sys_args=Args)
            else:
                raise __QS_Error__(f"错误的因子运算类型: '{operation_type}', 必须为 'PointOperation', 'TimeOperation', 'SectionOperation' 或者 'PanelOperation'")
        return defFactor
    return Decorator

if __name__=="__main__":
    @FactorOperation(operation_type="PointOperation", sys_args={"运算时点": "多时点"})
    def test_fun(f, idt, iid, x, args):
        return x[0] + 1
    
    from QuantStudio.FactorDataBase.FactorDB import DataFactor
    Factor1 = DataFactor(name="Factor1", data=1)
    
    Factor2 = test_fun("Factor2", [Factor1])
    
    print(Factor2)
    print(Factor2.Args)
    print("===")