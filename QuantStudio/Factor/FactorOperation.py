# -*- coding: utf-8 -*-
"""因子运算"""
import base64
import importlib
import os
import datetime as dt
from functools import partial
from typing import Optional, Literal, List, Any, Tuple, Callable, Union, Dict

import dill
import pandas as pd
import numpy as np
from pydantic import Field
from multiprocess import Event

from QuantStudio.Core import __QS_Error__, __QS_Object__
from QuantStudio.Core.Node import Node
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import Factor, DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame
from QuantStudio.Tools.AuxiliaryFun import partitionList


class FactorOperator(__QS_Object__):
    """因子算子"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        OperatorType: Literal["Point", "Time", "Section", "Panel"] = Field(title="算子类型", frozen=True)
        Name: str = Field(default="FactorOperator", title="名称", frozen=True, exclude=True)
        ModelArgs: Dict[str, Any] = Field(default={}, title="模型参数", frozen=True)
        Arity: Optional[int] = Field(default=None, ge=1, title="入参数量", frozen=True)
        DataType: Literal["double", "string", "object"] = Field(default="double",title="数据类型", frozen=True)
        Description: str = Field(default="", title="描述信息", frozen=False, exclude=True)
        Meta: Dict[str, Any] = Field(default={}, title="元信息", frozen=False, exclude=True)
        InputFormat: Literal["numpy", "pandas"] = Field(default="numpy", title="输入格式", repr=False, frozen=True)
        ExpandDescriptors: List[int] = Field(default=[], title="展开描述子", repr=False, frozen=True)
        DescriptorCompoundType: List[List[Tuple[str, Literal["double", "string", "object"]]]] = Field(default=[], title="描述子复合类型", repr=False, frozen=True)
        MultiMapping: bool = Field(default=False, title="多重映射", repr=False, frozen=True)
        CompoundType: List[Tuple[str, Literal["double", "string", "object"]]] = Field(default=[], title="复合类型", repr=False, frozen=True)
        
        def __init__(self, /, **data: Any) -> None:
            if ("DataType" not in data) and (data.get("CompoundType", []) or data.get("MultiMapping", False)):
                data["DataType"] = "object"
            return super().__init__(**data)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if "calculate" in self.__dict__:
            state["calculate"] = dill.dumps(self.calculate)
        return state
    
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

    # region 序列化 / 反序列化
    def _is_custom_calculate(self) -> bool:
        """判断 calculate 是否在实例上被替换过（非类继承）"""
        calculate = getattr(self, "calculate", None)
        if calculate is None:
            return False
        op_class = type(self)
        if "calculate" not in op_class.__dict__:
            return True
        cls_calc = op_class.__dict__["calculate"]
        if hasattr(cls_calc, "__code__") and hasattr(calculate, "__code__"):
            return cls_calc.__code__ is not calculate.__code__
        return cls_calc is not calculate

    def _serialize_calculate(self) -> Optional[dict]:
        """序列化自定义 calculate，返回 None 表示无需序列化"""
        calculate = getattr(self, "calculate", None)
        if calculate is None or not self._is_custom_calculate():
            return None
        # numpy ufunc
        if isinstance(calculate, np.ufunc):
            return {"type": "numpy_func", "name": calculate.__name__}
        # 可导入的函数
        module = getattr(calculate, "__module__", None)
        qualname = getattr(calculate, "__qualname__", None)
        if module and qualname and module != "__main__":
            try:
                obj = importlib.import_module(module)
                for part in qualname.split("."):
                    obj = getattr(obj, part)
                if obj is calculate:
                    return {"type": "func_ref", "module": module, "qualname": qualname}
            except (ImportError, AttributeError):
                pass
        # dill 兜底
        return {"type": "dill", "data": base64.b64encode(dill.dumps(calculate)).decode("ascii")}

    @staticmethod
    def _deserialize_calculate(op, ref: dict):
        if ref is None:
            return
        t = ref.get("type", "")
        if t == "func_ref":
            module = importlib.import_module(ref["module"])
            obj = module
            for part in ref["qualname"].split("."):
                obj = getattr(obj, part)
            op.calculate = obj
        elif t == "dill":
            op.calculate = dill.loads(base64.b64decode(ref["data"]))
        elif t == "numpy_func":
            op.calculate = getattr(np, ref["name"])

    def serialize(self) -> Dict[str, Any]:
        """序列化算子为 dict，包含 calculate_ref 和完整类路径"""
        result = super().serialize()
        result["__class__"] = type(self).__module__ + "." + type(self).__qualname__
        result["calculate_ref"] = self._serialize_calculate()
        return result

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "FactorOperator":
        """从序列化 dict 重建算子实例，恢复 calculate"""
        op = super().deserialize(data)
        cls._deserialize_calculate(op, data.get("calculate_ref"))
        return op
    # endregion
    
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

    def calcData(self, factor: Factor, ids: List[str], dts: List[dt.datetime], descriptor_data: list, dt_ruler: Optional[List[dt.datetime]]=None, section_ids: Optional[List[str]]=None, extra_dep_data: List[Any]=[]):
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
        Name: str = Field(default="PointOperator", title="名称", frozen=True, exclude=True)
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

    def _calcDataNumpyMultiIDMultiDT(self, factor, ids, dts, descriptor_data, ModelArgs, extra_dep_data):
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is not None:
            if self._QSArgs.DataType == 'double':
                StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
            else:
                StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
            descriptor_data = [iData[CalcMask] for iData in descriptor_data]
            dts = np.array(dts, dtype="O")[CalcMask].tolist()
            iStdData = self.calculate(factor, dts, ids, descriptor_data + extra_dep_data, ModelArgs)
            StdData[CalcMask, :] = iStdData
            return StdData
        else:
            return self.calculate(factor, dts, ids, descriptor_data + extra_dep_data, ModelArgs)

    def _calcDataNumpySingleIDSingleDT(self, factor, ids, dts, descriptor_data, ModelArgs, extra_dep_data):
        if self._QSArgs.DataType == 'double':
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else:
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
        if CalcDTs: CalcDTs = set(CalcDTs)
        for i, iDT in enumerate(dts):
            if (CalcDTs is not None) and (iDT not in CalcDTs): continue
            for j, jID in enumerate(ids):
                StdData[i, j] = self.calculate(factor, iDT, jID, [iData[i, j] for iData in descriptor_data] + extra_dep_data, ModelArgs)
        return StdData

    def _calcDataNumpySingleIDMultiDT(self, factor, ids, dts, descriptor_data, ModelArgs, extra_dep_data):
        if self._QSArgs.DataType == 'double':
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else:
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is None:
            for j, jID in enumerate(ids):
                StdData[:, j] = self.calculate(factor, dts, jID, [iData[:, j] for iData in descriptor_data] + extra_dep_data, ModelArgs)
        else:
            dts = np.array(dts, dtype="O")[CalcMask].tolist()
            for j, jID in enumerate(ids):
                StdData[CalcMask, j] = self.calculate(factor, dts, jID, [iData[CalcMask, j] for iData in descriptor_data] + extra_dep_data, ModelArgs)
        return StdData

    def _calcDataNumpyMultiIDSingleDT(self, factor, ids, dts, descriptor_data, ModelArgs, extra_dep_data):
        if self._QSArgs.DataType == 'double':
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype='float')
        else:
            StdData = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype='O')
        CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
        if CalcDTs: CalcDTs = set(CalcDTs)
        for i, iDT in enumerate(dts):
            if (CalcDTs is not None) and (iDT not in CalcDTs): continue
            StdData[i, :] = self.calculate(factor, iDT, ids, [iData[i, :] for iData in descriptor_data] + extra_dep_data, ModelArgs)
        return StdData

    def _calcDataNumpy(self, context: FactorContext, factor, ids, dts, descriptor_data, ModelArgs, extra_dep_data):
        if (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '多ID'):
            TargetFunc = self._calcDataNumpyMultiIDMultiDT
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '单ID'):
            TargetFunc = self._calcDataNumpySingleIDSingleDT
        elif (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '单ID'):
            TargetFunc = self._calcDataNumpySingleIDMultiDT
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '多ID'):
            TargetFunc = self._calcDataNumpyMultiIDSingleDT
        TaskExecutor = factor._QSArgs.TaskExecutor if factor._QSArgs.TaskExecutor is not None else context.TaskExecutor
        if (not factor._QSArgs.Parallel) or (TaskExecutor is None) or (context.MaxWorkers <= 1): return TargetFunc(factor=factor, ids=ids, dts=dts, descriptor_data=descriptor_data, ModelArgs=ModelArgs, extra_dep_data=extra_dep_data)
        Futures = []
        BatchSize = len(ids) // context.MaxWorkers + (len(ids) % context.MaxWorkers > 0)
        for i in range(context.MaxWorkers):
            iIDStartIdx, iIDEndIdx = i * BatchSize, (i + 1) * BatchSize
            iIDs = ids[iIDStartIdx:iIDEndIdx]
            if not iIDs: continue
            Futures.append(TaskExecutor.submit(TargetFunc, factor, iIDs, dts, [iData[:, iIDStartIdx:iIDEndIdx] for iData in descriptor_data], ModelArgs, extra_dep_data))
        return np.hstack([iFuture.result() for iFuture in Futures])

    def _calcDataPandas(self, factor, ids, dts, descriptor_data, ModelArgs, extra_dep_data):
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is not None:
            descriptor_data = Panel({f"d{i}": descriptor_data[i][CalcMask] for i in range(len(descriptor_data))}).to_frame(filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
        else:
            descriptor_data = Panel({f"d{i}": descriptor_data[i] for i in range(len(descriptor_data))}).to_frame(
                filter_observations=False).sort_index(axis=1, key=lambda x: x.str.replace("d", "").astype(int))
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
        if (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '多ID'):
            DTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
            StdData = self.calculate(factor, DTs, ids, [descriptor_data] + extra_dep_data, ModelArgs)
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
                    iStdData = self.calculate(factor, iDT, jID, [descriptor_data.loc[iDT].loc[jID]] + extra_dep_data, ModelArgs)
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
                iStdData = self.calculate(factor, CalcDTs, jID, [descriptor_data.loc[jID]] + extra_dep_data, ModelArgs)
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
                iStdData = self.calculate(factor, iDT, ids, [descriptor_data.loc[iDT]] + extra_dep_data, ModelArgs)
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

    def calcData(self, context:FactorContext, factor:Factor, ids:List[str], dts:List[dt.datetime], descriptor_data:List[np.ndarray | pd.DataFrame], dt_ruler:Optional[List[dt.datetime]]=None, section_ids:Optional[List[str]]=None, extra_dep_data:List[Any]=[]):
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, self._QSArgs.ModelArgs, extra_dep_data)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, self._QSArgs.ModelArgs, extra_dep_data)


class TimeOperator(FactorOperator):
    """时序算子, 运算依赖于被依赖因子在过去若干个时点以及单个ID的因子值"""
    
    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Time"] = Field(default="Time", title="算子类型", frozen=True)
        Name: str = Field(default="TimeOperator", title="名称", frozen=True, exclude=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        IDMode: Literal["单ID", "多ID"] = Field(default="单ID", title="运算ID", frozen=True)
        LookBack: List[int] = Field(default=[], title="回溯期数", frozen=True, description="描述子向前回溯的时点数(不包括当前时点)")
        # LookBackMode: List[Literal["滚动窗口", "扩张窗口"]] = Field(default=[], title="回溯模式", description="描述子的回溯模式", frozen=True)
        StartDT: List[Optional[dt.datetime]] = Field(default=[], title="起始时点", frozen=True, description="扩张窗口模式下描述子的起始时点, 如果为 None, 则使用回溯期数参数")
        iInitFactor: int = Field(default=-1, title="起始因子", ge=-1, frozen=True)
        DescriptorDTRuler: List[Optional[List[dt.datetime]]] = Field(
            default=[], title="描述子时点标尺", frozen=True,
            description="每个描述子的时点标尺, None 表示使用因子的时点标尺. "
                        "决定回溯的频率: 日度标尺回溯N天, 月度标尺回溯N个月."
        )

        def __init__(self, /, **data):
            Arity = data.get("Arity", 0)
            if Arity is None: Arity = 0
            if not data.get("LookBack", []): data["LookBack"] = [0] * Arity
            # if not data.get("LookBackMode", []): data["LookBackMode"] = ["滚动窗口"] * Arity
            if not data.get("StartDT", []): data["StartDT"] = [None] * Arity
            DescriptorDTRuler = data.get("DescriptorDTRuler", [])
            if not DescriptorDTRuler:
                data["DescriptorDTRuler"] = [None] * Arity
            elif len(DescriptorDTRuler) < Arity:
                data["DescriptorDTRuler"] = list(DescriptorDTRuler) + [None] * (Arity - len(DescriptorDTRuler))
            return super().__init__(**data)

        def model_post_init(self, context: Any, /) -> None:
            if self.Arity is not None:
                if self.Arity != len(self.LookBack):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBack({self.__pydantic_fields__['LookBack'].title}): {self.LookBack} 的长度不一致!")
                # if self.Arity != len(self.LookBackMode):
                #     raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBackMode({self.__pydantic_fields__['LookBackMode'].title}): {self.LookBackMode} 的长度不一致!")
                if self.Arity != len(self.StartDT):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 StartDT({self.__pydantic_fields__['StartDT'].title}): {self.StartDT} 的长度不一致!")
                # 自动填充 DescriptorDTRuler（兼容旧数据）
                if len(self.DescriptorDTRuler) < self.Arity:
                    object.__setattr__(self, "DescriptorDTRuler", list(self.DescriptorDTRuler) + [None] * (self.Arity - len(self.DescriptorDTRuler)))
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
    
    def _calcDataNumpySingleIDSingleDT(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
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
                StdData[iStartIdx+i, j] = self.calculate(factor, iDTs, jID, x + extra_dep_data, ModelArgs)
        return StdData[iStartIdx:, :]
    
    def _calcDataNumpySingleIDMultiDT(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
        for j, jID in enumerate(ids):
            StdData[iStartIdx:, j] = self.calculate(factor, DTRuler, jID, [kDescriptorData[:, j] for kDescriptorData in descriptor_data] + extra_dep_data, ModelArgs)
        StdData = StdData[iStartIdx:, :]
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is not None:
            StdData[~CalcMask, :] = None
        return StdData

    def _calcDataNumpyMultiIDSingleDT(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
        CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
        if CalcDTs: CalcDTs = set(CalcDTs)
        for i, iDT in enumerate(dts):
            if (CalcDTs is not None) and (iDT not in CalcDTs): continue
            iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
            x = []
            for k,kDescriptorData in enumerate(descriptor_data):
                kStartInd, kLen = StartIndAndLen[k]
                x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
            StdData[iStartIdx+i, :] = self.calculate(factor, iDTs, ids, x + extra_dep_data, ModelArgs)
        return StdData[iStartIdx:, :]
    
    def _calcDataNumpyMultiIDMultiDT(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
        StdData = self.calculate(factor, DTRuler, ids, descriptor_data + extra_dep_data, ModelArgs)[iStartIdx:, :]
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is not None:
            StdData[~CalcMask, :] = None
        return StdData

    def _calcDataNumpy(self, context, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
        if (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '多ID'):
            TargetFunc = self._calcDataNumpyMultiIDMultiDT
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '单ID'):
            TargetFunc = self._calcDataNumpySingleIDSingleDT
        elif (self._QSArgs.DTMode == '多时点') and (self._QSArgs.IDMode == '单ID'):
            TargetFunc = self._calcDataNumpySingleIDMultiDT
        elif (self._QSArgs.DTMode == '单时点') and (self._QSArgs.IDMode == '多ID'):
            TargetFunc = self._calcDataNumpyMultiIDSingleDT
        TaskExecutor = factor._QSArgs.TaskExecutor if factor._QSArgs.TaskExecutor is not None else context.TaskExecutor
        if (not factor._QSArgs.Parallel) or (TaskExecutor is None) or (context.MaxWorkers <= 1):
            return TargetFunc(factor=factor, ids=ids, dts=dts, descriptor_data=descriptor_data, DTRuler=DTRuler, StartIndAndLen=StartIndAndLen, MaxLookBack=MaxLookBack, MaxLen=MaxLen, iStartIdx=iStartIdx, ModelArgs=ModelArgs, StdData=StdData, extra_dep_data=extra_dep_data)
        Futures = []
        BatchSize = len(ids) // context.MaxWorkers + (len(ids) % context.MaxWorkers > 0)
        for i in range(context.MaxWorkers):
            iIDStartIdx, iIDEndIdx = i * BatchSize, (i + 1) * BatchSize
            iIDs = ids[iIDStartIdx:iIDEndIdx]
            if not iIDs: continue
            iStdData = StdData[:, iIDStartIdx:iIDEndIdx].copy()
            Futures.append(TaskExecutor.submit(TargetFunc, factor, iIDs, dts, [iData[:, iIDStartIdx:iIDEndIdx] for iData in descriptor_data], DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, iStdData, extra_dep_data))
        return np.hstack([iFuture.result() for iFuture in Futures])
    
    def _calcDataPandas(self, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
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
                    iStdData = self.calculate(factor, iDTs, jID, [jDescriptorData.loc[iDTs]] + extra_dep_data, ModelArgs)
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
                iStdData = self.calculate(factor, iDTs, ids, [descriptor_data.loc[iDTs]] + extra_dep_data, ModelArgs)
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
                iStdData = self.calculate(factor, DTRuler, jID, [descriptor_data.loc[jID]] + extra_dep_data, ModelArgs)
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
            StdData = self.calculate(factor, DTRuler, ids, [descriptor_data] + extra_dep_data, ModelArgs)
            StdData = self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask] = None
            return StdData
    
    def calcData(self, context:FactorContext, factor:Factor, ids:List[str], dts:List[dt.datetime], descriptor_data:List[np.ndarray | pd.DataFrame], dt_ruler:Optional[List[dt.datetime]]=None, section_ids:Optional[List[str]]=None, extra_dep_data:List[Any]=[]):
        if dt_ruler is None: dt_ruler = dts
        if self._QSArgs.iInitFactor >= 0:# 自身迭代
            StdData = descriptor_data[self._QSArgs.iInitFactor] = descriptor_data[self._QSArgs.iInitFactor].copy()
            iStartIdx = StdData.shape[0] - len(dts)
        else:
            if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(section_ids)), fill_value=np.nan, dtype='float')
            else: StdData = np.full(shape=(len(dts), len(section_ids)), fill_value=None, dtype='O')
            iStartIdx = 0
        StartIdx, EndIdx = np.searchsorted(dt_ruler, dts[0], side="left"), np.searchsorted(dt_ruler, dts[-1], side="right")
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1# StartIndAndLen: [(开始位置, 数据长度)], MaxLookBack: 最大回溯期, MaxLen: 最大数据长度
        for i in range(len(descriptor_data)):
            iLookBack = self._QSArgs.LookBack[i]
            if i == self._QSArgs.iInitFactor:
                iStartInd = iStartIdx
                iLookBack = (iLookBack if self._QSArgs.StartDT[i] is None else (max(0, StartIdx - np.searchsorted(dt_ruler, self._QSArgs.StartDT[i], side="left") + iLookBack)))
                iLen = (iLookBack + 1 if self._QSArgs.StartDT[i] is None else np.inf)
            elif self._QSArgs.StartDT[i] is None:
                iStartInd = iLookBack
                iLen = iLookBack + 1
            else:
                iStartInd = iLookBack = max(0, StartIdx - np.searchsorted(dt_ruler, self._QSArgs.StartDT[i], side="left") + iLookBack)
                iLen = np.inf
            StartIndAndLen.append((iStartInd, iLen))
            MaxLen = max(MaxLen, iLen)
            MaxLookBack = max(MaxLookBack, iLookBack)
        if StartIdx >= MaxLookBack: DTRuler = dt_ruler[StartIdx-MaxLookBack:EndIdx]
        else: raise __QS_Error__(f"因子 {factor.Name}(QSID: {factor.QSID}) 运算时超出时点范围, 理论上不应该走到这个位置!")
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData, extra_dep_data)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData, extra_dep_data)


class SectionOperator(FactorOperator):
    """截面算子, 运算依赖于被依赖因子在过去若干个时点以及整个截面的因子值"""

    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Section"] = Field(default="Section", title="算子类型", frozen=True)
        Name: str = Field(default="SectionOperator", title="名称", frozen=True, exclude=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
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
            iid: 当前待计算的 ID, 也即全体截面 ID
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
    
    def _calcDataNumpySingleDT(self, factor, ids, dts, descriptor_data, SectionIDs, ModelArgs, extra_dep_data):
        if self._QSArgs.DataType=="double": StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=np.nan, dtype="float")
        else: StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=None, dtype="O")
        CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
        if CalcDTs: CalcDTs = set(CalcDTs)
        for i, iDT in enumerate(dts):
            if (CalcDTs is not None) and (iDT not in CalcDTs): continue
            StdData[i, :] = self.calculate(factor, iDT, SectionIDs, [kDescriptorData[i] for kDescriptorData in descriptor_data] + extra_dep_data, ModelArgs)
        return pd.DataFrame(StdData, columns=SectionIDs).reindex(columns=ids).values

    def _calcDataNumpyMultiDT(self, factor, ids, dts, descriptor_data, SectionIDs, ModelArgs, extra_dep_data):
        if self._QSArgs.DataType=="double": StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=np.nan, dtype="float")
        else: StdData = np.full(shape=(len(dts), len(SectionIDs)), fill_value=None, dtype="O")
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is not None:
            descriptor_data = [iData[CalcMask] for iData in descriptor_data]
            dts = np.array(dts, dtype="O")[CalcMask].tolist()
            iStdData = self.calculate(factor, dts, SectionIDs, descriptor_data + extra_dep_data, ModelArgs)
            StdData[CalcMask, :] = iStdData
        else:
            StdData = self.calculate(factor, dts, SectionIDs, descriptor_data + extra_dep_data, ModelArgs)
        return pd.DataFrame(StdData, columns=SectionIDs).reindex(columns=ids).values

    def _calcDataNumpy(self, context, factor, ids, dts, descriptor_data, SectionIDs, ModelArgs, extra_dep_data):
        if self._QSArgs.DTMode == '多时点':
            TargetFunc = self._calcDataNumpyMultiDT
        elif self._QSArgs.DTMode == '单时点':
            TargetFunc = self._calcDataNumpySingleDT
        TaskExecutor = factor._QSArgs.TaskExecutor if factor._QSArgs.TaskExecutor is not None else context.TaskExecutor
        if (not factor._QSArgs.Parallel) or (TaskExecutor is None) or (context.MaxWorkers <= 1): 
            return TargetFunc(factor=factor, ids=ids, dts=dts, descriptor_data=descriptor_data, SectionIDs=SectionIDs, ModelArgs=ModelArgs, extra_dep_data=extra_dep_data)
        Futures = []
        BatchSize = len(dts) // context.MaxWorkers + (len(dts) % context.MaxWorkers > 0)
        for i in range(context.MaxWorkers):
            iStartIdx, iEndIdx = i * BatchSize, (i + 1) * BatchSize
            iDTs = dts[iStartIdx:iEndIdx]
            if not iDTs: continue
            Futures.append(TaskExecutor.submit(TargetFunc, factor, ids, iDTs, [iData[iStartIdx:iEndIdx] for iData in descriptor_data], SectionIDs, ModelArgs, extra_dep_data))
        return np.vstack([iFuture.result() for iFuture in Futures])
    
    def _calcDataPandas(self, factor, ids, dts, descriptor_data, SectionIDs, ModelArgs, extra_dep_data):
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
        CalcDTs = (np.array(dts, dtype="O")[CalcMask].tolist() if CalcMask is not None else dts)
        if self._QSArgs.DTMode=="单时点":
            StdData = []
            for i, iDT in enumerate(CalcDTs):
                iStdData = self.calculate(factor, iDT, SectionIDs, [iData.loc[iDT] for iData in descriptor_data] + extra_dep_data, ModelArgs)
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
            StdData = self.calculate(factor, CalcDTs, SectionIDs, descriptor_data + extra_dep_data, ModelArgs)
            return self._QS_adjOutputPandas(StdData, CompoundCols, CalcDTs, ids).reindex(index=dts)

    def calcData(self, context:FactorContext, factor:Factor, ids:List[str], dts:List[dt.datetime], descriptor_data:List[np.ndarray | pd.DataFrame], dt_ruler:Optional[List[dt.datetime]]=None, section_ids:Optional[List[str]]=None, extra_dep_data:List[Any]=[]):
        ModelArgs = self._QSArgs.ModelArgs
        if section_ids is None: section_ids = ids
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, section_ids, ModelArgs, extra_dep_data)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, section_ids, ModelArgs, extra_dep_data)


class PanelOperator(FactorOperator):
    """面板算子, 运算依赖于被依赖因子在单个时点和整个截面的因子值"""

    class __QS_ArgClass__(FactorOperator.__QS_ArgClass__):
        OperatorType: Literal["Panel"] = Field(default="Panel", title="算子类型", frozen=True)
        Name: str = Field(default="PanelOperator", title="名称", frozen=True, exclude=True)
        DTMode: Literal["单时点", "多时点"] = Field(default="单时点", title="运算时点", frozen=True)
        DescriptorSection: List[Optional[List[str]]] = Field(default=[], title="描述子截面", frozen=True, description="None 表示该描述子和当前因子的截面一致")
        LookBack: List[int] = Field(default=[], title="回溯期数", frozen=True, description="描述子向前回溯的时点数(不包括当前时点)")
        # LookBackMode: List[Literal["滚动窗口", "扩张窗口"]] = Field(default=[], title="回溯模式", description="描述子的回溯模式", frozen=True)
        StartDT: List[Optional[dt.datetime]] = Field(default=[], title="起始时点", frozen=True, description="如果描述子对应的该参数非 None 表示为扩张窗口模式, 该参数为描述子数据的起始时点")
        iInitFactor: int = Field(default=-1, title="起始因子", ge=-1, frozen=True)
        DescriptorDTRuler: List[Optional[List[dt.datetime]]] = Field(
            default=[], title="描述子时点标尺", frozen=True,
            description="每个描述子的时点标尺, None 表示使用因子的时点标尺. "
                        "决定回溯的频率: 日度标尺回溯N天, 月度标尺回溯N个月."
        )

        def __init__(self, /, **data):
            Arity = data.get("Arity", 0)
            if Arity is None: Arity = 0
            if not data.get("DescriptorSection", []): data["DescriptorSection"] = [None] * Arity
            if not data.get("LookBack", []): data["LookBack"] = [0] * Arity
            # if not data.get("LookBackMode", []): data["LookBackMode"] = ["滚动窗口"] * Arity
            if not data.get("StartDT", []): data["StartDT"] = [None] * Arity
            DescriptorDTRuler = data.get("DescriptorDTRuler", [])
            if not DescriptorDTRuler:
                data["DescriptorDTRuler"] = [None] * Arity
            elif len(DescriptorDTRuler) < Arity:
                data["DescriptorDTRuler"] = list(DescriptorDTRuler) + [None] * (Arity - len(DescriptorDTRuler))
            return super().__init__(**data)

        def model_post_init(self, context: Any, /) -> None:
            if self.Arity is not None:
                if self.Arity != len(self.DescriptorSection):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 DescriptorSection({self.__pydantic_fields__['DescriptorSection'].title}): {self.DescriptorSection} 的长度不一致!")
                if self.Arity != len(self.LookBack):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBack({self.__pydantic_fields__['LookBack'].title}): {self.LookBack} 的长度不一致!")
                # if self.Arity != len(self.LookBackMode):
                #     raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBackMode({self.__pydantic_fields__['LookBackMode'].title}): {self.LookBackMode} 的长度不一致!")
                if self.Arity != len(self.StartDT):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 StartDT({self.__pydantic_fields__['StartDT'].title}): {self.StartDT} 的长度不一致!")
                # 自动填充 DescriptorDTRuler（兼容旧数据）
                if len(self.DescriptorDTRuler) < self.Arity:
                    object.__setattr__(self, "DescriptorDTRuler", list(self.DescriptorDTRuler) + [None] * (self.Arity - len(self.DescriptorDTRuler)))
                if self.iInitFactor >= self.Arity:
                    raise __QS_Error__(f"算子{self.Name}的 iInitFactor({self.__pydantic_fields__['iInitFactor'].title}): {self.iInitFactor} 超出了 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity}!")
            return super().model_post_init(context)
    
    def calculate(self, f: Factor, idt: dt.datetime | List[dt.datetime], iid: str | List[str], x: list, args: dict):
        """算子的运算逻辑实现

        Args:
            f: 该算子所属的因子对象
            idt: 当前待计算的时点, 如果 DTMode 为多时点, 则该值为时点序列 list[datetime]
            iid: 当前待计算的 ID, 也即全体截面 ID
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
    
    def _calcDataNumpySingleDT(self, factor, ids, dts, descriptor_data, jStartIdx, jEndIdx, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
        CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
        if CalcDTs: CalcDTs = set(CalcDTs)
        for i, iDT in enumerate(dts):
            if (i < jStartIdx) or (i >= jEndIdx): continue
            if (CalcDTs is not None) and (iDT not in CalcDTs): continue
            iDTs = DTRuler[max(0, MaxLookBack+i+1-MaxLen):i+1+MaxLookBack]
            x = []
            for k, kDescriptorData in enumerate(descriptor_data):
                kStartInd, kLen = StartIndAndLen[k]
                x.append(kDescriptorData[max(0, kStartInd+1+i-kLen):kStartInd+1+i])
            StdData[iStartIdx+i, :] = self.calculate(factor, iDTs, SectionIDs, x + extra_dep_data, ModelArgs)
        return pd.DataFrame(StdData[iStartIdx+jStartIdx:iStartIdx+jEndIdx, :], columns=SectionIDs).reindex(columns=ids).values
    
    def _calcDataNumpyMultiDT(self, factor, ids, dts, descriptor_data, jStartIdx, jEndIdx, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
        StdData = self.calculate(factor, DTRuler, SectionIDs, descriptor_data + extra_dep_data, ModelArgs)
        CalcMask = factor._QS_getCalcDTs(dts, mask=True)
        if CalcMask is not None:
            StdData[~CalcMask, :] = None
        return pd.DataFrame(StdData, columns=SectionIDs).reindex(columns=ids).values

    def _calcDataNumpy(self, context, factor, ids, dts, descriptor_data, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
        if self._QSArgs.DTMode == '多时点':
            TargetFunc = self._calcDataNumpyMultiDT
        elif self._QSArgs.DTMode == '单时点':
            TargetFunc = self._calcDataNumpySingleDT
        TaskExecutor = factor._QSArgs.TaskExecutor if factor._QSArgs.TaskExecutor is not None else context.TaskExecutor
        if (not factor._QSArgs.Parallel) or (self._QSArgs.iInitFactor >= 0) or (TaskExecutor is None) or (context.MaxWorkers <= 1): 
            return TargetFunc(factor=factor, ids=ids, dts=dts, descriptor_data=descriptor_data, jStartIdx=0, jEndIdx=len(dts), DTRuler=DTRuler, SectionIDs=SectionIDs, StartIndAndLen=StartIndAndLen, MaxLookBack=MaxLookBack, MaxLen=MaxLen, iStartIdx=iStartIdx, ModelArgs=ModelArgs, StdData=StdData, extra_dep_data=extra_dep_data)
        Futures = []
        BatchSize = len(dts) // context.MaxWorkers + (len(dts) % context.MaxWorkers > 0)
        for j in range(context.MaxWorkers):
            jStartIdx, jEndIdx = j * BatchSize, min(len(dts), (j + 1) * BatchSize)
            jDTs = dts[jStartIdx:jEndIdx]
            if not jDTs: continue
            Futures.append(TaskExecutor.submit(TargetFunc, factor, ids, jDTs, descriptor_data, jStartIdx, jEndIdx, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data))
        return np.vstack([iFuture.result() for iFuture in Futures])
    
    def _calcDataPandas(self, factor, ids, dts, descriptor_data, DTRuler, SectionIDs, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, ModelArgs, StdData, extra_dep_data):
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
        if self._QSArgs.DTMode=='单时点':
            CalcDTs = factor._QS_getCalcDTs(dts, mask=False)
            if CalcDTs: CalcDTs = set(CalcDTs)
            StdData = []
            for i, iDT in enumerate(dts):
                if (CalcDTs is not None) and (iDT not in CalcDTs): continue
                iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
                iStdData = self.calculate(factor, iDTs, SectionIDs, [iData.loc[iDTs] for iData in descriptor_data] + extra_dep_data, ModelArgs)
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
            StdData = self.calculate(factor, DTRuler, SectionIDs, descriptor_data + extra_dep_data, ModelArgs)
            StdData = self._QS_adjOutputPandas(StdData, CompoundCols, dts, ids)
            CalcMask = factor._QS_getCalcDTs(dts, mask=True)
            if CalcMask is not None:
                StdData[~CalcMask] = None
            return StdData

    def calcData(self, context:FactorContext, factor:Factor, ids:List[str], dts:List[dt.datetime], descriptor_data:List[np.ndarray | pd.DataFrame], dt_ruler:Optional[List[dt.datetime]]=None, section_ids:Optional[List[str]]=None, extra_dep_data:List[Any]=[]):
        if dt_ruler is None: dt_ruler = dts
        if section_ids is None: section_ids = ids
        if self._QSArgs.iInitFactor >= 0:# 自身迭代
            StdData = descriptor_data[self._QSArgs.iInitFactor] = descriptor_data[self._QSArgs.iInitFactor].copy()
            iStartIdx = StdData.shape[0] - len(dts)
        else:
            if self._QSArgs.DataType=='double': StdData = np.full(shape=(len(dts), len(section_ids)), fill_value=np.nan, dtype='float')
            else: StdData = np.full(shape=(len(dts), len(section_ids)), fill_value=None, dtype='O')
            iStartIdx = 0
        StartIdx, EndIdx = np.searchsorted(dt_ruler, dts[0], side="left"), np.searchsorted(dt_ruler, dts[-1], side="right")
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1# StartIndAndLen: [(开始位置, 数据长度)], MaxLookBack: 最大回溯期, MaxLen: 最大数据长度
        for i in range(len(descriptor_data)):
            iLookBack = self._QSArgs.LookBack[i]
            if i == self._QSArgs.iInitFactor:
                iStartInd = iStartIdx
                iLookBack = (iLookBack if self._QSArgs.StartDT[i] is None else (max(0, StartIdx - np.searchsorted(dt_ruler, self._QSArgs.StartDT[i], side="left") + iLookBack)))
                iLen = (iLookBack + 1 if self._QSArgs.StartDT[i] is None else np.inf)
            elif self._QSArgs.StartDT[i] is None:
                iStartInd = iLookBack
                iLen = iLookBack + 1
            else:
                iStartInd = iLookBack = max(0, StartIdx - np.searchsorted(dt_ruler, self._QSArgs.StartDT[i], side="left") + iLookBack)
                iLen = np.inf
            StartIndAndLen.append((iStartInd, iLen))
            MaxLen = max(MaxLen, iLen)
            MaxLookBack = max(MaxLookBack, iLookBack)
        if StartIdx >= MaxLookBack: DTRuler = dt_ruler[StartIdx-MaxLookBack:EndIdx]
        else: raise __QS_Error__(f"因子 {factor.Name}(QSID: {factor.QSID}) 运算时超出时点范围, 理论上不应该走到这个位置!")
        if self._QSArgs.InputFormat == "numpy":
            return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, DTRuler, section_ids, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData, extra_dep_data)
        else:
            return self._calcDataPandas(factor, ids, dts, descriptor_data, DTRuler, section_ids, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData, extra_dep_data)


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
        Name: str = Field(default="DerivativeFactor", frozen=True, title="名称", exclude=True)
        Operator: FactorOperator = Field(title="算子", frozen=True)
        ModelArgs: dict = Field(default={}, title="参数", frozen=True)

    def __init__(self, descriptors: List[Factor], extra_deps: List[Node] = [], args: dict={}, config_file: Optional[str]=None, **kwargs):
        Operator = args["Operator"]._QS_validate(*descriptors, **kwargs.pop("operator_kwargs", {}))
        args = {"Name": Operator._QSArgs.Name} | args | {"Operator": Operator}
        super().__init__(descriptors=descriptors, extra_deps=extra_deps, args=args, config_file=config_file, **kwargs)
        self._Operator = self._QSArgs.Operator
        self._QS_checkConsistency()
        self.UserData = {}

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
        #     return context.SectionIDs

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


class PointOperation(DerivativeFactor):
    """基于单点算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Name: str = Field(default="PointOperation", frozen=True, title="名称", exclude=True)
        Operator: PointOperator = Field(title="算子", frozen=True)

    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.DataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs)
        FwdData = [FactorLocalContext(IDs=context.getID(self.QSID, ([context.PID] if Cached else (fwd_data.PIDs or [context.PID]))), DTs=CalcDTs, PIDs=fwd_data.PIDs)] * len(self._Descriptors)
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": CalcDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": CalcDTs})

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if bwd_data_list:
            SectionIDs = context.getID(self.QSID, [context.PID] if Cached else (local_context.PIDs or [context.PID]))
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not SectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=SectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                if self._Operator._QSArgs.InputFormat == "numpy":
                    StdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=CalcDTs, descriptor_data=[iBwdData.values for iBwdData in bwd_data_list[:len(self._Descriptors)]], dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                    StdData = pd.DataFrame(StdData, index=CalcDTs, columns=SectionIDs)
                else:
                    StdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=CalcDTs, descriptor_data=bwd_data_list[:len(self._Descriptors)], dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
            if Cached and (not StdData.empty):
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids={context.PID: SectionIDs}, pid=context.PID, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=self.QSID, dt_range=(CalcDTs[0], CalcDTs[-1]))
        if Cached:
            StdData = context.DataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__(f"因子 {self.Name}(QSID: {self.QSID}) 理论上不应该走到这个位置!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)


class TimeOperation(DerivativeFactor):
    """基于时序算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Name: str = Field(default="TimeOperation", frozen=True, title="名称", exclude=True)
        Operator: TimeOperator = Field(title="算子", frozen=True)
    
    def __init__(self, descriptors: List[Factor], extra_deps: List[Node] = [], args: dict={}, config_file: Optional[str]=None, **kwargs):
        args["Operator"] = Operator = args["Operator"]._QS_validate(*descriptors, **kwargs.pop("operator_kwargs", {}))
        if (Operator._QSArgs.iInitFactor >= 0) and (Operator._QSArgs.StartDT[Operator._QSArgs.iInitFactor] is None): # 自身迭代且为滚动窗口，给出警告且强制将缓存去掉
            Operator.Logger.warning(f"算子 {Operator.Name}(QSID: {Operator.QSID}) 为自身迭代且滚动窗口模式，在缓存的不同状态下产生的数据会不一致，所以该算子作用的因子将强制不使用缓存!")
            args["CacheEnabled"] = False
        return super().__init__(descriptors=descriptors, extra_deps=extra_deps, args=args, config_file=config_file, **kwargs)

    def _get_descriptor_dtruler(self, i: int, default: List[dt.datetime]) -> List[dt.datetime]:
        """获取描述子的时点标尺

        优先从算子的 DescriptorDTRuler 获取（描述子的频率），
        如果算子未配置，则使用默认值（因子的时点标尺）。

        Args:
            i: 描述子索引
            default: 默认时点标尺（因子的时点标尺）

        Returns:
            描述子的时点标尺
        """
        if (i < len(self._Operator._QSArgs.DescriptorDTRuler) and
            self._Operator._QSArgs.DescriptorDTRuler[i] is not None):
            return self._Operator._QSArgs.DescriptorDTRuler[i]
        return default

    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {})
        StartDT, EndDT = FactorState["dt_range"]
        if (self._Operator._QSArgs.iInitFactor >= 0) and (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor] is not None): # 自身迭代且为扩张窗口，修改自身的起始日为 StartDT[i]
            FactorState["dt_range"] = (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor], EndDT)
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
            # 获取描述子的时点标尺（决定回溯频率）
            iDescDTRuler = self._get_descriptor_dtruler(i, DTRuler)
            if self._Operator._QSArgs.StartDT[i] is None:# 指定起始时点, 滚动窗口模式，从当前时点回溯 LookBack[i] 期
                # 使用描述子的时点标尺计算回溯索引
                iDescStartIdx = np.searchsorted(iDescDTRuler, StartDT, side="left")
                iStartIdx = StartIdx - self._Operator._QSArgs.LookBack[i]
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                    iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
                else:
                    iEndDT = EndDT
            else:# 指定了起始时点, 扩张窗口模式, 以起始时点 StartDT[i] 的位置为准
                if self._Operator._QSArgs.StartDT[i] < DTRuler[0]:
                    raise __QS_Error__(f"对于因子 {self.Name}(QSID: {self.QSID}) 的描述子 '{iDescriptor.Name}'(QSID: {iDescriptor.QSID}), 参数 StartDT 为 {self._Operator._QSArgs.StartDT[i]}, 时点标尺长度不足, 起始时点为 {DTRuler[0]}")
                iStartDTIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以 StartDT[i] 的上一个时点为结束时点
                    iStartIdx = min(StartIdx, iStartDTIdx - self._Operator._QSArgs.LookBack[i])
                    iEndDT = min(DTRuler[max(0, iStartDTIdx - 1)], EndDT)
                else:
                    iStartIdx = min(StartIdx, iStartDTIdx) - self._Operator._QSArgs.LookBack[i]
                    iEndDT = EndDT
            if iStartIdx < 0:
                raise __QS_Error__(f"对于因子 {self.Name}(QSID: {self.QSID}) 的描述子 '{iDescriptor.Name}'(QSID: {iDescriptor.QSID}), 参数 StartDT 为 {self._Operator._QSArgs.StartDT[i]}, 参数 LookBack 为 {self._Operator._QSArgs.LookBack[i]}, 时点标尺长度不足, 超出了 {abs(iStartIdx)} 个时点")
            InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"DTRange": (DTRuler[iStartIdx], iEndDT)}))
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.DataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        DTRuler = context.DTRuler
        StartIdx, EndIdx = DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1])
        SectionIDs = context.getID(self.QSID, ([context.PID] if Cached else (fwd_data.PIDs or [context.PID])))
        FwdData = []
        for i in range(len(self._Descriptors)):
            # 获取描述子的时点标尺
            iDescDTRuler = self._get_descriptor_dtruler(i, DTRuler)
            if self._Operator._QSArgs.StartDT[i] is None:# 滚动窗口模式
                iStartIdx, iEndIdx = StartIdx - self._Operator._QSArgs.LookBack[i], EndIdx
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                    iEndIdx = StartIdx - 1
            else:
                iStartDTIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以 StartDT 的上一个时点为结束时点
                    iStartIdx = min(StartIdx, iStartDTIdx - self._Operator._QSArgs.LookBack[i], DTRuler.index(fwd_data.DTs[0]))
                    iEndIdx = min(EndIdx, iStartDTIdx - 1)
                else:
                    iStartIdx, iEndIdx = min(StartIdx, iStartDTIdx) - self._Operator._QSArgs.LookBack[i], EndIdx
            iDTs = iDescDTRuler[iStartIdx:iEndIdx+1] if iDescDTRuler is not DTRuler else DTRuler[iStartIdx:iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=SectionIDs, DTs=iDTs, PIDs=fwd_data.PIDs))
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": CalcDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": CalcDTs})

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if bwd_data_list:
            SectionIDs = context.getID(self.QSID, [context.PID] if Cached else (local_context.PIDs or [context.PID]))
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not SectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=SectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                NeedCalc, StartDropNum, EndDropNum = True, 0, 0
                if self._Operator._QSArgs.iInitFactor >= 0:# 自身迭代，拼接自身数据
                    i = self._Operator._QSArgs.iInitFactor
                    iDescriptorData = bwd_data_list[i]
                    if (iDescriptorData.index[0] <= CalcDTs[0]) and (iDescriptorData.index[-1] >= CalcDTs[-1]):# 描述子数据已经覆盖了自身的时点范围
                        StdData = bwd_data_list[i].reindex(index=CalcDTs)
                        NeedCalc = False
                    else:
                        DTRuler = context.DTRuler
                        iDTs = DTRuler[DTRuler.index(iDescriptorData.index[-1]) + 1: DTRuler.index(CalcDTs[-1]) + 1]
                        if Cached and context.DataCache.checkFactorDataExistence(key=self.QSID, pids=[context.PID]):# 如果有缓存，取用缓存数据
                            NonCachedDTRange = context.DataCache.getDTRange(key=self.QSID, dt_range=(CalcDTs[0], CalcDTs[-1]))
                            StartDropNum, EndDropNum = DTRuler.index(NonCachedDTRange[0]) - DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1]) - DTRuler.index(NonCachedDTRange[-1])
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = context.DataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=[context.PID], data_type=self._Operator._QSArgs.DataType)
                            StdData = StdData.reindex(index=iDTs)
                        else:
                            StartDropNum, EndDropNum = max(0, DTRuler.index(iDescriptorData.index[-1]) + 1 - DTRuler.index(CalcDTs[0])), 0
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = pd.DataFrame(None, index=iDTs, columns=SectionIDs)
                        bwd_data_list[i] = pd.concat([iDescriptorData.loc[DTRuler[DTRuler.index(iDTs[0]) - self._Operator._QSArgs.LookBack[i]]:], StdData], ignore_index=False)
                else:
                    AdjCalcDTs = CalcDTs
                if NeedCalc:
                    if self._Operator._QSArgs.InputFormat == "numpy":
                        DescriptorData = [iBwdData.values[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):iBwdData.shape[0]-EndDropNum] for i, iBwdData in enumerate(bwd_data_list[:len(self._Descriptors)])]
                        CalcStdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=AdjCalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                        CalcStdData = pd.DataFrame(CalcStdData, index=AdjCalcDTs, columns=SectionIDs)
                    else:
                        DescriptorData = [iBwdData.iloc[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):iBwdData.shape[0]-EndDropNum] for i, iBwdData in enumerate(bwd_data_list[:len(self._Descriptors)])]
                        CalcStdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=AdjCalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                    if self._Operator._QSArgs.iInitFactor >= 0:# 自身迭代，拼接自身数据
                        StdData.update(CalcStdData)
                        StdData = pd.concat([iDescriptorData, StdData], ignore_index=False)
                    else:
                        StdData = CalcStdData
            if Cached and (not StdData.empty):
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids={context.PID: SectionIDs}, pid=context.PID, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=self.QSID, dt_range=(StdData.index[0], StdData.index[-1]))
        if Cached:
            StdData = context.DataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__(f"因子 {self.Name}(QSID: {self.QSID}) 理论上不应该走到这个位置!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)


class SectionOperation(DerivativeFactor):
    """基于截面算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Name: str = Field(default="SectionOperation", frozen=True, title="名称", exclude=True)
        Operator: SectionOperator = Field(title="算子", frozen=True)
    
    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        for i, iDescriptor in enumerate(self._Descriptors):
            iSectionIDs = self._QS_getDescriptorSectionIDs(i, context=context)
            if iSectionIDs != InitData[i].SectionIDs:
                InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"SectionIDs": iSectionIDs}))
        if (len(context.PIDList) > 1) and (self.QSID not in context.Event):
            # context.Event[self.QSID] = context.ExtraData["mp_manager"].Event()
            context.Event[self.QSID] = Event()
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        if context.DataCache and self._QSArgs.CacheEnabled:
            DTRange = context.DataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        if context.DataCache and self._QSArgs.CacheEnabled and (len(context.PIDList) > 1):
            PID = context.PID
            DTPartition = partitionList(CalcDTs, len(context.PIDList))
            iCalcDTs = DTPartition[context.PIDList.index(PID)]
        else:
            iCalcDTs = CalcDTs
        FwdData = [FactorLocalContext(IDs=self._QS_getDescriptorSectionIDs(i, context), DTs=CalcDTs, PIDs=context.PIDList) for i, iDescriptor in enumerate(self.Descriptors)]
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": iCalcDTs, "TotalCalcDTs": CalcDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": iCalcDTs, "TotalCalcDTs": CalcDTs})
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        if bwd_data_list:
            iSectionIDs = context.getID(self.QSID, pids=None)
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not iSectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=iSectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                if self._Operator._QSArgs.InputFormat == "numpy":
                    DescriptorData = [iBwdData.reindex(index=CalcDTs).values for iBwdData in bwd_data_list[:len(self._Descriptors)]]
                    StdData = self._Operator.calcData(context=context, factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=iSectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                    StdData = pd.DataFrame(StdData, index=CalcDTs, columns=iSectionIDs)
                else:
                    DescriptorData = [iBwdData.reindex(CalcDTs) for iBwdData in bwd_data_list[:len(self._Descriptors)]]
                    StdData = self._Operator.calcData(context=context, factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=bwd_data_list, dt_ruler=context.DTRuler, section_ids=iSectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
            if context.DataCache and self._QSArgs.CacheEnabled and (not StdData.empty):
                PIDIDs = context.NodeState[self.QSID]["pid_ids"]
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=self.QSID, dt_range=(CalcDTs[0], CalcDTs[-1]))
        if len(context.PIDList) > 1:
            context.Sub2MainQueue.put(("Event", context.PID, (self.QSID, 1)))
            context.Event[self.QSID].wait()
            if "TotalCalcDTs" in local_context.ExtraData:
                TotalCalcDTs = local_context.ExtraData["TotalCalcDTs"]
                if context.DataCache:
                    context.DataCache.updateDTRange(key=self.QSID, dt_range=(TotalCalcDTs[0], TotalCalcDTs[-1]))
        if context.DataCache and self._QSArgs.CacheEnabled:
            StdData = context.DataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__(f"因子 {self.Name}(QSID: {self.QSID}) 理论上不应该走到这个位置!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)


class PanelOperation(DerivativeFactor):
    """基于面板算子的衍生因子"""

    class __QS_ArgClass__(DerivativeFactor.__QS_ArgClass__):
        Name: str = Field(default="PanelOperation", frozen=True, title="名称", exclude=True)
        Operator: PanelOperator = Field(title="算子", frozen=True)
    
    def __init__(self, descriptors: List[Factor], extra_deps: List[Node] = [], args: dict={}, config_file: Optional[str]=None, **kwargs):
        args["Operator"] = Operator = args["Operator"]._QS_validate(*descriptors, **kwargs.pop("operator_kwargs", {}))
        if (Operator._QSArgs.iInitFactor >= 0) and (Operator._QSArgs.StartDT[Operator._QSArgs.iInitFactor] is None): # 自身迭代且为滚动窗口，给出警告且强制将缓存去掉
            Operator.Logger.warning(f"算子 {Operator.Name}(QSID: {Operator.QSID}) 为自身迭代且滚动窗口模式，在缓存的不同状态下产生的数据会不一致，所以该算子作用的因子将强制不使用缓存!")
            args["CacheEnabled"] = False
        return super().__init__(descriptors=descriptors, extra_deps=extra_deps, args=args, config_file=config_file, **kwargs)

    def _get_descriptor_dtruler(self, i: int, default: List[dt.datetime]) -> List[dt.datetime]:
        """获取描述子的时点标尺

        优先从算子的 DescriptorDTRuler 获取（描述子的频率），
        如果算子未配置，则使用默认值（因子的时点标尺）。

        Args:
            i: 描述子索引
            default: 默认时点标尺（因子的时点标尺）

        Returns:
            描述子的时点标尺
        """
        if (i < len(self._Operator._QSArgs.DescriptorDTRuler) and
            self._Operator._QSArgs.DescriptorDTRuler[i] is not None):
            return self._Operator._QSArgs.DescriptorDTRuler[i]
        return default

    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {})
        StartDT, EndDT = FactorState["dt_range"]
        if (self._Operator._QSArgs.iInitFactor >= 0) and (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor] is not None): # 自身迭代且为扩张窗口，修改自身的起始日为 StartDT[i]
            FactorState["dt_range"] = (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor], EndDT)
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
            # 获取描述子的时点标尺
            iDescDTRuler = self._get_descriptor_dtruler(i, DTRuler)
            if self._Operator._QSArgs.StartDT[i] is None:# 未指定起始时点, 滚动窗口模式，从当前时点回溯 LookBack[i] 期
                iStartIdx = StartIdx - self._Operator._QSArgs.LookBack[i]
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                    iEndDT = DTRuler[max(StartIdx - 1, iStartIdx)]
                else:
                    iEndDT = EndDT
            else:# 指定了起始时点, 扩张窗口模式, 以起始时点 StartDT[i] 的位置为准
                if self._Operator._QSArgs.StartDT[i] < DTRuler[0]:
                    raise __QS_Error__(f"对于因子 {self.Name}(QSID: {self.QSID}) 的描述子 '{iDescriptor.Name}'(QSID: {iDescriptor.QSID}), 参数 StartDT 为 {self._Operator._QSArgs.StartDT[i]}, 时点标尺长度不足, 起始时点为 {DTRuler[0]}")
                iStartDTIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以 StartDT[i] 的上一个时点为结束时点
                    iStartIdx = min(StartIdx, iStartDTIdx - self._Operator._QSArgs.LookBack[i])
                    iEndDT = min(DTRuler[max(0, iStartDTIdx - 1)], EndDT)
                else:
                    iStartIdx = min(StartIdx, iStartDTIdx) - self._Operator._QSArgs.LookBack[i]
                    iEndDT = EndDT
            if iStartIdx < 0:
                raise __QS_Error__(f"对于因子 {self.Name}(QSID: {self.QSID}) 的描述子 '{iDescriptor.Name}'(QSID: {iDescriptor.QSID}), 参数 StartDT 为 {self._Operator._QSArgs.StartDT[i]}, 参数 LookBack 为 {self._Operator._QSArgs.LookBack[i]}, 时点标尺长度不足, 超出了 {abs(iStartIdx)} 个时点")
            InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"DTRange": (DTRuler[iStartIdx], iEndDT), "SectionIDs": self._QS_getDescriptorSectionIDs(i, context=context)}))
        if (len(context.PIDList) > 1) and (self.QSID not in context.Event):
            # context.Event[self.QSID] = context.ExtraData["mp_manager"].Event()
            context.Event[self.QSID] = Event()
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        DTRange = context.NodeState.get(self.QSID, {}).get("dt_range", None)
        if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)# 该因子没有要计算的数据, 理论上不应该走到
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.DataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)# 当前缓存已经覆盖所有数据
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs)
        if Cached and (len(context.PIDList) > 1):
            PID = context.PID
            i = self._Operator._QSArgs.iInitFactor
            if (i >= 0) and (self._Operator._QSArgs.StartDT[i] is not None):# 自身为扩张窗口模式
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
            # 获取描述子的时点标尺
            iDescDTRuler = self._get_descriptor_dtruler(i, DTRuler)
            if self._Operator._QSArgs.StartDT[i] is None:# 滚动窗口模式
                iStartIdx, iEndIdx = StartIdx - self._Operator._QSArgs.LookBack[i], EndIdx
                iResponsibleStartIdx, iResponsibleEndIdx = ResponsibleStartIdx - self._Operator._QSArgs.LookBack[i], ResponsibleEndIdx
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以当前时点的上一个时点为结束时点
                    iEndIdx = StartIdx - 1
                    iResponsibleEndIdx = ResponsibleStartIdx - 1
            else:
                iStartDTIdx = np.searchsorted(DTRuler, self._Operator._QSArgs.StartDT[i], side="left")
                if i==self._Operator._QSArgs.iInitFactor:# 当前描述子为自身初始值因子, 以 StartDT 的上一个时点为结束时点
                    iStartIdx = min(StartIdx, iStartDTIdx - self._Operator._QSArgs.LookBack[i], DTRuler.index(fwd_data.DTs[0]))
                    iEndIdx = min(EndIdx, iStartDTIdx - 1)
                    CalcDTs = DTRuler[min(StartIdx, DTRuler.index(fwd_data.DTs[0])):EndIdx]
                else:
                    iStartIdx, iEndIdx = min(StartIdx, iStartDTIdx) - self._Operator._QSArgs.LookBack[i], EndIdx
                iResponsibleStartIdx, iResponsibleEndIdx = iStartIdx, iEndIdx
            iDTs = iDescDTRuler[iStartIdx:iEndIdx+1] if iDescDTRuler is not DTRuler else DTRuler[iStartIdx:iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=self._QS_getDescriptorSectionIDs(i, context), DTs=iDTs, PIDs=context.PIDList))
            DescriptorDTs.append(iDescDTRuler[iResponsibleStartIdx:iResponsibleEndIdx+1] if iDescDTRuler is not DTRuler else DTRuler[iResponsibleStartIdx:iResponsibleEndIdx+1])
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": ResponsibleCalcDTs, "TotalCalcDTs": CalcDTs, "DescriptorDTs": DescriptorDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": ResponsibleCalcDTs, "TotalCalcDTs": CalcDTs, "DescriptorDTs": DescriptorDTs})
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if bwd_data_list:
            SectionIDs = context.getID(self.QSID, pids=None)
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not SectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=SectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                NeedCalc, StartDropNum, EndDropNum = True, 0, 0
                DescriptorDTs = local_context.ExtraData["DescriptorDTs"]
                if self._Operator._QSArgs.iInitFactor >= 0:# 自身迭代，拼接自身数据
                    i = self._Operator._QSArgs.iInitFactor
                    if (DescriptorDTs[i][0] <= CalcDTs[0]) and (DescriptorDTs[i][-1] >= CalcDTs[-1]):# 描述子数据已经覆盖了自身的时点范围
                        StdData = bwd_data_list[i].reindex(index=CalcDTs)
                        NeedCalc = False
                    else:
                        DTRuler = context.DTRuler
                        iDescriptorData = bwd_data_list[i].reindex(index=DescriptorDTs[i])
                        iDTs = DTRuler[DTRuler.index(iDescriptorData.index[-1]) + 1: DTRuler.index(CalcDTs[-1]) + 1]
                        if Cached and context.DataCache.checkFactorDataExistence(key=self.QSID, pids=[context.PID]):# 如果有缓存，取用缓存数据
                            NonCachedDTRange = context.DataCache.getDTRange(key=self.QSID, dt_range=(CalcDTs[0], CalcDTs[-1]))
                            StartDropNum, EndDropNum = DTRuler.index(NonCachedDTRange[0]) - DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1]) - DTRuler.index(NonCachedDTRange[-1])
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = context.DataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
                            StdData = StdData.reindex(index=iDTs)
                        else:
                            StartDropNum, EndDropNum = max(0, DTRuler.index(iDescriptorData.index[-1]) + 1 - DTRuler.index(CalcDTs[0])), 0
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = pd.DataFrame(None, index=iDTs, columns=SectionIDs)
                        bwd_data_list[i] = pd.concat([iDescriptorData.loc[DTRuler[DTRuler.index(iDTs[0]) - self._Operator._QSArgs.LookBack[i]]:], StdData], ignore_index=False)
                else:
                    AdjCalcDTs = CalcDTs
                if NeedCalc:
                    if self._Operator._QSArgs.InputFormat == "numpy":
                        DescriptorData = [(iBwdData.reindex(index=DescriptorDTs[i]).values[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):len(DescriptorDTs[i])-EndDropNum] if i != self._Operator._QSArgs.iInitFactor else iBwdData.values[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):iBwdData.shape[0]-EndDropNum]) for i, iBwdData in enumerate(bwd_data_list[:len(self._Descriptors)])]
                        CalcStdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=AdjCalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                        CalcStdData = pd.DataFrame(CalcStdData, index=AdjCalcDTs, columns=SectionIDs)
                    else:
                        DescriptorData = [(iBwdData.reindex(index=DescriptorDTs[i]).iloc[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):len(DescriptorDTs[i])-EndDropNum] if i != self._Operator._QSArgs.iInitFactor else iBwdData.iloc[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):iBwdData.shape[0]-EndDropNum]) for i, iBwdData in enumerate(bwd_data_list[:len(self._Descriptors)])]
                        CalcStdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=AdjCalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                    if self._Operator._QSArgs.iInitFactor >= 0:# 自身迭代，拼接自身数据
                        StdData.update(CalcStdData)
                        StdData = pd.concat([iDescriptorData, StdData], ignore_index=False)
                    else:
                        StdData = CalcStdData
            if Cached and (not StdData.empty):
                PIDIDs = context.NodeState[self.QSID]["pid_ids"]
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=self.QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=self.QSID, dt_range=(StdData.index[0], StdData.index[-1]))
        if len(context.PIDList) > 1:
            context.Sub2MainQueue.put(("Event", context.PID, (self.QSID, 1)))
            context.Event[self.QSID].wait()
            if "TotalCalcDTs" in local_context.ExtraData:
                TotalCalcDTs = local_context.ExtraData["TotalCalcDTs"]
                if Cached:
                    context.DataCache.updateDTRange(key=self.QSID, dt_range=(TotalCalcDTs[0], TotalCalcDTs[-1]))
        if Cached:
            StdData = context.DataCache.readFactorData(key=self.QSID, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__(f"因子 {self.Name}(QSID: {self.QSID}) 理论上不应该走到这个位置!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)
