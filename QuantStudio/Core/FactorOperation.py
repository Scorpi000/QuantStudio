# -*- coding: utf-8 -*-
"""因子运算"""
import gc
import datetime as dt
from functools import partial
from typing import Optional, Literal, List, Any
from multiprocessing import Queue, Event

import pandas as pd
import numpy as np
from pydantic import Field

from QuantStudio.Core import __QS_Error__, __QS_Object__
from QuantStudio.Core.Factor import Factor, DataFactor, FactorContext
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools.AuxiliaryFun import partitionList
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame


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

        def __QS_initArgValue__(self, args={}):
            if args.get("复合类型", []) or args.get("多重映射", False): args["数据类型"] = "object"
            return super().__QS_initArgValue__(args=args)

    def __init__(self, args={}, config_file=None, **kwargs):
        super().__init__(sys_args=args, config_file=config_file, **kwargs)
        self._QS_CachedOperators = {}

    @property
    def Name(self):
        return self._QSArgs.Name

    def _QS_checkArity(self, *x):
        Arity = len(x)
        if self._QSArgs.MaxArity == 0:
            if Arity != self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 和指定的入参数 {self._QSArgs.Arity} 不符!")
        elif self._QSArgs.MaxArity < 0:
            if Arity < self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 小于最小入参数 {self._QSArgs.Arity}!")
        else:
            if Arity > self._QSArgs.MaxArity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 大于最大入参数 {self._QSArgs.MaxArity}!")
            elif Arity < self._QSArgs.Arity:
                return (False, f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 小于最小入参数 {self._QSArgs.Arity}!")
        return (True, None)

    def _QS_makeOperator(self, *x, args: dict = {}, cached_id=None):
        isOK, Msg = self._QS_checkArity(*x)
        if not isOK: raise __QS_Error__(Msg)
        if not args:
            return self
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


# 算子工厂函数
# operator_type: 算子类型, 可选: 'Point', 'Time', 'Section', 'Panel'
# sys_args: 算子参数
def makeFactorOperator(func, operator_type, args={}, **kwargs):
    if not callable(func): raise __QS_Error__("func 必须是可调用对象!")
    if operator_type == "Point":
        FactorOperator = PointOperator(args=args, config_file=None, **kwargs)
    # elif operator_type == "Time":
    #     FactorOperator = TimeOperator(args=args, config_file=None, **kwargs)
    # elif operator_type == "Section":
    #     FactorOperator = SectionOperator(args=args, config_file=None, **kwargs)
    # elif operator_type == "Panel":
    #     FactorOperator = PanelOperator(args=args, config_file=None, **kwargs)
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


class PointOperation(DerivativeFactor):
    """单点运算"""

    def __init__(self, descriptors, args={}, config_file=None, **kwargs):
        args = args.copy()
        Operator = args.pop("Operator", None)
        if Operator is None: raise __QS_Error__("创建衍生因子必须指定算子!")
        if not isinstance(Operator, FactorOperator):
            Operator = makeFactorOperator(operator_type="Point", func=Operator, args=args, logger=descriptors[0]._QS_Logger)
        elif not isinstance(Operator, PointOperator):
            raise __QS_Error__(f"类型为 PointOperation 的衍生因子的算子类型必须为 PointOperator, 但传入的算子类型为 {Operator.__class__}")
        return super().__init__(descriptors=descriptors, args={"Operator": Operator, **args}, config_file=config_file, **kwargs)

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Any=None) -> Any:
        pass

    def readData(self, ids, dts, **kwargs):
        Context = self.BatchContext
        if Context is not None:
            return Context.readData(factors=[self], ids=ids, dts=dts, **kwargs).iloc[0]
        if self._Operator._QSArgs.InputFormat == "numpy":
            StdData = self._Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs).values for iDescriptor in self._Descriptors])
            return pd.DataFrame(StdData, index=dts, columns=ids)
        else:
            StdData = self._Operator.calcData(factor=self, ids=ids, dts=dts, descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs) for iDescriptor in self._Descriptors])
            return StdData

    def __QSBC_prepareCacheData__(self):
        Context = self.BatchContext
        DTRange = Context._DTRange.get(self._QSID, None)
        if DTRange is None: return 0
        DTRange = Context.getDTRange(self._QSID, DTRange)
        if DTRange is None: return 0
        PID = Context._iPID
        DTs = Context.getDateTime(DTRange)
        if not DTs: return 0
        IDs = Context.getID(self._QSID, [PID])
        if IDs:
            if self._Operator._QSArgs.InputFormat == "numpy":
                StdData = self._Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor.__QSBC_getData__(DTs, pids=[PID]).values for iDescriptor in self._Descriptors])
                StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
            else:
                StdData = self._Operator.calcData(factor=self, ids=IDs, dts=DTs, descriptor_data=[iDescriptor.__QSBC_getData__(DTs, pids=[PID]) for iDescriptor in self._Descriptors])
        else:
            for iDescriptor in self._Descriptors:
                iDescriptor.__QSBC_getData__(DTs, pids=[PID])
            StdData = pd.DataFrame(index=DTs, columns=IDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
        Context._Cache.writeFactorData(key=self._QSID, target_field="StdData", factor_data=StdData, pid_ids={PID: IDs}, pid=PID, if_exists="append")
        Context.updateDTRange(factor_id=self._QSID, dt_range=DTRange)
        return 0


if __name__ == "__main__":
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