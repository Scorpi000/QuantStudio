# -*- coding: utf-8 -*-
"""因子运算"""
import base64
import importlib
import copy
import types
import datetime as dt
from functools import partial
from typing import Optional, Literal, List, Any, Tuple, Callable, Union, Dict

import dill
import pandas as pd
import numpy as np
from pydantic import Field, ConfigDict, SerializeAsAny
from multiprocess import Event

from QuantStudio.Core import __QS_Error__, __QS_Object__, __QS_Args__
from QuantStudio.Core.Node import Node
from QuantStudio.Factor.Factor import Factor, DataFactor, FactorContext, FactorLocalContext, FactorInitData, makeFactorRunningKey
from QuantStudio.Tools.AuxiliaryFun import partitionList


def _serialize_operator_value(value):
    """将算子参数递归转换为 JSON 值，保留模型类型和非 JSON 值。"""
    if isinstance(value, __QS_Args__):
        model_type = type(value)
        return {"__qs_model__": base64.b64encode(dill.dumps(model_type)).decode("ascii"),
                "values": _serialize_operator_value(value.serialize() | (value.__pydantic_extra__ or {}))}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _serialize_operator_value(item) for key, item in value.items()}
    elif isinstance(value, list):
        return [_serialize_operator_value(item) for item in value]
    elif value is None or isinstance(value, (str, bool, int, float)):
        return value
    # tuple、函数及其他 Python 值使用已有的 dill 传输机制保留类型。
    return {"__qs_dill__": base64.b64encode(dill.dumps(value)).decode("ascii")}


def _deserialize_operator_value(value):
    """恢复 serialize 生成的可信算子参数及嵌套模型。"""
    if isinstance(value, list):
        return [_deserialize_operator_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__qs_model__", "values"}:
            model_type = dill.loads(base64.b64decode(value["__qs_model__"]))
            return model_type.deserialize(_deserialize_operator_value(value["values"]))
        if set(value) == {"__qs_dill__"}:
            return dill.loads(base64.b64decode(value["__qs_dill__"]))
        return {key: _deserialize_operator_value(item) for key, item in value.items()}
    return value


class OperatorModelArgs(__QS_Args__):
    """算子模型参数基类

    各算子声明自己的子类以描述其模型参数的元信息(类型/默认值/描述/取值范围等),
    下游(GUI 等)据此渲染参数表单。详见 docs/plans/factor_operator_args.md。

    子类继承 ``extra='forbid'``, 未知参数会报错; 而函数式算子(工厂创建)在未声明
    模型类时直接传入任意 dict, 因此基类放开 ``extra='allow'`` 以兼容旧用法。
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, frozen=True)

    def __init_subclass__(cls, **kwargs):
        # 声明的具体模型参数子类恢复严格校验(未知参数报错)
        super().__init_subclass__(**kwargs)
        cls.model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)


class OperandSpec(__QS_Args__):
    """算子可作用因子(操作数)的声明

    描述算子 ``__call__`` 接受的一个操作数槽位。若干 ``OperandSpec`` 组成算子
    argclass 上的 ``OperandInfo`` 字段。

    Args:
        Name: 该操作数的语义名, 如 f/mask/endog/exog
        Title: 面向用户的显示名
        Description: 该操作数的含义说明
        Required: True 为必填槽位, False 为可选槽位(如 mask/cat_data)
        Variadic: True 表示该参数接受多个因子
        KeywordOnly: True 表示只能通过关键字传入
        Grouped: True 表示多个因子以一个 list/tuple 参数传入，而非 *args
        MinNum: Variadic=True 时生效, 当前参数组的最少数量
        MaxNum: Variadic=True 时生效, None 表示不限
    """
    Name: str = Field(default="", title="名称")
    Title: str = Field(default="", title="标题")
    Description: str = Field(default="", title="描述")
    Required: bool = Field(default=True, title="必填")
    Variadic: bool = Field(default=False, title="可变元数")
    KeywordOnly: bool = Field(default=False, title="仅关键字")
    Grouped: bool = Field(default=False, title="因子列表")
    MinNum: int = Field(default=0, ge=0, title="最少个数")
    MaxNum: Optional[int] = Field(default=None, ge=1, title="最多个数")

    def model_post_init(self, context: Any, /) -> None:
        if self.Grouped and not self.Variadic:
            raise __QS_Error__("Grouped 只能用于 Variadic 操作数!")
        if self.Variadic and self.MaxNum is not None and self.MinNum > self.MaxNum:
            raise __QS_Error__("操作数 MinNum 不能大于 MaxNum!")
        return super().model_post_init(context)


def _operand_bounds(specs):
    """根据公开接口声明计算展开后的描述子总数范围。"""
    minimum, maximum = 0, 0
    for spec in specs:
        minimum += spec.MinNum if spec.Variadic else int(spec.Required)
        if maximum is not None:
            maximum = (None if spec.Variadic and spec.MaxNum is None else
                       maximum + (spec.MaxNum if spec.Variadic else 1))
    return minimum, maximum


def _copy_operator_value(value):
    """复制参数容器，保留模块和函数等不可深复制的计算对象。"""
    if isinstance(value, __QS_Args__):
        return type(value)(**{key: _copy_operator_value(item) for key, item in (value | {}).items()})
    if isinstance(value, dict):
        return {key: _copy_operator_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_operator_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_operator_value(item) for item in value)
    if isinstance(value, types.ModuleType) or callable(value):
        return value
    return copy.deepcopy(value)


class FactorOperator(__QS_Object__):
    """因子算子"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        OperatorType: Literal["Point", "Time", "Section", "Panel"] = Field(title="算子类型", frozen=True)
        Name: str = Field(default="FactorOperator", title="名称", frozen=True, exclude=True)
        ModelArgs: SerializeAsAny[OperatorModelArgs] = Field(default_factory=OperatorModelArgs, title="模型参数", frozen=True)
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(), title="操作数声明", frozen=True, exclude=True)
        Arity: Optional[int] = Field(default=None, ge=1, title="入参数量", frozen=True)
        DataType: Literal["double", "string", "object"] = Field(default="double",title="数据类型", frozen=True)
        Description: str = Field(default="", title="描述信息", frozen=False, exclude=True)
        Meta: Dict[str, Any] = Field(default={}, title="元信息", frozen=False, exclude=True)
        CompoundType: List[Tuple[str, Literal["double", "string", "object"]]] = Field(default=[], title="复合类型", repr=False, frozen=True)

        def __init__(self, /, **data: Any) -> None:
            if ("DataType" not in data) and data.get("CompoundType", []):
                data["DataType"] = "object"
            return super().__init__(**data)

        def model_post_init(self, context: Any, /) -> None:
            # 校验 Arity 与 OperandInfo 的一致性:
            # 可变元数算子的 Arity 由实际调用确定, 只校验上下界;
            # 必填槽位数 <= Arity <= 槽位总数
            OperandInfo = self.OperandInfo
            names = [spec.Name for spec in OperandInfo if spec.Name]
            if len(names) != len(set(names)):
                raise __QS_Error__("操作数声明的名称不能重复!")
            positional = [spec for spec in OperandInfo if not spec.KeywordOnly]
            tails = [i for i, spec in enumerate(positional) if spec.Variadic and not spec.Grouped]
            if tails and (len(tails) > 1 or tails[0] != len(positional) - 1):
                raise __QS_Error__("展开式可变操作数必须是最后一个位置参数!")
            if self.Arity is not None and OperandInfo:
                MinArity, MaxArity = _operand_bounds(OperandInfo)
                if (self.Arity < MinArity) or ((MaxArity is not None) and (self.Arity > MaxArity)):
                    raise __QS_Error__(f"算子 {self.Name} 的 Arity({self.Arity}) 超出了 OperandInfo 声明的范围 [{MinArity}, {MaxArity if MaxArity is not None else '不限'}]!")
            return super().model_post_init(context)
    
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

    @classmethod
    def _QS_fromArgs(cls, args, **kwargs):
        """从完整配置统一重建对象，不调用便捷构造函数。

        Args:
            args: 已包含默认值和派生配置的完整参数。
            kwargs: 日志等对象初始化参数。

        Returns:
            经过参数模型校验的新算子。
        """
        op = cls.__new__(cls)
        __QS_Object__.__init__(op, args=args, config_file=None, **kwargs)
        return op

    def _QS_updateArgs(self, args, updates):
        """更新与元数、窗口和输出类型有关的派生配置。"""
        old_model = self.Args.ModelArgs
        model = args["ModelArgs"]
        arity = args.get("Arity")
        for field in ("LookBack", "StartDT", "DescriptorSection"):
            if arity is not None and field in args and field not in updates:
                values = args[field]
                if len(values) != arity:
                    fill = values[0] if len(values) == 1 else (0 if field == "LookBack" else None)
                    args[field] = (values[:arity] + [fill] * max(0, arity - len(values)))
        if model.get("window") != old_model.get("window") and "LookBack" in args and "LookBack" not in updates:
            window = model.get("window")
            if window is not None:
                args["LookBack"] = [window - 1] * (arity or len(args["LookBack"]))
        if model.get("dtype") != old_model.get("dtype") and "DataType" not in updates:
            args["DataType"] = model.dtype
        return args

    def new(self, args={}, **kwargs):
        Args = _copy_operator_value(self._QSArgs | {}) | args
        model_values = _copy_operator_value(self.Args.ModelArgs | {})
        # 兼容原 operator_kwargs 中直接传入的模型构造形参。
        model_updates = {key: kwargs.pop(key) for key in list(kwargs) if key in model_values}
        supplied = args.get("ModelArgs", {})
        if isinstance(supplied, OperatorModelArgs):
            Args["ModelArgs"] = _copy_operator_value(supplied)
        else:
            Args["ModelArgs"] = type(self.Args.ModelArgs)(**(model_values | model_updates | supplied))
        Args = self._QS_updateArgs(Args, args)
        # 重建不再读取配置文件，防止本地配置覆盖保存的完整参数。
        kwargs.pop("config_file", None)
        NewOperator = self._QS_fromArgs(Args, **({"logger": self._QS_Logger} | kwargs))
        NewOperator._ConfigFile = self._ConfigFile
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
        result["__qsargs__"] = _serialize_operator_value(result["__qsargs__"])
        result["__class__"] = type(self).__module__ + "." + type(self).__qualname__
        result["calculate_ref"] = self._serialize_calculate()
        return result

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "FactorOperator":
        """从序列化 dict 重建算子实例，恢复 calculate"""
        class_path = data.get("__class__", "")
        if "." in class_path:
            module_path, class_name = class_path.rsplit(".", 1)
            cls = getattr(importlib.import_module(module_path), class_name)
        Args = cls.__QS_ArgClass__.deserialize(_deserialize_operator_value(data.get("__qsargs__", {})))
        # 不经 model_dump，避免恢复后的模型类型与 exclude 字段再次丢失。
        # 参数已完整恢复并校验，直接初始化对象，避免内置构造函数要求
        # func/aggr_ids 等位置参数，或重新推导并覆盖保存的配置。
        op = cls._QS_fromArgs(Args | {})
        cls._deserialize_calculate(op, data.get("calculate_ref"))
        return op
    # endregion
    
    def _QS_validate(self, *x, **kwargs):
        Arity = len(x)
        OperandInfo = self._QSArgs.OperandInfo
        # 可变元数算子: 校验实参个数是否落在 OperandInfo 声明的范围内
        if OperandInfo:
            MinArity, MaxArity = _operand_bounds(OperandInfo)
            if Arity < MinArity:
                raise __QS_Error__(f"因子算子 {self._QSArgs.Name} 至少需要 {MinArity} 个操作数, 实际传入 {Arity} 个!")
            if (MaxArity is not None) and (Arity > MaxArity):
                raise __QS_Error__(f"因子算子 {self._QSArgs.Name} 最多接受 {MaxArity} 个操作数, 实际传入 {Arity} 个!")
        if self._QSArgs.Arity is None:
            return self.new(args={"Arity": Arity}, **kwargs)
        elif Arity != self._QSArgs.Arity:
            raise __QS_Error__(f"因子算子 {self._QSArgs.Name} 实际传入的因子数量 {Arity} 和指定的入参数 {self._QSArgs.Arity} 不符!")
        else:
            return self

    def _QS_validateOperands(self, *operands, **named):
        """按公开接口绑定并校验各操作数，展开前分别检查参数组。"""
        position = 0
        flattened = []
        for spec in self.Args.OperandInfo:
            if spec.KeywordOnly:
                value = named.pop(spec.Name, None)
            elif spec.Name in named:
                if position < len(operands):
                    raise __QS_Error__(f"操作数 {spec.Name} 重复传入!")
                value = named.pop(spec.Name)
            elif spec.Variadic and not spec.Grouped:
                value = operands[position:]
                position = len(operands)
            else:
                value = operands[position] if position < len(operands) else None
                position += int(position < len(operands))
            if spec.Variadic:
                if value is None:
                    value = ()
                if not isinstance(value, (list, tuple)):
                    raise __QS_Error__(f"操作数 {spec.Name} 必须是因子列表或元组!")
                count = len(value)
                if count < spec.MinNum or (spec.MaxNum is not None and count > spec.MaxNum):
                    raise __QS_Error__(f"操作数 {spec.Name} 的数量 {count} 超出声明范围!")
                flattened.extend(value)
            elif spec.Required and value is None:
                raise __QS_Error__(f"缺少必填操作数 {spec.Name}!")
            elif value is not None:
                flattened.append(value)
        if position < len(operands) or named:
            raise __QS_Error__("传入了未声明的操作数!")
        return tuple(flattened)

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
        bound = kwargs.pop("_qs_bound", False)
        if self.Args.OperandInfo and not bound:
            named = {spec.Name: kwargs.pop(spec.Name) for spec in self.Args.OperandInfo if spec.Name in kwargs}
            x = self._QS_validateOperands(*x, **named)
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

    def calcData(self, context:FactorContext, factor:Factor, ids:List[str], dts:List[dt.datetime], descriptor_data:List[np.ndarray | pd.DataFrame], dt_ruler:Optional[List[dt.datetime]]=None, section_ids:Optional[List[str]]=None, extra_dep_data:List[Any]=[]):
        return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, self._QSArgs.ModelArgs, extra_dep_data)


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
        
        def __init__(self, /, **data):
            Arity = data.get("Arity", 0)
            if Arity is None: Arity = 0
            if not data.get("LookBack", []): data["LookBack"] = [0] * Arity
            # if not data.get("LookBackMode", []): data["LookBackMode"] = ["滚动窗口"] * Arity
            if not data.get("StartDT", []): data["StartDT"] = [None] * Arity
            return super().__init__(**data)
        
        def model_post_init(self, context: Any, /) -> None:
            if self.Arity is not None:
                if self.Arity != len(self.LookBack):
                    raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBack({self.__pydantic_fields__['LookBack'].title}): {self.LookBack} 的长度不一致!")
                # if self.Arity != len(self.LookBackMode):
                #     raise __QS_Error__(f"算子{self.Name}的 Arity({self.__pydantic_fields__['Arity'].title}): {self.Arity} 和 LookBackMode({self.__pydantic_fields__['LookBackMode'].title}): {self.LookBackMode} 的长度不一致!")
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
        bound = kwargs.pop("_qs_bound", False)
        if self.Args.OperandInfo and not bound:
            named = {spec.Name: kwargs.pop(spec.Name) for spec in self.Args.OperandInfo if spec.Name in kwargs}
            x = self._QS_validateOperands(*x, **named)
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
        return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData, extra_dep_data)


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
        bound = kwargs.pop("_qs_bound", False)
        if self.Args.OperandInfo and not bound:
            named = {spec.Name: kwargs.pop(spec.Name) for spec in self.Args.OperandInfo if spec.Name in kwargs}
            x = self._QS_validateOperands(*x, **named)
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
    
    def calcData(self, context:FactorContext, factor:Factor, ids:List[str], dts:List[dt.datetime], descriptor_data:List[np.ndarray | pd.DataFrame], dt_ruler:Optional[List[dt.datetime]]=None, section_ids:Optional[List[str]]=None, extra_dep_data:List[Any]=[]):
        ModelArgs = self._QSArgs.ModelArgs
        if section_ids is None: section_ids = ids
        return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, section_ids, ModelArgs, extra_dep_data)


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
        
        def __init__(self, /, **data):
            Arity = data.get("Arity", 0)
            if Arity is None: Arity = 0
            if not data.get("DescriptorSection", []): data["DescriptorSection"] = [None] * Arity
            if not data.get("LookBack", []): data["LookBack"] = [0] * Arity
            # if not data.get("LookBackMode", []): data["LookBackMode"] = ["滚动窗口"] * Arity
            if not data.get("StartDT", []): data["StartDT"] = [None] * Arity
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
        bound = kwargs.pop("_qs_bound", False)
        if self.Args.OperandInfo and not bound:
            named = {spec.Name: kwargs.pop(spec.Name) for spec in self.Args.OperandInfo if spec.Name in kwargs}
            x = self._QS_validateOperands(*x, **named)
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
        return self._calcDataNumpy(context, factor, ids, dts, descriptor_data, DTRuler, section_ids, StartIndAndLen, MaxLookBack, MaxLen, iStartIdx, self._QSArgs.ModelArgs, StdData, extra_dep_data)


FactorOperatorFunc = Callable[[Factor, dt.datetime | List[dt.datetime], str | List[str], list, dict], Any]

def makeFactorOperator(func: FactorOperatorFunc, operator_type: Literal['Point', 'Time', 'Section', 'Panel'], args:dict={},
                       operand_info:Optional[Tuple[OperandSpec, ...]]=None, model_args:Optional[type]=None, **kwargs) -> FactorOperator:
    """算子工厂函数, 给定一个函数创建一个因子算子对象

    Args:
        func: 定义了算子运算逻辑的函数
        operator_type: 算子类型
        args: 创建算子对象时传递给它的参数集
        operand_info: 操作数声明 ``Tuple[OperandSpec, ...]``, 描述算子可作用的因子槽位
        model_args: 模型参数类(``OperatorModelArgs`` 子类), 用于描述算子的模型参数元信息
        kwargs: 创建算子对象时传递给它的其他入参

    Returns:
        创建的因子算子对象
    """
    if not callable(func): raise __QS_Error__("func 必须是可调用对象!")
    if "Name" not in args: args = args | {"Name": func.__qualname__}
    if operand_info is not None:
        args = args | {"OperandInfo": tuple(operand_info)}
    # ModelArgs 双形态兼容: 已知模型类时, dict -> 模型实例; 否则 dict 交由 pydantic 校验
    MA = args.get("ModelArgs")
    if (model_args is not None) and isinstance(MA, dict):
        args = args | {"ModelArgs": model_args(**MA)}
    elif (model_args is not None) and (MA is None):
        args = args | {"ModelArgs": model_args()}
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

def FactorOperatorized(operator_type: Literal['Point', 'Time', 'Section', 'Panel'], args:dict={},
                       operand_info:Optional[Tuple[OperandSpec, ...]]=None, model_args:Optional[type]=None, **kwargs) -> Callable[[FactorOperatorFunc], FactorOperator]:
    """将函数转换成因子算子对象的装饰器

    Args:
        operator_type: 算子类型
        args: 创建算子对象时传递给它的参数集
        operand_info: 操作数声明 ``Tuple[OperandSpec, ...]``
        model_args: 模型参数类(``OperatorModelArgs`` 子类)
        kwargs: 创建算子对象时传递给它的其他入参

    Returns:
        算子装饰器
    """
    return partial(makeFactorOperator, operator_type=operator_type, args=args, operand_info=operand_info, model_args=model_args, **kwargs)


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
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=fwd_data.SectionIDs or self._QSArgs.SectionIDs, context=context)
        DTRange = context.NodeState.get(self.QSID, {}).get(RunningKey, {}).get("dt_range", None)
        if DTRange is None: 
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的 DTRange is None, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.DataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: 
                self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的缓存数据已覆盖所需的时点, 返回空的 forward 数据, 终止前向传递")
                return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs:
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 需要计算的时点序列为空, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, SectionIDs=fwd_data.SectionIDs)
        FwdData = [FactorLocalContext(IDs=context.getID(self.QSID, running_key=RunningKey, pids=([context.PID] if Cached else (fwd_data.PIDs or [context.PID]))), DTs=CalcDTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs or self._QSArgs.SectionIDs)] * len(self._Descriptors)
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs, ExtraData={"CalcDTs": CalcDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs, ExtraData={"CalcDTs": CalcDTs})

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=local_context.SectionIDs or self._QSArgs.SectionIDs, context=context)
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if bwd_data_list:
            SectionIDs = context.getID(self.QSID, running_key=RunningKey, pids=[context.PID] if Cached else (local_context.PIDs or [context.PID]))
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not SectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=SectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                StdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=CalcDTs, descriptor_data=[iBwdData.values for iBwdData in bwd_data_list[:len(self._Descriptors)]], dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                StdData = pd.DataFrame(StdData, index=CalcDTs, columns=SectionIDs)
            if Cached and (not StdData.empty):
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=RunningKey, target_field="StdData", factor_data=StdData, pid_ids={context.PID: SectionIDs}, pid=context.PID, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=RunningKey, dt_range=(CalcDTs[0], CalcDTs[-1]))
        if Cached:
            StdData = context.DataCache.readFactorData(key=RunningKey, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
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
    
    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=init_data.SectionIDs or self._QSArgs.SectionIDs, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {}).setdefault(RunningKey, {})
        StartDT, EndDT = FactorState["dt_range"]
        if (self._Operator._QSArgs.iInitFactor >= 0) and (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor] is not None): # 自身迭代且为扩张窗口，修改自身的起始日为 StartDT[i]
            FactorState["dt_range"] = (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor], EndDT)
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
            if self._Operator._QSArgs.StartDT[i] is None:# 指定起始时点, 滚动窗口模式，从当前时点回溯 LookBack[i] 期
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
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=fwd_data.SectionIDs or self._QSArgs.SectionIDs, context=context)
        DTRange = context.NodeState.get(self.QSID, {}).get(RunningKey, {}).get("dt_range", None)
        if DTRange is None: 
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的 DTRange is None, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.DataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: 
                self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的缓存数据已覆盖所需的时点, 返回空的 forward 数据, 终止前向传递")
                return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: 
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 需要计算的时点序列为空, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        DTRuler = context.DTRuler
        StartIdx, EndIdx = DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1])
        SectionIDs = context.getID(self.QSID, running_key=RunningKey, pids=([context.PID] if Cached else (fwd_data.PIDs or [context.PID])))
        FwdData = []
        for i in range(len(self._Descriptors)):
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
            iDTs = DTRuler[iStartIdx:iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=SectionIDs, DTs=iDTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs or self._QSArgs.SectionIDs))
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, SectionIDs=fwd_data.SectionIDs, PIDs=fwd_data.PIDs, ExtraData={"CalcDTs": CalcDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs, ExtraData={"CalcDTs": CalcDTs})

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=local_context.SectionIDs or self._QSArgs.SectionIDs, context=context)
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if bwd_data_list:
            SectionIDs = context.getID(self.QSID, running_key=RunningKey, pids=[context.PID] if Cached else (local_context.PIDs or [context.PID]))
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
                        if Cached and context.DataCache.checkFactorDataExistence(key=RunningKey, pids=[context.PID]):# 如果有缓存，取用缓存数据
                            NonCachedDTRange = context.DataCache.getDTRange(key=RunningKey, dt_range=(CalcDTs[0], CalcDTs[-1]))
                            StartDropNum, EndDropNum = DTRuler.index(NonCachedDTRange[0]) - DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1]) - DTRuler.index(NonCachedDTRange[-1])
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = context.DataCache.readFactorData(key=RunningKey, ipid=context.PID, target_field="StdData", pids=[context.PID], data_type=self._Operator._QSArgs.DataType)
                            StdData = StdData.reindex(index=iDTs)
                        else:
                            StartDropNum, EndDropNum = max(0, DTRuler.index(iDescriptorData.index[-1]) + 1 - DTRuler.index(CalcDTs[0])), 0
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = pd.DataFrame(None, index=iDTs, columns=SectionIDs)
                        bwd_data_list[i] = pd.concat([iDescriptorData.loc[DTRuler[DTRuler.index(iDTs[0]) - self._Operator._QSArgs.LookBack[i]]:], StdData], ignore_index=False)
                else:
                    AdjCalcDTs = CalcDTs
                if NeedCalc:
                    DescriptorData = [iBwdData.values[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):iBwdData.shape[0]-EndDropNum] for i, iBwdData in enumerate(bwd_data_list[:len(self._Descriptors)])]
                    CalcStdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=AdjCalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                    CalcStdData = pd.DataFrame(CalcStdData, index=AdjCalcDTs, columns=SectionIDs)
                    if self._Operator._QSArgs.iInitFactor >= 0:# 自身迭代，拼接自身数据
                        StdData.update(CalcStdData)
                        StdData = pd.concat([iDescriptorData, StdData], ignore_index=False)
                    else:
                        StdData = CalcStdData
            if Cached and (not StdData.empty):
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=RunningKey, target_field="StdData", factor_data=StdData, pid_ids={context.PID: SectionIDs}, pid=context.PID, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=RunningKey, dt_range=(StdData.index[0], StdData.index[-1]))
        if Cached:
            StdData = context.DataCache.readFactorData(key=RunningKey, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
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
            InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"SectionIDs": self._Operator._QSArgs.DescriptorSection[i] or InitData[i].SectionIDs or self._QSArgs.SectionIDs}))
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=init_data.SectionIDs or self._QSArgs.SectionIDs, context=context)
        if (len(context.PIDList) > 1) and (RunningKey not in context.Event):
            # context.Event[RunningKey] = context.ExtraData["mp_manager"].Event()
            context.Event[RunningKey] = Event()
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=fwd_data.SectionIDs or self._QSArgs.SectionIDs, context=context)
        NodeState = context.NodeState.get(self.QSID, {}).get(RunningKey, {})
        DTRange = NodeState.get("dt_range", None)
        if DTRange is None: 
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的 DTRange is None, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        if context.DataCache and self._QSArgs.CacheEnabled:
            DTRange = context.DataCache.getDTRange(self.QSID, DTRange)
            if DTRange is None: 
                self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的缓存数据已覆盖所需的时点, 返回空的 forward 数据, 终止前向传递")
                return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: 
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的需要计算的时点序列为空, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
        if context.DataCache and self._QSArgs.CacheEnabled and (len(context.PIDList) > 1):
            PID = context.PID
            DTPartition = partitionList(CalcDTs, len(context.PIDList))
            iCalcDTs = DTPartition[context.PIDList.index(PID)]
        else:
            iCalcDTs = CalcDTs
        SectionIDs = fwd_data.SectionIDs or self._QSArgs.SectionIDs or context.SectionIDs
        FwdData = [FactorLocalContext(IDs=self._Operator._QSArgs.DescriptorSection[i] or SectionIDs, DTs=CalcDTs, PIDs=context.PIDList, SectionIDs=self._Operator._QSArgs.DescriptorSection[i] or SectionIDs) for i, iDescriptor in enumerate(self.Descriptors)]
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs, ExtraData={"CalcDTs": iCalcDTs, "TotalCalcDTs": CalcDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs, ExtraData={"CalcDTs": iCalcDTs, "TotalCalcDTs": CalcDTs})
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=local_context.SectionIDs or self._QSArgs.SectionIDs, context=context)
        if bwd_data_list:
            iSectionIDs = context.getID(self.QSID, running_key=RunningKey, pids=None)
            CalcDTs = local_context.ExtraData["CalcDTs"]
            if (not iSectionIDs) or (not CalcDTs):
                StdData = pd.DataFrame(index=CalcDTs, columns=iSectionIDs, dtype=("float" if self._Operator._QSArgs.DataType == "double" else "O"))
            else:
                DescriptorData = [iBwdData.reindex(index=CalcDTs).values for iBwdData in bwd_data_list[:len(self._Descriptors)]]
                StdData = self._Operator.calcData(context=context, factor=self, ids=iSectionIDs, dts=CalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=iSectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                StdData = pd.DataFrame(StdData, index=CalcDTs, columns=iSectionIDs)
            if context.DataCache and self._QSArgs.CacheEnabled and (not StdData.empty):
                PIDIDs = context.NodeState[self.QSID][RunningKey]["pid_ids"]
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=RunningKey, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=RunningKey, dt_range=(CalcDTs[0], CalcDTs[-1]))
        if len(context.PIDList) > 1:
            context.Sub2MainQueue.put(("Event", context.PID, (RunningKey, 1)))
            context.Event[RunningKey].wait()
            if "TotalCalcDTs" in local_context.ExtraData:
                TotalCalcDTs = local_context.ExtraData["TotalCalcDTs"]
                if context.DataCache:
                    context.DataCache.updateDTRange(key=RunningKey, dt_range=(TotalCalcDTs[0], TotalCalcDTs[-1]))
        if context.DataCache and self._QSArgs.CacheEnabled:
            StdData = context.DataCache.readFactorData(key=RunningKey, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
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

    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=init_data.SectionIDs or self._QSArgs.SectionIDs, context=context)
        FactorState = context.NodeState.setdefault(self.QSID, {}).setdefault(RunningKey, {})
        StartDT, EndDT = FactorState["dt_range"]
        if (self._Operator._QSArgs.iInitFactor >= 0) and (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor] is not None): # 自身迭代且为扩张窗口，修改自身的起始日为 StartDT[i]
            FactorState["dt_range"] = (self._Operator._QSArgs.StartDT[self._Operator._QSArgs.iInitFactor], EndDT)
        DTRuler = context.DTRuler
        StartIdx = np.searchsorted(DTRuler, StartDT, side="left")
        for i, iDescriptor in enumerate(self._Descriptors):
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
            InitData[i] = InitData[i].__class__(**(InitData[i].model_dump() | {"DTRange": (DTRuler[iStartIdx], iEndDT), "SectionIDs": self._Operator._QSArgs.DescriptorSection[i] or InitData[i].SectionIDs or self._QSArgs.SectionIDs}))
        if (len(context.PIDList) > 1) and (RunningKey not in context.Event):
            # context.Event[RunningKey] = context.ExtraData["mp_manager"].Event()
            context.Event[RunningKey] = Event()
        return InitData
    
    def forward_compute(self, path: List[str], fwd_data: FactorLocalContext, context: FactorContext) -> Tuple[List[FactorLocalContext], FactorLocalContext]:
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=fwd_data.SectionIDs or self._QSArgs.SectionIDs, context=context)
        NodeState = context.NodeState.get(self.QSID, {}).get(RunningKey, {})
        DTRange = NodeState.get("dt_range", None)
        if DTRange is None: 
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的 DTRange is None, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)# 该因子没有要计算的数据, 理论上不应该走到
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if Cached:
            DTRange = context.DataCache.getDTRange(RunningKey, DTRange)
            if DTRange is None: 
                self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的缓存数据已覆盖所需的时点, 返回空的 forward 数据, 终止前向传递")
                return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)# 当前缓存已经覆盖所有数据
        CalcDTs = context.getDateTime(DTRange)
        if not CalcDTs: 
            self._QS_Logger.debug(f"因子 {self.Name}(QSID:{self.QSID}) 的需要计算的时点序列为空, 返回空的 forward 数据, 终止前向传递")
            return [], FactorLocalContext(DTs=fwd_data.DTs, IDs=fwd_data.IDs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs)
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
        SectionIDs = fwd_data.SectionIDs or self._QSArgs.SectionIDs or context.SectionIDs
        for i, iDescriptor in enumerate(self._Descriptors):
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
            iDTs = DTRuler[iStartIdx:iEndIdx+1]
            FwdData.append(FactorLocalContext(IDs=self._Operator._QSArgs.DescriptorSection[i] or SectionIDs, DTs=iDTs, PIDs=context.PIDList, SectionIDs=self._Operator._QSArgs.DescriptorSection[i] or SectionIDs))
            DescriptorDTs.append(DTRuler[iResponsibleStartIdx:iResponsibleEndIdx+1])
        if self._ExtraDeps:
            DefaultFwdData, _ = super().forward_compute(path=path, fwd_data=fwd_data, context=context)
            return FwdData + DefaultFwdData[len(self._Descriptors):], FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs, ExtraData={"CalcDTs": ResponsibleCalcDTs, "TotalCalcDTs": CalcDTs, "DescriptorDTs": DescriptorDTs})
        else:
            return FwdData, FactorLocalContext(IDs=fwd_data.IDs, DTs=fwd_data.DTs, PIDs=fwd_data.PIDs, SectionIDs=fwd_data.SectionIDs, ExtraData={"CalcDTs": ResponsibleCalcDTs, "TotalCalcDTs": CalcDTs, "DescriptorDTs": DescriptorDTs})
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        RunningKey = makeFactorRunningKey(qsid=self.QSID, section_ids=local_context.SectionIDs or self._QSArgs.SectionIDs, context=context)
        Cached = (context.DataCache and self._QSArgs.CacheEnabled)
        if bwd_data_list:
            SectionIDs = context.getID(self.QSID, running_key=RunningKey, pids=None)
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
                        if Cached and context.DataCache.checkFactorDataExistence(key=RunningKey, pids=[context.PID]):# 如果有缓存，取用缓存数据
                            NonCachedDTRange = context.DataCache.getDTRange(key=RunningKey, dt_range=(CalcDTs[0], CalcDTs[-1]))
                            StartDropNum, EndDropNum = DTRuler.index(NonCachedDTRange[0]) - DTRuler.index(CalcDTs[0]), DTRuler.index(CalcDTs[-1]) - DTRuler.index(NonCachedDTRange[-1])
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = context.DataCache.readFactorData(key=RunningKey, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
                            StdData = StdData.reindex(index=iDTs)
                        else:
                            StartDropNum, EndDropNum = max(0, DTRuler.index(iDescriptorData.index[-1]) + 1 - DTRuler.index(CalcDTs[0])), 0
                            AdjCalcDTs = CalcDTs[StartDropNum:len(CalcDTs)-EndDropNum]
                            StdData = pd.DataFrame(None, index=iDTs, columns=SectionIDs)
                        bwd_data_list[i] = pd.concat([iDescriptorData.loc[DTRuler[DTRuler.index(iDTs[0]) - self._Operator._QSArgs.LookBack[i]]:], StdData], ignore_index=False)
                else:
                    AdjCalcDTs = CalcDTs
                if NeedCalc:
                    DescriptorData = [(iBwdData.reindex(index=DescriptorDTs[i]).values[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):len(DescriptorDTs[i])-EndDropNum] if i != self._Operator._QSArgs.iInitFactor else iBwdData.values[(StartDropNum if self._Operator._QSArgs.StartDT[i] is None else 0):iBwdData.shape[0]-EndDropNum]) for i, iBwdData in enumerate(bwd_data_list[:len(self._Descriptors)])]
                    CalcStdData = self._Operator.calcData(context=context, factor=self, ids=SectionIDs, dts=AdjCalcDTs, descriptor_data=DescriptorData, dt_ruler=context.DTRuler, section_ids=SectionIDs, extra_dep_data=bwd_data_list[len(self._Descriptors):])
                    CalcStdData = pd.DataFrame(CalcStdData, index=AdjCalcDTs, columns=SectionIDs)
                    if self._Operator._QSArgs.iInitFactor >= 0:# 自身迭代，拼接自身数据
                        StdData.update(CalcStdData)
                        StdData = pd.concat([iDescriptorData, StdData], ignore_index=False)
                    else:
                        StdData = CalcStdData
            if Cached and (not StdData.empty):
                PIDIDs = context.NodeState[self.QSID][RunningKey]["pid_ids"]
                if context.Mode == "DEBUG": Meta = {"FactorName": self.Name, "DepName": [iDep.Name for iDep in self.Deps], "DepQSID": [iDep.QSID for iDep in self.Deps]}
                else: Meta = {}
                context.DataCache.writeFactorData(key=RunningKey, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=None, if_exists="append", data_type=self._Operator._QSArgs.DataType, meta=Meta)
                context.DataCache.updateDTRange(key=RunningKey, dt_range=(StdData.index[0], StdData.index[-1]))
        if len(context.PIDList) > 1:
            context.Sub2MainQueue.put(("Event", context.PID, (RunningKey, 1)))
            context.Event[RunningKey].wait()
            if "TotalCalcDTs" in local_context.ExtraData:
                TotalCalcDTs = local_context.ExtraData["TotalCalcDTs"]
                if Cached:
                    context.DataCache.updateDTRange(key=RunningKey, dt_range=(TotalCalcDTs[0], TotalCalcDTs[-1]))
        if Cached:
            StdData = context.DataCache.readFactorData(key=RunningKey, ipid=context.PID, target_field="StdData", pids=local_context.PIDs, data_type=self._Operator._QSArgs.DataType)
        elif not bwd_data_list:
            raise __QS_Error__(f"因子 {self.Name}(QSID: {self.QSID}) 理论上不应该走到这个位置!")
        return StdData.reindex(index=local_context.DTs, columns=local_context.IDs)
