# coding=utf-8
"""内置的因子运算"""
import datetime as dt
from typing import Optional, Dict, Union, List, Any, Callable, Tuple, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import Field
from numpy.lib import recfunctions as rfn

from QuantStudio.Core import __QS_Error__
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.FactorOperation import PointOperator, TimeOperator, SectionOperator, PanelOperator, PointOperation, TimeOperation, SectionOperation, PanelOperation, OperatorModelArgs, OperandSpec
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools import DataPreprocessingFun
from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame


# ----------------------单点运算--------------------------------
class AsTypeArgs(OperatorModelArgs):
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="转换后的数据类型")

class AsType(PointOperator):
    """数据类型转换"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: AsTypeArgs = Field(default_factory=AsTypeArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待转换数据类型的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化数据类型转换算子

        Args:
            dtype: 新的数据类型
        """
        Args = {"Name": "astype", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: AsTypeArgs) -> np.ndarray:
        if args.dtype=="double":
            return x[0].astype(float)
        elif args.dtype=="string":
            return x[0].astype(str)
        elif args.dtype=="object":
            return x[0].astype("O")
        else:
            raise Exception(f"不支持的数据类型: {args.dtype}")

class LogArgs(OperatorModelArgs):
    base: float = Field(default=np.e, gt=0, title="底数", description="对数的底数, 必须 > 0")

class Log(PointOperator):
    """对数"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: LogArgs = Field(default_factory=LogArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="取对数的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, base:float=np.e, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化对数算子

        Args:
            base: 底数
        """
        Args = {"Name": "log"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"base": base} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: LogArgs) -> np.ndarray:
        Data = x[0].astype(float)
        return np.log(np.where(Data>0, Data, np.nan)) / np.log(args.base)

class NotNull(PointOperator):
    """非NULL检测算子"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待检测的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化非NULL检测算子"""
        Args = {"Name": "notnull"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        return pd.notnull(x[0])

class IsInArgs(OperatorModelArgs):
    test_elements: List[Any] = Field(default=[], title="检测集合", description="待检测的取值集合")

class IsIn(PointOperator):
    """是否属于给定集合的检测算子"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: IsInArgs = Field(default_factory=IsInArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待检测的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, test_elements:List[Any]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化检测因子值是否属于给定集合的算子

        Args:
            test_elements: 给定的检测集合
        """
        Args = {"Name": "isin"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"test_elements": test_elements} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: IsInArgs) -> np.ndarray:
        return np.isin(x[0], args.test_elements).astype(float)

class ApplyArrayFuncArgs(OperatorModelArgs):
    func: Callable = Field(title="运算函数", description="施加到因子值上的函数")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="函数返回值的数据类型")

class ApplyArrayFunc(PointOperator):
    """施加对 array 整体运算的函数, 比如 numpy.floor"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: ApplyArrayFuncArgs = Field(default_factory=ApplyArrayFuncArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待运算的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, func:Callable, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 array 整体运算算子

        Args:
            func: 施加到每个因子值的函数
            dtype: func 函数返回值的数据类型
        """
        Args = {"Name": "applyArrayFunc", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: ApplyArrayFuncArgs) -> np.ndarray:
        return args.func(*x)

class ApplymapArgs(OperatorModelArgs):
    func: Callable[[Any], Any] = Field(default=id, title="映射函数", description="施加到每个因子值的函数")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="函数返回值的数据类型")

class Applymap(PointOperator):
    """map 操作"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: ApplymapArgs = Field(default_factory=ApplymapArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待映射的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, func:Callable[[Any], Any]=id, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 map 操作算子

        Args:
            func: 施加到每个因子值的函数
            dtype: func 函数返回值的数据类型
        """
        Args = {"Name": "applymap", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: ApplymapArgs) -> np.ndarray:
        return pd.DataFrame(x[0]).map(args.func).values

class WhereArgs(OperatorModelArgs):
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="返回值的数据类型")

class Where(PointOperator):
    """where 操作"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: WhereArgs = Field(default_factory=WhereArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="f", Title="真值", Required=True, Description="mask 为 True 时的取值"),
            OperandSpec(Name="mask", Title="条件", Required=True, Description="布尔因子, True 处取 f"),
            OperandSpec(Name="other", Title="假值", Required=True, Description="condition 为 False 时的取值"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 where 操作算子

        Args:
            dtype: where 操作返回值的数据类型
        """
        Args = {"Name": "where", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: WhereArgs) -> np.ndarray:
        return np.where(x[1], x[0], x[2])

    def __call__(self, f, mask, other, factor_args:dict={}, **kwargs) -> PointOperation:
        return super().__call__(f, mask, other, factor_args=factor_args, **kwargs)

class FetchArgs(OperatorModelArgs):
    pos: Union[int, str] = Field(default=0, title="提取位置", description="从复合因子中提取的字段位置或名称")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="提取结果的数据类型")
    compound_type: Optional[List[Tuple[str, Literal["double", "string", "object"]]]] = Field(default=None, title="复合类型", description="复合因子的字段类型定义")

class Fetch(PointOperator):
    """从复合因子中取出简单因子的操作算子"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: FetchArgs = Field(default_factory=FetchArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="复合因子", Required=True, Description="待提取的复合因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, pos:Union[int, str]=0, dtype:Literal["double", "string", "object"]="double", compound_type:List[Tuple[str, Literal["double", "string", "object"]]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "fetch", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"pos": pos, "dtype": dtype, "compound_type": compound_type} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: FetchArgs) -> np.ndarray:
        Data = x[0]
        CompoundType = args.compound_type
        if CompoundType and isinstance(args.pos, str):
            DefaultData = np.array([[None]], dtype="O")
            DefaultData[0, 0] = (None,) * len(CompoundType)
            DefaultData = DefaultData.repeat(Data.shape[0], axis=0).repeat(Data.shape[1], axis=1)
            Data = np.where(pd.notnull(Data), Data, DefaultData)
            DataType = np.dtype([(iCol, float if iType=="double" else "O") for iCol, iType in CompoundType])
        else:
            SampleData = Data[pd.notnull(Data)]
            if SampleData.shape[0]==0:
                return (np.full(Data.shape, fill_value=np.nan, dtype="float") if self._QSArgs.DataType=="double" else np.full(Data.shape, fill_value=None, dtype="O"))
            SampleData = SampleData[0]
            DataType = np.dtype([(str(i),(float if isinstance(SampleData[i], float) else "O")) for i in range(len(SampleData))])
        return Data.astype(DataType)[str(args.pos)]

    def __call__(self, x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        Operator = getattr(x, "Operator", None)
        if (Operator is not None) and (Operator._QSArgs.CompoundType != self._QSArgs.ModelArgs.compound_type):
            ModelArgs = self._QSArgs.ModelArgs.model_dump() | {"compound_type": Operator._QSArgs.CompoundType}
            return super(Fetch, self.new(args={"ModelArgs": ModelArgs})).__call__(x, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(x, factor_args=factor_args, **kwargs)

class SumArgs(OperatorModelArgs):
    all_nan: Any = Field(default=0, title="全空填充值", description="所有操作数均为空时的填充值")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="求和结果的数据类型")

class Sum(PointOperator):
    """求和"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: SumArgs = Field(default_factory=SumArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="操作数", Required=True, Variadic=True, MinNum=2, Description="参与求和的操作数, 至少两个"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, all_nan:Any=0, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "sum", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: SumArgs) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nansum(Data, axis=0)
        if args.dtype=="double": Rslt = Rslt.astype(float)
        Mask = (np.sum(pd.notnull(Data), axis=0) == 0)
        Rslt[Mask] = args.all_nan
        return Rslt

class MaxArgs(OperatorModelArgs):
    all_nan: Any = Field(default=np.nan, title="全空填充值", description="所有操作数均为空时的填充值")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="结果的数据类型")

class Max(PointOperator):
    """最大值"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: MaxArgs = Field(default_factory=MaxArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="操作数", Required=True, Variadic=True, MinNum=2, Description="参与取最大值操作数, 至少两个"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, all_nan:Any=np.nan, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "max", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: MaxArgs) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nanmax(Data, axis=0)
        if args.dtype=="double": Rslt = Rslt.astype(float)
        Mask = (np.sum(pd.notnull(Data), axis=0) == 0)
        Rslt[Mask] = args.all_nan
        return Rslt

class MinArgs(OperatorModelArgs):
    all_nan: Any = Field(default=np.nan, title="全空填充值", description="所有操作数均为空时的填充值")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="结果的数据类型")

class Min(PointOperator):
    """最小值"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: MinArgs = Field(default_factory=MinArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="操作数", Required=True, Variadic=True, MinNum=2, Description="参与取最小值操作数, 至少两个"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, all_nan:Any=np.nan, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "min", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: MinArgs) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nanmin(Data, axis=0)
        if args.dtype=="double": Rslt = Rslt.astype(float)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args.all_nan
        return Rslt

class RankArgs(OperatorModelArgs):
    ascending: bool = Field(default=True, title="升序", description="是否按升序排名")
    uniformization: bool = Field(default=True, title="归一化", description="排名是否除以总数归一化")

class Rank(PointOperator):
    """排名"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: RankArgs = Field(default_factory=RankArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="操作数", Required=True, Variadic=True, MinNum=2, Description="参与排名操作数, 至少两个"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, ascending:bool=True, uniformization:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rank"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"ascending": ascending, "uniformization": uniformization} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: RankArgs) -> np.ndarray:
        Data = np.array(x) * (float(args.ascending) * 2 - 1)
        Rslt = np.argsort(np.argsort(Data, axis=0), axis=0)[0].astype(float)
        Rslt[pd.isnull(Data[0])] = np.nan
        if args.uniformization:
            TotalNum = np.sum(pd.notnull(Data), axis=0)
            Rslt = Rslt / TotalNum
        return Rslt

class MeanArgs(OperatorModelArgs):
    weights: Optional[List[float]] = Field(default=None, title="权重", description="各操作数的权重, None 表示等权")
    ignore_nan_weight: bool = Field(default=True, title="忽略空权重", description="空值的权重是否忽略")

class Mean(PointOperator):
    """平均值"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: MeanArgs = Field(default_factory=MeanArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="操作数", Required=True, Variadic=True, MinNum=2, Description="参与求平均的操作数, 至少两个"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, weights:Optional[List[float]]=None, ignore_nan_weight:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "mean"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"weights": weights, "ignore_nan_weight": ignore_nan_weight} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: MeanArgs) -> np.ndarray:
        Data = np.array(x)
        Weights = args.weights
        if Weights is None:
            if args.ignore_nan_weight:
                return np.nanmean(Data, axis=0)
            Weights = [1] * Data.shape[0]
        Rslt = np.zeros(Data.shape[1:])
        WeightArray = np.zeros(Data.shape[1:])
        for i, iData in enumerate(Data):
            iMask = pd.notnull(iData)
            WeightArray += iMask * Weights[i]
            iData[~iMask] = 0.0
            Rslt += iData * Weights[i]
        if args.ignore_nan_weight:
            WeightArray[WeightArray==0.0] = np.nan
            return Rslt / WeightArray
        else:
            Rslt[WeightArray==0.0] = np.nan
            return Rslt / Data.shape[0]

class StdArgs(OperatorModelArgs):
    ddof: int = Field(default=1, ge=0, title="自由度", description="标准差计算中的自由度修正")
    all_nan: float = Field(default=np.nan, title="全空填充值", description="所有操作数均为空时的填充值")

class Std(PointOperator):
    """标准差"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: StdArgs = Field(default_factory=StdArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="操作数", Required=True, Variadic=True, MinNum=2, Description="参与求标准差的操作数, 至少两个"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, ddof:int=1, all_nan:float=np.nan, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "std"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "ddof": ddof} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: StdArgs) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nanstd(Data, axis=0, ddof=args.ddof)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args.all_nan
        return Rslt

class RegressArgs(OperatorModelArgs):
    intercept: bool = Field(default=True, title="含截距", description="回归是否包含截距项")
    output: Optional[Literal["alpha", "beta"]] = Field(default=None, title="输出项", description="None 返回复合结果, 否则只返回指定项")

class Regress(PointOperator):
    """OLS 回归"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        ModelArgs: RegressArgs = Field(default_factory=RegressArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="x", Title="观测序列", Required=True, Variadic=True, MinNum=2,
                        Description="按输入顺序组成观测序列，以序号为自变量回归"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, intercept:bool=True, output:Optional[Literal["alpha", "beta"]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        if output not in ("alpha", "beta", None):
            raise __QS_Error__(f"算子 Regress 的输入参数 output 只能取值为 'alpha' 或者 'beta', 不支持 '{output}'")
        Args = {"Name": "regress"} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double"), ("beta", "double")]
        else:
            Args["DataType"] = "double"
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: RegressArgs) -> np.ndarray:
        Y = np.array(x)
        X = np.arange(Y.shape[0]).astype("float").reshape((Y.shape[0], 1, 1)).repeat(Y.shape[1], axis=1).repeat(Y.shape[2], axis=2)
        X[pd.isnull(Y)] = np.nan
        XBar, YBar = np.nanmean(X, axis=0), np.nanmean(Y, axis=0)
        Beta = np.nansum((X - XBar) * (Y - YBar), axis=0) / np.nansum((X - XBar)**2, axis=0)
        Beta[np.isinf(Beta)] = np.nan
        if args.output=="beta": return Beta
        if args.intercept:
            Alpha = YBar - Beta * XBar
        else:
            Alpha = np.zeros(shape=YBar.shape)
        if args.output=="alpha": return Alpha
        return rfn.unstructured_to_structured(np.array([Alpha, Beta]).swapaxes(0, -1)).T

    def _QS_updateArgs(self, args, updates, operand_count=None):
        args = super()._QS_updateArgs(args, updates, operand_count)
        if args["ModelArgs"].output == self.Args.ModelArgs.output:
            return args
        if "CompoundType" not in updates:
            args["CompoundType"] = ([("alpha", "double"), ("beta", "double")]
                                    if args["ModelArgs"].output is None else [])
        args["DataType"] = "object" if args["ModelArgs"].output is None else "double"
        return args

class RegressChangeRate(PointOperator):
    """以回归方式计算增长率"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="x", Title="观测序列", Required=True, Variadic=True, MinNum=2,
                        Description="按输入顺序组成观测序列，以序号为自变量回归"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "regressChangeRate"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Y = np.array(x).astype(float)
        X = np.arange(Y.shape[0]).astype("float").reshape((Y.shape[0], 1, 1)).repeat(Y.shape[1], axis=1).repeat(Y.shape[2], axis=2)
        Denominator = np.abs(np.nanmean(Y, axis=0))
        X[pd.isnull(Y)] = np.nan
        X = X - np.nanmean(X, axis=0)
        Y = Y - np.nanmean(Y, axis=0)
        Numerator = np.nansum(X * Y, axis=0) / np.nansum(X**2, axis=0)
        Rslt = Numerator / Denominator
        Mask = (Denominator==0)
        Rslt[Mask] = np.sign(Numerator)[Mask]
        Rslt[np.isinf(Rslt)] = np.nan
        return Rslt

class ToList(PointOperator):
    """将多个因子转成值为 list 的单个因子"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="x", Title="输入因子", Required=True, Variadic=True, MinNum=1, Description="待合并的因子, 至少一个"),
            OperandSpec(Name="mask", Title="掩码因子", Required=False, KeywordOnly=True, Description="值为 1 的位置保留"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "tolist"} | args | {"DataType": "object", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: OperatorModelArgs) -> np.ndarray:
        Data = {i: iData for i, iData in enumerate(x)}
        if f.Args.ModelArgs["mask"]:
            Rslt = Panel(Data, major_axis=idt, minor_axis=iid).sort_index(axis=0).to_frame(filter_observations=False)
            Rslt = Rslt[Rslt.pop(0)==1]
            return Rslt.apply(lambda s: s.tolist(), axis=1).unstack().reindex(index=idt, columns=iid).values
        else:
            return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: s.tolist(), axis=1).unstack().values

    def __call__(self, *x:Factor, mask:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PointOperation:
        self._QS_validateOperands(*x, mask=mask)
        kwargs["_qs_bound"] = True
        if not x:
            raise __QS_Error__("ToList 至少需要一个待合并的因子!")
        factor_args = factor_args | {"ModelArgs": factor_args.get("ModelArgs", {}) | {"mask": mask is not None}}
        if mask is None:
            return super().__call__(*x, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(mask, *x, factor_args=factor_args, **kwargs)

class ToCompound(PointOperator):
    """将多个因子转成单个复合因子"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="x", Title="输入因子", Required=True, Variadic=True, MinNum=1, Description="待合成的因子, 至少一个"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "toCompound"} | args | {"DataType": "object", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = {i: iData for i, iData in enumerate(x)}
        return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: tuple(s), axis=1).unstack().values

    def __call__(self, *x:Factor, fields:Optional[List[str]]=None, factor_args:dict={}, **kwargs) -> PointOperation:
        if fields is None:
            DataTypes = [(iFactor.Name, iFactor.getMetaData(key="DataType")) for iFactor in x]
        else:
            DataTypes = [(x[i].Name if not iField else iField, x[i].getMetaData(key="DataType")) for i, iField in enumerate(fields)]
        return super(ToCompound, self.new(args={"CompoundType": DataTypes})).__call__(*x, factor_args=factor_args, **kwargs)

# ----------------------时序运算--------------------------------
class LagArgs(OperatorModelArgs):
    window: int = Field(default=1, ge=0, title="窗口", description="回溯的窗口期数")
    lag_period: int = Field(default=1, ge=0, title="滞后期", description="回溯的滞后阶数")

class Lag(TimeOperator):
    """按照时间回溯数据"""

    class __QS_ArgClass__(TimeOperator.__QS_ArgClass__):
        ModelArgs: LagArgs = Field(default_factory=LagArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="f", Title="输入因子", Required=True, Description="待回溯的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, lag_period:int=1, window:Optional[int]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        if window is None: window = lag_period
        Args = {"Name": "lag", "LookBack": [window], "DataType": "double"} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window, "lag_period": lag_period} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: LagArgs) -> np.ndarray:
        Data = x[0]
        if f._QSArgs.CalcDTRuler is None: return Data[self.Args["LookBack"][0]-args.lag_period:Data.shape[0]-args.lag_period]
        TargetDTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
        Data = pd.DataFrame(Data, index=idt)
        TargetData = Data.reindex(index=TargetDTs).values.copy()
        TargetData[args.lag_period:] = TargetData[:-args.lag_period]
        if self._QSArgs.DataType!="double":
            Data = pd.DataFrame(np.empty(Data.shape, dtype="O"), index=Data.index, columns=iid)
        else:
            Data = pd.DataFrame(index=Data.index, columns=iid, dtype="float")
        Data.loc[TargetDTs] = TargetData
        return Data.ffill().values[self.Args["LookBack"][0]:]

    def __call__(self, f:Factor, factor_args:dict={}, **kwargs) -> TimeOperation:
        DataType = f.getMetaData(key="DataType")
        if DataType != self._QSArgs.DataType:
            return super(Lag, self.new(args={"DataType": DataType})).__call__(f, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, factor_args=factor_args, **kwargs)

    def _QS_updateArgs(self, args, updates, operand_count=None):
        args = super()._QS_updateArgs(args, updates, operand_count)
        if "LookBack" not in updates and args["ModelArgs"].window != self.Args.ModelArgs.window:
            args["LookBack"] = [args["ModelArgs"].window]
        return args

class RollingRankArgs(OperatorModelArgs):
    window: int = Field(default=1, ge=1, title="窗口", description="滚动窗口期数")
    min_periods: int = Field(default=1, ge=1, title="最少期数", description="窗口内最少非空期数")
    ascending: bool = Field(default=True, title="升序", description="是否按升序排名")
    uniformization: bool = Field(default=True, title="归一化", description="排名是否除以总数归一化")

class RollingRank(TimeOperator):
    """滚动排名"""

    class __QS_ArgClass__(TimeOperator.__QS_ArgClass__):
        ModelArgs: RollingRankArgs = Field(default_factory=RollingRankArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待滚动排名的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, window:int=1, min_periods:int=1, ascending:bool=True, uniformization:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rollingRank", "LookBack": [window - 1]} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window, "min_periods": min_periods, "ascending": ascending, "uniformization": uniformization} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: RollingRankArgs) -> np.ndarray:
        Args = args.model_dump()
        Data = pd.DataFrame(x[0])
        if not Args.pop("ascending"):
            Data = - Data
        Uniformization = Args.pop("uniformization")
        Rslt = Data.rolling(**Args).rank().values[self.Args["LookBack"][0]:] - 1
        if Uniformization:
            Rslt = Rslt / Data.rolling(**Args).count().values[self.Args["LookBack"][0]:]
        return Rslt

class RollingMeanArgs(OperatorModelArgs):
    window: int = Field(default=1, ge=1, title="窗口", description="滚动窗口期数")
    min_periods: int = Field(default=1, ge=1, title="最少期数", description="窗口内最少非空期数")
    weights: Optional[List[float]] = Field(default=None, title="权重", description="各期权重, None 表示等权")

class RollingMean(TimeOperator):
    """滚动平均"""

    class __QS_ArgClass__(TimeOperator.__QS_ArgClass__):
        ModelArgs: RollingMeanArgs = Field(default_factory=RollingMeanArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待滚动平均的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, window:int=1, min_periods:int=1, weights:Optional[List[float]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        if weights is not None: window = len(weights)
        Args = {"Name": "rollingMean", "LookBack": [window - 1]} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] =  {"window": window, "min_periods": min_periods, "weights": weights} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: RollingMeanArgs) -> np.ndarray:
        Data = pd.DataFrame(x[0])
        Args = {"window": args.window, "min_periods": args.min_periods}
        weights = args.weights
        if not weights:
            return Data.rolling(**Args).mean().values[self.Args["LookBack"][0]:]
        else:
            weights = np.array(weights)
            return Data.rolling(**Args).apply(lambda x: np.nansum(x * weights) / np.nansum(pd.notnull(x) * weights), raw=True).values[self.Args["LookBack"][0]:]

class RollingApplyArgs(OperatorModelArgs):
    func: Callable[[np.ndarray], Any] = Field(default=np.nansum, title="运算函数", description="施加到滚动窗口上的函数")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="函数返回值的数据类型")
    window: int = Field(default=1, ge=1, title="窗口", description="滚动窗口期数")
    min_periods: int = Field(default=1, ge=1, title="最少期数", description="窗口内最少非空期数")

class RollingApply(TimeOperator):
    """滚动操作"""

    class __QS_ArgClass__(TimeOperator.__QS_ArgClass__):
        ModelArgs: RollingApplyArgs = Field(default_factory=RollingApplyArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待滚动运算的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, func:Callable[[np.ndarray], Any]=np.nansum, dtype:Literal["double", "string", "object"]="double", window:int=1, min_periods:int=1, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rollingApply", "LookBack": [window - 1], "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype, "window": window, "min_periods": min_periods} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: RollingApplyArgs) -> np.ndarray:
        func, dtype = args.func, args.dtype
        Data = np.lib.stride_tricks.sliding_window_view(x[0], window_shape=args.window, axis=0)
        Mask = (np.sum(~ np.isnan(Data), axis=-1) < args.min_periods)
        Data = np.apply_along_axis(func, axis=-1, arr=Data)
        if dtype=="double":
            Data = Data.astype(float)
            Data[Mask] = np.nan
        else:
            Data = Data.astype("O")
            Data[Mask] = None
        return Data

class RollingChangeRateArgs(OperatorModelArgs):
    window: int = Field(default=1, ge=1, title="窗口", description="滚动窗口期数")

class RollingChangeRate(TimeOperator):
    """滚动增长率"""

    class __QS_ArgClass__(TimeOperator.__QS_ArgClass__):
        ModelArgs: RollingChangeRateArgs = Field(default_factory=RollingChangeRateArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="x", Title="输入因子", Required=True, Description="待计算增长率的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, window:int=1, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rollingChangeRate", "LookBack": [window - 1]} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: RollingChangeRateArgs) -> np.ndarray:
        Data = x[0]
        Numerator = Data[args.window-1:]
        Denominator = Data[:-args.window+1]
        Rslt = (Numerator - Denominator) / np.abs(Denominator)
        Mask = (Denominator==0)
        Rslt[Mask] = np.nan
        Rslt[Mask & (Numerator>0)] = 1.0
        Rslt[Mask & (Numerator<0)] = -1.0
        Rslt[Mask & (Numerator==0)] = 0.0
        return Rslt[self.Args["LookBack"][0]-args.window+1:]

class RollingRegressArgs(OperatorModelArgs):
    window: int = Field(default=1, ge=1, title="窗口", description="滚动窗口期数")
    min_periods: int = Field(default=1, ge=1, title="最少期数", description="窗口内最少非空期数")
    intercept: bool = Field(default=True, title="含截距", description="回归是否包含截距项")
    output: Optional[str] = Field(default=None, pattern=r"^(alpha|r2|beta\d+)$", title="输出项", description="None 返回复合结果, 否则返回 alpha/r2/beta{i}")

class RollingRegress(TimeOperator):
    """滚动回归"""

    class __QS_ArgClass__(TimeOperator.__QS_ArgClass__):
        ModelArgs: RollingRegressArgs = Field(default_factory=RollingRegressArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="endog", Title="被解释变量", Required=True, Description="回归的因变量"),
            OperandSpec(Name="exog", Title="解释变量", Required=False, Variadic=True, MinNum=0, Description="回归的自变量, 可多个"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, window:int=1, min_periods:int=1, intercept:bool=True, output:Optional[str]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        operand_count = 1
        Args = {"Name": "rollingRegress"} | args | {"DataType": "double", "DTMode": "单时点", "IDMode": "单ID"}
        Args["ModelArgs"] = {"window": window, "min_periods": min_periods, "intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        Args["LookBack"] = [Args["ModelArgs"]["window"] - 1] * operand_count
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(max(1, operand_count - 1))] + [("r2", "double")]
        else:
            Args["DataType"] = "double"
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: str, x: List[np.ndarray], args: RollingRegressArgs) -> Union[float, Tuple[float]]:
        Y, X = x[0].astype(float), (np.array(x[1:], dtype=float).T if len(x)>1 else np.arange(0, x[0].shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        if np.sum(Mask) < args.min_periods: return (np.nan if args.output is not None else (np.nan,) * (2 + X.shape[1]))
        Y, X = Y[Mask], X[Mask]
        if args.intercept: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        if args.output is None: return (tuple(Rslt.params) if args.intercept else ((0, ) + tuple(Rslt.params))) + (Rslt.rsquared,)
        elif args.output=="alpha": return Rslt.params[0] if args.intercept else 0
        elif args.output=="r2": return Rslt.rsquared
        else: return Rslt.params[int(args.output[4:]) + int(args.intercept)]

    def __call__(self, endog:Factor, *exog:Factor, factor_args:dict={}, **kwargs) -> TimeOperation:
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)

    def _QS_updateArgs(self, args, updates, operand_count=None):
        args = super()._QS_updateArgs(args, updates, operand_count)
        if (operand_count is None or operand_count == getattr(self, "_OperandCount", None)) and args["ModelArgs"].output == self.Args.ModelArgs.output:
            return args
        if "CompoundType" not in updates:
            arity = operand_count or getattr(self, "_OperandCount", None) or 1
            args["CompoundType"] = ([("alpha", "double")] +
                [(f"beta{i}", "double") for i in range(max(1, arity - 1))] + [("r2", "double")]
                if args["ModelArgs"].output is None else [])
        args["DataType"] = "object" if args["ModelArgs"].output is None else "double"
        return args

# ----------------------截面运算--------------------------------
class SectionRankArgs(OperatorModelArgs):
    ascending: bool = Field(default=True, title="升序", description="是否按升序排名")
    uniformization: bool = Field(default=True, title="归一化", description="排名是否除以总数归一化")

class SectionRank(SectionOperator):
    """截面排名"""

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: SectionRankArgs = Field(default_factory=SectionRankArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="f", Title="待排名因子", Required=True, Description="待排名的因子"),
            OperandSpec(Name="mask", Title="掩码因子", Required=False, Description="掩码因子, 值为 1 的位置参与排名"),
            OperandSpec(Name="cat_data", Title="分组因子", Required=False, Description="分组因子, 按其取值分组分别排名"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, ascending:bool=True, uniformization:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        operand_count = 1
        Args = {"Name": "rankSection"} | args | {"DataType": "double", "DTMode": "多时点"}
        Args["ModelArgs"] = {"uniformization": uniformization, "ascending": ascending} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [None])[0]
        Args["DescriptorSection"] = [descriptor_ids] * operand_count
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        FactorData = x[0]
        Mask = (x[1].astype(bool) if f._QSArgs.ModelArgs["mask"] else [None] * FactorData.shape[0])
        CatData = (x[-1] if f._QSArgs.ModelArgs["cat_data"] else [None] * FactorData.shape[0])
        Rslt = np.full_like(FactorData, fill_value=np.nan)
        for i in range(FactorData.shape[0]):
            Rslt[i] = DataPreprocessingFun.standardizeRank(FactorData[i], mask=Mask[i], cat_data=CatData[i], perturbation=False, offset=0, **args)
        return Rslt

    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> SectionOperation:
        self._QS_validateOperands(f, mask, cat_data)
        kwargs["_qs_bound"] = True
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class AggregateArgs(OperatorModelArgs):
    aggr_func: Callable[[np.ndarray], Any] = Field(default=np.nansum, title="聚合函数", description="接收截面数据的聚合函数")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="聚合结果的数据类型")

class Aggregate(SectionOperator):
    """截面聚合

    对单个时点的截面数据进行聚合, 结果广播到所有 ID。
    支持通过 mask 掩码过滤数据, 通过 cat_data 按类别分组聚合。

    Args:
        aggr_func: 聚合函数, 接收 1D ndarray (截面值), 返回标量。
            当带 mask 时仅包含掩码为 1 的值; 当带 cat_data 时包含当前类别的截面值。
        descriptor_ids: 聚合的截面 ID 列表, None 表示使用因子自身截面
        dtype: 输出数据类型
    """

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: AggregateArgs = Field(default_factory=AggregateArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="f", Title="待聚合因子", Required=True, Description="待聚合的因子"),
            OperandSpec(Name="mask", Title="掩码因子", Required=False, Description="掩码因子, 值为 1 的位置参与聚合"),
            OperandSpec(Name="cat_data", Title="分组因子", Required=False, Description="分组因子, 按其取值分组分别聚合"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, aggr_func:Callable[[np.ndarray], Any]=np.nansum, descriptor_ids:Optional[List[str]]=None, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        operand_count = 1
        Args = {"Name": "aggregate"} | args | {"DataType": dtype, "DTMode": "单时点"}
        Args["ModelArgs"] = {"aggr_func": aggr_func, "dtype": dtype} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [descriptor_ids])[0]
        Args["DescriptorSection"] = [descriptor_ids] * operand_count
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: dt.datetime, iid: List[str], x: List[np.ndarray], args: AggregateArgs) -> np.ndarray:
        nID = len(iid)
        FactorData = x[0]
        if f._QSArgs.ModelArgs["mask"]:
            Mask = (x[1]==1)
        else:
            Mask = np.full(FactorData.shape, fill_value=True)
        AggrFunc = args.aggr_func
        if f._QSArgs.ModelArgs["cat_data"]:
            CatData = x[-1]
            Rslt = np.full(shape=(nID, ), fill_value=np.nan)
            if f._QSArgs.ModelArgs["section_chged"]:
                for i, iID in enumerate(iid):
                    iMask = ((CatData==iID) & Mask)
                    Rslt[i] = AggrFunc(FactorData[iMask])
            else:
                AllCats = pd.unique(CatData.flatten())
                for i, iCat in enumerate(AllCats):
                    if pd.isnull(iCat):
                        iMask = (pd.isnull(CatData) & Mask)
                    else:
                        iMask = ((CatData==iCat) & Mask)
                    Rslt[iMask] = AggrFunc(FactorData[iMask])
        else:
            Rslt = np.full(shape=(nID, ), fill_value=AggrFunc(FactorData[Mask]))
        return Rslt

    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> SectionOperation:
        self._QS_validateOperands(f, mask, cat_data)
        kwargs["_qs_bound"] = True
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None), "section_chged": (self._QSArgs.DescriptorSection[0] is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class Disaggregate(SectionOperator):
    """截面反聚合"""

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="f", Title="待反聚合因子", Required=True, Description="待反聚合的因子"),
            OperandSpec(Name="cat_data", Title="分组因子", Required=False, Description="分组因子, 按聚合 ID 反聚合"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, aggr_ids:List[str], disaggr_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "disaggregate", "DataType": "double"} | args | {"DTMode": "多时点"}
        DescriptorSection = Args.get("DescriptorSection", [aggr_ids, disaggr_ids])
        Args["DescriptorSection"] = DescriptorSection
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        nDT, nID = len(idt), len(iid)
        FactorData = x[0]
        if f._QSArgs.ModelArgs["cat_data"]:
            CatData = x[-1]
            Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
            for i, iID in enumerate(self.Args.DescriptorSection[0]):
                iMask = (CatData==iID)
                Rslt[iMask] = FactorData[:, [i]].repeat(nID, axis=1)[iMask]
        else:
            Rslt = FactorData.repeat(nID, axis=1)
        return Rslt

    def __call__(self, f:Factor, cat_data:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> SectionOperation:
        self._QS_validateOperands(f, cat_data)
        kwargs["_qs_bound"] = True
        Factors = [f]
        if cat_data is not None: Factors.append(cat_data)
        kwargs["operator_kwargs"] =  {"aggr_ids": self._QSArgs.DescriptorSection[0]} | kwargs.get("operator_kwargs", {})
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"cat_data": (cat_data is not None)}
        DataType = f.getMetaData(key="DataType")
        if DataType != self._QSArgs.DataType:
            return super(Disaggregate, self.new(args={"DataType": DataType}, **kwargs["operator_kwargs"])).__call__(*Factors, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class ConcatSectionArgs(OperatorModelArgs):
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="拼接结果的数据类型")

class ConcatSection(SectionOperator):
    """截面拼接"""

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: ConcatSectionArgs = Field(default_factory=ConcatSectionArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="x", Title="输入因子", Required=True, Variadic=True, MinNum=1, Description="待拼接的因子, 至少一个"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, descriptor_sections:List[List[str]]=[], dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        operand_count = max(1, len(descriptor_sections))
        Args = {"Name": "concatSection", "DataType": dtype} | args | {"DTMode": "多时点"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        DescriptorSection = Args.get("DescriptorSection", descriptor_sections)
        Args["DescriptorSection"] = DescriptorSection[:operand_count] + [None] * max(0, operand_count - len(DescriptorSection))
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: ConcatSectionArgs) -> np.ndarray:
        return pd.DataFrame(np.concatenate(x, axis=1), columns=sum(((iid if iIDs is None else iIDs) for iIDs in self.Args.DescriptorSection), [])).reindex(columns=iid).values

class ChgSectionArgs(OperatorModelArgs):
    id_map: Dict[str, str] = Field(default={}, title="ID 映射", description="{新ID: 旧ID} 的映射关系")

class ChgSection(SectionOperator):
    """修改截面"""

    # id_map: {新ID: 旧ID}
    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: ChgSectionArgs = Field(default_factory=ChgSectionArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(OperandSpec(Name="f", Title="输入因子", Required=True, Description="待修改截面的因子"),), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, old_ids:List[str], id_map:Dict[str, str]={}, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "chgSection", "DataType": "double"} | args | {"DTMode": "多时点"}
        Args["ModelArgs"] = {"id_map": id_map} | Args.get("ModelArgs", {})
        if "DescriptorSection" not in Args: Args["DescriptorSection"] = [old_ids]
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: ChgSectionArgs) -> np.ndarray:
        Data = x[0]
        IDMap = args.id_map
        OldIDs = self._QSArgs.DescriptorSection[0]
        Rslt = np.full(shape=(len(idt), len(iid)), fill_value=np.nan, dtype=Data.dtype)
        for i, iID in enumerate(iid):
            iOldID = IDMap.get(iID, None)
            if iOldID not in OldIDs: continue
            Rslt[:, i] = Data[:, OldIDs.index(iOldID)]
        return Rslt

    def __call__(self, f:Factor, factor_args:dict={}, **kwargs) -> SectionOperation:
        kwargs["operator_kwargs"] =  {"old_ids": self._QSArgs.DescriptorSection[0]} | kwargs.get("operator_kwargs", {})
        DataType = f.getMetaData(key="DataType")
        if DataType != self._QSArgs.DataType:
            return super(ChgSection, self.new(args={"DataType": DataType}, **kwargs["operator_kwargs"])).__call__(f, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, factor_args=factor_args, **kwargs)

class SectionRegressArgs(OperatorModelArgs):
    intercept: bool = Field(default=True, title="含截距", description="回归是否包含截距项")
    output: Optional[str] = Field(default=None, title="输出项", description="None 返回复合结果, 否则只返回指定项")

class SectionRegress(SectionOperator):
    """截面回归"""

    # output: alpha, beta{i}, resid
    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: SectionRegressArgs = Field(default_factory=SectionRegressArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="endog", Title="被解释变量", Required=True, Description="回归的因变量"),
            OperandSpec(Name="exog", Title="解释变量", Required=False, Variadic=True, MinNum=0, Description="回归的自变量, 可多个"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, intercept:bool=True, output:Optional[str]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        operand_count = 1
        Args = {"Name": "regressSection"} | args | {"DTMode": "单时点"}
        Args["ModelArgs"] = {"intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [None])[0]
        Args["DescriptorSection"] = [descriptor_ids] * operand_count
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(max(1, operand_count - 1))] + [("resid", "double")]
        else:
            Args["DataType"] = "double"
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: dt.datetime, iid: List[str], x: List[np.ndarray], args: SectionRegressArgs) -> np.ndarray:
        Y, X = x[0].astype(float), (np.array(x[1:], dtype=float).T if len(x)>1 else np.arange(0, x[0].shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        Y, X = Y[Mask], X[Mask]
        if args.intercept: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        Beta = (tuple(Rslt.params) if args.intercept else (0, ) + tuple(Rslt.params))
        Resid = np.full(shape=Mask.shape, fill_value=np.nan)
        Resid[Mask] = Rslt.resid
        if args.output is None: return [Beta+(Resid[i], ) for i in range(len(iid))]
        elif args.output=="alpha": Rslt = Beta[0]
        elif args.output == "resid": return Resid
        else: Rslt = Beta[int(args.output[4:]) + 1]
        return np.full(shape=(len(iid),), fill_value=Rslt)

    def __call__(self, endog:Factor, *exog:Factor, factor_args:dict={}, **kwargs) -> SectionOperation:
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)

    def _QS_updateArgs(self, args, updates, operand_count=None):
        args = super()._QS_updateArgs(args, updates, operand_count)
        if (operand_count is None or operand_count == getattr(self, "_OperandCount", None)) and args["ModelArgs"].output == self.Args.ModelArgs.output:
            return args
        if "CompoundType" not in updates:
            arity = operand_count or getattr(self, "_OperandCount", None) or 1
            args["CompoundType"] = ([("alpha", "double")] +
                [(f"beta{i}", "double") for i in range(max(1, arity - 1))] + [("resid", "double")]
                if args["ModelArgs"].output is None else [])
        args["DataType"] = "object" if args["ModelArgs"].output is None else "double"
        return args

class AggregateComponentArgs(OperatorModelArgs):
    aggr_func: Callable[[np.ndarray], Any] = Field(default=np.nanmean, title="聚合函数", description="接收 2D ndarray 的聚合函数")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="聚合结果的数据类型")
    include_component_code: bool = Field(default=False, title="含成分代码", description="传给聚合函数的数据是否包含成分代码列")

class AggregateComponent(SectionOperator):
    """聚合成分数据

    按成分列表聚合。component 因子的每个 ID 对应一个成分 ID 列表, 算子从
    component_data 和 exog 中取值, 展开成分后按 code 分组聚合。

    Args:
        aggr_func: 聚合函数, 接收 2D ndarray。
            如果 include_component_code 为 False, 则 shape=(n_components, nEndog+nExog)，列顺序为 [component_data_0, ..., exog_0, ...], 行为当前 code 下的各成分。
            如果 include_component_code 为 True, 则 shape=(n_components, 1+nEndog+nExog)，列顺序为 [component_code, component_data_0, ..., exog_0, ...], 行为当前 code 下的各成分。
        descriptor_ids: exog 数据的截面 ID 列表, None 表示无 exog
        dtype: 输出数据类型
    """

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        ModelArgs: AggregateComponentArgs = Field(default_factory=AggregateComponentArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="component", Title="成分因子", Required=True, Description="每个 ID 对应一个成分 ID 列表的因子"),
            OperandSpec(Name="component_data", Title="成分数据", Required=False, Variadic=True, Grouped=True, MinNum=0, Description="成分对应的数据因子, 可多个"),
            OperandSpec(Name="exog", Title="外生数据", Required=False, Variadic=True, Grouped=True, MinNum=0, Description="外生数据因子, 可多个"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, aggr_func:Callable[[np.ndarray], Any]=np.nanmean, include_component_code:bool=False, descriptor_ids:Optional[List[str]]=None, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        ModelArgs = args.get("ModelArgs", {})
        Args = {"Name": "aggregateComponent"} | args | {"DataType": dtype, "DTMode": "单时点"}
        Args["ModelArgs"] = {"aggr_func": aggr_func, "dtype": dtype, "include_component_code": include_component_code} | ModelArgs
        Args.setdefault("DescriptorSection", [descriptor_ids] * (1))
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: dt.datetime, iid: List[str], x: List[np.ndarray], args: AggregateComponentArgs) -> np.ndarray:
        Data = x[0:1+f.Args.ModelArgs["endog_num"]]
        Data = pd.DataFrame(Data, columns=iid).T
        Data = expandListElementDataFrame(Data, expand_index=True, dropna=True)
        Data.columns = ["code", "component_code"] + Data.columns[2:].tolist()
        if Data.empty:
            return np.full((len(iid), ), np.nan, dtype=float) if args.dtype=="double" else np.full_like((len(iid), ), None, dtype="O")
        Exog = x[1+f.Args.ModelArgs["endog_num"]:1+f.Args.ModelArgs["endog_num"]+f.Args.ModelArgs["exog_num"]]
        if Exog:
            Exog = pd.DataFrame(Exog, columns=f.Operator.Args.DescriptorSection[-1]).T
            Data = pd.merge(Data, Exog, how="left", left_on=["component_code"], right_index=True)
        AggrFunc = args.aggr_func
        if args.include_component_code:
            def _aggr_func(df):
                return AggrFunc(df.values)
        else:
            def _aggr_func(df):
                return AggrFunc(df.iloc[:, 1:].values)
        Rslt = Data.groupby(["code"]).apply(_aggr_func, include_groups=False)
        return Rslt.reindex(index=iid).values

    def __call__(self, component:Factor, component_data:List[Factor]=[], exog:List[Factor]=[], factor_args:dict={}, **kwargs) -> SectionOperation:
        self._QS_validateOperands(component, component_data, exog)
        kwargs["_qs_bound"] = True
        operator_kwargs = dict(kwargs.pop("operator_kwargs", {}))
        descriptor_ids = operator_kwargs.pop("descriptor_ids", self.Args.DescriptorSection[-1])
        args = operator_kwargs.pop("args", {}) | {
            "DescriptorSection": [None] * (1 + len(component_data)) + [descriptor_ids] * len(exog),
        }
        factor_args = factor_args | {"ModelArgs": factor_args.get("ModelArgs", {}) | {
            "endog_num": len(component_data), "exog_num": len(exog),
        }}
        return super(AggregateComponent, self.new(args=args, **operator_kwargs)).__call__(
            component, *component_data, *exog, factor_args=factor_args, **kwargs)

# ----------------------面板运算--------------------------------
class PanelRegressArgs(OperatorModelArgs):
    window: int = Field(default=1, ge=1, title="窗口", description="面板窗口期数")
    intercept: bool = Field(default=True, title="含截距", description="回归是否包含截距项")
    output: Optional[str] = Field(default=None, title="输出项", description="None 返回复合结果, 否则只返回指定项")

class PanelRegress(PanelOperator):
    """面板回归"""

    # output: alpha, beta{i}, resid
    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        ModelArgs: PanelRegressArgs = Field(default_factory=PanelRegressArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="endog", Title="被解释变量", Required=True, Description="回归的因变量"),
            OperandSpec(Name="exog", Title="解释变量", Required=False, Variadic=True, MinNum=0, Description="回归的自变量, 可多个"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, window:int=1, intercept:bool=True, output:Optional[str]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        operand_count = 1
        Args = {"Name": "regressPanel"} | args | {"DTMode": "单时点"}
        Args["ModelArgs"] = {"window": window, "intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [None])[0]
        Args["DescriptorSection"] = [descriptor_ids] * operand_count
        Args["LookBack"] = [Args["ModelArgs"]["window"] - 1] * operand_count
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(max(1, operand_count - 1))] + [("resid", "double")]
        else:
            Args["DataType"] = "double"
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: PanelRegressArgs) -> np.ndarray:
        Y = x[0].astype(float).flatten()
        X = (np.array(x[1:], dtype=float).reshape((len(x)-1, -1)).T if len(x)>1 else np.arange(0, Y.shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        Y, X = Y[Mask], X[Mask]
        if args.intercept: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        Beta = (tuple(Rslt.params) if args.intercept else (0, )+tuple(Rslt.params))
        Resid = np.full(shape=Mask.shape, fill_value=np.nan)
        Resid[Mask] = Rslt.resid
        Resid = Resid[-len(iid):]
        if args.output is None: return [Beta+(Resid[i], ) for i in range(len(iid))]
        elif args.output=="alpha": Rslt = Beta[0]
        elif args.output == "resid": return Resid
        else: Rslt = Beta[int(args.output[4:]) + 1]
        return np.full(shape=(len(iid),), fill_value=Rslt)

    def __call__(self, endog:Factor, *exog:Factor, factor_args:dict={}, **kwargs) -> PanelOperation:
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)

    def _QS_updateArgs(self, args, updates, operand_count=None):
        args = super()._QS_updateArgs(args, updates, operand_count)
        if (operand_count is None or operand_count == getattr(self, "_OperandCount", None)) and args["ModelArgs"].output == self.Args.ModelArgs.output:
            return args
        if "CompoundType" not in updates:
            arity = operand_count or getattr(self, "_OperandCount", None) or 1
            args["CompoundType"] = ([("alpha", "double")] +
                [(f"beta{i}", "double") for i in range(max(1, arity - 1))] + [("resid", "double")]
                if args["ModelArgs"].output is None else [])
        args["DataType"] = "object" if args["ModelArgs"].output is None else "double"
        return args

class AggregatePanelArgs(OperatorModelArgs):
    aggr_func: Callable[[np.ndarray], Any] = Field(default=np.nansum, title="聚合函数", description="接收 2D ndarray 的聚合函数")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="聚合结果的数据类型")
    window: int = Field(default=1, ge=1, title="窗口", description="回溯窗口大小, 1 表示仅当期")

class AggregatePanel(PanelOperator):
    """截面聚合时序数据

    对 window 期内的截面数据进行聚合, 结果广播到所有 ID。
    与 Aggregate 的区别在于支持时间窗口回溯, aggr_func 接收整个窗口的数据。
    支持通过 mask 掩码过滤, 通过 cat_data 按类别分组聚合。

    Args:
        aggr_func: 聚合函数, 接收 2D ndarray, shape=(window, nID_filtered)。
            当无 mask/cat_data 时为 (window, nID);
            当带 mask 时列维度为掩码为 1 的 ID;
            当带 cat_data 时为当前类别的截面值。
        window: 回溯窗口大小, 1 表示仅当期
        descriptor_ids: 聚合的截面 ID 列表, None 表示使用因子自身截面
        dtype: 输出数据类型
    """

    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        ModelArgs: AggregatePanelArgs = Field(default_factory=AggregatePanelArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="f", Title="待聚合因子", Required=True, Description="待聚合的因子"),
            OperandSpec(Name="mask", Title="掩码因子", Required=False, Description="掩码因子, 值为 1 的位置参与聚合"),
            OperandSpec(Name="cat_data", Title="分组因子", Required=False, Description="分组因子, 按其取值分组分别聚合"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, aggr_func:Callable[[np.ndarray], Any]=np.nansum, window:int=1, descriptor_ids:Optional[List[str]]=None, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        operand_count = 1
        Args = {"Name": "aggregatePanel"} | args | {"DataType": dtype, "DTMode": "单时点"}
        Args["ModelArgs"] = {"aggr_func": aggr_func, "dtype": dtype, "window": window} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [descriptor_ids])[0]
        Args["DescriptorSection"] = [descriptor_ids] * operand_count
        Args["LookBack"] = [Args["ModelArgs"]["window"] - 1] + [0] * (operand_count - 1)
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: dt.datetime, iid: List[str], x: List[np.ndarray], args: AggregatePanelArgs) -> np.ndarray:
        nID = len(iid)
        FactorData = x[0]
        if f._QSArgs.ModelArgs["mask"]:
            Mask = (x[1][0]==1)
        else:
            Mask = np.full((FactorData.shape[1], ), fill_value=True)
        AggrFunc = args.aggr_func
        if f._QSArgs.ModelArgs["cat_data"]:
            CatData = x[-1][0]
            Rslt = np.full(shape=(nID, ), fill_value=np.nan)
            if f._QSArgs.ModelArgs["section_chged"]:
                for i, iID in enumerate(iid):
                    iMask = ((CatData==iID) & Mask)
                    Rslt[i] = AggrFunc(FactorData[:, iMask])
            else:
                AllCats = pd.unique(CatData.flatten())
                for i, iCat in enumerate(AllCats):
                    if pd.isnull(iCat):
                        iMask = (pd.isnull(CatData) & Mask)
                    else:
                        iMask = ((CatData==iCat) & Mask)
                    Rslt[iMask] = AggrFunc(FactorData[:, iMask])
        else:
            Rslt = np.full(shape=(nID, ), fill_value=AggrFunc(FactorData[:, Mask]))
        return Rslt

    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        self._QS_validateOperands(f, mask, cat_data)
        kwargs["_qs_bound"] = True
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None), "section_chged": (self._QSArgs.DescriptorSection[0] is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class AggregateComponentPanelArgs(OperatorModelArgs):
    aggr_func: Callable[[np.ndarray], Any] = Field(default=np.nanmean, title="聚合函数", description="按 code 分组调用的聚合函数")
    dtype: Literal["double", "string", "object"] = Field(default="double", title="数据类型", description="聚合结果的数据类型")
    window: int = Field(default=1, ge=1, title="窗口", description="exog 的回溯窗口大小, 1 表示仅当期")

class AggregateComponentPanel(PanelOperator):
    """聚合成分时序数据

    AggregateComponent 的面板版本, exog 支持时间窗口回溯。
    component 和 component_data (endog) 只取当期数据, exog 回溯 window 期。
    无 exog 时等价于 AggregateComponent (但 aggr_func 接收 DataFrame)。

    Args:
        aggr_func: 聚合函数, 接收 DataFrame, 按 code 分组后调用。
            无 exog 时列: [component_code, endog_0, ...]
            有 exog 时列: [component_code, datetime, endog_0, ..., exog_0, ...]
            endog 数据在时间维度上重复, exog 数据按 datetime 对应不同时点的值。
            由 aggr_func 自行决定如何处理时间维度。
        window: exog 的回溯窗口大小, 1 表示仅当期
        descriptor_ids: exog 数据的截面 ID 列表, None 表示无 exog
        dtype: 输出数据类型
    """

    class __QS_ArgClass__(PanelOperator.__QS_ArgClass__):
        ModelArgs: AggregateComponentPanelArgs = Field(default_factory=AggregateComponentPanelArgs, title="模型参数")
        OperandInfo: Tuple[OperandSpec, ...] = Field(default=(
            OperandSpec(Name="component", Title="成分因子", Required=True, Description="每个 ID 对应一个成分 ID 列表的因子"),
            OperandSpec(Name="component_data", Title="成分数据", Required=False, Variadic=True, Grouped=True, MinNum=0, Description="成分对应的数据因子, 可多个"),
            OperandSpec(Name="exog", Title="外生数据", Required=False, Variadic=True, Grouped=True, MinNum=0, Description="外生数据因子, 可多个"),
        ), title="操作数声明", frozen=True, exclude=True)

    def __init__(self, aggr_func:Callable[[np.ndarray], Any]=np.nanmean, window:int=1, descriptor_ids:Optional[List[str]]=None, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        ModelArgs = args.get("ModelArgs", {})
        Args = {"Name": "aggregateComponentPanel"} | args | {"DataType": dtype, "DTMode": "单时点"}
        Args["ModelArgs"] = {"aggr_func": aggr_func, "dtype": dtype, "window": window} | ModelArgs
        Args.setdefault("DescriptorSection", [descriptor_ids] * (1))
        Args.setdefault("LookBack", [Args["ModelArgs"]["window"] - 1] * (1))
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: AggregateComponentPanelArgs) -> np.ndarray:
        nID = len(iid)
        AggrFunc = args.aggr_func
        EndogNum = f.Args.ModelArgs["endog_num"]
        ExogNum = f.Args.ModelArgs["exog_num"]
        DesIDs = f.Operator.Args.DescriptorSection[-1]
        Exog = x[1+EndogNum:1+EndogNum+ExogNum]
        ComponentData = x[0][0]
        EndogData = [x[1+e][0] for e in range(EndogNum)]
        BaseData = pd.DataFrame([ComponentData] + EndogData, columns=iid).T
        BaseData = expandListElementDataFrame(BaseData, expand_index=True, dropna=True)
        BaseData.columns = ["code", "component_code"] + [f"endog_{i}" for i in range(EndogNum)]
        if BaseData.empty:
            return np.full((nID, ), np.nan, dtype=float) if args.dtype=="double" else np.full((nID, ), None, dtype="O")
        if not Exog:
            def _aggr_func(df):
                return AggrFunc(df[["component_code"] + [f"endog_{i}" for i in range(EndogNum)]])
            Rslt = BaseData.groupby(["code"]).apply(_aggr_func)
            return Rslt.reindex(index=iid).values
        nRow = len(BaseData)
        nWindow = Exog[0].shape[0]
        # 展开: 每条成分 × 每个时点
        PanelData = BaseData.iloc[np.repeat(np.arange(nRow), nWindow)].reset_index(drop=True)
        PanelData.insert(1, "datetime", np.tile(idt, nRow))
        # exog 长表: 每个 exog shape (nWindow, len(DesIDs)), 展平为 (nWindow*len(DesIDs), ...)
        ExogTable = pd.DataFrame({
            "component_code": np.tile(DesIDs, nWindow),
            "datetime": np.repeat(idt, len(DesIDs)),
        } | {f"exog_{i}": Exog[i].ravel() for i in range(ExogNum)})
        PanelData = pd.merge(PanelData, ExogTable, on=["component_code", "datetime"], how="left")
        def _aggr_func(df):
            return AggrFunc(df[["component_code", "datetime"] + [f"endog_{i}" for i in range(EndogNum)] + [f"exog_{i}" for i in range(ExogNum)]])
        Rslt = PanelData.groupby(["code"]).apply(_aggr_func)
        return Rslt.reindex(index=iid).values

    def __call__(self, component:Factor, component_data:List[Factor]=[], exog:List[Factor]=[], factor_args:dict={}, **kwargs) -> PanelOperation:
        self._QS_validateOperands(component, component_data, exog)
        kwargs["_qs_bound"] = True
        operator_kwargs = dict(kwargs.pop("operator_kwargs", {}))
        descriptor_ids = operator_kwargs.pop("descriptor_ids", self.Args.DescriptorSection[-1])
        operand_count = 1 + len(component_data) + len(exog)
        window = operator_kwargs.get("window", self.Args.ModelArgs.window)
        model = operator_kwargs.get("args", {}).get("ModelArgs", {})
        window = model.get("window", window)
        args = operator_kwargs.pop("args", {}) | {
            "DescriptorSection": [None] * (1 + len(component_data)) + [descriptor_ids] * len(exog),
            "LookBack": [0] * (1 + len(component_data)) + [window - 1] * len(exog),
            "StartDT": [None] * operand_count,
        }
        factor_args = factor_args | {"ModelArgs": factor_args.get("ModelArgs", {}) | {
            "endog_num": len(component_data), "exog_num": len(exog),
        }}
        return super(AggregateComponentPanel, self.new(args=args, **operator_kwargs)).__call__(
            component, *component_data, *exog, factor_args=factor_args, **kwargs)


if __name__=="__main__":
    from functools import partial
    from QuantStudio.Factor.Factor import DataFactor
    
    np.random.seed(0)
    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Factor1 = DataFactor(name="Factor1", data=1)
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    
    qs_sum = Sum(all_nan="", dtype="string")
    rolling_std = RollingApply(func=partial(np.nanstd, ddof=1), window=3, min_periods=3)
    rank_section = SectionRank(uniformization=False)
    aggr_sum = Aggregate(aggr_func=np.nanprod, descriptor_ids=IDs)
    rolling_regress = RollingRegress(window=2)
    disaggr = Disaggregate(aggr_ids=["000000.HST"])
    chg_section = ChgSection(old_ids=IDs, args={"DataType": "string"})
    
    Factor3 = Log(base=np.e)(Factor2, factor_name="Factor3")
    Factor4 = RollingApply(func=np.nansum, window=2, min_periods=2)(Factor2, factor_name="Factor4")
    Factor5 = rolling_std(Factor2, factor_name="Factor5")
    Factor6 = aggr_sum(Factor2, Factor1, Factor1, factor_name="Factor7")
    Factor7 = qs_sum(Factor1, Factor2)
    Factor8 = rolling_regress(Factor1, Factor2)
    Factor9 = rank_section(Factor1, mask=Factor2)
    Factor10 = rank_section(Factor1, cat_data=Factor2)
    
    # print(Factor1.readData(ids=IDs, dts=DTs))
    # print(Factor2.readData(ids=IDs, dts=DTs))
    # print(Factor3.readData(ids=IDs, dts=DTs))
    # print(Factor4.readData(ids=IDs, dts=DTs))
    # print(Factor5.readData(ids=IDs, dts=DTs))
    # print(Factor6.readData(ids=IDs, dts=DTs))
    # print(Factor7.readData(ids=IDs, dts=DTs))
    print(Factor9.QSID)
    print(Factor10.QSID)
    
    print("===")
