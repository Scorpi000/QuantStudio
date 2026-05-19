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
from QuantStudio.Factor.FactorOperation import PointOperator, TimeOperator, SectionOperator, PanelOperator, PointOperation, TimeOperation, SectionOperation, PanelOperation
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools import DataPreprocessingFun


# ----------------------单点运算--------------------------------
class AsType(PointOperator):
    """数据类型转换"""

    def __init__(self, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化数据类型转换算子
        
        Args:
            dtype: 新的数据类型
        """
        Args = {"Name": "astype", "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        if args["dtype"]=="double":
            return x[0].astype(float)
        elif args["dtype"]=="string":
            return x[0].astype(str)
        elif args["dtype"]=="object":
            return x[0].astype("O")
        else:
            raise Exception(f"不支持的数据类型: {args['dtype']}")
    
class Log(PointOperator):
    """对数"""

    def __init__(self, base:float=np.e, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化对数算子
        
        Args:
            base: 底数
        """
        Args = {"Name": "log"} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"base": base} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = x[0].astype(float)
        return np.log(np.where(Data>0, Data, np.nan)) / np.log(args["base"])

class NotNull(PointOperator):
    """非NULL检测算子"""

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化非NULL检测算子"""
        Args = {"Name": "notnull"} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        return pd.notnull(x[0])

class IsIn(PointOperator):
    """是否属于给定集合的检测算子"""

    def __init__(self, test_elements:List[Any]=[], args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化检测因子值是否属于给定集合的算子

        Args:
            test_elements: 给定的检测集合
        """
        Args = {"Name": "isin"} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"test_elements": test_elements} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        return np.isin(x[0], args["test_elements"]).astype(float)

class ApplyArrayFunc(PointOperator):
    """施加对 array 整体运算的函数, 比如 numpy.floor"""

    def __init__(self, func:Callable, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 array 整体运算算子

        Args:
            func: 施加到每个因子值的函数
            dtype: func 函数返回值的数据类型
        """
        Args = {"Name": "applyArrayFunc", "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        return args["func"](*x)

class Applymap(PointOperator):
    """map 操作"""

    def __init__(self, func:Callable[[Any], Any]=id, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 map 操作算子

        Args:
            func: 施加到每个因子值的函数
            dtype: func 函数返回值的数据类型
        """
        Args = {"Name": "applymap", "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        return pd.DataFrame(x[0]).map(args["func"]).values

class Where(PointOperator):
    """where 操作"""

    def __init__(self, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 where 操作算子

        Args:
            dtype: where 操作返回值的数据类型
        """
        Args = {"Name": "where", "DataType": dtype} | args | {"Arity": 3, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        return np.where(x[1], x[0], x[2])
    
    def __call__(self, f, mask, other, factor_args:dict={}, **kwargs) -> PointOperation:
        return super().__call__(f, mask, other, factor_args=factor_args, **kwargs)

class Fetch(PointOperator):
    """从复合因子中取出简单因子的操作算子"""

    def __init__(self, pos:Union[int, str]=0, dtype:Literal["double", "string", "object"]="double", compound_type:List[Tuple[str, Literal["double", "string", "object"]]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "fetch", "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"pos": pos, "dtype": dtype, "compound_type": compound_type} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = x[0]
        CompoundType = args["compound_type"]
        if CompoundType and isinstance(args["pos"], str):
            DefaultData = np.array([[None]], dtype="O")
            DefaultData[0, 0] = (None,) * len(CompoundType)
            DefaultData = DefaultData.repeat(Data.shape[0], axis=0).repeat(Data.shape[1], axis=1)
            Data = np.where(pd.notnull(Data), Data, DefaultData)
            DataType = np.dtype([(iCol, float if iType=="double" else "O") for iCol, iType in CompoundType])
        else:
            SampleData = Data[pd.notnull(Data)]
            if SampleData.shape[0]==0:
                return (np.full(Data.shape, fill_value=np.nan, dtype="float") if f._QSArgs.DataType=="double" else np.full(Data.shape, fill_value=None, dtype="O"))
            SampleData = SampleData[0]
            DataType = np.dtype([(str(i),(float if isinstance(SampleData[i], float) else "O")) for i in range(len(SampleData))])
        return Data.astype(DataType)[str(args["pos"])]
    
    def __call__(self, x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        Operator = getattr(x, "Operator", None)
        if (Operator is not None) and (Operator._QSArgs.CompoundType != self._QSArgs.ModelArgs["compound_type"]):
            ModelArgs = self._QSArgs.ModelArgs | {"compound_type": Operator._QSArgs.CompoundType}
            return super(Fetch, self.new(args={"ModelArgs": ModelArgs})).__call__(x, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(x, factor_args=factor_args, **kwargs)

class Sum(PointOperator):
    """求和"""

    def __init__(self, all_nan:Any=0, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "sum", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nansum(Data, axis=0)
        if args["dtype"]=="double": Rslt = Rslt.astype(float)
        Mask = (np.sum(pd.notnull(Data), axis=0) == 0)
        Rslt[Mask] = args["all_nan"]
        return Rslt

class Max(PointOperator):
    """最大值"""

    def __init__(self, all_nan:Any=np.nan, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "max", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nanmax(Data, axis=0)
        if args["dtype"]=="double": Rslt = Rslt.astype(float)
        Mask = (np.sum(pd.notnull(Data), axis=0) == 0)
        Rslt[Mask] = args["all_nan"]
        return Rslt

class Min(PointOperator):
    """最小值"""

    def __init__(self, all_nan:Any=np.nan, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "min", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nanmin(Data, axis=0)
        if args["dtype"]=="double": Rslt = Rslt.astype(float)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt

class Rank(PointOperator):
    """排名"""

    def __init__(self, ascending:bool=True, uniformization:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rank"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"ascending": ascending, "uniformization": uniformization} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = np.array(x) * (float(args["ascending"]) * 2 - 1)
        Rslt = np.argsort(np.argsort(Data, axis=0), axis=0)[0].astype(float)
        Rslt[pd.isnull(Data[0])] = np.nan
        if args["uniformization"]:
            TotalNum = np.sum(pd.notnull(Data), axis=0)
            Rslt = Rslt / TotalNum
        return Rslt

class Mean(PointOperator):
    """平均值"""

    def __init__(self, weights:Optional[List[float]]=None, ignore_nan_weight:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "mean"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"weights": weights, "ignore_nan_weight": ignore_nan_weight} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = np.array(x)
        Weights = args["weights"]
        if Weights is None:
            if args["ignore_nan_weight"]:
                return np.nanmean(Data, axis=0)
            Weights = [1] * Data.shape[0]
        Rslt = np.zeros(Data.shape[1:])
        WeightArray = np.zeros(Data.shape[1:])
        for i, iData in enumerate(Data):
            iMask = pd.notnull(iData)
            WeightArray += iMask * Weights[i]
            iData[~iMask] = 0.0
            Rslt += iData * Weights[i]
        if args["ignore_nan_weight"]:
            WeightArray[WeightArray==0.0] = np.nan
            return Rslt / WeightArray
        else:
            Rslt[WeightArray==0.0] = np.nan
            return Rslt / Data.shape[0]

class Std(PointOperator):
    """标准差"""

    def __init__(self, ddof:int=1, all_nan:float=np.nan, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "std"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "ddof": ddof} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = np.array(x)
        Rslt = np.nanstd(Data, axis=0, ddof=args["ddof"])
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt

class Regress(PointOperator):
    """OLS 回归"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=2, title="入参数", frozen=True)
    
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
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Y = np.array(x)
        X = np.arange(Y.shape[0]).astype("float").reshape((Y.shape[0], 1, 1)).repeat(Y.shape[1], axis=1).repeat(Y.shape[2], axis=2)
        X[pd.isnull(Y)] = np.nan
        XBar, YBar = np.nanmean(X, axis=0), np.nanmean(Y, axis=0)
        Beta = np.nansum((X - XBar) * (Y - YBar), axis=0) / np.nansum((X - XBar)**2, axis=0)
        Beta[np.isinf(Beta)] = np.nan
        if args["output"]=="beta": return Beta
        if args["intercept"]:
            Alpha = YBar - Beta * XBar
        else:
            Alpha = np.zeros(shape=YBar.shape)
        if args["output"]=="alpha": return Alpha
        return rfn.unstructured_to_structured(np.array([Alpha, Beta]).swapaxes(0, -1)).T

class RegressChangeRate(PointOperator):
    """以回归方式计算增长率"""
    
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=2, title="入参数", frozen=True)
    
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

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "tolist"} | args | {"DataType": "object", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"mask": False} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = {i: iData for i, iData in enumerate(x)}
        if args["mask"]:
            Rslt = Panel(Data, major_axis=idt, minor_axis=iid).sort_index(axis=0).to_frame(filter_observations=False)
            Rslt = Rslt[Rslt.pop(0)==1]
            return Rslt.apply(lambda s: s.tolist(), axis=1).unstack().reindex(index=idt, columns=iid).values
        else:
            return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: s.tolist(), axis=1).unstack().values
    
    def __call__(self, *x:Factor, mask:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PointOperation:
        if mask is None:
            return super(ToList, self.new(args={"ModelArgs": {"mask": False}})).__call__(*x, factor_args=factor_args, **kwargs)
        else:
            return super(ToList, self.new(args={"ModelArgs": {"mask": True}})).__call__(mask, *x, factor_args=factor_args, **kwargs)

class ToCompound(PointOperator):
    """将多个因子转成单个复合因子"""

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
class Lag(TimeOperator):
    """按照时间回溯数据"""

    def __init__(self, lag_period:int=1, window:Optional[int]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        if window is None: window = lag_period
        Args = {"Name": "lag", "LookBack": [window], "DataType": "double"} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window, "lag_period": lag_period} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = x[0]
        if f._QSArgs.CalcDTRuler is None: return Data[self.Args["LookBack"][0]-args['lag_period']:Data.shape[0]-args['lag_period']]
        TargetDTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
        Data = pd.DataFrame(Data, index=idt)
        TargetData = Data.reindex(index=TargetDTs).values.copy()
        TargetData[args['lag_period']:] = TargetData[:-args['lag_period']]
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

class RollingRank(TimeOperator):
    """滚动排名"""

    def __init__(self, window:int=1, min_periods:int=1, ascending:bool=True, uniformization:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rollingRank", "LookBack": [window - 1]} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window, "min_periods": min_periods, "ascending": ascending, "uniformization": uniformization} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        args = args.copy()
        Data = pd.DataFrame(x[0])
        if not args.pop("ascending"):
            Data = - Data
        Uniformization = args.pop("uniformization")
        Rslt = Data.rolling(**args).rank().values[self.Args["LookBack"][0]:] - 1
        if Uniformization:
            Rslt = Rslt / Data.rolling(**args).count().values[self.Args["LookBack"][0]:]
        return Rslt

class RollingMean(TimeOperator):
    """滚动平均"""

    def __init__(self, window:int=1, min_periods:int=1, weights:Optional[List[float]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        if weights is not None: window = len(weights)
        Args = {"Name": "rollingMean", "LookBack": [window - 1]} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] =  {"window": window, "min_periods": min_periods, "weights": weights} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = pd.DataFrame(x[0])
        Args = args.copy()
        weights = Args.pop("weights")
        if not weights:
            return Data.rolling(**Args).mean().values[self.Args["LookBack"][0]:]
        else:
            weights = np.array(weights)
            return Data.rolling(**Args).apply(lambda x: np.nansum(x * weights) / np.nansum(pd.notnull(x) * weights), raw=True).values[self.Args["LookBack"][0]:]

class RollingApply(TimeOperator):
    """滚动操作"""

    def __init__(self, func:Callable[[np.ndarray], Any]=np.nansum, dtype:Literal["double", "string", "object"]="double", window:int=1, min_periods:int=1, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rollingApply", "LookBack": [window - 1], "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype, "window": window, "min_periods": min_periods} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        args = args.copy()
        func, dtype = args.pop("func"), args.pop("dtype")
        # Data = pd.DataFrame(x[0])
        # return Data.rolling(**args).apply(func, raw=True).values[self.Args["LookBack"][0]:]
        Data = np.lib.stride_tricks.sliding_window_view(x[0], window_shape=args["window"], axis=0)
        Mask = (np.sum(~ np.isnan(Data), axis=-1) < args["min_periods"])
        Data = np.apply_along_axis(func, axis=-1, arr=Data)
        if dtype=="double":
            Data = Data.astype(float)
            Data[Mask] = np.nan
        else:
            Data = Data.astype("O")
            Data[Mask] = None
        return Data

class RollingChangeRate(TimeOperator):
    """滚动增长率"""

    def __init__(self, window:int=1, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rollingChangeRate", "LookBack": [window - 1]} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = x[0]
        Numerator = Data[args["window"]-1:]
        Denominator = Data[:-args["window"]+1]
        Rslt = (Numerator - Denominator) / np.abs(Denominator)
        Mask = (Denominator==0)
        Rslt[Mask] = np.nan
        Rslt[Mask & (Numerator>0)] = 1.0
        Rslt[Mask & (Numerator<0)] = -1.0
        Rslt[Mask & (Numerator==0)] = 0.0
        return Rslt[self.Args["LookBack"][0]-args["window"]+1:]

class RollingRegress(TimeOperator):
    """滚动回归"""

    def __init__(self, window:int=1, min_periods:int=1, intercept:bool=True, output:Optional[Literal["alpha", "beta", "r2"]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "rollingRegress"} | args | {"DataType": "double", "DTMode": "单时点", "IDMode": "单ID"}
        Args["ModelArgs"] = {"window": window, "min_periods": min_periods, "intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        Args["LookBack"] = [Args["ModelArgs"]["window"] - 1] * Arity
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(Arity)] + [("r2", "double")]
        else:
            Args["DataType"] = "double"
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: str, x: List[np.ndarray], args: dict) -> Union[float, Tuple[float]]:
        Y, X = x[0].astype(float), (np.array(x[1:], dtype=float).T if len(x)>1 else np.arange(0, x[0].shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        if np.sum(Mask) < args["min_periods"]: return (np.nan if args["output"] is not None else (np.nan,) * (2 + X.shape[1]))
        Y, X = Y[Mask], X[Mask]
        if args["intercept"]: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        if args["output"] is None: return (tuple(Rslt.params) if args["intercept"] else ((0, ) + tuple(Rslt.params))) + (Rslt.rsquared,)
        elif args["output"]=="alpha": return Rslt.params[0] if args["intercept"] else 0
        elif args["output"]=="r2": return Rslt.rsquared
        else: return Rslt.params[int(args["output"][4:]) + int(args["intercept"])]
        
    def __call__(self, endog:Factor, *exog:Factor, factor_args:dict={}, **kwargs) -> TimeOperation:
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)

# ----------------------截面运算--------------------------------
class SectionRank(SectionOperator):
    """截面排名"""

    def __init__(self, ascending:bool=True, uniformization:bool=True, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "rankSection"} | args | {"DataType": "double", "DTMode": "多时点"}
        Args["ModelArgs"] = {"uniformization": uniformization, "ascending": ascending} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [None])[0]
        Args["DescriptorSection"] = [descriptor_ids] * Arity
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
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class Aggregate(SectionOperator):
    """截面聚合"""

    def __init__(self, aggr_func:Callable[[np.ndarray], Any]=np.nansum, descriptor_ids:Optional[List[str]]=None, dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "aggregate"} | args | {"DataType": dtype, "DTMode": "单时点"}
        Args["ModelArgs"] = {"aggr_func": aggr_func, "dtype": dtype} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [descriptor_ids])[0]
        Args["DescriptorSection"] = [descriptor_ids] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f: Factor, idt: dt.datetime, iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        nID = len(iid)
        FactorData = x[0]
        if f._QSArgs.ModelArgs["mask"]:
            Mask = (x[1]==1)
        else:
            Mask = np.full(FactorData.shape, fill_value=True)
        AggrFunc = args["aggr_func"]
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
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None), "section_chged": (self._QSArgs.DescriptorSection[0] is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class Disaggregate(SectionOperator):
    """截面反聚合"""

    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=1, le=2, title="入参数", frozen=True)
    
    def __init__(self, aggr_ids:List[str], disaggr_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "disaggregate"} | args | {"DataType": "double", "DTMode": "多时点"}
        DescriptorSection = Args.get("DescriptorSection", [aggr_ids, disaggr_ids])
        if len(DescriptorSection) < Arity: DescriptorSection.append(disaggr_ids)
        elif len(DescriptorSection) > Arity: DescriptorSection = DescriptorSection[:Arity]
        Args["DescriptorSection"] = DescriptorSection
        return super().__init__(args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        nDT, nID = len(idt), len(iid)
        FactorData = x[0]
        if args["cat_data"]:
            CatData = x[-1]
            Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
            for i, iID in enumerate(self.Args.DescriptorSection[0]):
                iMask = (CatData==iID)
                Rslt[iMask] = FactorData[:, [i]].repeat(nID, axis=1)[iMask]
        else:
            Rslt = FactorData.repeat(nID, axis=1)
        return Rslt
    
    def __call__(self, f:Factor, cat_data:Optional[Factor]=None, factor_args:dict={}, **kwargs) -> SectionOperation:
        Factors = [f]
        if cat_data is not None: Factors.append(cat_data)
        kwargs["operator_kwargs"] =  {"aggr_ids": self._QSArgs.DescriptorSection[0]} | kwargs.get("operator_kwargs", {})
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"cat_data": (cat_data is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)

class ConcatSection(SectionOperator):
    """截面拼接"""

    def __init__(self, descriptor_sections:List[List[str]]=[], dtype:Literal["double", "string", "object"]="double", args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or max(1, len(descriptor_sections))
        Args = {"Name": "concatSection", "DataType": dtype} | args | {"DTMode": "多时点"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        DescriptorSection = Args.get("DescriptorSection", descriptor_sections)
        Args["DescriptorSection"] = DescriptorSection[:Arity] + [None] * max(0, Arity - len(DescriptorSection))
        return super().__init__(args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        return pd.DataFrame(np.concatenate(x, axis=1), columns=sum(((iid if iIDs is None else iIDs) for iIDs in self.Args.DescriptorSection), [])).reindex(columns=iid).values

class ChgSection(SectionOperator):
    """修改截面"""

    # id_map: {新ID: 旧ID}
    def __init__(self, old_ids:List[str], id_map:Dict[str, str]={}, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "chgSection", "DataType": "double"} | args | {"Arity": 1, "DTMode": "多时点"}
        Args["ModelArgs"] = {"id_map": id_map} | Args.get("ModelArgs", {})
        if "DescriptorSection" not in Args: Args["DescriptorSection"] = [old_ids]
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Data = x[0]
        IDMap = args["id_map"]
        OldIDs = f._QSArgs.DescriptorSection[0]
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

class SectionRegress(SectionOperator):
    """截面回归"""

    # output: alpha, beta{i}, resid
    def __init__(self, intercept:bool=True, output:Optional[str]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "regressSection"} | args | {"DTMode": "单时点"}
        Args["ModelArgs"] = {"intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [None])[0]
        Args["DescriptorSection"] = [descriptor_ids] * Arity
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(Arity-1)] + [("resid", "double")]
        else:
            Args["DataType"] = "double"        
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: dt.datetime, iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Y, X = x[0].astype(float), (np.array(x[1:], dtype=float).T if len(x)>1 else np.arange(0, x[0].shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        Y, X = Y[Mask], X[Mask]
        if args["intercept"]: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        Beta = (tuple(Rslt.params) if args["intercept"] else (0, ) + tuple(Rslt.params))
        Resid = np.full(shape=Mask.shape, fill_value=np.nan)
        Resid[Mask] = Rslt.resid
        if args["output"] is None: return [Beta+(Resid[i], ) for i in range(len(iid))]
        elif args["output"]=="alpha": Rslt = Beta[0]
        elif args["output"] == "resid": return Resid
        else: Rslt = Beta[int(args["output"][4:]) + 1]
        return np.full(shape=(len(iid),), fill_value=Rslt)
        
    def __call__(self, endog:Factor, *exog:Factor, factor_args:dict={}, **kwargs) -> SectionOperation:
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)

# ----------------------面板运算--------------------------------
class PanelRegress(PanelOperator):
    """面板回归"""

    # output: alpha, beta{i}, resid
    def __init__(self, window:int=1, intercept:bool=True, output:Optional[str]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "regressPanel"} | args | {"DTMode": "单时点"}
        Args["ModelArgs"] = {"window": window, "intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [descriptor_ids])[0]
        Args["DescriptorSection"] = [descriptor_ids] * Arity
        Args["LookBack"] = [Args["ModelArgs"]["window"] - 1] * Arity
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(Arity-1)] + [("resid", "double")]
        else:
            Args["DataType"] = "double"
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        Y = x[0].astype(float).flatten()
        X = (np.array(x[1:], dtype=float).reshape((len(x)-1, -1)).T if len(x)>1 else np.arange(0, Y.shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        Y, X = Y[Mask], X[Mask]
        if args["intercept"]: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        Beta = (tuple(Rslt.params) if args["intercept"] else (0, )+tuple(Rslt.params))
        Resid = np.full(shape=Mask.shape, fill_value=np.nan)
        Resid[Mask] = Rslt.resid
        Resid = Resid[-len(iid):]
        if args["output"] is None: return [Beta+(Resid[i], ) for i in range(len(iid))]
        elif args["output"]=="alpha": Rslt = Beta[0]
        elif args["output"] == "resid": return Resid
        else: Rslt = Beta[int(args["output"][4:]) + 1]
        return np.full(shape=(len(iid),), fill_value=Rslt)
    
    def __call__(self, endog:Factor, *exog:Factor, factor_args:dict={}, **kwargs) -> PanelOperation:
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)


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
