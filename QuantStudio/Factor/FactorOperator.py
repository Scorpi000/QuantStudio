# coding=utf-8
"""内置的因子运算"""
import json
import datetime as dt
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import Field
from numpy.lib import recfunctions as rfn

from QuantStudio.Core import __QS_Error__
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.FactorOperation import PointOperator, TimeOperator, SectionOperator, PanelOperator
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools import DataPreprocessingFun


# ----------------------单点运算--------------------------------
class AsType(PointOperator):
    def __init__(self, dtype:str="double", args={}, config_file=None, **kwargs):
        Args = {"Name": "astype", "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        if args["dtype"]=="double":
            return x[0].astype(float)
        elif args["dtype"]=="string":
            return x[0].astype(str)
        elif args["dtype"]=="object":
            return x[0].astype("O")
        else:
            raise Exception(f"不支持的数据类型: {args['dtype']}")
    
class Log(PointOperator):
    def __init__(self, base:float=np.e, args={}, config_file=None, **kwargs):
        Args = {"Name": "log"} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"base": base} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0].astype(float)
        return np.log(np.where(Data>0, Data, np.nan)) / np.log(args["base"])

class NotNull(PointOperator):
    def __init__(self, args={}, config_file=None, **kwargs):
        Args = {"Name": "notnull"} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.notnull(x[0])

class IsIn(PointOperator):
    def __init__(self, test_elements=[], args={}, config_file=None, **kwargs):
        Args = {"Name": "isin"} | args | {"Arity": 1,"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"test_elements": test_elements} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.isin(x[0], args["test_elements"]).astype(float)

class Applymap(PointOperator):
    def __init__(self, func=id, dtype:str="double", args={}, config_file=None, **kwargs):
        Args = {"Name": "applymap", "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).map(args["func"]).values
    
class Where(PointOperator):
    def __init__(self, dtype:str="double", args={}, config_file=None, **kwargs):
        Args = {"Name": "where", "DataType": dtype} | args | {"Arity": 3, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.where(x[1], x[0], x[2])
    
    def __call__(self, f, mask, other, factor_args:Dict={}, **kwargs):
        return super().__call__(f, mask, other, factor_args=factor_args, **kwargs)

class Fetch(PointOperator):
    def __init__(self, pos:Union[int, str]=0, dtype:str="double", compound_type=None, args={}, config_file=None, **kwargs):
        Args = {"Name": "fetch", "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"pos": pos, "dtype": dtype, "compound_type": compound_type} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
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

class Strftime(PointOperator):
    def __init__(self, dt_format:str="%Y%m%d", args={}, config_file=None, **kwargs):
        Args = {"Name": "strftime"} | args | {"Arity": 1, "DataType": "string", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dt_format": dt_format} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        DTFormat = args["dt_format"]
        return pd.DataFrame(x[0]).map(lambda x: x.strftime(DTFormat) if pd.notnull(x) else None).values

class Strptime(PointOperator):
    def __init__(self, dt_format:str="%Y%m%d", args={}, config_file=None, **kwargs):
        Args = {"Name": "strptime"} | args | {"Arity": 1, "DataType": "object", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"dt_format": dt_format} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        DTFormat = args["dt_format"]
        return pd.DataFrame(x[0]).map(lambda x: dt.datetime.strptime(x, DTFormat) if pd.notnull(x) else None).values

class Sum(PointOperator):
    def __init__(self, all_nan=0, dtype:str="double", args={}, config_file=None, **kwargs):
        Args = {"Name": "sum", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nansum(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt

class Max(PointOperator):
    def __init__(self, all_nan=np.nan, dtype:str="double", args={}, config_file=None, **kwargs):
        Args = {"Name": "max", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanmax(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    

class Min(PointOperator):
    def __init__(self, all_nan=np.nan, dtype:str="double", args={}, config_file=None, **kwargs):
        Args = {"Name": "min", "DataType": dtype} | args | {"DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": np.nan, "dtype": dtype} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanmin(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    

class Rank(PointOperator):
    def __init__(self, ascending:bool=True, uniformization:bool=True, args={}, config_file=None, **kwargs):
        Args = {"Name": "rank"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"ascending": ascending, "uniformization": uniformization} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x) * (float(args["ascending"]) * 2 - 1)
        Rslt = np.argsort(np.argsort(Data, axis=0), axis=0)[0].astype(float)
        Rslt[pd.isnull(Data[0])] = np.nan
        if args["uniformization"]:
            TotalNum = np.sum(pd.notnull(Data), axis=0)
            Rslt = Rslt / TotalNum
        return Rslt
    

class Mean(PointOperator):
    def __init__(self, weights=None, ignore_nan_weight=True, args={}, config_file=None, **kwargs):
        Args = {"Name": "mean"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"weights": weights, "ignore_nan_weight": ignore_nan_weight} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
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
    def __init__(self, ddof=1, all_nan:float=np.nan, args={}, config_file=None, **kwargs):
        Args = {"Name": "std"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"all_nan": all_nan, "ddof": ddof} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanstd(Data, axis=0, ddof=args["ddof"])
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    

class Regress(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=2, title="入参数", frozen=True)
    
    def __init__(self, intercept=True, output:Optional[str]=None, args={}, config_file=None, **kwargs):
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
    
    def calculate(self, f, idt, iid, x, args):
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
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=2, title="入参数", frozen=True)
    
    def __init__(self, args={}, config_file=None, **kwargs):
        Args = {"Name": "regressChangeRate"} | args | {"DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Y = np.array(x)
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
    def __init__(self, args={}, config_file=None, **kwargs):
        Args = {"Name": "tolist"} | args | {"DataType": "object", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"mask": False} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = {i: iData for i, iData in enumerate(x)}
        if args["mask"]:
            Rslt = Panel(Data, major_axis=idt, minor_axis=iid).sort_index(axis=0).to_frame(filter_observations=False)
            Rslt = Rslt[Rslt.pop(0)==1]
            return Rslt.apply(lambda s: s.tolist(), axis=1).unstack().reindex(index=idt, columns=iid).values
        else:
            return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: s.tolist(), axis=1).unstack().values
    
    def __call__(self, *factors, mask=None, factor_args:Dict={}, **kwargs):
        if mask is None:
            return super(ToList, self.new(args={"ModelArgs": {"mask": False}})).__call__(*factors, factor_args=factor_args, **kwargs)
        else:
            return super(ToList, self.new(args={"ModelArgs": {"mask": True}})).__call__(mask, *factors, factor_args=factor_args, **kwargs)

class ToJson(PointOperator):
    def __init__(self, args={}, config_file=None, **kwargs):
        Args = {"Name": "tojson"} | args | {"DataType": "string", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).map(lambda v: json.dumps(v, ensure_ascii=False) if pd.notnull(v) else None).values

class ToCompound(PointOperator):
    def __init__(self, args={}, config_file=None, **kwargs):
        Args = {"Name": "toCompound"} | args | {"DataType": "object", "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = {i: iData for i, iData in enumerate(x)}
        return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: tuple(s), axis=1).unstack().values
    
    def __call__(self, *factors, fields=None, factor_args:Dict={}, **kwargs):
        if fields is None:
            DataTypes = [(iFactor.Name, iFactor.getMetaData(key="DataType")) for iFactor in factors]
        else:
            DataTypes = [(factors[i].Name if not iField else iField, factors[i].getMetaData(key="DataType")) for i, iField in enumerate(fields)]
        return super(ToCompound, self.new(args={"CompoundType": DataTypes})).__call__(*factors, factor_args=factor_args, **kwargs)

# ----------------------时序运算--------------------------------
class Lag(TimeOperator):
    def __init__(self, lag_period=1, window=1, dt_change_fun=None, args={}, config_file=None, **kwargs):
        Args = {"Name": "lag", "LookBack": [window], "DataType": "double"} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window, "lag_period": lag_period, "dt_change_fun": dt_change_fun} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0]
        if args.get('dt_change_fun', None) is None: return Data[self.Args["LookBack"][0]-args['lag_period']:Data.shape[0]-args['lag_period']]
        TargetDTs = args['dt_change_fun'](idt)
        Data = pd.DataFrame(Data, index=idt)
        TargetData = Data.reindex(index=TargetDTs).values
        TargetData[args['lag_period']:] = TargetData[:-args['lag_period']]
        if self._QSArgs.DataType!="double":
            Data = pd.DataFrame(np.empty(Data.shape,dtype="O"),index=Data.index,columns=iid)
        else:
            Data = pd.DataFrame(index=Data.index,columns=iid,dtype="float")
        Data.loc[TargetDTs] = TargetData
        return Data.fillna(method='pad').values[self.Args["LookBack"][0]:]
    
    def __call__(self, f: Factor, factor_args:Dict={}, **kwargs):
        DataType = f.getMetaData(key="DataType")
        if DataType != self._QSArgs.DataType:
            return super(Lag, self.new(args={"DataType": DataType})).__call__(f, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, factor_args=factor_args, **kwargs)

class RollingRank(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, ascending:bool=True, uniformization:bool=True, args={}, config_file=None, **kwargs):
        Args = {"Name": "rollingRank", "LookBack": [window - 1]} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window, "min_periods": min_periods, "ascending": ascending, "uniformization": uniformization} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
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
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, weights=None, args={}, config_file=None, **kwargs):
        if weights is not None: window = len(weights)
        Args = {"Name": "rollingMean", "LookBack": [window - 1]} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] =  {"window": window, "min_periods": min_periods, "win_type": win_type, "weights": weights} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        Args = args.copy()
        weights = Args.pop("weights")
        if not weights:
            return Data.rolling(**Args).mean().values[self.Args["LookBack"][0]:]
        else:
            weights = np.array(weights)
            return Data.rolling(**Args).apply(lambda x: np.nansum(x * weights) / np.nansum(pd.notnull(x) * weights), raw=True).values[self.Args["LookBack"][0]:]
    
        
class RollingApply(TimeOperator):
    def __init__(self, func=np.nansum, dtype:str="double", window:int=1, min_periods:int=1, win_type:Optional[str]=None, args={}, config_file=None, **kwargs):
        Args = {"Name": "rollingApply", "LookBack": [window - 1], "DataType": dtype} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"func": func, "dtype": dtype, "window": window, "min_periods": min_periods, "win_type": win_type} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        args = args.copy()
        func, dtype = args.pop("func"), args.pop("dtype")
        return Data.rolling(**args).apply(func, raw=True).values[self.Args["LookBack"][0]:]

class RollingChangeRate(TimeOperator):
    def __init__(self, window:int=1, args={}, config_file=None, **kwargs):
        Args = {"Name": "rollingChangeRate", "LookBack": [window - 1]} | args | {"Arity": 1, "DataType": "double", "DTMode": "多时点", "IDMode": "多ID"}
        Args["ModelArgs"] = {"window": window} | Args.get("ModelArgs", {})
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
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
    def __init__(self, window:int=1, min_periods:int=1, intercept=True, output:Optional[str]=None, args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "rollingRegress"} | args | {"DataType": "double", "DTMode": "单时点", "IDMode": "单ID"}
        Args["ModelArgs"] = {"window": window, "min_periods": min_periods, "intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        Args["LookBack"] = [Args["ModelArgs"]["window"] - 1] * Arity
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double"), ("beta", "double")]
        else:
            Args["DataType"] = "double"
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Y, X = x[0].astype(float), (np.array(x[1:], dtype=float).T if len(x)>1 else np.arange(0, x[0].shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        if np.sum(Mask) < args["min_periods"]: return (np.nan if args["output"] is not None else (np.nan,) * (1 + X.shape[1]))
        Y, X = Y[Mask], X[Mask]
        if args["intercept"]: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        if args["output"] is None: return tuple(Rslt.params) if args["intercept"] else (0, )+tuple(Rslt.params)
        elif args["output"]=="alpha": return Rslt.params[0] if args["intercept"] else 0
        else: return Rslt.params[int(args["output"][4:]) + int(args["intercept"])]
        
    def __call__(self, endog:Factor, *exog, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)

# ----------------------截面运算--------------------------------
class SectionRank(SectionOperator):
    def __init__(self, ascending:bool=True, uniformization:bool=True, args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "rankSection"} | args | {"DataType": "double", "DTMode": "多时点", "OutputMode": "全截面"}
        Args["ModelArgs"] = {"uniformization": uniformization, "ascending": ascending} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [None])[0]
        Args["DescriptorSection"] = [descriptor_ids] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        FactorData = x[0]
        Mask = (x[1].astype(bool) if f.UserData["mask"] else [None] * FactorData.shape[0])
        CatData = (x[-1] if f.UserData["cat_data"] else [None] * FactorData.shape[0])
        Rslt = np.full_like(FactorData, fill_value=np.nan)
        for i in range(FactorData.shape[0]):
            Rslt[i] = DataPreprocessingFun.standardizeRank(FactorData[i], mask=Mask[i], cat_data=CatData[i], perturbation=False, offset=0, **args)
        return Rslt
    
    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, *, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        f = super().__call__(*Factors, factor_args=factor_args, **kwargs)
        f.UserData = {"mask": (mask is not None), "cat_data": (cat_data is not None)}
        return f

class Aggregate(SectionOperator):
    def __init__(self, aggr_func=np.nansum, descriptor_ids=None, dtype="double", args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "aggregate"} | args | {"DataType": dtype, "DTMode": "单时点", "OuptutMode": "全截面"}
        Args["ModelArgs"] = {"aggr_func": aggr_func, "dtype": dtype} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [descriptor_ids])[0]
        Args["DescriptorSection"] = [descriptor_ids] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f, idt, iid, x, args):
        nID = len(iid)
        FactorData = x[0]
        if f.UserData["mask"]:
            Mask = (x[1]==1)
        else:
            Mask = np.full(FactorData.shape, fill_value=True)
        AggrFunc = args["aggr_func"]
        if f.UserData["cat_data"]:
            CatData = x[-1]
            Rslt = np.full(shape=(nID, ), fill_value=np.nan)
            if f.UserData["section_chged"]:
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

    def __call__(self, f, mask=None, cat_data=None, *, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        f = super().__call__(*Factors, factor_args=factor_args, **kwargs)
        f.UserData = {"mask": (mask is not None), "cat_data": (cat_data is not None), "section_chged": (self._QSArgs.DescriptorSection[0] is not None)}
        return f

class Disaggregate(SectionOperator):
    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        Arity: Optional[int] = Field(default=None, ge=1, le=2, title="入参数", frozen=True)
    
    def __init__(self, aggr_ids, disaggr_ids=None, args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "disaggregate"} | args | {"DataType": "double", "DTMode": "多时点", "OutputMode": "全截面"}
        DescriptorSection = Args.get("DescriptorSection", [aggr_ids, disaggr_ids])
        if len(DescriptorSection) < Arity: DescriptorSection.append(disaggr_ids)
        elif len(DescriptorSection) > Arity: DescriptorSection = DescriptorSection[:Arity]
        Args["DescriptorSection"] = DescriptorSection
        return super().__init__(args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f, idt, iid, x, args):
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
    
    def __call__(self, f, cat_data=None, *, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if cat_data is not None: Factors.append(cat_data)
        kwargs["operator_kwargs"] =  {"aggr_ids": self._QSArgs.DescriptorSection[0]} | kwargs.get("operator_kwargs", {})
        f = super().__call__(*Factors, factor_args=factor_args, **kwargs)
        f.UserData = {"cat_data": (cat_data is not None)}
        return f

class ConcatSection(SectionOperator):
    def __init__(self, descriptor_sections=[], dtype="double", args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "concatSection", "DataType": dtype} | args | {"DTMode": "多时点", "OutputMode": "全截面"}
        Args["ModelArgs"] = {"dtype": dtype} | Args.get("ModelArgs", {})
        DescriptorSection = Args.get("DescriptorSection", descriptor_sections)
        Args["DescriptorSection"] = DescriptorSection[:Arity] + [None] * max(0, Arity - len(DescriptorSection))
        return super().__init__(args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(np.concatenate(x, axis=1), columns=sum(((iid if iIDs is None else iIDs) for iIDs in self.Args.DescriptorSection), [])).reindex(columns=iid).values
    

class ChgSection(SectionOperator):
    # id_map: {新ID: 旧ID}
    def __init__(self, old_ids, id_map={}, args={}, config_file=None, **kwargs):
        Args = {"Name": "chgSection", "DataType": "double"} | args | {"Arity": 1, "DTMode": "多时点", "OutputMode": "全截面"}
        Args["ModelArgs"] = {"id_map": id_map} | Args.get("ModelArgs", {})
        if "DescriptorSection" not in Args: Args["DescriptorSection"] = [old_ids]
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0]
        IDMap = args["id_map"]
        OldIDs = f._QSArgs.DescriptorSection[0]
        Rslt = np.full(shape=(len(idt), len(iid)), fill_value=np.nan, dtype=Data.dtype)
        for i, iID in enumerate(iid):
            iOldID = IDMap.get(iID, None)
            if iOldID not in OldIDs: continue
            Rslt[:, i] = Data[:, OldIDs.index(iOldID)]
        return Rslt
    
    def __call__(self, f, factor_args:Dict={}, **kwargs):
        kwargs["operator_kwargs"] =  {"old_ids": self._QSArgs.DescriptorSection[0]} | kwargs.get("operator_kwargs", {})
        DataType = f.getMetaData(key="DataType")
        if DataType != self._QSArgs.DataType:
            return super(ChgSection, self.new(args={"DataType": DataType}, **kwargs["operator_kwargs"])).__call__(f, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, factor_args=factor_args, **kwargs)

class SectionRegress(SectionOperator):
    # output: alpha, beta{i}, resid
    def __init__(self, intercept=True, output:Optional[str]=None, descriptor_ids=None, args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "regressSection"} | args | {"DTMode": "单时点", "OutputMode": "全截面"}
        Args["ModelArgs"] = {"intercept": intercept, "output": output} | Args.get("ModelArgs", {})
        descriptor_ids = Args.get("DescriptorSection", [descriptor_ids])[0]
        Args["DescriptorSection"] = [descriptor_ids] * Arity
        if Args["ModelArgs"]["output"] is None:
            Args["DataType"] = "object"
            Args["CompoundType"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(Arity-1)] + [("resid", "double")]
        else:
            Args["DataType"] = "double"        
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Y, X = x[0].astype(float), (np.array(x[1:], dtype=float).T if len(x)>1 else np.arange(0, x[0].shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        Y, X = Y[Mask], X[Mask]
        if args["intercept"]: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        Beta = (tuple(Rslt.params) if args["intercept"] else (0, )+tuple(Rslt.params))
        Resid = np.full(shape=Mask.shape, fill_value=np.nan)
        Resid[Mask] = Rslt.resid
        if args["output"] is None: return [Beta+(Resid[i], ) for i in range(len(iid))]
        elif args["output"]=="alpha": Rslt = Beta[0]
        elif args["output"] == "resid": return Resid
        else: Rslt = Beta[int(args["output"][4:]) + 1]
        return np.full(shape=(len(iid),), fill_value=Rslt)
        
    def __call__(self, endog:Factor, *exog, factor_args:Dict={}, **kwargs):
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)

# ----------------------面板运算--------------------------------
class PanelRegress(PanelOperator):
    # output: alpha, beta{i}, resid
    def __init__(self, window:int=1, intercept=True, output:Optional[str]=None, descriptor_ids=None, args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "regressPanel"} | args | {"DTMode": "单时点", "OutputMode": "全截面"}
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
    
    def calculate(self, f, idt, iid, x, args):
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
        
    def __call__(self, endog:Factor, *exog, factor_args:Dict={}, **kwargs):
        return super().__call__(endog, *exog, factor_args=factor_args, **kwargs)


if __name__=="__main__":
    from functools import partial
    from QuantStudio.Core.Factor import DataFactor
    
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
    Factor7 = aggr_sum(Factor2, Factor1, Factor1, factor_name="Factor7")
    Factor8 = qs_sum(Factor1, Factor2)
    Factor9 = rolling_regress(Factor1, Factor2)
    Factor10 = rank_section(Factor1, Factor2)
    Factor11 = disaggr(Factor1)
    Factor12 = chg_section(Factor1)
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    #print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    #print(Factor6.readData(ids=IDs, dts=DTs))
    print(Factor7.readData(ids=IDs, dts=DTs))
    
    print("===")
