# coding=utf-8
"""内置的因子运算"""
import json
import datetime as dt
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.lib import recfunctions as rfn

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PointOperator, TimeOperator, SectionOperator, PanelOperator
from QuantStudio.Tools import DataPreprocessingFun
from QuantStudio.Tools.api import Panel


# ----------------------单点运算--------------------------------
class AsType(PointOperator):
    def __init__(self, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "astype", "入参数": 1, "最大入参数": 1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"dtype": dtype}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        if args["dtype"]=="double":
            return x[0].astype(float)
        elif args["dtype"]=="string":
            return x[0].astype(str)
        elif args["dtype"]=="object":
            return x[0].astype("O")
        else:
            raise Exception(f"不支持的数据类型: {args['dtype']}")
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(f, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Log(PointOperator):
    def __init__(self, base:float=np.e, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "log", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {"base": base}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0].astype(float)
        return np.log(np.where(Data>0, Data, np.nan)) / np.log(args["base"])
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class NotNull(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "notnull", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.notnull(x[0])
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class IsIn(PointOperator):
    def __init__(self, test_elements=[], sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "isin", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {"test_elements": test_elements}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.isin(x[0], args["test_elements"]).astype(float)
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Applymap(PointOperator):
    def __init__(self, func=id, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "applymap", "入参数": 1, "最大入参数": 1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"func": func, "dtype": dtype}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).applymap(args["func"]).values
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(f, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Where(PointOperator):
    def __init__(self, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "where", "入参数": 3, "最大入参数": 3, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"dtype": dtype}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.where(x[1], x[0], x[2])
    
    def __call__(self, f, mask, other, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(f, mask, other, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, mask, other, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Fetch(PointOperator):
    def __init__(self, pos:Union[int, str]=0, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "fetch", "入参数": 1, "最大入参数": 1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"pos": pos, "dtype": dtype, "compound_type": None}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
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
    
    def __call__(self, f, pos:Union[int, str]=0, dtype:str="double", factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({"pos": pos, "dtype": dtype, "compound_type": None})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        CompoundType = getattr(f._QSArgs, "CompoundType", None)
        if CompoundType:
            if isinstance(factor_args["参数"]["pos"], str):
                factor_args["参数"]["dtype"] = dict(CompoundType)[factor_args["参数"]["pos"]]
            else:
                factor_args["参数"]["pos"], factor_args["参数"]["dtype"] = CompoundType[int(factor_args["参数"]["pos"])]
            factor_args["参数"]["compound_type"] = CompoundType
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(f, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Strftime(PointOperator):
    def __init__(self, dt_format:str="%Y%m%d", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "strftime", "入参数": 1, "最大入参数": 1, "数据类型": "string", "运算时点": "多时点", "运算ID": "多ID", "参数": {"dt_format": dt_format}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        DTFormat = args["dt_format"]
        return pd.DataFrame(x[0]).applymap(lambda x: x.strftime(DTFormat) if pd.notnull(x) else None).values
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Strptime(PointOperator):
    def __init__(self, dt_format:str="%Y%m%d", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "strptime", "入参数": 1, "最大入参数": 1, "数据类型": "object", "运算时点": "多时点", "运算ID": "多ID", "参数": {"dt_format": dt_format}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        DTFormat = args["dt_format"]
        return pd.DataFrame(x[0]).applymap(lambda x: dt.datetime.strptime(x, DTFormat) if pd.notnull(x) else None).values
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Sum(PointOperator):
    def __init__(self, all_nan:float=0, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "sum", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": all_nan, "dtype": dtype}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nansum(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(*factors, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Max(PointOperator):
    def __init__(self, all_nan:float=np.nan, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "max", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": np.nan, "dtype": dtype}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanmax(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(*factors, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Min(PointOperator):
    def __init__(self, all_nan:float=np.nan, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "min", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": np.nan, "dtype": dtype}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanmin(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, all_nan:float=np.nan, dtype:str="double", factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(*factors, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Rank(PointOperator):
    def __init__(self, ascending:bool=True, uniformization:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rank", "入参数": 1, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {"ascending": ascending, "uniformization": uniformization}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x) * (float(args["ascending"]) * 2 - 1)
        Rslt = np.argsort(np.argsort(Data, axis=0), axis=0)[0]
        Rslt[pd.isnull(Data[0])] = np.nan
        if args["uniformization"]:
            TotalNum = np.sum(pd.notnull(Data), axis=0)
            Rslt = Rslt / TotalNum
        return Rslt
    
    def __call__(self, f, *peers, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        return super().__call__(f, *peers, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Mean(PointOperator):
    def __init__(self, weights=None, ignore_nan_weight=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "mean", "入参数": 1, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {"weights": weights, "ignore_nan_weight": ignore_nan_weight}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
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
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Std(PointOperator):
    def __init__(self, ddof=1, all_nan:float=np.nan, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "std", "入参数": 1, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": all_nan, "ddof": ddof}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanstd(Data, axis=0, ddof=args["ddof"])
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Regress(PointOperator):
    def __init__(self, intercept=True, output:Optional[str]=None, sys_args={}, config_file=None, **kwargs):
        if output not in ("alpha", "beta"):
            raise __QS_Error__(f"算子 Regress 的输入参数 output 只能取值为 'alpha' 或者 'beta', 不支持 '{output}'")
        Args = {"名称": "regress", "入参数": 2, "最大入参数": -1, "运算时点": "多时点", "运算ID": "多ID", "参数": {"intercept": intercept, "output": output}}
        if output is None:
            Args["数据类型"] = "object"
            Args["复合类型"] = [("alpha", "double"), ("beta", "double")]
        else:
            Args["数据类型"] = "double"
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
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
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["参数"] = Args
        DataType = ("object" if Args["output"] is None else "double")
        if DataType!=self._QSArgs.DataType:
            OpArgs = {"数据类型": DataType}
            if DataType=="object": OpArgs["复合类型"] = [("alpha", "double"), ("beta", "double")]
            return super().__call__(*factors, args=OpArgs, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RegressChangeRate(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "regressChangeRate", "入参数": 2, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
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
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class ToList(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "tolist", "入参数": 1, "最大入参数": -1, "数据类型": "object", "运算时点": "多时点", "运算ID": "多ID", "参数": {"mask": False}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = {i: iData for i, iData in enumerate(x)}
        if args["mask"]:
            Rslt = Panel(Data, major_axis=idt, minor_axis=iid).sort_index(axis=0).to_frame(filter_observations=False)
            Rslt = Rslt[Rslt.pop(0)==1]
            return Rslt.apply(lambda s: s.tolist(), axis=1).unstack().reindex(index=idt, columns=iid).values
        else:
            return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: s.tolist(), axis=1).unstack().values
    
    def __call__(self, *factors, mask=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = {"mask": mask is not None}
        factor_args = factor_args.copy()
        factor_args["参数"] = Args
        if mask is None:
            return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(mask, *factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class ToJson(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "tojson", "入参数": 1, "最大入参数": 1, "数据类型": "string", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).applymap(lambda v: json.dumps(v, ensure_ascii=False) if pd.notnull(v) else None).values
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class ToCompound(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "toCompound", "入参数": 1, "最大入参数": -1, "数据类型": "object", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = {i: iData for i, iData in enumerate(x)}
        return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: tuple(s), axis=1).unstack().values
    
    def __call__(self, *factors, fields=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        if fields is None:
            DataTypes = [(iFactor.Name, iFactor.getMetaData(key="DataType")) for iFactor in factors]
        else:
            DataTypes = [(factors[i].Name if not iField else iField, factors[i].getMetaData(key="DataType")) for i, iField in enumerate(fields)]
        return super().__call__(*factors, args={"复合类型": DataTypes}, factor_name=factor_name, factor_args=factor_args, **kwargs)


# ----------------------时序运算--------------------------------
class Lag(TimeOperator):
    def __init__(self, lag_period=1, window=1, dt_change_fun=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "lag", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window], "参数": {"window": window, "lag_period": lag_period, "dt_change_fun": dt_change_fun}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0]
        if args.get('dt_change_fun', None) is None: return Data[self.Args["回溯期数"][0]-args['lag_period']:Data.shape[0]-args['lag_period']]
        TargetDTs = args['dt_change_fun'](idt)
        Data = pd.DataFrame(Data, index=idt)
        TargetData = Data.reindex(index=TargetDTs).values
        TargetData[args['lag_period']:] = TargetData[:-args['lag_period']]
        if f.DataType!="double":
            Data = pd.DataFrame(np.empty(Data.shape,dtype="O"),index=Data.index,columns=iid)
        else:
            Data = pd.DataFrame(index=Data.index,columns=iid,dtype="float")
        Data.loc[TargetDTs] = TargetData
        return Data.fillna(method='pad').values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]]
        factor_args["参数"] = Args
        DataType = f.getMetaData(key="DataType")
        if DataType==self._QSArgs["数据类型"]:
            return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, args={"数据类型": DataType}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingSum(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingSum", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).rolling(**args).sum().values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] - 1 > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]-1]
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingMax(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingMax", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).rolling(**args).max().values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] - 1 > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]-1]
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingMin(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingMin", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).rolling(**args).min().values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] - 1 > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]-1]
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingRank(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, ascending:bool=True, uniformization:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingRank", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type, "ascending": ascending, "uniformization": uniformization}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        args = args.copy()
        Data = pd.DataFrame(x[0])
        if not args.pop("ascending"):
            Data = - Data
        Uniformization = args.pop("uniformization")
        Rslt = Data.rolling(**args).rank().values[self.Args["回溯期数"][0]:] - 1
        if Uniformization:
            Rslt = Rslt / Data.rolling(**args).count().values[self.Args["回溯期数"][0]:]
        return Rslt
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] - 1 > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]-1]
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingMean(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, weights=None, sys_args={}, config_file=None, **kwargs):
        if weights is not None: window = len(weights)
        Args = {"名称": "rollingMean", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type, "weights": weights}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        if not args["weights"]:
            return Data.rolling(**args).mean().values[self.Args["回溯期数"][0]:]
        else:
            Args = args.copy()
            weights = np.array(Args.pop("weights"))
            return Data.rolling(**Args).apply(lambda x: np.nansum(x * weights) / np.nansum(pd.notnull(x) * weights), raw=True).values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] - 1 > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]-1]
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


class RollingStd(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, ddof=1, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingStd", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type, "ddof": ddof}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        args = args.copy()
        ddof = args.pop("ddof")
        return Data.rolling(**args).apply(lambda x:np.nanstd(x, ddof=ddof), raw=True).values[self.Args["回溯期数"][0]:]
        
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] - 1 > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]-1]
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingChangeRate(TimeOperator):
    def __init__(self, window:int=1, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingChangeRate", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [window-1], "参数": {"window": window}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
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
        return Rslt
        
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        if ("回溯期数" not in factor_args) and (Args["window"] - 1 > self._QSArgs["回溯期数"][0]): factor_args["回溯期数"] = [Args["window"]-1]
        factor_args["参数"] = Args
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingRegress(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, intercept=True, output:Optional[str]=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingRegress", "入参数": 1, "最大入参数": -1, "数据类型": "double", "运算时点": "单时点", "运算ID": "单ID", "回溯期数": [window-1], "参数": {"window": window, "min_periods": min_periods, "intercept": intercept, "output": output}}
        Args.update(sys_args)
        if output is None:
            Args["数据类型"] = "object"
            Args["复合类型"] = [("alpha", "double"), ("beta", "double")]
        else:
            Args["数据类型"] = "double"        
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
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
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        Factors = [endog] + exog
        if "回溯期数" not in factor_args:
            if Args["window"] - 1 > self._QSArgs["回溯期数"][0]: factor_args["回溯期数"] = [Args["window"]-1]
            else: factor_args["回溯期数"] = [self._QSArgs["回溯期数"][0]] * len(Factors)
        Outputs = [None, "alpha"] + [f"beta{i}" for i in range(len(exog))] if exog else (None, "alpha", "beta")
        if Args["output"] not in Outputs:
            raise __QS_Error__(f"算子 RollingRegress 的参数 output 只能取值为 {Outputs}, 不支持 {Args['output']}")
        factor_args["参数"] = Args
        DataType = ("object" if Args["output"] is None else "double")
        if DataType!=self._QSArgs.DataType:
            OpArgs = {"数据类型": DataType}
            if DataType=="object": OpArgs["复合类型"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(len(exog))]
            return super().__call__(*Factors, args=OpArgs, factor_name=factor_name, factor_args=factor_args, **kwargs)
        elif DataType=="object":
            CompoundType = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(len(exog))]
            if CompoundType!=self._QSArgs.CompoundType:
                return super().__call__(*Factors, args={"复合类型": CompoundType}, factor_name=factor_name, factor_args=factor_args, **kwargs)                
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

# ----------------------截面运算--------------------------------
class SectionRank(SectionOperator):
    def __init__(self, ascending:bool=True, uniformization:bool=True, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rankSection", "入参数": 1, "最大入参数": 3, "数据类型": "double", "运算时点": "多时点", "输出形式": "全截面", "参数": {"uniformization": uniformization, "ascending": ascending, "mask": False, "cat_data": False}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        FactorData, args = x[0], args.copy()
        Mask = (x[1].astype(bool) if args.pop("mask") else [None] * FactorData.shape[0])
        CatData = (x[-1] if args.pop("cat_data") else [None] * FactorData.shape[0])
        Rslt = np.full_like(FactorData, fill_value=np.nan)
        for i in range(FactorData.shape[0]):
            Rslt[i] = DataPreprocessingFun.standardizeRank(FactorData[i], mask=Mask[i], cat_data=CatData[i], perturbation=False, offset=0, **args)
        return Rslt
    
    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, *, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        Args["mask"] = (mask is not None)
        Args["cat_data"] = (cat_data is not None)
        factor_args["参数"] = Args
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Aggregate(SectionOperator):
    def __init__(self, aggr_fun=np.nansum, descriptor_ids=None, dtype="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "aggregate", "入参数": 1, "最大入参数": 3, "数据类型": dtype, "运算时点": "单时点", "输出形式": "全截面", "描述子截面": [descriptor_ids], "参数": {"aggr_fun": aggr_fun, "mask": False, "cat_data": False, "dtype": dtype, "section_chged": (descriptor_ids is not None)}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f, idt, iid, x, args):
        nID = len(iid)
        FactorData = x[0]
        if args["mask"]:
            Mask = (x[1]==1)
        else:
            Mask = np.full(FactorData.shape, fill_value=True)
        AggrFun = args["aggr_fun"]
        if args["cat_data"]:
            CatData = x[-1]
            Rslt = np.full(shape=(nID, ), fill_value=np.nan)
            if args["section_chged"]:
                for i, iID in enumerate(iid):
                    iMask = ((CatData==iID) & Mask)
                    Rslt[i] = AggrFun(FactorData[iMask])
            else:
                AllCats = pd.unique(CatData.flatten())
                for i, iCat in enumerate(AllCats):
                    if pd.isnull(iCat):
                        iMask = (pd.isnull(CatData) & Mask)
                    else:
                        iMask = ((CatData==iCat) & Mask)
                    Rslt[iMask] = AggrFun(FactorData[iMask])
        else:
            Rslt = np.full(shape=(nID, ), fill_value=AggrFun(FactorData[Mask]))
        return Rslt

    def __call__(self, f, mask=None, cat_data=None, *, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        Args["mask"] = (mask is not None)
        Args["cat_data"] = (cat_data is not None)
        factor_args["描述子截面"] = [kwargs.get("descriptor_ids", self._QSArgs.DescriptorSection[0])] * len(Factors)
        Args["section_chged"] = (factor_args["描述子截面"][0] is not None)
        factor_args["参数"] = Args
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(*Factors, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Disaggregate(SectionOperator):
    def __init__(self, aggr_ids, disaggr_ids=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "disaggregate", "入参数": 1, "最大入参数": 2, "数据类型": "double", "运算时点": "多时点", "输出形式": "全截面", "描述子截面": [aggr_ids, disaggr_ids], "参数": {"cat_data": False}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f, idt, iid, x, args):
        nDT, nID = len(idt), len(iid)
        FactorData = x[0]
        if args["cat_data"]:
            CatData = x[-1]
            Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
            for i, iID in enumerate(f.Args["描述子截面"][0]):
                iMask = (CatData==iID)
                Rslt[iMask] = FactorData[:, [i]].repeat(nID, axis=1)[iMask]
        else:
            Rslt = FactorData.repeat(nID, axis=1)
        return Rslt
    
    def __call__(self, f, cat_data=None, *, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if cat_data is not None: Factors.append(cat_data)
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        Args["cat_data"] = (cat_data is not None)
        if cat_data is not None:
            factor_args["描述子截面"] = [kwargs.get("aggr_ids", self._QSArgs.DescriptorSection[0]), kwargs.get("disaggr_ids", self._QSArgs.DescriptorSection[-1])]
        else:
            factor_args["描述子截面"] = [kwargs.get("aggr_ids", self._QSArgs.DescriptorSection[0])]
        factor_args["参数"] = Args
        DataType = f.getMetaData(key="DataType")
        if DataType!=self._QSArgs["数据类型"]:
            return super().__call__(*Factors, args={"数据类型": DataType}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class ConcatSection(SectionOperator):
    def __init__(self, descriptor_sections=None, dtype="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "concatSection", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "输出形式": "全截面", "描述子截面": descriptor_sections, "参数": {"dtype": dtype}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
        
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(np.concatenate(x, axis=1), columns=sum(((iid if iIDs is None else iIDs) for iIDs in f.Args["描述子截面"]), [])).reindex(columns=iid).values
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["描述子截面"] = kwargs.get("descriptor_sections", self._QSArgs.DescriptorSection)
        factor_args["参数"] = Args
        if len(factors)!=len(factor_args["描述子截面"]): raise __QS_Error__("算子 ConcatSection: 描述子个数与描述子截面个数不一致!")
        if factor_args["参数"]["dtype"]!=self._QSArgs["参数"]["dtype"]:
            return super().__call__(*factors, args={"数据类型": factor_args["参数"]["dtype"]}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)
       
class ChgSection(SectionOperator):
    # id_map: {新ID: 旧ID}
    def __init__(self, old_ids, id_map={}, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "chgSection", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "输出形式": "全截面", "描述子截面": [old_ids], "参数": {"id_map": id_map}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
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
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        factor_args["描述子截面"] = [kwargs.get("old_ids", self._QSArgs.DescriptorSection[0])]
        factor_args["参数"] = Args
        DataType = f.getMetaData(key="DataType")
        if DataType!=self._QSArgs["数据类型"]:
            return super().__call__(f, args={"数据类型": DataType}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class SectionRegress(SectionOperator):
    def __init__(self, intercept=True, output:Optional[str]=None, descriptor_ids=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "regressSection", "入参数": 1, "最大入参数": -1, "运算时点": "单时点", "输出形式": "全截面", "描述子截面": [descriptor_ids], "参数": {"intercept": intercept, "output": output}}
        Args.update(sys_args)
        if output is None:
            Args["数据类型"] = "object"
            Args["复合类型"] = [("alpha", "double"), ("beta", "double")]
        else:
            Args["数据类型"] = "double"        
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Y, X = x[0].astype(float), (np.array(x[1:], dtype=float).T if len(x)>1 else np.arange(0, x[0].shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        Y, X = Y[Mask], X[Mask]
        if args["intercept"]: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        if args["output"] is None: return [(tuple(Rslt.params) if args["intercept"] else (0, )+tuple(Rslt.params))] * len(iid)
        elif args["output"]=="alpha": Rslt = (Rslt.params[0] if args["intercept"] else 0)
        else: Rslt = (Rslt.params[int(args["output"][4:]) + int(args["intercept"])])
        return np.full(shape=(len(iid),), fill_value=Rslt)
        
    def __call__(self, endog:Factor, *exog, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        Factors = [endog] + exog
        Outputs = [None, "alpha"] + [f"beta{i}" for i in range(len(exog))] if exog else (None, "alpha", "beta")
        if Args["output"] not in Outputs:
            raise __QS_Error__(f"算子 SectionRegress 的参数 output 只能取值为 {Outputs}, 不支持 {Args['output']}")
        factor_args["参数"] = Args
        factor_args["描述子截面"] = [kwargs.get("descriptor_ids", self._QSArgs.DescriptorSection[0])] * len(Factors)
        DataType = ("object" if Args["output"] is None else "double")
        if DataType!=self._QSArgs.DataType:
            OpArgs = {"数据类型": DataType}
            if DataType=="object": OpArgs["复合类型"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(len(exog))]
            return super().__call__(*Factors, args=OpArgs, factor_name=factor_name, factor_args=factor_args, **kwargs)
        elif DataType=="object":
            CompoundType = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(len(exog))]
            if CompoundType!=self._QSArgs.CompoundType:
                return super().__call__(*Factors, args={"复合类型": CompoundType}, factor_name=factor_name, factor_args=factor_args, **kwargs)                
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

# ----------------------面板运算--------------------------------
class PanelRegress(PanelOperator):
    def __init__(self, window:int=1, intercept=True, output:Optional[str]=None, descriptor_ids=None, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "regressPanel", "入参数": 1, "最大入参数": -1, "运算时点": "单时点", "输出形式": "全截面", "回溯期数": [window-1], "描述子截面": [descriptor_ids], "参数": {"window": window, "intercept": intercept, "output": output}}
        Args.update(sys_args)
        if output is None:
            Args["数据类型"] = "object"
            Args["复合类型"] = [("alpha", "double"), ("beta", "double")]
        else:
            Args["数据类型"] = "double"        
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Y = x[0].astype(float).flatten()
        X = (np.array(x[1:], dtype=float).reshape((len(x)-1, -1)).T if len(x)>1 else np.arange(0, Y.shape[0]).reshape((-1, 1)))
        Mask = (~ (np.isnan(Y) | np.any(np.isnan(X), axis=1)))
        Y, X = Y[Mask], X[Mask]
        if args["intercept"]: X = sm.add_constant(X, prepend=True)
        Rslt = sm.OLS(Y, X).fit()
        if args["output"] is None: return [(tuple(Rslt.params) if args["intercept"] else (0, )+tuple(Rslt.params))] * len(iid)
        elif args["output"]=="alpha": Rslt = (Rslt.params[0] if args["intercept"] else 0)
        else: Rslt = (Rslt.params[int(args["output"][4:]) + int(args["intercept"])])
        return np.full(shape=(len(iid),), fill_value=Rslt)
        
    def __call__(self, endog:Factor, *exog, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = self._QSArgs["参数"].copy()
        Args.update({iKey: kwargs[iKey] for iKey in Args if iKey in kwargs})
        factor_args = factor_args.copy()
        Args.update(factor_args.get("参数", {}))
        Factors = [endog] + exog
        if "回溯期数" not in factor_args:
            if Args["window"] - 1 > self._QSArgs["回溯期数"][0]: factor_args["回溯期数"] = [Args["window"]-1]
            else: factor_args["回溯期数"] = [self._QSArgs["回溯期数"][0]] * len(Factors)
        Outputs = [None, "alpha"] + [f"beta{i}" for i in range(len(exog))] if exog else (None, "alpha", "beta")
        if Args["output"] not in Outputs:
            raise __QS_Error__(f"算子 PanelRegress 的参数 output 只能取值为 {Outputs}, 不支持 {Args['output']}")
        factor_args["参数"] = Args
        factor_args["描述子截面"] = [kwargs.get("descriptor_ids", self._QSArgs.DescriptorSection[0])] * len(Factors)
        DataType = ("object" if Args["output"] is None else "double")
        if DataType!=self._QSArgs.DataType:
            OpArgs = {"数据类型": DataType}
            if DataType=="object": OpArgs["复合类型"] = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(len(exog))]
            return super().__call__(*Factors, args=OpArgs, factor_name=factor_name, factor_args=factor_args, **kwargs)
        elif DataType=="object":
            CompoundType = [("alpha", "double")] + [(f"beta{i}", "double") for i in range(len(exog))]
            if CompoundType!=self._QSArgs.CompoundType:
                return super().__call__(*Factors, args={"复合类型": CompoundType}, factor_name=factor_name, factor_args=factor_args, **kwargs)                
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


if __name__=="__main__":
    from QuantStudio.FactorDataBase.FactorDB import DataFactor
    
    np.random.seed(0)
    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Factor1 = DataFactor(name="Factor1", data=1)
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    
    rolling_sum = RollingSum(window=3, min_periods=3)
    rank_section = RankSection()    
    
    Factor3 = Log(base=np.e)(Factor2, factor_name="Factor3")
    
    Factor4 = RollingSum(window=2, min_periods=2)(Factor2, factor_name="Factor4")
    Factor5 = rolling_sum(Factor2, factor_name="Factor5")
    Factor7 = rank_section(Factor2, Factor1, Factor1, factor_name="Factor7")
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    #print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    #print(Factor6.readData(ids=IDs, dts=DTs))
    print(Factor7.readData(ids=IDs, dts=DTs))
    
    print("===")
