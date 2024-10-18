# coding=utf-8
"""内置的因子运算"""
import json
import datetime as dt
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PointOperator, TimeOperator, SectionOperator
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
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class IsNull(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "isnull", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.isnull(x[0])
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
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

class Sign(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "sign", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.sign(x[0].astype(float))
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Ceil(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "ceil", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.ceil(x[0].astype(float))
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Floor(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "floor", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.floor(x[0].astype(float))
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Fix(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "fix", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.fix(x[0].astype(float))
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Clip(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "clip", "入参数": 1, "最大入参数": 3, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.clip(x[0].astype(float), x[1].astype(float), x[2].astype(float))
    
    def __call__(self, f, a_min, a_max, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, a_min, a_max, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class IsIn(PointOperator):
    def __init__(self, test_elements=[], sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "isin", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {"test_elements": test_elements}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.isin(x[0], args["test_elements"])
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Applymap(PointOperator):
    def __init__(self, func=id, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "applymap", "入参数": 1, "最大入参数": 1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"func": func}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).applymap(args["func"]).values
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Where(PointOperator):
    def __init__(self, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "where", "入参数": 3, "最大入参数": 3, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return np.where(x[1], x[0], x[2])
    
    def __call__(self, f, mask, other, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, mask, other, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Fetch(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "fetch", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
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
        CompoundType = getattr(f._QSArgs, "CompoundType", None)
        if CompoundType:
            if isinstance(pos, str):
                dtype = dict(CompoundType)[pos]
            else:
                pos, dtype = CompoundType[int(pos)]
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"pos": pos, "compound_type": CompoundType})
        return super().__call__(f, args=({} if dtype=="double" else {"数据类型": dtype}), factor_name=factor_name, factor_args=factor_args, **kwargs)

class Strftime(PointOperator):
    def __init__(self, dt_format:str="%Y%m%d", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "strftime", "入参数": 1, "最大入参数": 1, "数据类型": "string", "运算时点": "多时点", "运算ID": "多ID", "参数": {"dt_format": dt_format}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0]
        DTFormat = args["dt_format"]
        return pd.DataFrame(Data).applymap(lambda x: x.strftime(DTFormat) if pd.notnull(x) else None).values
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Strptime(PointOperator):
    def __init__(self, dt_format:str="%Y%m%d", is_datetime:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "strptime", "入参数": 1, "最大入参数": 1, "数据类型": "object", "运算时点": "多时点", "运算ID": "多ID", "参数": {"dt_format": dt_format, "is_datetime": is_datetime}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0]
        DTFormat = args["dt_format"]
        if args["is_datetime"]:
            return pd.DataFrame(Data).applymap(lambda x: dt.datetime.strptime(x, DTFormat) if pd.notnull(x) else None).values
        else:
            return pd.DataFrame(Data).applymap(lambda x: dt.datetime.strptime(x, DTFormat).date() if pd.notnull(x) else None).values
    
    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


class FromTimestamp(PointOperator):
    def __init__(self, unit:float=1e9, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "fromtimestamp", "入参数": 1, "最大入参数": 1, "数据类型": "object", "运算时点": "多时点", "运算ID": "多ID", "参数": {"unit": unit}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).applymap(lambda x: dt.datetime.fromtimestamp(x / args["unit"]) if pd.notnull(x) else None).values

    def __call__(self, f, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class NanSum(PointOperator):
    def __init__(self, all_nan:float=0, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "nansum", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": all_nan}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nansum(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class NanProd(PointOperator):
    def __init__(self, all_nan:float=1, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "nanprod", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": all_nan}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanprod(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class NanMax(PointOperator):
    def __init__(self, all_nan:float=np.nan, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "nanmax", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": np.nan}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanmax(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class NanMin(PointOperator):
    def __init__(self, all_nan:float=np.nan, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "nanmin", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": np.nan}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanmin(Data, axis=0)
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, all_nan:float=np.nan, dtype:str="double", factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class NanMean(PointOperator):
    def __init__(self, weights=None, ignore_nan_weight=True, dtype:str="double", sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "nanmean", "入参数": 1, "最大入参数": -1, "数据类型": dtype, "运算时点": "多时点", "运算ID": "多ID", "参数": {"weights": weights, "ignore_nan_weight": ignore_nan_weight}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(Data)
        Weights = args["weights"]
        if Weights is None:
            if args["ignore_nan_weight"]:
                return np.nanmean(Data,axis=0)
            Weights = [1] * len(Data)
        Rslt = np.zeros(Data.shape[1:])
        WeightArray = np.zeros(Data[0].shape)
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
            return Rslt / len(Data)
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class NanStd(PointOperator):
    def __init__(self, ddof=1, all_nan:float=np.nan, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "nanstd", "入参数": 1, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {"all_nan": all_nan, "ddof": ddof}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = np.array(x)
        Rslt = np.nanstd(Data, axis=0, ddof=args["ddof"])
        Mask = (np.sum(pd.notnull(Data), axis=0)==0)
        Rslt[Mask] = args["all_nan"]
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RegressChangeRate(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "regress_change_rate", "入参数": 2, "最大入参数": -1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        Y = np.array(x)
        X = np.arange(Y.shape[0]).astype("float").reshape((Y.shape[0], 1, 1)).repeat(Y.shape[1], axis=1).repeat(Y.shape[2], axis=2)
        Denominator = np.abs(np.nanmean(Y, axis=0))
        X[pd.isnull(Y)] = np.nan
        X = X - np.nanmean(X, axis=0)
        Y = Y - np.nanmean(Y, axis=0)
        Numerator = np.nansum(X*Y, axis=0) / np.nansum(X**2, axis=0)
        Rslt = Numerator / Denominator
        Mask = (Denominator==0)
        Rslt[Mask] = np.sign(Numerator)[Mask]
        Rslt[np.isinf(Rslt)] = np.nan
        return Rslt
    
    def __call__(self, *factors, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(*factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class ToList(PointOperator):
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "tolist", "入参数": 1, "最大入参数": -1, "数据类型": "object", "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
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
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"mask": (mask is not None)})
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
    def __init__(self, lag_period=1, window=1, dt_change_fun=None, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "lag", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window, "lag_period": lag_period, "dt_change_fun": dt_change_fun}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window]
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
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Diff(TimeOperator):
    def __init__(self, n:int=1, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "diff", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1], "参数": {"n": int(n)}}
        if auto_lookback:
            Args["回溯期数"] = [int(n)]
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        nDT = x[0].shape[0] - self.Args["回溯期数"][0]
        return np.diff(x[0], n=args["n"], axis=0)[-nDT:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class FillNa(TimeOperator):
    def __init__(self, value=None, lookback=1, sys_args={}, config_file=None, **kwargs):
        if value is None:
            Args = {"名称": "fillna", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [lookback], "参数": {"fill_value": False, "lookback": lookback}}
        else:
            Args = {"名称": "fillna", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [0, 0], "参数": {"fill_value": True, "lookback": lookback}}
        Args.update(sys_args)
        self._Value = value
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data, Val = x
        if not args["fill_value"]:
            LookBack = args["lookback"]
            return pd.DataFrame(Data).fillna(method="pad", limit=LookBack).values[LookBack:]
        else:
            return np.where(pd.notnull(Data), Data, Val)
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        if self._Value is None:
            return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, self._Value, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingSum(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingSum", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window-1]
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        return Data.rolling(**args).sum().values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingProd(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingProd", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "单时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window, "min_periods": min_periods}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window-1]
        Args.update(args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Rslt = np.nanprod(x[0], axis=0)
        Rslt[np.sum(pd.notnull(x[0]), axis=0) < args["min_periods"]] = np.nan
        return Rslt
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingMean(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, weights=None, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        if weights is not None: window = len(weights)
        Args = {"名称": "rollingMean", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window-1]
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        if "weights" not in args:
            return Data.rolling(**args).mean().values[self.Args["回溯期数"][0]:]
        else:
            Args = args.copy()
            weights = np.array(Args.pop("weights"))
            return Data.rolling(**Args).apply(lambda x: np.nansum(x * weights) / np.nansum(pd.notnull(x) * weights), raw=True).values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


class RollingStd(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, ddof=1, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingStd", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type, "ddof": ddof}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window-1]
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        args = args.copy()
        ddof = args.pop("ddof")
        return Data.rolling(**args).apply(lambda x:np.nanstd(x, ddof=ddof), raw=True).values[self.Args["回溯期数"][0]:]
        
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingSkew(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingSkew", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window-1]
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).rolling(**args).skew().values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingKurt(TimeOperator):
    def __init__(self, window:int=1, min_periods:int=1, win_type:Optional[str]=None, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingKurt", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window, "min_periods": min_periods, "win_type": win_type}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window-1]
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return pd.DataFrame(x[0]).rolling(**args).kurt().values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class RollingChangeRate(TimeOperator):
    def __init__(self, window:int=1, auto_lookback:bool=True, sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "rollingChangeRate", "入参数": 1, "最大入参数": 1, "数据类型": "double", "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": window}}
        if auto_lookback and (window is not None):
            Args["回溯期数"] = [window-1]
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
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


# ----------------------截面运算--------------------------------
class StandardizeRank(SectionOperator):
    def __init__(self, ascending:bool=True, uniformization:bool=True, perturbation:bool=False, offset:float=0.5, other_handle:str='填充None', sys_args={}, config_file=None, **kwargs):
        Args = {"名称": "standardizeRank", "入参数": 1, "最大入参数": 3, "数据类型": "double", "运算时点": "多时点", "输出形式": "全截面", "参数": {"uniformization": uniformization, "perturbation": perturbation, "offset": offset, "other_handle": other_handle}}
        Args.update(sys_args)
        return super().__init__(sys_args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        FactorData, args = x[0], args.copy()
        Mask = (x[1].astype(bool) if args.pop("mask") else [None] * FactorData.shape[0])
        CatData = (x[-1] if args.pop("cat_data") else [None] * FactorData.shape[0])
        Rslt = np.full_like(FactorData, fill_value=np.nan)
        for i in range(FactorData.shape[0]):
            Rslt[i] = DataPreprocessingFun.standardizeRank(FactorData[i], mask=Mask[i], cat_data=CatData[i], **args)
        return Rslt
    
    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, *, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"mask": (mask is not None), "cat_data": (cat_data is not None)})
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


if __name__=="__main__":
    from QuantStudio.FactorDataBase.FactorDB import DataFactor
    
    np.random.seed(0)
    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Factor1 = DataFactor(name="Factor1", data=1)
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    
    rolling_sum = RollingSum(window=3, min_periods=3)
    standardize_rank = StandardizeRank()    
    
    Factor3 = Log(base=np.e)(Factor2, factor_name="Factor3")
    
    Factor4 = RollingSum(window=2, min_periods=2)(Factor2, factor_name="Factor4")
    Factor5 = rolling_sum(Factor2, factor_name="Factor5")
    Factor7 = standardize_rank(Factor2, Factor1, Factor1, factor_name="Factor7")
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    #print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    #print(Factor6.readData(ids=IDs, dts=DTs))
    print(Factor7.readData(ids=IDs, dts=DTs))
    
    print("===")
