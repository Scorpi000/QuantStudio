# coding=utf-8
"""内置的因子运算"""
import datetime as dt
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PointOperator, TimeOperator, SectionOperator
from QuantStudio.Tools import DataPreprocessingFun


# ----------------------单点运算--------------------------------
class Log(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "log", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID", "参数": {"base": np.e}}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0].astype(float)
        return np.log(np.where(Data>0, Data, np.nan)) / np.log(args["base"])
    
    def __call__(self, f, base:Optional[float]=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"base": base} if base is not None else {})
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Where(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "where", "入参数": 3, "最大入参数": 3, "运算时点": "多时点", "运算ID": "多ID", "参数": {}}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return np.where(x[1], x[0], x[2])
    
    def __call__(self, f, mask, other, dtype:str="double", factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(f, mask, other, args=({} if dtype=="double" else {"数据类型": dtype}), factor_name=factor_name, factor_args=factor_args, **kwargs)

class Fetch(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "fetch", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID", "参数": {"pos": 0}}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
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
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "strftime", "入参数": 1, "最大入参数": 1, "数据类型": "string", "运算时点": "多时点", "运算ID": "多ID", "参数": {"dt_format": "%Y%m%d"}}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0]
        DTFormat = args["dt_format"]
        return pd.DataFrame(Data).applymap(lambda x: x.strftime(DTFormat) if pd.notnull(x) else None).values
    
    def __call__(self, f, dt_format:str="%Y%m%d", factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"dt_format": dt_format})
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Strptime(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "strptime", "入参数": 1, "最大入参数": 1, "数据类型": "object", "运算时点": "多时点", "运算ID": "多ID", "参数": {"dt_format": "%Y%m%d"}}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        Data = x[0]
        DTFormat = args["dt_format"]
        if args["is_datetime"]:
            return pd.DataFrame(Data).applymap(lambda x: dt.datetime.strptime(x, DTFormat) if pd.notnull(x) else None).values
        else:
            return pd.DataFrame(Data).applymap(lambda x: dt.datetime.strptime(x, DTFormat).date() if pd.notnull(x) else None).values
    
    def __call__(self, f, dt_format:str="%Y%m%d", is_datetime:bool=True, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"dt_format": dt_format, "is_datetime": is_datetime})
        return super().__call__(f, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


# ----------------------时序运算--------------------------------
class RollingSum(TimeOperator):
    class __QS_ArgClass__(TimeOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "rollingSum", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID", "回溯期数": [1-1], "参数": {"window": 1, "min_periods": 1, "win_type": None}}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        Data = pd.DataFrame(x[0])
        return Data.rolling(**args).sum().values[self.Args["回溯期数"][0]:]
    
    def __call__(self, f:Factor, window:Optional[int]=None, min_periods:Optional[int]=None, win_type:Optional[str]=None, auto_lookback:bool=True, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Args = ({"回溯期数": [window-1]} if auto_lookback and (window is not None) else {})
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"window": window, "min_periods": min_periods, "win_type": win_type})
        return super().__call__(f, args=Args, factor_name=factor_name, factor_args=factor_args, **kwargs)


# ----------------------截面运算--------------------------------
class StandardizeRank(SectionOperator):
    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "standardizeRank", "入参数": 1, "最大入参数": 3, "运算时点": "多时点", "输出形式": "全截面"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        FactorData, args = x[0], args.copy()
        Mask = (x[1].astype(bool) if args.pop("mask") else [None] * FactorData.shape[0])
        CatData = (x[-1] if args.pop("cat_data") else [None] * FactorData.shape[0])
        Rslt = np.full_like(FactorData, fill_value=np.nan)
        for i in range(FactorData.shape[0]):
            Rslt[i] = DataPreprocessingFun.standardizeRank(FactorData[i], mask=Mask[i], cat_data=CatData[i], **args)
        return Rslt
    
    def __call__(self, f:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, *, ascending:bool=True, uniformization:bool=True, perturbation:bool=False, offset:float=0.5, other_handle:str='填充None', factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"mask": (mask is not None), "cat_data": (cat_data is not None), "ascending": ascending, "uniformization": uniformization, "perturbation": perturbation, "offset": offset, "other_handle": other_handle})
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


log = Log()
rolling_sum = RollingSum()
standardize_rank = StandardizeRank()

if __name__=="__main__":
    from QuantStudio.FactorDataBase.FactorDB import DataFactor
    
    np.random.seed(0)
    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Factor1 = DataFactor(name="Factor1", data=1)
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    
    Factor3 = Log()(Factor2, factor_name="Factor3")
    
    Factor4 = RollingSum()(Factor2, window=2, min_periods=2, factor_name="Factor4")
    Factor5 = rolling_sum(Factor2, window=2, min_periods=2, factor_name="Factor5")
    Factor6 = rolling_sum(Factor2, window=3, min_periods=3, factor_name="Factor6")
    Factor7 = standardize_rank(Factor2, Factor1, Factor1, factor_name="Factor7")
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    #print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    print(Factor6.readData(ids=IDs, dts=DTs))
    print(Factor7.readData(ids=IDs, dts=DTs))
    
    print("===")
