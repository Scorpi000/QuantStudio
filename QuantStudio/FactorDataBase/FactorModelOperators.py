# coding=utf-8
"""多因子模型相关的因子算子"""
import datetime as dt
from typing import Optional, Dict

import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import SectionOperator

class Corr(SectionOperator):
    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "calcCorr", "入参数": 2, "最大入参数": -1, "运算时点": "单时点", "输出形式": "全截面", "输入格式": "pandas"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        Return = x[0].pop("d0")
        if args["mask"]: 
            Mask = x[0].pop("d1").astype(bool)
            Return[~Mask] = np.nan
        Rslt = x[0].corrwith(Return, method=args["corr_method"])
        Rslt.index = iid
        return Rslt
    
    def __call__(self, f:Factor, *factors, mask:Optional[Factor]=None, descriptor_ids=None, corr_method:str="spearman", factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = [f]
        if mask is not None: Factors.append(mask)
        if factors: Factors += factors
        else: raise __QS_Error__(f"算子 {self.__class__}: 必须至少指定一个因子!")
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"mask": (mask is not None), "corr_method": corr_method})
        if descriptor_ids is not None: factor_args["描述子截面"] = [descriptor_ids] * len(Factors)
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


class IC(Corr):
    class __QS_ArgClass__(Corr.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "calcIC"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def __call__(self, rtn:Factor, *factors, mask:Optional[Factor]=None, descriptor_ids=None, corr_method:str="spearman", factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        return super().__call__(rtn, *factors, mask=mask, descriptor_ids=descriptor_ids, corr_method=corr_method, factor_name=factor_name, factor_args=factor_args, **kwargs)

class Breadth(SectionOperator):
    class __QS_ArgClass__(SectionOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "calcBreadth", "入参数": 1, "最大入参数": -1, "运算时点": "多时点", "输出形式": "全截面"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        if args["mask"]: 
            Rslt = np.array(x[:-1])
            Mask = x[-1].astype(bool)
            Rslt[:, ~Mask] = np.nan
        else:
            Rslt = np.array(x)
        return np.sum(pd.notnull(Rslt), axis=2).T
    
    def __call__(self, *factors, mask:Optional[Factor]=None, descriptor_ids=None, factor_name:Optional[str]=None, factor_args:Dict={}, **kwargs):
        Factors = []
        if factors: Factors += factors
        else: raise __QS_Error__(f"算子 {self.__class__}: 必须至少指定一个因子!")
        if mask is not None: Factors.append(mask)
        factor_args = factor_args.copy()
        factor_args.setdefault("参数", {}).update({"mask": (mask is not None)})
        if descriptor_ids is not None: factor_args["描述子截面"] = [descriptor_ids] * len(Factors)
        return super().__call__(*Factors, args={}, factor_name=factor_name, factor_args=factor_args, **kwargs)


if __name__=="__main__":
    from QuantStudio.FactorDataBase.FactorDB import DataFactor
    
    np.random.seed(0)
    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Mask = DataFactor(name="Mask", data=pd.DataFrame(np.random.randint(0, 2, size=(len(DTs), len(IDs))).astype(bool), index=DTs, columns=IDs))
    Rtn = DataFactor(name="Return", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    Factor1 = DataFactor(name="Factor1", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    
    #FIC = IC()(Rtn, Factor1, Factor2, descriptor_ids=IDs, factor_name="IC")
    #print(FIC.readData(ids=["Factor1", "Factor2"], dts=DTs))
    
    FBreadth = Breadth()(Factor1, Factor2, mask=Rtn, descriptor_ids=IDs, factor_name="Breadth")
    print(FBreadth.readData(ids=["Factor1", "Factor2"], dts=DTs))
    
    print("===")
    