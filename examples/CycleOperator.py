# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd

import QuantStudio.api as QS
import QuantStudio.FactorDataBase.FactorOperators as fo
from QuantStudio.FactorDataBase.FactorDB import DataFactor, Factorize
from QuantStudio.FactorDataBase.FactorOperation import makeFactorOperator, FactorOperatorized

# 带环的因子定义

# 构造用于测试的基础因子
np.random.seed(0)
IDs = [f"00000{i}.SZ" for i in range(1, 6)]
DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(7)]
DTRuler = [dt.datetime(2020, 1, 1) - dt.timedelta(i+1) for i in range(7)] + DTs
f0 = DataFactor(name="f0", data=1)

@FactorOperatorized(operator_type="Time", sys_args={"名称": "func1", "入参数": 1, "运算时点": "单时点", "运算ID": "多ID"})
def func1(f, idt, iid, x, args):
    if x[0].shape[0]==0:
        return np.ones(())
f1 = func1(f0, factor_name="f1", factor_args={"回溯期数": [1-1]})

@FactorOperatorized(operator_type="Point", sys_args={"名称": "func2", "入参数": 1, "运算时点": "多时点", "运算ID": "多ID"})
def func2(f, idt, iid, x, args):
    return np.nansum(x[0], axis=0)
f2 = func2(f1, factor_name="f2", factor_args={})
f1._Descriptor = [f2]

#@FactorOperatorized(operator_type="Point", sys_args={"名称": "func3", "入参数": 1, "运算时点": "多时点", "运算ID": "多ID"})
#def func3(f, idt, iid, x, args):
    #return np.nansum(x[0], axis=0)
#f3 = func3(f2, factor_name="f3", factor_args={})



if __name__=="__main__":
    # 输出数据
    #print(f1.Name, f1.readData(ids=IDs, dts=DTs, dt_ruler=DTRuler), sep="\n")
    #print(f2.Name, f2.readData(ids=IDs, dts=DTs), sep="\n")
    print(f3.Name, f3.readData(ids=IDs, dts=DTs), sep="\n")
    
    print("===")