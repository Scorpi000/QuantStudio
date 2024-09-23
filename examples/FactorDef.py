# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd

import QuantStudio.api as QS
import QuantStudio.FactorDataBase.FactorOperators as fd
from QuantStudio.FactorDataBase.FactorDB import DataFactor, Factorize
from QuantStudio.FactorDataBase.FactorOperation import makeFactorOperator, FactorOperatorized


# 构造用于测试的基础因子
np.random.seed(0)
IDs = [f"00000{i}.SZ" for i in range(1, 6)]
DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(7)]
Open = DataFactor(name="Open", data=pd.DataFrame(np.random.rand(len(DTs), len(IDs)) * 10, index=DTs, columns=IDs))
Close = DataFactor(name="Close", data=pd.DataFrame(np.random.rand(len(DTs), len(IDs)) * 10, index=DTs, columns=IDs))
Volume = DataFactor(name="Volume", data=pd.DataFrame(np.random.rand(len(DTs), len(IDs)) * 10000, index=DTs, columns=IDs))
Industry = DataFactor(name="Industry", data=pd.Series(np.random.choice(["Fin", "TMT", "Ind"], size=(len(IDs),)), index=IDs))
    
# 表达式方式
Mid = Factorize((Open + Close) / 2, factor_name="Mid")

# 内置算子
Vol_5d = fd.rolling_sum(Volume, window=5, min_priods=1, factor_name="Volume_5d")
    
# 自定义算子, 装饰器方式
@FactorOperatorized(operator_type="Time", sys_args={"名称": "rolling_5d_sum", "入参数": 1, "运算时点": "单时点", "运算ID": "多ID", "回溯期数": [5-1]})
def rolling5DSum(f, idt, iid, x, args):
    return np.nansum(x[0], axis=0)
Vol_5d_Decorator = rolling5DSum(Volume, factor_name="Volume_5d_Decorator")

# 自定义算子, 工厂函数方式
def calcRank(f, idt, iid, x, args):
    return np.argsort(np.argsort(x[0]))
calc_rank = makeFactorOperator(calcRank, operator_type="Section", sys_args={"名称": "rank", "入参数": 1, "运算时点": "单时点", "输出形式": "全截面"})    
MidRank = calc_rank(Mid, factor_name="MidRank")

# 自定义算子, 直接实例化方式, 不推荐
from QuantStudio.FactorDataBase.FactorOperation import SectionOperation
MidRank_Inst = SectionOperation(name="MidRank_Inst", descriptors=[Mid], sys_args={"算子": calcRank, "运算时点": "单时点"})

# 输出数据
print(Open.Name, Open.readData(ids=IDs, dts=DTs), sep="\n")
print(Close.Name, Close.readData(ids=IDs, dts=DTs), sep="\n")
print(Volume.Name, Volume.readData(ids=IDs, dts=DTs), sep="\n")
print(Mid.Name, Mid.readData(ids=IDs, dts=DTs), sep="\n")
print(Vol_5d.Name, Vol_5d.readData(ids=IDs, dts=DTs), sep="\n")
print(Vol_5d_Decorator.Name, Vol_5d_Decorator.readData(ids=IDs, dts=DTs), sep="\n")
print(MidRank.Name, MidRank.readData(ids=IDs, dts=DTs), sep="\n")
print(MidRank_Inst.Name, MidRank_Inst.readData(ids=IDs, dts=DTs), sep="\n")