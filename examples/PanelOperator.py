# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd

import QuantStudio.api as QS
import QuantStudio.FactorDataBase.FactorOperators as fo
from QuantStudio.FactorDataBase.FactorDB import DataFactor, Factorize
from QuantStudio.FactorDataBase.FactorOperation import makeFactorOperator, FactorOperatorized

# 构造用于测试的基础因子
np.random.seed(0)
IDs = [f"00000{i}.SZ" for i in range(1, 6)]
SectionIDs = IDs + [f"00000{i}.SZ" for i in range(6, 10)]
DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(7)]
DTRuler = [dt.datetime(2020, 1, 1) - dt.timedelta(i+1) for i in range(7)] + DTs
Open = DataFactor(name="Open", data=pd.DataFrame(np.random.rand(len(DTs), len(IDs)) * 10, index=DTs, columns=IDs))
Close = DataFactor(name="Close", data=pd.DataFrame(np.random.rand(len(DTs), len(IDs)) * 10, index=DTs, columns=IDs))
Volume = DataFactor(name="Volume", data=pd.DataFrame(np.random.rand(len(DTs), len(IDs)) * 10000, index=DTs, columns=IDs))
Industry = DataFactor(name="Industry", data=pd.Series(np.random.choice(["Fin", "TMT", "Ind"], size=(len(IDs),)), index=IDs))

# 滚动窗口
@FactorOperatorized(operator_type="Panel", sys_args={"名称": "rolling_sum", "入参数": 1, "运算时点": "单时点", "输出形式": "全截面"})
def rollingSum(f, idt, iid, x, args):
    return np.nansum(x[0], axis=0)
Vol5d = rollingSum(Volume, factor_name="Volume_5d", factor_args={"回溯期数": [5-1]})

# 扩张窗口
@FactorOperatorized(operator_type="Panel", sys_args={"名称": "expanding_sum", "入参数": 2, "运算时点": "单时点", "输出形式": "全截面"})
def expandingSum(f, idt, iid, x, args):
    return x[0][-1] + x[1][0]
CumVol = expandingSum(Volume, 0, factor_name="Volume_Cum", factor_args={"回溯期数": [5-1, 2-1], "起始因子": 1})


if __name__=="__main__":
    # 输出数据
    print(Volume.Name, Volume.readData(ids=IDs, dts=DTs), sep="\n")
    print(Vol5d.Name, Vol5d.readData(ids=IDs, dts=DTs), sep="\n")
    print(CumVol.Name, CumVol.readData(ids=IDs, dts=DTs, dt_ruler=DTRuler, section_ids=SectionIDs), sep="\n")
    
    print("===")