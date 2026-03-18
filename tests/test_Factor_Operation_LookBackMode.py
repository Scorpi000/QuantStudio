# -*- coding: utf-8 -*-
"""测试因子运算的回溯模式"""
import os
import datetime as dt
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

import numpy as np
import pandas as pd

from QuantStudio.Core.CalcEngine import Engine, ParallelEngine
from QuantStudio.Factor.Factor import FactorContext, DataFactor, FactorLocalContext, FactorInitData
from QuantStudio.Factor.FactorOperation import FactorOperatorized
from QuantStudio.Factor.FactorCache import FeatherFactorCache


np.random.seed(0)
nDT, nID = 23, 3
IDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]


InitData = DataFactor(data=1, args={"Name": "init"})
# Open = DataFactor(data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTRuler, columns=IDs), args={"Name": "open"})
# Close = DataFactor(data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTRuler, columns=IDs), args={"Name": "close"})
Open = DataFactor(data=2, args={"Name": "open"})
Close = DataFactor(data=3, args={"Name": "close"})

# ------------------------------ PanelOperation -------------------------------------
# # 无自身迭代
# @FactorOperatorized(operator_type="Panel", args={
#     "Arity": 3, "DTMode": "单时点", "OutputMode": "全截面",
#     "LookBack": [0, 2, 2], 
#     "StartDT": [dt.datetime(2025, 1, 5), dt.datetime(2025, 1, 4), None], 
#     "iInitFactor": -1
# })
# def PanelFunc(f, idt, iid, x, args):
#     print("="*20)
#     print(idt[-1], x[0].shape[0])
#     print(x[0])
#     return np.zeros((len(iid), ))

# 自身迭代且为扩展窗口
@FactorOperatorized(operator_type="Panel", args={
    "Arity": 3, "DTMode": "单时点", "OutputMode": "全截面",
    "LookBack": [1, 2, 2], 
    "StartDT": [dt.datetime(2025, 1, 5), dt.datetime(2025, 1, 4), None], 
    "iInitFactor": 0
})
def PanelFunc(f, idt, iid, x, args):
    print("="*20)
    print(idt[-1], x[0].shape[0])
    print(x[0])
    return np.zeros((len(iid), ))

# # 自身迭代且为滚动窗口
# @FactorOperatorized(operator_type="Panel", args={
#     "Arity": 3, "DTMode": "单时点", "OutputMode": "全截面",
#     "LookBack": [1, 2, 3], 
#     "StartDT": [None, dt.datetime(2025, 1, 4), None], 
#     "iInitFactor": 0
# })
# def PanelFunc(f, idt, iid, x, args):
#     print("="*20)
#     print(idt[-1], x[0].shape[0])
#     print(x[0])
#     return np.zeros((len(iid), ))

PanelFactor = PanelFunc(InitData, Open, Close, factor_args={"Name": "PanelFactor"})
NodeList = [PanelFactor]

# ------------------------------ TimeOperation -------------------------------------
# # 无自身迭代
# @FactorOperatorized(operator_type="Time", args={
#     "Arity": 3, "DTMode": "单时点", "IDMode": "多ID",
#     "LookBack": [0, 2, 3], 
#     "StartDT": [dt.datetime(2025, 1, 5), dt.datetime(2025, 1, 4), None], 
#     "iInitFactor": -1
# })
# def TimeFunc(f, idt, iid, x, args):
#     print("="*20)
#     print(idt[-1], x[0].shape[0])
#     print(x[0])
#     return np.zeros((len(iid), ))

# # 自身迭代且为扩展窗口
# @FactorOperatorized(operator_type="Time", args={
#     "Arity": 3, "DTMode": "单时点", "IDMode": "多ID",
#     "LookBack": [1, 2, 2], 
#     "StartDT": [dt.datetime(2025, 1, 5), dt.datetime(2025, 1, 4), None], 
#     "iInitFactor": 0
# })
# def TimeFunc(f, idt, iid, x, args):
#     print("="*20)
#     print(idt[-1], x[0].shape[0])
#     print(x[0])
#     return np.zeros((len(iid), ))

# # 自身迭代且为滚动窗口
# @FactorOperatorized(operator_type="Time", args={
#     "Arity": 3, "DTMode": "单时点", "IDMode": "多ID",
#     "LookBack": [1, 2, 2], 
#     "StartDT": [None, dt.datetime(2025, 1, 4), None], 
#     "iInitFactor": 0
# })
# def TimeFunc(f, idt, iid, x, args):
#     print("="*20)
#     print(idt[-1], x[0].shape[0])
#     print(x[0])
#     return np.zeros((len(iid), ))

# TimeFactor = TimeFunc(InitData, Open, Close, factor_args={"Name": "TimeFactor"})
# NodeList = [TimeFactor]


with FeatherFactorCache(args={"DTRuler": DTRuler, "CacheDir": "/mnt/d/Data/Cache/DevCache", "StartMode": "new"}) as Cache:
    with FactorContext(DTRuler=DTRuler, DefaultSectionIDs=IDs, FactorDataCache=Cache) as Context:
        with Engine() as ExecEngine:
            DTs = DTRuler[2:-3]
            Rslt = ExecEngine.run(
                NodeList, 
                Context, 
                fwd_data_list=[FactorLocalContext(DTs=DTs, IDs=IDs)] * len(NodeList), 
                init_data_list=[FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=IDs)] * len(NodeList)
            )
            # 测试在有缓存的情况下能否接续计算
            DTs = DTRuler[-3:]
            Rslt1 = ExecEngine.run(
                NodeList, 
                Context, 
                fwd_data_list=[FactorLocalContext(DTs=DTs, IDs=IDs)] * len(NodeList), 
                init_data_list=[FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=IDs)] * len(NodeList)
            )


print(Rslt[0])
print(Rslt1[0])
print("===")