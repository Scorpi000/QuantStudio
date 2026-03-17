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
nDT, nID = 20, 3
IDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]


# Open = DataFactor(data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTRuler, columns=IDs), args={"Name": "open"})
# Close = DataFactor(data=pd.DataFrame(np.random.rand(nDT, nID) * 10, index=DTRuler, columns=IDs), args={"Name": "close"})
Open = DataFactor(data=2, args={"Name": "open"})
Close = DataFactor(data=3, args={"Name": "close"})
InitData = DataFactor(data=1, args={"Name": "init"})

@FactorOperatorized(operator_type="Panel", args={
    "Arity": 3, "DTMode": "单时点", "OutputMode": "全截面",
    "LookBack": [1, 2, 3], 
    "LookBackMode": ["扩张窗口", "扩张窗口", "滚动窗口"],
    "StartDT": [dt.datetime(2025, 1, 2), None, None], 
    "iInitFactor": 0
})
def PanelFunc(f, idt, iid, x, args):
    return np.zeros((len(iid), ))

PanelFactor = PanelFunc(InitData, Open, Close, factor_args={"Name": "PanelFactor"})

NodeList = [PanelFactor]
DTs = DTRuler[-5:]
with FeatherFactorCache(args={"DTRuler": DTRuler, "CacheDir": "/mnt/d/Data/Cache/DevCache", "StartMode": "new"}) as Cache:
    with FactorContext(DTRuler=DTRuler, DefaultSectionIDs=IDs, FactorDataCache=Cache) as Context:
        with Engine() as ExecEngine:
            Rslt = ExecEngine.run(
                NodeList, 
                Context, 
                fwd_data_list=[FactorLocalContext(DTs=DTs, IDs=IDs)] * len(NodeList), 
                init_data_list=[FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=IDs)] * len(NodeList)
            )

print(Rslt[0])
print("===")