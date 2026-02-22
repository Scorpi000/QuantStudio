import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core.Node import Node
from QuantStudio.Core.CalcEngine import Engine, ParallelEngine
from QuantStudio.Core.Factor import DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Core.FactorCache import FeatherCache
from QuantStudio.BackTest.BackTestModel import BTInitData, BTLocalContext
from QuantStudio.BackTest.SectionFactor.IC import IC, ICOutput, ICReport


if __name__ == "__main__":
    np.random.seed(0)
    nDT, nID = 10, 5
    SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
    DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
    IDs, DTs = SectionIDs[:3], DTRuler[-5:]
    
    Factor1 = DataFactor(data=1, args={"Name": "Factor1"})
    Factor2 = DataFactor(data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs), args={"Name": "Factor2"})
    FactorIC = IC(lookback=1)(Factor1, price=Factor2)
    BTModule = ICReport(ICOutput(FactorIC, args={"RollingAvgPeriod": 2}))
    
    ExecEngine = Engine()
    Cache = FeatherCache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "CacheDir": Path("~/Project/Data/FactorCache"), "PIDs": ["0"]})
    Cache.start()
    Context = FactorContext(
        PID="0",
        PIDList=["0"],
        DTRuler=DTRuler,
        DefaultSectionIDs=SectionIDs,
        IDSplit="连续切分",
        FactorDataCache=Cache
    )
    NodeList = [Factor2, BTModule]
    FwdDataList = [FactorLocalContext(DTs=DTs, IDs=IDs), BTLocalContext(DTs=DTs)]
    InitDataList = [FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=SectionIDs), BTInitData(DTRange=(DTs[0], DTs[-1]))]
    Rslt = ExecEngine.run(NodeList, Context, fwd_data_list=FwdDataList, init_data_list=InitDataList)
    
    print(Rslt)
    
    print("===")
