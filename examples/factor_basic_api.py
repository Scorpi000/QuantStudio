# -*- coding: utf-8 -*-
"""因子基本API"""
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.HDF5DB import HDF5DB


if __name__ == "__main__":
    SDB = HDF5DB(args={}).connect()

    np.random.seed(0)
    nDT, nID = 10, 5
    SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
    DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
    IDs, DTs = SectionIDs[:3], DTRuler[-5:]

    Cache = FeatherFactorCache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "CacheDir": r"C:\Users\hst\Project\Data\FactorCache", "PIDs": ["0"], "StartMode": "new"})
    Cache.start()
    Context = FactorContext(
        PID="0",
        PIDList=["0"],
        DTRuler=DTRuler,
        SectionIDs=SectionIDs,
        IDSplit="连续切分",
        DataCache=Cache
    )
    LocalContext = FactorLocalContext(DTs=DTs, IDs=IDs)
    FactorList = [Factor1, Factor2, Factor3, Factor4, Factor5, Factor6, Factor7]
    Rslt = ExecEngine.run(FactorList, Context, fwd_data_list=[LocalContext]*len(FactorList), init_data_list=[{"dt_range": (DTs[0], DTs[-1]), "section_ids": SectionIDs}]*len(FactorList))
    print(Rslt)
    
    print("===")