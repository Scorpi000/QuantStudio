# coding=utf-8
import sys
import datetime as dt

import numpy as np
import pandas as pd

import QuantStudio.api as QS


if __name__=="__main__":
    from QuantStudio.FactorDataBase.FactorDB import DataFactor, BatchContext
    from QuantStudio.FactorDataBase.FactorCache import HDF5Cache
    from QuantStudio.Tools.DateTimeFun import getNaturalDay, getMonthLastDateTime
    
    np.random.seed(0)
    IDs = [f"{str(i).zfill(6)}.SZ" for i in range(1, 21)]
    SectionIDs = IDs
    DTRuler = getNaturalDay(dt.datetime(2019, 1, 1), dt.datetime(2020, 12, 31))
    DTs = getNaturalDay(dt.datetime(2020, 1, 1), dt.datetime(2020, 12, 31))
    MonthDTRuler = getMonthLastDateTime(DTRuler)
    MonthDTs = getMonthLastDateTime(DTs)
    Mask = DataFactor(name="Mask", data=pd.DataFrame(np.random.randint(0, 2, size=(len(DTRuler), len(IDs))).astype(bool), index=DTRuler, columns=IDs))
    Industry = DataFactor(name="Industry", data=pd.Series(np.random.choice(["Fin", "TMT", "Ind"], size=(len(IDs),)), index=IDs))    
    #Rtn = DataFactor(name="Return", data=pd.DataFrame(np.random.randn(len(DTRuler), len(IDs)), index=DTRuler, columns=IDs))
    Price = DataFactor(name="Price", data=pd.DataFrame(np.random.rand(len(DTRuler), len(IDs)) * 10, index=DTRuler, columns=IDs))
    Factor1 = DataFactor(name="Factor1", data=pd.DataFrame(np.random.randn(len(DTRuler), len(IDs)), index=DTRuler, columns=IDs))
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTRuler), len(IDs)), index=DTRuler, columns=IDs))
    
    Model = QS.BackTest.BackTestModel()
    
    iModule = QS.BackTest.SectionFactor.IC(Price, Factor1, Factor2, section_ids=SectionIDs, name="IC", sys_args={
        "排序方向": "降序",
        "计算时点": MonthDTRuler,
        "回溯期数": 1,
        "相关性算法": "spearman",
        "滚动平均期数": 3
    })
    Model.Modules.append(iModule)
    
    iModule = QS.BackTest.SectionFactor.QuantilePortfolio(Factor1, Price, section_ids=SectionIDs, name="分位数组合", sys_args={
        "排序方向": "降序",
        "分组数": 3,
        "调仓时点": MonthDTRuler,
        "价格缺失": "沿用前值"
    })
    Model.Modules.append(iModule)
    
    Cache = HDF5Cache(sys_args={"缓存目录": "/home/hst/桌面/Cache"})
    Context = BatchContext(Cache, sys_args={
        "时点标尺": DTRuler,
        "截面ID": SectionIDs,
        "IO并发数": 0,
        "计算并发数": 0,
        "清空缓存": False
    })
    
    with Context:
        Model.run(DTs)
    
    Output = Model.output()
    
    print("===")
