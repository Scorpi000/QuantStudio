import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']# 指定默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False# 正确显示负号

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Core.Node import DTLocalContext, DTInitData
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Factor.FactorCache import FeatherFactorCache
import QuantStudio.Factor.FactorOperator as fo
from QuantStudio.BackTest.BackTestModel import BTReport
from QuantStudio.BackTest.Strategy.Strategy import MakeAccount, AccountReport
from QuantStudio.BackTest.SectionFactor.Portfolio import CalcPortfolioNV
from QuantStudio.Tools.DateTimeFun import getNaturalDay, getMonthLastDateTime


if __name__=="__main__":
    np.random.seed(0)
    SectionIDs = [f"{str(i).zfill(6)}.SZ" for i in range(1, 21)]
    IDs = SectionIDs
    DTRuler = getNaturalDay(dt.datetime(2019, 1, 1), dt.datetime(2020, 12, 31))
    DTs = getNaturalDay(dt.datetime(2020, 1, 1), dt.datetime(2020, 12, 31))
    MonthDTRuler = getMonthLastDateTime(DTRuler)
    MonthDTs = getMonthLastDateTime(DTs)

    Mask = DataFactor(data=pd.DataFrame(np.random.randint(0, 2, size=(len(DTRuler), len(SectionIDs))).astype(bool), index=DTRuler, columns=SectionIDs), args={"Name": "Mask"})
    Industry = DataFactor(data=pd.Series(np.random.choice(["Fin", "TMT", "Ind"], size=(len(SectionIDs),)), index=SectionIDs, dtype=pd.StringDtype(storage="python")), args={"Name": "Industry", "DataType": "string"})    
    #Rtn = DataFactor(data=pd.DataFrame(np.random.randn(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs), args={"Name": "Return"})
    Price = DataFactor(data=pd.DataFrame(np.random.rand(len(DTRuler), len(SectionIDs)) * 10, index=DTRuler, columns=SectionIDs), args={"Name": "Price"})
    Factor1 = DataFactor(data=pd.DataFrame(np.random.randn(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs), args={"Name": "Factor1"})
    Factor2 = DataFactor(data=pd.DataFrame(np.random.randn(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs), args={"Name": "Factor2"})
    Weight = DataFactor(data=pd.DataFrame(np.random.rand(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs), args={"Name": "Weight"})

    InitCash = 1e6
    PortfolioSignal = (np.random.randn(len(DTRuler), len(SectionIDs)) > 0).astype(float)
    PortfolioSignal = DataFactor(data=pd.DataFrame(PortfolioSignal / np.sum(PortfolioSignal, axis=1, keepdims=True), index=DTRuler, columns=SectionIDs), args={"Name": "Signal"})
    
    Account = MakeAccount(signal_type="目标权重", start_dt=DTs[0], init_cash=InitCash)(last_price=Price, signal=PortfolioSignal)
    StrategyAmt = fo.Fetch(pos=2, dtype="double")(Account)

    StrategyNV = CalcPortfolioNV(start_dt=DTs[0], descriptor_ids=SectionIDs)(PortfolioSignal, price=Price, init_nv=InitCash)

    StrategyReport = AccountReport(account=Account, bmk_nv=StrategyNV, args={"GenReport": True})

    PIDList = ["0"]
    ExecEngine = Engine()
    # PIDList = ["0-0", "0-1"]
    # ExecEngine = ParallelEngine()
    Cache = FeatherFactorCache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "PIDs": PIDList, "CacheDir": "/mnt/d/Data/Cache/DevCache", "StartMode": "new"})
    Cache.start()
    Context = FactorContext(
        PID="0",
        PIDList=PIDList,
        DTRuler=DTRuler,
        SectionIDs=SectionIDs,
        DataCache=Cache
    )
    NodeList = [PortfolioSignal, Account, StrategyAmt, StrategyNV, StrategyReport]
    FwdDataList = [FactorLocalContext(DTs=DTs, IDs=IDs)] * len(NodeList)
    InitDataList = [FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=SectionIDs)] * len(NodeList)
    Rslt = ExecEngine.run(NodeList, Context, fwd_data_list=FwdDataList, init_data_list=InitDataList)

    print(Rslt[0])
    print("===")
