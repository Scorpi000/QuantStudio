import webbrowser
import datetime as dt

import numpy as np
import pandas as pd
from lxml import etree

from QuantStudio.Core.CalcEngine import Engine, ParallelEngine
from QuantStudio.Factor.Factor import DataFactor, FactorContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.BackTest.BackTestModel import BTInitData, BTLocalContext, BTReport
from QuantStudio.BackTest.SectionFactor.IC import CalcIC, IC, ICDecay
from QuantStudio.BackTest.SectionFactor.Portfolio import makeQuantilePortfolio, MultiPortfolio, CalcPortfolioNV
from QuantStudio.BackTest.SectionFactor.Correlation import CalcFactorTurnover, FactorTurnover, CalcSectionCorrelation, SectionCorrelation
from QuantStudio.BackTest.SectionFactor.ReturnDecomposition import CalcFamaMacBethRegression, FamaMacBethRegression
from QuantStudio.Tools.DateTimeFun import getNaturalDay, getMonthLastDateTime


if __name__ == "__main__":
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

    FactorIC = CalcIC(lookback=1, period_lookback=1, descriptor_ids=SectionIDs)(Factor1, price=Price)
    ICModule = IC(FactorIC, args={"RollingAvgPeriod": 2, "GenReport": True})

    ICDecayModule = ICDecay(ic_list=[CalcIC(lookback=i, period_lookback=i, descriptor_ids=SectionIDs)(Factor1, price=Price) for i in range(1, 4)], args={"GenReport": True})

    Mask = (Factor1 > 0)
    QuantilePortfolioList = makeQuantilePortfolio(Factor1, mask=Mask, cat_data=Industry, weight=Weight, descriptor_ids=SectionIDs, rebalance_dts=MonthDTRuler, group_num=3)
    calcPortfolioNV = CalcPortfolioNV(descriptor_ids=SectionIDs)
    PortfolioNVList = [calcPortfolioNV(iPortfolio, price=Price, init_nv=1) for iPortfolio in QuantilePortfolioList]
    QuantilePortfolioModule = MultiPortfolio(PortfolioNVList, portfolio_list=QuantilePortfolioList, args={"RebalanceDTs": MonthDTRuler, "GenReport": True})

    TurnoverFactor = CalcFactorTurnover(lookback=1, period_lookback=1, descriptor_ids=SectionIDs)(Factor1)
    FactorTurnoverModule = FactorTurnover(TurnoverFactor, args={"GenReport": True})

    SectionCorrelationFactor = CalcSectionCorrelation(descriptor_ids=SectionIDs)(Factor1, Factor2)
    SectionCorrelationModule = SectionCorrelation(SectionCorrelationFactor, args={"GenReport": True})

    FamaMacBethFactor = CalcFamaMacBethRegression(descriptor_ids=SectionIDs)(Factor1, Factor2, price=Price)
    FamaMacBethModule = FamaMacBethRegression(FamaMacBethFactor, args={"GenReport": True})
    
    ExecEngine = Engine()
    Cache = FeatherFactorCache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "PIDs": ["0"], "CacheDir": r"D:\Data\DevCache", "ClearStart": True})
    Cache.start()
    Context = FactorContext(
        PID="0",
        PIDList=["0"],
        DTRuler=DTRuler,
        DefaultSectionIDs=SectionIDs,
        IDSplit="连续切分",
        FactorDataCache=Cache
    )
    NodeList = [ICModule, ICDecayModule, QuantilePortfolioModule, FactorTurnoverModule, SectionCorrelationModule, FamaMacBethModule]
    Report = BTReport(bt_node_list=NodeList)
    FwdDataList = [BTLocalContext(DTs=DTs)]
    InitDataList = [BTInitData(DTRange=(DTs[0], DTs[-1]))]
    Rslt = ExecEngine.run([Report], Context, fwd_data_list=FwdDataList, init_data_list=InitDataList)
    
    Output = Rslt[0]
    print(Output)

    # 生成 HTML 报告
    Tree = etree.ElementTree(etree.HTML(Output["Report"]))
    Tree.write("BTReport.html")
    webbrowser.open("BTReport.html")
    
    print("===")
