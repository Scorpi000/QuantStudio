import webbrowser
import datetime as dt

import numpy as np
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']# 指定默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False# 正确显示负号

from QuantStudio.Core.CalcEngine import Engine, ParallelEngine
from QuantStudio.Factor.Factor import DataFactor, FactorContext
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.BackTest.BackTestModel import BTInitData, BTLocalContext, BTReport
from QuantStudio.BackTest.SectionFactor.IC import CalcIC, IC, ICDecay
from QuantStudio.BackTest.SectionFactor.Portfolio import makeQuantilePortfolio, MultiPortfolio, CalcPortfolioNV
from QuantStudio.BackTest.SectionFactor.Correlation import CalcFactorTurnover, FactorTurnover, CalcSectionCorrelation, SectionCorrelation
from QuantStudio.BackTest.SectionFactor.ReturnDecomposition import CalcFamaMacBethRegression, FamaMacBethRegression
from QuantStudio.Tools.DateTimeFun import getNaturalDay, getMonthLastDateTime


if __name__ == "__main__1":
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


if __name__=="__main__":
    # 参数设置
    from QuantStudio.Factor.HDF5DB import HDF5DB
    HDB = HDF5DB(args={"MainDir": r"D:\Data\TestHDF5DB"}).connect()

    StartDT, EndDT = dt.datetime(2014, 1, 1), dt.datetime(2026, 2, 28)# 数据起止时间
    TestStartDT, TestEndDT = dt.datetime(2019, 1, 1), EndDT# 测试起止时间

    FT = HDB.getTable("stock_cn_day_bar_adj_backward_nafilled")
    DTRuler = FT.getDateTime(start_dt=StartDT, end_dt=EndDT)
    TestDTs = FT.getDateTime(start_dt=TestStartDT, end_dt=TestEndDT)
    SectionIDs = IDs = FT.getID()

    # 再平衡时点序列
    from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime
    BalanceDTs = getMonthLastDateTime(DTRuler)# 月末

    FT = HDB.getTable("stock_cn_status")
    Mask = (FT.getFactor("if_trading")==1)

    FT = HDB.getTable("stock_cn_day_bar_adj_backward_nafilled")
    Price = FT.getFactor("close")

    FT = HDB.getTable("stock_cn_industry")
    Industry = FT.getFactor("citic2019_level1")

    FT = HDB.getTable("stock_cn_factor_value")
    FactorList = [FT.getFactor(iFactorName) for iFactorName in ["bp_lr", "ep_ttm"]]

    # 回测节点列表
    NodeList = []

    # Rank IC
    FactorIC = CalcIC(lookback=31, period_lookback=1, corr_method="spearman", descriptor_ids=SectionIDs)(*FactorList, price=Price, mask=Mask, cat_data=Industry, factor_args={"CalcDTRuler": BalanceDTs})
    ICNode = IC(FactorIC, args={"RollingAvgPeriod": 2, "GenReport": True})
    NodeList.append(ICNode)

    # IC 衰减
    ICDecayNode = ICDecay(ic_list=[CalcIC(lookback=31*i, period_lookback=i, descriptor_ids=SectionIDs)(*FactorList, price=Price, mask=Mask, cat_data=Industry, factor_args={"CalcDTRuler": BalanceDTs}) for i in range(1, 13)], args={"GenReport": True})
    NodeList.append(ICDecayNode)

    # 分位数组合
    calcPortfolioNV = CalcPortfolioNV(descriptor_ids=SectionIDs)
    for iFactor in FactorList:
        iQuantilePortfolioList = makeQuantilePortfolio(iFactor, mask=Mask, cat_data=Industry, weight=None, descriptor_ids=SectionIDs, rebalance_dts=BalanceDTs, group_num=5)
        iPortfolioNVList = [calcPortfolioNV(iPortfolio, price=Price, init_nv=1, factor_args={"Name": f"P{i}"}) for i, iPortfolio in enumerate(iQuantilePortfolioList)]
        iQuantilePortfolioNode = MultiPortfolio(nv_list=iPortfolioNVList, portfolio_list=iQuantilePortfolioList, args={"RebalanceDTs": BalanceDTs, "GenReport": True, "Name": f"{iFactor.Name}-分位数组合"})
        NodeList.append(iQuantilePortfolioNode)

    # 因子换手率
    TurnoverFactor = CalcFactorTurnover(lookback=31, period_lookback=1, descriptor_ids=SectionIDs)(*FactorList, mask=Mask)
    FactorTurnoverNode = FactorTurnover(TurnoverFactor, args={"GenReport": True})
    NodeList.append(FactorTurnoverNode)

    # 截面相关性
    SectionCorrelationFactor = CalcSectionCorrelation(descriptor_ids=SectionIDs)(*FactorList, mask=Mask)
    SectionCorrelationNode = SectionCorrelation(SectionCorrelationFactor, args={"GenReport": True})
    NodeList.append(SectionCorrelationNode)

    # Fama-MacBeth 回归
    FamaMacBethFactor = CalcFamaMacBethRegression(descriptor_ids=SectionIDs)(*FactorList, price=Price, mask=Mask, cat_data=Industry)
    FamaMacBethModule = FamaMacBethRegression(FamaMacBethFactor, args={"GenReport": True})
    NodeList.append(FamaMacBethModule)

    Report = BTReport(bt_node_list=NodeList)

    with FeatherFactorCache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "PIDs": ["0"], "CacheDir": r"D:\Data\DevCache", "ClearStart": True}) as Cache:
        with FactorContext(
            PID="0",
            PIDList=["0"],
            DTRuler=DTRuler,
            DefaultSectionIDs=SectionIDs,
            FactorDataCache=Cache
        ) as Context:
            with Engine() as ExecEngine:
                Output, = ExecEngine.run([Report], Context, fwd_data_list=[BTLocalContext(DTs=TestDTs)], init_data_list=[BTInitData(DTRange=(TestDTs[0], TestDTs[-1]))])

    print(Output["Report"])