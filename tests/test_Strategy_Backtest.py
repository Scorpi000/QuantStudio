# -*- coding: utf-8 -*-
"""策略回测对比"""
import faulthandler
faulthandler.enable()
import datetime as dt

import numpy as np
import pandas as pd
pd.set_option("future.infer_string", False)# 禁用 PyArrow 字符串后端

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Factor.FactorCache import FeatherFactorCache
import QuantStudio.Factor.FactorOperator as fo
from QuantStudio.BackTest.Strategy.Strategy import MakeAccount, MakeStrategy
from QuantStudio.BackTest.SectionFactor.Portfolio import CalcPortfolioNV, CalcMaskPortfolio
from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime, getNaturalDay


if __name__=="__main__":
    StartDT, EndDT = dt.datetime(2024, 1, 1), dt.datetime(2025, 12, 31)# 数据起止时间
    TestStartDT, TestEndDT = dt.datetime(2025, 1, 31), EndDT# 测试起止时间
    DTRuler = getNaturalDay(start_date=StartDT, end_date=EndDT)
    TestDTs = getNaturalDay(start_date=TestStartDT, end_date=TestEndDT)
    SectionIDs = [f"{str(i).zfill(6)}.SZ" for i in range(1, 101)]
    IDs = SectionIDs

    # MOCK 数据
    np.random.seed(0)
    Mask = DataFactor(data=pd.DataFrame(np.random.choice([0, 1], size=(len(DTRuler), len(SectionIDs)), p=[0.1, 0.9]).astype(bool), index=DTRuler, columns=SectionIDs), args={"Name": "Mask"})
    Industry = DataFactor(data=pd.Series(np.random.choice(["Fin", "TMT", "Ind"], size=(len(SectionIDs),)), index=SectionIDs, dtype=pd.StringDtype(storage="python")), args={"Name": "Industry", "DataType": "string"})    
    Price = DataFactor(data=pd.DataFrame(np.random.rand(len(DTRuler), len(SectionIDs)) * 10, index=DTRuler, columns=SectionIDs), args={"Name": "Price"})
    ExpectedReturn = DataFactor(data=pd.DataFrame(np.random.randn(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs), args={"Name": "ExpectedReturn"})
    Factor2 = DataFactor(data=pd.DataFrame(np.random.randn(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs), args={"Name": "Factor2"})
    Weight = DataFactor(data=pd.DataFrame(np.random.rand(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs), args={"Name": "Weight"})

    # 再平衡时点序列
    BalanceDTs = getMonthLastDateTime(DTRuler)# 月末
    
    # 构造投资组合
    ExpectedRank = fo.SectionRank(ascending=True, uniformization=True)(ExpectedReturn, mask=Mask, cat_data=Industry)
    Portfolio = CalcMaskPortfolio(descriptor_ids=SectionIDs)(
        mask=(ExpectedRank >= 0.7),
        weight=None,
        cat_data=Industry,
        cat_weight=Weight,
        factor_args={"CalcDTRuler": BalanceDTs, "Name": "Portfolio"}
    )

    # 迭代式回测
    Account = MakeAccount(signal_type="目标权重", init_cash=1e6, short_allowed=False, start_dt=TestDTs[0])(
        last_price=Price, signal=Portfolio, 
        factor_args={"Name": "迭代式回测"}
    )

    # 向量化回测
    PortfolioNV1 = CalcPortfolioNV(descriptor_ids=SectionIDs, start_dt=TestDTs[0], calc_type="numpy")(Portfolio, price=Price, init_nv=1e6, portfolio_name_list=["Portfolio"], factor_args={"Name": "向量化回测-numpy"})
    PortfolioNV2 = CalcPortfolioNV(descriptor_ids=SectionIDs, start_dt=TestDTs[0], calc_type="pandas")(Portfolio, price=Price, init_nv=1e6, portfolio_name_list=["Portfolio"], factor_args={"Name": "向量化回测-pandas"})

    NodeList = [Account, PortfolioNV1, PortfolioNV2]
    FwdDataList = [FactorLocalContext(DTs=TestDTs, IDs=SectionIDs), FactorLocalContext(DTs=TestDTs, IDs=["Portfolio"]), FactorLocalContext(DTs=TestDTs, IDs=["Portfolio"])]
    InitDataList = [FactorInitData(DTRange=(TestDTs[0], TestDTs[-1]), SectionIDs=SectionIDs), FactorInitData(DTRange=(TestDTs[0], TestDTs[-1]), SectionIDs=["Portfolio"]), FactorInitData(DTRange=(TestDTs[0], TestDTs[-1]), SectionIDs=["Portfolio"])]

    with FeatherFactorCache(args={"DTRuler": DTRuler, "StartMode": "new", "CacheDir": None}) as Cache:
        with FactorContext(DTRuler=DTRuler, DefaultSectionIDs=SectionIDs, FactorDataCache=Cache) as Context:
            with Engine() as ExecEngine:
                Rslt = ExecEngine.run(
                    NodeList, Context, 
                    fwd_data_list=FwdDataList, 
                    init_data_list=InitDataList
                )

    NV = {}
    Cash = Rslt[0].iloc[:, 0].apply(lambda x: x[0])
    iNV = Cash + Rslt[0].map(lambda x: x[2]).sum(axis=1)
    iNV = iNV / iNV.iloc[0]
    NV["迭代式回测"] = iNV
    NV["向量化回测-numpy"] = Rslt[1].iloc[:, 0] / Rslt[1].iloc[0, 0]
    NV["向量化回测-pandas"] = Rslt[2].iloc[:, 0] / Rslt[2].iloc[0, 0]
    NV = pd.DataFrame(NV)
    print("迭代式回测 - 向量化回测-numpy 最大误差: ", (NV["迭代式回测"] - NV["向量化回测-numpy"]).abs().max())
    print("向量化回测-pandas - 向量化回测-numpy 最大误差: ", (NV["向量化回测-pandas"] - NV["向量化回测-numpy"]).abs().max())

    # with pd.ExcelWriter("./test_Strategy_Backtest.xlsx", engine="openpyxl") as xlsFile:
    #     NV.to_excel(xlsFile, sheet_name="回测净值", index=True, header=True)

    print("===")