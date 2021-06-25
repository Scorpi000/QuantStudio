# coding=utf-8
"""业绩分析相关函数"""
import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__

# 单期 Brinson 模型
# data: DataFrame({"portfolio": 期初持有的投资组合权重, "benchmark": 期初基准投资组合权重, 
#                               "return": 证券收益率, "category": 资产类别}, index=IDs)
# 返回: DataFrame(index=[资产类别], columns=["portfolio", "benchmark", "allocation", "selection", "interaction", "allocation_adj"])
def calcBrinsonSinglePeriod(data):
    CatWeight = data.loc[:, ["portfolio", "benchmark", "category"]].groupby(["category"]).sum()
    CatReturn = (data.loc[:, ["portfolio", "benchmark"]].T * data["return"]).T.groupby(by=data["category"]).sum()
    CatReturn = CatReturn / CatWeight
    Rslt = pd.DataFrame({"portfolio": CatWeight["portfolio"] * CatReturn["portfolio"], 
                         "benchmark": CatWeight["benchmark"] * CatReturn["benchmark"], 
                         "allocation": (CatWeight["portfolio"] - CatWeight["benchmark"]) * CatReturn["benchmark"], 
                         "selection": CatWeight["benchmark"] * (CatReturn["portfolio"] - CatReturn["benchmark"]), 
                         "interaction": (CatWeight["portfolio"] - CatWeight["benchmark"]) * (CatReturn["portfolio"] - CatReturn["benchmark"]),
                         "allocation_adj": (CatWeight["portfolio"] - CatWeight["benchmark"]) * (CatReturn["portfolio"] - CatReturn["benchmark"].sum())})
    return Rslt

# 单期 Campisi 模型, TODO
# data: DataFrame({"portfolio": 期初持有的投资组合权重, "benchmark": 期初基准投资组合权重, 
#                               "coupon": 债券票面利率(年化), "yield": 债券到期收益率(年化), "modified_duration": 修正久期(年化)
#                               "delta_yield_treasury": 对应久期的国债收益率变化}, index=IDs)
# delta_t: 时间区间(年化)
# 返回: DataFrame(index=IDs, columns=["income", "coupon", "convergence", "treasury", "credit"])
def calcCampisiSinglePeriod(data, delta_t):
    pass
    