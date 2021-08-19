# coding=utf-8
"""业绩分析相关函数"""
import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__
from QuantStudio.Tools.MathFun import interpolateHermite

# 单期 Brinson 模型
# data: DataFrame({"portfolio": 期初持有的投资组合权重, 
#                               "benchmark": 期初基准投资组合权重, 
#                               "return": 证券收益率, 
#                               "category": 资产类别}, index=IDs)
# 返回: DataFrame(index=[资产类别], columns=["portfolio", "benchmark", "allocation", "selection", "interaction", "allocation_adj"])
def calcBrinsonSinglePeriod(data):
    CatWeight = data.loc[:, ["portfolio", "benchmark", "category"]].groupby(["category"]).sum()
    CatReturn = (data.loc[:, ["portfolio", "benchmark"]].T * data["return"]).T.groupby(by=data["category"]).sum()
    CatReturn = CatReturn / CatWeight
    CatReturn[CatWeight<=0] = 0.0
    Rslt = pd.DataFrame({"portfolio": CatWeight["portfolio"] * CatReturn["portfolio"], 
                         "benchmark": CatWeight["benchmark"] * CatReturn["benchmark"], 
                         "allocation": (CatWeight["portfolio"] - CatWeight["benchmark"]) * CatReturn["benchmark"], 
                         "selection": CatWeight["benchmark"] * (CatReturn["portfolio"] - CatReturn["benchmark"]), 
                         "interaction": (CatWeight["portfolio"] - CatWeight["benchmark"]) * (CatReturn["portfolio"] - CatReturn["benchmark"]),
                         "allocation_adj": (CatWeight["portfolio"] - CatWeight["benchmark"]) * (CatReturn["portfolio"] - CatReturn["benchmark"].sum())})
    return Rslt

# 单期债券的 Campisi 模型
# data: DataFrame({"coupon": 债券票面利率(年化), 
#                               "modified_duration": 期初修正久期(年化), 
#                               "yield_0": 期初债券到期收益率(年化),
#                               "yield_t": 期末债券到期收益率(年化)}, index=IDs)
# delta_t: 时间区间(年化)
# yield_curve: 国债收益率曲线, pd.DataFrame(index=[0, 1], columns=[期限]), 第一行为期初曲线, 第二行为期末曲线
# 返回: DataFrame(index=IDs, columns=["income", "coupon", "convergence", "yield_curve", "treasury", "credit"])
def calcCampisiSinglePeriod(data, delta_t, yield_curve):
    # 计算对应期限国债收益率
    iCurve = yield_curve.iloc[0].dropna()
    TYield_0 = interpolateHermite(iCurve.index.values, iCurve.values, data["modified_duration"].values)
    iCurve = yield_curve.iloc[1].dropna()
    TYield_t = interpolateHermite(iCurve.index.values, iCurve.values, data["modified_duration"].values)
    # 单只债券的收益率分解
    BondReturn = pd.DataFrame(data["yield_0"] * delta_t, columns=["income"])
    BondReturn["coupon"] = data["coupon"] * delta_t
    BondReturn["convergence"] = BondReturn["income"] - BondReturn["coupon"]
    BondReturn["yield_curve"] = - data["modified_duration"] * (data["yield_t"] - data["yield_0"])
    BondReturn["treasury"] = - data["modified_duration"] * (TYield_t - TYield_0)
    BondReturn["credit"] = BondReturn["yield_curve"] - BondReturn["treasury"]
    return BondReturn