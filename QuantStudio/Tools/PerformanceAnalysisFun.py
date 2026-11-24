# coding=utf-8
"""业绩分析相关函数"""
import numpy as np
import pandas as pd

from QuantStudio.Tools.MathFun import interpolateHermite


def calcBrinsonSignlePeriodExt(data):
    """单期 Brinson 模型，基于外部计算的资产权重和收益率
    
    Args:
        data: DataFrame({
                "p_weight": 目标投资组合期初资产权重, 
                "bmk_weight": 基准投资组合期初资产权重, 
                "p_return": 目标投资组合资产收益率, 
                "bmk_return": 基准投资组合资产收益率
            }, index=[资产])
    Returns:
        DataFrame({
            "portfolio": 目标投资组合收益率, 
            "benchmark": 基准投资组合收益率, 
            "allocation": 资产配置收益, 
            "selection": 个券选择收益,
            "interaction": 交互效应,
            "allocation_adj": 修正的资产配置收益
        }, index=[资产])
    """
    return pd.DataFrame({
        "portfolio": data["p_weight"] * data["p_return"], 
        "benchmark": data["bmk_weight"] * data["bmk_return"], 
        "allocation": (data["p_weight"] - data["bmk_weight"]) * data["bmk_return"], 
        "selection": data["bmk_weight"] * (data["p_return"] - data["bmk_return"]), 
        "interaction": (data["p_weight"] - data["bmk_weight"]) * (data["p_return"] - data["bmk_return"]),
        "allocation_adj": (data["p_weight"] - data["bmk_weight"]) * (data["p_return"] - data["bmk_return"].sum())
    })


def calcBrinsonSinglePeriod(data):
    """单期 Brinson 模型

    Args:
        data: DataFrame({
                "portfolio": 目标投资组合期初权重, 
                "benchmark": 基准投资组合期初权重, 
                "return": 证券收益率, 
                "category": 资产类别
            }, index=[证券代码])
    Returns:
        DataFrame({
            "portfolio": 目标投资组合收益率, 
            "benchmark": 基准投资组合收益率, 
            "allocation": 资产配置收益, 
            "selection": 个券选择收益,
            "interaction": 交互效应,
            "allocation_adj": 修正的资产配置收益,
            "p_weight": 目标投资组合期初资产权重, 
            "bmk_weight": 基准投资组合期初资产权重, 
            "p_return": 目标投资组合资产收益率, 
            "bmk_return": 基准投资组合资产收益率
        }, index=[资产类别])
    """
    CatWeight = data.loc[:, ["portfolio", "benchmark", "category"]].groupby(["category"]).sum()
    CatReturn = (data.loc[:, ["portfolio", "benchmark"]].T * data["return"]).T.groupby(by=data["category"]).sum()
    CatReturn = CatReturn / CatWeight
    CatReturn[CatWeight<=0] = 0.0
    return pd.DataFrame({
        "portfolio": CatWeight["portfolio"] * CatReturn["portfolio"], 
        "benchmark": CatWeight["benchmark"] * CatReturn["benchmark"], 
        "allocation": (CatWeight["portfolio"] - CatWeight["benchmark"]) * CatReturn["benchmark"], 
        "selection": CatWeight["benchmark"] * (CatReturn["portfolio"] - CatReturn["benchmark"]), 
        "interaction": (CatWeight["portfolio"] - CatWeight["benchmark"]) * (CatReturn["portfolio"] - CatReturn["benchmark"]),
        "allocation_adj": (CatWeight["portfolio"] - CatWeight["benchmark"]) * (CatReturn["portfolio"] - CatReturn["benchmark"].sum()),
        "p_weight": CatWeight["portfolio"],
        "bmk_weight": CatWeight["benchmark"],
        "p_return": CatReturn["portfolio"],
        "bmk_return": CatReturn["benchmark"]
    })


def calcCarinoAdjustFactor(data):
    """计算多期 Brinson 归因中各期的 Carino 调整因子。

    Args:
        data: 各期组合与基准收益率，DataFrame(index=[期],
            columns=["portfolio", "benchmark"])。收益率采用小数形式，且大于 -1。

    Returns:
        pd.Series: 与输入同索引的各期调整因子 ``k_t / k``。其中 ``k_t``
        为当期组合和基准收益率的对数差商，``k`` 为累计收益率的对数差商。
    """
    portfolio = data["portfolio"].to_numpy(dtype=float)
    benchmark = data["benchmark"].to_numpy(dtype=float)
    if (not np.isfinite(portfolio).all() or not np.isfinite(benchmark).all()
            or (portfolio <= -1).any() or (benchmark <= -1).any()):
        raise ValueError("组合和基准收益率必须为有限数且大于 -1")

    excess = portfolio - benchmark
    period_k = 1 / (1 + benchmark)
    unequal = excess != 0
    period_k[unequal] = np.log1p(excess[unequal] / (1 + benchmark[unequal])) / excess[unequal]

    portfolio_total = np.prod(1 + portfolio) - 1
    benchmark_total = np.prod(1 + benchmark) - 1
    total_excess = portfolio_total - benchmark_total
    if total_excess == 0:
        total_k = 1 / (1 + benchmark_total)
    else:
        total_k = np.log1p(total_excess / (1 + benchmark_total)) / total_excess
    return pd.Series(period_k / total_k, index=data.index, name="adjust_factor")


def calcCampisiSinglePeriod(data, delta_t, yield_curve):
    """债券的单期 Campisi 模型

    Args:
        data: DataFrame({
            "coupon": 债券票面利率(年化), 
            "modified_duration": 期初修正久期(年化), 
            "yield_0": 期初债券到期收益率(年化),
            "yield_t": 期末债券到期收益率(年化)
            }, index=[债券])
        delta_t: 时间区间(年化)
        yield_curve: 国债收益率曲线, pd.DataFrame(index=[0, 1], columns=[期限]), 第一行为期初曲线, 第二行为期末曲线
    
    Returns: DataFrame(index=[债券], columns=["income", "coupon", "convergence", "yield_curve", "treasury", "credit"])
    """
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
