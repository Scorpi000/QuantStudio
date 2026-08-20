# coding=utf-8
"""宏观数据处理函数"""
import datetime as dt
from typing import Dict

import pandas as pd
import numpy as np


def cleanMacroPublDate(
    df: pd.DataFrame, 
    cutoff_date: Dict[str, dt.datetime] = {},
    detect_lag_outlier: bool = True, 
    outlier_mad_factor: float = 3.0,
    lag_quantile: float = 0.75
) -> pd.DataFrame:
    """智能清洗宏观数据发布日的函数。

    核心逻辑：
    1. 截止日早于 cutoff_date 的记录直接视为不可靠
    2. 同一 ID 下，若某个 PublDate 对应多个不同的 EndDate，该批次不可靠
    3. 即使 PublDate 只对应一个 EndDate，若其 lag 远超该指标的典型滞后，也视为不可靠
    对不可靠记录，用 `EndDate + 保守滞后天数` 作为估计。

    Args:
        df: 包含 ID, EndDate, PublDate 的 DataFrame
        cutoff_date: {指标ID: datetime}, 截止日早于此日期的记录直接视为不可靠
        detect_lag_outlier: 是否启用基于 lag 统计的异常值检测
        outlier_mad_factor: 异常值判定的 MAD 倍数，lag > median + factor * MAD 视为异常
        lag_quantile: 用于估计修正滞后的分位数（0~1）
    """
    result = df.copy()
    result["EndDate"] = pd.to_datetime(result["EndDate"])
    result["PublDate"] = pd.to_datetime(result["PublDate"])

    # ---- 第零轮：cutoff_date 规则 ----
    cutoff_unreliable = pd.Series(False, index=result.index)
    if cutoff_date:
        for ind_id, cutoff in cutoff_date.items():
            cutoff = pd.Timestamp(cutoff)
            cutoff_unreliable |= (result["ID"] == ind_id) & (result["EndDate"] < cutoff)

    # ---- 第一轮：PublDate 对应多个 EndDate 的批次不可靠 ----
    no_publ = result["PublDate"].isna()
    group_key = result.groupby(["ID", "PublDate"])["EndDate"].transform("nunique")
    multi_end = (group_key > 1) | no_publ

    # ---- 第二轮：基于 lag 统计的异常值检测（仅对未被前两轮标记的记录）----
    is_unreliable = cutoff_unreliable | multi_end
    if detect_lag_outlier:
        candidate = ~(cutoff_unreliable | multi_end)
        single = result[candidate].copy()
        single["lag"] = (single["PublDate"] - single["EndDate"]).dt.days
        lag_grouped = single.groupby("ID")["lag"]
        lag_median = lag_grouped.median()
        lag_mad = lag_grouped.apply(lambda x: np.median(np.abs(x - x.median())))
        lag_stats = pd.DataFrame({"median": lag_median, "mad": lag_mad})
        lag_stats["mad"] = lag_stats["mad"].replace(0, np.nan)
        single["id_median"] = single["ID"].map(lag_stats["median"])
        single["id_mad"] = single["ID"].map(lag_stats["mad"])
        single["lag_outlier"] = (single["lag"] - single["id_median"]) > (outlier_mad_factor * single["id_mad"])
        single["lag_outlier"] = single["lag_outlier"].fillna(False)
        is_unreliable.loc[single[single["lag_outlier"].astype(bool)].index] = True
    is_reliable = ~is_unreliable

    # ---- 计算保守滞后（75分位，仅从可靠记录推算）----
    reliable = result[is_reliable].copy()
    reliable["lag"] = (reliable["PublDate"] - reliable["EndDate"]).dt.days
    typical_lag = reliable.groupby("ID")["lag"].quantile(lag_quantile)

    # ---- 修正不可靠记录：EndDate + 保守滞后 ----
    unreliable = result[is_unreliable].copy()
    unreliable["final_PublDate"] = unreliable["PublDate"]
    if not unreliable.empty:
        id_lag = unreliable["ID"].map(typical_lag)
        # 无可靠记录时回退到 0
        estimated_lag = id_lag.fillna(0)
        unreliable["final_PublDate"] = unreliable["EndDate"] + pd.to_timedelta(estimated_lag, unit="D")

    reliable["final_PublDate"] = reliable["PublDate"]

    # 合并
    result = result.merge(
        pd.concat([reliable[["final_PublDate"]], unreliable[["final_PublDate"]]])
          .rename_axis(None),
        left_index=True, right_index=True, how="left"
    )
    result["final_PublDate"] = result["final_PublDate"].fillna(result["PublDate"])

    # 输出列
    result["lag_days"] = (result["PublDate"] - result["EndDate"]).dt.days
    result["is_reliable"] = is_reliable.values
    result["final_PublDate"] = pd.to_datetime(result["final_PublDate"]).dt.normalize()

    return result

def cleanMacroPublDateDynamic(
    df: pd.DataFrame, 
    cutoff_date: Dict[str, dt.datetime] = {},
    detect_lag_outlier: bool = True, 
    outlier_mad_factor: float = 3.0,
    lag_quantile: float = 0.75
) -> pd.DataFrame:
    """智能清洗宏观数据发布日的函数（动态滞后版本）。

    与 clean_macro_publ_date 的区别：滞后天数不再使用固定值，而是基于可靠记录
    拟合 lag 随 EndDate 变化的线性趋势，并加上残差的指定分位作为保守偏移。

    Args:
        df: 包含 ID, EndDate, PublDate 的 DataFrame
        cutoff_date: {指标ID: datetime}, 截止日早于此日期的记录直接视为不可靠
        detect_lag_outlier: 是否启用基于 lag 统计的异常值检测
        outlier_mad_factor: 异常值判定的 MAD 倍数，lag > median + factor * MAD 视为异常
        lag_quantile: 残差分位数 / 样本不足时的固定滞后分位数
    """
    result = df.copy()
    result["EndDate"] = pd.to_datetime(result["EndDate"])
    result["PublDate"] = pd.to_datetime(result["PublDate"])

    # ---- 第零轮：cutoff_date 规则 ----
    cutoff_unreliable = pd.Series(False, index=result.index)
    if cutoff_date:
        for ind_id, cutoff in cutoff_date.items():
            cutoff = pd.Timestamp(cutoff)
            cutoff_unreliable |= (result["ID"] == ind_id) & (result["EndDate"] < cutoff)

    # ---- 第一轮：PublDate 对应多个 EndDate 的批次不可靠 ----
    no_publ = result["PublDate"].isna()
    group_key = result.groupby(["ID", "PublDate"])["EndDate"].transform("nunique")
    multi_end = (group_key > 1) | no_publ

    # ---- 第二轮：基于 lag 统计的异常值检测 ----
    is_unreliable = cutoff_unreliable | multi_end
    if detect_lag_outlier:
        candidate = ~(cutoff_unreliable | multi_end)
        single = result[candidate].copy()
        single["lag"] = (single["PublDate"] - single["EndDate"]).dt.days
        lag_grouped = single.groupby("ID")["lag"]
        lag_median = lag_grouped.median()
        lag_mad = lag_grouped.apply(lambda x: np.median(np.abs(x - x.median())))
        lag_stats = pd.DataFrame({"median": lag_median, "mad": lag_mad})
        lag_stats["mad"] = lag_stats["mad"].replace(0, np.nan)
        single["id_median"] = single["ID"].map(lag_stats["median"])
        single["id_mad"] = single["ID"].map(lag_stats["mad"])
        single["lag_outlier"] = (single["lag"] - single["id_median"]) > (outlier_mad_factor * single["id_mad"])
        single["lag_outlier"] = single["lag_outlier"].fillna(False)
        is_unreliable.loc[single[single["lag_outlier"].astype(bool)].index] = True
    is_reliable = ~is_unreliable

    # ---- 拟合 lag 随 EndDate 的线性趋势 ----
    reliable = result[is_reliable].copy()
    reliable["lag"] = (reliable["PublDate"] - reliable["EndDate"]).dt.days
    reliable["end_ts"] = reliable["EndDate"].astype(np.int64) / 1e9 / 86400  # 转为天数

    trend_model = {}  # {ID: (slope, intercept, residual_q75)}
    for ind_id, grp in reliable.groupby("ID"):
        if len(grp) < 3:
            # 样本不足，退化为固定值
            q75 = grp["lag"].quantile(lag_quantile)
            trend_model[ind_id] = (0.0, q75, 0.0)
        else:
            x = grp["end_ts"].values
            y = grp["lag"].values
            # 最小二乘线性拟合
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            residuals = y - predicted
            residual_q75 = np.percentile(residuals, lag_quantile * 100)
            trend_model[ind_id] = (slope, intercept, residual_q75)

    # ---- 动态滞后估计函数 ----
    def _estimate_lag(ind_id, end_date):
        """基于 EndDate 预测保守滞后天数。"""
        if ind_id not in trend_model:
            return np.nan
        slope, intercept, residual_q75 = trend_model[ind_id]
        end_ts = pd.Timestamp(end_date).value / 1e9 / 86400
        return max(slope * end_ts + intercept + residual_q75, 0)

    # ---- 修正不可靠记录：EndDate + 动态滞后 ----
    unreliable = result[is_unreliable].copy()
    unreliable["final_PublDate"] = unreliable["PublDate"]
    if not unreliable.empty:
        estimated_lag = unreliable.apply(
            lambda r: _estimate_lag(r["ID"], r["EndDate"]), axis=1
        )
        # 无趋势模型时回退到 0
        estimated_lag = estimated_lag.fillna(0)
        unreliable["final_PublDate"] = unreliable["EndDate"] + pd.to_timedelta(estimated_lag, unit="D")

    reliable["final_PublDate"] = reliable["PublDate"]

    # 合并
    result = result.merge(
        pd.concat([reliable[["final_PublDate"]], unreliable[["final_PublDate"]]])
          .rename_axis(None),
        left_index=True, right_index=True, how="left"
    )
    result["final_PublDate"] = result["final_PublDate"].fillna(result["PublDate"])

    # 输出列
    result["lag_days"] = (result["PublDate"] - result["EndDate"]).dt.days
    result["is_reliable"] = is_reliable.values
    result["final_PublDate"] = pd.to_datetime(result["final_PublDate"]).dt.normalize()

    return result


if __name__=="__main__":
    from QuantStudio.Factor.JYDB import JYDB

    JYDB = JYDB().connect()
    SQLStr = """
        SELECT IndicatorCode AS ID, EndDate, InfoPublDate AS PublDate
        FROM C_ED_MacroIndicatorData
        WHERE indicatorcode = 110251594
        ORDER BY EndDate
    """
    df_test = JYDB.fetchall(sql_str=SQLStr, header=False)
    df_test = pd.DataFrame(df_test, columns=["ID", "EndDate", "PublDate"])

    print("=== 原始数据 ===")
    print(df_test[['EndDate', 'PublDate']])

    # 设置截止日期为 2023-02-01，则 1月31日的记录会被强制标记
    result = cleanMacroPublDate(df_test, cutoff_date=None)

    print("\n=== 处理结果 ===")
    print(result[['EndDate', 'PublDate', 'lag_days', 'is_reliable', 'final_PublDate']])

    # 查看被修正的记录
    unreliable = result[~result['is_reliable']]
    if not unreliable.empty:
        print("\n=== 被修正的不可靠记录 ===")
        print(unreliable[['EndDate', 'PublDate', 'lag_days', 'final_PublDate']])