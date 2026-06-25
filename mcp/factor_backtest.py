# -*- coding: utf-8 -*-
"""
因子回测 MCP Server

基于 QuantStudio 框架，对指定因子进行 IC（信息系数）回测。

功能：
    - 从聚源 JYDB 加载因子数据和价格数据
    - 使用 QuantStudio CalcIC 算子计算因子 IC 序列
    - 输出 IC 均值、标准差、ICIR、t 统计量、胜率等统计指标

工具列表：
    - backtest_factor_ic: 回测因子的 IC 表现

使用方式（以 stdio 模式运行）：
    D:\\miniforge\\envs\\QS312\\python.exe factor_backtest.py

注册到 Claude Code：
    claude mcp add --transport stdio --scope local factor_backtest -- \\
        "D:\\miniforge\\envs\\QS312\\python.exe" \\
        "D:\\Project\\QuantStudio\\mcp\\factor_backtest.py" \\
        --env "PYTHONPATH=D:\\Project\\QuantStudio;D:\\Project\\QSAgent;D:\\Project\\QSResearch"
"""

import datetime as dt
import json
from typing import Optional, List, Literal, Any, Dict

import numpy as np
import pandas as pd
from fastmcp import FastMCP

from QuantStudio.Factor import JYDB
from QuantStudio.Factor.Factor import DataFactor
from QuantStudio.BackTest.SectionFactor.IC import CalcIC

mcp = FastMCP(name="factor_backtest")


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _get_jydb() -> JYDB:
    """创建并连接 JYDB 实例。"""
    jydb = JYDB()
    jydb.connect()
    return jydb


def _load_factor_df(
    jydb: JYDB,
    table_name: str,
    field_names: List[str],
    ids: List[str],
    dts: List[dt.datetime],
) -> pd.DataFrame:
    """从 JYDB 加载因子数据，返回 DataFrame(index=dts, columns=ids)。"""
    ft = jydb.getTable(table_name)
    raw = ft.readData(factor_names=field_names, ids=ids, dts=dts)
    # raw 可能是 Panel(items=field_names, major_axis=dts, minor_axis=ids)
    # 也可能是 DataFrame（单因子多字段场景）
    if hasattr(raw, "iloc") and hasattr(raw, "shape") and len(raw.shape) == 3:
        # Panel: 取第一个因子
        return raw.iloc[0]
    return raw


def _calc_ic_statistics(
    ic_df: pd.DataFrame,
    breadth_df: pd.DataFrame,
    rolling_avg_period: int = 12,
) -> pd.DataFrame:
    """由 IC 序列和截面宽度计算汇总统计，对标 IC.backward_compute 的输出。"""
    stats = pd.DataFrame(index=ic_df.columns)
    stats["IC 均值"] = ic_df.mean()
    stats["IC 标准差"] = ic_df.std()
    stats["IC 最小值"] = ic_df.min()
    stats["IC 最大值"] = ic_df.max()
    stats["ICIR"] = stats["IC 均值"] / stats["IC 标准差"]
    stats["平均截面宽度"] = breadth_df.mean()
    stats["IC × √N"] = stats["IC 均值"] * np.sqrt(stats["平均截面宽度"])
    stats["有效期数"] = pd.notnull(ic_df).sum()
    stats["t 统计量"] = stats["有效期数"] ** 0.5 * stats["ICIR"]
    ic_ma = ic_df.rolling(window=rolling_avg_period, min_periods=1).mean()
    stats["IC 移动平均(最新)"] = ic_ma.iloc[-1]
    stats["胜率"] = (ic_df > 0).sum() / pd.notnull(ic_df).sum()
    return stats


# ---------------------------------------------------------------------------
# MCP 工具
# ---------------------------------------------------------------------------

@mcp.tool()
def backtest_factor_ic(
    factor_table: str,
    factor_fields: List[str],
    start_date: str,
    end_date: str,
    price_table: str = "A股日度行情",
    price_field: str = "收盘价",
    corr_method: Literal["spearman", "pearson", "kendall"] = "spearman",
    period_lookback: int = 1,
    stock_ids: Optional[List[str]] = None,
    rolling_avg_period: int = 12,
) -> str:
    """回测指定因子的 IC（信息系数）表现。

    计算因子值与未来收益率之间的截面秩相关性（默认 Spearman），
    返回 IC 均值、ICIR、t 统计量、胜率等统计指标。

    Args:
        factor_table: JYDB 中的因子表名称，例如 "A股日度行情"
        factor_fields: 待回测的因子字段列表，例如 ["涨跌幅", "换手率"]
        start_date: 起始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        price_table: 价格数据所在表名，默认 "A股日度行情"
        price_field: 价格字段名，默认 "收盘价"
        corr_method: 相关性方法，可选 spearman / pearson / kendall
        period_lookback: 因子回溯期数，1 表示上期因子值对应当期收益率
        stock_ids: 股票 ID 列表，默认 None 使用全体 A 股
        rolling_avg_period: IC 移动平均期数，默认 12

    Returns:
        JSON 字符串，包含：
        - params: 回测参数摘要
        - ic_stats: 各因子的 IC 统计指标
        - ic_time_series_tail: 最近 20 期的 IC 序列
    """
    # ---- 1. 连接数据库，获取交易日和股票列表 ----
    jydb = _get_jydb()
    dts = jydb.getTradeDay(
        start_date=dt.datetime.strptime(start_date, "%Y-%m-%d"),
        end_date=dt.datetime.strptime(end_date, "%Y-%m-%d"),
        exchange="SSE",
    )
    if not dts:
        return json.dumps({"error": "指定日期范围内无交易日"}, ensure_ascii=False)

    if stock_ids is None:
        stock_ids = jydb.getStockID(
            index_id="全体A股",
            date=dt.datetime.strptime(end_date, "%Y-%m-%d").date(),
            is_current=False,
        )
    if not stock_ids:
        return json.dumps({"error": "未能获取股票列表"}, ensure_ascii=False)

    # ---- 2. 加载价格因子 ----
    try:
        price_df = _load_factor_df(jydb, price_table, [price_field], stock_ids, dts)
    except Exception as e:
        return json.dumps({"error": f"加载价格因子失败: {e}"}, ensure_ascii=False)

    # ---- 3. 加载测试因子 ----
    try:
        factor_df = _load_factor_df(jydb, factor_table, factor_fields, stock_ids, dts)
    except Exception as e:
        return json.dumps({"error": f"加载测试因子失败: {e}"}, ensure_ascii=False)

    if factor_df.empty:
        return json.dumps({"error": "测试因子数据为空"}, ensure_ascii=False)

    # ---- 4. 构建 DataFactor，使用 CalcIC 算子 ----
    price_factor = DataFactor(data=price_df, args={"Name": price_field})

    # 若 factor_fields 只一个字段，factor_df 直接是单个因子的 DataFrame；
    # 若是 Panel 则先取出各因子
    if hasattr(factor_df, "shape") and len(factor_df.shape) == 3:
        # 多因子 Panel
        test_factors = [
            DataFactor(data=factor_df.iloc[i], args={"Name": factor_fields[i]})
            for i in range(factor_df.shape[0])
        ]
    else:
        test_factors = [DataFactor(data=factor_df, args={"Name": factor_fields[0]})]

    ic_operator = CalcIC(
        descriptor_ids=stock_ids,
        lookback=31,
        period_lookback=period_lookback,
        corr_method=corr_method,
    )

    ic_factor = ic_operator(
        *test_factors,
        price=price_factor,
        factor_name_list=factor_fields,
    )

    # ---- 5. 运行引擎计算 IC ----
    try:
        ic_data = ic_factor.readData(
            ids=factor_fields,
            dts=dts,
            dt_ruler=dts,
            section_ids=factor_fields,
        )
    except Exception as e:
        return json.dumps({"error": f"IC 计算失败: {e}"}, ensure_ascii=False)

    # ---- 6. 解析 IC 结果 ----
    # CalcIC 输出复合类型，每个单元格为 (IC 值, 截面宽度)
    ic_values = ic_data.map(
        lambda x: x[0] if isinstance(x, tuple) and pd.notnull(x) else np.nan
    )
    breadth_values = ic_data.map(
        lambda x: x[1] if isinstance(x, tuple) and pd.notnull(x) else np.nan
    )

    ic_values = ic_values.dropna(how="all", axis=0)
    breadth_values = breadth_values.reindex(index=ic_values.index)

    if ic_values.empty:
        return json.dumps({"error": "IC 计算结果为空"}, ensure_ascii=False)

    # ---- 7. 汇总统计 ----
    stats = _calc_ic_statistics(ic_values, breadth_values, rolling_avg_period)
    tail_n = min(20, len(ic_values))
    recent_ic = ic_values.tail(tail_n)

    result: Dict[str, Any] = {
        "params": {
            "factor_table": factor_table,
            "factor_fields": factor_fields,
            "start_date": start_date,
            "end_date": end_date,
            "price_table": price_table,
            "price_field": price_field,
            "corr_method": corr_method,
            "period_lookback": period_lookback,
            "rolling_avg_period": rolling_avg_period,
            "num_trade_days": len(dts),
            "num_stocks": len(stock_ids),
        },
        "ic_stats": json.loads(
            stats.to_json(orient="index", force_ascii=False)
        ),
        "ic_time_series_tail": json.loads(
            recent_ic.to_json(orient="index", force_ascii=False, date_format="iso")
        ),
    }

    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

app = mcp.http_app()

if __name__ == "__main__":
    mcp.run(transport="stdio")
