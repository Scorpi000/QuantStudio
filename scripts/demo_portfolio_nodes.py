# -*- coding: utf-8 -*-
"""演示组合优化原生节点和实际持仓连续优化。

实现逻辑：
    使用两只虚拟证券与固定协方差组装均值方差目标、预算和权重约束，
    先独立求解；再加入绑定账户状态的换手约束，运行收盘决策、下一交易日
    开盘执行的共享现金账户。成交额上限使第一笔订单部分成交，后续决策
    使用真实数量和价格漂移后的权重。

使用方法：
    在项目根目录使用项目 Python 环境运行：
    ``python scripts/demo_portfolio_nodes.py``。
    示例完全使用内存数据，不连接数据库，不发送订单，不写入账户或数据文件。
"""
import datetime as dt
import logging

import numpy as np
import pandas as pd

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Risk.api import DataRiskTable
from QuantStudio.PortfolioConstructor.api import (
    MeanVarianceObjectiveNode, BudgetConstraintNode,
    WeightConstraintNode, TurnoverConstraintNode, PortfolioProblem, CVXPortfolio, PortfolioWeights,
)
from QuantStudio.BackTest.Strategy.api import OptimizedStrategy


def main():
    """组装并运行两种执行方式，打印目标和真实成交的差异。"""
    logging.getLogger("QS").setLevel(logging.WARNING)
    ids = ["A", "B"]
    dates = [dt.datetime(2026, 1, day) for day in (1, 2, 3, 4)]
    alpha = DataFactor(pd.DataFrame([[.6, .2], [-1., 2.], [-1., 2.], [-1., 2.]],
                                    index=dates, columns=ids))
    close = DataFactor(pd.DataFrame([[10., 10.], [12., 10.], [12., 10.], [12., 10.]],
                                    index=dates, columns=ids))
    risk = DataRiskTable(pd.DataFrame(np.eye(2), index=ids, columns=ids))
    objective = MeanVarianceObjectiveNode.template(args={"ExpectedReturnCoef": 1.}).bind(
        expected_return=alpha, risk_table=risk).build()
    budget = BudgetConstraintNode.template(args={"UpLimit": .8, "DownLimit": .8}).build()
    limit = WeightConstraintNode.template().build()
    problem = PortfolioProblem.template().bind(objective=objective, constraints=(budget, limit)).build()
    solution = CVXPortfolio.template(args={"SectionIDs": ids}).bind(problem=problem).build()
    weights = PortfolioWeights.template().bind(source=solution).build()
    print("独立求解的目标权重：")
    print(weights.readData(ids=ids, section_ids=ids, dts=dates).round(6))

    turnover = TurnoverConstraintNode.template(args={
        "HoldingSource": "账户状态", "AmtMultiple": 0., "UpLimit": 1.,
    }).build()
    rolling = PortfolioProblem.template().bind(
        objective=objective, constraints=(budget, limit, turnover)).build()
    account = OptimizedStrategy.template(args={
        "StartDT": dates[0], "SectionIDs": ids, "InitialCash": 1000.,
    }).bind(problem=rolling, mark_price=close, execution_price=DataFactor(10.),
            buy_amt_limit=DataFactor(200.)).build()
    result, = Engine().run([account], FactorContext(DTRuler=dates, SectionIDs=ids),
                           fwd_data_list=[FactorLocalContext(DTs=dates, IDs=ids)])
    print("\n每天实际成交的数量：")
    print(result.trades.round(6))
    print("\n收盘实际证券权重（剩余为现金）：")
    print(result.realized_weights.round(6))
    print("\n第二次决策使用的实际持仓权重：")
    print(pd.Series(result.snapshots[dates[1]].weights, index=ids).round(6))
    print("\n最后一日收盘计划尚未执行：", result.pending_target is not None)


if __name__ == "__main__":
    main()
