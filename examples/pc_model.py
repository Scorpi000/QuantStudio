# -*- coding: utf-8 -*-
"""组合收益目标、预算、逐期权重上限和换手约束，生成图中的权重因子。"""
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor
from QuantStudio.PortfolioConstructor.api import (
    OptimizationModel, ObjectiveSpec, ConstraintSpec, MakeOptimizedPortfolio,
    MeanVarianceObjective, BudgetConstraint, WeightConstraint, TurnoverConstraint,
)


def main():
    ids = ["A", "B"]
    dts = [dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2)]
    cov = pd.DataFrame(np.diag([1., 4.]), index=ids, columns=ids)
    alpha = DataFactor(pd.DataFrame([[.1, .2], [.3, .1]], index=dts, columns=ids))
    cap = DataFactor(pd.DataFrame([[.35, 1.], [.8, 1.]], index=dts, columns=ids))
    p0 = DataFactor(pd.DataFrame([[.3, .7]] * 2, index=dts, columns=ids))

    model = OptimizationModel(
        objective=ObjectiveSpec(MeanVarianceObjective,
            args={"ExpectedReturnCoef": 1., "RiskAversionCoef": 1.},
            inputs={"expected_return": alpha, "cov": cov}),
        constraints=[
            ConstraintSpec(BudgetConstraint, args={"UpLimit": 1., "DownLimit": 1.}),
            ConstraintSpec(WeightConstraint, inputs={"up_limit": cap, "down_limit": 0.}),
            ConstraintSpec(TurnoverConstraint, args={"AmtMultiple": 0., "UpLimit": .2},
                           inputs={"p0": p0}),
        ],
    )
    weights = MakeOptimizedPortfolio(model)(factor_args={"Name": "目标权重", "SectionIDs": ids})
    result = weights.readData(ids=ids, section_ids=ids, dts=dts, dt_ruler=dts)
    np.testing.assert_allclose(result.values, [[.35, .65], [.4, .6]], atol=1e-5)
    print(result)
    return result


if __name__ == "__main__":
    main()
