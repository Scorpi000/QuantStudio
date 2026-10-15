# -*- coding: utf-8 -*-
"""使用内存数据，通过计算图生成均值方差目标权重。"""
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import DataFactor
from QuantStudio.PortfolioConstructor.api import MakeMeanVariancePortfolio


def main():
    ids = ["A", "B"]
    dts = [dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2)]
    cov = pd.DataFrame(np.diag([1., 4.]), index=ids, columns=ids)
    expected_return = DataFactor(pd.DataFrame([[.1, .2], [.3, .1]], index=dts, columns=ids))

    operator = MakeMeanVariancePortfolio(cov=cov, expected_return_coef=1., risk_aversion_coef=1.)
    weights = operator(expected_return, up_limit=1., down_limit=0.,
                       factor_args={"Name": "目标权重", "SectionIDs": ids})
    result = weights.readData(ids=ids, dts=dts, section_ids=ids, dt_ruler=dts)
    np.testing.assert_allclose(result.values, [[.78, .22], [.84, .16]], atol=1e-5)
    print(result)


if __name__ == "__main__":
    main()
