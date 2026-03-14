# coding=utf-8
import sys
sys.path.append(r"C:\Users\hst\Project\vectorbt")
import time
from typing import Tuple

import numpy as np
import pandas as pd

from QuantStudio.Tools.StrategyTestFun import backtestPortfolioStrategyWithMargin, backtestPortfolioStrategy_pd, backtestPortfolioStrategy
from QuantStudio.Tools.DataPreprocessingFun import numpy_ffill


def backtestPortfolioStrategy_v0(portfolio: np.ndarray, price: np.ndarray, fee:float | np.ndarray=0.0) -> Tuple[np.ndarray, np.ndarray]:
    Mask = (~ np.all(np.isnan(portfolio), axis=1))
    MaskedReturn, MaskedTurnover = backtestPortfolioStrategyWithMargin(portfolio[Mask], price[Mask], fee=fee)
    if np.all(Mask):
        return np.cumprod(1 + MaskedReturn), MaskedTurnover
    MaskedNV = np.full(shape=Mask.shape, fill_value=np.nan)
    MaskedNV[Mask] = np.cumprod(1 + MaskedReturn)
    MaskedNV = numpy_ffill(MaskedNV)
    MaskedNV[np.isnan(MaskedNV)] = 1
    portfolio = np.r_[np.zeros((1, portfolio.shape[1])), portfolio]
    portfolio = numpy_ffill(portfolio, axis=0, limit=None)
    MaskedPrice = price.copy()
    MaskedPrice[~Mask] = np.nan
    MaskedPrice[0] = price[0]
    MaskedPrice = numpy_ffill(MaskedPrice, axis=0, limit=None)
    MaskedReturn = np.zeros_like(MaskedPrice)
    MaskedReturn[1:] = price[1:] / MaskedPrice[:-1] - 1
    MaskedReturn = np.where(np.isinf(MaskedReturn), np.sign(MaskedReturn), MaskedReturn)
    MaskedReturn[np.isnan(MaskedReturn)] = 0.0
    PortfolioNV = 1 + np.nansum(portfolio[:-1] * MaskedReturn, axis=1)
    PortfolioNV = MaskedNV * PortfolioNV
    Turnover = np.zeros(shape=PortfolioNV.shape)
    Turnover[Mask] = MaskedTurnover
    return PortfolioNV, Turnover


np.random.seed(0)
nDT, nID = 1000, 3000
Portfolio = np.random.rand(nDT, nID)
Portfolio = Portfolio / np.sum(Portfolio, axis=1, keepdims=True)
Mask = (np.random.randn(nDT) > 0)
Portfolio[Mask] = np.nan
# print(Portfolio)
Price = np.cumprod(np.exp(0.03 * np.random.randn(nDT, nID)), axis=0)


# numpy v0 版本
StartT = time.perf_counter()
NV1, Turnover1 = backtestPortfolioStrategy_v0(Portfolio, Price)
# Mask = (~ np.all(np.isnan(Portfolio), axis=1))
# Return1, Turnover1 = testPortfolioStrategy(Portfolio[Mask], Price[Mask])
# NV1 = np.cumprod(Return1 + 1)
print(time.perf_counter() - StartT)
print(NV1[-1])

# pandas 版本
iTmp = pd.DataFrame(Portfolio).dropna(how="all", axis=0)
StartT = time.perf_counter()
NV2 = backtestPortfolioStrategy_pd(iTmp, pd.DataFrame(Price))
print(time.perf_counter() - StartT)
print(NV2.iloc[-1])

# numpy 版本
StartT = time.perf_counter()
NV3, Turnover3 = backtestPortfolioStrategy(Portfolio, Price)
print(time.perf_counter() - StartT)
print(NV3[-1])

# # vectorbt 版本
# import vectorbt as vbt
# from vectorbt.portfolio.enums import SizeType

# # 转换为 pandas 格式  
# price_df = pd.DataFrame(Price)  
# weights_df = pd.DataFrame(Portfolio)  

# StartT = time.perf_counter()
# # 创建投资组合进行回测
# pf = vbt.Portfolio.from_orders(  
#     price_df,  
#     size=weights_df,  
#     size_type=SizeType.TargetPercent,
#     cash_sharing=True,  
#     call_seq='auto'  # 自动处理再平衡顺序  
# )
# # 获取回测结果  
# total_return = pf.total_return()
# print(time.perf_counter() - StartT)
# print(total_return)

print("===")