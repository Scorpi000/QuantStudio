# -*- coding: utf-8 -*-
"""策略测试"""

from .Strategy import MakeAccount, MakeStrategy, AccountStats, AccountReport
from . import AllocationStrategy
from . import TimingStrategy
try:
    from .OptimizedStrategy import (
        OptimizedStrategy, StrategyAccountFactor, StrategyTargetWeights, StrategyRealizedWeights,
    )
except ModuleNotFoundError as error:
    # 连续优化依赖可选求解器，普通账户在未安装 CVXPY 时仍可使用。
    if error.name != "cvxpy":
        raise
