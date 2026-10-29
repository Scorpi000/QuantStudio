# -*- coding: utf-8 -*-
"""组合优化节点传递的当期问题、账户快照和计算结果。"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AccountSnapshot:
    """决策时点的实际账户；现金不参与证券权重归一化。"""

    datetime: datetime
    ids: tuple[str, ...]
    cash: float
    positions: np.ndarray
    prices: np.ndarray
    available_positions: np.ndarray
    version: int = 0

    def __reduce__(self):
        """跨进程还原时重新校验并保护账户数组。"""
        return type(self), (self.datetime, self.ids, self.cash, self.positions, self.prices, self.available_positions, self.version)

    def __post_init__(self):
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("账户证券标签不能重复")
        for name in ("positions", "prices", "available_positions"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.shape != (len(self.ids),):
                raise ValueError(f"账户 {name} 与完整截面的长度不同")
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @property
    def wealth(self):
        """按当前已知价格计算净资产。"""
        held = self.positions != 0
        if not np.isfinite(self.cash) or not np.isfinite(self.positions).all():
            raise ValueError("账户现金或持仓数量无效")
        if np.any(held & (~np.isfinite(self.prices) | (self.prices <= 0))):
            raise ValueError("已有持仓缺少有效估值价格")
        value = self.cash + np.sum(self.positions[held] * self.prices[held])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("账户净资产必须为有限正数")
        return float(value)

    @property
    def weights(self):
        """返回证券权重，未投资部分保留为现金。"""
        amounts = np.zeros(len(self.ids))
        held = self.positions != 0
        amounts[held] = self.positions[held] * self.prices[held]
        return amounts / self.wealth


@dataclass(frozen=True)
class PeriodComponentData:
    """组件已取得的外生数据，以及需要延后绑定的账户字段。"""

    component: type
    args: dict
    inputs: dict
    ids: tuple[str, ...]
    origin: str
    holding_source: str = "不使用"
    wealth_source: str = "不使用"


@dataclass(frozen=True)
class PeriodProblem:
    """一个完整证券截面的目标、约束和股票池。"""

    datetime: datetime
    ids: tuple[str, ...]
    mask: np.ndarray
    objective: PeriodComponentData
    constraints: tuple[PeriodComponentData, ...]


@dataclass(frozen=True)
class SolveResult:
    """求解状态和诊断；失败与未调仓不共用成功状态。"""

    status: str
    weights: np.ndarray | None
    info: dict = field(default_factory=dict)

    def __reduce__(self):
        """跨进程还原时保留权重数组的只读约定。"""
        return type(self), (self.status, self.weights, self.info)

    def __post_init__(self):
        if self.weights is not None:
            weights = np.array(self.weights, copy=True)
            weights.flags.writeable = False
            object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class PortfolioResult:
    """可被权重与诊断出口共享的独立优化结果。"""

    weights: pd.DataFrame
    diagnostics: dict[datetime, SolveResult]


@dataclass(frozen=True)
class StrategyResult:
    """一次连续账户运行的全部视图。"""

    weights: pd.DataFrame
    realized_weights: pd.DataFrame
    account: pd.DataFrame
    diagnostics: dict[datetime, SolveResult]
    snapshots: dict[datetime, AccountSnapshot]
    orders: pd.DataFrame
    trades: pd.DataFrame
    fees: pd.DataFrame
    execution_diagnostics: dict[datetime, Any]
    final_state: AccountSnapshot
    pending_target: np.ndarray | None
