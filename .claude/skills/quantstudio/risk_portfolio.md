# 风险模型与组合优化

## 风险模型

### 导入

```python
from QuantStudio.Risk.api import HDF5FRDB, HDF5RDB
```

### 风险数据模型

```
风险库 (RiskDB) → 风险表 (RiskTable) → 时点数据 (风险矩阵)
```

- 风险矩阵 Panel: items=时点, major_axis=证券代码, minor_axis=证券代码
- 每个时点数据为 DataFrame(index=证券代码, columns=证券代码)

多因子风险模型结构:
- `FactorCovMatrix` — 因子协方差阵 DataFrame(index=因子, columns=因子)
- `FactorExposure` — 因子暴露 DataFrame(index=证券代码, columns=因子)
- `SpecificRisk` — 特异性风险 Series(index=证券代码)
- `CovMatrix` — 证券协方差阵 DataFrame(index=证券代码, columns=证券代码)
- `FactorReturn` / `SpecificReturn` — 收益分解

## 组合优化

### 导入

```python
from QuantStudio.PortfolioConstructor.api import (
    MeanVarianceObjective, MaxDiversificationObjective, RiskBudgetObjective,
    BudgetConstraint, WeightConstraint, FactorExposeConstraint,
    VolatilityConstraint, ExpectedReturnConstraint,
    TurnoverConstraint, NonZeroNumConstraint, CVXPC
)
```

### 优化目标

- **MeanVarianceObjective**: $\max \gamma\mu^T w - \frac{\lambda}{2}w^T\Sigma w - TC(w)$
- **MaxDiversificationObjective**: 最大化分散化
- **RiskBudgetObjective**: $\min \sum (RC_i - b_i R(w))^2$，当 $b_i=1/n$ 时为风险平价

### 约束条件

| 约束类 | 说明 |
|--------|------|
| `BudgetConstraint` | $\mathbf{1}^T w = a$ |
| `WeightConstraint` | $a \le w-w_b \le b$ |
| `FactorExposeConstraint` | $x^T(w-w_b)=a$，风格中性 |
| `VolatilityConstraint` | $(w-w_b)^T\Sigma(w-w_b) \le \sigma^2$ |
| `ExpectedReturnConstraint` | $\mu^T w \ge a$ |
| `TurnoverConstraint` | 换手率约束 |
| `NonZeroNumConstraint` | $\text{nnz}(w) \le N$ |

### 求解器

```python
from QuantStudio.PortfolioConstructor.api import CVXPC
# 基于 cvxpy 的凸优化求解器（需安装 cvxpy）
```
