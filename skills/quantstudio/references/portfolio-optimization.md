# 组合优化

组合优化位于 `QuantStudio.PortfolioConstructor`，核心是"目标函数 + 约束条件 → 求解器"的设计模式。

## 架构设计

```
OptimizationObjective.genObjective()  →  {'type': 'Quadratic', 'Sigma': ..., 'Mu': ...}
Constraint.genConstraint() × N        →  [{'type': 'Box', ...}, {'type': 'LinearEq', ...}]
PortfolioConstructor.solve()          →  (权重数组, 求解信息)
```

## 优化目标

### MeanVarianceObjective — 均值方差优化

Markowitz 均值方差模型：

```python
from QuantStudio.PortfolioConstructor.BasePC import MeanVarianceObjective

Objective = MeanVarianceObjective(
    mask=Mask,                    # bool array(n)，股票池
    expected_return=None,         # array(n)，预期收益
    p0=None,                      # array(n)，初始组合
    bmk=None,                     # array(n)，基准组合
    cov=None,                     # array(n,n)，证券协方差阵（方式一）
    # 或使用因子形式（方式二）：
    factor_cov=None,              # array(k,k)，因子协方差
    factor_data=None,             # array(n,k)，因子暴露
    specific_risk=None,           # array(n)，特异性风险
    args={
        "Benchmark": False,            # 是否相对基准优化
        "ExpectedReturnCoef": 0.0,     # 收益项系数 γ（0=纯风险最小化）
        "RiskAversionCoef": 1.0,       # 风险厌恶系数 λ
        "TurnoverPenaltyCoef": 0.0,    # 双边换手惩罚 λ₁
        "BuyPenaltyCoef": 0.0,         # 买入惩罚 λ₂
        "SellPenaltyCoef": 0.0,        # 卖出惩罚 λ₃
    }
)
```

### RiskBudgetObjective — 风险预算优化

```python
from QuantStudio.PortfolioConstructor.BasePC import RiskBudgetObjective

Objective = RiskBudgetObjective(
    mask=Mask,
    budget=None,       # 风险预算，None=等风险预算（风险平价）
    cov=None,
    # 或 factor_cov + factor_data + specific_risk
)
```

### MaxDiversificationObjective — 最大分散化优化

```python
from QuantStudio.PortfolioConstructor.BasePC import MaxDiversificationObjective
```

## 约束条件

所有约束继承自 `Constraint`，都有 `DropPriority` 参数（默认 -1=不可松弛），求解失败时按优先级从高到低松弛。

### BudgetConstraint — 预算约束

```python
from QuantStudio.PortfolioConstructor.BasePC import BudgetConstraint

# 全额投资
BudgetConstraint(mask=Mask, args={"UpLimit": 1, "DownLimit": 1})
# 相对基准偏离
BudgetConstraint(mask=Mask, bmk=Bmk, args={"UpLimit": 0.05, "DownLimit": -0.05, "Benchmark": True})
```

### WeightConstraint — 权重约束

```python
from QuantStudio.PortfolioConstructor.BasePC import WeightConstraint

# 纯多头
WeightConstraint(mask=Mask, up_limit=1, down_limit=0)
# 个股权重上限
WeightConstraint(mask=Mask, up_limit=0.1)
# 相对基准偏离
WeightConstraint(mask=Mask, bmk=Bmk, up_limit=0.02, down_limit=-0.02, args={"Benchmark": True})
```

### FactorExposeConstraint — 因子暴露约束

```python
from QuantStudio.PortfolioConstructor.BasePC import FactorExposeConstraint

# 行业中性
FactorExposeConstraint(mask=Mask, factor_data=IndustryDummy, bmk=Bmk,
    args={"UpLimit": 0, "DownLimit": 0, "Benchmark": True, "FactorType": "类别型"})
```

### VolatilityConstraint — 波动率约束

```python
from QuantStudio.PortfolioConstructor.BasePC import VolatilityConstraint

# 年化波动率 ≤ 10%
VolatilityConstraint(mask=Mask, cov=Cov, args={"UpLimit": 0.10})
# 跟踪误差 ≤ 3%
VolatilityConstraint(mask=Mask, bmk=Bmk, cov=Cov, args={"UpLimit": 0.03, "Benchmark": True})
```

### TurnoverConstraint — 换手率约束

```python
from QuantStudio.PortfolioConstructor.BasePC import TurnoverConstraint

TurnoverConstraint(mask=Mask, p0=P0,
    args={"ConstraintType": "总换手限制", "UpLimit": 0.5})
```

### NonZeroNumConstraint — 持仓数量约束

```python
from QuantStudio.PortfolioConstructor.BasePC import NonZeroNumConstraint
# 注意：引入 0-1 整数变量，转为 MIP 问题，求解难度大增

NonZeroNumConstraint(mask=Mask, args={"UpLimit": 50})
```

## CVXPC — 凸优化求解器

```python
from QuantStudio.PortfolioConstructor.CVXPC import CVXPC

PC = CVXPC(mask=Mask, objective=Objective, constraints=ConstraintList,
    args={"OptimOption": {"solver": cvx.SCIP, "verbose": True}})

Portfolio, Info = PC.solve()
# Portfolio: array(n)，最优权重，不在股票池的为 NaN
# Info: {"status": 1/0, "msg": ..., "solver_name": ..., "solve_time": ..., ...}
```

### 求解器选择

| 求解器 | 适用场景 |
|--------|----------|
| `cvx.SCIP` | MIP（含 NonZeroNum 约束） |
| `cvx.CLARABEL` | 锥规划（RiskBudget 等） |
| `cvx.OSQP` | 二次规划（ADMM 算子分裂） |
| `cvx.ECOS` | 锥规划（内点法） |

### 约束松弛机制

约束有 `DropPriority`：求解失败时，从最高优先级开始逐级松弛直到找到可行解。`Info["ReleasedConstraint"]` 记录了被松弛的约束。

## 模型对比

| 模型 | 输入要求 | 特点 |
|------|----------|------|
| 最小方差 | 仅需协方差 | 对参数估计较不敏感，持仓可能集中 |
| 均值方差 | 预期收益 + 协方差 | 收益估计误差影响大 |
| 风险预算 | 协方差 + 风险预算 | 风险分散化，权重均衡 |
