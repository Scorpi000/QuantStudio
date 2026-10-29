# -*- coding: utf-8 -*-
"""将节点问题绑定为当期数值问题，复用既有求解和约束松弛内核。"""
import numpy as np

from .BasePC import BudgetConstraint, WeightConstraint, MeanVarianceObjective, OptimizationObjective
from .CVXPC import CVXPC
from .OptimizationModel import ObjectiveSpec, ConstraintSpec, _covariance_mask
from .Types import SolveResult


def _build_component(data, mask, ids, snapshot, logger):
    """为一个组件显式绑定账户权重和净资产。"""
    inputs = dict(data.inputs)
    for source, name, attribute in ((data.holding_source, "p0", "weights"),
                                    (data.wealth_source, "wealth", "wealth")):
        if source == "账户状态":
            if snapshot is None:
                raise ValueError(f"{data.component.__name__} 需要实际账户状态")
            inputs[name] = getattr(snapshot, attribute)
        elif source == "外部因子" and inputs.get(name) is None:
            raise ValueError(f"{data.component.__name__} 缺少外部 {name} 输入")
        elif source == "不使用" and inputs.get(name) is not None:
            raise ValueError(f"{name} 已绑定，请显式选择外部因子来源")
    spec_type = ObjectiveSpec if issubclass(data.component, OptimizationObjective) else ConstraintSpec
    result = spec_type(data.component, args=data.args, inputs=inputs).build(mask, ids, lambda value: value, logger)
    for name in ("_Cov", "_FactorCov"):
        covariance = getattr(result, name, None)
        if covariance is not None:
            if name == "_Cov":
                risk_mask = _covariance_mask(mask, getattr(result, "_Bmk", None))
                covariance = covariance[np.ix_(risk_mask, risk_mask)]
            if not covariance.size:
                continue
            scale = max(1., float(np.max(np.abs(covariance))))
            if np.linalg.eigvalsh(covariance).min() < -1e-10 * scale:
                raise ValueError("协方差矩阵必须为半正定矩阵")
    if getattr(result, "_FactorCov", None) is not None:
        for name in ("_FactorData", "_SpecificRisk"):
            value = getattr(result, name, None)
            if value is None or not np.isfinite(value).all():
                raise ValueError("因子风险数据必须完整覆盖证券截面且为有限数值")
    return result


def _constraint_residual(constraint, x):
    """计算统一数学约束的最大违反量。"""
    kind = constraint["type"]
    flat = lambda name: np.asarray(constraint[name]).reshape(-1)
    if kind == "Box":
        return max(np.max(flat("lb") - x, initial=0.), np.max(x - flat("ub"), initial=0.))
    if kind == "LinearIn":
        return np.max(constraint["A"] @ x - flat("b"), initial=0.)
    if kind == "LinearEq":
        return np.max(np.abs(constraint["Aeq"] @ x - flat("beq")), initial=0.)
    if kind == "L1":
        return np.sum(np.abs(x - flat("c"))) - constraint["l"]
    if kind == "Pos":
        return np.maximum(x - flat("c_pos"), 0).sum() - constraint["l_pos"]
    if kind == "Neg":
        return np.maximum(flat("c_neg") - x, 0).sum() - constraint["l_neg"]
    if kind == "Quadratic":
        sigma = constraint.get("Sigma")
        if sigma is None:
            sigma = constraint["X"] @ constraint["F"] @ constraint["X"].T + np.diag(flat("Delta"))
        return float(x @ sigma @ x + flat("Mu") @ x - constraint["q"])
    if kind == "NonZeroNum":
        return np.count_nonzero(np.abs(x - flat("b")) > 1e-6) - constraint["N"]
    raise ValueError(f"无法验证约束数学形式 {kind}")


def solve_period(problem, *, snapshot=None, optim_option=None, tolerance=1e-5,
                 accept_inaccurate=False, logger=None, buy_allowed=None, sell_allowed=None):
    """绑定并求解一个截面，账户物理限制始终为不可释放约束。

    Args:
        problem: 已准备的当期问题。
        snapshot: 实际账户快照；None 表示独立求解。
        optim_option: CVXPY 求解选项。
        tolerance: 生效约束的最大允许残差。
        accept_inaccurate: 是否接受求解器的不精确最优状态。
        logger: 框架日志对象。
        buy_allowed: 决策时已知的买入许可。
        sell_allowed: 决策时已知的卖出许可。

    Returns:
        包含权重或失败诊断的 SolveResult。输入无效时抛出异常。
    """
    ids = problem.ids
    mask = np.asarray(problem.mask, dtype=bool).copy()
    if snapshot is not None:
        if snapshot.ids != ids:
            raise ValueError("账户与问题的证券标签或顺序不同")
        if not issubclass(problem.objective.component, MeanVarianceObjective):
            raise ValueError("首版连续账户只支持均值方差目标")
        w0 = snapshot.weights
        mask |= snapshot.positions != 0
    if not mask.any():
        # 无候选且已清仓时，只有真实满足所有约束的现金组合才可成功。
        _build_component(problem.objective, mask, ids, snapshot, logger)
        constraints = [_build_component(data, mask, ids, snapshot, logger) for data in problem.constraints]
        if all(_constraint_residual(item, np.array([])) <= tolerance
               for constraint in constraints for item in constraint.genConstraint()):
            return SolveResult("success", np.zeros(len(ids)), {"status": 1, "ReleasedConstraint": []})
        return SolveResult("failed", None, {"status": 0, "msg": "可选证券为空"})
    objective = _build_component(problem.objective, mask, ids, snapshot, logger)
    constraints = [_build_component(data, mask, ids, snapshot, logger) for data in problem.constraints]
    if snapshot is not None:
        available = np.asarray(snapshot.available_positions)
        if (not np.isfinite(available).all() or np.any(snapshot.positions < 0)
                or np.any(available < 0) or np.any(available > snapshot.positions)):
            raise ValueError("账户可卖数量无效")
        lower = np.zeros(len(ids))
        held = snapshot.positions > 0
        lower[held] = ((snapshot.positions[held] - available[held]) * snapshot.prices[held]
                       / snapshot.wealth)
        upper = np.full(len(ids), np.inf)
        upper[~problem.mask] = w0[~problem.mask]
        if buy_allowed is not None:
            upper[~buy_allowed] = np.minimum(upper[~buy_allowed], w0[~buy_allowed])
        if sell_allowed is not None:
            lower[~sell_allowed] = w0[~sell_allowed]
        constraints.extend([
            WeightConstraint(mask, up_limit=upper, down_limit=lower),
            BudgetConstraint(mask, args={"UpLimit": 1., "DownLimit": 0.}),
        ])
    CVXPC.validate_model(type(objective), tuple(type(item) for item in constraints))
    weights, info = CVXPC(mask, objective, constraints,
                        args={"OptimOption": optim_option or {}}, logger=logger).solve()
    allowed_status = {"optimal", "optimal_inaccurate"} if accept_inaccurate else {"optimal"}
    if (weights is None or info.get("status") != 1 or info.get("msg") not in allowed_status
            or not np.isfinite(weights[mask]).all()):
        return SolveResult("failed", None, info)
    released = {int(item.split("-", 1)[0]) for item in info.get("ReleasedConstraint", [])}
    residual = 0.
    for index, constraint in enumerate(constraints):
        if index not in released:
            for math_constraint in constraint.genConstraint():
                value = float(_constraint_residual(math_constraint, weights[mask]))
                residual = max(residual, value) if np.isfinite(value) else np.inf
    info = dict(info, max_residual=residual,
                constraint_origins=[data.origin for data in problem.constraints])
    if not np.isfinite(residual) or residual > tolerance:
        return SolveResult("failed", None, dict(info, status=0, msg="生效约束残差超限"))
    return SolveResult("relaxed" if released else "success", np.where(mask, weights, 0.), info)
