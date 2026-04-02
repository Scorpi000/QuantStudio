import numpy as np
import pandas as pd
import cvxpy as cvx

from QuantStudio.PortfolioConstructor.CVXPC import CVXPC
from QuantStudio.PortfolioConstructor.BasePC import MeanVarianceObjective, BudgetConstraint, WeightConstraint

np.random.seed(0)
nID = 10
Mask = np.full(shape=(nID,), fill_value=True, dtype=np.bool)
ExpectedReturn = np.random.randn(nID)
Cov = np.cov(np.random.randn(100*nID, nID), rowvar=False)
P0 = np.random.rand(nID)
P0 = P0 / np.sum(P0)
Bmk = np.random.rand(nID)
Bmk = Bmk / np.sum(Bmk)


Objective = MeanVarianceObjective(mask=Mask, expected_return=ExpectedReturn, p0=P0, cov=Cov, args={"ExpectedReturnCoef": 1.0, "RiskAversionCoef": 1})
ConstraintList = [
    WeightConstraint(mask=Mask, args={"UpLimit": 1, "DownLimit": 0}),
    BudgetConstraint(mask=Mask, args={"UpLimit": 1, "DownLimit": 1})
]

PC = CVXPC(mask=Mask, objective=Objective, constraints=ConstraintList, args={"OptimOption": {"solver": cvx.SCIP}})
P, Info = PC.solve()
print(np.where(np.abs(P) > 1e-8, P, 0))
print(Info)

print("===")