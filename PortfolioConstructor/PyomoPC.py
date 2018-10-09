# -*- coding: utf-8 -*-
"""基于 pyomo 模块的投资组合构造器(TODO)"""
import pandas as pd
import numpy as np
import pyomo

from .BasePC import PortfolioConstructor

MeanVarianceModel = pyomo.environ.AbstractModel()

model.m = Param(within=NonNegativeIntegers)
model.n = Param(within=NonNegativeIntegers)

model.I = RangeSet(1, model.m)
model.J = RangeSet(1, model.n)

model.a = Param(model.I, model.J)
model.b = Param(model.I)
model.c = Param(model.J)

model.x = Var(model.J, domain=NonNegativeReals)

def obj_expression(model):
    return summation(model.c, model.x)

model.OBJ = Objective(rule=obj_expression)

def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i,j] * model.x[j] for j in model.J) >= model.b[i]


class PyomoPC(PortfolioConstructor):
    """基于 pyomo 模块的投资组合构造器"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        return
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        ErrorCode = self._MatlabEng.connect(engine_name=None,option="-desktop")
        if ErrorCode!=1: return np.zeros(nvar)+np.nan
        Eng = self._MatlabEng.acquireEngine()
        Eng.clear(nargout=0)
        self._transmitObjective(prepared_objective)
        for iType in prepared_constraints: self._transmitConstraint(prepared_constraints[iType])
        Eng.workspace['nVar'] = float(nvar)
        Eng.workspace["Options"] = prepared_option
        getattr(Eng, self._MatlabScript)(nargout=0)
        ResultInfo = Eng.workspace['ResultInfo']
        if ResultInfo["Status"]==1:
            x = Eng.workspace['x']
            x = np.array(x).reshape(nvar)
        else:
            x = None
        self._MatlabEng.releaseEngine()
        return (x, ResultInfo)