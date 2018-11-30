# -*- coding: utf-8 -*-
"""基于 MATLAB 的投资组合构造器"""
import os
from multiprocessing import Lock

import numpy as np
import pandas as pd
import matlab
import matlab.engine

from .BasePC import PortfolioConstructor
from QuantStudio import __QS_MainPath__, __QS_Error__

class MatlabPC(PortfolioConstructor):
    """基于 MATLAB 的投资组合构造器"""
    def __init__(self, matlab_eng=None, lock=None, sys_args={}, config_file=None, **kwargs):
        self._Option = kwargs.get("option", "-desktop")
        self._MatlabEng = matlab_eng
        self._EngLock = lock
        self._MatlabScript = ""
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if (self._MatlabEng is not None) and (self._MatlabEng._check_matlab()):
            with self._EngLock:
                if not self._MatlabEng.matlab.engine.isEngineShared(): self._MatlabEng.matlab.engine.shareEngine()
                state["_MatlabEng"] = self._MatlabEng.matlab.engine.engineName()
        else:
            state["_MatlabEng"] = None
        return state
    def __setstate__(self, state):
        super().__setstate__(state)
        if self._MatlabEng is not None:
            self._MatlabEng = matlab.engine.connect_matlab(name=self._MatlabEng)
    # 传递优化目标变量
    def _transmitObjective(self, prepared_objective):
        MatlabVar = {}
        for iVar,iValue in prepared_objective.items():
            if isinstance(iValue, np.ndarray): MatlabVar[iVar] = matlab.double(iValue.tolist())
            elif isinstance(iValue, str): MatlabVar[iVar] = iValue
            else: MatlabVar[iVar] = matlab.double([iValue])
        self._MatlabEng.workspace['Objective'] = MatlabVar
        return 0
    # 传递约束条件变量
    def _transmitConstraint(self, prepared_constraints):
        if isinstance(prepared_constraints, dict):
            MatlabVar = {}
            for iVar, iValue in prepared_constraints.items():
                if isinstance(iValue,np.ndarray): MatlabVar[iVar] = matlab.double(iValue.tolist())
                elif isinstance(iValue,str): MatlabVar[iVar] = iValue
                else: MatlabVar[iVar] = matlab.double([iValue])
            self._MatlabEng.workspace[prepared_constraints['type']+'_Constraint'] = MatlabVar
            return 0
        else:
            MatlabVar = []
            for i, iConstraint in enumerate(prepared_constraints):
                iMatlabVar = {}
                for jVar,jValue in iConstraint.items():
                    if isinstance(jValue, np.ndarray): iMatlabVar[jVar] = matlab.double(jValue.tolist())
                    elif isinstance(jValue,str): iMatlabVar[jVar] = jValue
                    else: iMatlabVar[jVar] = matlab.double([jValue])
                MatlabVar.append(iMatlabVar)
            self._MatlabEng.workspace[MatlabVar[0]['type']+'_Constraint'] = MatlabVar
            return 0
    # 启动 MATLAB
    def _startEng(self):
        if self._MatlabEng is None:
            self._MatlabEng = matlab.engine.start_matlab(option=self._Option)
        if self._EngLock is None: self._EngLock = Lock()
        return 0
    def init(self):
        if self.OptimObjective.Type=="均值方差目标": self._MatlabScript = "solveMeanVariance"
        elif self.OptimObjective.Type=="风险预算目标": self._MatlabScript = "solveRiskBudget"
        elif self.OptimObjective.Type=="最大夏普率目标": self._MatlabScript = "solveMaxSharpe"
        elif self.OptimObjective.Type=="最大分散化目标": self._MatlabScript = "solveMaxDiversification"
        else: raise __QS_Error__("不支持的优化目标: '%s'" % self.OptimObjective.Type)
        return super().init()
    def _genOption(self):
        Options = {"Display":"0", "Solver":"Default"}
        Options.update(self.OptimOption)
        return Options
    # 调用 MATLAB 求解优化问题
    def _solve(self, nvar, prepared_objective, prepared_constraints, prepared_option):
        self._startEng()
        with self._EngLock:
            self._MatlabEng.cd(__QS_MainPath__+os.sep+"Matlab")
            self._MatlabEng.clear(nargout=0)
            self._transmitObjective(prepared_objective)
            for iType in prepared_constraints: self._transmitConstraint(prepared_constraints[iType])
            self._MatlabEng.workspace["nVar"] = float(nvar)
            self._MatlabEng.workspace["Options"] = prepared_option
            getattr(self._MatlabEng, self._MatlabScript)(nargout=0)
            ResultInfo = self._MatlabEng.workspace['ResultInfo']
            if ResultInfo["Status"]==1:
                x = self._MatlabEng.workspace['x']
                x = np.array(x).reshape(nvar)
            else:
                x = None
        return (x, ResultInfo)