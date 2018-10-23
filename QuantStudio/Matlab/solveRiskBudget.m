if isfield(Objective,'X')
    Sigma = Objective.X*Objective.F*Objective.X'+diag(Objective.Delta);
else
    Sigma = Objective.Sigma;
end
% 根据问题的规模使用不同的算法
if nVar>=1e3% 规模较大, 变换问题
    [x,ResultInfo] = RiskBudgetSolverByYalmip(nVar,Sigma,Objective.b);
    if ResultInfo.Status==0
        [x,ResultInfo] = RiskBudgetSolverByFmincon(nVar,Sigma,Objective.b);
    end
else% 规模较小或者变换问题的求解失败, 直接求解
    [x,ResultInfo] = RiskBudgetSolverByFmincon(nVar,Sigma,Objective.b);
    if ResultInfo.Status==0
        [x,ResultInfo] = RiskBudgetSolverByYalmip(nVar,Sigma,Objective.b);
    end
end