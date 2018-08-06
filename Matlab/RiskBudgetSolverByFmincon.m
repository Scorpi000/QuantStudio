function [x,ResultInfo] = RiskBudgetSolverByFmincon(nvar,cov_matrix,b)
x0 = RiskBudgetSol_rho0(cov_matrix, b);
OptObjective = @(x) RiskBudgetObjectiveFun2(x, cov_matrix, b);
OptimOptions = optimoptions('fmincon','Algorithm','interior-point','Display','off','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'HessianFcn',@(x,lambda) RiskBudgetHessianFcn(x,lambda,cov_matrix,b));
[x,fval,exitflag,output] = fmincon(OptObjective,x0,[],[],ones(1,nvar),1,zeros(nvar,1),[],[],OptimOptions);
ResultInfo = struct();
if exitflag>0
    ResultInfo.Status = 1;
else
    ResultInfo.Status = 0;
end
ResultInfo.ErrorCode = exitflag;
ResultInfo.Msg = output.message;
end