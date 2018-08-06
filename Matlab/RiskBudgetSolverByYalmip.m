function [x,ResultInfo] = RiskBudgetSolverByYalmip(nvar,cov_matrix,b)
x = sdpvar(nvar,1);
c = (b'*log(b))-min([1e-4,1/nvar]);
Constraints = [crossentropy(b,x)>=c,x>=zeros(nvar,1)];
OptObjective = (x'*cov_matrix*x);
OptimOptions = sdpsettings();
OptimOptions.verbose = 0;
ResultInfo = optimize(Constraints,OptObjective,OptimOptions);
if ResultInfo.problem==0
    ResultInfo.Status = 1;
else
    ResultInfo.Status = 0;
end
ResultInfo.ErrorCode = ResultInfo.problem;
ResultInfo.Msg = ResultInfo.info;
x = value(x);
x = x./sum(x);
yalmip('clear');
end