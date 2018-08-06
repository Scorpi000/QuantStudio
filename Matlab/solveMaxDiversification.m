x = sdpvar(nVar,1);
OptimOptions = sdpsettings();
OptimOptions.solver = 'cplex';
Constraints = [ones(1,nVar)*x==1,x>=zeros(nVar,1)];
if isfield(Objective,'X')
    Sigma = Objective.X*Objective.F*Objective.X'+diag(Objective.Delta);
else
    Sigma = Objective.Sigma;
end
D = diag(1./diag(Sigma).^0.5);
P = D*Sigma*D;
OptObjective = (x'*P*x);
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
x = D*x;
x = x./sum(x);
yalmip('clear');