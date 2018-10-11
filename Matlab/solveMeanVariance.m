x = sdpvar(nVar,1);
genYalmipConstraint;% 生成约束条件
%% 设定优化目标
if ~isempty(strfind(Objective.type,'Linear'))
    OptObjective = (Objective.f' * x);
elseif ~isempty(strfind(Objective.type,'Quadratic'))
    if isfield(Objective,'X')
        OptObjective = (x'*(Objective.X*Objective.F*Objective.X'+diag(Objective.Delta))*x + Objective.Mu'*x);
    else
        OptObjective = (x'*Objective.Sigma*x + Objective.Mu'*x);
    end
end
if isfield(Objective,'lambda1')
    OptObjective = OptObjective + Objective.lambda1 * norm(x-Objective.c,1);
end
if isfield(Objective,'lambda2')
    OptObjective = OptObjective + Objective.lambda2 * sum(abs(x-Objective.c_pos)+(x-Objective.c_pos))/2;
end
if isfield(Objective,'lambda3')
    OptObjective = OptObjective + Objective.lambda3 * sum(abs(x-Objective.c_neg)-(x-Objective.c_neg))/2;
end
%% 设置优化选项
OptimOptions = sdpsettings();
if isfield(Options,'Display') && (~isequal(Options.Display,'Default'))
    OptimOptions.verbose = str2double(Options.Display);
end
if isfield(Options,'Solver') && (~isequal(Options.Solver,'Default'))
    OptimOptions.solver = Options.Solver;
end
%% 求解
ResultInfo = optimize(Constraints,OptObjective,OptimOptions);
if ResultInfo.problem==0
    ResultInfo.Status = 1;
else
    ResultInfo.Status = 0;
end
ResultInfo.ErrorCode = ResultInfo.problem;
ResultInfo.Msg = ResultInfo.info;
x = value(x);
yalmip('clear');