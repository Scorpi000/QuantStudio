x = sdpvar(nVar,1);
OptimOptions = sdpsettings();
genYalmipConstraint;% 设定约束条件
[x,ResultInfo] = MaxSharpeSolverByYalmip(x,Objective,Constraints,OptimOptions);
yalmip('clear');