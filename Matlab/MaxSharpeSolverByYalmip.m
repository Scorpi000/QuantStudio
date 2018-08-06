function [y,ResultInfo] = MaxSharpeSolverByYalmip(x,Objective,Constraints,Options)
% 求解分子可取到的最大值和最小值
OptObjective = (Objective.f' * x);
ResultInfo = optimize(Constraints,OptObjective);
if ResultInfo.problem==0
    NumeratorMin = Objective.f'*value(x)+Objective.f0;
else
    NumeratorMin = -inf;
end
OptObjective = (-Objective.f' * x);
ResultInfo = optimize(Constraints,OptObjective);
if ResultInfo.problem==0
    NumeratorMax = Objective.f'*value(x)+Objective.f0;
else
    NumeratorMax = inf;
end
% 求解
if isfield(Objective,'X')
    Sigma = (Objective.X*Objective.F*Objective.X'+diag(Objective.Delta));
else
    Sigma = Objective.Sigma;
end
if isfield(Options,'Display') && (~isequal(Options.Display,'Default'))
    if isequal(Options.Display,'0')
        OptimOptions = optimset('Display','off');
    elseif isequal(Options.Display,'1')
        OptimOptions = optimset('Display','final');
    else
        OptimOptions = optimset('Display','iter');
    end
else
    OptimOptions = optimset('Display','notify');
end
    function y = ObjectFun(r)
        y = solveMinDenominator(r,x,Sigma,Objective.Mu,Objective.f,Objective.f0,Constraints);
        y = -r/(y'*Sigma*y+Objective.Mu'*y+Objective.q)^0.5;
    end
[r,~,exitflag] = fminbnd(@ObjectFun,NumeratorMin,NumeratorMax,OptimOptions);
if exitflag==1
    y = solveMinDenominator(r,x,Sigma,Objective.Mu,Objective.f,Objective.f0,Constraints);
    ResultInfo = struct('ErrorCode',0,'Status',1,'Msg','Successfully solved');
    return;
end
y = value(x);
ResultInfo = struct('ErrorCode',exitflag,'Status',0,'Msg','Failed');
if exitflag==0
    ResultInfo.Msg = 'Maximum iterations exceeded';
elseif exitflag==-2
    ResultInfo.Msg = 'Infeasible problem';
elseif exitflag==-1
    ResultInfo.Msg = 'Unknown error';
end
end

function y = solveMinDenominator(r,x,Sigma,Mu,f,f0,Constraints)
OptObjective = (x'*Sigma*x + Mu'*x);
NewConstraints = Constraints + [f'*x==r-f0];
a = optimize(NewConstraints,OptObjective);
y = value(x);
end