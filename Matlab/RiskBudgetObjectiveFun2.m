function [f,g,H] = RiskBudgetObjectiveFun2(w,cov,b)
% Calculate objective f
nVar = length(w);
u = cov*w;
ub = u./b;
Var = w'*u;
wb = ones(nVar,nVar)*diag(w.*ub./(Var^0.5));
f = sum(sum((wb-wb').^2));
if nargout==1
    return;
end

% gradient required
wb = w./b;
g = 4./Var .* (nVar.*w.*ub.^2 + nVar*cov*(u.*wb.^2) - nVar./Var.*((w.^2)'*(ub.^2)).*u - (w'*ub).*(ub + cov*wb - (w'*ub)./Var.*u));
if nargout==2
    return;
end

% Hessian required
H = nVar.*diag(ub.^2) + 2*nVar*diag(wb.*ub)*cov - (ub*ub') - ub*(cov*wb)' - (w'*ub)*diag(1./b)*cov ...
    + 2*nVar*cov*diag(wb.*ub) + nVar*cov*diag(wb.^2)*cov - (cov*wb)*(ub + cov*wb)' ...
    - (w'*ub)*cov*diag(1./b) - 2.*nVar./Var.*(u*(w.*ub.^2 + cov*(u.*wb.^2))') ...
    - nVar/Var*((w.^2)'*(ub.^2))*(cov - 2./Var.*(u*u')) + 2./Var.*(w'*ub).*((u*ub') + (u*(cov*wb)')) ...
    + 1./Var.*(w'*ub).^2.*(cov - 2./Var.*(u*u'));
H = 4./Var.*H - 2./Var.*(g*u');
end