function H = RiskBudgetHessianFcn(w,lambda,cov,b)
% Hessian of objective
nVar = length(w);
u = cov*w;
ub = u./b;
wb = w./b;
Var = w'*u;

g = 4./Var .* (nVar.*w.*ub.^2 + nVar*cov*(u.*wb.^2) - nVar./Var.*((w.^2)'*(ub.^2)).*u - (w'*ub).*(ub + cov*wb - (w'*ub)./Var.*u));

H = nVar.*diag(ub.^2) + 2*nVar*diag(wb.*ub)*cov - (ub*ub') - ub*(cov*wb)' - (w'*ub)*diag(1./b)*cov ...
    + 2*nVar*cov*diag(wb.*ub) + nVar*cov*diag(wb.^2)*cov - (cov*wb)*(ub + cov*wb)' ...
    - (w'*ub)*cov*diag(1./b) - 2.*nVar./Var.*(u*(w.*ub.^2 + cov*(u.*wb.^2))') ...
    - nVar/Var*((w.^2)'*(ub.^2))*(cov - 2./Var.*(u*u')) + 2./Var.*(w'*ub).*((u*ub') + (u*(cov*wb)')) ...
    + 1./Var.*(w'*ub).^2.*(cov - 2./Var.*(u*u'));
H = 4./Var.*H - 2./Var.*(g*u');
end