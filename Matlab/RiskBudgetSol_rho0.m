function w = RiskBudgetSol_rho0(cov,b)
sigma = diag(cov).^0.5;
w = b.^0.5./sigma;
w = w./sum(w);
end