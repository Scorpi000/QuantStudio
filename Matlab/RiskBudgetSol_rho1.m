function w = RiskBudgetSol_rho1(cov,b)
sigma = diag(cov).^0.5;
w = b./sigma;
w = w./sum(w);
end