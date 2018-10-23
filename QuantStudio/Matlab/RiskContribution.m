function CR = RiskContribution(w,cov)
sigma_w = cov*w;
CR = w.*sigma_w./(w'*sigma_w).^0.5;
end