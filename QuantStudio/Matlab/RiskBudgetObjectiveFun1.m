function y = RiskBudgetObjectiveFun1(w,cov,b)
CR = RiskContribution(w,cov);
y = sum((CR-b.*sum(CR)).^2);
end