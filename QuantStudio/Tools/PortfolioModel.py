# coding=utf-8
"""投资组合模型"""
import numpy as np
from scipy.optimize import minimize_scalar
import cvxpy as cvx

# 组合特征
def calcPortfolioCharacter(cov, w):
    return np.dot(cov, w) / np.dot(w, np.dot(cov, w))

# 特征组合
# min w' * cov * w
# s.t. A' * w = 1
# 解析解: w = cov^-1 * A * (A' * cov^-1 * A)^-1 * 1
def calcCharacteristicPortfolio(cov, a, *other_a):
    A = np.c_[(a,)+other_a]
    h = np.dot(np.linalg.inv(cov), A)
    return np.dot(np.dot(h, np.linalg.inv(np.dot(A.T, h))), np.ones((A.shape[1],)))

# 因子组合
# min w' * cov * w
# s.t. X' * w = b = [1, 0, ...]
# 解析解: w = cov^-1 * X * (X' * cov^-1 * X)^-1 * b
def calcFactorPortfolio(cov, target_factor, *other_factor):
    X = np.c_[(target_factor,)+other_factor]
    h = np.dot(np.linalg.inv(cov), X)
    l = np.zeros((X.shape[1],))
    l[0] = 1
    return np.dot(np.dot(h, np.linalg.inv(np.dot(X.T, h))), l)

# 最小方差组合
# min w' * cov * w
# # s.t. 1' * w = 1
def calcMinVarPortfolio(cov, allow_short=True, lb=None, ub=None):
    if allow_short and (lb is None) and (ub is None):
        return calcCharacteristicPortfolio(cov, np.ones((cov.shape[0], )))
    else:
        w = cvx.Variable(np.shape(cov)[0])
        Constraints = [cvx.sum(w)==1]
        if not allow_short:
            Constraints += [w>=0, w<=1]
        if lb is not None:
            Constraints.append(w>=lb)
        if ub is not None:
            Constraints.append(w<=ub)
        # 假设默认精度为 1e-6, 通过 adj_coef 调整目标函数值使其不要过小
        adj_coef = max((1, 1e-4 / np.min(np.diag(cov))))
        Prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, adj_coef * cov)), Constraints)
        Prob.solve(verbose=False)
        if Prob.status != cvx.OPTIMAL:
            raise Exception("Min variance problem solving fails: '%s'" % (str(Prob.status), ))
        return w.value
# 有效组合
# 如果给定目标收益率 r：
# min w' * cov * w
# s.t. mu' * w = r
#       1' * w = 1
#        0<=w<=1, if allow_short=False
# 如果给定目标波动率 sigma：
# max mu' * w
# s.t. w' * cov * w <= sigma**2
#       1' * w = 1
#        0<=w<=1, if allow_short=False
# 如果给定风险厌恶系数 risk_aversion
# max mu' * w - risk_aversion * w' * cov * w
# s.t. 1' * w = 1
#        0<=w<=1, if allow_short=False
def calcEfficientPortfolio(cov, mu, allow_short=True, lb=None, ub=None, r=None, sigma=None, risk_aversion=None, **kwargs):
    # 有解析解的情形
    if (r is not None) and allow_short and (lb is None) and (ub is None):
        return calcCharacteristicPortfolio(cov, np.ones(shape=np.shape(mu)), mu / r)
    if (r is None) and (sigma is None) and (risk_aversion is not None) and allow_short and (lb is None) and (ub is None):
        InvCov = np.linalg.inv(cov)
        l = np.ones(shape=np.shape(mu))
        return 1 / (2*risk_aversion) * np.dot(InvCov, mu) + (1 - np.dot(l, np.dot(InvCov, mu)) / (2*risk_aversion)) / np.dot(l, np.dot(InvCov, l)) * np.dot(InvCov, l)
    # 无解析解的情形
    verbose  = kwargs.pop("verbose", False)
    w = cvx.Variable(np.shape(mu)[0])
    Constraints = [cvx.sum(w)==1]
    if not allow_short:
        Constraints += [w>=0, w<=1]
    if lb is not None:
        Constraints.append(w>=lb)
    if ub is not None:
        Constraints.append(w<=ub)
    if r is not None:
        if r<np.min(mu):
            raise Exception("Target return is less than the minimum of all returns")
        elif r>np.max(mu):
            raise Exception("Target return is greater than the maximum of all returns")
        else:
            Constraints.append(mu @ w == r)
            Prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, cov)), Constraints)
    elif sigma is not None:
        Constraints.append(cvx.quad_form(w, cov) <= sigma**2)
        Prob = cvx.Problem(cvx.Maximize(mu @ w), Constraints)
        Prob.solve(verbose=verbose, **kwargs)
        if Prob.status == cvx.OPTIMAL:
            return w.value
        else:
            MinVarPortfolio = calcMinVarPortfolio(cov, allow_short=allow_short, lb=lb, ub=ub)
            if sigma < np.dot(MinVarPortfolio, np.dot(cov, MinVarPortfolio))**0.5:
                raise Exception("Target volatility is less than the minimum volatility!")
            else:
                raise Exception("Efficient portfolio problem solving fails: '%s'" % (str(Prob.status), ))
    elif risk_aversion is not None:
        Prob = cvx.Problem(cvx.Maximize(mu @ w - cvx.quad_form(w, risk_aversion * cov)), Constraints)
    else:
        raise Exception("One of the target return, target volatility and risk aversion must be given!")
    Prob.solve(verbose=verbose, **kwargs)
    if Prob.status == cvx.OPTIMAL:
        return w.value
    else:
        raise Exception("Efficient portfolio problem solving fails: '%s'" % (str(Prob.status), ))

# 最大夏普率组合
# min (mu' * w - rf) / (w' * cov * w) ** (1/2)
# s.t. 1' * w = 1
#        0<=w<=1, if allow_short=False
def calcMaxSharpeRatioPortfolio(cov, mu, rf, allow_short=True, lb=None, ub=None):
    if allow_short and (lb is None) and (ub is None):
        h = np.dot(np.linalg.inv(cov), mu - rf)
        l = np.dot(np.ones(np.shape(h)), h)
        if l<=0:
            raise Exception("There is no portfolio maximizing Sharpe ratio!")
        return h / l
    w = cvx.Variable(np.shape(mu)[0])
    TargetReturn = cvx.Parameter(1)
    # 假设默认精度为 1e-6, 通过 adj_coef 调整目标函数值使其不要过小
    adj_coef = max((1, 1e-4 / np.min(np.diag(cov))))
    Constraints = [mu @ w == TargetReturn, cvx.sum(w)==1]
    if not allow_short:
        Constraints += [w>=0, w<=1]
    if lb is not None:
        Constraints.append(w>=lb)
    if ub is not None:
        Constraints.append(w<=ub)
    Prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, adj_coef * cov)), Constraints)
    def _maxSharpeRatio(target_r):
        TargetReturn.value = np.array([target_r])
        Prob.solve(warm_start=True, verbose=False)
        # 返回夏普率的相反数
        if Prob.status == cvx.OPTIMAL:
            return - (target_r - rf) / (Prob.value / adj_coef) ** 0.5
        else:
            return 9999
    if (lb is None) and (ub is None):
        MaxReturn, MinReturn = np.max(mu), np.min(mu)
    else:
        MaxReturnProb = cvx.Problem(cvx.Maximize(mu @ w), Constraints[1:])
        MaxReturn = MaxReturnProb.solve() - 1e-5
        MinReturnProb = cvx.Problem(cvx.Minimize(mu @ w), Constraints[1:])
        MinReturn = MinReturnProb.solve() + 1e-5
    Res = minimize_scalar(_maxSharpeRatio, bounds=(MinReturn, MaxReturn), method="bounded")
    if not Res.success:
        raise Exception("Max Sharpe ratio problem solving fails!")
    TargetReturn.value = np.array([Res.x])
    Prob.solve(warm_start=True, verbose=False)
    if Prob.status == cvx.OPTIMAL:
        w = w.value
        w[w<=0] = 0
        return w / np.sum(w)
    else:
        raise Exception("Max Sharpe ratio problem solving fails: '%s'" % (str(Prob.status), ))

# 无约束的有效前沿
# min w' * cov * w
# s.t. mu' * w = r
# 解析解: r = (mu' * cov^-1 * mu)^(1/2) * sigma
def calcEfficientFrontierWithoutConstraint(cov, mu, r=None, sigma=None, num=100):
    k = np.dot(np.dot(mu, np.linalg.inv(cov)), mu) ** 0.5
    if r is None:
        if sigma is None:
            MaxSigma = np.max(np.diag(cov)**0.5)
            sigma = np.linspace(0, MaxSigma, num)
        r = k * sigma
    else:
        sigma = r / k
    return (r, sigma)

# 有效前沿
# min w' * cov * w
# s.t. mu' * w = r
#       1' * w = 1
#        0<=w<=1, if allow_short=False
def _calcEfficientFrontierNoShort(cov, mu, r=None, sigma=None, num=100):
    w = cvx.Variable(np.shape(mu)[0])
    if (sigma is None) or (r is not None):
        if r is None:
            w_MinVar = calcMinVarPortfolio(cov, allow_short=False)
            #MinVar = np.dot(w_MinVar, np.dot(cov, w_MinVar))
            #if MinVar<=np.min(np.diag(cov)):# 最小方差组合求解成功
                #MinMu = np.dot(mu, w_MinVar)
            #else:
                #MinMu = np.min(mu)
            MinMu = np.dot(mu, w_MinVar)
            r = np.linspace(MinMu, np.max(mu), num)
        TargetReturn = cvx.Parameter(1)
        # 假设默认精度为 1e-6, 通过 adj_coef 调整目标函数值使其不要过小
        adj_coef = max((1, 1e-4 / np.min(np.diag(cov))))
        Prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, adj_coef * cov)), [mu @ w == TargetReturn, cvx.sum(w)==1, w>=0, w<=1])
        sigma = np.full(shape=np.shape(r), fill_value=np.nan)
        for i, ir in enumerate(r):
            TargetReturn.value = np.array([ir])
            Prob.solve(warm_start=True, verbose=False)
            if Prob.status == cvx.OPTIMAL:
                sigma[i] = (Prob.value / adj_coef) ** 0.5
        #Idx = np.nanargmin(sigma)
        #sigma = sigma[Idx:]
        #r = r[Idx:]
    else:
        TargetVar = cvx.Parameter(1)
        Prob = cvx.Problem(cvx.Maximize(mu @ w), [cvx.quad_form(w, cov) <= TargetVar, cvx.sum(w)==1, w>=0, w<=1])
        r = np.full(shape=np.shape(sigma), fill_value=np.nan)
        for i, isigma in enumerate(sigma):
            TargetVar.value = np.array([isigma**2])
            Prob.solve(warm_start=True, verbose=False)
            if Prob.status == cvx.OPTIMAL:
                r[i] = Prob.value
    return (r, sigma)

def calcEfficientFrontier(cov, mu, rf=None, allow_short=True, r=None, sigma=None, num=100):
    if rf is None:# 不考虑无风险利率
        if allow_short:
            x = np.c_[np.ones(np.shape(mu)), mu]
            M = np.dot(np.dot(x.T, np.linalg.inv(cov)), x)
            if r is None:
                if sigma is None:
                    MaxSigma = np.max(np.diag(cov)**0.5)
                    MinSigma = 1 / M[0, 0] ** 0.5
                    sigma = np.linspace(MinSigma, MaxSigma, num)
                r = (M[0, 1] + ((M[0, 0]*sigma**2 - 1) * np.linalg.det(M)) ** 0.5) / M[0, 0]
            else:
                sigma = ((M[0, 0]*r**2 - 2*M[0, 1]*r + M[1, 1]) / np.linalg.det(M)) ** 0.5
            return (r, sigma)
        else:
            return _calcEfficientFrontierNoShort(cov, mu, sigma=sigma, r=r, num=num)
    else:
        # 求解最大夏普率组合
        w_MSR = calcMaxSharpeRatioPortfolio(cov=cov, mu=mu, rf=rf, allow_short=allow_short)
        r_MSR = np.dot(mu, w_MSR)
        sigma_MSR = np.dot(np.dot(w_MSR, cov), w_MSR) ** 0.5
        if r is None:
            if sigma is None:
                MaxSigma = np.max(np.diag(cov)**0.5)
                sigma = np.linspace(0, MaxSigma, num)
            r = rf + (r_MSR - rf) / sigma_MSR * sigma
        else:
            sigma = sigma_MSR / (r_MSR - rf) * (r - rf)
            sigma[sigma<0] = np.nan
        return (r, sigma)

# Black-Litterman 模型预期收益率和协方差阵
def calcBlackLittermanReturnCov(cov, pi, tau, pick_matrix, view, omega):
    p_sigma = np.dot(pick_matrix, cov)
    temp = tau * np.dot(p_sigma.T, np.linalg.inv(tau * np.dot(p_sigma, pick_matrix.T) + omega))
    mu_BL = pi + np.dot(temp, view - np.dot(pick_matrix, pi))
    sigma_BL = (1 + tau) * cov - tau * np.dot(temp, p_sigma)
    return (mu_BL, sigma_BL)

# 风险预算模型
def calcRiskBudgetPortfolio(cov, b=None):
    nVar = np.shape(cov)[0]
    if b is None:
        b = np.full(shape=(nVar,), fill_value=1 / nVar)
    w = cvx.Variable(nVar)
    c = np.dot(b, np.log(b)) - min(1e-4, 1 / nVar)
    Prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, cov)), [w >= 0, b @ cvx.log(w) >= c])
    Prob.solve(verbose=False)
    if Prob.status == cvx.OPTIMAL:
        return w.value / np.sum(w.value)
    else:
        raise Exception("Risk parity problem solving fails!")

# 最大分散化模型
def calcMaxDiversificationPortfolio(cov):
    nVar = np.shape(cov)[0]
    w = cvx.Variable(nVar)
    D = np.diag(1 / np.diag(cov)**0.5)
    P = np.dot(np.dot(D, cov), D)
    Prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, P)), [w >= 0, cvx.sum(w) == 1])
    Prob.solve(verbose=False)
    if Prob.status == cvx.OPTIMAL:
        w = np.dot(D, w.value)
        return w / np.sum(w)
    else:
        raise Exception("Max diversification problem solving fails!")

if __name__=="__main__":
    #np.random.seed(0)
    #V = np.cov(np.random.normal(0.05, 0.3, size=(5, 100)))
    #mu = np.random.normal(0.05, 0.3, 5)
    V = np.array([[5.83620527e-02, -5.03222806e-05, 4.53809891e-06, 2.74778102e-03],
                  [-5.03222806e-05, 2.16942544e-04, 3.80028792e-06, 3.01588333e-05],
                  [4.53809891e-06, 3.80028792e-06, 2.97896953e-06, 9.41050834e-06],
                  [2.74778102e-03, 3.01588333e-05, 9.41050834e-06, 2.47097475e-02]])
    mu = np.array([0.07327771, 0.04035562, 0.03249029, 0.05633103])
    
    p = calcEfficientPortfolio(V, mu, allow_short=False, r=None, sigma=None, risk_aversion=2)
    
    #beta = np.random.rand(5) * 2
    #print(calcCharacteristicPortfolio(V, mu, beta))
    #w = calcMaxSharpeRatioPortfolio(V, mu, 0.02, allow_short=False)
    #w = calcRiskBudgetPortfolio(V)
    #w = calcMaxDiversificationPortfolio(V)
    
    #r1, sigma1 = calcEfficientFrontierWithoutConstraint(V, mu)
    #r2, sigma2 = calcEfficientFrontier(V, mu, allow_short=True)
    #r3, sigma3 = calcEfficientFrontier(V, mu, allow_short=False)
    #r4, sigma4 = calcEfficientFrontier(V, mu, rf=0.02, allow_short=True)
    #r5, sigma5 = calcEfficientFrontier(V, mu, rf=0.02, allow_short=False)
    import matplotlib.pyplot as plt
    Fig, Axes = plt.subplots(1, 1)
    #Axes.plot(sigma1, r1, label="Efficient Frontier without Constraint")
    #Axes.plot(sigma2, r2, label="Efficient Frontier Fully Invested")
    #Axes.plot(sigma3, r3, label="Efficient Frontier Fully Invested without Short")
    #Axes.plot(sigma4, r4, label="Efficient Frontier Fully Invested with Risk Free Rate")
    Axes.plot(sigma5, r5, label="Efficient Frontier Fully Invested without Short with Risk Free Rate")
    Axes.legend(loc="best")
    plt.show()
    print("===")