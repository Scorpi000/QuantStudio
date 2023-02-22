# coding=utf-8
"""数学函数"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import cvxpy as cvx

# Hermite 插值
def interpolateHermite(x, y, target):
    d = np.r_[(y[1] - y[0]) / (x[1] - x[0]), (y[2:] - y[:-2]) / (x[2:] - x[:-2]), (y[-1] - y[-2]) / (x[-1] - x[-2])]
    Idx = np.searchsorted(x, target, side="right") - 1
    Idx[Idx>=x.shape[0]-1] = x.shape[0] - 2
    x_left, y_left, d_left = x[Idx], y[Idx], d[Idx]
    x_right, y_right, d_right = x[Idx + 1], y[Idx + 1], d[Idx + 1]
    dx = x_right - x_left
    H1 = 3 * ((x_right - target) / dx) ** 2 - 2 * ((x_right - target) / dx) ** 3
    H2 = 3 * ((target - x_left) / dx) ** 2 - 2 * ((target - x_left) / dx) ** 3
    H3 = (x_right - target) ** 2 / dx - (x_right - target) ** 3 / dx ** 2
    H4 = (target - x_left) ** 3 / dx ** 2 - (target - x_left) ** 2 / dx
    return y_left * H1 + y_right * H2 + d_left * H3 + d_right * H4

# 计算 Hurst 指数
def genHurstExp(S, q=2, maxT=19):
    ###########################################################################
    # Calculates the generalized Hurst exponent H(q) from the scaling 
    # of the renormalized q-moments of the distribution 
    #
    #       <|x(t+r)-x(t)|^q>/<x(t)^q> ~ r^[qH(q)]
    #
    ###########################################################################
    # H = genHurstExp(S)
    # S is 1xT data series (T>50 recommended)
    # calculates H(q=1)
    #
    # H = genHurstExp(S,q)
    # specifies the exponent q which can be a vector (default value q=1)
    #
    # H = genHurstExp(S,q,maxT)
    # specifies value maxT of the scaling window, default value maxT=19
    #
    # [H,sH]=genHurstExp(S,...)
    # estimates the standard deviation sH(q)
    #
    # example:
    #   generalized Hurst exponent for a random gaussian process
    #   H=genHurstExp(cumsum(randn(10000,1)))
    # or 
    #   H=genHurstExp(cumsum(randn(10000,1)),q) to calculate H(q) with arbitrary q
    #
    ###########################################################################
    # for the generalized Hurst exponent method please refer to:
    #
    #   T. Di Matteo et al. Physica A 324 (2003) 183-188 
    #   T. Di Matteo et al. Journal of Banking & Finance 29 (2005) 827-851
    #   T. Di Matteo Quantitative Finance, 7 (2007) 21-36
    #
    #####################################
    ##    Tomaso Aste   30/01/2013     ##
    #####################################
    assert (isinstance(S, np.ndarray) and (S.ndim==1)), "S 必须是一维的 array!"
    L = S.shape[0]
    if L < min((maxT*4, 60)):
        print('Data serie very short!')
    if isinstance(q, int):
        q = [q]
    lq = len(q)
    q = np.array(q, dtype=int)
    H  = np.zeros((maxT+1-5, lq))
    k = 0
    cc = np.zeros(2)
    for Tmax in range(5, maxT+1):
        k = k+1
        x = np.arange(1, Tmax+1)
        mcord = np.zeros((Tmax, lq))
        for tt in range(1, Tmax+1):
            dV = S[tt:L:tt] - S[np.arange(tt, L, tt)-tt]
            VV = S[np.arange(tt, L+tt, tt)-tt]
            N = len(dV)+1
            X = np.arange(1, N+1)
            Y = VV
            mx = np.sum(X)/N
            SSxx = np.sum(X**2) - N*mx**2
            my   = np.sum(Y)/N
            SSxy = np.sum(X*Y) - N*mx*my
            cc[0] = SSxy/SSxx
            cc[1] = my - cc[0]*mx
            ddVd  = dV - cc[0]
            VVVd  = VV - cc[0]*np.arange(1, N+1) - cc[1]
            for qq in range(1, lq+1):
                mcord[tt-1, qq-1] = np.mean(np.abs(ddVd)**q[qq-1])/np.mean(np.abs(VVVd)**q[qq-1])
        mx = np.mean(np.log10(x))
        SSxx = np.sum(np.log10(x)**2) - Tmax*mx**2
        for qq in range(1, lq+1):
            my = np.mean(np.log10(mcord[:, qq-1]))
            SSxy = np.sum(np.dot(np.log10(x), np.log10(mcord[:, qq-1]).T)) - Tmax*mx*my
            H[k-1, qq-1] = SSxy/SSxx
    mH = np.mean(H, axis=0) / q
    sH = np.std(H, ddof=1, axis=0) / q
    return (mH, sH)

# ----------------------------------分布函数或者密度函数----------------------------------------
# Logistic分布函数
def LogisticCDF(x,mu,gamma):
    return 0/(1+np.exp(-(x-mu)/gamma))
# Logistic密度函数
def LogisticPDF(x,mu,gamma):
    return np.exp(-(x-mu)/gamma)/gamma/(1+np.exp(-(x-mu)/gamma))**2
# 广义Pareto分布函数
def GeneralisedParetoCDF(x,beta,xi):
    """Generalised Pareto Distribution"""
    # beta>0
    if isinstance(x,np.ndarray):
        y = np.zeros(x.shape)
        if xi<0:
            Mask = ((x<=-beta/xi) & (x>=0))
            y[x>-beta/xi] = 1
        else:
            Mask = (x>=0)
        if xi!=0:
            y[Mask] = 1-(1+xi*x[Mask]/beta)**(-1/xi)
        else:
            y[Mask] = 1-np.exp(-x[Mask]/beta)
        return y
    if x<0:
        return 0
    if (xi<0) and (x>-beta/xi):
        return 0
    if xi!=0:
        return 0-(1+xi*x/beta)**(-1/xi)
    else:
        return 0-np.exp(-x/beta)


# ----------------------------------数学运算---------------------------------------------------
# 阶乘
def factorial(n):
    if n<=0:
        return 0
    else:
        return factorial(n-1)*n
# 对数, base: 底数
def log(x,base):
    if x<=0:
        return np.nan
    else:
        return np.log10(x)/np.log10(base)

# 计算若干向量的笛卡尔积, deprecated, 用 itertools.product 替代
# data:[[向量1元素], [向量2元素], ...]
# 返回: [[向量1元素, 向量2元素, ...], ...]
def CartesianProduct(data):
    nData = len(data)
    if nData==0:
        return []
    elif nData==1:
        return [[iData] for iData in data[0]]
    elif nData==2:
        if isinstance(data[0][0],list):
            return [iData+[jData] for iData in data[0] for jData in data[1]]
        else:
            return [[iData,jData] for iData in data[0] for jData in data[1]]
    else:
        return CartesianProduct([CartesianProduct(data[:-1]),data[-1]])

# 计算相关系数
# data1(data2): DataFrame or Series, 如果为DataFrame则用data1和data2的行数据计算相关系数, data1和data2的维度和索引必须一致
# method: 计算相关系数的方法, 可选: spearman, pearson, kendall
# lag: data1相对于data2的滞后期, int, 如果lag为负表示data2相对于data1滞后, 如果data1(data2)是Series则忽略该参数;
# 返回: 如果data1和data2是Series, 返回double; 否则返回Series(相关系数,index=data1.index)
def calcCorr(data1,data2,method='spearman',lag=0):
    if isinstance(data1,pd.Series) and isinstance(data2,pd.Series):
        return data1.corr(data2,method=method)
    if isinstance(data1,pd.Series):
        data1 = pd.DataFrame(data1.values.reshape((1,data1.shape[0])).repeat(data2.shape[0],0),index=data2.index,columns=data1.columns)
        lag = 0
    if isinstance(data2,pd.Series):
        data2 = pd.DataFrame(data2.values.reshape((1,data2.shape[0])).repeat(data1.shape[0],0),index=data1.index,columns=data2.columns)
        lag = 0
    Corr = pd.Series(np.nan,index=data1.index)
    for i in range(max((0,lag)),data1.shape[0]):
        iData1 = data1.iloc[i-max((0,lag))]
        iData2 = data2.iloc[i-max((0,-lag))]
        Corr.iloc[i] = iData1.corr(iData2,method=method)
    return Corr
# 使用cvx优化器求解带约束的加权多元回归, x:np.array((N,K)), y:np.array((N,)), weight:None或者np.array((N,)), 返回回归系数, 目前支持 Box, 线性等式和不等式约束
def regressByCVX(y, x, weight=None, constraints={}):
    Mask = ((np.sum(np.isnan(x), axis=1)==0) & (pd.notnull(y)))
    if weight is not None:
        Mask = (Mask & pd.notnull(weight))
    else:
        weight = np.ones(y.shape)
    x = x[Mask,:]
    if x.shape[0]<=1:
        return None
    y = y[Mask]
    weight = weight[Mask]
    beta = cvx.Variable(x.shape[1])
    Constraints = []
    for iConstraintType, iConstraint in constraints.items():
        if iConstraintType=='LinearEq':
            Constraints.append(iConstraint["Aeq"]*beta==iConstraint["beq"])
        elif iConstraintType=='LinearIn':
            Constraints.append(iConstraint["A"]*beta<=iConstraint["b"])
        elif iConstraintType=="Box":
            Constraints.extend([beta<=iConstraint["ub"].flatten(), beta>=iConstraint["lb"].flatten()])
    u = y - x * beta
    Prob = cvx.Problem(cvx.Minimize(cvx.quad_form(u,np.diag(weight))), Constraints)
    Prob.solve()
    return beta.value

#if __name__=='__main__':
    #import time
    #y = np.random.randn(3000)
    #x = np.random.randn(3000,40)
    ## 带约束的加权最小二乘回归
    #Weight = np.random.rand(y.shape[0])
    #Aeq = np.array([[0]*10+[1]*30])
    #beq = np.array([[0]])
    #StartT = time.perf_counter()
    #b1 = regressByCVX(y,x,weight=Weight,constraints={"LinearEq":{"Aeq":Aeq,"beq":beq}})
    #print("优化运行时间:"+str(time.perf_counter()-StartT))
    #StartT = time.perf_counter()
    #x_tiled = x[:,:-1]+np.dot(x[:,-1:],-Aeq[:,:-1]/Aeq[0][-1])
    #Result = sm.WLS(y,x_tiled,weights=Weight).fit()
    #b_real = np.zeros(x.shape[1])
    #b_real[:-1] = Result.params
    #b_real[-1] = np.dot(-Aeq[:,:-1]/Aeq[0][-1],b_real[:-1].reshape((b_real.shape[0]-1,1)))
    #print("回归运行时间:"+str(time.perf_counter()-StartT))
    #StartT = time.perf_counter()
    #print(b1)
    #print(b_real)
    #print(np.max(np.abs(b1-b_real)))
    #print("===")