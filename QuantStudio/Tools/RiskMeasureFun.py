# -*- coding: utf-8 -*-
"""风险测度方法"""
import numpy as np
from scipy.stats import kurtosis,norm,skew
from scipy.optimize import minimize
from scipy.integrate import quad

from .MathFun import GeneralisedParetoCDF

# ------------------------------------EVT(Extreme Value Theory)-----------------------------------
def estimate_u(x):
    x = x[~np.isnan(x)]
    Avg = x.mean()
    K = kurtosis(x)
    while K>=3:
        Pos = np.abs(x-Avg).argmax()
        x = np.delete(x,Pos,0)
        Avg = x.mean()
        K = kurtosis(x)
    return np.abs(x).max()

def MLFun(beta,xi,y):
    """beta和xi估计的最大似然函数(取了负值,最小化)"""
    # y = x-u,x>=u
    Nu = y.shape[0]
    if xi<0:
        Temp = np.zeros_like(y)
        Temp[y<-beta/xi] = np.log(1+xi*y[y<-beta/xi]/beta)
        return Nu*np.log(beta)+(1+xi)/xi*Temp.sum()
    elif xi>0:
        return Nu*np.log(beta)+(1+xi)/xi*np.log(1+xi*y/beta).sum()
    else:
        return Nu*np.log(beta)+1/beta*y.sum()

def MLDerFun(beta,xi,y):
    """beta和xi估计的最大似然函数(取了负值)的导数"""
    Der = np.zeros(2)
    Nu = y.shape[0]
    Der[0] = Nu/beta+(1+xi)/beta*(y/(beta+xi*y)).sum()
    if xi!=0:
        Der[1] = -1/xi**2*np.log(1+y/beta*xi).sum()+(1+1/xi)*(y/(beta+y*xi)).sum()
    else:
        Der[1] = y.sum()/beta-(y**2).sum()/2/beta**2
    return Der
def estimateArg(x,u):
    """估计beta和xi"""
    y = x-u
    y = y[y>0]
    Objective = lambda z:MLFun(z[0],z[1],y)
    DerFun = lambda z:MLDerFun(z[0],z[1],y)
    z0 = [1,0]
    z = minimize(Objective, z0, method='BFGS', jac=DerFun,options={'disp': True})
    return (z.x[0],z.x[1])

def estimateVaR_EVT(x,beta,xi,u,alpha):
    N = x.shape[0]
    Nu = (x>u).sum()
    if xi!=0:
        return u+beta/xi*((alpha*N/Nu)**(-xi)-1)
    else:
        return u-beta*np.log(alpha*N/Nu)

def estimateES(x,beta,xi,u,alpha,var=None):
    if var is None:
        var = estimateVaR_EVT(x,beta,xi,u,alpha)
    return (var+beta-xi*u)/(1-xi)

# ------------------------------------VaR, CVaR-----------------------------------
def estimateVaR(x,alpha,method):
    N = x.shape[0]
    Avg = x.mean()
    Std = x.std()
    if method=='历史模拟':
        VaR = np.percentile(x,alpha)
        CVaR = x[x<VaR].mean()
    elif method=='正态分布':
        VaR = norm.ppf(alpha,loc=Avg,scale=Std)
        CVaR = Avg-Std/alpha*norm.pdf((VaR-Avg)/Std)
    elif method=='Cornish-Fisher':
        x = (x-Avg)/Std
        S = skew(x)
        K = kurtosis(x)-3
        q = norm.ppf(alpha)
        VaR = Avg+Std*(q+1/6*(q**2-1)*S+1/24*(q**3-3*q)*K-1/36*(2*q**3-5*q)*S**2)
        m1 = quad(lambda x:x*1/(2*np.pi)**0.5*np.exp(-x**2/2),-np.inf,q)[0]/alpha
        m2 = quad(lambda x:x**2*1/(2*np.pi)**0.5*np.exp(-x**2/2),-np.inf,q)[0]/alpha
        m3 = quad(lambda x:x**3*1/(2*np.pi)**0.5*np.exp(-x**2/2),-np.inf,q)[0]/alpha
        CVaR = Avg+Std*(m1+1/6*(m2-1)*S+1/24*(m3-3*m1)*K-1/36*(2*m3-5*m1)*S**2)
    return (VaR,CVaR)