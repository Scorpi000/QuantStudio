# -*- coding: utf-8 -*-
"""风险模型相关方法"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from QuantStudio.Tools.AuxiliaryFun import getExpWeight
from QuantStudio.Tools.DataTypeConversionFun import DummyVarTo01Var

# 使用 EWMA 方法估计样本协方差矩阵
# ret: 收益率, array, 行是日期, 列是 ID; forcast_num: 向前预测的期数; half_life: 时间指数权重半衰期
def estimateSampleCovMatrix_EWMA(ret, forcast_num=1, half_life=np.inf):
    Weight = np.flipud(np.array(getExpWeight(ret.shape[0],half_life)))
    Weight = np.repeat(np.reshape(Weight,(ret.shape[0],1)),ret.shape[1],axis=1)
    Mask = (~np.isnan(ret))
    Weight = Weight*Mask
    WeightSqrt = Weight**0.5
    ret = np.copy(ret)
    ret[~Mask] = 0.0
    ret = ret*WeightSqrt
    TotalWeight = np.dot(WeightSqrt.T, WeightSqrt)
    TotalWeight[TotalWeight==0] = np.nan
    BiasAdjust = (TotalWeight**2-np.dot(Weight.T,Weight))
    BiasAdjust[BiasAdjust==0] = np.nan
    BiasAdjust = TotalWeight**2/BiasAdjust
    AvgRet1 = np.dot(ret.T,WeightSqrt)/TotalWeight
    AvgRet2 = np.dot(WeightSqrt.T,ret)/TotalWeight
    CovMatrix = (np.dot(ret.T,ret)/TotalWeight-AvgRet1*AvgRet2)*BiasAdjust
    Vol = np.diag(CovMatrix)**0.5
    Temp = ((np.dot((ret**2).T,Mask)/TotalWeight-AvgRet1**2)*(np.dot(Mask.T,ret**2)/TotalWeight-AvgRet2**2))**0.5*BiasAdjust
    Temp[Temp==0] = np.nan
    CovMatrix = (CovMatrix/Temp*Vol).T*Vol
    return (CovMatrix.T+CovMatrix)/2*forcast_num
# 将协方差阵分解为: 波动率*相关系数矩阵*波动率, cov_matrix: 协方差矩阵, array
def decomposeCov2Corr(cov_matrix):
    Vol = 1/np.diag(cov_matrix)**0.5
    Corr = (1/Vol)*(cov_matrix/Vol).T
    Corr = (Corr+Corr.T)/2
    Corr[Corr>1.0] = 1.0
    Corr[Corr<-1.0] = -1.0
    Corr = Corr - np.diag(np.diag(Corr)) + np.eye(Corr.shape[0])
    return (Corr,Vol)
# 给定协方差阵, 计算平均相关系数, cov_matrix: 协方差矩阵, array
def calcAvgCorr(cov_matrix):
    CorrMatrix,_ = decomposeCov2Corr(cov_matrix)
    CorrMatrix = dropRiskMatrixNA(pd.DataFrame(CorrMatrix)).values
    return (np.nansum(CorrMatrix)-np.nansum(np.diag(CorrMatrix)))/CorrMatrix.shape[0]/(CorrMatrix.shape[0]-1)
# 带一个线性约束的加权多元线性回归, x: array((N,K)), y: array((N,)), weight: None 或者 array((N,)), 返回回归系数
def regressWithOneLinearEqConstraint(y, x, weight=None, Aeq=None, beq=None, statistics=False):
    Mask = ((np.sum(np.isnan(x), axis=1)==0) & (pd.notnull(y)))
    if weight is not None:
        Mask = (Mask & pd.notnull(weight))
    else:
        weight = np.ones(y.shape)
    x = x[Mask,:]
    if x.shape[0]<=1: return None
    y = y[Mask]
    weight = weight[Mask]
    if (Aeq is not None) and (beq is not None):
        NonZeroInd = np.arange(0,Aeq.shape[0])[Aeq!=0]
        if NonZeroInd.shape[0]==0: return None
        NonZeroInd = NonZeroInd[0]
        yy = y-x[:,NonZeroInd]*beq/Aeq[NonZeroInd]
        if NonZeroInd==0:
            xx = -np.dot(x[:,NonZeroInd:NonZeroInd+1],Aeq[NonZeroInd+1:].reshape((1,Aeq.shape[0]-1-NonZeroInd))/Aeq[NonZeroInd])+x[:,1+NonZeroInd:]
        elif NonZeroInd==x.shape[1]-1:
            xx = x[:,:NonZeroInd]-np.dot(x[:,NonZeroInd:NonZeroInd+1],Aeq[:NonZeroInd].reshape((1,NonZeroInd))/Aeq[NonZeroInd])
        else:
            xx = np.hstack((x[:,:NonZeroInd]-np.dot(x[:,NonZeroInd:NonZeroInd+1],Aeq[:NonZeroInd].reshape((1,NonZeroInd))/Aeq[NonZeroInd]),-np.dot(x[:,NonZeroInd:NonZeroInd+1],Aeq[NonZeroInd+1:].reshape((1,Aeq.shape[0]-1-NonZeroInd))/Aeq[NonZeroInd])+x[:,1+NonZeroInd:]))
        Result = sm.WLS(yy,xx,weights=weight).fit()
        beta = np.zeros(x.shape[1])
        beta[NonZeroInd] = (beq-np.sum(Result.params*np.append(Aeq[:NonZeroInd],Aeq[NonZeroInd+1:])))/Aeq[NonZeroInd]
        beta[:NonZeroInd] = Result.params[:NonZeroInd]
        beta[NonZeroInd+1:] = Result.params[NonZeroInd:]
    else:
        Result = sm.WLS(y,x,weights=weight).fit()
        beta = Result.params
    if not statistics: return beta
    Statistics = {"R2":1-np.sum(weight*(y-np.dot(x,beta))**2)/np.sum(weight*y**2)}
    Statistics["R2_adj"] = 1-(1-Statistics["R2"])*(y.shape[0]-1)/(y.shape[0]-beta.shape[0]-1+((Aeq is not None) and (beq is not None)))
    return (beta, Statistics)

# 计算回归权重
def calcRegressWeight(cap_data, percentile=0.95):
    Weight = cap_data**0.5
    Quantile = Weight.quantile(percentile)
    Weight[Weight>Quantile] = Quantile
    Weight = Weight / Weight.sum()
    return Weight

# 计算市场收益率, ret_data: Series(收益率,index=[ID]), weight: Series(权重,index=[ID]), mask: Series(True or False,index=[ID])
def calcMarketReturn(ret, weight=None):
    Mask = pd.notnull(ret)
    if weight is not None:
        Mask = (Mask & pd.notnull(weight))
    else:
        weight = pd.Series(1.0,index=ret.index)
    ret = ret[Mask]
    weight = weight[Mask]
    return (ret*weight).sum()/weight.sum()

# 计算残差收益的极端部分, resid_ret: 残差收益率, array
def calcRetOutlier(resid_ret):
    RobustStd = 1.4826*np.nanmedian(np.abs(resid_ret-np.nanmedian(resid_ret)))
    Outlier = np.zeros(resid_ret.shape)
    Temp = np.abs(resid_ret)-4*RobustStd
    Mask = (Temp>0)
    Outlier[Mask] = np.sign(resid_ret[Mask])*Temp[Mask]
    return Outlier

# 补充Proxy Asset
def addProxySample(ret, factor_data, industry_data, weight, market_ret):
    ProxyID = []
    ProxyWeight = []
    ProxyRet = []
    ProxyIndustry = []
    AllIndustries = np.unique(industry_data)
    for i,iIndustry in enumerate(AllIndustries):
        iMask = (industry_data==iIndustry)
        iWeight = weight[iMask]
        iTotalWeight = iWeight.sum()
        if iTotalWeight==0:
            continue
        nSample = (iTotalWeight)**2/(iWeight**2).sum()
        if nSample<5:
            ProxyID.append('Proxy-'+iIndustry)
            ProxyWeight.append((5-nSample)*iTotalWeight/nSample)
            ProxyRet.append(market_ret)
            ProxyIndustry.append(iIndustry)
    nProxy = len(ProxyID)
    if nProxy>0:
        factor_data = factor_data.append(pd.DataFrame(np.zeros((nProxy,factor_data.shape[1])),index=ProxyID,columns=factor_data.columns))
        industry_data = industry_data.append(pd.Series(ProxyIndustry,index=ProxyID))
        ret = ret.append(pd.Series(ProxyRet,index=ProxyID))
        weight = weight.append(pd.Series(ProxyWeight,index=ProxyID))
    return (ret,factor_data,industry_data,weight)

# 使用 EWMA 方法和 Newey-West 方法估计协方差, jret, kret: np.array(收益率), weight: np.array(权重), delta: kret相对于jret的滞后期
# 假设 jret、kret 以及 weight 均没有nan
def calcCovariance(jret, kret, weight, delta=0):
    nLen = jret.shape[0]-np.abs(delta)
    weight = weight[:nLen]
    weight = weight/np.nansum(weight)
    if delta>=0:
        jret = jret[:nLen]
        kret = kret[kret.shape[0]-nLen:]
    else:
        kret = kret[:nLen]
        jret = jret[jret.shape[0]-nLen]
    return np.nansum(weight*(jret-np.nansum(jret*weight))*(kret-np.nanmean(kret*weight)))

# 使用 EWMA 方法和 Newey-West 方法估计协方差矩阵, ret: 收益率, DataFrame(收益率, index=[日期], columns=[ID])
# forcast_num: 向前预测的期数; auto_corr_num: 考虑有自相关性的最大期数; half_life: 时间指数权重半衰期; calc_cov: 是否计算协方差, False的话只返回方差(Series)
def estimateCovMatrix(ret, forcast_num=21, auto_corr_num=10, half_life=480, calc_cov=True):
    if forcast_num>auto_corr_num+1:
        N = auto_corr_num
    else:
        N = forcast_num - 1
    Weight = np.flipud(np.array(getExpWeight(ret.shape[0],half_life)))
    if calc_cov:
        CovMatrix = pd.DataFrame(np.nan, index=ret.columns, columns=ret.columns)
        for j,jCol in enumerate(ret.columns):
            for kCol in ret.columns[j:]:
                jRet = ret[jCol].values
                kRet = ret[kCol].values
                Covs = np.zeros(N*2+1)
                Coefs = np.zeros(N*2+1)
                for Delta in range(-N,N+1):
                    Coefs[Delta+N] = N+1-np.abs(Delta)
                    Covs[Delta+N] = calcCovariance(jRet, kRet, Weight, Delta)
                CovMatrix[jCol][kCol] = np.nansum(Coefs*Covs)
                CovMatrix[kCol][jCol] = CovMatrix[jCol][kCol]
    else:
        Columns = ret.columns
        CovMatrix = np.zeros(ret.shape[1])+np.nan
        ret = ret.values
        for j in range(ret.shape[1]):
            jRet = ret[:, j]
            Covs = np.zeros(N*2+1)
            Coefs = np.zeros(N*2+1)
            for Delta in range(-N, N+1):
                Coefs[Delta+N] = N+1-np.abs(Delta)
                Covs[Delta+N] = calcCovariance(jRet, jRet, Weight, Delta)
            CovMatrix[j] = np.nansum(Coefs*Covs)
        CovMatrix = pd.Series(CovMatrix, index=Columns)
    CovMatrix = CovMatrix * forcast_num / (N+1)
    return CovMatrix
# 使对称矩阵正定, 对于非正特征值以小正数替换
def makeMatrixPositiveDefinite(target_matrix, epsilon=1e-6):
    D,Q = np.linalg.eig(target_matrix)
    D[D<=0] = epsilon
    return np.dot(np.dot(Q,np.diag(D)),Q.T)
# 估计因子收益率和特异性收益率, 使用Barra EUE3的方法, 参见EUE3
# ret: Series(股票收益率,index=[ID]); factor_data: DataFrame(因子暴露,index=[ID],columns=[因子名]);
# industry_data: Series(行业名称,index=[ID]); weight: Series(回归权重,index=[ID]);
# estu: Series(是否属于Estimation Universe,0 or 1,index=[ID]); cap: Series(市值,index=[ID])
# all_industries: [所有的行业名]
def estimateFactorAndSpecificReturn_EUE3(ret, factor_data, industry_data, weight, estu, cap, all_industries):
    # 准备用于回归的数据
    ESTUMask = ((estu==1) & pd.notnull(weight))
    ESTUFactorData = factor_data.loc[ESTUMask,:]
    ESTURet = ret.loc[ESTUMask]
    ESTUWeight = weight.loc[ESTUMask]
    ESTUWeight = calcRegressWeight(ESTUWeight,percentile=0.95)
    ESTUIndustry = industry_data.loc[ESTUMask]
    ESTUMarketRet = calcMarketReturn(ESTURet,ESTUWeight)
    # 添加Proxy Asset
    ESTURet,ESTUFactorData,ESTUIndustry,ESTUWeight = addProxySample(ESTURet,ESTUFactorData,ESTUIndustry,ESTUWeight,ESTUMarketRet)
    # 展开行业因子成0-1变量
    ESTUIndustryDummy = DummyVarTo01Var(ESTUIndustry,ignore_na=True)
    # 计算回归的限制条件
    ESTUIndustryCap = [(ESTUIndustryDummy.iloc[:,i]*ESTUWeight).sum() for i in range(ESTUIndustryDummy.shape[1])]
    Aeq = np.array([0]*(1+ESTUFactorData.shape[1])+ESTUIndustryCap)
    beq = 0.0
    Y = ESTURet.values
    X = np.hstack((np.ones((ESTURet.shape[0],1)),ESTUFactorData.values,ESTUIndustryDummy.values))
    # 第一次回归
    FactorReturn = regressWithOneLinearEqConstraint(Y,X,ESTUWeight.values,Aeq,beq)
    SpecificReturn = Y-np.dot(X,FactorReturn)
    # 计算残差收益的异常部分
    ResidOutlier = pd.Series(calcRetOutlier(SpecificReturn),index=ESTURet.index)
    # 第二次回归
    Y = Y-ResidOutlier.values
    FactorReturn,Statistics = regressWithOneLinearEqConstraint(Y,X,ESTUWeight.values,Aeq,beq,True)
    _,iStatistics = regressWithOneLinearEqConstraint(Y,X[:,0:1],ESTUWeight.values,None,None,True)
    Statistics["R2_市场因子"] = iStatistics["R2"]
    Statistics["R2_adj_市场因子"] = iStatistics["R2_adj"]
    _,iStatistics = regressWithOneLinearEqConstraint(Y,X[:,0:ESTUFactorData.shape[1]+1],ESTUWeight.values,None,None,True)
    Statistics["R2_风险因子"] = iStatistics["R2"]
    Statistics["R2_adj_风险因子"] = iStatistics["R2_adj"]
    _,iStatistics = regressWithOneLinearEqConstraint(Y,X[:,ESTUFactorData.shape[1]+1:],ESTUWeight.values,None,None,True)
    Statistics["R2_行业因子"] = iStatistics["R2"]
    Statistics["R2_adj_行业因子"] = iStatistics["R2_adj"]
    # 生成因子收益率
    FactorReturn = pd.Series(FactorReturn,["Market"]+list(factor_data.columns)+list(ESTUIndustryDummy.columns))
    FactorReturn = FactorReturn[["Market"]+list(factor_data.columns)+all_industries]
    FactorReturn[pd.isnull(FactorReturn)] = ESTUMarketRet
    # 生成特异性收益率
    ESTUIndustryDummy = DummyVarTo01Var(industry_data,ignore_na=True)
    ESTUIndustryDummy = ESTUIndustryDummy.loc[:,all_industries]
    ESTUIndustryDummy = ESTUIndustryDummy.where(pd.notnull(ESTUIndustryDummy),0.0)
    X = np.hstack((np.ones((ret.shape[0],1)),factor_data.values,ESTUIndustryDummy.values))
    SpecificReturn = pd.Series(ret.values-np.dot(X,FactorReturn.values),index=ret.index)
    return (FactorReturn,SpecificReturn,pd.DataFrame(X,index=factor_data.index,columns=["Market"]+list(factor_data.columns)+all_industries),Statistics)
# 估计因子协方差矩阵, 使用Barra CHE2的方法, 参见CHE2附录A
# factor_ret: DataFrame(因子收益率,index=[日期],columns=[ID]); forcast_num: 向前预测的期数;
# auto_corr_num: 考虑有自相关性的最大期数; half_life_corr: 估计相关系数的时间指数权重半衰期;
# half_life_vol: 估计波动率的时间指数权重半衰期;
def estimateFactorCov_CHE2(factor_ret,forcast_num=21,auto_corr_num=10,half_life_corr=480,half_life_vol=90):
    FactorCov = estimateCovMatrix(factor_ret,forcast_num=forcast_num,auto_corr_num=auto_corr_num,half_life=half_life_corr)
    VolatilityReciprocal = np.diag(1/np.diag(FactorCov)**0.5)
    CorrMatrix = np.dot(np.dot(VolatilityReciprocal,FactorCov),VolatilityReciprocal)
    VolatilityDiag = estimateCovMatrix(factor_ret,forcast_num=forcast_num,half_life=half_life_vol,calc_cov=False)
    VolatilityDiag = np.diag(VolatilityDiag**0.5)
    FactorCov = np.dot(np.dot(VolatilityDiag,CorrMatrix),VolatilityDiag)
    return pd.DataFrame(makeMatrixPositiveDefinite(FactorCov),index=factor_ret.columns,columns=factor_ret.columns)
# 计算blending coefficient
# specific_ret: DataFrame(收益率,index=[日期],columns=[ID]);
def calcBlendingCoefficient(specific_ret):
    Gamma = {}
    for iID in specific_ret.columns:
        iSpecificRet = specific_ret[iID]
        iSpecificRet = iSpecificRet[pd.notnull(iSpecificRet)].values
        ih = iSpecificRet.shape[0]
        if ih==0:
            Gamma[iID]=0
            continue
        iRobustStd = 1/1.35*(np.percentile(iSpecificRet,75)-np.percentile(iSpecificRet,25))
        iSpecificRet[iSpecificRet>10*iRobustStd] = 10*iRobustStd
        iSpecificRet[iSpecificRet<-10*iRobustStd] = -10*iRobustStd
        iStd = np.std(iSpecificRet)
        iZVal = np.abs((iStd-iRobustStd)/iRobustStd)
        Gamma[iID] = min((1,max((0,(ih-60)/120))))*min((1,max((0,np.exp(1-iZVal)))))
    Gamma = pd.Series(Gamma,name='Gamma')
    Gamma[pd.isnull(Gamma)] = 0
    return Gamma
    
# 计算Structural forcast of specific risk
def calcSTRSpecificRisk(gamma, std_ts, factor_data, cap):
    # 准备回归数据
    IDs = gamma[gamma==1].index.tolist()# 选择gamma值为1的ID
    Y = std_ts.loc[IDs]
    Y[Y==0] = np.nan
    FactorData = factor_data.loc[IDs, :]
    FactorData = FactorData.loc[:, FactorData.abs().sum()!=0]
    RegWeight = calcRegressWeight(cap).loc[IDs]
    # 回归
    Coef = regressWithOneLinearEqConstraint(np.log(Y.values), FactorData.values, RegWeight.values)
    # 估计Scale Multiplier
    Temp = Y.values / np.exp(np.dot(FactorData.values, Coef))
    Mask = (pd.notnull(Temp) & pd.notnull(RegWeight.values))
    E0 = np.nansum(Temp[Mask] * RegWeight.values[Mask]) / np.nansum(RegWeight.values[Mask])
    # 计算Structural forcast of specific risk
    return pd.Series(np.exp(np.dot(factor_data.loc[:, FactorData.columns].values, Coef)) * E0, index=std_ts.index)

# 估计特异性风险, 使用Barra EUE3的方法, 参见EUE3
# specific_ret: DataFrame(收益率,index=[日期],columns=[ID]); forcast_num: 向前预测的期数;
# auto_corr_num: 考虑有自相关性的最大期数; half_life: 时间指数权重半衰期;
def estimateSpecificRisk_EUE3(specific_ret, factor_data, cap, forcast_num=21, auto_corr_num=10, half_life=480):
    Std_TS = estimateCovMatrix(specific_ret, forcast_num=forcast_num, auto_corr_num=auto_corr_num, half_life=half_life, calc_cov=False)**0.5
    Gamma = calcBlendingCoefficient(specific_ret)
    Std_STR = calcSTRSpecificRisk(Gamma, Std_TS, factor_data, cap)
    return Std_TS*Gamma + (1-Gamma)*Std_STR

# Eigenfactor Risk Adjustment
# factor_cov: DataFrame(因子协方差,index=[因子名],columns=[因子名]);
# 返回DataFrame(修正的因子协方差,index=[因子名],columns=[因子名]);
def EigenfactorRiskAdjustment(factor_cov, monte_carlo_num=1000, date_num=480, ignore_num=9, a=1.4, forcast_num=21, auto_corr_num=10, half_life_corr=480, half_life_vol=90):
    nFactor = factor_cov.shape[0]
    D0,U0 = np.linalg.eig(factor_cov.values)
    D0 = np.diag(D0)
    v = np.zeros(nFactor)
    for i in range(monte_carlo_num):
        ib = np.zeros((date_num,nFactor))
        for j in range(nFactor):
            ib[:,j] = np.random.randn(date_num)*D0[j]**0.5
        iFactorCov = estimateFactorCov_CHE2(pd.DataFrame(ib,columns=factor_cov.columns),forcast_num=forcast_num,auto_corr_num=auto_corr_num,half_life_corr=half_life_corr,half_life_vol=half_life_vol)
        iD,iU = np.linalg.eig(iFactorCov.values)
        iD_tiled = np.dot(iU.T,np.dot(factor_cov.values,iU))
        v += np.diag(iD_tiled)/np.diag(iD)
    v = (v/monte_carlo_num)**0.5
    Cof = np.polyfit(np.linspace(ignore_num+1,nFactor,nFactor-ignore_num),v[ignore_num:],2)
    p = np.poly1d(Cof)
    vp = p(np.linspace(1,nFactor,nFactor))
    vs = a*(vp-1)+1
    D0 = D0*vs**2
    return pd.DataFrame(np.dot(U0,np.dot(D0,U0.T)),index=factor_cov.index,columns=factor_cov.columns)
# Bayesian Shrinkage
# specific_risk: Series(特异性风险,index=[ID]); factor_data: DataFrame(因子暴露,index=[ID],columns=[因子名]);
# 返回修正后的特异性风险: Series(特异性风险,index=[ID])
def BayesianShrinkage(specific_risk, cap,quantile_num=10, q=0.1):
    Rslt = pd.Series(np.nan,index=specific_risk.index)
    Mask = pd.notnull(specific_risk)
    specific_risk = specific_risk[Mask]
    cap = cap[Mask]
    for i in range(quantile_num):
        if i==0:
            iIDs = cap[cap<cap.quantile((i+1)/quantile_num)].index
        elif i==quantile_num-1:
            iIDs = cap[cap>=cap.quantile(i/quantile_num)].index
        else:
            iIDs = cap[(cap>=cap.quantile(i/quantile_num)) & (cap<cap.quantile((i+1)/quantile_num))].index
        iSpecificRisk = specific_risk[iIDs]
        iStd = (iSpecificRisk*cap[iIDs]).sum()/cap[iIDs].sum()
        iDelta = ((iSpecificRisk-iStd)**2).sum()/iSpecificRisk.shape[0]
        iv = q*(iSpecificRisk-iStd).abs()/(q*(iSpecificRisk-iStd).abs()+iDelta)
        Rslt[iIDs] = iSpecificRisk*iv+(1-iv)*iStd
    return Rslt

# Volatility Regime Adjustment
# ret: DataFrame(收益率,index=[日期(频率为日)],columns=[ID或者因子]);
# forcast_volitility: DataFrame(波动率预测,index=[预测日期],columns=[ID或者因子]);
# half_life: 计算乘子的半衰期; forcast_num: 预测期数, 如果为<=0的数据则用forcast_volitility的日期间隔计算收益
# 返回调整乘子
def VolatilityRegimeAdjustment(ret, forcast_volitility, half_life=90, forcast_num=21):
    BiasStats = pd.Series(np.nan,index=forcast_volitility.index)
    for i,iDate in enumerate(forcast_volitility.index):
        iInd = (ret.index<=iDate).sum()-1
        if forcast_num>0:
            if ret.shape[0]>=iInd+forcast_num+1:
                iRet = (1+ret.iloc[iInd+1:iInd+forcast_num+1]).prod()-1
            else:
                continue
        else:
            if i==forcast_volitility.shape[0]-1:
                continue
            else:
                iNextDate = forcast_volitility.index[i+1]
                iNextInd = (ret.index<=iNextDate).sum()-1
                iRet = (1+ret.iloc[iInd+1:iNextInd+1]).prod()-1
        iTemp = (iRet/forcast_volitility.loc[iDate])**2
        BiasStats.loc[iDate] = iTemp.sum()/pd.notnull(iTemp).sum()
    BiasStats = BiasStats[pd.notnull(BiasStats)]
    Weight = getExpWeight(ret.shape[0],half_life=half_life,is_unitized=False)
    Weight.reverse()
    Weight = pd.Series(Weight,index=ret.index)
    Weight = Weight[BiasStats.index]/Weight[BiasStats.index].sum()
    return (Weight*BiasStats**2).sum()**0.5

# 去掉风险矩阵的缺失值
def dropRiskMatrixNA(risk_matrix):
    risk_matrix = risk_matrix.dropna(how='all',axis=0)
    risk_matrix = risk_matrix.loc[:,risk_matrix.index]
    risk_matrix = risk_matrix.dropna(how='any',axis=0)
    return risk_matrix.loc[:,risk_matrix.index]