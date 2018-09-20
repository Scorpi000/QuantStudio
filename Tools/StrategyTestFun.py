# coding=utf-8
"""策略回测相关函数"""
import os
import sys
import shelve
import time

import numpy as np
import pandas as pd
from scipy.stats import skew,kurtosis,norm
from scipy.integrate import quad
import statsmodels.api as sm

from QuantStudio.Tools.DateTimeFun import DateStr2Datetime,getCurrentDateStr,getNaturalDay
from QuantStudio.Tools.FileFun import listDirFile,readCSV2Pandas

# 迭代法求解在考虑交易费且无交易限制假设下执行交易后的财富值
# p_holding: 当前持有的投资组合, Series(权重,index=[ID])
# p_target: 目标投资组合, Series(权重,index=[ID])
# wealth: 当前财富值
# transfee_rate: 交易费率
# epsilon: 容忍误差
# 返回: 考虑交易费执行交易后的财富值
# 算法: 迭代求解方程: x = wealth-sum(abs(x*p_target-wealth*p_holding))*transfee_rate, 其中x是未知量
def calcWealthAfterTrade(p_holding,p_target,wealth,transfee_rate,epsilon=1e-6):
    x_pre = 0.0
    x = wealth
    while abs(x-x_pre)>=epsilon:
        x_pre = x
        x = wealth - (x_pre*p_target-wealth*p_holding).abs().sum()*transfee_rate
    return x
# 计算投资组合的换手率
# old_p,new_p: {ID:投资比例}或者其他“类字典”形式
def calcTurnover(old_p, new_p):
    TurnOver = 0
    AllID = set(new_p.keys()).union(set(old_p.keys()))
    for iID in AllID:
        if (iID in old_p) and (iID in new_p):
            TurnOver += abs(new_p[iID]-old_p[iID])
        elif (iID not in old_p) and (iID in new_p):
            TurnOver += new_p[iID]
        elif (iID in old_p) and (iID not in new_p):
            TurnOver += old_p[iID]
    return TurnOver

# 计算投资组合收益率, portfolio:{ID:投资比例}或者其他“类字典”形式, return_rate:{ID:收益率}或者其他“类字典”形式
def calcPortfolioReturn(portfolio, return_rate):
    PortfolioReturn = 0
    for iID in portfolio:
        iRetRate = return_rate.get(iID)
        if pd.notnull(iRetRate):
            PortfolioReturn += portfolio[iID]*return_rate[iID]
        else:
            PortfolioReturn += portfolio[iID]*(-1)
    return PortfolioReturn
# 计算收益率序列, wealth_seq: 净值序列, array; init_wealth: 初始财富, 若为None使用wealth_seq的第一个元素
def calcYieldSeq(wealth_seq, init_wealth=None):
    YieldSeq = np.zeros(wealth_seq.shape)
    YieldSeq[1:] = (wealth_seq[1:]-wealth_seq[:-1])/wealth_seq[:-1]
    if init_wealth is not None:
        YieldSeq[0] = (wealth_seq[0]-init_wealth)/init_wealth
    return YieldSeq
# 计算净值序列, yield_seq: 收益率序列, array
def calcWealthSeq(yield_seq, init_wealth=None):
    WealthSeq = np.cumprod(1+yield_seq,axis=0)
    if init_wealth is not None:
        WealthSeq = np.append(1,WealthSeq)*init_wealth
    return WealthSeq
# 计算多空组合收益率序列, long_yield, short_yield: 多空收益率序列, array; rebalance_index: 再平衡位置索引, array, None 表示每个时点均进行再平衡
def calcLSYield(long_yield, short_yield, rebalance_index=None):
    if rebalance_index is None: return long_yield - short_yield
    LSYield = np.zeros(long_yield.shape[0])
    if not rebalance_index: return LSYield
    nRebalance, iRebalanceInd = len(rebalance_index), 0
    for i in range(rebalance_index[0], long_yield.shape[0]):
        if iRebalanceInd>0: LSYield[i] = LNAV * long_yield[i] - SNAV * short_yield[i]
        if (iRebalanceInd==nRebalance) or (i<rebalance_index[iRebalanceInd]):
            LNAV *= (1+long_yield[i])
            SNAV *= (1+short_yield[i])
        else:
            LNAV, SNAV = 1.0, 1.0
            iRebalanceInd += 1
    return LSYield
# 计算年化收益率, wealth_seq: 净值序列, array
def calcAnnualYield(wealth_seq, num_per_year=252, start_date=None, end_date=None):
    if (start_date is not None) and (end_date is not None):
        nYear = (end_date - start_date).days / 365
    else:
        nYear = (wealth_seq.shape[0] - 1) / num_per_year
    return (wealth_seq[-1] / wealth_seq[0])**(1/nYear) - 1
# 计算年化波动率, wealth_seq: 净值序列, array
def calcAnnualVolatility(wealth_seq, num_per_year=252, start_date=None, end_date=None):
    YieldSeq = calcYieldSeq(wealth_seq)
    if (start_date is not None) and (end_date is not None):
        num_per_year = (YieldSeq.shape[0] - 1) / ((end_date - start_date).days / 365)
    return np.nanstd(YieldSeq,axis=0) * num_per_year**0.5
# 计算滚动年化收益率, wealth_seq: 净值序列, array
def calcRollingAnnualYieldSeq(wealth_seq, window=252, min_window=252, num_per_year=252):
    RollingAnnualYieldSeq = np.zeros(wealth_seq.shape)+np.nan
    for i,iWealth in enumerate(wealth_seq):
        if i>=window:
            iStartInd = i-window
        elif i>=min_window:
            iStartInd = i-min_window
        else:
            continue
        iPreWealth = wealth_seq[iStartInd]
        nYear = (i-iStartInd)/num_per_year
        RollingAnnualYieldSeq[i] = (iWealth/iPreWealth)**(1/nYear)-1
    return RollingAnnualYieldSeq
# 计算滚动年化波动率, wealth_seq: 净值序列, array
def calcRollingAnnualVolatilitySeq(wealth_seq, window=252, min_window=252, num_per_year=252):
    YieldSeq = calcYieldSeq(wealth_seq)
    RollingAnnualVolatilitySeq = np.zeros(YieldSeq.shape)+np.nan
    for i,iWealth in enumerate(YieldSeq):
        if i>=window:
            iStartInd = i-window
        elif i>=min_window:
            iStartInd = i-min_window
        else:
            continue
        iYieldSeq = YieldSeq[iStartInd:i+1]
        RollingAnnualVolatilitySeq[i] = np.nanstd(iYieldSeq,axis=0)*num_per_year**0.5
    return RollingAnnualVolatilitySeq
# 计算扩张年化收益率, wealth_seq: 净值序列, array
def calcExpandingAnnualYieldSeq(wealth_seq, min_window=252, num_per_year=252):
    ExpandingAnnualYieldSeq = np.zeros(wealth_seq.shape)+np.nan
    for i,iWealth in enumerate(wealth_seq):
        if i<min_window:
            continue
        nYear = i/num_per_year
        ExpandingAnnualYieldSeq[i] = (iWealth/wealth_seq[0])**(1/nYear)-1
    return ExpandingAnnualYieldSeq
# 计算滚动年化波动率, wealth_seq: 净值序列, array
def calcExpandingAnnualVolatilitySeq(wealth_seq, min_window=252, num_per_year=252):
    YieldSeq = calcYieldSeq(wealth_seq)
    ExpandingAnnualVolatilitySeq = np.zeros(YieldSeq.shape)+np.nan
    for i,iWealth in enumerate(YieldSeq):
        if i<min_window:
            continue
        iYieldSeq = YieldSeq[0:i+1]
        ExpandingAnnualVolatilitySeq[i] = np.nanstd(iYieldSeq,axis=0)*num_per_year**0.5
    return ExpandingAnnualVolatilitySeq
# 计算 beta, wealth_seq: 净值序列, array, market_wealth_seq: 市场收益率序列, array
def calcBeta(wealth_seq, market_wealth_seq):
    YieldSeq = calcYieldSeq(wealth_seq)
    MarketYieldSeq = calcYieldSeq(market_wealth_seq)
    Mask = (pd.notnull(MarketYieldSeq) & pd.notnull(YieldSeq))
    return np.cov(YieldSeq[Mask],MarketYieldSeq[Mask])[0,1]/np.nanvar(MarketYieldSeq)
# 计算收益率的 Lower Partial Moment, wealth_seq: 净值序列, array
def calcLPM(wealth_seq, threshold=0.0, order=2):
    YieldSeq = calcYieldSeq(wealth_seq)
    # This method returns a lower partial moment of the returns
    # Create an array has same length as wealth_seq containing the minimum return threshold
    ThresholdArray = np.empty(YieldSeq.shape[0])
    ThresholdArray.fill(threshold)
    # Calculate the difference between the threshold and the returns
    Diff = ThresholdArray - YieldSeq
    # Set the minimum of each to 0
    Diff = Diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.nansum(Diff ** order) / np.sum(pd.notnull(YieldSeq))
# 计算收益率的 Higher Partial Moment, wealth_seq: 净值序列, array
def calcHPM(wealth_seq, threshold=0.0, order=2):
    YieldSeq = calcYieldSeq(wealth_seq)
    # This method returns a lower partial moment of the returns
    # Create an array has same length as wealth_seq containing the minimum return threshold
    ThresholdArray = np.empty(YieldSeq.shape[0])
    ThresholdArray.fill(threshold)
    # Calculate the difference between the threshold and the returns
    Diff = YieldSeq - ThresholdArray
    # Set the minimum of each to 0
    Diff = Diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.nansum(Diff ** order) / np.sum(pd.notnull(YieldSeq))
# 计算夏普比率, wealth_seq: 净值序列, array
def calcSharpeRatio(wealth_seq, risk_free_rate=0.0, expected_return=None):
    YieldSeq = calcYieldSeq(wealth_seq)
    if expected_return is None:
        return (np.nanmean(YieldSeq) - risk_free_rate) / np.nanstd(YieldSeq)
    else:
        return (expected_return - risk_free_rate) / np.nanstd(YieldSeq)
# 计算特雷诺比率, wealth_seq: 净值序列, array, market_wealth_seq: 市场净值序列, array
def calcTreynorRatio(wealth_seq, market_wealth_seq, risk_free_rate=0.0, expected_return=None):
    YieldSeq = calcYieldSeq(wealth_seq)
    MarketYieldSeq = calcYieldSeq(market_wealth_seq)
    if expected_return is None:
        return (np.nanmean(YieldSeq) - risk_free_rate) / calcBeta(wealth_seq, market_wealth_seq)
    else:
        return (expected_return - risk_free_rate) / calcBeta(wealth_seq, market_wealth_seq)
# 计算调整的夏普比率, 参见: The Relative Merits of Investable Hedge Fund Indices and of Funds of Hedge Funds in Optimal Passive Portfolios, wealth_seq: 净值序列, array
def calcAdjustedSharpeRatio(wealth_seq, risk_free_rate=0.0, expected_return=None):
    SR = calcSharpeRatio(wealth_seq, risk_free_rate=risk_free_rate, expected_return=expected_return)
    YieldSeq = calcYieldSeq(wealth_seq)
    Skewness = skew(YieldSeq,axis=0,nan_policy='omit')
    Kurtosis = kurtosis(YieldSeq,axis=0,nan_policy='omit')
    return SR*(1+Skewness/6*SR-(Kurtosis-3)/24*SR**2)
# 计算信息比率, wealth_seq: 净值序列, array, benchmark_wealth_seq: 基准净值序列, array
def calcInformationRatio(wealth_seq, benchmark_wealth_seq):
    YieldSeq = calcYieldSeq(wealth_seq)
    BenchmarkYieldSeq = calcYieldSeq(benchmark_wealth_seq)
    Diff = YieldSeq - BenchmarkYieldSeq
    return np.nanmean(Diff) / np.nanstd(Diff)
# 计算 Modigliani Ratio, wealth_seq: 净值序列, array, benchmark_wealth_seq: 基准净值序列, array
def calcModiglianiRatio(wealth_seq, benchmark_wealth_seq, risk_free_rate=0.0, expected_return=None):
    YieldSeq = calcYieldSeq(wealth_seq)
    BenchmarkYieldSeq = calcYieldSeq(benchmark_wealth_seq)
    np_rf = np.empty(YieldSeq.shape[0])
    np_rf.fill(risk_free_rate)
    rdiff = YieldSeq - np_rf
    bdiff = BenchmarkYieldSeq - np_rf
    if expected_return is None:
        return (np.nanmean(YieldSeq) - risk_free_rate) * (np.nanstd(rdiff) / np.nanstd(bdiff)) + risk_free_rate
    else:
        return (expected_return - risk_free_rate) * (np.nanstd(rdiff) / np.nanstd(bdiff)) + risk_free_rate
# 计算 Sortino Ratio, wealth_seq: 净值序列, array
def calcSortinoRatio(wealth_seq, risk_free_rate=0.0, target_return=0.0, expected_return=None):
    if expected_return is None:
        return (np.nanmean(calcYieldSeq(wealth_seq)) - risk_free_rate) / calcLPM(wealth_seq, threshold=target_return, order=2)**0.5
    else:
        return (expected_return - risk_free_rate) / calcLPM(wealth_seq, threshold=target_return, order=2)**0.5
# 计算 Omega Ratio, wealth_seq: 净值序列, array
def calcOmegaRatio(wealth_seq, risk_free_rate=0.0, target_return=0.0, expected_return=None):
    if expected_return is None:
        return (np.nanmean(calcYieldSeq(wealth_seq)) - risk_free_rate) / calcLPM(wealth_seq, target_return, 1)
    else:
        return (expected_return - risk_free_rate) / calcLPM(wealth_seq, target_return, 1)
# 计算 Kappa3 Ratio, wealth_seq: 净值序列, array
def calcKappaThreeRatio(wealth_seq, risk_free_rate=0.0, target_return=0.0, expected_return=None):
    if expected_return is None:
        return (np.nanmean(calcYieldSeq(wealth_seq)) - risk_free_rate) / calcLPM(wealth_seq, target_return, 3)**(1/3)
    else:
        return (expected_return - risk_free_rate) / calcLPM(wealth_seq, target_return, 3)**(1/3)
# 计算盈亏比率, wealth_seq: 净值序列, array
def calcGainLossRatio(wealth_seq, target_return=0.0):
    return calcHPM(wealth_seq, target_return, 1) / calcLPM(wealth_seq, target_return, 1)
# 计算 Upside Potential Ratio, wealth_seq: 净值序列, array
def calcUpsidePotentialRatio(wealth_seq, target_return=0):
    return calcHPM(wealth_seq, target_return, 1) / calcLPM(wealth_seq, target_return, 2)**0.5
# 计算 VaR 和 CVaR, wealth_seq: 净值序列, array
def calcVaR(wealth_seq, alpha=0.05, method="Historical"):
    YieldSeq = calcYieldSeq(wealth_seq)
    if method=='Historical':
        VaR = np.nanpercentile(YieldSeq,alpha)
        CVaR = np.nanmean(YieldSeq[YieldSeq<VaR])
        return (VaR,CVaR)
    Avg = np.nanmean(YieldSeq)
    Std = np.nanstd(YieldSeq)
    if method=='Norm':
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
    else:
        VaR = np.nan
        CVaR = np.nan
    return (VaR,CVaR)
# 计算财富上升波段, wealth_seq: 净值序列, array
def calcUpPeriod(wealth_seq):
    UpDates = []
    UpWealth = []
    UpOrDownSeq = (np.diff(wealth_seq)>=0)
    NowIsUp = False
    tempInd = 0
    for i in range(1,UpOrDownSeq.shape[0]):
        if (not NowIsUp) and (UpOrDownSeq[i]):
            UpDates.append(i-1)
            UpWealth.append(wealth_seq[i-1])
            UpDates.append(i)
            UpWealth.append(wealth_seq[i])
            NowIsUp = True
            tempInd = tempInd+2
        elif (NowIsUp) and (UpOrDownSeq[i]):
            UpDates[tempInd-1] = i
            UpWealth[tempInd-1] = wealth_seq[i]
        elif (NowIsUp) and (not UpOrDownSeq[i]):
            NowIsUp = False
    return (np.array(UpWealth),np.array(UpDates))
# 计算财富下降波段, wealth_seq: 净值序列, array
def calcDownPeriod(wealth_seq):
    DownDates = []
    DownWealth = []
    UpOrDownSeq = (np.diff(wealth_seq)<0)
    NowIsDown = False
    tempInd = 0
    for i in range(1,len(UpOrDownSeq)):
        if (not NowIsDown) and (UpOrDownSeq[i]):
            DownDates.append(i-1)
            DownWealth.append(wealth_seq[i-1])
            DownDates.append(i)
            DownWealth.append(wealth_seq[i])
            NowIsDown = True
            tempInd = tempInd+2
        elif (NowIsDown) and (UpOrDownSeq[i]):
            DownDates[tempInd-1] = i
            DownWealth[tempInd-1] = wealth_seq[i]
        elif (NowIsDown) and (not UpOrDownSeq[i]):
            NowIsDown = False
    return (np.array(DownWealth),np.array(DownDates))
# 计算回撤序列, wealth_seq: 净值序列, array; 返回(最大回撤,最大回撤开始位置,最大回撤结束位置,回撤序列)
def calcDrawdown(wealth_seq):
    High = wealth_seq[0]
    DrawdownSeq = np.zeros(wealth_seq.shape)
    MaxDrawdownRate = 0.0
    DrawdownStartDate = 0
    MaxDrawdownStartDate = 0
    MaxDrawdownEndDate = 0
    for i,iWealth in enumerate(wealth_seq[1:]):
        if iWealth<High:# 当前还未回复到历史高点
            iDrawdownRate = iWealth/High-1
            DrawdownSeq[i+1] = iDrawdownRate
            if iDrawdownRate<MaxDrawdownRate:
                MaxDrawdownRate = iDrawdownRate
                MaxDrawdownStartDate = DrawdownStartDate
                MaxDrawdownEndDate = i+1
        else:
            High = iWealth
            DrawdownStartDate = i+1
    if MaxDrawdownEndDate>MaxDrawdownStartDate:
        return (MaxDrawdownRate,MaxDrawdownStartDate,MaxDrawdownEndDate,DrawdownSeq)
    else:
        return (MaxDrawdownRate,None,None,DrawdownSeq)
# 计算最大回撤率, 最大回撤开始日期, 最大回撤结束日期, wealth_seq: 净值序列, array; 返回(最大回撤,最大回撤开始时间,最大回撤结束时间)
def calcMaxDrawdownRate(wealth_seq):
    MaxDrawdownRate = 0
    MaxDrawdownStartDate = 0
    MaxDrawdownEndDate = 0
    for i,iWealth in enumerate(wealth_seq[1:]):
        iMaxInd = np.argmax(wealth_seq[:i+1])
        iMaxWealth = wealth_seq[:i+1][iMaxInd]
        if iMaxWealth!=0:
            iMinInd = np.argmin(wealth_seq[i:])
            iMinWealth = wealth_seq[i:][iMinInd]
            iMaxDrawdownRate = (iMaxWealth-iMinWealth)/iMaxWealth
        else:
            continue
        if iMaxDrawdownRate>MaxDrawdownRate:
            MaxDrawdownStartDate = iMaxInd
            MaxDrawdownEndDate = iMinInd
            MaxDrawdownRate = iMaxDrawdownRate
    return (MaxDrawdownRate,MaxDrawdownStartDate,MaxDrawdownEndDate)
# 计算最大回撤率(加速版), 最大回撤开始日期, 最大回撤结束日期, wealth_seq: 净值序列, array; 返回(最大回撤,最大回撤开始时间,最大回撤结束时间)
def calcMaxDrawdownRateEx(wealth_seq):
    iBackwardMaxInd = 0
    iBackwardMaxWealth = wealth_seq[0]
    iForwardMinInd = np.argmin(wealth_seq)
    iForwardMinWealth = wealth_seq[iForwardMinInd]
    MaxDrawdownRate = (iBackwardMaxWealth-iForwardMinWealth)/iBackwardMaxWealth
    BackwardMaxInd = iBackwardMaxInd
    ForwardMinInd = iForwardMinInd
    for i,iWealth in enumerate(wealth_seq[1:]):
        if iWealth>iBackwardMaxWealth:
            iBackwardMaxWealth = iWealth
            iBackwardMaxInd = i+1
        if iForwardMinInd==i:
            iForwardMinInd = np.argmin(wealth_seq[i+1:])+i+1
            iForwardMinWealth = wealth_seq[iForwardMinInd]
        iMaxDrawdownRate = ((iBackwardMaxWealth-iForwardMinWealth)/iBackwardMaxWealth if iBackwardMaxWealth!=0 else np.nan)
        if iMaxDrawdownRate>MaxDrawdownRate:
            MaxDrawdownRate = iMaxDrawdownRate
            BackwardMaxInd = iBackwardMaxInd
            ForwardMinInd = iForwardMinInd
    return (MaxDrawdownRate,BackwardMaxInd,ForwardMinInd)
# 计算最大回撤率(二次加速版), 最大回撤开始日期, 最大回撤结束日期, wealth_seq: 净值序列, array; 返回(最大回撤,最大回撤开始时间,最大回撤结束时间)
def calcMaxDrawdownRateExEx(wealth_seq):
    High = wealth_seq[0]
    MaxDrawdownRate = 0.0
    DrawdownStartInd = 0
    MaxDrawdownStartInd= 0
    MaxDrawdownEndInd = 0
    for i,iWealth in enumerate(wealth_seq[1:]):
        if iWealth<=High:# 当前还未回复到历史高点
            iDrawdownRate = iWealth/High-1
            if iDrawdownRate<MaxDrawdownRate:
                MaxDrawdownRate = iDrawdownRate
                MaxDrawdownStartInd = DrawdownStartInd
                MaxDrawdownEndInd = i+1
        else:
            High = iWealth
            DrawdownStartInd = i+1
    if MaxDrawdownEndInd>MaxDrawdownStartInd:
        return (abs(MaxDrawdownRate),MaxDrawdownStartInd,MaxDrawdownEndInd)
    else:
        return (abs(MaxDrawdownRate),None,None)
# 计算给定期限的最大回撤
def calcPeriodDrawdown(wealth_seq, tau):
    # Returns the draw-down given time period tau
    pos = wealth_seq.shape[0] - 1
    pre = pos - tau
    drawdown = 0.0
    # Find the maximum drawdown given period tau
    while pre >= 0:
        dd_i = (wealth_seq[pos] / wealth_seq[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    return drawdown
# 计算给定期限内的平均回撤
def calcAverageDrawdown(wealth_seq, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, wealth_seq.shape[0]):
        drawdown_i = calcPeriodDrawdown(wealth_seq, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = drawdowns[0]
    for i in range(1, periods):
        total_dd += drawdowns[i]
    return total_dd / periods
# 计算给定期限内的平均回撤平方
def calcAverageDrawdownSquared(wealth_seq, periods):
    # Returns the average maximum drawdown squared over n periods
    drawdowns = []
    for i in range(0, wealth_seq.shape[0]):
        drawdown_i = calcPeriodDrawdown(wealth_seq, i)**2
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns,reverse=True)
    total_dd = drawdowns[0]
    for i in range(1, periods):
        total_dd += drawdowns[i]
    return total_dd / periods
# 计算 Calmar Ratio
def calcCalmarRatio(wealth_seq, risk_free_rate=0.0, expected_return=None):
    MaxDrawdownRate,MaxDrawdownStartDate,MaxDrawdownEndDate = calcMaxDrawdownRate(wealth_seq)
    if expected_return is None:
        Denominator = np.nanmean(calcYieldSeq(wealth_seq)) - risk_free_rate
    else:
        Denominator = expected_return - risk_free_rate
    if MaxDrawdownRate==0:
        return np.sign(Denominator)*np.inf
    return Denominator / MaxDrawdownRate
# 计算 Sterling Ratio
def calcSterlingRatio(wealth_seq, periods, risk_free_rate=0.0, expected_return=None):
    AverageDrawdown = calcAverageDrawdown(wealth_seq,periods)
    if expected_return is None:
        Denominator = np.nanmean(calcYieldSeq(wealth_seq)) - risk_free_rate
    else:
        Denominator = expected_return - risk_free_rate
    if AverageDrawdown==0:
        return np.sign(Denominator)*np.inf
    return Denominator / AverageDrawdown
# 计算 Burke Ratio
def calcBurkeRatio(wealth_seq, periods, risk_free_rate=0.0, expected_return=None):
    AverageDrawdownSquared = calcAverageDrawdownSquared(wealth_seq,periods)
    if expected_return is None:
        Denominator = np.nanmean(calcYieldSeq(wealth_seq)) - risk_free_rate
    else:
        Denominator = expected_return - risk_free_rate
    if AverageDrawdownSquared==0:
        return np.sign(Denominator)*np.inf
    return Denominator / AverageDrawdownSquared**0.5
# 以更高的频率扩充净值序列
def _densifyWealthSeq(wealth_seq, dates, date_ruler=None):
    if date_ruler is None:
        return (wealth_seq, dates)
    else:
        try:
            date_ruler = date_ruler[date_ruler.index(dates[0]):date_ruler.index(dates[-1])+1]
        except:
            return (wealth_seq,dates)
    nDate = len(date_ruler)
    if nDate<=len(dates):
        return (wealth_seq,dates)
    DenseWealthSeq = np.zeros((nDate,)+wealth_seq.shape[1:])
    DenseWealthSeq[0] = wealth_seq[0]
    for i,iDate in enumerate(dates[1:]):
        iStartInd = date_ruler.index(dates[i])
        iInd = date_ruler.index(iDate)
        iAvgYield = np.array([(wealth_seq[i+1]/wealth_seq[i])**(1/(iInd-iStartInd))-1])
        iWealthSeq = np.cumprod(np.repeat(iAvgYield,iInd-iStartInd,axis=0)+1,axis=0)*wealth_seq[i]
        DenseWealthSeq[iStartInd+1:iInd+1] = iWealthSeq
    return (DenseWealthSeq,date_ruler)
# 生成策略的统计指标
def summaryStrategy(wealth_seq, dates, date_ruler=None, init_wealth=None):
    wealth_seq, dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    YieldSeq = calcYieldSeq(wealth_seq, init_wealth)
    if init_wealth is None: init_wealth = wealth_seq[0]
    nCol = (wealth_seq.shape[1] if wealth_seq.ndim>1 else 1)
    StartDate, EndDate = dates[0], dates[-1]
    SummaryIndex = ['起始日', '结束日']
    SummaryData = [np.array([StartDate]*nCol), np.array([EndDate]*nCol)]
    SummaryIndex.append('日期数')
    SummaryData.append(np.zeros(nCol) + len(dates))
    SummaryIndex.append('总收益率')
    SummaryData.append(wealth_seq[-1] / init_wealth - 1)
    SummaryIndex.append('年化收益率')
    SummaryData.append(calcAnnualYield(wealth_seq, start_date=StartDate, end_date=EndDate))
    SummaryIndex.append('年化波动率')
    SummaryData.append(calcAnnualVolatility(wealth_seq, start_date=StartDate, end_date=EndDate))
    SummaryIndex.append('Sharpe比率')
    SummaryData.append(SummaryData[4] / SummaryData[5])
    SummaryIndex.append('胜率')
    SummaryData.append(np.sum(YieldSeq>=0, axis=0) / np.sum(pd.notnull(YieldSeq), axis=0))
    SummaryIndex.extend(("最大回撤率", "最大回撤开始日期", "最大回撤结束日期"))
    if wealth_seq.ndim==1:
        MaxDrawdownRate, MaxDrawdownStartPos, MaxDrawdownEndPos, _ = calcDrawdown(wealth_seq=wealth_seq)
        SummaryData.extend((np.abs(MaxDrawdownRate), dates[MaxDrawdownStartPos], dates[MaxDrawdownEndPos]))
    else:
        MaxDrawdownRate, MaxDrawdownStartDate, MaxDrawdownEndDate = [], [], []
        for i in range(nCol):
            iMaxDrawdownRate, iMaxDrawdownStartPos, iMaxDrawdownEndPos, _ = calcDrawdown(wealth_seq=wealth_seq[:, i])
            MaxDrawdownRate.append(np.abs(iMaxDrawdownRate))
            MaxDrawdownStartDate.append((dates[iMaxDrawdownStartPos] if iMaxDrawdownStartPos is not None else None))
            MaxDrawdownEndDate.append((dates[iMaxDrawdownEndPos] if iMaxDrawdownEndPos is not None else None))
        SummaryData.extend((np.array(MaxDrawdownRate), np.array(MaxDrawdownStartDate), np.array(MaxDrawdownEndDate)))
    return pd.DataFrame(SummaryData, index=SummaryIndex)
# 计算每年的收益率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcReturnPerYear(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    YearYield = []
    Years = []
    PreDate = dates[0]
    StartInd = 0
    for i,iDate in enumerate(dates[1:]):
        if iDate[:4]!=PreDate[:4]:# 进入新的年度
            Years.append(PreDate[:4])
            YearYield.append(DenseWealthSeq[i]/DenseWealthSeq[StartInd]-1)
            StartInd = i
        PreDate = iDate
    Years.append(iDate[:4])
    YearYield.append(DenseWealthSeq[-1]/DenseWealthSeq[StartInd]-1)
    return pd.DataFrame(YearYield,index=Years)
# 计算每年的波动率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcVolatilityPerYear(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    YearVol = []
    Years = []
    PreDate = dates[0]
    StartInd = 0
    for i,iDate in enumerate(dates[1:]):
        if iDate[:4]!=PreDate[:4]:# 进入新的年度
            Years.append(PreDate[:4])
            iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:i+1])
            YearVol.append(np.nanstd(iYieldSeq,axis=0)*(i-StartInd)**0.5)
            StartInd = i
        PreDate = iDate
    Years.append(iDate[:4])
    iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:])
    YearVol.append(np.nanstd(iYieldSeq,axis=0)*(i-StartInd)**0.5)
    return pd.DataFrame(YearVol,index=Years)
# 计算每年的最大回撤, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcMaxDrawdownPerYear(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    YearMD = []
    Years = []
    PreDate = dates[0]
    StartInd = 0
    for i,iDate in enumerate(dates[1:]):
        if iDate[:4]!=PreDate[:4]:# 进入新的年度
            Years.append(PreDate[:4])
            iWealthSeq = DenseWealthSeq[StartInd:i+1]
            if iWealthSeq.ndim==1:
                YearMD.append(calcMaxDrawdownRateExEx(iWealthSeq)[0])
            else:
                YearMD.append(np.array([calcMaxDrawdownRateExEx(iWealthSeq[:,j])[0] for j in range(iWealthSeq.shape[1])]))
            StartInd = i
        PreDate = iDate
    Years.append(iDate[:4])
    iWealthSeq = DenseWealthSeq[StartInd:]
    if iWealthSeq.ndim==1:
        YearMD.append(calcMaxDrawdownRateExEx(iWealthSeq)[0])
    else:
        YearMD.append(np.array([calcMaxDrawdownRateExEx(iWealthSeq[:,j])[0] for j in range(iWealthSeq.shape[1])]))
    return pd.DataFrame(YearMD,index=Years)
# 计算每年每月的收益率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcReturnPerYearMonth(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    MonthYield = []
    Months = []
    PreDate = dates[0]
    StartInd = 0
    for i,iDate in enumerate(dates[1:]):
        if iDate[:6]!=PreDate[:6]:# 进入新的月度
            Months.append(PreDate[:6])
            MonthYield.append(DenseWealthSeq[i]/DenseWealthSeq[StartInd]-1)
            StartInd = i
        PreDate = iDate
    Months.append(iDate[:6])
    MonthYield.append(DenseWealthSeq[-1]/DenseWealthSeq[StartInd]-1)
    return pd.DataFrame(MonthYield,index=Months)
# 计算每年每月的波动率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcVolatilityPerYearMonth(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    MonthVol = []
    Months = []
    PreDate = dates[0]
    StartInd = 0
    for i,iDate in enumerate(dates[1:]):
        if iDate[:6]!=PreDate[:6]:# 进入新的月度
            Months.append(PreDate[:6])
            iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:i+1])
            MonthVol.append(np.nanstd(iYieldSeq,axis=0)*(i-StartInd)**0.5)
            StartInd = i
        PreDate = iDate
    Months.append(iDate[:6])
    iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:])
    MonthVol.append(np.nanstd(iYieldSeq,axis=0)*(i-StartInd)**0.5)
    return pd.DataFrame(MonthVol,index=Months)
# 计算每年每月的最大回撤, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcMaxDrawdownPerYearMonth(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    MonthMD = []
    Months = []
    PreDate = dates[0]
    StartInd = 0
    for i,iDate in enumerate(dates[1:]):
        if iDate[:6]!=PreDate[:6]:# 进入新的月度
            Months.append(PreDate[:6])
            iWealthSeq = DenseWealthSeq[StartInd:i+1]
            if iWealthSeq.ndim==1:
                MonthMD.append(calcMaxDrawdownRateExEx(iWealthSeq)[0])
            else:
                MonthMD.append(np.array([calcMaxDrawdownRateExEx(iWealthSeq[:,j])[0] for j in range(iWealthSeq.shape[1])]))
            StartInd = i
        PreDate = iDate
    Months.append(iDate[:6])
    iWealthSeq = DenseWealthSeq[StartInd:]
    if iWealthSeq.ndim==1:
        MonthMD.append(calcMaxDrawdownRateExEx(iWealthSeq)[0])
    else:
        MonthMD.append(np.array([calcMaxDrawdownRateExEx(iWealthSeq[:,j])[0] for j in range(iWealthSeq.shape[1])]))
    return pd.DataFrame(MonthMD,index=Months)
# 计算每个年度月平均收益率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcAvgReturnPerMonth(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    MonthYield = np.zeros((12,)+DenseWealthSeq.shape[1:])
    MonthNum = np.zeros(12)
    PreDate = dates[0]
    StartInd = 0
    for i,iDate in enumerate(dates[1:]):
        if iDate[:6]!=PreDate[:6]:# 进入新的月度
            iTargetMonth = int(PreDate[4:6])
            MonthYield[iTargetMonth-1] += DenseWealthSeq[i]/DenseWealthSeq[StartInd]-1
            MonthNum[iTargetMonth-1] += 1
            StartInd = i
        PreDate = iDate
    MonthYield[int(iDate[4:6])-1] += DenseWealthSeq[-1]/DenseWealthSeq[StartInd]-1
    MonthNum[int(iDate[4:6])-1] += 1
    for i in range(12):
        if MonthNum[i]==0:
            MonthYield[i] = np.nan
        else:
            MonthYield[i] = MonthYield[i]/MonthNum[i]
    return pd.DataFrame(MonthYield,index=[i+1 for i in range(12)])
# 计算每个周度日平均收益率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcAvgReturnPerWeekday(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    WeekdayYield = np.zeros((7,)+DenseWealthSeq.shape[1:])
    WeekdayNum = np.zeros(7)
    for i,iDate in enumerate(dates[1:]):
        iWeekday = DateStr2Datetime(iDate).weekday()
        WeekdayNum[iWeekday-1] += 1
        WeekdayYield[iWeekday-1] += DenseWealthSeq[i+1]/DenseWealthSeq[i]-1
    for i in range(7):
        if WeekdayNum[i]==0:
            WeekdayYield[i] = np.nan
        else:
            WeekdayYield[i] = WeekdayYield[i]/WeekdayNum[i]
    return pd.DataFrame(WeekdayYield,index=[i+1 for i in range(7)])
# 计算每个月度日平均收益率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcAvgReturnPerMonthday(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    MonthdayYield = np.zeros((31,)+DenseWealthSeq.shape[1:])
    MonthdayNum = np.zeros(31)
    for i,iDate in enumerate(dates[1:]):
        iMonthday = DateStr2Datetime(iDate).day
        MonthdayNum[iMonthday-1] += 1
        MonthdayYield[iMonthday-1] += DenseWealthSeq[i+1]/DenseWealthSeq[i]-1
    for i in range(31):
        if MonthdayNum[i]==0:
            MonthdayYield[i] = np.nan
        else:
            MonthdayYield[i] = MonthdayYield[i]/MonthdayNum[i]
    return pd.DataFrame(MonthdayYield,index=[i+1 for i in range(31)])
# 计算每个年度日平均收益率, wealth_seq: 净值序列, dates: 日期序列, date_ruler: 日期标尺
def calcAvgReturnPerYearday(wealth_seq, dates, date_ruler=None):
    DenseWealthSeq,dates = _densifyWealthSeq(wealth_seq, dates, date_ruler)
    YeardayYield = np.zeros((366,)+DenseWealthSeq.shape[1:])
    YeardayNum = np.zeros(366)
    YeardaySeq = getNaturalDay("20000101","20001231")
    YeardaySeq = [iDate[4:] for iDate in YeardaySeq]
    for i,iDate in enumerate(dates[1:]):
        iInd = YeardaySeq.index(iDate[4:])
        YeardayNum[iInd] += 1
        YeardayYield[iInd] += DenseWealthSeq[i+1]/DenseWealthSeq[i]-1
    for i in range(5):
        if YeardayNum[i]==0:
            YeardayYield[i] = np.nan
        else:
            YeardayYield[i] = YeardayYield[i]/YeardayNum[i]
    return pd.DataFrame(YeardayYield,index=YeardaySeq)
# T-M 二项式模型, 评价择时能力和选股能力
def calcTMModel(wealth_seq, market_wealth_seq, risk_free_rate=0.0):
    Y = calcYieldSeq(wealth_seq)-risk_free_rate
    X = np.ones((Y.shape[0],3))
    X[:,1] = calcYieldSeq(market_wealth_seq)-risk_free_rate
    X[:,2] = X[:,1]**2
    Rslt = sm.OLS(Y,X,missing='drop').fit()
    return Rslt.params
# H-M 双贝塔模型, 评价择时能力和选股能力
def calcHMModel(wealth_seq, market_wealth_seq, risk_free_rate=0.0):
    Y = calcYieldSeq(wealth_seq)-risk_free_rate
    X = np.ones((Y.shape[0],3))
    X[:,1] = calcYieldSeq(market_wealth_seq)-risk_free_rate
    X[:,2] = X[:,1]*(X[:,1]>0)
    Rslt = sm.OLS(Y,X,missing='drop').fit()
    return Rslt.params
# C-L 模型, 评价择时能力和选股能力
def calcCLModel(wealth_seq, market_wealth_seq, risk_free_rate=0.0):
    Y = calcYieldSeq(wealth_seq)-risk_free_rate
    X = np.ones((Y.shape[0],3))
    rM = calcYieldSeq(market_wealth_seq)-risk_free_rate
    X[:,1] = rM*(rM<0)
    X[:,2] = rM*(rM>=0)
    Rslt = sm.OLS(Y,X,missing='drop').fit()
    return Rslt.params
# 生成策略的Excel报告
def genStrategyExcelReport(save_path,template_path,strategy_name,output,save_type="新建"):
    if (template_path=='') or (save_path==''):
        return 0
    if strategy_name=='':
        NewSheetName = "策略表现"
    else:
        NewSheetName = strategy_name
    import shutil
    import xlwings as xw
    if (save_type=='新建') or (not os.path.exists(save_path)):
        shutil.copy(template_path, save_path)
        xlBook = xw.Book(save_path)
        xlBook.sheets['策略表现'].name = NewSheetName
    else:
        xlBook = xw.Book(save_path)
        TempBook = xw.Book(template_path)
        TempBook.sheets["策略表现"].api.Copy(Before=xlBook.sheets[0].api)
        TempBook.close()
        AllSheetNames = [iSheet.name for iSheet in xlBook.sheets]
        if (NewSheetName in AllSheetNames) and (NewSheetName!=xlBook.sheets[xlBook.sheets.count-1].name):
            xlBook.app.display_alerts = False
            xlBook.sheets[NewSheetName].delete()
            xlBook.app.display_alerts = True
        xlBook.sheets[0].name = NewSheetName
    CurSheet = xlBook.sheets[NewSheetName]
    # 写入日期序列
    CurSheet[1,0].expand().clear_contents()
    Dates = [iDate[:4]+'-'+iDate[4:6]+'-'+iDate[6:] for iDate in output['净值'].index]
    CurSheet[1,0].options(transpose=True).value = Dates
    # 写入净值数据
    Data = output['净值'].iloc[:,3:].copy()
    if Data.shape[1]==1:# 只有策略多头, 补充策略多空, 基准, 对冲
        Data['策略多空'] = Data['策略多头']
        Data['基准'] = 1.0
        Data['对冲'] = Data['策略多头']
    elif Data.shape[1]==2:# 只有策略多头, 策略多空, 补充基准, 对冲
        Data['基准'] = 1.0
        Data['对冲'] = Data['策略多头']
    elif Data.shape[1]==3:# 只有策略多头, 基准, 对冲, 补充策略多空
        Data.insert(1,'策略多空',Data['策略多头'])
    CurSheet[1,1].value = Data.values
    # 写入收益率数据
    CurSheet[2,5].expand().clear_contents()
    Data = output['收益率'].copy()
    if Data.shape[1]==1:# 只有策略多头, 补充策略多空, 基准, 对冲
        Data['策略多空'] = Data['策略多头']
        Data['基准'] = 0.0
        Data['对冲'] = Data['策略多头']
    elif Data.shape[1]==2:# 只有策略多头, 策略多空, 补充基准, 对冲
        Data['基准'] = 0.0
        Data['对冲'] = Data['策略多头']
    elif Data.shape[1]==3:# 只有策略多头, 基准, 对冲, 补充策略多空
        Data.insert(1,'策略多空',Data['策略多头'])
    CurSheet[2,5].value = Data.values
    # 写入回撤数据
    CurSheet[1,9].expand().clear_contents()
    Data = output['回撤'].copy()
    if Data.shape[1]==1:# 只有策略多头, 补充策略多空, 基准, 对冲
        Data['策略多空'] = Data['策略多头']
        Data['基准'] = 0.0
        Data['对冲'] = Data['策略多头']
    elif Data.shape[1]==2:# 只有策略多头, 策略多空, 补充基准, 对冲
        Data['基准'] = 0.0
        Data['对冲'] = Data['策略多头']
    elif Data.shape[1]==3:# 只有策略多头, 基准, 对冲, 补充策略多空
        Data.insert(1,'策略多空',Data['策略多头'])
    CurSheet[1,9].value = Data.values
    # 绘制净值, 收益率, 回撤图像
    DataLen = output['净值'].shape[0]
    Chrt = CurSheet.charts["多头净值"].api[1]
    Chrt.SeriesCollection(1).Values = CurSheet[1:DataLen+1,1].api
    Chrt.SeriesCollection(2).Values = CurSheet[1:DataLen+1,3].api
    Chrt.SeriesCollection(3).Values = CurSheet[1:DataLen+1,9].api
    Chrt.SeriesCollection(1).XValues = CurSheet[1:DataLen+1,0].api
    Chrt = CurSheet.charts["多空净值收益率"].api[1]
    Chrt.SeriesCollection(1).Values = CurSheet[1:DataLen+1,2].api
    Chrt.SeriesCollection(2).Values = CurSheet[1:DataLen+1,6].api
    Chrt.SeriesCollection(1).XValues = CurSheet[1:DataLen+1,0].api
    Chrt = CurSheet.charts["对冲净值收益率"].api[1]
    Chrt.SeriesCollection(1).Values = CurSheet[1:DataLen+1,4].api
    Chrt.SeriesCollection(2).Values = CurSheet[1:DataLen+1,8].api
    Chrt.SeriesCollection(1).XValues = CurSheet[1:DataLen+1,0].api
    Chrt = CurSheet.charts["多空净值回撤"].api[1]
    Chrt.SeriesCollection(1).Values = CurSheet[1:DataLen+1,2].api
    Chrt.SeriesCollection(2).Values = CurSheet[1:DataLen+1,10].api
    Chrt.SeriesCollection(1).XValues = CurSheet[1:DataLen+1,0].api
    Chrt = CurSheet.charts["对冲净值回撤"].api[1]
    Chrt.SeriesCollection(1).Values = CurSheet[1:DataLen+1,4].api
    Chrt.SeriesCollection(2).Values = CurSheet[1:DataLen+1,12].api
    Chrt.SeriesCollection(1).XValues = CurSheet[1:DataLen+1,0].api
    # 写入换手率数据
    CurSheet[1,14].expand().clear_contents()
    CurSheet["Q:Q"].clear_contents()
    CurSheet["Q1"].value = "滚动平均"
    Data = output['换手率']
    Data = Data[Data['换手率']!=0]
    Dates = [iDate[:4]+'-'+iDate[4:6]+'-'+iDate[6:] for iDate in Data.index]
    CurSheet[1,14].options(transpose=True).value = Dates
    CurSheet[1,15].value = Data.values
    nYear = ((DateStr2Datetime(Data.index[-1])-DateStr2Datetime(Data.index[0])).days)/365
    nAvg = round(Data.shape[0]/nYear)
    AvgData = Data.copy()
    for i in range(Data.shape[0]):
        if i<nAvg-1:
            AvgData['换手率'].iloc[i] = np.nan
        else:
            AvgData['换手率'].iloc[i] = Data['换手率'].iloc[i-nAvg+1:i+1].mean()
    if Data.shape[0]>=nAvg:
        CurSheet[nAvg,16].value = AvgData.iloc[nAvg-1:,:].values
    Chrt = CurSheet.charts["换手率"].api[1]
    Chrt.SeriesCollection(1).Values = CurSheet[1:Data.shape[0]+1,15].api
    Chrt.SeriesCollection(2).Values = CurSheet[1:Data.shape[0]+1,16].api
    Chrt.SeriesCollection(1).XValues = CurSheet[1:Data.shape[0]+1,14].api
    # 写入统计数据
    CurSheet[1,18].expand().clear_contents()
    Index = ["总收益率","年化收益率","年化波动率","Sharpe比率","最大回撤率","最大回撤开始日期","最大回撤结束日期","胜率"]
    CurSheet[1,18].options(transpose=True).value = Index
    CurSheet[1,19].value = output['统计数据'].loc[Index,["策略多头","策略多空","基准","对冲"]].values
    # 写入年度统计数据
    CurSheet[0,24].expand().clear_contents()
    if "年度统计" in output:
        Data = output['年度统计']
        CurSheet[0,24].value = '年份'
        CurSheet[0,25].value = list(Data.columns)
        CurSheet[1,24].options(transpose=True).value = list(Data.index)
        CurSheet[1,25].value = Data.values
        Chrt = CurSheet.charts["对冲年度统计"].api[1]
        Chrt.SeriesCollection(1).Values = CurSheet[1:Data.shape[0]+1,25].api
        Chrt.SeriesCollection(1).XValues = CurSheet[1:Data.shape[0]+1,24].api
        if Data.shape[1]>=3:
            Chrt.SeriesCollection(2).Values = CurSheet[1:Data.shape[0]+1,25+Data.shape[1]-2].api
            Chrt.SeriesCollection(3).Values = CurSheet[1:Data.shape[0]+1,25+Data.shape[1]-1].api
        else:
            Chrt.SeriesCollection(3).Delete()
            Chrt.SeriesCollection(2).Delete()
        Chrt = CurSheet.charts["多空年度统计"].api[1]
        Chrt.SeriesCollection(1).Values = CurSheet[1:Data.shape[0]+1,25].api
        Chrt.SeriesCollection(1).XValues = CurSheet[1:Data.shape[0]+1,24].api
        if (Data.shape[1]==2) or (Data.shape[1]==4):
            Chrt.SeriesCollection(2).Values = CurSheet[1:Data.shape[0]+1,26].api
        else:
            Chrt.SeriesCollection(2).Delete()
        CurSheet.api.ListObjects.Add(1, CurSheet[0:Data.shape[0]+1,24:25+Data.shape[1]].api,True,1).Name = "YearStatisticsTable"
        CurSheet.api.ListObjects("YearStatisticsTable").TableStyle = "TableStyleLight9"
        StartCol = 31
    else:
        CurSheet.charts["对冲年度统计"].delete()
        CurSheet.charts["多空年度统计"].delete()
        for i in range(6):
            CurSheet.api.Columns(25).Delete()
        StartCol = 25
    # 写入月度统计数据
    CurSheet[0,StartCol-1].expand().clear_contents()
    if "月度统计" in output:
        Data = output['月度统计']
        CurSheet[0,StartCol-1].value = '月份'
        CurSheet[0,StartCol].value = list(Data.columns)
        CurSheet[1,StartCol-1].options(transpose=True).value = list(Data.index)
        CurSheet[1,StartCol].value = Data.values
        Chrt = CurSheet.charts["对冲月度统计"].api[1]
        Chrt.SeriesCollection(1).Values = CurSheet[1:Data.shape[0]+1,StartCol].api
        Chrt.SeriesCollection(1).XValues = CurSheet[1:Data.shape[0]+1,StartCol-1].api
        if Data.shape[1]>=3:
            Chrt.SeriesCollection(2).Values = CurSheet[1:Data.shape[0]+1,StartCol+Data.shape[1]-2].api
            Chrt.SeriesCollection(3).Values = CurSheet[1:Data.shape[0]+1,StartCol+Data.shape[1]-1].api
        else:
            Chrt.SeriesCollection(3).Delete()
            Chrt.SeriesCollection(2).Delete()
        Chrt = CurSheet.charts["多空月度统计"].api[1]
        Chrt.SeriesCollection(1).Values = CurSheet[1:Data.shape[0]+1,StartCol].api
        Chrt.SeriesCollection(1).XValues = CurSheet[1:Data.shape[0]+1,StartCol-1].api
        if (Data.shape[1]==2) or (Data.shape[1]==4):
            Chrt.SeriesCollection(2).Values = CurSheet[1:Data.shape[0]+1,StartCol+1].api
        else:
            Chrt.SeriesCollection(2).Delete()
        CurSheet.api.ListObjects.Add(1, CurSheet[0:Data.shape[0]+1,StartCol-1:StartCol+Data.shape[1]].api,True,1).Name = "MonthStatisticsTable"
        CurSheet.api.ListObjects("MonthStatisticsTable").TableStyle = "TableStyleLight9"
    else:
        CurSheet.charts["对冲月度统计"].delete()
        CurSheet.charts["多空月度统计"].delete()
    xlBook.save()
    xlBook.app.quit()
    return 0
# 加载CSV文件投资组合信号, 返回: {日期:信号}
def loadCSVFilePortfolioSignal(csv_path):
    FileSignals = {}
    if not os.path.isfile(csv_path):
        return FileSignals
    with open(csv_path) as CSVFile:
        FirstLine = CSVFile.readline()
    if len(FirstLine.split(","))==3:
        file_type="纵向排列"
    else:
        file_type="横向排列"
    if file_type=='横向排列':
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True)
        temp = list(CSVDF.columns)
        nCol = len(temp)
        AllSignalDates = [str(int(temp[i])) for i in range(0,nCol,2)]
        for i in range(int(nCol/2)):
            iDate = CSVDF.columns[i*2]
            iSignal = CSVDF.iloc[:,i*2:i*2+2]
            iSignal = iSignal[pd.notnull(iSignal.iloc[:,1])].set_index([iDate]).iloc[:,0]
            FileSignals[AllSignalDates[i]] = iSignal
    else:
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True,header=0)
        AllSignalDates = pd.unique(CSVDF.iloc[:,0])
        AllColumns = list(CSVDF.columns)
        for iDate in AllSignalDates:
            iSignal = CSVDF.ix[CSVDF.iloc[:,0]==iDate,1:]
            iSignal = iSignal.set_index(AllColumns[1:2])
            iSignal = iSignal[AllColumns[2]]
            FileSignals[str(iDate)] = iSignal
    return FileSignals
# 将投资组合信号写入CSV文件
def writePortfolioSignal2CSV(signals, csv_path):
    AllDates = list(signals.keys())
    AllDates.sort()
    nDate = len(AllDates)
    nID = 0
    IDNums = [signals[iDate].shape[0] for iDate in AllDates]
    if IDNums==[]:
        np.savetxt(csv_path,np.array([]),fmt='%s',delimiter=',')
        return 0
    nID = max(IDNums)
    SignalArray = np.array([('',)*nDate*2]*(nID+1),dtype='O')
    for i,iDate in enumerate(AllDates):
        SignalArray[:IDNums[i]+1,2*i:2*i+2] = np.array([(iDate,'')]+list(signals[iDate].items()))
    try:
        np.savetxt(csv_path,SignalArray,fmt='%s',delimiter=',')
    except:
        return -1
    return 0
# 加载CSV文件择时信号, 返回: {时间戳: 信号}
def loadCSVFileTimingSignal(csv_path):
    FileSignals = {}
    if not os.path.isfile(csv_path):
        return FileSignals
    with open(csv_path) as CSVFile:
        FirstLine = CSVFile.readline().split(",")
    if (len(FirstLine)!=2) or (DateStr2Datetime(FirstLine[1]) is not None):
        file_type="横向排列"
    else:
        file_type="纵向排列"
    if file_type=='横向排列':
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True,header=0,index_col=None)
        CSVDF = CSVDF.T
    else:
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True,header=None,index_col=0)
    CSVDF = CSVDF.iloc[:,0]
    for iDate in CSVDF.index:
        FileSignals[str(iDate)] = CSVDF[iDate]
    return FileSignals
# 生成随机投资组合, ids: 初始股票池
def genRandomPortfolio(ids, target_num=20, weight=None):
    IDs = np.random.choice(np.array(ids),target_num,replace=False)
    IDs.sort()
    if weight is None:
        return pd.Series(1/IDs.shape[0],index=IDs)
    Portfolio = weight[IDs]
    Portfolio = Portfolio[pd.notnull(Portfolio) & (Portfolio!=0)]
    return Portfolio/Portfolio.sum()
# 以筛选的方式形成投资组合
def genPortfolioByFiltration(factor_data, ascending=False, target_num=20, target_quantile=0.1, weight=None):
    factor_data = factor_data[pd.notnull(factor_data)]
    factor_data = factor_data.sort_values(inplace=False,ascending=ascending)
    if target_num is not None:
        TargetIDs = set(factor_data.iloc[:target_num].index)
    else:
        TargetIDs = set(factor_data.index)
    if target_quantile is not None:
        if ascending:
            TargetIDs = set(factor_data[factor_data<=factor_data.quantile(target_quantile)].index).intersection(TargetIDs)
        else:
            TargetIDs = set(factor_data[factor_data<=factor_data.quantile(target_quantile)].index).intersection(TargetIDs)
    TargetIDs = list(TargetIDs)
    TargetIDs.sort()
    Portfolio = weight[TargetIDs]
    Portfolio = Portfolio[pd.notnull(Portfolio) & (Portfolio!=0)]
    return Portfolio/Portfolio.sum()