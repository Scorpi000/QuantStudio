# coding=utf-8
"""策略相关函数"""
import os
import sys
import shutil
import datetime as dt
import time

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, norm
from scipy.integrate import quad
import statsmodels.api as sm

from QuantStudio.Tools.DateTimeFun import getDateSeries
from QuantStudio.Tools.FileFun import listDirFile, readCSV2Pandas
from QuantStudio import __QS_Error__

# 迭代法求解在考虑交易费且无交易限制假设下执行交易后的财富值
# p_holding: 当前持有的投资组合, Series(权重,index=[ID])
# p_target: 目标投资组合, Series(权重,index=[ID])
# wealth: 当前财富值
# transfee_rate: 交易费率
# epsilon: 容忍误差
# 返回: 考虑交易费执行交易后的财富值
# 算法: 迭代求解方程: x = wealth-sum(abs(x*p_target-wealth*p_holding))*transfee_rate, 其中x是未知量
def calcWealthAfterTrade(p_holding, p_target, wealth, transfee_rate, epsilon=1e-6):
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
    nRebalance, iPreRebalanceInd = len(rebalance_index), -1
    LNAV, SNAV = 1.0, 1.0
    for i in range(rebalance_index[0], long_yield.shape[0]):
        if (iPreRebalanceInd==nRebalance-1) or ((iPreRebalanceInd>=0) and (i<=rebalance_index[iPreRebalanceInd+1])):
            LSYield[i] = LNAV * long_yield[i] - SNAV * short_yield[i]
            LNAV *= (1 + long_yield[i])
            SNAV *= (1 + short_yield[i])
        if (iPreRebalanceInd<nRebalance-1) and (i==rebalance_index[iPreRebalanceInd+1]):
            LNAV, SNAV = 1.0, 1.0
            iPreRebalanceInd += 1
    return LSYield
# 计算年化收益率, wealth_seq: 净值序列, array
def calcAnnualYield(wealth_seq, num_per_year=252, start_dt=None, end_dt=None):
    if (start_dt is not None) and (end_dt is not None):
        nYear = (end_dt - start_dt).days / 365
    else:
        nYear = (wealth_seq.shape[0] - 1) / num_per_year
    return (wealth_seq[-1] / wealth_seq[0])**(1/nYear) - 1
# 计算年化波动率, wealth_seq: 净值序列, array
def calcAnnualVolatility(wealth_seq, num_per_year=252, start_dt=None, end_dt=None):
    YieldSeq = calcYieldSeq(wealth_seq)
    if (start_dt is not None) and (end_dt is not None):
        num_per_year = (YieldSeq.shape[0] - 1) / ((end_dt - start_dt).days / 365)
    return np.nanstd(YieldSeq, axis=0) * num_per_year**0.5
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
# 计算扩张年化波动率, wealth_seq: 净值序列, array
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
        x = (YieldSeq-Avg) / Std
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
    for i in range(UpOrDownSeq.shape[0]):
        if (not NowIsUp) and (UpOrDownSeq[i]):
            UpDates.append(i)
            UpWealth.append(wealth_seq[i])
            UpDates.append(i+1)
            UpWealth.append(wealth_seq[i+1])
            NowIsUp = True
            tempInd = tempInd+2
        elif (NowIsUp) and (UpOrDownSeq[i]):
            UpDates[tempInd-1] = i+1
            UpWealth[tempInd-1] = wealth_seq[i+1]
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
    for i in range(UpOrDownSeq.shape[0]):
        if (not NowIsDown) and (UpOrDownSeq[i]):
            DownDates.append(i)
            DownWealth.append(wealth_seq[i])
            DownDates.append(i+1)
            DownWealth.append(wealth_seq[i+1])
            NowIsDown = True
            tempInd = tempInd+2
        elif (NowIsDown) and (UpOrDownSeq[i]):
            DownDates[tempInd-1] = i+1
            DownWealth[tempInd-1] = wealth_seq[i+1]
        elif (NowIsDown) and (not UpOrDownSeq[i]):
            NowIsDown = False
    return (np.array(DownWealth),np.array(DownDates))
# 计算回撤序列, wealth_seq: 净值序列, array; 返回: (回撤率序列, 回撤期序列), (array, array)
def calcDrawdown(wealth_seq):
    HighWater = wealth_seq[0]# 高水位线
    Drawdown = np.zeros(wealth_seq.shape)# 回撤序列
    DrawdownDuration = np.zeros(wealth_seq.shape)# 回撤期
    for i, iWealth in enumerate(wealth_seq[1:]):
        HighWater = np.maximum(HighWater, iWealth)
        Drawdown[i+1] = iWealth / HighWater - 1
        DrawdownDuration[i+1] = (DrawdownDuration[i] + 1) * (Drawdown[i+1] < 0)
    return (Drawdown, DrawdownDuration)
# 计算最大回撤率, wealth_seq: 净值序列, array; 返回(最大回撤, 最大回撤开始位置, 最大回撤结束位置)
def calcMaxDrawdownRate(wealth_seq):
    HighWater = wealth_seq[0]# 高水位线
    MaxDrawdownRate = 0.0
    DrawdownStartInd = MaxDrawdownStartInd = MaxDrawdownEndInd = 0
    for i, iWealth in enumerate(wealth_seq[1:]):
        if iWealth<=HighWater:# 当前还未回复到历史高点
            iDrawdownRate = iWealth / HighWater - 1
            if iDrawdownRate<MaxDrawdownRate:
                MaxDrawdownRate = iDrawdownRate
                MaxDrawdownStartInd = DrawdownStartInd
                MaxDrawdownEndInd = i+1
        else:
            HighWater = iWealth
            DrawdownStartInd = i+1
    if MaxDrawdownEndInd>MaxDrawdownStartInd: return (MaxDrawdownRate, MaxDrawdownStartInd, MaxDrawdownEndInd)
    else: return (MaxDrawdownRate, None, None)
# 计算最长回撤期, wealth_seq: 净值序列, array; 返回(最长回撤期, 最长回撤开始位置, 最长回撤结束位置)
def calcMaxDrawdownDuration(wealth_seq):
    HighWater = wealth_seq[0]# 高水位线
    MaxDrawdownDuration = DrawdownDuration = MaxDrawdownEndInd = 0
    for i, iWealth in enumerate(wealth_seq[1:]):
        HighWater = max(HighWater, iWealth)
        DrawdownDuration = (DrawdownDuration + 1) * ( iWealth / HighWater < 1)
        if DrawdownDuration>MaxDrawdownDuration:
            MaxDrawdownDuration = DrawdownDuration
            MaxDrawdownEndInd = i+1
    if MaxDrawdownDuration>0: return (MaxDrawdownDuration, MaxDrawdownEndInd - MaxDrawdownDuration, MaxDrawdownEndInd)
    else: return (MaxDrawdownDuration, None, None)
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
    MaxDrawdownRate, _, _ = calcMaxDrawdownRate(wealth_seq)
    if expected_return is None:
        Denominator = np.nanmean(calcYieldSeq(wealth_seq)) - risk_free_rate
    else:
        Denominator = expected_return - risk_free_rate
    if MaxDrawdownRate==0: return np.sign(Denominator)*np.inf
    return Denominator / abs(MaxDrawdownRate)
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
def _densifyWealthSeq(wealth_seq, dts, dt_ruler=None):
    if dt_ruler is None: return (wealth_seq, dts)
    try:
        dt_ruler = dt_ruler[dt_ruler.index(dts[0]):dt_ruler.index(dts[-1])+1]
    except:
        return (wealth_seq, dts)
    nDT = len(dt_ruler)
    if nDT<=len(dts): return (wealth_seq, dts)
    DenseWealthSeq = np.zeros((nDT, ) + wealth_seq.shape[1:])
    DenseWealthSeq[0] = wealth_seq[0]
    for i, iDT in enumerate(dts[1:]):
        iStartInd = dt_ruler.index(dts[i])
        iInd = dt_ruler.index(iDT)
        iAvgYield = np.array([(wealth_seq[i+1] / wealth_seq[i])**(1 / (iInd-iStartInd)) - 1])
        iWealthSeq = np.cumprod(np.repeat(iAvgYield, iInd-iStartInd, axis=0)+1, axis=0) * wealth_seq[i]
        DenseWealthSeq[iStartInd+1:iInd+1] = iWealthSeq
    return (DenseWealthSeq, dt_ruler)
# 生成策略的统计指标
def summaryStrategy(wealth_seq, dts, dt_ruler=None, init_wealth=None):
    nCol = (wealth_seq.shape[1] if wealth_seq.ndim>1 else 1)
    if nCol==1: wealth_seq = wealth_seq.reshape((wealth_seq.shape[0], 1))
    wealth_seq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    YieldSeq = calcYieldSeq(wealth_seq, init_wealth)
    if init_wealth is None: init_wealth = wealth_seq[0]
    StartDT, EndDT = dts[0], dts[-1]
    SummaryIndex = ['起始时点', '结束时点']
    SummaryData = [np.array([StartDT]*nCol), np.array([EndDT]*nCol)]
    SummaryIndex.append('时点数')
    SummaryData.append(np.zeros(nCol) + len(dts))
    SummaryIndex.append('总收益率')
    SummaryData.append(wealth_seq[-1] / init_wealth - 1)
    SummaryIndex.append('年化收益率')
    SummaryData.append(calcAnnualYield(wealth_seq, start_dt=StartDT, end_dt=EndDT))
    SummaryIndex.append('年化波动率')
    SummaryData.append(calcAnnualVolatility(wealth_seq, start_dt=StartDT, end_dt=EndDT))
    SummaryIndex.append('Sharpe比率')
    SummaryData.append(SummaryData[4] / SummaryData[5])
    SummaryIndex.append('胜率')
    SummaryData.append(np.sum(YieldSeq>=0, axis=0) / np.sum(pd.notnull(YieldSeq), axis=0))
    SummaryIndex.extend(("最大回撤率", "最大回撤开始时点", "最大回撤结束时点"))
    MaxDrawdownRate, MaxDrawdownStartDT, MaxDrawdownEndDT = [], [], []
    for i in range(nCol):
        iMaxDrawdownRate, iMaxDrawdownStartPos, iMaxDrawdownEndPos = calcMaxDrawdownRate(wealth_seq=wealth_seq[:, i])
        MaxDrawdownRate.append(np.abs(iMaxDrawdownRate))
        MaxDrawdownStartDT.append((dts[iMaxDrawdownStartPos] if iMaxDrawdownStartPos is not None else None))
        MaxDrawdownEndDT.append((dts[iMaxDrawdownEndPos] if iMaxDrawdownEndPos is not None else None))
    SummaryData.extend((np.array(MaxDrawdownRate), np.array(MaxDrawdownStartDT), np.array(MaxDrawdownEndDT)))
    return pd.DataFrame(SummaryData, index=SummaryIndex)
# 计算每年的收益率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcReturnPerYear(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    Years, YearYield = [], []
    PreDT, StartInd = dts[0], 0
    for i, iDT in enumerate(dts[1:]):
        if iDT.year!=PreDT.year:# 进入新的年度
            Years.append(str(PreDT.year))
            YearYield.append(DenseWealthSeq[i]/DenseWealthSeq[StartInd]-1)
            StartInd = i
        PreDT = iDT
    Years.append(str(iDT.year))
    YearYield.append(DenseWealthSeq[-1] / DenseWealthSeq[StartInd] - 1)
    return pd.DataFrame(YearYield, index=Years)
# 计算每年的波动率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcVolatilityPerYear(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    Years, YearVol = [], []
    PreDT, StartInd = dts[0], 0
    for i, iDT in enumerate(dts[1:]):
        if iDT.year!=PreDT.year:# 进入新的年度
            Years.append(str(PreDT.year))
            iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:i+1])
            YearVol.append(np.nanstd(iYieldSeq, axis=0)*(i-StartInd)**0.5)
            StartInd = i
        PreDT = iDT
    Years.append(str(iDT.year))
    iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:])
    YearVol.append(np.nanstd(iYieldSeq, axis=0) * (i-StartInd)**0.5)
    return pd.DataFrame(YearVol, index=Years)
# 计算每年的最大回撤, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcMaxDrawdownPerYear(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    Years, YearMD = [], []
    PreDT, StartInd = dts[0], 0
    for i, iDT in enumerate(dts[1:]):
        if iDT.year!=PreDT.year:# 进入新的年度
            Years.append(str(PreDT.year))
            iWealthSeq = DenseWealthSeq[StartInd:i+1]
            if iWealthSeq.ndim==1:
                YearMD.append(abs(calcMaxDrawdownRate(iWealthSeq)[0]))
            else:
                YearMD.append(np.array([abs(calcMaxDrawdownRate(iWealthSeq[:, j])[0]) for j in range(iWealthSeq.shape[1])]))
            StartInd = i
        PreDT = iDT
    Years.append(str(iDT.year))
    iWealthSeq = DenseWealthSeq[StartInd:]
    if iWealthSeq.ndim==1:
        YearMD.append(abs(calcMaxDrawdownRate(iWealthSeq)[0]))
    else:
        YearMD.append(np.array([abs(calcMaxDrawdownRate(iWealthSeq[:,j])[0]) for j in range(iWealthSeq.shape[1])]))
    return pd.DataFrame(YearMD, index=Years)
# 计算每年每月的收益率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcReturnPerYearMonth(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    MonthYield, Months = [], []
    PreDT, StartInd = dts[0], 0
    for i, iDT in enumerate(dts[1:]):
        if (iDT.year!=PreDT.year) or (iDT.month!=PreDT.month):# 进入新的月度
            Months.append(PreDT.strftime("%Y%m"))
            MonthYield.append(DenseWealthSeq[i] / DenseWealthSeq[StartInd] - 1)
            StartInd = i
        PreDT = iDT
    Months.append(iDT.strftime("%Y%m"))
    MonthYield.append(DenseWealthSeq[-1] / DenseWealthSeq[StartInd] - 1)
    return pd.DataFrame(MonthYield, index=Months)
# 计算每年每月的波动率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcVolatilityPerYearMonth(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    MonthVol, Months = [], []
    PreDT, StartInd = dts[0], 0
    for i, iDT in enumerate(dts[1:]):
        if (iDT.year!=PreDT.year) or (iDT.month!=PreDT.month):# 进入新的月度
            Months.append(PreDT.strftime("%Y%m"))
            iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:i+1])
            MonthVol.append(np.nanstd(iYieldSeq, axis=0) * (i-StartInd)**0.5)
            StartInd = i
        PreDT = iDT
    Months.append(iDT.strftime("%Y%m"))
    iYieldSeq = calcYieldSeq(DenseWealthSeq[StartInd:])
    MonthVol.append(np.nanstd(iYieldSeq, axis=0) * (i-StartInd)**0.5)
    return pd.DataFrame(MonthVol, index=Months)
# 计算每年每月的最大回撤, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcMaxDrawdownPerYearMonth(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    MonthMD, Months = [], []
    PreDT, StartInd = dts[0], 0
    for i, iDT in enumerate(dts[1:]):
        if (iDT.year!=PreDT.year) or (iDT.month!=PreDT.month):# 进入新的月度
            Months.append(PreDT.strftime("%Y%m"))
            iWealthSeq = DenseWealthSeq[StartInd:i+1]
            if iWealthSeq.ndim==1:
                MonthMD.append(abs(calcMaxDrawdownRate(iWealthSeq)[0]))
            else:
                MonthMD.append(np.array([abs(calcMaxDrawdownRate(iWealthSeq[:,j])[0]) for j in range(iWealthSeq.shape[1])]))
            StartInd = i
        PreDT = iDT
    Months.append(iDT.strftime("%Y%m"))
    iWealthSeq = DenseWealthSeq[StartInd:]
    if iWealthSeq.ndim==1:
        MonthMD.append(abs(calcMaxDrawdownRate(iWealthSeq)[0]))
    else:
        MonthMD.append(np.array([abs(calcMaxDrawdownRate(iWealthSeq[:,j])[0]) for j in range(iWealthSeq.shape[1])]))
    return pd.DataFrame(MonthMD, index=Months)
# 计算每个年度月平均收益率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcAvgReturnPerMonth(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    MonthYield = np.zeros((12, ) + DenseWealthSeq.shape[1:])
    MonthNum = np.zeros(12)
    PreDT, StartInd = dts[0], 0
    for i, iDT in enumerate(dts[1:]):
        if (iDT.year!=PreDT.year) or (iDT.month!=PreDT.month):# 进入新的月度
            iTargetMonth = PreDT.month
            MonthYield[iTargetMonth-1] += DenseWealthSeq[i] / DenseWealthSeq[StartInd] - 1
            MonthNum[iTargetMonth-1] += 1
            StartInd = i
        PreDT = iDT
    MonthYield[iDT.month-1] += DenseWealthSeq[-1] / DenseWealthSeq[StartInd] - 1
    MonthNum[iDT.month-1] += 1
    for i in range(12):
        if MonthNum[i]==0: MonthYield[i] = np.nan
        else: MonthYield[i] = MonthYield[i] / MonthNum[i]
    return pd.DataFrame(MonthYield, index=[i+1 for i in range(12)])
# 计算每个周度日平均收益率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcAvgReturnPerWeekday(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    WeekdayYield = np.zeros((7, ) + DenseWealthSeq.shape[1:])
    WeekdayNum = np.zeros(7)
    for i, iDT in enumerate(dts[1:]):
        iWeekday = iDT.weekday()
        WeekdayNum[iWeekday-1] += 1
        WeekdayYield[iWeekday-1] += DenseWealthSeq[i+1] / DenseWealthSeq[i] - 1
    for i in range(7):
        if WeekdayNum[i]==0: WeekdayYield[i] = np.nan
        else: WeekdayYield[i] = WeekdayYield[i] / WeekdayNum[i]
    return pd.DataFrame(WeekdayYield, index=[i+1 for i in range(7)])
# 计算每个月度日平均收益率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcAvgReturnPerMonthday(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    MonthdayYield = np.zeros((31, ) + DenseWealthSeq.shape[1:])
    MonthdayNum = np.zeros(31)
    for i, iDT in enumerate(dts[1:]):
        iMonthday = iDT.day
        MonthdayNum[iMonthday-1] += 1
        MonthdayYield[iMonthday-1] += DenseWealthSeq[i+1] / DenseWealthSeq[i] - 1
    for i in range(31):
        if MonthdayNum[i]==0: MonthdayYield[i] = np.nan
        else: MonthdayYield[i] = MonthdayYield[i] / MonthdayNum[i]           
    return pd.DataFrame(MonthdayYield, index=[i+1 for i in range(31)])
# 计算每个年度日平均收益率, wealth_seq: 净值序列, dts: 时间序列, dt_ruler: 时间标尺
def calcAvgReturnPerYearday(wealth_seq, dts, dt_ruler=None):
    DenseWealthSeq, dts = _densifyWealthSeq(wealth_seq, dts, dt_ruler)
    YeardayYield = np.zeros((366, ) + DenseWealthSeq.shape[1:])
    YeardayNum = np.zeros(366)
    YeardaySeq = [iDT.strftime("%m%d") for iDT in getDateSeries(start_dt=dt.date(2000, 1, 1), end_dt=dt.date(2000, 12, 31))]
    for i, iDT in enumerate(dts[1:]):
        iInd = YeardaySeq.index(iDT.strftime("%m%d"))
        YeardayNum[iInd] += 1
        YeardayYield[iInd] += DenseWealthSeq[i+1] / DenseWealthSeq[i] - 1
    for i in range(366):
        if YeardayNum[i]==0: YeardayYield[i] = np.nan
        else: YeardayYield[i] = YeardayYield[i] / YeardayNum[i]
    return pd.DataFrame(YeardayYield, index=YeardaySeq)
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
    X[:,2] = X[:,1] * (X[:,1]>0)
    Rslt = sm.OLS(Y,X,missing='drop').fit()
    return Rslt.params
# C-L 模型, 评价择时能力和选股能力
def calcCLModel(wealth_seq, market_wealth_seq, risk_free_rate=0.0):
    Y = calcYieldSeq(wealth_seq)-risk_free_rate
    X = np.ones((Y.shape[0],3))
    rM = calcYieldSeq(market_wealth_seq)-risk_free_rate
    X[:,1] = rM * (rM<0)
    X[:,2] = rM * (rM>=0)
    Rslt = sm.OLS(Y,X,missing='drop').fit()
    return Rslt.params
# 加载CSV文件投资组合信号, 返回: {时点: 信号}
def loadCSVFilePortfolioSignal(csv_path):
    FileSignals = {}
    if not os.path.isfile(csv_path): raise __QS_Error__("文件: '%s' 不存在" % csv_path)
    with open(csv_path) as CSVFile:
        FirstLine = CSVFile.readline()
    if len(FirstLine.split(","))!=3:# 横向排列
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True)
        temp = list(CSVDF.columns)
        nCol = len(temp)
        AllSignalDates = [str(int(temp[i])) for i in range(0,nCol,2)]
        for i in range(int(nCol/2)):
            iDT = CSVDF.columns[i*2]
            iSignal = CSVDF.iloc[:,i*2:i*2+2]
            iSignal = iSignal[pd.notnull(iSignal.iloc[:,1])].set_index([iDT]).iloc[:,0]
            FileSignals[AllSignalDates[i]] = iSignal
    else:# 纵向排列
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True,header=0)
        AllSignalDates = pd.unique(CSVDF.iloc[:,0])
        AllColumns = list(CSVDF.columns)
        for iDT in AllSignalDates:
            iSignal = CSVDF.iloc[:, 1:][CSVDF.iloc[:,0]==iDT]
            iSignal = iSignal.set_index(AllColumns[1:2])
            iSignal = iSignal[AllColumns[2]]
            FileSignals[str(iDT)] = iSignal
    return FileSignals
# 将投资组合信号写入CSV文件
def writePortfolioSignal2CSV(signals, csv_path):
    AllDates = list(signals.keys())
    AllDates.sort()
    nDate = len(AllDates)
    nID = 0
    IDNums = [signals[iDT].shape[0] for iDT in AllDates]
    if IDNums==[]:
        np.savetxt(csv_path,np.array([]),fmt='%s',delimiter=',')
        return 0
    nID = max(IDNums)
    SignalArray = np.array([('',)*nDate*2]*(nID+1),dtype='O')
    for i, iDT in enumerate(AllDates):
        SignalArray[:IDNums[i]+1,2*i:2*i+2] = np.array([(iDT,'')]+list(signals[iDT].items()))
    try:
        np.savetxt(csv_path,SignalArray,fmt='%s',delimiter=',')
    except:
        return -1
    return 0
# 加载CSV文件择时信号, 返回: {时点: 信号}
def loadCSVFileTimingSignal(csv_path):
    FileSignals = {}
    if not os.path.isfile(csv_path):
        return FileSignals
    with open(csv_path) as CSVFile:
        FirstLine = CSVFile.readline().split(",")
    if (len(FirstLine)!=2) or (FirstLine[1] is not None):# 横向排列
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True,header=0,index_col=None)
        CSVDF = CSVDF.T
    else:# 纵向排列
        CSVDF = readCSV2Pandas(csv_path,detect_file_encoding=True,header=None,index_col=0)
    CSVDF = CSVDF.iloc[:,0]
    for iDT in CSVDF.index:
        FileSignals[str(iDT)] = CSVDF[iDT]
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
# 生成期货连续合约的价格序列
# id_map: 连续合约每一期的月合约 ID, Series(ID)
# price: 月合约的价格序列, DataFrame(价格, index=id_map.index, columns=[月合约ID])
# adj_direction: 调整方向, 可选: "前复权"(最后一期价格不变), "后复权"(第一期价格不变)
# adj_type: 调整方式, 可选: "收益率不变", "价差不变","价格不变"
# rollover_ahead: 合约展期是否提前一期, bool
# 返回: Series(价格, index=id_map.index)
def genContinuousContractPrice(id_map, price, adj_direction="前复权", adj_type="收益率不变", rollover_ahead=False):
    IDIndex = dict(zip(price.columns.tolist(), np.arange(price.shape[1])))
    AdjPrice = price.values[np.arange(id_map.shape[0]), list(map(lambda x: IDIndex.get(x, 0), id_map.tolist()))]
    AdjPrice[pd.isnull(id_map)] = np.nan
    if adj_type=="收益率不变":
        if adj_direction=="前复权":
            for i in range(id_map.shape[0]-1, 0, -1):
                iID, iPreID = id_map.iloc[i], id_map.iloc[i-1]
                if pd.isnull(iID) or pd.isnull(iPreID)  or (iID==iPreID): continue
                iAdj = price[iID].iloc[i-rollover_ahead] / price[iPreID].iloc[i-rollover_ahead]
                if pd.isnull(iAdj): iAdj = price[iID].iloc[i] / price[iPreID].iloc[i-1]
                AdjPrice[:i] = AdjPrice[:i] * iAdj
        elif adj_direction=="后复权":
            for i in range(1, id_map.shape[0]):
                iID, iPreID = id_map.iloc[i], id_map.iloc[i-1]
                if pd.isnull(iID) or pd.isnull(iPreID)  or (iID==iPreID): continue
                iAdj = price[iPreID].iloc[i-rollover_ahead] / price[iID].iloc[i-rollover_ahead]
                if pd.isnull(iAdj): iAdj = price[iPreID].iloc[i-1] / price[iID].iloc[i]
                AdjPrice[i:] = AdjPrice[i:] * iAdj
        else: raise __QS_Error__("不支持的调整方向: '%s'" % adj_direction)
    elif adj_type=="价差不变":
        if adj_direction=="前复权":
            for i in range(id_map.shape[0]-1, 0, -1):
                iID, iPreID = id_map.iloc[i], id_map.iloc[i-1]
                if pd.isnull(iID) or pd.isnull(iPreID)  or (iID==iPreID): continue
                iAdj = price[iPreID].iloc[i-rollover_ahead] - price[iID].iloc[i-rollover_ahead]
                if pd.isnull(iAdj): iAdj = price[iPreID].iloc[i-1] - price[iID].iloc[i]
                AdjPrice[:i] = AdjPrice[:i] - iAdj
        elif adj_direction=="后复权":
            for i in range(1, id_map.shape[0]):
                iID, iPreID = id_map.iloc[i], id_map.iloc[i-1]
                if pd.isnull(iID) or pd.isnull(iPreID)  or (iID==iPreID): continue
                iAdj = price[iPreID].iloc[i-rollover_ahead] - price[iID].iloc[i-rollover_ahead]
                if pd.isnull(iAdj): iAdj = price[iPreID].iloc[i-1] - price[iID].iloc[i]
                AdjPrice[i:] = AdjPrice[i:] + iAdj
        else: raise __QS_Error__("不支持的调整方向: '%s'" % adj_direction)
    elif adj_type!="价格不变": raise __QS_Error__("不支持的调整方式: '%s'" % adj_type)
    return pd.Series(AdjPrice, index=id_map.index)

# 给定持仓数量的策略向量化回测(非自融资策略)
# num_units: 每期的持仓数量, array(shape=(nDT, nID)), nDT: 时点数, nID: ID 数
# price: 价格序列, array(shape=(nDT, nID))
# fee: 手续费率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# long_margin: 多头保证金率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# short_margin: 空头保证金率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# 返回: (Return, PNL, Margin, Amount), (array(shape=(nDT, )), array(shape=(nDT, nID)), array(shape=(nDT, nID)), array(shape=(nDT, nID)))
def testNumStrategy(num_units, price, fee=0.0, long_margin=1.0, short_margin=1.0):
    Amount = (num_units * price)# shape=(nDT, nID)
    Margin = np.clip(Amount, 0, np.inf) * long_margin - np.clip(Amount, -np.inf, 0) * short_margin# shape=(nDT, nID)
    MoneyIn = np.r_[0, np.nansum(np.clip(Margin, 0, np.inf), axis=1)]# shape=(nDT+1, )
    Mask = (MoneyIn[:-1]==0)
    MoneyIn[:-1][Mask] = np.diff(MoneyIn)[Mask]
    num_units, price = np.r_[np.zeros((1, num_units.shape[1])), num_units], np.r_[np.zeros((1, price.shape[1])), price]# shape=(nDT+1, nID)
    PNL = np.diff(price, axis=0) * num_units[:-1] - np.abs(np.diff(num_units, axis=0) * price[1:]) * fee# shape=(nDT, nID)
    Return = np.nansum(PNL, axis=1) / MoneyIn[:-1]# shape=(nDT, )
    Mask = np.isinf(Return)
    Return[Mask] = np.sign(Return)[Mask]
    Return[np.isnan(Return)] = 0.0
    return (Return, PNL, Margin, Amount)

# 给定投资组合(持仓金额比例)的策略向量化回测(自融资策略)
# portfolio: 每期的投资组合, array(shape=(nDT, nID)), nDT: 时点数, nID: ID 数
# price: 价格序列, array(shape=(nDT, nID))
# fee: 手续费率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# long_margin: 多头保证金率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# short_margin: 空头保证金率, scalar, array(shape=(nID,)), array(shape=(nDT, nID))
# borrowing_rate: 借款利率, scalar, array(shape=(nDT, ))
# lending_rate: 贷款利率, scalar, array(shape=(nDT, ))
# 返回: (Return, Turnover), (array(shape=(nDT, )), array(shape=(nDT, )))
def testPortfolioStrategy(portfolio, price, fee=0.0, long_margin=1.0, short_margin=-1.0, borrowing_rate=0.0, lending_rate=0.0):
    Return = np.zeros_like(price)
    Return[1:] = price[1:] / price[:-1] - 1
    Mask = np.isinf(Return)
    Return[Mask] = np.sign(Return)[Mask]
    Return[np.isnan(Return)] = 0.0
    portfolio = np.r_[np.zeros((1, portfolio.shape[1])), portfolio]
    Return = np.nansum(portfolio[:-1] * Return, axis=1)
    Margin = np.nansum(np.clip(portfolio[:-1], 0, np.inf) * long_margin - np.clip(portfolio[:-1], -np.inf, 0) * short_margin, axis=1)
    Return += np.clip(1 - Margin, 0, np.inf) * borrowing_rate - np.clip(Margin - 1, 0, np.inf) * lending_rate
    Turnover = np.abs(np.diff(portfolio, axis=0))
    Return -=  np.nansum(Turnover * fee, axis=1)
    return (Return, np.nansum(Turnover, axis=1))