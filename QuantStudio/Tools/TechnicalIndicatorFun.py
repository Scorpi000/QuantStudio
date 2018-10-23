# -*- coding: utf-8 -*-
"""技术指标"""
import pandas as pd
import numpy as np

# --------------------------------------------------------------均线-----------------------------------------------------------------------------
# 简单移动平均线
# p: 价格序列, array
# n_ma: 计算平均值的样本长度, int
# min_periods: 最小计算窗口, int
# 返回: array, 计算窗口长度不足的填充np.nan
def MA(p, n_ma=5, min_periods=5):
    MA = np.zeros(p.shape)+np.nan
    for i in range(min_periods-1,MA.shape[0],1):
        MA[i] = np.nanmean(p[max((i-n_ma+1,0)):i+1],axis=0)
    return MA

# 指数移动平均线
# p: 价格序列, array
# n_ma: 计算窗口长度, int
# init_value: 初始值, 如果为None, 以p的第一个值为初始值
# 返回: array
def EMA(p, n_ma=5, init_value=None):
    nDim = np.ndim(p)
    if nDim==1:
        p = np.reshape(p,(p.shape[0],1))
    EMA = np.zeros(p.shape)+np.nan
    SmoothCoef = 2/(n_ma+1)
    if init_value is None:
        PreEMA = np.copy(p[0])
    else:
        PreEMA = np.copy(init_value)
    for i in range(0,EMA.shape[0],1):
        Mask = pd.isnull(PreEMA)
        PreEMA[Mask] = p[i][Mask]
        EMA[i] = SmoothCoef*p[i]+(1-SmoothCoef)*PreEMA
        PreEMA = np.copy(EMA[i])
    if nDim==1:
        return EMA[:,0]
    else:
        return EMA

# 自适应均线
# p: 价格序列, array
# n_er: 计算效率系数的窗口长度, int
# n_fast: 快均线的窗口长度, int
# n_slow: 慢均线的窗口长度, int
# min_periods: 最小计算窗口, int
# init_value: 初始值, 如果为None, 以p的第一个值为初始值
# 返回: array, 计算窗口长度不足的填充np.nan
def AMA(p, n_er=10, n_fast=2, n_slow=60, min_periods=10, init_value=None):
    nDim = np.ndim(p)
    if nDim==1:
        p = np.reshape(p,(p.shape[0],1))
    AMA = np.zeros(p.shape)+np.nan
    if init_value is None:
        PreAMA = np.copy(p[min_periods-1])
    else:
        PreAMA = np.copy(init_value)
    for i in range(min_periods-1,AMA.shape[0],1):
        iStartInd = max((i-n_er+1,0))
        iDirection = np.abs(p[i]-p[iStartInd])
        iVol = np.nansum(np.abs(np.diff(p[iStartInd:i+1],axis=0)),axis=0)
        iER = iDirection/iVol
        iSmooth = (iER*2/(n_fast+1)+(1-iER)*2/(n_slow+1))**2
        AMA[i] = p[i]*iSmooth+(1-iSmooth)*PreAMA
        PreAMA = np.copy(AMA[i])
    if nDim==1:
        return AMA[:,0]
    else:
        return AMA

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 指数平滑异同平均线(MACD)
# p: 价格序列, array
# n_short: 短期均线的窗口长度, int, 使用EMA计算
# n_long: 长期均线的窗口长度, int, 使用EMA计算
# n_dea: 计算 DEA 的窗口长度, int
# min_periods: 最小计算窗口, int
# init_sema: 初始值, 如果为None, 以p的第一个值为初始值
# init_lema: 初始值, 如果为None, 以p的第一个值为初始值
# init_dea: 初始值, 如果为None, 以dif的第一个值为初始值
# 返回: (DIF, DEA, MACD, SEMA, LEMA)
# SEMA: 短期移动平均线, 使用EMA计算, array
# LEMA: 长期移动平均线, 使用EMA计算, array
# DIF: SEMA - LEMA, array
# DEA: DIF的移动平均值, 使用EMA计算, array
# MACD: 2 * (DIF - DEA), array
def MACD(p, n_short=12, n_long=26, n_dea=9, return_pos=None, init_sema=None, init_lema=None, init_dea=None):
    SEMA = EMA(p, n_short, init_sema)
    LEMA = EMA(p,n_long, init_lema)
    DIF = SEMA - LEMA
    DEA = EMA(DIF, n_dea, init_dea)
    MACD = 2 * (DIF - DEA)
    if return_pos is None:
        return (DIF, DEA, MACD, SEMA, LEMA)
    else:
        return (DIF, DEA, MACD, SEMA, LEMA)[return_pos]
# 随机指标(KDJ)
# p_close: 收盘价序列, array
# p_high: 最高价序列, array
# p_low: 最低价序列, array
# n_rsv: 计算 RSV 的窗口长度, int
# n_k: 计算 K 的窗口长度(M1), int
# n_d: 计算 D 的窗口长度(M2), int
# 返回: (K, D, J, RSV)
# 未成熟随机值(RSV) = （收盘价－N日内最低价） / （N日内最高价－N日内最低价） × 100
# K: RSV 的 M1 日移动平均, array
# D: K 值的 M2 日移动平均, array
# J: 3 × K － 2 × D, array
def KDJ(p_close, p_high, p_low, n_rsv=9, n_k=3, n_d=3, min_periods=9, return_pos=None, init_rsv=None, init_k=None, init_d=None):
    nDim = np.ndim(p_close)
    if nDim==1:
        p_close = np.reshape(p_close,(p_close.shape[0],1))
        p_high = np.reshape(p_high,(p_high.shape[0],1))
        p_low = np.reshape(p_low,(p_low.shape[0],1))
    RSV = np.zeros(p_close.shape)+np.nan
    for i in range(min_periods-1,p_close.shape[0],1):
        iStartInd = max((i-n_rsv+1,0))
        iMin = np.nanmin(p_low[iStartInd:i+1],axis=0)
        iMax = np.nanmax(p_high[iStartInd:i+1],axis=0)
        RSV[i] = (p_close[i]-iMin)/(iMax-iMin)
        iMask = (iMin==iMax)
        RSV[i][iMask] = np.nan
    RSV = RSV*100
    K = EMA(RSV, n_k, init_k)
    D = EMA(K, n_d, init_d)
    J = 3*K - 2*D
    if nDim==1:
        Rslt = (K[:,0],D[:,0],J[:,0],RSV[:,0])
    else:
        Rslt = (K,D,J,RSV)
    if return_pos is None:
        return Rslt
    else:
        return Rslt[return_pos]
# 威廉指标(WR)
# p_close: 收盘价序列, array
# p_high: 最高价序列, array
# p_low: 最低价序列, array
# n_wr: 计算 WR 的窗口长度, int
# min_periods: 最小计算窗口, int
# 返回: WR: （N日内最高价 - 收盘价） / （N日内最高价 - N日内最低价） * 100, array
def WR(p_close, p_high, p_low, n_wr=10, min_periods=10):
    nDim = np.ndim(p_close)
    if nDim==1:
        p_close = np.reshape(p_close,(p_close.shape[0],1))
        p_high = np.reshape(p_high,(p_high.shape[0],1))
        p_low = np.reshape(p_low,(p_low.shape[0],1))
    WR = np.zeros(p_close.shape)+np.nan
    for i in range(min_periods-1,p_close.shape[0],1):
        iStartInd = max((i-n_wr+1,0))
        iMin = np.nanmin(p_high[iStartInd:i+1],axis=0)
        iMax = np.nanmax(p_low[iStartInd:i+1],axis=0)
        WR[i] = (iMax-p_close[i])/(iMax-iMin)*100
        iMask = (iMin==iMax)
        WR[i][iMask] = np.nan
    if nDim==1:
        return WR[:,0]
    else:
        return WR
# 相对强弱指标(RSI)
# p: 价格序列, array
# n_rsi: 计算 RSI 的窗口长度, int
# min_periods: 最小计算窗口, int
# 返回: RSI: N日内上涨幅度累计 / N日内上涨及下跌幅度累计 * 100, array
def RSI(p, n_rsi=6, min_periods=6):
    nDim = np.ndim(p)
    if nDim==1:
        p = np.reshape(p,(p.shape[0],1))
    RSI = np.zeros(p.shape)+np.nan
    Ret = np.zeros(p.shape)+np.nan
    Ret[1:] = p[1:]/p[:-1]-1
    for i in range(min_periods,p.shape[0],1):
        iRet = Ret[max((i-n_rsi+1,0)):i+1]
        iTemp = np.nansum(np.abs(iRet),axis=0)
        RSI[i] = np.nansum(iRet*(iRet>0),axis=0)/iTemp*100
        iMask = (iTemp==0)
        RSI[iMask] = np.nan
    if nDim==1:
        return RSI[:,0]
    else:
        return RSI
# 三重指数平滑移动平均指标(TRIX)
# p: 价格序列, Series()
# n_p: 计算均线的窗口长度, int
# m: 计算MATRIX的窗口长度, int
# 返回(TRIX, MATRIX, TR)
# TR: p的N日指数移动平均的N日指数移动平均的N日指数移动平均, 使用EMA计算, array
# TRIX: (TR-昨日TR)/昨日TR*100, array
# MATRIX: TRIX的M日简单移动平均, 使用MA计算, array
def TRIX(p, n_tr=12, m=20, return_pos=None):
    TR = EMA(EMA(EMA(p,n_tr),n_tr),n_tr)
    TRIX = np.zeros(TR.shape)+np.nan
    TRIX[1:] = (TR[1:]/TR[:-1]-1)*100
    MATRIX = MA(TRIX, m)
    return (TRIX, MATRIX, TR)