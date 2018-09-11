# coding=utf-8
"""因子测试相关函数"""
import numpy as np
import pandas as pd

from QuantStudio.FunLib.MathFun import CartesianProduct
from QuantStudio.FunLib.AuxiliaryFun import partitionList,getClassMask, genAvailableName

# 计算月度平均
def getMonthAvg(data):
    Dates = pd.Series(data.index,index=data.index)
    MonthAvg = pd.DataFrame(0.0,index=[str(i)+"月" for i in range(1,13,1)],columns=data.columns)
    for i in range(1,13,1):
        i = str(i)
        MonthAvg.loc[i+"月",:] = data[Dates.str.slice(4,6)==("0"*(2-len(i))+i)].mean()
    return MonthAvg

# 计算 IC(Information Coefficient)
# factor_data: 因子数据, array
# return_data: 收益率数据, array
# cat_data: 分类数据, array, 如果非空, 类别内调整收益率数据
# weight_data: 计算类别收益率的权重数据, array
# factor_data, return_data, cat_data, weight_data的长度必须一致
# method: 计算相关系数的方法, 可选: spearman, pearson, kendall
def calcIC(factor_data, return_data, cat_data=None, weight_data=None, method='spearman'):
    if weight_data is None:
        weight_data = np.ones(factor_data.shape)
    if cat_data is not None:
        cat_data[pd.isnull(cat_data)] = None
        AllCats = pd.unique(cat_data)
        for iCat in AllCats:
            if pd.isnull(iCat):
                iMask = pd.isnull(cat_data)
            else:
                iMask = (cat_data==iCat)
            iWeight = weight_data[iMask]
            return_data[iMask] -= np.nansum(return_data[iMask]*iWeight)/np.nansum(iWeight)
    Mask = (pd.notnull(factor_data) & pd.notnull(return_data))
    return np.corrcoef(factor_data[Mask],return_data[Mask],method=method)

# 计算分位数组合收益率
# factor_data: 因子数据, DataFrame(index=[日期],columns=[ID]) or Series(index=[ID]), 如果为DataFrame则用行数据
# return_data: 收益率数据, DataFrame(index=[日期],columns=[ID]) or Series(index=[ID]), 如果为DataFrame则用行数据
# mask: 过滤条件, None or DataFrame(index=[日期],columns=[ID]) or Series(index=[ID]), 如果为DataFrame则用行数据
# cat_data: 分类数据, [DataFrame(index=[日期],columns=[ID]) or Series(index=[ID])], 如果为DataFrame则用行数据, 如果非空, 类别内分别分组
# weight_data: 形成投资组合的权重数据, None or DataFrame(index=[日期],columns=[ID]) or Series(index=[ID]), 如果为DataFrame则用行数据
# factor_data, return_data, mask, cat_data[i], weight_data的行列维度和索引必须一致
# ascending: 是否升序排列, 可选: True or False
# n_group: 分组数, int, >0;
# 返回: 如果factor_data等为DataFrame, 返回(DataFrame(收益率,index=[日期],columns=[i]), [[分位数组合]]), 分位数组合: Series(index=[ID])
# 返回: 如果factor_data等为Series, 返回([收益率], [分位数组合]), 分位数组合: Series(index=[ID])
def calcQuantilePortfolio(factor_data, return_data, mask=None, *cat_data, weight_data=None, ascending=False, n_group=10):
    if isinstance(factor_data,pd.Series):
        if mask is not None:
            factor_data = factor_data[mask]
        factor_data = factor_data[pd.notnull(factor_data)]
        factor_data = factor_data.sort_values(ascending=ascending,inplace=False)
        if cat_data!=():
            cat_data = pd.DataFrame(list(cat_data),columns=cat_data[0].index).T
            cat_data = cat_data.loc[factor_data.index]
            cat_data = cat_data.where(pd.notnull(cat_data),np.nan)
            AllCats = CartesianProduct([list(iCatData[iCat].unique()) for iCat in cat_data])
        else:
            AllCats = [None]
            cat_data = factor_data
        PortfolioIDList = [[] for i in range(n_group)]
        for iCat in AllCats:
            iMask = getClassMask(iCat,cat_data)
            iPortfolioIDList = partitionList(list(factor_data[iMask].index),n_group)
            for j,jIDs in enumerate(iPortfolioIDList):
                PortfolioIDList[j] += jIDs
        weight_data = (weight_data[factor_data.index] if weight_data is not None else pd.Series(1.0,index=factor_data.index))
        return_data = return_data[factor_data.index]
        Portfolio = []
        PortfolioReturn = []
        for jIDs in PortfolioIDList:
            jWeight = weight_data.loc[jIDs]
            jPortfolio = jWeight[pd.notnull(jWeight)]/jWeight.sum()
            Portfolio.append(jPortfolio)
            PortfolioReturn.append((jPortfolio*return_data[jPortfolio.index]).sum())
        return (PortfolioReturn,Portfolio)
    PortfolioReturn = pd.DataFrame(0.0,index=factor_data.index,columns=[i for i in range(n_group)])
    Portfolio = []
    for i in range(factor_data.shape[0]):
        iFactorData = factor_data.iloc[i]
        if i<factor_data.shape[0]-1:
            iReturnData = return_data.iloc[i+1]
        else:
            iReturnData = pd.Series(np.nan,index=factor_data.columns)
        iMask = (mask.iloc[i] if mask is not None else None)
        iCatData = [jCatData.iloc[i] for jCatData in cat_data]
        iWeightData = (weight_data.iloc[i] if weight_data is not None else None)
        iPortfolioReturn,iPortfolio = calcQuantilePortfolio(iFactorData,iReturnData,iMask,*iCatData,weight_data=iWeightData,ascending=ascending,n_group=n_group)
        PortfolioReturn.iloc[(i+1)%factor_data.shape[0]] = iPortfolioReturn
        Portfolio.append(iPortfolio)
    return (PortfolioReturn,Portfolio)