# coding=utf-8
"""内置的因子运算(因子模型相关)"""
import datetime as dt
import uuid

import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PointOperation, TimeOperation, SectionOperation, PanelOperation
from QuantStudio.FactorDataBase.FactorTools import _genMultivariateOperatorInfo, _genOperatorData
from QuantStudio.Tools import DataPreprocessingFun
from QuantStudio.Tools.AuxiliaryFun import distributeEqual
from QuantStudio.Tools.StrategyTestFun import testPortfolioStrategy_pd, calcLSYield

# 组合收益率
def _portfolio_return(f, idt, iid, x, args):
    Data = _genOperatorData(f,idt,iid,x,args)
    Mask = (Data[0]==1)
    if args["OperatorArg"]["price"]:
        Price = Data[1]
    else:
        Price = np.nancumprod(Data[1] + 1, axis=0)
    if args["OperatorArg"]["weight_data"]:
        Portfolio = Data[-1]
    else:
        Portfolio = np.ones(Mask.shape)
    if args["OperatorArg"]["rebalance_dt_fun"] is None:
        RebalanceDTs = idt
    else:
        RebalanceDTs = args["OperatorArg"]["rebalance_dt_fun"](idt)
    DescriptorIDs = (iid if f.DescriptorSection[0] is None else f.DescriptorSection[0])
    Mask = pd.DataFrame(Mask, index=idt, columns=DescriptorIDs).loc[RebalanceDTs]
    Price = pd.DataFrame(Price, index=idt, columns=DescriptorIDs)
    Portfolio = pd.DataFrame(Portfolio, index=idt, columns=DescriptorIDs).loc[RebalanceDTs]
    Portfolio = (Portfolio[Mask].T / Portfolio[Mask].sum(axis=1)).T
    NV = testPortfolioStrategy_pd(Portfolio, Price)
    return np.repeat(np.reshape(NV.pct_change().values[f.LookBack[0]:], (-1, 1)), len(iid), axis=1)

def portfolio_return(mask, price=None, return_data=None, look_back=1, rebalance_dt_fun=None, weight_data=None, descriptor_ids=None, **kwargs):
    Factors = [mask]
    if (price is None) and (return_data is None):
        raise __QS_Error__("变量 price 或者 return_data 必须指定其一!")
    elif price is not None:
        Factors.append(price)
    else:
        Factors.append(return_data)
    if weight_data is not None:
        Factors.append(weight_data)
    Descriptors,Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"rebalance_dt_fun":rebalance_dt_fun, "price":(price is not None), "weight_data":(weight_data is not None)}
    return PanelOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_portfolio_return,"参数":Args,"回溯期数":[look_back]*len(Descriptors),"运算时点":"多时点", "输出形式":"全截面", "描述子截面":[descriptor_ids]*len(Descriptors)},**kwargs)

# 多空收益率
def _long_short_return(f, idt, iid, x, args):
    Data = _genOperatorData(f,idt,iid,x,args)
    LReturn, SReturn = Data
    if args["OperatorArg"]["rebalance_dt_fun"] is None:
        RebalanceIdx = None
    else:
        RebalanceDTs = args["OperatorArg"]["rebalance_dt_fun"](idt)
        RebalanceIdx = pd.Series(np.arange(len(idt)), index=idt).loc[RebalanceDTs].tolist()
    LSReturn = calcLSYield(LReturn, SReturn, rebalance_index=RebalanceIdx)
    return LSReturn[f.LookBack[0]:]

def long_short_return(long_return, short_return, look_back=1, rebalance_dt_fun=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(long_return, short_return)
    Args["OperatorArg"] = {"rebalance_dt_fun":rebalance_dt_fun}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_long_short_return,"参数":Args,"回溯期数":[look_back]*len(Descriptors),"运算时点":"多时点", "运算ID":"单ID"},**kwargs)

# 截面分组
def _section_group(f, idt, iid, x, args):
    Data = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd]==1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData==1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd+CatData]
        CatData = np.array(list(zip(*CatData)))
    Rslt = np.zeros(FactorData.shape)+np.nan
    GroupNum = OperatorArg.pop("group_num")
    RemainderPos = OperatorArg.pop("remainder_pos")
    for i in range(FactorData.shape[0]):
        iRank = DataPreprocessingFun.standardizeRank(FactorData[i], mask=(Mask[i] if Mask is not None else None), cat_data=(CatData[i].T if CatData is not None else None), **OperatorArg)
        iTotalNum = np.sum(pd.notnull(iRank))
        iGroupNums = np.cumsum(np.array(distributeEqual(iTotalNum, GroupNum, remainder_pos=RemainderPos))) / iTotalNum
        iGroup = np.searchsorted(iGroupNums, iRank, side="right").astype(np.float)
        iGroup[pd.isnull(iRank)] = np.nan
        Rslt[i] = iGroup
    return Rslt

def section_group(f, group_num=10, mask=None, cat_data=None, ascending=True, perturbation=False, other_handle='填充None', remainder_pos="middle", **kwargs):
    Factors = [f]
    OperatorArg = {}
    if mask is not None:
        Factors.append(mask)
        OperatorArg["mask"] = 1
    else:
        OperatorArg["mask"] = None
    if isinstance(cat_data,Factor):
        Factors.append(cat_data)
        OperatorArg["cat_data"] = 1
    elif isinstance(cat_data,list):
        Factors += cat_data
        OperatorArg["cat_data"] = len(cat_data)
    else:
        OperatorArg["cat_data"] = None
    Descriptors,Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"group_num": group_num, "ascending":ascending,"uniformization": True,"perturbation":perturbation,"offset": 0,"other_handle":other_handle, "remainder_pos": remainder_pos}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_section_group,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)

# 分位数组合收益率: target_group 可选: 0~group_num-1, "L-S"
def quantile_portfolio_return(f, price=None, return_data=None, target_group="L-S", group_num=10, look_back=1, rebalance_dt_fun=None, weight_data=None, descriptor_ids=None, 
                              mask=None, market_mask=None, cat_data=None, ascending=True, perturbation=False, remainder_pos="middle", **kwargs):
    if (target_group!="L-S") and (target_group not in np.arange(group_num)):
        raise __QS_Error__("输入变量 target_group 的可选值: %s" % (str(np.arange(group_num).tolist()+["L-S"]), ))
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    SectionGroup = section_group(f, group_num=group_num, mask=mask, cat_data=cat_data, ascending=ascending, perturbation=perturbation, remainder_pos=remainder_pos)
    if target_group=="L-S":
        LReturn = portfolio_return((SectionGroup==0), price=price, return_data=return_data, look_back=look_back, rebalance_dt_fun=rebalance_dt_fun, weight_data=weight_data, descriptor_ids=descriptor_ids)
        SReturn = portfolio_return((SectionGroup==group_num-1), price=price, return_data=return_data, look_back=look_back, rebalance_dt_fun=rebalance_dt_fun, weight_data=weight_data, descriptor_ids=descriptor_ids)
        PortfolioReturn = long_short_return(LReturn, SReturn, look_back=look_back, rebalance_dt_fun=rebalance_dt_fun, factor_name=FactorName, **kwargs)
    else:
        PortfolioReturn = portfolio_return((SectionGroup==target_group), price=price, return_data=return_data, look_back=look_back, rebalance_dt_fun=rebalance_dt_fun, weight_data=weight_data, descriptor_ids=descriptor_ids, factor_name=FactorName, **kwargs)
        if market_mask is not None:
            MarketReturn = portfolio_return(market_mask, price=price, return_data=return_data, look_back=look_back, rebalance_dt_fun=rebalance_dt_fun, weight_data=weight_data, descriptor_ids=descriptor_ids)
            PortfolioReturn = long_short_return(PortfolioReturn, MarketReturn, look_back=look_back, rebalance_dt_fun=rebalance_dt_fun, factor_name=FactorName, **kwargs)
    return PortfolioReturn