# coding=utf-8
"""内置的因子运算(因子模型相关)"""
import datetime as dt
import uuid
import json

import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PointOperation, TimeOperation, SectionOperation, PanelOperation
from QuantStudio.FactorDataBase.FactorTools import _genMultivariateOperatorInfo
from QuantStudio.Tools import DataPreprocessingFun
from QuantStudio.Tools.AuxiliaryFun import distributeEqual
from QuantStudio.Tools.StrategyTestFun import testPortfolioStrategy_pd

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
        iGroupNums = np.cumsum(np.array(distributeEqual(iTotalNum, GroupNum, remainder_pos=RemainderPos)))
        Rslt[i] = np.searchsorted(iGroupNums, iRank, side="right")
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
    Args["OperatorArg"] = {"group_num": group_num, "ascending":ascending,"uniformization": False,"perturbation":perturbation,"offset": 0,"other_handle":other_handle, "remainder_pos": remainder_pos}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_section_group,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)

# 因子分位数多空组合收益率(TODO)
def _calcQuantilePortfolio(factor_data, return_data, balance_dts=None, mask=None, *cat_data, weight_data=None, ascending=False, n_group=10):
    for i in range(factor_data.shape[0]):
        pass
    
def _aggr_quantile_portfolio_return(f, idt, iid, x, args):
    Data = _genOperatorData(f,idt,iid,x,args)
    FactorData = Data[0]
    ReturnData = Data[1]
    StartInd = 2
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
    return None
    
def aggr_quantile_portfolio_return(f, return_data, look_back=1, rebalance_dt_fun=None, mask=None, cat_data=None, weight_data=None, **kwargs):
    Factors = [f, return_data]
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
    if weight_data is not None:
        Factors.append(weight_data)
        OperatorArg["weight_data"] = 1
    else:
        OperatorArg["weight_data"] = None
    Descriptors,Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"rebalance_dt_fun":rebalance_dt_fun}
    Args["OperatorArg"].update(OperatorArg)
    return PanelOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_quantile_portfolio_return,"参数":Args,"回溯期数":[look_back]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)