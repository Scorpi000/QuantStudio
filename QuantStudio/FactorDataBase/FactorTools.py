# coding=utf-8
"""内置的因子运算"""
import datetime as dt
import uuid
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from QuantStudio import __QS_Error__
from QuantStudio.Tools.api import Panel
from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PointOperation, TimeOperation, SectionOperation
from QuantStudio.Tools import DataPreprocessingFun


def _genMultivariateOperatorInfo(*factors):
    Args = {}
    Descriptors = []
    for i,iFactor in enumerate(factors):
        iInd = str(i+1)
        if isinstance(iFactor, Factor):# 第i个操作子为因子
            if iFactor.Name=="":# 第i个因子为中间运算因子
                Args["Fun"+iInd] = iFactor.Operator
                Args["Arg"+iInd] = iFactor.ModelArgs
                Args["SepInd"+iInd] = Args.get("SepInd"+str(i),0)+len(iFactor.Descriptors)
                Descriptors += iFactor.Descriptors
            else:# 第i个因子为最终因子
                Args["SepInd"+iInd] = Args.get("SepInd"+str(i),0)+1
                Descriptors += [iFactor]
        else:# 第i个操作子为标量
            Args["Data"+iInd] = iFactor
            Args["SepInd"+iInd] = Args.get("SepInd"+str(i),0)
    Args["nData"] = len(factors)
    return (Descriptors, Args)
def _genOperatorData(f, idt, iid, x, args):
    Data = []
    for i in range(args["nData"]):
        iInd = str(i+1)
        iFun = args.get("Fun"+iInd,None)
        if iFun is not None:
            Data.append(iFun(f, idt, iid, x[args.get("SepInd"+str(i),0):args.get("SepInd"+iInd,args["nData"])], args["Arg"+iInd]))
        else:
            if "Data"+iInd in args:
                Data.append(args["Data"+iInd])
            else:
                Data.append(x[args.get("SepInd"+str(i),0)])
    return Data
# ----------------------单点运算--------------------------------
def _astype(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return Data.astype(dtype=args["OperatorArg"]["dtype"])
def astype(f, dtype, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"dtype":dtype}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_astype, "参数":Args, "运算时点":"多时点", "运算ID":"多ID"}, **kwargs)
def _log(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    Data[Data<=0] = np.nan
    return np.log(Data.astype(float))/np.log(args["OperatorArg"]["base"])
def log(f, base=np.e, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"base":base}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_log, "参数":Args, "运算时点":"多时点", "运算ID":"多ID"}, **kwargs)
def _isnull(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return pd.isnull(Data)
def isnull(f, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_isnull,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _notnull(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return pd.notnull(Data)
def notnull(f, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_notnull, "参数":Args, "运算时点":"多时点", "运算ID":"多ID"}, **kwargs)
def _sign(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return np.sign(Data.astype(float))
def sign(f, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_sign,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _ceil(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return np.ceil(Data.astype(float))
def ceil(f, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_ceil,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _floor(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return np.floor(Data.astype(float))
def floor(f, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_floor,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _fix(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return np.fix(Data.astype(float))
def fix(f, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_fix,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _isin(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return np.isin(Data, args["OperatorArg"]["test_elements"])
def isin(f, test_elements, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"test_elements":test_elements}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_isin,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _applymap(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    Func = args["OperatorArg"]["func"]
    return Data.applymap(Func).values
def applymap(f, func=id, data_type="double", **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"func":func}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_applymap,"参数":Args,"运算时点":"多时点","运算ID":"多ID","数据类型":data_type}, **kwargs)
def _map_value(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    Mapping = pd.Series(args["OperatorArg"]["mapping"])
    TargetShape = Data.shape
    Data = Data.flatten(order="C")
    Rslt = np.full(shape=Data.shape, fill_value=None, dtype=Mapping.dtype)
    Mask = pd.notnull(Data)
    Rslt[Mask] = Mapping.reindex(index=Data[Mask]).values
    return Rslt.reshape(TargetShape)
def map_value(f, mapping, data_type="double", **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"mapping":mapping}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_map_value,"参数":Args,"运算时点":"多时点","运算ID":"多ID", "数据类型":data_type}, **kwargs)
def _fetch(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    if isinstance(args["OperatorArg"]["pos"], str):
        return Data.astype(args["OperatorArg"]["dtype"])[args["OperatorArg"]["pos"]]
    SampleData = Data[0,0]
    DataType = np.dtype([(str(i),(np.float if isinstance(SampleData[i], float) else "O")) for i in range(len(SampleData))])
    return Data.astype(DataType)[str(args["OperatorArg"]["pos"])]
def fetch(f,pos=0,dtype="double", **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"pos":pos,"dtype":dtype}
    if isinstance(pos,str):
        Args["OperatorArg"]['dtype'] = f.TempData['dtype']
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_fetch,"参数":Args,"运算时点":"多时点","运算ID":"多ID","数据类型":dtype}, **kwargs)
def _where(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    return np.where(Data[1],Data[0],Data[2])
def where(f,mask,other,data_type="double",**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f,mask,other)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_where, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "数据类型":data_type}, **kwargs)
def _replace(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    ValueMap = args["OperatorArg"]["value_map"]
    if f.DataType=="double":
        Rslt = np.full_like(Data, fill_value=np.nan, dtype="float")
    else:
        Rslt = np.full_like(Data, fill_value=None, dtype="O")
    for iKey, iVal in ValueMap.items():
        if pd.isnull(iKey):
            Rslt[pd.isnull(Data)] = iVal
        else:
            Rslt[Data==iKey] = iVal
    return Rslt
def replace(f, value_map, data_type="double",**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"value_map":value_map}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_replace, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "数据类型":data_type}, **kwargs)
def _clip(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    return np.clip(Data[0].astype(float),Data[1],Data[2])
def clip(f,a_min,a_max,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f,a_min,a_max)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_clip,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nansum(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    Data = np.array(Data)
    Rslt = np.nansum(Data,axis=0)
    Mask = (np.sum(pd.notnull(Data),axis=0)==0)
    Rslt[Mask] = args["OperatorArg"]["all_nan"]
    return Rslt
def nansum(*factors,all_nan=0,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    Args["OperatorArg"] = {"all_nan":all_nan}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nansum,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanprod(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    Data = np.array(Data)
    Rslt = np.nanprod(Data,axis=0)
    Mask = (np.sum(pd.notnull(Data),axis=0)==0)
    Rslt[Mask] = args["OperatorArg"]["all_nan"]
    return Rslt
def nanprod(*factors,all_nan=1,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    Args["OperatorArg"] = {"all_nan":all_nan}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanprod,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanmax(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    return np.nanmax(np.array(Data),axis=0)
def nanmax(*factors,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanmax,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanmin(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    return np.nanmin(np.array(Data),axis=0)
def nanmin(*factors,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanmin,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanargmax(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    Data = np.array(Data)
    Mask = pd.isnull(Data)
    Data[Mask] = -np.inf
    Rslt = np.nanargmax(Data,axis=0)
    Mask = (np.sum(Mask,axis=0)==Data.shape[0])
    Rslt[Mask] = np.nan   
    return Rslt
def nanargmax(*factors,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanargmax,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanargmin(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    Data = np.array(Data)
    Mask = pd.isnull(Data)
    Data[Mask] = np.inf
    Rslt = np.nanargmin(Data,axis=0)
    Mask = (np.sum(Mask,axis=0)==Data.shape[0])
    Rslt[Mask] = np.nan   
    return Rslt
def nanargmin(*factors,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanargmin,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanmean(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    Weights = args["OperatorArg"]["weights"]
    if Weights is None:
        if args["OperatorArg"]["ignore_nan_weight"]:
            return np.nanmean(np.array(Data),axis=0)
        Weights = [1]*len(Data)
    Rslt = np.zeros(Data[0].shape)
    WeightArray = np.zeros(Data[0].shape)
    for i,iData in enumerate(Data):
        iMask = pd.notnull(iData)
        WeightArray += iMask*Weights[i]
        iData[~iMask] = 0.0
        Rslt += iData*Weights[i]
    if args["OperatorArg"]["ignore_nan_weight"]:
        WeightArray[WeightArray==0.0] = np.nan
        return Rslt/WeightArray
    else:
        Rslt[WeightArray==0.0] = np.nan
        return Rslt/len(Data)
def nanmean(*factors,weights=None,ignore_nan_weight=True,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    Args["OperatorArg"] = {"weights":weights,"ignore_nan_weight":ignore_nan_weight}
    return PointOperation(kwargs.get("factor_name",str(uuid.uuid1())),Descriptors,{"算子":_nanmean,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanstd(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    return np.nanstd(np.array(Data),axis=0,ddof=args["OperatorArg"]["ddof"])
def nanstd(*factors,ddof=1,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    Args["OperatorArg"] = {"ddof":ddof}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanstd,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanvar(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    return np.nanvar(np.array(Data),axis=0,ddof=args["OperatorArg"]["ddof"])
def nanvar(*factors, ddof=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    Args["OperatorArg"] = {"ddof":ddof}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanvar,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanmedian(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    return np.nanmedian(np.array(Data),axis=0)
def nanmedian(*factors, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanmedian,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nanquantile(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    return np.nanpercentile(np.array(Data),args["OperatorArg"]["quantile"]*100,axis=0)
def nanquantile(*factors, quantile=0.5, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    Args["OperatorArg"] = {"quantile":quantile}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nanquantile,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nancount(f,idt,iid,x,args):
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)]
    return np.nansum(pd.isnull(np.array(Data)),axis=0)
def nancount(*factors, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nancount,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _regress_change_rate(f,idt,iid,x,args):
    Y = np.array([(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for iData in _genOperatorData(f,idt,iid,x,args)])
    X = np.arange(Y.shape[0]).astype("float").reshape((Y.shape[0],1,1)).repeat(Y.shape[1],axis=1).repeat(Y.shape[2],axis=2)
    Denominator = np.abs(np.nanmean(Y,axis=0))
    X[pd.isnull(Y)] = np.nan
    X = X - np.nanmean(X, axis=0)
    Y = Y - np.nanmean(Y, axis=0)
    Numerator = np.nansum(X*Y,axis=0)/np.nansum(X**2,axis=0)
    Rslt = Numerator/Denominator
    Mask = (Denominator==0)
    Rslt[Mask] = np.sign(Numerator)[Mask]
    Rslt[np.isinf(Rslt)] = np.nan
    return Rslt
def regress_change_rate(*factors, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_regress_change_rate,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _tolist(f,idt,iid,x,args):
    Data = {i: (iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for i, iData in enumerate(_genOperatorData(f,idt,iid,x,args))}
    return Panel(Data).sort_index(axis=0).to_frame(filter_observations=False).apply(lambda s: s.tolist(), axis=1).unstack().values
def tolist(*factors,**kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*factors)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_tolist,"参数":Args,"运算时点":"多时点","运算ID":"多ID","数据类型":"object"}, **kwargs)
def _to_json(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return pd.DataFrame(Data).applymap(lambda v: json.dumps(v, ensure_ascii=False) if pd.notnull(v) else None).values
def to_json(f, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_to_json, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "数据类型":"string"}, **kwargs)
def _report_period_delta(last, prev):
    if pd.isnull(last) or pd.isnull(prev): return np.nan
    last, prev = pd.to_datetime(last), pd.to_datetime(prev)
    return ((last.year - prev.year) * 12 + last.month - prev.month) / 3.0
def _single_quarter(f,idt,iid,x,args):
    Last, LastPeriod, Prev, PrevPeriod = _genOperatorData(f,idt,iid,x,args)
    f = np.vectorize(lambda x: (pd.to_datetime(x).strftime("%m%d")=="0331") if pd.notnull(x) else False)
    Rslt = Last - Prev
    Mask = f(LastPeriod)
    Rslt[Mask] = Last[Mask]
    f = np.vectorize(_report_period_delta)
    Rslt[(f(LastPeriod, PrevPeriod)!=1) & (~Mask)] = np.nan
    return Rslt
def single_quarter(last, last_period, prev, pre_period, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(last, last_period, prev, pre_period)
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_single_quarter,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _strftime(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    DTFormat = args["OperatorArg"]["dt_format"]
    return pd.DataFrame(Data).applymap(lambda x: x.strftime(DTFormat) if pd.notnull(x) else None).values
def strftime(f, dt_format="%Y%m%d", **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"dt_format":dt_format}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_strftime, "参数":Args, "数据类型":"string", "运算时点":"多时点", "运算ID":"多ID"}, **kwargs)
def _strptime(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    DTFormat = args["OperatorArg"]["dt_format"]
    if args["OperatorArg"]["is_datetime"]:
        return pd.DataFrame(Data).applymap(lambda x: dt.datetime.strptime(x, DTFormat) if pd.notnull(x) else None).values
    else:
        return pd.DataFrame(Data).applymap(lambda x: dt.datetime.strptime(x, DTFormat).date() if pd.notnull(x) else None).values
def strptime(f, dt_format="%Y%m%d", is_datetime=True, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"dt_format":dt_format, "is_datetime":is_datetime}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_strptime, "参数":Args, "数据类型":"object", "运算时点":"多时点", "运算ID":"多ID"}, **kwargs)
def _fromtimestamp(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return pd.DataFrame(Data).applymap(lambda x: dt.datetime.fromtimestamp(x / args["OperatorArg"]["unit"]) if pd.notnull(x) else None).values
def fromtimestamp(f, unit=1e9, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"unit":unit}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())), Descriptors, {"算子":_fromtimestamp, "参数":Args, "数据类型":"object", "运算时点":"多时点", "运算ID":"多ID"}, **kwargs)
# ----------------------时间序列运算--------------------------------
def _rolling_mean(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    if "weights" not in args["OperatorArg"]:
        return Data.rolling(**args["OperatorArg"]).mean().values[args["OperatorArg"]["window"]-1:]
    else:
        weights = np.array(args["OperatorArg"]["weights"])
        TotalWeight = np.nansum(weights)
        return Data.rolling(**args["OperatorArg"]).apply(lambda x: np.nansum(x * weights) / np.nansum(pd.notnull(x) * weights),raw=True).values[args["OperatorArg"]["window"]-1:]
def rolling_mean(f, window, min_periods=1, win_type=None, weights=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    if weights is None:
        return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_mean,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
    else:
        Args["OperatorArg"]["window"] = len(weights)
        Args["OperatorArg"]["weights"] = weights
        return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_mean,"参数":Args,"回溯期数":[len(weights)-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_sum(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).sum().values[args["OperatorArg"]["window"]-1:]
def rolling_sum(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_sum,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_prod(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    Rslt = np.nanprod(Data, axis=0)
    Rslt[np.sum(pd.notnull(Data), axis=0)<args["OperatorArg"]["min_periods"]] = np.nan
    return Rslt
def rolling_prod(f, window, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_prod,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"单时点","运算ID":"多ID"}, **kwargs)
def _rolling_std(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return Data.rolling(**OperatorArg).apply(lambda x:np.nanstd(x, **SubOperatorArg), raw=True).values[args["OperatorArg"]["window"]-1:]
def rolling_std(f, window, min_periods=1, win_type=None, ddof=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type,"SubOperatorArg":{"ddof":ddof}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_std,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_max(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).max().values[args["OperatorArg"]["window"]-1:]
def rolling_max(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_max,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_min(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).min().values[args["OperatorArg"]["window"]-1:]
def rolling_min(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_min,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_argmax(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).apply(np.nanargmax).values[args["OperatorArg"]["window"]-1:]
def rolling_argmax(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_argmax,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_argmin(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).apply(np.nanargmin).values[args["OperatorArg"]["window"]-1:]
def rolling_argmin(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_argmin,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_median(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).median().values[args["OperatorArg"]["window"]-1:]
def rolling_median(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_median,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_skew(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).skew().values[args["OperatorArg"]["window"]-1:]
def rolling_skew(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_skew,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_kurt(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).kurt().values[args["OperatorArg"]["window"]-1:]
def rolling_kurt(f, window, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_kurt,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_var(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return Data.rolling(**OperatorArg).apply(lambda x:np.nanvar(x, **SubOperatorArg), raw=True).values[args["OperatorArg"]["window"]-1:]
def rolling_var(f, window, min_periods=1, win_type=None, ddof=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type,"SubOperatorArg":{"ddof":ddof}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_var,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_quantile(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return Data.rolling(**OperatorArg).quantile(**SubOperatorArg).values[args["OperatorArg"]["window"]-1:]
def rolling_quantile(f, window, quantile=0.5, min_periods=1, win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type,"SubOperatorArg":{"quantile":quantile}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_quantile,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_count(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.rolling(**args["OperatorArg"]).count().values[args["OperatorArg"]["window"]-1:]
def rolling_count(f, window, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_count,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_change_rate(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    Numerator = Data[args["OperatorArg"]["window"]-1:]
    Denominator = Data[:-args["OperatorArg"]["window"]+1]
    Rslt = (Numerator-Denominator)/np.abs(Denominator)
    Mask = (Denominator==0)
    Rslt[Mask] = np.nan
    Rslt[Mask & (Numerator>0)] = 1.0
    Rslt[Mask & (Numerator<0)] = -1.0
    Rslt[Mask & (Numerator==0)] = 0.0
    return Rslt
def rolling_change_rate(f, window, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"window":window}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_change_rate,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_rank(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    if args["OperatorArg"]["ascending"]:
        Rslt = Data.rolling(**args["OperatorArg"]["RollingArg"]).apply(lambda s: np.sort(s).searchsorted(s[-1]), raw=True).values[args["OperatorArg"]["RollingArg"]["window"]-1:].astype(float)
    else:
        Rslt = Data.rolling(**args["OperatorArg"]["RollingArg"]).apply(lambda s: np.sort(-s).searchsorted(-s[-1]), raw=True).values[args["OperatorArg"]["RollingArg"]["window"]-1:].astype(float)
    Rslt[pd.isnull(Data[args["OperatorArg"]["RollingArg"]["window"]-1:]).values] = np.nan
    return Rslt
def rolling_rank(f, window, min_periods=1, win_type=None, ascending=True, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"ascending": ascending}
    Args["OperatorArg"]["RollingArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_rank,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_mean(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).mean().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_mean(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_mean,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_sum(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).sum().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_sum(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_sum,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_std(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return Data.expanding(**OperatorArg).std(**SubOperatorArg).values[args["OperatorArg"]["min_periods"]-1:]
def expanding_std(f, min_periods=1, ddof=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods,"SubOperatorArg":{"ddof":ddof}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_std,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_max(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).max().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_max(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_max,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_min(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).min().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_min(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_min,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_median(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).median().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_median(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_median,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_skew(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).skew().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_skew(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_skew,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_kurt(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).kurt().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_kurt(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_kurt,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_var(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return Data.expanding(**OperatorArg).var(**SubOperatorArg).values[args["OperatorArg"]["min_periods"]-1:]
def expanding_var(f, min_periods=1, ddof=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods,"SubOperatorArg":{"ddof":ddof}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_var,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_quantile(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg",{})
    return Data.expanding(**OperatorArg).quantile(**SubOperatorArg).values[args["OperatorArg"]["min_periods"]-1:]
def expanding_quantile(f, quantile=0.5, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods,"SubOperatorArg":{"quantile":quantile}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_quantile,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_count(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.expanding(**args["OperatorArg"]).count().values[args["OperatorArg"]["min_periods"]-1:]
def expanding_count(f, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_count,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _ewm_mean(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    return Data.ewm(**args["OperatorArg"]).mean().values[args["OperatorArg"]["min_periods"]-1:]
def ewm_mean(f, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"com":com,"span":span,"halflife":halflife,"alpha":alpha,
                           "min_periods":min_periods,"adjust":adjust,"ignore_na":ignore_na}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_ewm_mean,"参数":Args,"回溯期数":[min_periods]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _ewm_std(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg",{})
    return Data.ewm(**OperatorArg).std(**SubOperatorArg).values[args["OperatorArg"]["min_periods"]-1:]
def ewm_std(f, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, bias=False, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"com":com,"span":span,"halflife":halflife,"alpha":alpha,"min_periods":min_periods,
                           "adjust":adjust,"ignore_na":ignore_na,"SubOperatorArg":{"bias":bias}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_ewm_std,"参数":Args,"回溯期数":[min_periods]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _ewm_var(f,idt,iid,x,args):
    Data = pd.DataFrame(_genOperatorData(f,idt,iid,x,args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg",{})
    return Data.ewm(**OperatorArg).var(**SubOperatorArg).values[args["OperatorArg"]["min_periods"]-1:]
def ewm_var(f, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, bias=False, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"com":com,"span":span,"halflife":halflife,"alpha":alpha,"min_periods":min_periods,
                           "adjust":adjust,"ignore_na":ignore_na,"SubOperatorArg":{"bias":bias}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_ewm_var,"参数":Args,"回溯期数":[min_periods]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_cov(f,idt,iid,x,args):
    Data1,Data2 = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg",{})
    return pd.DataFrame(Data1).rolling(**OperatorArg).cov(pd.DataFrame(Data2),**SubOperatorArg).values[args["OperatorArg"]["window"]-1:]
def rolling_cov(f1, f2, window, min_periods=1, win_type=None, ddof=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f1,f2)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type,"SubOperatorArg":{"ddof":ddof}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_cov,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _rolling_corr(f,idt,iid,x,args):
    Data1,Data2 = _genOperatorData(f,idt,iid,x,args)
    Method = args["OperatorArg"]["method"]
    if Method=="pearson":
        return pd.DataFrame(Data1).rolling(window=args["OperatorArg"]["window"], min_periods=args["OperatorArg"]["min_periods"], win_type=args["OperatorArg"]["win_type"]).corr(pd.DataFrame(Data2)).values[args["OperatorArg"]["window"]-1:]
    Mask = np.sum(pd.notnull(Data1) & pd.notnull(Data2), axis=0)
    Rslt = pd.DataFrame(Data1).corrwith(pd.DataFrame(Data2), axis=0, drop=False, method=Method).values
    Rslt[Mask<args["OperatorArg"]["min_periods"]] = np.nan
    return Rslt
def rolling_corr(f1, f2, window, min_periods=1, method="pearson", win_type=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f1,f2)
    Args["OperatorArg"] = {"window":window,"min_periods":min_periods,"win_type":win_type,"method":method}
    if method=="pearson":
        return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_corr,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
    else:
        return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_corr,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"单时点","运算ID":"多ID"}, **kwargs)
def _rolling_regress(f,idt,iid,x,args):
    X = _genOperatorData(f,idt,iid,x,args)
    Y = X[0]
    if args["OperatorArg"]['constant']:
        X = np.array([np.ones(Y.shape)]+X[1:])
    else:
        X = np.array(X[1:])
    Window = args["OperatorArg"]['window']
    Weight = (0.5**(1/args["OperatorArg"]['half_life']))**np.arange(Window)
    Weight = Weight[::-1]/np.sum(Weight)
    Rslt = np.empty((Y.shape[0]-Window+1,Y.shape[1]),dtype="O")
    for i in range(Rslt.shape[0]):
        for j in range(Rslt.shape[1]):
            iY = Y[i:i+Window,j]
            iX = X[:,i:i+Window,j].T
            try:
                iRslt = sm.WLS(iY,iX,missing='drop').fit()
                Rslt[i,j] = tuple(iRslt.params)+tuple(iRslt.tvalues)+(iRslt.fvalue,iRslt.rsquared,iRslt.rsquared_adj)
            except:
                Rslt[i,j] = (np.nan,)*int(X.shape[0]*2+3)
    return Rslt
def rolling_regress(Y, *X, window=20, constant=True, half_life=np.inf, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(*((Y,)+X))
    Args["OperatorArg"] = {"window":window,"constant":constant,"half_life":half_life}
    nX = len(X)
    f = TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_regress,"参数":Args,"回溯期数":[window-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID","数据类型":"object"}, **kwargs)
    if constant:
        DataType = [('alpha',np.float)]+[('beta'+str(i),np.float) for i in range(nX)]
        DataType += [('t_alpha',np.float)]+[('t_beta'+str(i),np.float) for i in range(nX)]
    else:
        DataType = [('beta'+str(i),np.float) for i in range(nX)]
        DataType += [('t_beta'+str(i),np.float) for i in range(nX)]
    DataType += [('fvalue',np.float),('rsquared',np.float),('rsquared_adj',np.float)]
    f.TempData["dtype"] = DataType
    return f
def _rolling_regress_change(f,idt,iid,x,args):
    Y = _genOperatorData(f,idt,iid,x,args)[0]
    X = np.arange(Y.shape[0]).astype("float").reshape((Y.shape[0], 1)).repeat(Y.shape[1], axis=1)
    Mask = pd.isnull(Y)
    X[Mask] = np.nan
    X = X - np.nanmean(X, axis=0)
    Y = Y - np.nanmean(Y, axis=0)
    Rslt = np.nansum(X * Y, axis=0) / np.nansum(X ** 2, axis=0)
    Rslt[Y.shape[0] - np.sum(Mask)<args["OperatorArg"]["min_periods"]] = np.nan
    Rslt[np.isinf(Rslt)] = np.nan
    return Rslt
def rolling_regress_change(f, window=20, min_periods=2, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_rolling_regress_change,"参数":Args,"运算时点":"单时点","运算ID":"多ID"}, **kwargs)
def _expanding_cov(f,idt,iid,x,args):
    Data1,Data2 = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg",{})
    return pd.DataFrame(Data1).expanding(**OperatorArg).cov(pd.DataFrame(Data2),**SubOperatorArg).values[args["OperatorArg"]["min_periods"]-1:]
def expanding_cov(f1, f2, min_periods=1, ddof=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f1,f2)
    Args["OperatorArg"] = {"min_periods":min_periods,"SubOperatorArg":{"ddof":ddof}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_cov,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _expanding_corr(f,idt,iid,x,args):
    Data1,Data2 = _genOperatorData(f,idt,iid,x,args)
    return pd.DataFrame(Data1).expanding(**args["OperatorArg"]).corr(pd.DataFrame(Data2)).values[args["OperatorArg"]["min_periods"]-1:]
def expanding_corr(f1, f2, min_periods=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f1,f2)
    Args["OperatorArg"] = {"min_periods":min_periods}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_expanding_corr,"参数":Args,"回溯期数":[min_periods-1]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _ewm_cov(f,idt,iid,x,args):
    Data1,Data2 = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg",{})
    return pd.DataFrame(Data1).ewm(**OperatorArg).cov(pd.DataFrame(Data2),**SubOperatorArg).values[args["OperatorArg"]["min_periods"]-1:]
def ewm_cov(f1, f2, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, bias=False, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f1,f2)
    Args["OperatorArg"] = {"com":com,"span":span,"halflife":halflife,"min_periods":min_periods,
                           "adjust":adjust,"ignore_na":ignore_na,"SubOperatorArg":{"bias":bias}}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_ewm_cov,"参数":Args,"回溯期数":[min_periods]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _ewm_corr(f,idt,iid,x,args):
    Data1,Data2 = _genOperatorData(f,idt,iid,x,args)
    return pd.DataFrame(Data1).ewm(**args["OperatorArg"]).corr(pd.DataFrame(Data2)).values[args["OperatorArg"]["min_periods"]-1:]
def ewm_corr(f1, f2, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f1,f2)
    Args["OperatorArg"] = {"com":com,"span":span,"halflife":halflife,"min_periods":min_periods,
                           "adjust":adjust,"ignore_na":ignore_na}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_ewm_corr,"参数":Args,"回溯期数":[min_periods]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _lag(f,idt,iid,x,args):
    if args["OperatorArg"]['dt_change_fun'] is None: return x[0][args["OperatorArg"]['window']-args["OperatorArg"]['lag_period']:x[0].shape[0]-args["OperatorArg"]['lag_period']]
    TargetDTs = args["OperatorArg"]['dt_change_fun'](idt)
    Data = pd.DataFrame(x[0], index=idt)
    TargetData = Data.reindex(index=TargetDTs).values
    TargetData[args["OperatorArg"]['lag_period']:] = TargetData[:-args["OperatorArg"]['lag_period']]
    if f.DataType!="double":
        Data = pd.DataFrame(np.empty(Data.shape,dtype="O"),index=Data.index,columns=iid)
    else:
        Data = pd.DataFrame(index=Data.index,columns=iid,dtype="float")
    Data.loc[TargetDTs] = TargetData
    return Data.fillna(method='pad').values[args["OperatorArg"]['window']:]
def lag(f, lag_period=1, window=1, dt_change_fun=None, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"lag_period":lag_period,"window":window,"dt_change_fun":dt_change_fun}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_lag,"参数":Args,"回溯期数":[window]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _diff(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    return np.diff(Data, n=args["OperatorArg"]['n'], axis=0)
def diff(f, n=1, **kwargs):
    Descriptors,Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"n":n}
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_diff,"参数":Args,"回溯期数":[n]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _fillna(f,idt,iid,x,args):
    Data, Val = _genOperatorData(f,idt,iid,x,args)
    if not args["OperatorArg"]["fill_value"]:
        LookBack = args["OperatorArg"]["lookback"]
        return pd.DataFrame(Data).fillna(method="pad", limit=LookBack).values[LookBack:]
    else:
        return np.where(pd.notnull(Data),Data,Val)
def fillna(f, value=None, lookback=1, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f, value)
    Args["OperatorArg"] = {"lookback":lookback, "fill_value": (value is not None)}
    if value is None:
        return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_fillna,"参数":Args,"回溯期数":[lookback]*len(Descriptors),"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
    else:
        return PointOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_fillna,"参数":Args,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
def _nav(f, idt, iid, x, args):
    Price = x[0]
    Return, = _genOperatorData(f, idt, iid, x[1:], args)
    if Price.shape[0]<=Return.shape[0]:
        NAV = np.nancumprod(Return + 1, axis=0)
    else:
        NAV = Price[-Return.shape[0]-1, :] * np.nancumprod(Return + 1, axis=0)
    NAV[pd.isnull(Return)] = np.nan
    return NAV
def nav(ret, init=None, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(ret)
    return TimeOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_nav,"参数":Args,"回溯期数":[0]*len(Descriptors),"自身回溯期数":1,"自身回溯模式":"扩张窗口","自身初始值":init,"运算时点":"多时点","运算ID":"多ID"}, **kwargs)
# ----------------------单截面运算--------------------------------
def _standardizeZScore(f,idt,iid,x,args):
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
        StartInd += 1
    elif CatData is not None:
        CatData = Data[StartInd:StartInd+CatData]
        StartInd += len(CatData)
        CatData = np.array(list(zip(*CatData)))
    AvgWeight = OperatorArg.pop("avg_weight")
    if AvgWeight is not None:
        AvgWeight = Data[StartInd]
        StartInd += 1
    DispersionWeight = OperatorArg.pop("dispersion_weight")
    if DispersionWeight is not None:
        DispersionWeight = Data[StartInd]
    Rslt = np.zeros(FactorData.shape)+np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.standardizeZScore(FactorData[i],mask=(Mask[i] if Mask is not None else None),
                                                         cat_data=(CatData[i].T if CatData is not None else None),
                                                         avg_weight=(AvgWeight[i] if AvgWeight is not None else None),
                                                         dispersion_weight=(DispersionWeight[i] if DispersionWeight is not None else None),
                                                         **OperatorArg)
    return Rslt
def standardizeZScore(f, mask=None, cat_data=None, avg_statistics="平均值", dispersion_statistics="标准差", avg_weight=None, dispersion_weight=None, other_handle='填充None', **kwargs):
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
    if avg_weight is not None:
        Factors.append(avg_weight)
        OperatorArg["avg_weight"] = 1
    else:
        OperatorArg["avg_weight"] = None
    if dispersion_weight is not None:
        Factors.append(dispersion_weight)
        OperatorArg["dispersion_weight"] = 1
    else:
        OperatorArg["dispersion_weight"] = None
    Descriptors,Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"avg_statistics":avg_statistics,"dispersion_statistics":dispersion_statistics,"other_handle":other_handle}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_standardizeZScore,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)
def _standardizeRank(f,idt,iid,x,args):
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
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.standardizeRank(FactorData[i],mask=(Mask[i] if Mask is not None else None),
                                                       cat_data=(CatData[i].T if CatData is not None else None),
                                                       **OperatorArg)
    return Rslt
def standardizeRank(f, mask=None, cat_data=None, ascending=True, uniformization=True, perturbation=False, offset=0.5, other_handle='填充None', **kwargs):
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
    Args["OperatorArg"] = {"ascending":ascending,"uniformization":uniformization,"perturbation":perturbation,"offset":offset,"other_handle":other_handle}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_standardizeRank,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)
def _standardizeQuantile(f,idt,iid,x,args):
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
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.standardizeQuantile(FactorData[i],mask=(Mask[i] if Mask is not None else None),
                                                           cat_data=(CatData[i].T if CatData is not None else None),
                                                           **OperatorArg)
    return Rslt
def standardizeQuantile(f, mask=None, cat_data=None, ascending=True, perturbation=False, other_handle='填充None', **kwargs):
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
    Args["OperatorArg"] = {"ascending":ascending,"perturbation":perturbation,"other_handle":other_handle}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_standardizeQuantile,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)
def _fillNaNByVal(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[1]==1)
    Rslt = np.zeros(FactorData.shape)+np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.fillNaNByVal(FactorData[i],mask=(Mask[i] if Mask is not None else None),**OperatorArg)
    return Rslt
def fillNaNByVal(f, mask=None, value=0.0, **kwargs):
    Factors = [f]
    OperatorArg = {}
    if mask is not None:
        Factors.append(mask)
        OperatorArg["mask"] = 1
    else:
        OperatorArg["mask"] = None
    Descriptors,Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"value":value}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_fillNaNByVal,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)
def _fillNaNByFun(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[1]==1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData==1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd+CatData]
        CatData = np.array(list(zip(*CatData)))
    ValFun = OperatorArg.pop("val_fun")
    if ValFun=="平均值":
        ValFun = (lambda x,n:np.zeros(n)+np.nanmean(x))
    elif ValFun=="中位数":
        ValFun = (lambda x,n:np.zeros(n)+np.nanmedian(x))
    elif ValFun=="最大值":
        ValFun = (lambda x,n:np.zeros(n)+np.nanmax(x))
    elif ValFun=="最小值":
        ValFun = (lambda x,n:np.zeros(n)+np.nanmin(x))
    elif ValFun=="高斯随机数":
        ValFun = (lambda x,n:np.random.randn(n)*np.nanstd(x)+np.nanmean(x))
    elif ValFun=="均匀随机数":
        ValFun = (lambda x,n:np.random.rand(n)*(np.nanmax(x)-np.nanmin(x))+np.nanmin(x))
    Rslt = np.zeros(FactorData.shape)+np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.fillNaNByFun(FactorData[i],mask=(Mask[i] if Mask is not None else None),
                                                    cat_data=(CatData[i].T if CatData is not None else None),
                                                    val_fun=ValFun,**OperatorArg)
    return Rslt
def fillNaNByFun(f, mask=None, cat_data=None, val_fun="平均值", **kwargs):
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
    Args["OperatorArg"] = {"val_fun":val_fun}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_fillNaNByFun,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)
def _fillNaNByRegress(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    X = OperatorArg.pop("X")
    if X==1:
        X = Data[StartInd]
        StartInd += 1
    elif X is not None:
        X = Data[StartInd:StartInd+X]
        StartInd += len(X)
        X = np.array(list(zip(*X)))
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd]==1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData==1:
        CatData = Data[StartInd]
        StartInd += 1
    elif CatData is not None:
        CatData = Data[StartInd:StartInd+CatData]
        StartInd += len(CatData)
        CatData = np.array(list(zip(*CatData)))
    DummyData = OperatorArg.pop("dummy_data")
    if DummyData==1:
        DummyData = Data[StartInd]
    elif DummyData is not None:
        DummyData = Data[StartInd:StartInd+DummyData]
        StartInd += len(DummyData)
        DummyData = np.array(list(zip(*DummyData)))
    Rslt = np.zeros(FactorData.shape)+np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.fillNaNByRegress(FactorData[i],X=(X[i].T if X is not None else None),
                                                        mask=(Mask[i] if Mask is not None else None),
                                                        cat_data=(CatData[i].T if CatData is not None else None),
                                                        dummy_data=(DummyData[i].T if DummyData is not None else None),
                                                        **OperatorArg)
    return Rslt
def fillNaNByRegress(Y, X, mask=None, cat_data=None, constant=False, dummy_data=None, drop_dummy_na=False, **kwargs):
    Factors = [Y]
    OperatorArg = {}
    if isinstance(X,Factor):
        Factors.append(X)
        OperatorArg["X"] = 1
    elif isinstance(X,list):
        Factors += X
        OperatorArg["X"] = len(X)
    else:
        OperatorArg["X"] = None
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
    if isinstance(dummy_data,Factor):
        Factors.append(dummy_data)
        OperatorArg["dummy_data"] = 1
    elif isinstance(dummy_data,list):
        Factors += dummy_data
        OperatorArg["dummy_data"] = len(dummy_data)
    else:
        OperatorArg["dummy_data"] = None
    Descriptors,Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"drop_dummy_na":drop_dummy_na,"constant":constant}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_fillNaNByRegress,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)
def _winsorize(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[1]==1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData==1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd+CatData]
        CatData = np.array(list(zip(*CatData)))
    Rslt = np.zeros(FactorData.shape)+np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.winsorize(FactorData[i],mask=(Mask[i] if Mask is not None else None),
                                                 cat_data=(CatData[i].T if CatData is not None else None),
                                                 **OperatorArg)
    return Rslt
def winsorize(f, mask=None, cat_data=None, method='截断', avg_statistics="平均值", dispersion_statistics="标准差", std_multiplier=3, std_tmultiplier=3.5, other_handle='填充None', **kwargs):
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
    Args["OperatorArg"] = {"method":method,"avg_statistics":avg_statistics,"dispersion_statistics":dispersion_statistics,"std_multiplier":std_multiplier,"std_tmultiplier":std_tmultiplier,"other_handle":other_handle}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_winsorize,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)
def _orthogonalize(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    X = OperatorArg.pop("X")
    if X==1:
        X = Data[StartInd]
        StartInd += 1
    elif X is not None:
        X = Data[StartInd:StartInd+X]
        StartInd += len(X)
        X = np.array(list(zip(*X)))
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd]==1)
        StartInd += 1
    DummyData = OperatorArg.pop("dummy_data")
    if DummyData==1:
        DummyData = Data[StartInd]
    elif DummyData is not None:
        DummyData = Data[StartInd:StartInd+DummyData]
        DummyData = np.array(list(zip(*DummyData)))
    Rslt = np.zeros(FactorData.shape)+np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.orthogonalize(FactorData[i],X=(X[i].T if X is not None else None),
                                                     mask=(Mask[i] if Mask is not None else None),
                                                     dummy_data=(DummyData[i].T if DummyData is not None else None),
                                                     **OperatorArg)
    return Rslt
def orthogonalize(Y, X, mask=None, constant=False, dummy_data=None, drop_dummy_na=False, other_handle='填充None', **kwargs):
    Factors = [Y]
    OperatorArg = {}
    if isinstance(X,Factor):
        Factors.append(X)
        OperatorArg["X"] = 1
    elif isinstance(X,list):
        Factors += X
        OperatorArg["X"] = len(X)
    else:
        OperatorArg["X"] = None
    if mask is not None:
        Factors.append(mask)
        OperatorArg["mask"] = 1
    else:
        OperatorArg["mask"] = None
    if isinstance(dummy_data,Factor):
        Factors.append(dummy_data)
        OperatorArg["dummy_data"] = 1
    elif isinstance(dummy_data,list):
        Factors += dummy_data
        OperatorArg["dummy_data"] = len(dummy_data)
    else:
        OperatorArg["dummy_data"] = None
    Descriptors,Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"drop_dummy_na":drop_dummy_na,"constant":constant,"other_handle":other_handle}
    Args["OperatorArg"].update(OperatorArg)
    return SectionOperation(kwargs.pop("factor_name", str(uuid.uuid1())),Descriptors,{"算子":_orthogonalize,"参数":Args,"运算时点":"多时点","输出形式":"全截面"}, **kwargs)

# ----------------------多截面运算--------------------------------
def _aggregate(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nID = len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    AggrFun = args["OperatorArg"]["aggr_fun"]
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nID, ), fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                Rslt[i] = AggrFun(FactorData[iMask])
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                Rslt[iMask] = AggrFun(FactorData[iMask])
    else:
        Rslt = np.full(shape=(nID, ), fill_value=AggrFun(FactorData[Mask]))
    return Rslt
def aggregate(f, aggr_fun=np.nansum, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"aggr_fun":aggr_fun, "Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggregate, "参数":Args, "运算时点":"单时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _disaggregate(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        for i, iID in enumerate(args["OperatorArg"]["aggr_ids"]):
            iMask = (CatData==iID)
            Rslt[iMask] = FactorData[:, [i]].repeat(nID, axis=1)[iMask]
    else:
        Rslt = FactorData.repeat(nID, axis=1)
    return Rslt
def disaggregate(f, aggr_ids, cat_data=None, disaggr_ids=None, **kwargs):# 将聚合因子分解成为普通因子
    if (len(aggr_ids)>1) and (cat_data is None): raise __QS_Error__("解聚合算子 disaggregate: 缺少类别因子!")
    Factors = [f]
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    DescriptorIDs = [aggr_ids] * Args.get("SepInd1", 0) + [disaggr_ids] * (len(Descriptors) - Args.get("SepInd1", 0))
    Args["OperatorArg"] = {"aggr_ids":aggr_ids, "CatData":(cat_data is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_disaggregate, "参数":Args, "运算时点":"多时点", "描述子截面":DescriptorIDs}, **kwargs)
def _disaggregate_list(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    Rslt = pd.DataFrame(Data[0], columns=["ID"], index=idt)
    HasValue = (len(Data)>1)
    if HasValue:
        Rslt["Value"] = Data[1]
    Rslt = Rslt.reset_index()
    Rslt.columns = ["DT", "ID"] + ["Value"] * HasValue
    Rslt = Rslt[pd.notnull(Rslt["ID"])]
    if Rslt.shape[0]==0: return np.full(shape=(len(idt), len(iid)), fill_value=None)
    Rslt["Len"] = Rslt["ID"].apply(len)
    Rslt["DT"] = Rslt.pop("DT").apply(lambda x: [x]) * Rslt["Len"]
    if HasValue:
        Rslt["Value"] = Rslt.loc[:, ["Value", "Len"]].apply(lambda s: (s.iloc[0] + [None]*(s.iloc[1]-len(s.iloc[0])))[:s.iloc[1]] if isinstance(s.iloc[0], list) else [s.iloc[0]] * s.iloc[1], axis=1)
    Rslt.pop("Len")
    Rslt = pd.DataFrame(Rslt.sum(axis=0).tolist(), index=Rslt.columns).T
    if not HasValue:
        Rslt["Value"] = 1
    Rslt = Rslt.set_index(["DT", "ID"])["Value"].unstack()
    if Rslt.columns.intersection(iid).shape[0]==0: return np.full(shape=(len(idt), len(iid)), fill_value=None)
    return Rslt.reindex(index=idt, columns=iid).values
def disaggregate_list(f, f_id, f_value=None, f_value_id=None, disaggr_ids=None, **kwargs):# 将值为 list 的聚合因子分解成为普通因子
    Factors = [f]
    if f_value is not None:
        Factors.append(f_value)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    if f_value_id is None: f_value_id = f_id
    DescriptorIDs = [[f_id]] * Args.get("SepInd1", 0) + [[f_value_id]] * (len(Descriptors) - Args.get("SepInd1", 0))
    Args["OperatorArg"] = {"f_id":f_id, "f_value_id": f_value_id}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_disaggregate_list, "参数":Args, "运算时点":"多时点", "描述子截面":DescriptorIDs}, **kwargs)
def _aggr_sum(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                Rslt[:, i] = np.nansum(iMask * FactorData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                Rslt[iMask] = np.nansum(iMask * FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        Rslt = np.nansum(FactorData * Mask, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_sum(f, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_sum, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_prod(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanprod(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            iData = np.full(shape=FactorData.shape, fill_value=np.nan)
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanprod(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanprod(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_prod(f, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_prod, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_max(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if FactorData.shape[1]==0: return np.full(shape=(nDT, nID), fill_value=np.nan)
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanmax(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanmax(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanmax(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_max(f, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_max, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_min(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if FactorData.shape[1]==0: return np.full(shape=(nDT, nID), fill_value=np.nan)
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanmin(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanmin(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanmin(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_min(f, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_min, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_mean(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["Weight"]:
        WeightData = Data[2-args["OperatorArg"]["Mask"]]
    else:
        WeightData = np.ones(FactorData.shape)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                Rslt[:, i] = np.nansum(iMask * WeightData * FactorData, axis=1) / np.nansum(iMask * WeightData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                Rslt[iMask] = (np.nansum(iMask * WeightData * FactorData, axis=1) / np.nansum(iMask * WeightData, axis=1)).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        Rslt = (np.nansum(FactorData * WeightData * Mask, axis=1) / np.nansum(WeightData * Mask, axis=1)).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_mean(f, mask=None, cat_data=None, weight_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if weight_data is not None:
        Factors.append(weight_data)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"Mask":(mask is not None), "Weight":(weight_data is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_mean, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_std(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanstd(iData, axis=1, ddof=args["OperatorArg"]["ddof"])
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanstd(iData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanstd(FactorData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_std(f, ddof=1, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"ddof":ddof, "Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_std, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_var(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanvar(iData, axis=1, ddof=args["OperatorArg"]["ddof"])
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanvar(iData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanvar(FactorData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_var(f, ddof=1, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"ddof":ddof, "Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_var, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_median(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanmedian(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanmedian(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanmedian(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_median(f, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_median, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_quantile(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanpercentile(iData, q=args["OperatorArg"]["quantile"]*100, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanpercentile(iData, q=args["OperatorArg"]["quantile"]*100, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanpercentile(FactorData, q=args["OperatorArg"]["quantile"]*100, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_quantile(f, quantile=0.5, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"quantile":quantile, "Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_quantile, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _aggr_count(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    nDT, nID = len(idt), len(iid)
    FactorData = pd.notnull(Data[0])
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1]==1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData==iID) & Mask)
                Rslt[:, i] = np.nansum(iMask * FactorData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData==iCat) & Mask)
                Rslt[iMask] = np.nansum(iMask * FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        Rslt = np.nansum(FactorData * Mask, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt
def aggr_count(f, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"Mask":(mask is not None), "CatData":(cat_data is not None), "SectionChged":(descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_aggr_count, "参数":Args, "运算时点":"多时点", "描述子截面":[descriptor_ids]*len(Descriptors)}, **kwargs)
def _merge(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)
    Rslt = np.concatenate(Data, axis=1)
    return pd.DataFrame(Rslt, columns=args["OperatorArg"]["descriptor_ids"]).reindex(columns=iid).values
def merge(factors, descriptor_ids, data_type="object", **kwargs):
    if len(factors)!=len(descriptor_ids): raise __QS_Error__("描述子个数与描述子截面个数不一致!")
    Descriptors, Args = _genMultivariateOperatorInfo(*factors)
    Args["OperatorArg"] = {"descriptor_ids": sum(descriptor_ids, [])}
    DescriptorIDs = []
    for i in range(len(factors)):
        StartInd, EndInd = Args.get("SepInd"+str(i), 0), Args.get("SepInd"+str(i+1), 0)
        DescriptorIDs += [descriptor_ids[i]] * (EndInd - StartInd)
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子":_merge, "参数":Args, "运算时点":"多时点", "描述子截面":DescriptorIDs, "数据类型": data_type}, **kwargs)
def _chg_ids(f,idt,iid,x,args):
    Data = _genOperatorData(f,idt,iid,x,args)[0]
    IDMap = args["OperatorArg"]["id_map"]
    OldIDs = f.DescriptorSection[0]
    Rslt = np.full(shape=(len(idt), len(iid)), fill_value=np.nan, dtype=Data.dtype)
    for i, iID in enumerate(iid):
        iOldID = IDMap.get(iID, None)
        if iOldID not in OldIDs: continue
        Rslt[:, i] = Data[:, OldIDs.index(iOldID)]
    return Rslt
def chg_ids(f, old_ids, id_map={}, **kwargs):# id_map: {新ID:旧ID}
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"id_map":id_map}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    DataType = f.getMetaData(key="DataType")
    if DataType is None: DataType = "object"
    return SectionOperation(FactorName, Descriptors, {"算子":_chg_ids, "参数":Args, "运算时点":"多时点", "描述子截面":[old_ids]*len(Descriptors), "数据类型":DataType}, **kwargs)
def _map_section(f,idt,iid,x,args):
    Data, Mapping = _genOperatorData(f,idt,iid,x,args)
    MappingIDs = (iid if not args["OperatorArg"]["mapping_ids"] else args["OperatorArg"]["mapping_ids"])
    Mapping = pd.Series(Mapping, index=MappingIDs)
    return Mapping.reindex(index=Data).values
def map_section(f, mapping, mapping_ids, **kwargs):
    Descriptors, Args = _genMultivariateOperatorInfo(f, mapping)
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    DataType = mapping.getMetaData(key="DataType")
    if DataType is None: DataType = "object"
    DescriptorIDs = []
    StartInd, EndInd = Args.get("SepInd0", 0), Args.get("SepInd1", 0)
    DescriptorIDs += [None] * (EndInd - StartInd)
    StartInd, EndInd = Args.get("SepInd1", 0), Args.get("SepInd2", 0)
    DescriptorIDs += [mapping_ids] * (EndInd - StartInd)
    Args["OperatorArg"] = {"mapping_ids": mapping_ids}
    return SectionOperation(FactorName, Descriptors, {"算子":_map_section, "参数":Args, "运算时点":"单时点", "描述子截面":DescriptorIDs, "数据类型":DataType}, **kwargs)