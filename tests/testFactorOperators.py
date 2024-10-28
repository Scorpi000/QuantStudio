# coding=utf-8
import datetime as dt

import numpy as np
import pandas as pd

import QuantStudio.FactorDataBase.FactorOperators as fo
from QuantStudio.FactorDataBase.FactorDB import DataFactor


IDs = [f"00000{i}.SZ" for i in range(1, 6)]
SectionIDs = IDs + [f"00000{i}.SZ" for i in range(6, 10)]
DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
DTRuler = [dt.datetime(2020, 1, 1) - dt.timedelta(i+1) for i in range(7)] + DTs

np.random.seed(0)
Factor1 = DataFactor(name="Factor1", data=1)
Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTRuler), len(SectionIDs)), index=DTRuler, columns=SectionIDs))
Factor3 = DataFactor(name="Factor3", data=pd.DataFrame(np.random.choice(5, size=(len(DTRuler), len(SectionIDs))).astype(str), index=DTRuler, columns=SectionIDs))
print(Factor1.Name, Factor1.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")
print(Factor2.Name, Factor2.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")
print(Factor3.Name, Factor3.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

## ----------------------单点运算--------------------------------
#Factor = fo.AsType(dtype="double")(Factor3)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Log()(Factor2, base=2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.NotNull()(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.IsIn(test_elements=["2", "3"])(Factor3)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Applymap(func=lambda x: x + 1)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Where()(Factor2, Factor2 > 0, 0)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#CompoundFactor = DataFactor(name="CompoundFactor", data=(1, "a"), sys_args={"数据类型": "object"})
#print(CompoundFactor.Name, CompoundFactor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")
#Factor = fo.Fetch()(CompoundFactor, pos=1, dtype="string")
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = DataFactor(name="DateTimeFactor", data=dt.datetime(2023, 1, 1), sys_args={"数据类型": "object"})
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")
#Factor = fo.Strftime(dt_format="%Y%m%d")(Factor)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")
#Factor = fo.Strptime(dt_format="%Y%m%d")(Factor)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Sum()(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Max()(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Min()(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Rank()(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Mean()(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Std()(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Regress(intercept=True, output="beta")(Factor1, Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RegressChangeRate()(Factor1, Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.ToList()(Factor1, Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = DataFactor(name="JsonFactor", data={"a": 1, "b": [2, 3]}, sys_args={"数据类型": "object"})
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")
#Factor = fo.ToJson()(Factor)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.ToCompound()(Factor1, Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")


## ----------------------时序运算--------------------------------
#Factor = fo.Lag(lag_period=2, window=2)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingSum(window=2, min_periods=2)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingMax(window=2, min_periods=2)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingMin(window=2, min_periods=2)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingRank(window=2, min_periods=2, uniformization=False)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingMean(window=2, min_periods=2)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingStd(window=2, min_periods=2)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingChangeRate(window=2)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.RollingRegress(window=2, min_periods=2, intercept=False, output="beta0")(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")


## ----------------------截面运算--------------------------------
#Factor = fo.SectionRank(ascending=True, uniformization=False)(Factor2)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.Aggregate(aggr_func=np.sum)(Factor2, descriptor_ids=IDs)
#print(Factor.Name, Factor.readData(ids=["000000.HST"], dts=DTs), sep="\n", end="\n\n")
#Factor = fo.Disaggregate(aggr_ids=["000000.HST"], disaggr_ids=IDs)(Factor)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")

#iIDs = [f"60000{i}.SH" for i in range(3)]
#ConcatFactor = DataFactor(name="ConcatFactor", data=pd.DataFrame(np.random.randn(len(DTRuler), len(iIDs)), index=DTRuler, columns=iIDs))
#print(ConcatFactor.Name, ConcatFactor.readData(ids=iIDs, dts=DTs), sep="\n", end="\n\n")
#Factor = fo.ConcatSection(descriptor_sections=[IDs, iIDs])(Factor2, ConcatFactor)
#print(Factor.Name, Factor.readData(ids=IDs+iIDs, dts=DTs), sep="\n", end="\n\n")

#Factor = fo.ChgSection(old_ids=IDs, id_map={"AHA"+iID: iID for iID in IDs})(Factor2)
#print(Factor.Name, Factor.readData(ids=["AHA"+iID for iID in IDs], dts=DTs), sep="\n", end="\n\n")

#Factor = fo.SectionRegress(intercept=False, output="beta0")(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")


## ----------------------面板运算--------------------------------
#Factor = fo.PanelRegress(window=2, intercept=False, output="beta0")(Factor2, Factor1)
#print(Factor.Name, Factor.readData(ids=IDs, dts=DTs), sep="\n", end="\n\n")


print("===")