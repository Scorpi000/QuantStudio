# -*- coding: utf-8 -*-
"""因子运算"""
import time
from concurrent.futures import ThreadPoolExecutor
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.FactorOperation import SectionOperation, PanelOperation, makeFactorOperator, FactorOperatorized

np.random.seed(0)
nDT, nID = 10, 10
SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
IDs, DTs = SectionIDs[:3], DTRuler
Factor1 = DataFactor(name="Factor1", data=1)
Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(SectionIDs)), index=DTs, columns=SectionIDs))


# 表达式方式
Factor3 = rename(Factor1 + Factor2, factor_name="Factor3")


# 工厂函数方式
def test_point(f, idt, iid, x, args):
    def fib(n):
        if n < 2:
            return n
        return fib(n-1) + fib(n-2)
    fib(35)
    return (x[0] + x[1]) ** (1/3)
test_point = makeFactorOperator(test_point, operator_type="Point", args={"Arity": 2, "DTMode": "单时点", "IDMode": "单ID"})
Factor4 = test_point(Factor1, Factor2, factor_args={"Name": "Factor4"})


# 装饰器方式
@FactorOperatorized(operator_type="Time", args={"Arity": 1, "IDMode": "多ID", "LookBack": [3 - 1]})
def test_time(f, idt, iid, x, args):
    return np.nansum(x[0], axis=0)
Factor5 = test_time(Factor1, factor_args={"Name": "Factor5", "Meta": {"Description": "我是 Factor5!"}})
print(Factor5.getMetaData(key="Description"))


# 直接实例化方式, 不推荐
def test_section(f, idt, iid, x, args):
    return np.argsort(np.argsort(x[0]))
test_section = makeFactorOperator(test_section, operator_type="Section", args={"Arity": 1, "DTMode": "单时点", "DescriptorSection": [SectionIDs]})
Factor6 = SectionOperation(descriptors=[Factor2], args={"Name": "Factor6", "Operator": test_section})


def test_panel(f, idt, iid, x, args):
    return np.argsort(np.argsort(x[0][0]))
test_panel = makeFactorOperator(test_panel, operator_type="Panel", args={"DTMode": "单时点", "LookBack": [1 - 1], "DescriptorSection": [SectionIDs]})
Factor7 = PanelOperation(descriptors=[Factor2], args={"Name": "Factor7", "Operator": test_panel})


if __name__ == "__main__":
    # print(Factor1.readData(ids=IDs, dts=DTs))
    # print(Factor2.readData(ids=IDs, dts=DTs))
    # print(Factor3.readData(ids=IDs, dts=DTs))
    # print(Factor4.readData(ids=IDs, dts=DTs))
    # print(Factor5.readData(ids=IDs, dts=DTs))
    # print(Factor6.readData(ids=IDs, dts=DTs))
    # print(Factor7.readData(ids=IDs, dts=DTs))
    
    with ThreadPoolExecutor(max_workers=10) as Executor:
        StartT = time.perf_counter()
        with FeatherFactorCache(args={"DTRuler": DTRuler, "CacheDir": r"D:\Data\Cache\DevCache", "StartMode": "new"}) as Cache:
            with FactorContext(DTRuler=DTRuler, SectionIDs=SectionIDs, DataCache=Cache, TaskExecutor=Executor, MaxWorkers=10) as Context:
                with Engine() as ExecEngine:
                    Rslt = ExecEngine.run(
                        [Factor4], 
                        Context, 
                        fwd_data_list=[FactorLocalContext(DTs=DTs, IDs=SectionIDs)], 
                        init_data_list=[FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=SectionIDs)]
                    )
        print(time.perf_counter() - StartT)
    # print(Rslt[0])

    StartT = time.perf_counter()
    with FeatherFactorCache(args={"DTRuler": DTRuler, "CacheDir": r"D:\Data\Cache\DevCache", "StartMode": "new"}) as Cache:
        with FactorContext(DTRuler=DTRuler, SectionIDs=SectionIDs, DataCache=Cache, MaxWorkers=0) as Context:
            with Engine() as ExecEngine:
                Rslt = ExecEngine.run(
                    [Factor4], 
                    Context, 
                    fwd_data_list=[FactorLocalContext(DTs=DTs, IDs=SectionIDs)], 
                    init_data_list=[FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=SectionIDs)]
                )
    print(time.perf_counter() - StartT)
    # print(Rslt[0])

    # ExecEngine = Engine()
    # Cache = FeatherFactorCache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "CacheDir": r"C:\Users\hst\Desktop\Cache", "PIDs": ["0"], "StartMode": "new"})
    # Cache.start()
    # Context = FactorContext(
    #     PID="0",
    #     PIDList=["0"],
    #     DTRuler=DTRuler,
    #     SectionIDs=SectionIDs,
    #     IDSplit="连续切分",
    #     DataCache=Cache
    # )
    # LocalContext = FactorLocalContext(DTs=DTs, IDs=IDs)
    # FactorList = [Factor1, Factor2, Factor3, Factor4, Factor5, Factor6, Factor7]
    # Rslt = ExecEngine.run(FactorList, Context, fwd_data_list=[LocalContext]*len(FactorList), init_data_list=[{"dt_range": (DTs[0], DTs[-1]), "section_ids": SectionIDs}]*len(FactorList))
    # print(Rslt)
    
    print("===")