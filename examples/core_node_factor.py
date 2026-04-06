import datetime as dt
from typing import Any, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio.Core.Node import Node
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Factor.Factor import Factor, DataFactor, FactorContext, FactorLocalContext
from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.FactorOperator import SectionRank
from QuantStudio.Factor.BaoStockDB import BaoStockDB
from QuantStudio.Factor.HDF5DB import HDF5DB
from QuantStudio.Factor.FactorCache import HDF5Cache, FeatherCache
from QuantStudio.Factor.FactorOperation import PointOperation, makeFactorOperator
from QuantStudio.Factor.FactorStorer import FactorStorer


if __name__ == "__main__":
    np.random.seed(0)
    nDT, nID = 10, 5
    SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
    DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
    IDs, DTs = SectionIDs[:3], DTRuler[-5:]
    
    Factor1 = DataFactor(data=1, args={"Name": "Factor1"})
    Factor2 = DataFactor(data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs), args={"Name": "Factor2"})
    
    def add(f, idt, iid, x, args):
        print("add")
        return x[0] + x[1]
    
    Factor3 = PointOperation(descriptors=[Factor1, Factor2], args={
        "Name": "Factor3",
        "Operator": makeFactorOperator(
            add,
            operator_type="Point",
            args={
                "DTMode": "多时点",
                "IDMode": "多ID"
            }
        ),
        #"CacheEnabled": False
    })

    Factor4 = rename(Factor3 + 1, factor_name="Factor4")
    Factor5 = rename(Factor2, factor_name="Factor5")
    
    Factor6 = SectionRank()(Factor3, factor_args={"Name": "Factor6"})
    
    ExecEngine = Engine()
    Cache = FeatherCache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "CacheDir": r"C:\Users\hst\Project\Data\FactorCache", "PIDs": ["0"]})
    Cache.start()
    Context = FactorContext(
        DTRuler=DTRuler,
        SectionIDs=SectionIDs,
        DataCache=Cache
    )
    LocalContext = FactorLocalContext(DTs=DTs, IDs=IDs)
    FactorList = [Factor2, Factor3, Factor4, Factor6]
    Rslt = ExecEngine.run(FactorList, Context, fwd_data_list=[LocalContext]*len(FactorList), init_data_list=[{"dt_range": (DTs[0], DTs[-1]), "section_ids": SectionIDs}]*len(FactorList))
    # TDB = HDF5DB().connect()
    # Storer = FactorStorer(deps=[Factor2, Factor3], args={"TargetFDB": TDB, "TargetTable": "test_table1", "IfExists": "update"})
    # Rslt = ExecEngine.run([Storer], Context, fwd_data_list=[LocalContext], init_data_list=[{"dt_range": (DTs[0], DTs[-1]), "section_ids": SectionIDs}])
    print(Rslt)
    
    print("===")

if __name__=="__main__1":
    BSDB = BaoStockDB().connect()
    print(BSDB.TableNames)

    FT = BSDB.getTable("A股K线数据")
    Open = FT.getFactor("open")
    Close = FT.getFactor("close")

    np.random.seed(0)
    nDT, nID = 10, 5
    SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
    DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
    IDs, DTs = SectionIDs[:3], DTRuler[-5:]

    ExecEngine = Engine()
    Cache = HDF5Cache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "CacheDir": r"C:\Users\hst\Desktop\Cache", "PIDs": ["0"]})
    Cache.start()
    Context = FactorContext(
        DTRuler=DTRuler,
        SectionIDs=SectionIDs,
        DataCache=Cache
    )
    LocalContext = FactorLocalContext(DTs=DTs, IDs=IDs)
    FactorList = [Open, Close]
    Rslt = ExecEngine.run(FactorList, Context, fwd_data_list=[LocalContext]*len(FactorList), init_data_list=[{"dt_range": (DTs[0], DTs[-1]), "section_ids": SectionIDs}]*len(FactorList))
    print(Rslt)

    # ExecEngine = ParallelEngine(args={"IOConcurrentNum": None, "RsltMerger": lambda l: pd.concat(l, join="outer", axis=1, ignore_index=False).sort_index(axis=1)})
    # PIDList = [f"0-{i}" for i in range(2)]
    # Cache = HDF5Cache(args={"DTRuler": DTRuler, "MinDTUnit": dt.timedelta(1), "CacheDir": r"C:\Users\hst\Desktop\Cache", "PIDs": PIDList})
    # Cache.start()
    # Context = FactorContext(
    #     PIDList=PIDList,
    #     DTRuler=DTRuler,
    #     SectionIDs=SectionIDs,
    #     DataCache=Cache
    # )
    # LocalContext = FactorLocalContext(dts=DTs, ids=IDs)
    # FactorList = [Open, Close]
    # Rslt = ExecEngine.run(FactorList, Context, fwd_data_list=[LocalContext]*len(FactorList), init_data_list=[{"dt_range": (DTs[0], DTs[-1]), "section_ids": SectionIDs}]*len(FactorList))
    # print(Rslt)

    print("===")



# class Factor(Node):
#     class __QS_ArgClass__(Node.__QS_ArgClass__):
#         lookback: List[int] = Field(default=[])
#
#     def __init__(self, deps: List["Node"] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
#         if "name" not in args: args["name"] = "factor"
#         return super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)
#
#     def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
#         DTRuler = context.dt_ruler
#         StartIdx = DTRuler.index(fwd_data[0])
#         return [DTRuler[max(StartIdx - iLookback, 0):] for iLookback in self._QSArgs.lookback], DTRuler[StartIdx:]
#
#     def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any = None) -> Any:
#         if bwd_data_list:
#             return bwd_data_list[np.argmax(self._QSArgs.lookback)]
#         else:
#             return local_context

# if __name__=="__main__1":
#     class FactorContext(Context):
#         dt_ruler: List[int]
#
#     Factor1 = Factor(deps=[], args={"name": "f1"})
#     Factor2 = Factor(deps=[], args={"name": "f2"})
#     Factor3 = Factor(deps=[Factor1, Factor2], args={"lookback": [0, 1], "name": "f3"})
#
#     Engine = SimpleEngine()
#     # Engine = RecursiveEngine()
#     TestContext = FactorContext(dt_ruler=list(range(10)))
#     Rslt = Engine.run([Factor3], TestContext, fwd_data_list=[[3, 4]])
#     print(Rslt)
#
#     print("===")


# class Account(Node):
#     def __init__(self, deps: List["Node"] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
#         if "name" not in args: args["name"] = "account"
#         return super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)
#
#     def init_compute(self, path: List[str], init_data: Any, context: Context):
#         if self.QSID in path:
#             print("Account terminate init")
#             return []
#         print("Account init")
#         StartDT = context.node_state.setdefault(self.QSID, {}).get("start_dt", None)
#         if not StartDT: StartDT = init_data[0]
#         else: StartDT = min(StartDT, init_data[0])
#         context.node_state[self.QSID]["start_dt"] = StartDT
#
#         return [init_data]
#
#     def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
#         print(f"Account forward compute: {fwd_data}")
#         if not fwd_data:
#             print(f"Account terminate forward compute!")
#             return [], []
#         return [fwd_data], fwd_data
#
#     def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any = None) -> Any:
#         print(f"Account backward compute: {local_context}")
#         return pd.Series(0, index=local_context)

# class Signal(Node):
#     def __init__(self, deps: List["Node"] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
#         if "name" not in args: args["name"] = "signal"
#         return super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)
#
#     def init_compute(self, path: List[str], init_data: Any, context: Context):
#         if self.QSID in path:
#             print("Signal terminate init")
#             return []
#         print("Signal init")
#         StartDT = context.node_state.setdefault(self.QSID, {}).get("start_dt", None)
#         if not StartDT: StartDT = init_data[0]
#         else: StartDT = min(StartDT, init_data[0])
#         context.node_state[self.QSID]["start_dt"] = StartDT
#         DTRuler = context.dt_ruler
#         StartIdx = DTRuler.index(init_data[0])
#         return [DTRuler[StartIdx-1:StartIdx]+init_data]
#
#     def forward_compute(self, path: List[str], fwd_data: Any, context: Context) -> Tuple[List[Any], Any]:
#         print(f"Signal forward compute: {fwd_data}")
#         if not fwd_data: return [[]], []
#         DTRuler = context.dt_ruler
#         StartDT = context.node_state[self.QSID]["start_dt"]
#         AccountStartDT = DTRuler[DTRuler.index(StartDT) - 1]
#         StartIdx = DTRuler.index(fwd_data[0])
#         MaxStartIdx = DTRuler.index(max(AccountStartDT, DTRuler[StartIdx - 1]))
#         return [DTRuler[MaxStartIdx:StartIdx] + fwd_data[:-1]], fwd_data
#
#     def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any = None) -> Any:
#         print(f"Signal backward compute: {local_context}")
#         if bwd_data_list:
#             return pd.Series(len(bwd_data_list[0]), index=local_context)
#         else:
#             return pd.Series()
#
# if __name__ == "__main__1":
#     Account = Account(deps=[])
#     Signal = Signal(deps=[Account])
#     Account.Deps.append(Signal)
#
#     print(Account.QSID)
#     print(Signal.QSID)
#
#     # Engine = Engine()
#     Engine = StackEngine()
#     class IterContext(Context):
#         dt_ruler: List[dt.datetime]
#         dts: List[dt.datetime]
#     TestContext = IterContext(
#         dt_ruler=[dt.datetime(2025, 1, 1)+dt.timedelta(i) for i in range(20)],
#         dts=[dt.datetime(2025, 1, 5)+dt.timedelta(i) for i in range(3)]
#     )
#     Rslt = Engine.run([Signal, Account], TestContext, init_data_list=[TestContext.dts]*2, fwd_data_list=[TestContext.dts]*2)
#     print("Signal: ", Rslt[0])
#     print("Account: ", Rslt[1])

    print("===")