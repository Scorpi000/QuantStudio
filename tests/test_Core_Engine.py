import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Core.TreeEngine import TreeEngine
from QuantStudio.Factor.Factor import DataFactor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.Factor.FactorOperation import PointOperation, makeFactorOperator


if __name__ == "__main__":
    np.random.seed(0)
    nDT, nID = 10, 5
    SectionIDs = [str(i).zfill(6) + ".SZ" for i in range(1, nID + 1)]
    DTRuler = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(nDT)]
    IDs, DTs = SectionIDs[:3], DTRuler[-5:]
    
    Factor1 = DataFactor(data=1, args={"Name": "Factor1"}, qs_id="Factor1")
    Factor2 = DataFactor(data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs), args={"Name": "Factor2"}, qs_id="Factor2")
    
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
    }, qs_id="Factor3")

    Factor4 = (Factor3 + 1).new(args={"Name": "Factor4"}, qs_id="Factor4")

    NodeList = [Factor1, Factor2, Factor4, Factor3]
    LocalContext = FactorLocalContext(DTs=DTs, IDs=IDs)
    InitData = FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=SectionIDs)

    with FeatherFactorCache(args={"DTRuler": DTRuler, "CacheDir": "./data/Cache", "StartMode": "new"}) as Cache:
        with FactorContext(DTRuler=DTRuler, SectionIDs=SectionIDs, DataCache=Cache) as Context:
            # with Engine() as ExecEngine:
            with TreeEngine(args={"CalcConcurrentNum": 4, "CalcConcurrentMode": "Thread"}) as ExecEngine:
                Rslt = ExecEngine.run(NodeList, Context, fwd_data_list=[LocalContext]*len(NodeList), init_data_list=[InitData]*len(NodeList))
    
    # PIDList = [f"0-{i}" for i in range(4)]
    # with FeatherFactorCache(args={"PIDs": PIDList, "DTRuler": DTRuler, "CacheDir": "./data/Cache", "StartMode": "new"}) as Cache:
    #     with FactorContext(PIDList=PIDList, DTRuler=DTRuler, SectionIDs=SectionIDs, DataCache=Cache) as Context:
    #         with ParallelEngine() as ExecEngine:
    #             Rslt = ExecEngine.run(NodeList, Context, fwd_data_list=[LocalContext]*len(NodeList), init_data_list=[InitData]*len(NodeList))
    
    for iRslt in Rslt:
        print(iRslt)
    print("===")