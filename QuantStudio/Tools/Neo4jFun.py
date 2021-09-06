# coding=utf-8
from QuantStudio import __QS_Object__
from QuantStudio.FactorDataBase.FactorOperation import DerivativeFactor

# 写入参数集
def writeArgs(args, arg_name=None, tx=None, node=None, node_var=None, var="a", parent_var=None):
    Parameters = {}
    if node is not None:
        if node_var is None:
            CypherStr = f"MERGE {node}"
        else:
            CypherStr = ""
            var = node_var
    else:
        if parent_var is not None:
            CypherStr = f"MERGE ({parent_var}) - [:参数 {{Name: '{arg_name}'}}] -> ({var}:`参数集`)"
        else:
            CypherStr = f"CREATE ({var}:`参数集`)"
    if args:
        args = args.copy()
        SubArgs = {}
        for iArg in sorted(args.keys()):
            if isinstance(args[iArg], dict):
                SubArgs[iArg] = args.pop(iArg)
            elif isinstance(args[iArg], __QS_Object__):
                iObj = args.pop(iArg)
                SubArgs[iArg] = iObj.Args
                SubArgs[iArg]["ObjClass"] = str(type(iObj))
            elif callable(args[iArg]):
                args[iArg] = args[iArg].__name__
        if args:
            CypherStr += f" SET {var} += ${var}"
            Parameters[var] = args
        for i, iArg in enumerate(SubArgs.keys()):
            iCypherStr, iParameters = writeArgs(SubArgs[iArg], arg_name=iArg, tx=None, node=None, var=var+"_a"+str(i), parent_var=var)
            CypherStr += " " + iCypherStr
            Parameters.update(iParameters)
    if tx is not None: tx.run(CypherStr, parameters=Parameters)
    return CypherStr, Parameters

# 写入因子, id_var: {id(对象): 变量名}
def writeFactor(factors, tx=None, id_var={}):
    CypherStr, Parameters = "", {}
    for iFactor in factors:
        iFID = id(iFactor)
        iVar = f"f{iFID}"
        iNode = f"({iVar}:`因子` {{Name: '{iFactor.Name}'}})"
        if not isinstance(iFactor, DerivativeFactor):# 基础因子
            iFT = iFactor.FactorTable
            if iFID not in id_var:
                if iFT is not None:# 有上层因子表
                    iFTVar = f"ft{id(iFT)}"
                    iFTStr, iFTParameters = writeFactorTable(iFT, tx=None, var=iFTVar, id_var=id_var)
                    if iFTStr: CypherStr += " "+iFTStr
                    Parameters.update(iFTParameters)
                    CypherStr += f" CREATE {iNode} - [:`产生于` {{`原始因子`: '{iFactor._NameInFT}'}}] -> ({iFTVar})"
                else:
                    CypherStr += f" CREATE {iNode}"
                iArgStr, iParameters = writeArgs(iFactor.Args, tx=None, node=iNode, node_var=iVar)
                if iArgStr: CypherStr += " "+iArgStr
                Parameters.update(iParameters)
                id_var[iFID] = iVar
        else:# 衍生因子
            iSubStr, iSubParameters = writeFactor(iFactor.Descriptors, tx=None, id_var=id_var)
            if iSubStr: CypherStr += " "+iSubStr
            Parameters.update(iSubParameters)
            if iFID not in id_var:
                CypherStr += f" CREATE {iNode}"
                iArgStr, iParameters = writeArgs(iFactor.Args, tx=None, node=iNode, node_var=iVar)
                if iArgStr: CypherStr += " "+iArgStr
                Parameters.update(iParameters)
                for jDescriptor in iFactor.Descriptors:
                    CypherStr += f" MERGE ({iVar}) - [:`依赖`] -> ({id_var[id(jDescriptor)]})"
                id_var[iFID] = iVar
    if tx is not None: tx.run(CypherStr, parameters=Parameters)
    return CypherStr, Parameters

# 写入因子表
def writeFactorTable(ft, tx=None, var="ft", id_var={}):
    CypherStr, Parameters = "", {}
    FTNode = f"({var}:`因子表` {{Name: '{ft.Name}'}})"
    # 写入因子库
    FDB = ft.FactorDB
    FTID, FDBID = id(ft), id(FDB)
    FDBVar = f"fdb{FDBID}"
    if FDB is not None:# 有上层因子库, 非自定义因子表
        if FDBID not in id_var:
            FDBStr, FDBParameters = writeFactorDB(FDB, tx=None, var=FDBVar)
            CypherStr += " "+FDBStr
            Parameters.update(FDBParameters)
            id_var[FDBID] = FDBVar
        # 写入因子表
        if FTID not in id_var:
            CypherStr += f" MERGE {FTNode} - [:`属于`] -> ({FDBVar})"
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, tx=None, node=FTNode, node_var=var)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            id_var[FTID] = var
            # 写入因子
            for iFactorName in ft.FactorNames:
                CypherStr += f" MERGE (:`因子` {{Name: '{iFactorName}'}}) - [:`属于`] -> ({var})"
    else:# 无上层因子库, 自定义因子表
        if FTID not in id_var:
            FStr, FParameters = writeFactor([ft.getFactor(iFactorName) for iFactorName in ft.FactorNames], tx=None, id_var=id_var)
            if FStr: CypherStr += " "+FStr
            Parameters.update(FParameters)
            CypherStr += f" CREATE {FTNode} "
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, tx=None, node=FTNode, node_var=var)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            for iFactorName in ft.FactorNames:
                CypherStr += f" MERGE ({var}) - [:`包含因子`] -> ({id_var[id(ft.getFactor(iFactorName))]})"
            id_var[FTID] = var
    if tx is not None: tx.run(CypherStr, parameters=Parameters)
    return CypherStr, Parameters

# 写入因子库
def writeFactorDB(fdb, tx=None, var="fdb"):
    Node = f"({var}:`因子库`:`{fdb.__class__.__name__}` {{`Name`: '{fdb.Name}'}})"
    CypherStr = "MERGE "+Node
    Args = fdb.Args
    Args.pop("用户名", None)
    Args.pop("密码", None)
    ArgStr, Parameters = writeArgs(Args, arg_name=None, tx=None, node=Node, node_var=var)
    if ArgStr: CypherStr += " " + ArgStr
    if tx is not None: tx.run(CypherStr, parameters=Parameters)
    return CypherStr, Parameters

if __name__=="__main__":
    import QuantStudio.api as QS
    from QuantStudio.FactorDataBase.Neo4jDB import Neo4jDB
    iDB = Neo4jDB()
    iDB.connect()
    
    #iFT = iDB.getTable("BenchmarkIndexFactor")
    #print(iFT.Args)
    #iFactor = iFT.getFactor("收盘价")
    #print(iFactor.Args)
    #TestArgs = {
        #"a": 1, 
        #"b": {
            #"b_a": "3", 
            #"b_b": [1,2,3], 
            #"b_c": {}
        #}
    #}
    #with iDB.session() as Session:
        #with Session.begin_transaction() as tx:
            #CypherStr, Parameters = writeArgs(TestArgs, arg_name=None, tx=tx, node=None, var="test", parent_var=None)
    #print(CypherStr)
    
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    TableName = "BenchmarkIndexFactor"
    FT = HDB.getTable(TableName)
    CFT = QS.FactorDB.CustomFT(name="aha_ft")
    High, Low = FT.getFactor("最高价", new_name="high"), FT.getFactor("最低价", new_name="low")
    CFT.addFactors(factor_list=[High, QS.FactorDB.Factorize((High+Low)/2, factor_name="mid")])
    with iDB.session() as Session:
        with Session.begin_transaction() as tx:
            CypherStr, Parameters = writeFactorTable(CFT, tx=tx, var=f"ft{id(CFT)}")
    iDB.disconnect()