# coding=utf-8
import pickle
import importlib

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.FactorDataBase.FactorOperation import DerivativeFactor

# 写入参数集
def writeArgs(args, arg_name=None, parent_var=None, var=None, tx=None):
    Parameters = {}
    if parent_var is not None:
        if var is None: var = parent_var + "_a"
        if arg_name is not None:
            Relation = f"[:参数 {{Name: '{arg_name}'}}]"
        else:
            Relation = f"[:参数 ]"
        CypherStr = f"MERGE ({parent_var}) - {Relation} -> ({var}:`参数集`)"
    else:
        if var is None: var = "a"
        CypherStr = f"CREATE ({var}:`参数集`)"
    if isinstance(args, __QS_Object__):
        Parameters[var] = {"_PickledObject": pickle.dumps(args)}
        args = args.Args
    else:
        Parameters[var] = {}
    if isinstance(args, dict):
        args = args.copy()
        SubArgs = {}
        for iArg in sorted(args.keys()):
            if isinstance(args[iArg], (dict, __QS_Object__)):
                SubArgs[iArg] = args.pop(iArg)
            elif callable(args[iArg]):
                args[iArg] = pickle.dumps(args[iArg])
        if args:
            CypherStr += f" SET {var} += ${var}"
            Parameters[var].update(args)
        for i, iArg in enumerate(SubArgs.keys()):
            iCypherStr, iParameters = writeArgs(SubArgs[iArg], arg_name=iArg, parent_var=var, var=var+"_a"+str(i), tx=None)
            CypherStr += " " + iCypherStr
            Parameters.update(iParameters)
    else:
        raise __QS_Error__(f"不支持的因子集类型 : {type(args)}")
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
                iArgStr, iParameters = writeArgs(iFactor.Args, arg_name=None, parent_var=iVar, tx=None)
                if iArgStr: CypherStr += " "+iArgStr
                Parameters.update(iParameters)
                id_var[iFID] = iVar
        else:# 衍生因子
            iSubStr, iSubParameters = writeFactor(iFactor.Descriptors, tx=None, id_var=id_var)
            if iSubStr: CypherStr += " "+iSubStr
            Parameters.update(iSubParameters)
            if iFID not in id_var:
                CypherStr += f" CREATE {iNode}"
                iArgStr, iParameters = writeArgs(iFactor.Args, arg_name=None, parent_var=iVar, tx=None)
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
            CypherStr += f" MERGE {FTNode} - [:`属于因子库`] -> ({FDBVar})"
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, parent_var=var,  tx=None)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            id_var[FTID] = var
            # 写入因子
            for iFactorName in ft.FactorNames:
                CypherStr += f" MERGE (:`因子` {{Name: '{iFactorName}'}}) - [:`属于因子表`] -> ({var})"
    else:# 无上层因子库, 自定义因子表
        if FTID not in id_var:
            FStr, FParameters = writeFactor([ft.getFactor(iFactorName) for iFactorName in ft.FactorNames], tx=None, id_var=id_var)
            if FStr: CypherStr += " "+FStr
            Parameters.update(FParameters)
            CypherStr += f" CREATE {FTNode} "
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, parent_var=var, tx=None)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            for iFactorName in ft.FactorNames:
                CypherStr += f" MERGE ({var}) - [:`包含因子`] -> ({id_var[id(ft.getFactor(iFactorName))]})"
            id_var[FTID] = var
    if tx is not None: tx.run(CypherStr, parameters=Parameters)
    return CypherStr, Parameters

# 写入因子库
def writeFactorDB(fdb, tx=None, var="fdb"):
    Class = str(fdb.__class__)
    Class = Class[Class.index("'")+1:]
    Class = Class[:Class.index("'")]
    Node = f"({var}:`因子库`:`{fdb.__class__.__name__}` {{`Name`: '{fdb.Name}', `_Class`: '{Class}'}})"
    CypherStr = "MERGE "+Node
    Args = fdb.Args
    Args.pop("用户名", None)
    Args.pop("密码", None)
    ArgStr, Parameters = writeArgs(Args, arg_name=None, parent_var=var, tx=None)
    if ArgStr: CypherStr += " " + ArgStr
    if tx is not None: tx.run(CypherStr, parameters=Parameters)
    return CypherStr, Parameters

# 读取参数集
def readArgs(node, node_id=None, tx=None):
    if node_id is None:
        CypherStr = f"MATCH p={node} - [:`参数`*1..] -> (args:`参数集`) RETURN length(p) AS Level, [r IN relationships(p) | r.Name] AS Path, properties(args) AS Args ORDER BY Level"
    else:
        CypherStr = f"MATCH p=(n) - [:`参数`*1..] -> (args:`参数集`) WHERE id(n)={node_id} RETURN length(p) AS Level, [r IN relationships(p) | r.Name] AS Path, properties(args) AS Args ORDER BY Level"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    for i, iRslt in enumerate(Rslt):
        iArgs = iRslt[2]
        if "_PickledObject" in iArgs:
            iObj = iArgs.pop("_PickledObject")
            try:
                iObj = pickle.loads(iObj)
            except Exception as e:
                print(e)
            else:
                for iArgName, iVal in enumerate(iArgs.items()):
                    iObj[iArgName] = iVal
                iArgs = iObj
        for iArgName, iVal in enumerate(iArgs.items()):
            if isinstance(iVal, bytes):
                try:
                    iArgs[iArgName] = pickle.loads(iVal)
                except Exception as e:
                    print(e)
        if i==0:
            Args = iArgs
        else:
            iDict = Args
            for iKey in iRslt[1][1:-1]:
                iSubDict = iDict.get(iKey, {})
                if iKey not in iDict:
                    iDict[iKey] = iSubDict
                iDict = iSubDict
            iDict[iRslt[1][-1]] = iArgs
    return Args

# 读取因子库
def readFactorDB(name, labels=["因子库"], properties={}, node_id=None, tx=None):
    if node_id is None:
        if name is not None:
            properties["Name"] = name
        Constraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in properties.items()))
        CypherStr = f"MATCH (fdb:`{'`:`'.join(labels)}` {{{Constraint}}}) RETURN fdb"
    else:
        CypherStr = f"MATCH (fdb) WHERE id(fdb)={node_id} RETURN fdb"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    FDBs = []
    for iRslt in Rslt:
        iProperties = dict(iRslt[0])
        if "_PickledObject" in iProperties:
            iFDB = iProperties.pop("_PickledObject")
            try:
                iFDB = pickle.loads(iFDB)
            except Exception as e:
                print(e)
            else:
                iArgs = readArgs(None, node_id=iRslt[0].id, tx=tx)
                for iArgName, iVal in enumerate(iArgs.items()):
                    iFDB[iArgName] = iVal
                FDBs.append(iFDB)
                continue
        try:
            Class = iProperties["_Class"].split(".")
            #exec(f"from {'.'.join(Class[:-1])} import {Class[-1]} as FDBClass")
            iModule = importlib.import_module('.'.join(Class[:-1]))
            FDBClass = getattr(iModule, Class[-1])
        except Exception as e:
            raise __QS_Error__(f"无法还原因子库({iProperties})对象: {e}")
        else:
            iArgs = readArgs(None, node_id=iRslt[0].id, tx=tx)
            iFDB = FDBClass(sys_args=iArgs)
        FDBs.append(iFDB)
    if len(FDBs)>1:
        return FDBs
    else:
        return FDBs[0]