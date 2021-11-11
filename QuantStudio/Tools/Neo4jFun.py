# coding=utf-8
import pickle
import importlib

import numpy as np
import pandas as pd

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.FactorDataBase.FactorOperation import DerivativeFactor
from QuantStudio.FactorDataBase.FactorDB import CustomFT

# 写入参数集
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的参数集节点 id
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
            elif callable(args[iArg]) or isinstance(args[iArg], (list, tuple, pd.Series, pd.DataFrame, np.ndarray)):
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
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({var})"
        return tx.run(CypherStr, parameters=Parameters).values()[0][0]    

# 写入因子, id_var: {id(对象): 变量名}
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的 [因子节点 id]
def writeFactor(factors, tx=None, id_var=None, write_other_fundamental_factor=True):
    if id_var is None: id_var = {}
    CypherStr, Parameters, NodeVars = "", {}, []
    for iFactor in factors:
        iFID = id(iFactor)
        iVar = f"f{iFID}"
        iClass = str(iFactor.__class__)
        iClass = iClass[iClass.index("'")+1:]
        iClass = iClass[:iClass.index("'")]
        if not isinstance(iFactor, DerivativeFactor):# 基础因子
            iFT = iFactor.FactorTable
            if iFID not in id_var:
                iNode = f"({iVar}:`因子`:`基础因子` {{Name: '{iFactor.Name}', `_Class`: '{iClass}'}})"
                if iFT is not None:# 有上层因子表
                    iFTVar = f"ft{id(iFT)}"
                    iFTStr, iFTParameters = writeFactorTable(iFT, tx=None, var=iFTVar, id_var=id_var, write_other_fundamental_factor=write_other_fundamental_factor)
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
            iSubStr, iSubParameters = writeFactor(iFactor.Descriptors, tx=None, id_var=id_var, write_other_fundamental_factor=write_other_fundamental_factor)
            if iSubStr: CypherStr += " "+iSubStr
            Parameters.update(iSubParameters)
            if iFID not in id_var:
                iNode = f"({iVar}:`因子`:`衍生因子` {{Name: '{iFactor.Name}', `_Class`: '{iClass}'}})"
                CypherStr += f" CREATE {iNode}"
                iArgStr, iParameters = writeArgs(iFactor.Args, arg_name=None, parent_var=iVar, tx=None)
                if iArgStr: CypherStr += " "+iArgStr
                Parameters.update(iParameters)
                for j, jDescriptor in enumerate(iFactor.Descriptors):
                    CypherStr += f" MERGE ({iVar}) - [:`依赖` {{Order: {j}}}] -> ({id_var[id(jDescriptor)]})"
                id_var[iFID] = iVar
        NodeVars.append(iVar)
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({'), id('.join(NodeVars)})"
        return tx.run(CypherStr, parameters=Parameters).values()[0]

# 删除因子
def deleteFactor(factor_name, labels=["因子"], ft=None, fdb=None, del_descriptors=True, tx=None):
    raise NotImplemented

# 写入因子表
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的因子表节点 id
def writeFactorTable(ft, tx=None, var="ft", id_var=None, write_other_fundamental_factor=True):
    if id_var is None: id_var = {}
    CypherStr, Parameters = "", {}
    Class = str(ft.__class__)
    Class = Class[Class.index("'")+1:]
    Class = Class[:Class.index("'")]
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
            FTNode = f"({var}:`因子表`:`库因子表` {{Name: '{ft.Name}', `_Class`: '{Class}'}})"
            CypherStr += f" MERGE {FTNode} - [:`属于因子库`] -> ({FDBVar})"
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, parent_var=var,  tx=None)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            id_var[FTID] = var
            # 写入因子
            if write_other_fundamental_factor:
                for iFactorName in ft.FactorNames:
                    CypherStr += f" MERGE (:`因子`:`基础因子` {{Name: '{iFactorName}'}}) - [:`属于因子表`] -> ({var})"
    else:# 无上层因子库, 自定义因子表
        if FTID not in id_var:
            FStr, FParameters = writeFactor([ft.getFactor(iFactorName) for iFactorName in ft.FactorNames], tx=None, id_var=id_var, write_other_fundamental_factor=write_other_fundamental_factor)
            if FStr: CypherStr += " "+FStr
            Parameters.update(FParameters)
            FTNode = f"({var}:`因子表`:`自定义因子表` {{Name: '{ft.Name}', `_Class`: '{Class}'}})"
            CypherStr += f" CREATE {FTNode} "
            FTArgs = ft.Args
            FTArgs.pop("遍历模式", None)
            ArgStr, FTParameters  = writeArgs(FTArgs, arg_name=None, parent_var=var, tx=None)
            if ArgStr: CypherStr += " "+ArgStr
            Parameters.update(FTParameters)
            for iFactorName in ft.FactorNames:
                CypherStr += f" MERGE ({var}) - [:`包含因子`] -> ({id_var[id(ft.getFactor(iFactorName))]})"
            id_var[FTID] = var
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({var})"
        return tx.run(CypherStr, parameters=Parameters).values()[0][0]

# 删除因子表
def deleteFactorTable(table_name, labels=["因子表"], fdb=None, del_factors=True, del_descriptors=True, tx=None):
    raise NotImplementedError

# 写入因子库
# tx: None 返回 Cypher 语句和参数
# tx: 非 None 返回写入的因子库节点 id
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
    if tx is None:
        return CypherStr, Parameters
    else:
        CypherStr += f" RETURN id({var})"
        return tx.run(CypherStr, parameters=Parameters).values()[0][0]

# 删除因子库
def deleteFactorDB(fdb_name, tx=None):
    raise NotImplementedError

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
        for iArgName, iVal in iArgs.items():
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
def readFactorDB(labels=["因子库"], properties={}, node_id=None, tx=None, **kwargs):
    if node_id is None:
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
            iModule = importlib.import_module('.'.join(Class[:-1]))
            FDBClass = getattr(iModule, Class[-1])
        except Exception as e:
            raise __QS_Error__(f"无法还原因子库({iProperties})对象: {e}")
        else:
            iArgs = readArgs(None, node_id=iRslt[0].id, tx=tx)
            iFDB = FDBClass(sys_args=iArgs, **kwargs)
        FDBs.append(iFDB)
    if len(FDBs)>1:
        return FDBs
    else:
        return FDBs[0]

# 读取因子表
def readFactorTable(labels=["因子表"], properties={}, node_id=None, tx=None, **kwargs):
    if node_id is None:
        Constraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in properties.items()))
        CypherStr = f"MATCH (ft:`{'`:`'.join(labels)}` {{{Constraint}}}) RETURN ft" 
    else:
        CypherStr = f"MATCH (ft) WHERE id(ft) = {node_id} RETURN ft"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    FTs = []
    for iRslt in Rslt:
        iProperties = dict(iRslt[0])
        iFTID, iTableName = iRslt[0].id, iProperties["Name"]
        if "_PickledObject" in iProperties:
            iFT = iProperties.pop("_PickledObject")
            try:
                iFT = pickle.loads(iFT)
            except Exception as e:
                print(e)
            else:
                iArgs = readArgs(None, node_id=iFTID, tx=tx)
                for iArgName, iVal in enumerate(iArgs.items()):
                    iFT[iArgName] = iVal
                FTs.append(iFT)
                continue
        iArgs = readArgs(None, iFTID, tx=tx)
        if "库因子表" in iRslt[0].labels:# 库因子表, 构造库
            CypherStr = f"MATCH (ft:`因子表`) - [:`属于因子库`] -> (fdb:`因子库`) WHERE id(ft)={iFTID} RETURN id(fdb)"
            iFDBID = tx.run(CypherStr).values()[0][0]
            iFDB = readFactorDB(node_id=iFDBID, tx=tx, **kwargs)
            iFT = iFDB.getTable(iTableName, args=iArgs)
        else:# 自定义因子表
            CypherStr = f"MATCH (ft:`因子表`) - [:`包含因子`] -> (f:`因子`) WHERE id(ft)={iFTID} RETURN collect(DISTINCT id(f))"
            iFactorIDs = tx.run(CypherStr).values()[0][0]
            iFactors = readFactor(node_ids=iFactorIDs, tx=tx, **kwargs)
            iFT = CustomFT(iTableName, sys_args=iArgs, **kwargs)
            iFT.addFactors(factor_list=iFactors)
        FTs.append(iFT)
    if len(FTs)>1:
        return FTs
    else:
        return FTs[0]

# 读取因子
def readFactor(labels=["因子"], properties={}, node_ids=None, tx=None, id_fdb={}, id_ft={}, id_factor={}, **kwargs):
    if node_ids is None:
        Constraint = ','.join(((f"`{iKey}` : '{iVal}'" if isinstance(iVal, str) else f"`{iKey}` : {iVal}") for iKey, iVal in properties.items()))
        CypherStr = f"MATCH (f:`{'`:`'.join(labels)}` {{{Constraint}}}) RETURN f" 
    else:
        CypherStr = f"MATCH (f) WHERE id(f) IN {str(node_ids)} RETURN f"
    if tx is None: return CypherStr
    Rslt = tx.run(CypherStr).values()
    if not Rslt: return None
    Factors = []
    for iRslt in Rslt:
        iProperties = dict(iRslt[0])
        iFactorID, iFactorName = iRslt[0].id, iProperties["Name"]
        if iFactorID in id_factor:
            Factors.append(id_factor[iFactorID])
            continue
        if "_PickledObject" in iProperties:
            iFactor = iProperties.pop("_PickledObject")
            try:
                iFactor = pickle.loads(iFactor)
            except Exception as e:
                print(e)
            else:
                iArgs = readArgs(None, node_id=iFactorID, tx=tx)
                for iArgName, iVal in enumerate(iArgs.items()):
                    iFactor[iArgName] = iVal
                id_factor[iFactorID] = iFactor
                Factors.append(iFactor)
                continue
        if "基础因子" in iRslt[0].labels:# 基础因子, 构造表和库
            CypherStr = f"MATCH (f:`因子`) - [r:`产生于`|`属于因子表`] -> (ft:`因子表`) - [:`属于因子库`] -> (fdb:`因子库`) WHERE id(f)={iFactorID} RETURN r.`原始因子`, id(ft), ft.Name, id(fdb)"
            iRawName, iFTID, iFTName, iFDBID = tx.run(CypherStr).values()[0]
            if iFDBID in id_fdb:
                iFDB = id_fdb[iFDBID]
            else:
                id_fdb[iFDBID] = iFDB = readFactorDB(node_id=iFDBID, tx=tx, **kwargs)
            if iFTID in id_ft:
                iFT = id_ft[iFTID]
            else:
                id_ft[iFTID] = iFT = iFDB.getTable(iFTName)
            iArgs = readArgs(None, iFactorID, tx=tx)
            id_factor[iFactorID] = iFactor = iFT.getFactor(iRawName, args=iArgs, new_name=iFactorName)
        else:# 衍生因子
            CypherStr = f"MATCH (f:`因子`) - [r:`依赖`] -> (d:`因子`) WHERE id(f)={iFactorID} RETURN id(d) ORDER BY r.`Order`"
            iDescriptorIDs = [iRslt[0] for iRslt in tx.run(CypherStr).values()]
            iDescriptors = readFactor(node_ids=iDescriptorIDs, tx=tx, id_fdb=id_fdb, id_ft=id_ft, id_factor=id_factor, **kwargs)
            iArgs = readArgs(None, iFactorID, tx=tx)
            iClass = iProperties["_Class"].split(".")
            iModule = importlib.import_module('.'.join(iClass[:-1]))
            iFactorClass = getattr(iModule, iClass[-1])            
            id_factor[iFactorID] = iFactor = iFactorClass(iFactorName, iDescriptors, sys_args=iArgs, **kwargs)
        Factors.append(iFactor)
    return Factors