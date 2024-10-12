# coding=utf-8
import gc
import time
import html
import datetime as dt
from collections import OrderedDict
import concurrent.futures
from multiprocessing import Queue

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from traits.api import Instance, Str, List, Int, Enum, Dict

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.FactorDataBase import __QS_BatchContext__
from QuantStudio.FactorDataBase.FactorCache import FactorCache
from QuantStudio.Tools.api import Panel
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio.Tools.AuxiliaryFun import startMultiProcess, partitionListMovingSampling, partitionList
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.DataTypeConversionFun import dict2html


# 因子库, 只读, 接口类
# 数据库由若干张因子表组成
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorDB(__QS_Object__):
    """因子库"""
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name = Str("因子库", arg_type="String", label="名称", order=-100)
    @property
    def Name(self):
        return self._QSArgs.Name
    # ------------------------------数据源操作---------------------------------
    # 链接到数据库
    def connect(self):
        return self
    # 断开到数据库的链接
    def disconnect(self):
        return 0
    # 检查数据库是否可用
    def isAvailable(self):
        return True
    # -------------------------------表的操作---------------------------------
    # 表名, 返回: [表名]
    @property
    def TableNames(self):
        return []
    # 返回因子表对象
    def getTable(self, table_name, args={}):
        return None
    def __getitem__(self, table_name):
        return self.getTable(table_name)
    def _repr_html_(self):
        return f"<b>名称</b>: {html.escape(self.Name)}<br/>" + super()._repr_html_()
    
    def equals(self, other):
        if self is other: return True
        if not isinstance(other, FactorDB): return False
        if not (isinstance(other, type(self)) or isinstance(self, type(other))): return False
        if self._QSArgs != other._QSArgs: return False
        return True


# 支持写入的因子库, 接口类
class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""
    # -------------------------------表的操作---------------------------------
    # 重命名表. 必须具体化
    def renameTable(self, old_table_name, new_table_name):
        raise NotImplementedError
    # 删除表. 必须具体化
    def deleteTable(self, table_name):
        raise NotImplementedError
    # 设置表的元数据. 必须具体化
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        raise NotImplementedError
    # 设置因子定义代码
    # if_exists: append, update
    def setFactorDef(self, table_name, def_file, if_exists="update"):
        raise NotImplementedError
    # --------------------------------因子操作-----------------------------------
    # 对一张表的因子进行重命名. 必须具体化
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        raise NotImplementedError
    # 删除一张表中的某些因子. 必须具体化
    def deleteFactor(self, table_name, factor_names):
        raise NotImplementedError
    # 设置因子的元数据. 必须具体化
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        raise NotImplementedError
    # 写入数据, if_exists: append, update. data_type: dict like, {因子名:数据类型}, 必须具体化
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        raise NotImplementedError


# 因子表, 接口类
# 因子表可看做一个独立的数据集或命名空间, 可看做 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])
# 因子表的数据有三个维度: 时间点, ID, 因子
# 时间点数据类型是 datetime.datetime, ID 和因子名称的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorTable(__QS_Object__):
    """因子表"""
    
    def __init__(self, name, fdb=None, sys_args={}, config_file=None, **kwargs):
        self._Name = name
        self._FactorDB = fdb# 因子表所属的因子库, None 表示自定义的因子表
        self._QS_GroupArgs = None# 准备原始数据决定分组的参数集，如果为 None，表示每个因子单独分组
        self._QS_LookbackArgs = ("回溯天数",)
        self._QS_RawDataMaskCols = ["QS_ID", "QS_DT"]
        self._BatchContext = None# 批量运算运行时环境
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    
    @property
    def Name(self):
        return self._Name
    
    @property
    def FactorDB(self):
        return self._FactorDB
    
    @property
    def BatchContext(self):
        if self._BatchContext is not None:
            return self._BatchContext
        elif __QS_BatchContext__:
            return __QS_BatchContext__[-1]
        return None
    
    @BatchContext.setter
    def BatchContext(self, value):
        if not isinstance(value, BatchContext):
            raise __QS_Error__("BatchContext 必须是批量运算运行时环境对象")
        self._BatchContext = value
    
    # -------------------------------表的信息---------------------------------
    # 获取表的元数据
    def getMetaData(self, key=None, args={}):
        if key is None: return pd.Series()
        return None
    # -------------------------------维度信息-----------------------------------
    # 返回所有因子名
    @property
    def FactorNames(self):
        return []
    # 获取因子对象
    def getFactor(self, ifactor_name, args={}, new_name=None):
        Args = self._QSArgs.to_dict()
        Args.update(args)
        iFactor = Factor(name=ifactor_name, ft=self, sys_args={"因子表参数": Args}, logger=self._QS_Logger)
        if new_name is not None: iFactor.Name = new_name
        return iFactor
    
    # 获取因子的元数据
    def getFactorMetaData(self, factor_names, key=None, args={}):
        if key is None: return pd.DataFrame(index=factor_names, dtype=np.dtype("O"))
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 获取 ID 序列
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []
    # 获取 ID 的 Mask, 返回: Series(True or False, index=[ID])
    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None: ids = self.getID(idt=idt, args=args)
        if not id_filter_str: return pd.Series(True, index=ids)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval(CompiledIDFilterStr)
    # 获取过滤后的 ID
    def getFilteredID(self, idt, ids=None, id_filter_str=None, args={}):
        if not id_filter_str: return self.getID(idt=idt, args=args)
        if ids is None: ids = self.getID(idt=idt, args=args)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval("temp["+CompiledIDFilterStr+"].index.tolist()")
    # 获取时间点序列
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    
    # 准备原始数据的接口
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        return None
    
    # 计算数据的接口, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        return None
    
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, factor_names, ids, dts, args={}, **kwargs):
        Context = self.BatchContext
        if Context is None:
            return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args), factor_names=factor_names, ids=ids, dts=dts, args=args)
        return Context.readData(factors=[self.getFactor(iFactorName, args=args) for iFactorName in factor_names], ids=ids, dts=dts, **kwargs)
        
    def __getitem__(self, key):
        if isinstance(key, str): return self.getFactor(key)
        elif isinstance(key, tuple): key += (slice(None),) * (3 - len(key))
        else: key = (key, slice(None), slice(None))
        if len(key)>3: raise IndexError("QuantStudio.FactorDataBase.FactorDB.FactorTable: Too many indexers")
        FactorNames, DTs, IDs = key
        if FactorNames==slice(None): FactorNames = self.FactorNames
        elif isinstance(FactorNames, str): FactorNames = [FactorNames]
        if DTs==slice(None): DTs = None
        elif isinstance(DTs, dt.datetime): DTs = [DTs]
        if IDs==slice(None): IDs = None
        elif isinstance(IDs, str): IDs = [IDs]
        Data = self.readData(FactorNames, IDs, DTs)
        return Data.loc[key]
    
    def _repr_html_(self):
        HTML = f"<b>名称</b>: {html.escape(self.Name)}<br/>"
        HTML += f"<b>来源因子库</b>: {html.escape(self.FactorDB.Name) if self.FactorDB is not None else ''}<br/>"
        HTML += f"<b>因子列表</b>: {html.escape(str(self.FactorNames))}<br/>"
        MetaData = self.getMetaData()
        MetaData = MetaData[~MetaData.index.str.contains("_QS")]
        HTML += f"<b>元信息</b>: {dict2html(MetaData)}"
        return HTML + super()._repr_html_()
    
    def equals(self, other):
        if self is other: return True
        if not isinstance(other, FactorTable): return False
        if not (isinstance(other, type(self)) or isinstance(self, type(other))): return False
        if not self._FactorDB.equals(other._FactorDB): return False
        if not (self._Name != other._Name): return False
        if self._QSArgs != other._QSArgs: return False
        return True

    
    
    # -------------------------------批量运算环境---------------------------------
    # 获取因子表准备原始数据的分组信息, [{"FactorIDs": [因子 QSID], "FT": 因子表对象, "RawFactorNames": {原始因子名}, "DTRange": (起始时点, 结束时点), "SectionIDs": [ID], "Args": {参数}, "RawDataKey":str}]
    def __QSBC_genGroupInfo__(self, factors, context):
        if self._QS_GroupArgs is None:# 每个因子单独准备
            Groups = []
            for iFactor in factors:
                iRawDataKey = iFactor._QSID
                context._FactorRawDataKeys.setdefault(iFactor._QSID, iRawDataKey)
                iDTRange = context._DTRange[iFactor._QSID]
                if iDTRange is not None:
                    Groups.append({
                        "FactorIDs": [iFactor._QSID], 
                        "FT": self, 
                        "RawFactorNames": {iFactor._NameInFT}, 
                        "DTRange": iDTRange, 
                        "SectionIDs": context._FactorSectionIDs.get(iFactor._QSID, context._SectionIDs), 
                        "Args": iFactor._QSArgs.FTArgs,
                        "RawDataKey": iRawDataKey
                    })
            return Groups
        ConditionGroup = {}
        for iFactor in factors:
            iDTRange = context._DTRange[iFactor._QSID]
            if iDTRange is None: continue
            iConditions = ",".join([f"{iArgName}:{str(iFactor._QSArgs.FTArgs[iArgName])}" for iArgName in sorted(iFactor._QSArgs.FTArgs.keys()) if iArgName in self._QS_GroupArgs])
            iRawDataKey = str(hash((self._QSID, iConditions)))
            context._FactorRawDataKeys.setdefault(iFactor._QSID, iRawDataKey)
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {
                    "FactorIDs": [iFactor._QSID],
                    "FT": self,
                    "RawFactorNames": {iFactor._NameInFT},
                    "DTRange": iDTRange,
                    "SectionIDs": context._FactorSectionIDs.get(iFactor._QSID, context._SectionIDs),
                    "Args":iFactor._QSArgs.FTArgs,
                    "RawDataKey": iRawDataKey
                }
            else:
                iSectionIDs = context._FactorSectionIDs.get(iFactor._QSID, context._SectionIDs)
                if iSectionIDs != ConditionGroup[iConditions]["SectionIDs"]:
                    ConditionGroup[iConditions]["SectionIDs"] = set(ConditionGroup[iConditions]["SectionIDs"]).union(iSectionIDs)
                ConditionGroup[iConditions]["FactorIDs"].append(iFactor._QSID)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["DTRange"] = (min(iDTRange[0], ConditionGroup[iConditions]["DTRange"][0]), max(iDTRange[1], ConditionGroup[iConditions]["DTRange"][1]))
                for jLookbackArg in self._QS_LookbackArgs:
                    if jLookbackArg in ConditionGroup[iConditions]["Args"]:
                        ConditionGroup[iConditions]["Args"][jLookbackArg] = max(ConditionGroup[iConditions]["Args"][jLookbackArg], iFactor._QSArgs.FTArgs[jLookbackArg])
        return list(ConditionGroup.values())
        
    def __QSBC_saveRawData__(self, raw_data, key, target_fields, pid_ids, **kwargs):
        if (raw_data is None) or raw_data.empty: return 0
        Context = self.BatchContext
        Cache = Context._Cache
        MaskCols = raw_data.columns.intersection(self._QS_RawDataMaskCols).tolist()
        CommonCols = raw_data.columns.difference(target_fields).tolist()
        for iFactorName in target_fields:
            iRawData = raw_data.loc[:, CommonCols+[iFactorName]]
            iKey = key+"-"+iFactorName
            iOldData = Cache.readRawData(iKey, target_fields=None, pids=None)
            if iOldData:
                iOldData = iOldData["RawData"]
                iOldData["QS_Mask"] = 1
                iRawData = pd.merge(iRawData, iOldData.loc[:, [*MaskCols, "QS_Mask"]], how="left", left_on=MaskCols, right_on=MaskCols)
                iOldData.pop("QS_Mask")
                iRawData = pd.concat([iOldData, iRawData[iRawData.pop("QS_Mask").isnull()]], ignore_index=True).sort_values(MaskCols)
            Cache.writeRawData(iKey, {"RawData": iRawData}, pid_ids, id_col="QS_ID", if_exists="replace")
    
    
# 自定义因子表
class CustomFT(FactorTable):
    """自定义因子表"""
    
    class __QS_ArgClass__(FactorTable.__QS_ArgClass__):
        MetaData = Dict({}, arg_type="Dict", label="元信息", order=-6, eq_arg=False)
    
    def __init__(self, name, sys_args={}, config_file=None, **kwargs):
        self._DateTimes = []# 数据源可提取的最长时点序列，[datetime.datetime]
        self._IDs = []# 数据源可提取的最长ID序列，['600000.SH']
        self._Factors = {}# 因子对象, {因子名: 因子对象}
        
        self._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))# 数据源中因子的来源信息, index=[因子名]
        self._TableArgDict = {}# 数据源中的表和参数信息, {id(FT) : (FT, [args]), id(None) : ([Factor], [args])}
        
        self._IDFilterStr = None# ID 过滤条件字符串, "@收益率>0", 给定日期, 数据源的 getID 将返回过滤后的 ID 序列
        self._CompiledIDFilter = {}# 编译过的过滤条件字符串以及对应的因子列表, {条件字符串: (编译后的条件字符串,[因子])}
        self._isStarted = False# 数据源是否启动
        return super().__init__(name=name, fdb=None, sys_args=sys_args, config_file=config_file, **kwargs)
    
    @property
    def FactorNames(self):
        return sorted(self._Factors)
    def getMetaData(self, key=None, args={}):
        if key is None: return pd.Series(self._QSArgs.MetaData)
        return self._QSArgs.MetaData.get(key, None)
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if key is not None: return pd.Series({iFactorName: self._Factors[iFactorName].getMetaData(key) for iFactorName in factor_names})
        else: return pd.DataFrame({iFactorName: self._Factors[iFactorName].getMetaData(key) for iFactorName in factor_names}).T
    def getFactor(self, ifactor_name, args={}, new_name=None):
        iFactor = self._Factors[ifactor_name]
        if new_name is not None: iFactor.Name = new_name
        return iFactor
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if (start_dt is not None) or (end_dt is not None):
            DateTimes = np.array(self._DateTimes, dtype="O")
            if start_dt is not None: DateTimes = DateTimes[DateTimes>=start_dt]
            if end_dt is not None: DateTimes = DateTimes[DateTimes<=end_dt]
            return DateTimes.tolist()
        else:
            return self._DateTimes
    def getID(self, ifactor_name=None, idt=None, args={}):
        return self._IDs
    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None: ids = self.getID(idt=idt, args=args)
        OldIDFilterStr = self.setIDFilter(id_filter_str)
        if self._IDFilterStr is None:
            self._IDFilterStr = OldIDFilterStr
            return pd.Series(True, index=ids)
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        self._IDFilterStr = OldIDFilterStr
        return eval(CompiledFilterStr)
    def getFilteredID(self, idt, ids=None, id_filter_str=None, args={}):
        OldIDFilterStr = self.setIDFilter(id_filter_str)
        if ids is None: ids = self.getID(idt=idt, args=args)
        if self._IDFilterStr is None:
            self._IDFilterStr = OldIDFilterStr
            return ids
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        if CompiledFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        self._IDFilterStr = OldIDFilterStr
        return eval("temp["+CompiledFilterStr+"].index.tolist()")
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        return Panel({iFactorName:self._Factors[iFactorName].readData(ids=ids, dts=dts, dt_ruler=self._DateTimes, section_ids=self._IDs) for iFactorName in factor_names}, items=factor_names, major_axis=dts, minor_axis=ids)
    
    def equals(self, other):
        if self is other: return True
        if not super().equals(other): return False
        if self._DateTimes != other._DateTimes: return False
        if self._IDs != other._IDs: return False
        if self._Factors != other._Factors: return False
        return True
    
    # ---------------新的接口------------------
    # 添加因子, factor_list: 因子对象列表
    def addFactors(self, factor_list=[], factor_table=None, factor_names=None, replace=True, args={}):
        if replace:
            FactorNames = {iFactor.Name for iFactor in factor_list}
            if factor_table is not None:
                FactorNames = set(FactorNames).union(factor_table.FactorNames if factor_names is None else factor_names)
            FactorNames = sorted(FactorNames.intersection(self.FactorNames))
            if FactorNames: self.deleteFactors(factor_names=FactorNames)
        for iFactor in factor_list:
            if iFactor.Name in self._Factors:
                raise __QS_Error__("因子: '%s' 有重名!" % iFactor.Name)
            self._Factors[iFactor.Name] = iFactor
        if factor_table is None: return 0
        if factor_names is None: factor_names = factor_table.FactorNames
        for iFactorName in factor_names:
            if iFactorName in self._Factors: raise __QS_Error__("因子: '%s' 有重名!" % iFactorName)
            iFactor = factor_table.getFactor(iFactorName, args=args)
            self._Factors[iFactor.Name] = iFactor
        return 0
    # 删除因子, factor_names = None 表示删除所有因子
    def deleteFactors(self, factor_names=None):
        if factor_names is None: factor_names = self.FactorNames
        for iFactorName in factor_names:
            if iFactorName not in self._Factors: continue
            self._Factors.pop(iFactorName, None)
        return 0
    # 重命名因子
    def renameFactor(self, factor_name, new_factor_name):
        if factor_name not in self._Factors: raise __QS_Error__("因子: '%s' 不存在!" % factor_name)
        if (new_factor_name!=factor_name) and (new_factor_name in self._Factors): raise __QS_Error__("因子: '%s' 有重名!" % new_factor_name)
        self._Factors[new_factor_name] = self._Factors.pop(factor_name)
        return 0
    # 设置时间点序列
    def setDateTime(self, dts):
        self._DateTimes = sorted(dts)
    # 设置 ID 序列
    def setID(self, ids):
        self._IDs = sorted(ids)
    # ID 过滤条件
    @property
    def IDFilterStr(self):
        return self._IDFilterStr
    # 设置 ID 过滤条件, id_filter_str, '@收益率$>0'
    def setIDFilter(self, id_filter_str):
        OldIDFilterStr = self._IDFilterStr
        if not id_filter_str:
            self._IDFilterStr = None
            return OldIDFilterStr
        elif not isinstance(id_filter_str, str): raise __QS_Error__("条件字符串必须为字符串或者为 None!")
        CompiledIDFilter = self._CompiledIDFilter.get(id_filter_str, None)
        if CompiledIDFilter is not None:# 该条件已经编译过
            self._IDFilterStr = id_filter_str
            return OldIDFilterStr
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__(f"条件字符串有误: {id_filter_str}")
        self._IDFilterStr = id_filter_str
        self._CompiledIDFilter[id_filter_str] = (CompiledIDFilterStr, IDFilterFactors)
        return OldIDFilterStr


# 因子
# 因子可看做一个 DataFrame(index=[时间点], columns=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class Factor(__QS_Object__):
    """因子"""
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        FTArgs = Dict(arg_type="Dict", label="因子表参数", order=-2, mutable=False)
        Meta = Dict(arg_type="Dict", label="元信息", order=-1)
        
    def __init__(self, name, ft, sys_args={}, config_file=None, **kwargs):
        self._FactorTable = ft# 因子所属的因子表, None 表示衍生因子
        self._NameInFT = name# 因子在所属的因子表中的名字
        self.Name = name# 因子对外显示的名称
        self._BatchContext = None# 批量运算运行时环境
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    
    @property
    def FactorTable(self):
        return self._FactorTable
    
    @property
    def Descriptors(self):
        return []
    
    @property
    def BatchContext(self):
        if self._BatchContext is not None:
            return self._BatchContext
        elif __QS_BatchContext__:
            return __QS_BatchContext__[-1]
        return None
    
    @BatchContext.setter
    def BatchContext(self, value):
        if not isinstance(value, BatchContext):
            raise __QS_Error__("BatchContext 必须是批量运算运行时环境对象")
        self._BatchContext = value
    
    # 获取因子的元数据
    def getMetaData(self, key=None):
        return self._FactorTable.getFactorMetaData(factor_names=[self._NameInFT], key=key, args=self._QSArgs.FTArgs).loc[self._NameInFT]
    
    # 获取 ID 序列
    def getID(self, idt=None):
        if self._FactorTable is not None: return self._FactorTable.getID(ifactor_name=self._NameInFT, idt=idt, args=self._QSArgs.FTArgs)
        return []
    
    # 获取时间点序列
    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        if self._FactorTable is not None: return self._FactorTable.getDateTime(ifactor_name=self._NameInFT, iid=iid, start_dt=start_dt, end_dt=end_dt, args=self._QSArgs.FTArgs)
        return []
    
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, ids, dts, **kwargs):
        Context = self.BatchContext
        if Context is None:
            return self._FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, args=self._QSArgs["因子表参数"]).iloc[0]
        return Context.readData(factors=[self], ids=ids, dts=dts, **kwargs).iloc[0]
    
    def __getitem__(self, key):
        if isinstance(key, tuple): key += (slice(None),) * (2 - len(key))
        else: key = (key, slice(None))
        if len(key)>2: raise IndexError("QuantStudio.FactorDataBase.FactorDB.Factor: Too many indexers")
        DTs, IDs = key
        if DTs==slice(None): DTs = None
        elif isinstance(DTs, dt.datetime): DTs = [DTs]
        if IDs==slice(None): IDs = None
        elif isinstance(IDs, str): IDs = [IDs]
        Data = self.readData(IDs, DTs)
        return Data.loc[key]
    
    def _repr_html_(self):
        HTML = f"<b>名称</b>: {html.escape(self.Name)}<br/>"
        HTML += f"<b>来源因子表</b>: {html.escape(self.FactorTable.Name) if self.FactorTable is not None else ''}<br/>"
        HTML += f"<b>原始名称</b>: {html.escape(self._NameInFT) if self.FactorTable is not None else ''}<br/>"
        HTML += f"<b>描述子列表</b>: {html.escape(str([iFactor.Name for iFactor in self.Descriptors]))}<br/>"
        MetaData = self.getMetaData()
        MetaData = MetaData[~MetaData.index.str.contains("_QS")]
        HTML += f"<b>元信息</b>: {dict2html(MetaData)}"
        return HTML + super()._repr_html_()
    
    def equals(self, other):
        if self is other: return True
        if not isinstance(other, Factor): return False
        if not (isinstance(other, type(self)) or isinstance(self, type(other))): return False
        if not self._FactorTable.equals(other._FactorTable): return False
        if not (self._NameInFT != other._NameInFT): return False
        if self._QSArgs != other._QSArgs: return False
        if len(self.Descriptors) != len(other.Descriptors): return False
        for i, iDescriptor in enumerate(self.Descriptors):
            if iDescriptor != other.Descriptors[i]:
                return False
        return True
    
    # -----------------------------批量运算环境-------------------------------------
    # 初始化，生成必备的计算信息, context: 批量计算运行时环境, dt_range: 新的起始结束时点, (StartDT, EndDT), prepare_ids: 新的截面 ID 序列
    def __QSBC_initOperation__(self, context, dt_range, section_ids):
        DTRange = context._DTRange.get(self._QSID, None)
        if DTRange is None:
            context._DTRange[self.QSID] = dt_range
        else:
            context._DTRange[self.QSID] = (min(DTRange[0], dt_range[0]), max(DTRange[1], dt_range[1]))
        SectionIDs = context._FactorSectionIDs.setdefault(self._QSID, section_ids)
        if section_ids != SectionIDs:
            raise __QS_Error__("因子 %s (QSID: %s) 指定了不同的截面!" % (self.Name, self.QSID))
    
    # 准备缓存数据
    def __QSBC_prepareCacheData__(self, dt_range):
        Context = self.BatchContext
        DTs = Context.getDateTime(dt_range)
        if not DTs: return 0
        RawKey = Context._FactorRawDataKeys.get(self._QSID, None)
        PIDIDs = Context.getPIDID(self._QSID)
        iSectionIDs = PIDIDs[Context._iPID]
        if RawKey is None: 
            RawData = None
        else:
            RawData = Context._Cache.readRawData(key=RawKey + "-" + self._NameInFT, target_fields=None, pids=[Context._iPID])
        if RawData:
            if len(RawData)==1: RawData = RawData["RawData"]
            StdData = self._FactorTable.__QS_calcData__(RawData, factor_names=[self._NameInFT], ids=iSectionIDs, dts=DTs, args=self._QSArgs.FTArgs).iloc[0]
        else:
            StdData = self._FactorTable.readData(factor_names=[self._NameInFT], ids=iSectionIDs, dts=DTs, args=self._QSArgs.FTArgs).iloc[0]
        Context._Cache.writeFactorData(key=self._QSID, target_field="StdData", factor_data=StdData, pid_ids=PIDIDs, pid=Context._iPID, if_exists="append")
        Context.updateDTRange(factor_id=self._QSID, dt_range=dt_range)
        Context._Cache.writeFactorData(key=self._QSID, target_field="DTRange", factor_data=Context._CachedDTRange[self._QSID], pid_ids=None, pid=Context._iPID, if_exists="replace")
        return 0
    
    # 获取因子数据, pid=None表示取所有进程的数据
    def __QSBC_getData__(self, dts, pids=None, **kwargs):
        Context = self.BatchContext
        DTRange = Context.getDTRange(self._QSID, (dts[0], dts[-1]))
        if DTRange is not None:# 需要准备缓存数据
            self.__QSBC_prepareCacheData__(DTRange)
        StdData = Context.Cache.readFactorData(key=self._QSID, target_field="StdData", pids=pids)
        IDs = Context.getID(self._QSID, pids=pids)
        return StdData.reindex(index=dts, columns=IDs)
    
    # -----------------------------重载运算符-------------------------------------
    def __add__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import add
        return add(self, other)
    
    def __radd__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import add
        return add(other, self)
    
    def __sub__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import sub
        return sub(self, other)
    
    def __rsub__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import sub
        return sub(other, self)
    
    def __mul__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import mul
        return mul(self, other)
    
    def __rmul__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import mul
        return mul(other, self)
    
    def __pow__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import qs_pow
        return qs_pow(self, other)
    
    def __rpow__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import qs_pow
        return qs_pow(other, self)
    
    def __truediv__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import div
        return div(self, other)
    
    def __rtruediv__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import div
        return div(other, self)
    
    def __floordiv__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import floordiv
        return floordiv(self, other)
    
    def __rfloordiv__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import floordiv
        return floordiv(other, self)
    
    def __mod__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import mod
        return mod(self, other)
        
    def __rmod__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import mod
        return mod(other, self)
    
    def __and__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import qs_and
        return qs_and(self, other)
        
    def __rand__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import qs_and
        return qs_and(other, self)
    
    def __or__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import qs_or
        return qs_or(self, other)        
    
    def __ror__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import qs_or
        return qs_or(other, self)
    
    def __xor__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import xor
        return xor(self, other)
        
    def __rxor__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import xor
        return xor(other, self)
    
    def __lt__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import lt
        return lt(self, other)
    
    def __le__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import le
        return le(self, other)
    
    def __eq__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import eq
        return eq(self, other)
    
    def __ne__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import neq
        return neq(self, other)
    
    def __gt__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import gt
        return gt(self, other)
    
    def __ge__(self, other):
        from QuantStudio.FactorDataBase.BasicOperators import ge
        return ge(self, other)
    
    def __neg__(self):
        from QuantStudio.FactorDataBase.BasicOperators import neg
        return neg(self)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        from QuantStudio.FactorDataBase.BasicOperators import qs_abs
        return qs_abs(self)
    
    def __invert__(self):
        from QuantStudio.FactorDataBase.BasicOperators import qs_not
        return qs_not(self)


# 直接赋予数据产生的因子
# data: DataFrame(index=[时点], columns=[ID])
class DataFactor(Factor):
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=0, option_range=("double", "string", "object"))
        LookBack = Int(0, arg_type="Integer", label="回溯天数", order=1)
    
    def __init__(self, name, data, sys_args={}, config_file=None, **kwargs):
        if  isinstance(data, pd.Series):
            if data.index.is_all_dates:
                self._DataContent = "DateTime"
            else:
                self._DataContent = "ID"
            if "数据类型" not in sys_args:
                try:
                    data = data.astype(float)
                except:
                    sys_args["数据类型"] = "object"
                else:
                    sys_args["数据类型"] = "double"
        elif isinstance(data, pd.DataFrame):
            self._DataContent = "Factor"
            if "数据类型" not in sys_args:
                try:
                    data = data.astype(float)
                except:
                    sys_args["数据类型"] = "object"
                else:
                    sys_args["数据类型"] = "double"
        else:
            self._DataContent = "Value"
            if "数据类型" not in sys_args:
                if isinstance(data, str): sys_args["数据类型"] = "string"
                else:
                    try:
                        data = float(data)
                    except:
                        sys_args["数据类型"] = "object"
                    else:
                        sys_args["数据类型"] = "double"
        self._Data = data
        return super().__init__(name=name, ft=None, sys_args=sys_args, config_file=None, **kwargs)
    
    @property
    def Data(self):
        return self._Data
    
    @property
    def DataContent(self):
        return self._DataContent
    
    def getMetaData(self, key=None, args={}):
        DataType = args.get("数据类型", self._QSArgs.DataType)
        if key is None: return pd.Series({"DataType": DataType})
        elif key=="DataType": return DataType
        return None
    def getID(self, idt=None):
        Context = self.BatchContext
        if Context is not None: return Context._QSArgs["截面ID"]
        if self._DataContent=="Factor":
            return self._Data.columns.tolist()
        elif self._DataContent=="ID":
            return self._Data.index.tolist()
        else:
            return []
    
    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        Context = self.BatchContext
        if Context is not None: return Context._QSArgs["时点标尺"]
        if self._DataContent in ("DateTime", "Factor"):
            return self._Data.index.tolist()
        else:
            return []
    
    def readData(self, ids, dts, **kwargs):
        if self._DataContent=="Value":
            if self._QSArgs.DataType in ("double", "string"):
                return pd.DataFrame(self._Data, index=dts, columns=ids)
            else:
                return pd.DataFrame([(self._Data,)*len(ids)]*len(dts), index=dts, columns=ids)
        elif self._DataContent=="ID":
            Data = pd.DataFrame(self._Data.values.reshape((1, self._Data.shape[0])).repeat(len(dts), axis=0), index=dts, columns=self._Data.index)
        elif self._DataContent=="DateTime":
            Data = pd.DataFrame(self._Data.values.reshape((self._Data.shape[0], 1)).repeat(len(ids), axis=1), index=self._Data.index, columns=ids)
        else:
            Data = self._Data
        if Data.columns.intersection(ids).shape[0]==0:
            return pd.DataFrame(index=dts, columns=ids, dtype=("O" if self._QSArgs.DataType!="double" else float))
        if self._QSArgs.LookBack==0: return Data.reindex(index=dts, columns=ids)
        else: return fillNaByLookback(Data.reindex(index=sorted(Data.index.union(dts)), columns=ids), lookback=self._QSArgs.LookBack*24.0*3600).loc[dts, :]
    
    def equals(self, other):
        if self is other: return True
        if not super().equals(other): return False
        if self._DataContent != other._DataContent: return False
        if not (isinstance(self._Data, type(other._Data)) or isinstance(other._Data, type(self._Data))): return False
        if isinstance(self._Data, (pd.DataFrame, pd.Series)) and (not self._Data.equals(other._Data)): return False
        return (self._Data == other._Data)
    
    def __QSBC_prepareCacheData__(self, ids=None):
        return self._Data
    
    def __QSBC_getData__(self, dts, pids=None, **kwargs):
        IDs = self.BatchContext.getID(self._QSID, pids=pids)
        return self.readData(ids=IDs, dts=dts)


# 为因子表达式产生的因子更改其他信息
def Factorize(factor_object, factor_name, args={}, **kwargs):
    factor_object.Name = factor_name
    for iArg, iVal in args.items(): factor_object._QSArgs[iArg] = iVal
    if "logger" in kwargs: factor_object._QS_Logger = kwargs["logger"]
    return factor_object


# 批量运算运行时环境
class BatchContext(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        DTRuler = List(dt.datetime, arg_type="DateTimeList", label="时点标尺", order=0)
        MinDTUnit = Instance(klass=dt.timedelta, factory=dt.timedelta(1), arg_type="TimeDelta", label="最小时间单位", order=1)
        DefaultSectionIDs = List(Str, arg_type="IDList", label="截面ID", order=1.5)
        FactorSectionIDs = Dict(arg_type="IDList", label="特别截面ID", order=2)# {因子对象的 QSID: [ID]}
        IOConcurrentNum = Int(100, arg_type="Integer", label="IO并发数", order=3)
        CalcConcurrentNum = Int(0, arg_type="Integer", label="计算并发数", order=4)
        IDSplit = Enum("连续切分", "间隔切分", arg_type="SingleOption", label="ID切分", order=5)
        WriteBatchNum = Int(1, arg_type="Integer", label="写入批次", order=6)
        Verbose = Enum(True, False, arg_type="Bool", label="打印信息", order=7)
        ClearCache = Enum(True, False, arg_type="Bool", label="清空缓存", order=8)
    
    def __init__(self, cache:FactorCache, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        self._isStarted = False# 是否开始
        
        # 多次运行的相关信息
        self._FactorDict = {}# 因子字典, {因子 QSID: 因子}, 包括所有的因子, 即衍生因子所依赖的描述子也在内
        self._CachedDTRange = {}# 已经缓存的因子数据时点范围, {因子 QSID: DataFrame(columns=["StartDT", "EndDT"])}
        self._FactorSectionIDs = self._QSArgs.FactorSectionIDs.copy()
        self._FactorRawDataKeys = {}# {因子 QSID: (因子表 QSID, ConditionStr, RawFactorName)}
        self._Cache = cache# 因子缓存对象
        
        # 单次运行的相关信息
        self._Factors = []# 因子列表, 当前生成数据的因子
        self._iFactorDict = {}# 当前运行所涉及到的因子, {因子 QSID: 因子}
        self._DTRange = {}# 当前运行需要计算的时点范围
        self._SectionIDs = None# 当前运行需要计算的截面 ID
        self._RawFactorIDs = []# 当前基本因子 QSID 列表, 即需要准备原始数据的因子
        self._DateTimes = None# 当前运行的时点序列
        self._IDs = None# 当前运行的 ID 序列
        self._iPID = "0"# 对象所在的进程 ID
        self._PIDs = []# 所有的计算进程 ID, 单进程下默认为"0", 多进程为"0-i"
        self._PID_IDs = {}# 每个计算进程分配的 ID 列表, {PID: [ID]}
        self._Event = {}# {因子 QSID: (Sub2MainQueue, Event)}, 用于多进程同步的 Event 数据
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._isStarted and (self not in __QS_BatchContext__):
            __QS_BatchContext__.append(self)
    
    @property
    def Cache(self):
        return self._Cache
    
    def start(self):
        self._isStarted = True
        if self._QSArgs.CalcConcurrentNum == 0:# 串行模式
            self._PIDs = [self._iPID]
        else:
            self._PIDs = [f"{self._iPID}-{i}" for i in range(self._QSArgs.CalcConcurrentNum)]
        self._Cache._QSArgs.PIDs = self._PIDs
        self._Cache.start()
        __QS_BatchContext__.append(self)
    
    def end(self):
        self._isStarted = False
        __QS_BatchContext__.remove(self)
        if self._QSArgs.ClearCache:
            self._Cache.clearFactorData()
            self._Cache.clearRawData()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
        return (exc_type is None)
    
    def _genFactorDict(self, factors, factor_dict):
        for iFactor in factors:
            factor_dict[iFactor.QSID] = iFactor
            factor_dict.update(self._genFactorDict(iFactor.Descriptors, factor_dict))
            if iFactor.FactorTable is not None: self._RawFactorIDs.add(iFactor.QSID)
        return factor_dict
    
    # 获取给定因子真正需要计算的时点范围
    def getDTRange(self, factor_id, dt_range):
        CachedDTRange = self._CachedDTRange.get(factor_id, None)
        if CachedDTRange is None:
            return dt_range
        else:
            StartIdx = CachedDTRange[(CachedDTRange["StartDT"]<=dt_range[0]) & (CachedDTRange["EndDT"]>=dt_range[0])]
            EndIdx = CachedDTRange[(CachedDTRange["StartDT"]<=dt_range[1]) & (CachedDTRange["EndDT"]>=dt_range[1])]
            if StartIdx.empty and EndIdx.empty:# 新区间起始结束点均在空档里
                return dt_range
            elif (not StartIdx.empty) and (not EndIdx.empty):# 新区间起始结束点均在已有区间里
                StartIdx, EndIdx = StartIdx.index[0], EndIdx.index[0]
                if StartIdx==EndIdx:# 已有区间完全覆盖新区间
                    return None
                else:# 新区间跨区间
                    return dt_range
            elif StartIdx.empty and (not EndIdx.empty):# 新区间起始点在空档里, 结束点在已有区间里
                EndDT = EndIdx["StartDT"].iloc[0] - self._QSArgs.MinDTUnit
                return (dt_range[0], EndDT)
            else:# 新区间起始点在已有区间里, 结束点在空档里
                StartDT = StartIdx["EndDT"].iloc[0] + self._QSArgs.MinDTUnit
                return (StartDT, dt_range[1])
    
    def updateDTRange(self, factor_id, dt_range):
        CachedDTRange = self._CachedDTRange.get(factor_id, None)
        if CachedDTRange is None:
            self._CachedDTRange[factor_id] = pd.DataFrame([dt_range], columns=["StartDT", "EndDT"])
            return
        StartIdx = CachedDTRange[(CachedDTRange["StartDT"]<=dt_range[0]) & (CachedDTRange["EndDT"]>=dt_range[0]-self._QSArgs.MinDTUnit)]
        EndIdx = CachedDTRange[(CachedDTRange["StartDT"]<=dt_range[1]+self._QSArgs.MinDTUnit) & (CachedDTRange["EndDT"]>=dt_range[1])]
        nRange = CachedDTRange.shape[0]
        if StartIdx.empty and EndIdx.empty:# 新区间起始结束点均在空档里
            CachedDTRange = CachedDTRange[~((CachedDTRange["StartDT"]>=dt_range[0]) & (CachedDTRange["EndDT"]<=dt_range[1]))]
            CachedDTRange.loc[nRange] = dt_range
            self._CachedDTRange[factor_id] = CachedDTRange.sort_values(["StartDT"], ignore_index=True)
        elif (not StartIdx.empty) and (not EndIdx.empty):# 新区间起始结束点均在已有区间里
            StartIdx, EndIdx = StartIdx.index[0], EndIdx.index[0]
            if StartIdx==EndIdx:# 已有区间完全覆盖新区间
                return
            else:# 新区间跨区间
                StartDT, EndDT = CachedDTRange.at[StartIdx, "StartDT"], CachedDTRange.at[EndIdx, "EndDT"]
                CachedDTRange = CachedDTRange[(CachedDTRange.index<StartIdx) | (CachedDTRange.index>EndIdx)]
                CachedDTRange.loc[nRange] = (StartDT, EndDT)
                self._CachedDTRange[factor_id] = CachedDTRange.sort_values(["StartDT"], ignore_index=True)
        elif StartIdx.empty and (not EndIdx.empty):# 新区间起始点在空档里, 结束点在已有区间里
            EndDT = EndIdx["EndDT"].iloc[0]
            CachedDTRange = CachedDTRange[~((CachedDTRange["StartDT"]>=dt_range[0]) & (CachedDTRange["EndDT"]<=EndDT))]
            CachedDTRange.loc[nRange] = (dt_range[0], EndDT)
            self._CachedDTRange[factor_id] = CachedDTRange.sort_values(["StartDT"], ignore_index=True)
        else:# 新区间起始点在已有区间里, 结束点在空档里
            StartDT = StartIdx["StartDT"].iloc[0]
            CachedDTRange = CachedDTRange[~((CachedDTRange["StartDT"]>=StartDT) & (CachedDTRange["EndDT"]<=dt_range[1]))]
            CachedDTRange.loc[nRange] = (StartDT, dt_range[1])
            self._CachedDTRange[factor_id] = CachedDTRange.sort_values(["StartDT"], ignore_index=True)
    
    def getDateTime(self, dt_range):
        StartIdx, EndIdx = np.searchsorted(self._QSArgs.DTRuler, dt_range[0], side="left"), np.searchsorted(self._QSArgs.DTRuler, dt_range[1], side="right")
        return self._QSArgs.DTRuler[StartIdx:EndIdx]
    
    # 划分 ID
    def splitID(self, ids):
        nPrcs = self._QSArgs.CalcConcurrentNum
        if nPrcs<=0: return [ids]
        if self._QSArgs.IDSplit == "连续切分":
            SubIDs = partitionList(ids, nPrcs)
        elif self._QSArgs.IDSplit == "间隔切分":
            SubIDs = partitionListMovingSampling(ids, nPrcs)
        else:
            raise __QS_Error__(f"不支持的 ID 切分方式: {self._QSArgs.IDSplit}")
        return SubIDs
    
    def getPIDID(self, factor_id):
        SectionIDs = self._FactorSectionIDs.get(factor_id, self._SectionIDs)
        if (SectionIDs is None) or (SectionIDs == self._SectionIDs):
            return self._PID_IDs
        else:
            ParentPID = "-".join(self._iPID.split("-")[:-1])
            return {f"{ParentPID}-{i}": iIDs for i, iIDs in enumerate(self.splitID(SectionIDs))}
    
    def getID(self, factor_id, pids=None):
        SectionIDs = self._FactorSectionIDs.get(factor_id, self._SectionIDs)
        if pids is not None:
            PIDIDs = self.getPIDID(factor_id=factor_id)
            return sorted(sum((PIDIDs[iPID] for iPID in pids), []))
        elif SectionIDs is None:
            return self._SectionIDs
        else:
            return SectionIDs
    
    # 单次运算开始前的初始化
    def _initContext(self, **kwargs):
        # 检查时点, ID 序列的合法性
        if not self._DateTimes: raise __QS_Error__("运算时点序列不能为空!")
        if not self._IDs: raise __QS_Error__("运算 ID 序列不能为空!")
        # 检查时点标尺是否合适
        DTs = pd.Series(np.arange(0, len(self._QSArgs.DTRuler)), index=self._QSArgs.DTRuler).reindex(index=self._DateTimes)
        if pd.isnull(DTs).any(): raise __QS_Error__("运算时点序列超出了时点标尺!")
        #elif (DTs.diff().iloc[1:]!=1).any(): raise __QS_Error__("运算时点序列的频率与时点标尺不一致!")
        
        self._postprocessContext()
        self._DTRange = {}
        
        # 收集所有的因子以及需要准备原始数据的因子
        self._RawFactorIDs = set()# 基本因子 QSID 列表, 即需要准备原始数据的因子
        self._iFactorDict = {}
        self._iFactorDict = self._genFactorDict(self._Factors, self._iFactorDict)
        self._RawFactorIDs = sorted(self._RawFactorIDs)
        self._FactorDict.update(self._iFactorDict)
        
        # 遍历所有因子对象, 调用其初始化方法, 生成所有因子的时点范围, 生成其需要准备原始数据的截面 ID
        for iFactor in self._Factors:
            iFactor.__QSBC_initOperation__(self, (self._DateTimes[0], self._DateTimes[-1]), self._SectionIDs)
        
        # 分配每个子进程的计算 ID 序列
        if self._QSArgs.CalcConcurrentNum == 0:# 串行模式
            self._PID_IDs = {self._iPID: self._SectionIDs}
        else:
            SubIDs = self.splitID(self._SectionIDs)
            self._PID_IDs = {self._PIDs[i]: iIDs for i, iIDs in enumerate(SubIDs)}
    
    # 单次运算结束后的处理
    def _postprocessContext(self):
        if self._QSArgs.CalcConcurrentNum > 0:
            # 从缓存中更新 DTRange
            for iFactorID, iFactor in self._iFactorDict.items():
                self._CachedDTRange[iFactorID] = self._Cache.readFactorData(key=iFactorID, target_field="DTRange", pids=self._PIDs[0])
    
    # 并发的原始数据准备
    # task: {"FactorIDs": [因子 QSID], "FT": 因子表对象, "RawFactorNames": {原始因子名}, "DTRange": (起始时点, 结束时点), "SectionIDs": [ID], "Args": {参数}}
    def _parallel_prepare(self, task):
        iRawFactorNames = sorted(task["RawFactorNames"])
        iRawData = task["FT"].__QS_prepareRawData__(iRawFactorNames, task["SectionIDs"], task["DTRange"], task["Args"])
        iSectionIDs = task["SectionIDs"]
        if iSectionIDs==self._SectionIDs:
            task["FT"].__QSBC_saveRawData__(iRawData, task["RawDataKey"], iRawFactorNames, pid_ids=self._PID_IDs)
        else:
            PIDIDs = {self._PIDs[i]: iIDs for i, iIDs in enumerate(self.splitID(sorted(iSectionIDs)))}
            task["FT"].__QSBC_saveRawData__(iRawData, task["RawDataKey"], iRawFactorNames, pid_ids=PIDIDs)
        return 0    
    
    # 准备原始数据
    def _prepare(self, **kwargs):
        # 分组准备数据
        InitGroups = {}# {因子表 QSID: (因子表, [因子])}
        for iFactorID in self._RawFactorIDs:
            iFactor = self._FactorDict[iFactorID]
            iFTID = iFactor.FactorTable.QSID
            if iFTID not in InitGroups:
                InitGroups[iFTID] = (iFactor.FactorTable, [iFactor])
            else:
                InitGroups[iFTID][1].append(iFactor)
        # 执行原始数据准备
        if self._QSArgs.IOConcurrentNum<=0:
            for iFTID, iInitGroup in InitGroups.items():
                for jTask in iInitGroup[0].__QSBC_genGroupInfo__(iInitGroup[1], self):
                    self._parallel_prepare(jTask)
        else:
            Futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._QSArgs.IOConcurrentNum) as Executor:
                for iFTID, iInitGroup in InitGroups.items():
                    for jTask in iInitGroup[0].__QSBC_genGroupInfo__(iInitGroup[1], self):
                        jFuture = Executor.submit(self._parallel_prepare, jTask)
                        Futures.append(jFuture)
                with ProgressBar(max_value=len(Futures)) as ProgBar:
                    for iFuture in concurrent.futures.as_completed(Futures):
                        iFuture.result()
                        ProgBar.update(ProgBar.value+1)
        return 0
    
    # 并发的因子计算
    def _parallel_write2FDB(self, task):
        self._iPID = task["PID"]
        # 分配任务
        TDB, TableName, SpecificTarget = task["FactorDB"], task["TableName"], task["specific_target"]
        if SpecificTarget:
            TaskDispatched = OrderedDict()# {(id(FactorDB), TableName) : (FatorDB, [Factor], [FactorName])}
            for iFactor in self._Factors:
                iDB, iTableName, iTargetFactorName = SpecificTarget.get(iFactor.Name, (None, None, None))
                if iDB is None: iDB = TDB
                if iTableName is None: iTableName = TableName
                if iTargetFactorName is None: iTargetFactorName = iFactor.Name
                iDBTable = (id(iDB), iTableName)
                if iDBTable in TaskDispatched:
                    TaskDispatched[iDBTable][1].append(iFactor)
                    TaskDispatched[iDBTable][2].append(iTargetFactorName)
                else:
                    TaskDispatched[iDBTable] = (iDB, [iFactor], [iTargetFactorName])
        else:
            TaskDispatched = {(id(TDB), TableName): (TDB, self._Factors, [iFactor.Name for iFactor in self._Factors])}
        # 执行任务
        nTask = len(self._Factors)
        nDT = len(self._DateTimes)
        TaskCount, BatchNum = 0, self._QSArgs.WriteBatchNum
        if self._QSArgs.CalcConcurrentNum <= 0:# 运行模式为串行
            with ProgressBar(max_value=nTask) as ProgBar:
                for i, iTask in enumerate(TaskDispatched):
                    iDB, iFactors, iTargetFactorNames = TaskDispatched[iTask]
                    iTableName = iTask[1]
                    if hasattr(iDB, "writeFactorData"):
                        for j, jFactor in enumerate(iFactors):
                            jData = jFactor.__QSBC_getData__(dts=self._DateTimes, pids=[task["PID"]])
                            if self._FactorSectionIDs.get(jFactor.QSID, None) is not None:
                                jData = jData.reindex(columns=self._IDs)
                            iDB.writeFactorData(jData, iTableName, iTargetFactorNames[j], if_exists=task["if_exists"], data_type=jFactor.getMetaData(key="DataType"), **task["kwargs"])
                            jData = None
                            TaskCount += 1
                            ProgBar.update(TaskCount)
                    else:
                        iFactorNum = len(iFactors)
                        iBatchNum = (iFactorNum if BatchNum<=0 else BatchNum)
                        iDTLen= int(np.ceil(nDT / iBatchNum))
                        iDataTypes = {iTargetFactorNames[j]:jFactor.getMetaData(key="DataType") for j, jFactor in enumerate(iFactors)}
                        for j in range(iBatchNum):
                            jDTs = list(self._DateTimes[j*iDTLen:(j+1)*iDTLen])
                            if jDTs:
                                jData = {}
                                for k, kFactor in enumerate(iFactors):
                                    ijkData = kFactor.__QSBC_getData__(dts=jDTs, pids=[task["PID"]])
                                    if self._FactorSectionIDs.get(kFactor.QSID, None) is not None:
                                        ijkData = ijkData.reindex(columns=self._IDs)
                                    jData[iTargetFactorNames[k]] = ijkData
                                    if j==0:
                                        TaskCount += 0.5
                                        ProgBar.update(TaskCount)
                                jData = Panel(jData, items=iTargetFactorNames, major_axis=jDTs)
                                iDB.writeData(jData, iTableName, if_exists=task["if_exists"], data_type=iDataTypes, **task["kwargs"])
                                jData = None
                            TaskCount += 0.5 * iFactorNum / iBatchNum
                            ProgBar.update(TaskCount)
        else:
            for i, iTask in enumerate(TaskDispatched):
                iDB, iFactors, iTargetFactorNames = TaskDispatched[iTask]
                iTableName = iTask[1]
                if hasattr(iDB, "writeFactorData"):
                    for j, jFactor in enumerate(iFactors):
                        if self._FactorSectionIDs.get(jFactor.QSID, None) is not None:
                            jData = jFactor.__QSBC_getData__(dts=self._DateTimes, pids=None)
                            jData = jData.reindex(columns=self._PID_IDs[task["PID"]])
                        else:
                            jData = jFactor.__QSBC_getData__(dts=self._DateTimes, pids=[task["PID"]])
                        iDB.writeFactorData(jData, iTableName, iTargetFactorNames[j], if_exists=task["if_exists"], data_type=jFactor.getMetaData(key="DataType"), **task["kwargs"])
                        jData = None
                        task["Sub2MainQueue"].put((task["PID"], 1, None))
                else:
                    iFactorNum = len(iFactors)
                    iBatchNum = (iFactorNum if BatchNum <= 0 else BatchNum)
                    iDTLen= int(np.ceil(nDT / iBatchNum))
                    iDataTypes = {iTargetFactorNames[j]:jFactor.getMetaData(key="DataType") for j, jFactor in enumerate(iFactors)}
                    for j in range(iBatchNum):
                        jDTs = list(self._DateTimes[j*iDTLen:(j+1)*iDTLen])
                        if jDTs:
                            jData = {}
                            for k, kFactor in enumerate(iFactors):
                                ijkData = kFactor.__QSBC_getData__(dts=jDTs, pids=[task["PID"]])
                                if self._FactorSectionIDs.get(kFactor.QSID, None) is not None:
                                    ijkData = ijkData.reindex(columns=self._IDs)
                                jData[iTargetFactorNames[k]] = ijkData
                                if j==0: task["Sub2MainQueue"].put((task["PID"], 0.5, None))
                            jData = Panel(jData, items=iTargetFactorNames, major_axis=jDTs)
                            iDB.writeData(jData, iTableName, if_exists=task["if_exists"], data_type=iDataTypes, **task["kwargs"])
                            jData = None
                        task["Sub2MainQueue"].put((task["PID"], 0.5 * iFactorNum / iBatchNum, None))
        return 0
    
    # 因子计算
    def _write2FDB(self, factor_db, table_name, if_exists, specific_target, **kwargs):
        Task = {"PID": "0", "FactorDB": factor_db, "TableName": table_name, "if_exists": if_exists, "specific_target": specific_target, "kwargs": kwargs}
        if self._QSArgs.CalcConcurrentNum <= 0:
            self._parallel_write2FDB(Task)
        else:
            nPrcs = len(self._PIDs)
            nTask = len(self._Factors) * nPrcs
            EventState = {iFactorID: 0 for iFactorID in self._Event}
            Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=self._parallel_write2FDB, arg=Task, main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            with ProgressBar(max_value=nTask) as ProgBar:
                while True:
                    nEvent = len(EventState)
                    if nEvent > 0:
                        FactorIDs = tuple(EventState.keys())
                        for iFactorID in FactorIDs:
                            iQueue = self._Event[iFactorID][0]
                            while not iQueue.empty():
                                jInc = iQueue.get()
                                EventState[iFactorID] += jInc
                            if EventState[iFactorID] >= nPrcs:
                                self._Event[iFactorID][1].set()
                                EventState.pop(iFactorID)
                    while ((not Sub2MainQueue.empty()) or (nEvent == 0)) and (iProg < nTask):
                        iPID, iSubProg, iMsg = Sub2MainQueue.get()
                        iProg += iSubProg
                        ProgBar.update(iProg)
                    if iProg >= nTask: break
            for iPID, iPrcs in Procs.items(): iPrcs.join()
        return 0
    
    def write2FDB(self, factors, ids, dts, factor_db, table_name, if_exists="update", section_ids=None, specific_target={}, **kwargs):
        if not isinstance(factor_db, WritableFactorDB): raise __QS_Error__("因子数据库: %s 不可写入!" % factor_db.Name)
        print("==========因子运算==========\n1. 原始数据准备\n")
        TotalStartT = time.perf_counter()
        self._Factors = factors
        self._DateTimes = sorted(dts)
        self._IDs = sorted(ids)
        self._SectionIDs = (self._QSArgs.DefaultSectionIDs if section_ids is None else sorted(section_ids))
        
        self._initContext(**kwargs)
        self._prepare(**kwargs)
        print(("耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n2. 因子数据计算\n")
        StartT = time.perf_counter()
        self._write2FDB(factor_db, table_name, if_exists, specific_target, **kwargs)
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+"\n3. 运算后处理\n")
        StartT = time.perf_counter()
        #self._postprocessContext()
        factor_db.connect()
        print(('耗时 : %.2f' % (time.perf_counter()-StartT, ))+"\n"+("总耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
        return 0
    
    # 并发的因子计算
    def _parallel_calculate(self, task):
        self._iPID = task["PID"]
        Data = {}
        if self._QSArgs.CalcConcurrentNum <= 0:# 运行模式为串行
            TaskCount = 0
            with ProgressBar(max_value=len(self._Factors)) as ProgBar:
                for j, jFactor in enumerate(self._Factors):
                    Data[jFactor.Name] = jFactor.__QSBC_getData__(dts=self._DateTimes, pids=[task["PID"]])
                    TaskCount += 1
                    ProgBar.update(TaskCount)
        else:
            for j, jFactor in enumerate(self._Factors):
                jData = jFactor.__QSBC_getData__(dts=self._DateTimes, pids=[task["PID"]])
                task["Sub2MainQueue"].put((task["PID"], 1, (jFactor.Name, jData)))
                Data[jFactor.Name] = jData
        return Data
    
    # 因子计算
    def _calculate(self, **kwargs):
        Task = {"PID": "0", "kwargs": kwargs}
        if self._QSArgs.CalcConcurrentNum <= 0:
            Data = self._parallel_calculate(Task)
            Data = Panel({iFactorName: iData.reindex(columns=self._IDs) for iFactorName, iData in Data.items()})
        else:
            #nPrcs = len(self._PIDs)
            #nTask = len(self._Factors) * nPrcs
            #EventState = {iFactorID: 0 for iFactorID in self._Event}
            #Futures, Sub2MainQueue = [], Queue()
            #Task["Sub2MainQueue"] = Sub2MainQueue
            #with concurrent.futures.ProcessPoolExecutor(max_workers=nPrcs) as Executor:
                #for iPID in self._PIDs:
                    #Task["PID"] = iPID
                    #jFuture = Executor.submit(self._parallel_calculate, Task)
                    #Futures.append(jFuture)
                #iProg = 0
                #with ProgressBar(max_value=nTask) as ProgBar:
                    #while True:
                        #nEvent = len(EventState)
                        #if nEvent > 0:
                            #FactorIDs = tuple(EventState.keys())
                            #for iFactorID in FactorIDs:
                                #iQueue = self._Event[iFactorID][0]
                                #while not iQueue.empty():
                                    #jInc = iQueue.get()
                                    #EventState[iFactorID] += jInc
                                #if EventState[iFactorID] >= nPrcs:
                                    #self._Event[iFactorID][1].set()
                                    #EventState.pop(iFactorID)
                        #while ((not Sub2MainQueue.empty()) or (nEvent == 0)) and (iProg < nTask):
                            #iPID, iSubProg, iMsg = Sub2MainQueue.get()
                            #iProg += iSubProg
                            #ProgBar.update(iProg)
                        #if iProg >= nTask: break
                #Data = {}
                #for iFuture in concurrent.futures.as_completed(Futures):
                    #for jFactorName, jData in iFuture.result().items():
                        #Data.setdefault(jFactorName, []).append(jData)
            nPrcs = len(self._PIDs)
            nTask = len(self._Factors) * nPrcs
            EventState = {iFactorID: 0 for iFactorID in self._Event}
            Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=self._parallel_calculate, arg=Task, main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            Data = {}
            with ProgressBar(max_value=nTask) as ProgBar:
                while True:
                    nEvent = len(EventState)
                    if nEvent > 0:
                        FactorIDs = tuple(EventState.keys())
                        for iFactorID in FactorIDs:
                            iQueue = self._Event[iFactorID][0]
                            while not iQueue.empty():
                                jInc = iQueue.get()
                                EventState[iFactorID] += jInc
                            if EventState[iFactorID] >= nPrcs:
                                self._Event[iFactorID][1].set()
                                EventState.pop(iFactorID)
                    while ((not Sub2MainQueue.empty()) or (nEvent == 0)) and (iProg < nTask):
                        iPID, iSubProg, iMsg = Sub2MainQueue.get()
                        iProg += iSubProg
                        ProgBar.update(iProg)
                        Data.setdefault(iMsg[0], []).append(iMsg[1])
                    if iProg >= nTask: break
            for iPID, iPrcs in Procs.items(): iPrcs.join()            
            Data = Panel({jFactorName: pd.concat(jData, join="outer", axis=1, ignore_index=False).reindex(columns=self._IDs) for jFactorName, jData in Data.items()})
        return Data
    
    def readData(self, factors, ids, dts, section_ids=None, **kwargs):
        print("==========因子运算==========\n1. 原始数据准备\n")
        TotalStartT = time.perf_counter()
        self._Factors = factors
        self._DateTimes = sorted(dts)
        self._IDs = sorted(ids)
        self._SectionIDs = (self._QSArgs.DefaultSectionIDs if section_ids is None else sorted(section_ids))
        
        self._initContext(**kwargs)
        self._prepare(**kwargs)
        print(("耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n2. 因子数据计算\n")
        StartT = time.perf_counter()
        Data = self._calculate(**kwargs)
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+"\n3. 运算后处理\n")
        StartT = time.perf_counter()
        #self._postprocessContext()
        print(('耗时 : %.2f' % (time.perf_counter()-StartT, ))+"\n"+("总耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
        return Data
