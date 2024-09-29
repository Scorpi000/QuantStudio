# coding=utf-8
import os
import gc
import time
import uuid
import html
import mmap
import pickle
import datetime as dt
from collections import OrderedDict
from multiprocessing import Process, Queue, cpu_count
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from traits.api import Instance, Str, List, Int, Enum, ListStr, Either, Directory, Dict

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
        self._BatchContext = None# 批量运算时环境
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
    
    # 查找因子对象, def_path: 以/分割的因子查找路径, 比如 年化收益率/0/1
    def searchFactor(self, factor_name=None, def_path=None, only_one=True, raise_error=True):
        if def_path is not None:
            def_path = def_path.split("/")
            if def_path[0] not in self.FactorNames:
                return None
            iFactor = self.getFactor(def_path[0])
            for iIdx in def_path[1:]:
                try:
                    iFactor = iFactor.Descriptors[int(iIdx)]
                except:
                    if raise_error:
                        raise __QS_Error__(f"查找不到因子: {def_path}")
                    return None
            if (factor_name is not None) and (iFactor.Name != factor_name):
                if raise_error:
                    if factor_name is not None:
                        raise __QS_Error__(f"查找不到因子({factor_name}): {def_path}")
                    else:
                        raise __QS_Error__(f"查找不到因子: {def_path}")
                return None
            else:
                return iFactor
        elif factor_name is not None:
            def _searchFactor(factors, factor_name):
                Factors = []
                for iFactor in factors:
                    if iFactor.Name == factor_name:
                        Factors.append(iFactor)
                    Factors += _searchFactor(iFactor.Descriptors, factor_name)
                return Factors

            Factors = []
            for iFactorName in self.FactorNames:
                iFactor = self.getFactor(iFactorName)
                if iFactorName == factor_name:
                    Factors.append(iFactor)
                Factors += _searchFactor(iFactor.Descriptors, factor_name)
            if only_one:
                if len(Factors) == 1:
                    return Factors[0]
                elif len(Factors) == 0:
                    if raise_error:
                        raise __QS_Error__(f"查找不到因子: {factor_name}")
                    else:
                        return None
                else:
                    if raise_error:
                        raise __QS_Error__(f"因子({factor_name}) 不止一个!")
                    else:
                        return None
            else:
                return Factors
        else:
            raise __QS_Error__("参数 def_path 和 factor_name 不能同时为 None!")

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
    # -------------------------------读取数据---------------------------------
    # 准备原始数据
    # context dict: 
    #     FactorStartDT: {因子QSID: datetime}, 因子数据需要的最早时点
    #     FactorEndDT: datetime, 因子数据需要的结束时点
    #     DTRuler: [datetime], 时点标尺
    # executor: 线程池或者进程池, None 表示非并行
    def _prepareRawData(self, factors, context, executor=None):
        if self._QS_GroupArgs is None:# 每个因子单独准备
            StartDT = dt.datetime.now()
            FactorNames, RawFactorNames = [], set()
            for iFactor in factors:
                FactorNames.append(iFactor.Name)
                RawFactorNames.add(iFactor._NameInFT)
                StartDT = min((StartDT, operation_mode._FactorStartDT[iFactor.QSID]))
            EndDT = operation_mode.DateTimes[-1]
            StartInd, EndInd = operation_mode.DTRuler.index(StartDT), operation_mode.DTRuler.index(EndDT)
            return [(self, FactorNames, list(RawFactorNames), operation_mode.DTRuler[StartInd:EndInd+1], {})]            
    
    # 获取因子表准备原始数据的分组信息, [(因子表对象, [因子名], [原始因子名], [时点], {参数})]
    def __QS_genGroupInfo__(self, factors, context):
        if self._QS_GroupArgs is None: return# 每个因子单独准备
        StartDT = dt.datetime.now()
        FactorNames, RawFactorNames = [], set()
        for iFactor in factors:
            FactorNames.append(iFactor.Name)
            RawFactorNames.add(iFactor._NameInFT)
            StartDT = min((StartDT, operation_mode._FactorStartDT[iFactor.QSID]))
        EndDT = operation_mode.DateTimes[-1]
        StartInd, EndInd = operation_mode.DTRuler.index(StartDT), operation_mode.DTRuler.index(EndDT)
        return [(self, FactorNames, list(RawFactorNames), operation_mode.DTRuler[StartInd:EndInd+1], {})]
    
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ConditionGroup = {}
        for iFactor in factors:
            iConditions = ";".join([iArgName+":"+str(iFactor._QSArgs[iArgName]) for iArgName in iFactor._QSArgs.ArgNames if iArgName in self._QS_GroupArgs])
            if iConditions not in ConditionGroup:
                ConditionGroup[iConditions] = {
                    "FactorNames":[iFactor.Name],
                    "RawFactorNames":{iFactor._NameInFT},
                    "StartDT":operation_mode._FactorStartDT[iFactor.Name],
                    "args":iFactor.Args.to_dict()
                }
            else:
                ConditionGroup[iConditions]["FactorNames"].append(iFactor.Name)
                ConditionGroup[iConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ConditionGroup[iConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ConditionGroup[iConditions]["StartDT"])
                if "回溯天数" in ConditionGroup[iConditions]["args"]:
                    ConditionGroup[iConditions]["args"]["回溯天数"] = max(ConditionGroup[iConditions]["args"]["回溯天数"], iFactor._QSArgs.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iConditions in ConditionGroup:
            StartInd = operation_mode.DTRuler.index(ConditionGroup[iConditions]["StartDT"])
            Groups.append((self, ConditionGroup[iConditions]["FactorNames"], list(ConditionGroup[iConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ConditionGroup[iConditions]["args"]))
        return Groups
    
    def __QS_saveRawData__(self, raw_data, factor_names, cache: FactorCache, pid_ids, file_name, **kwargs):
        return cache.writeRawData(file_name, raw_data, target_fields=factor_names, additional_data=kwargs.get("additional_data", {}), pid_ids=pid_ids)
    
    # 准备原始数据的接口
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        return None
    
    # 计算数据的接口, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        return None
    
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, factor_names, ids, dts, args={}):
        return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args), factor_names=factor_names, ids=ids, dts=dts, args=args)
    
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
    def equals(self, other):
        if self is other: return True
        if not super().equals(other): return False
        if self._DateTimes != other._DateTimes: return False
        if self._IDs != other._IDs: return False
        if self._Factors != other._Factors: return False
        return True


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
        return self._FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, args=self._QSArgs.Args).loc[self._NameInFT]
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
    
    # 获取数据的开始时点, start_dt:新起始时点, dt_dict: 当前所有因子的时点信息: {因子名 : 开始时点}, id_dict: 当前所有因子的准备原始数据的截面 ID 信息: {因子名 : ID 序列}
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        OldStartDT = dt_dict.get(self.QSID, start_dt)
        dt_dict[self.QSID] = (start_dt if start_dt<OldStartDT else OldStartDT)
        PrepareIDs = id_dict.setdefault(self.QSID, prepare_ids)
        if prepare_ids != PrepareIDs:
            raise __QS_Error__("因子 %s (QSID: %s) 指定了不同的截面!" % (self.Name, self.QSID))
    # 准备缓存数据
    def __QS_prepareCacheData__(self, ids=None):
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = self._OperationMode.DTRuler[StartInd:EndInd+1]
        RawData, PrepareIDs = self._OperationMode._Cache.readRawData(self._RawDataFile, self._OperationMode._iPID, self._NameInFT)
        if PrepareIDs is None:
            PrepareIDs = self._OperationMode._FactorPrepareIDs[self.Name]
            if PrepareIDs is None: PrepareIDs = self._OperationMode._PID_IDs[self._OperationMode._iPID]
        if RawData is not None:
            StdData = self._FactorTable.__QS_calcData__(RawData, factor_names=[self._NameInFT], ids=PrepareIDs, dts=DTs, args=self.Args).iloc[0]
        else:
            StdData = self._FactorTable.readData(factor_names=[self._NameInFT], ids=PrepareIDs, dts=DTs, args=self.Args).iloc[0]
        self._OperationMode._Cache.writeFactorData(self.Name+str(self._OperationMode._FactorID[self.Name]), StdData, pid=self._OperationMode._iPID)
        self._isCacheDataOK = True
        return StdData
    # 获取因子数据, pid=None表示取所有进程的数据
    def _QS_getData(self, dts, pids=None, **kwargs):
        if not self._isCacheDataOK:# 若没有准备好缓存数据, 准备缓存数据
            self.__QS_prepareCacheData__()
        StdData = self._OperationMode._Cache.readFactorData(self.Name+str(self._OperationMode._FactorID[self.Name]), pids=pids)
        if pids is not None:
            StdData = StdData.reindex(index=list(dts))
        elif self._OperationMode._FactorPrepareIDs[self.Name] is None:
            StdData = StdData.reindex(index=list(dts), columns=self._OperationMode.IDs)
        else:
            StdData = StdData.reindex(index=list(dts), columns=self._OperationMode._FactorPrepareIDs[self.Name])
        gc.collect()
        return StdData
    
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
        if (self._OperationMode is not None) and (self._OperationMode._isStarted): return self._OperationMode.IDs
        if self._DataContent=="Factor":
            return self._Data.columns.tolist()
        elif self._DataContent=="ID":
            return self._Data.index.tolist()
        else:
            return []
    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted): return self._OperationMode.DateTimes
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
    def __QS_prepareCacheData__(self, ids=None):
        return self._Data
    def _QS_getData(self, dts, pids=None, **kwargs):
        IDs = kwargs.get("ids", None)
        if IDs is None:
            IDs = self._OperationMode._FactorPrepareIDs[self.Name]
            if IDs is None:
                if pids is None:
                    IDs = list(self._OperationMode.IDs)
                else:
                    IDs = []
                    for iPID in pids: IDs.extend(self._OperationMode._PID_IDs[iPID])
            else:
                if pids is not None:
                    PrepareIDs = partitionListMovingSampling(IDs, len(self._OperationMode._PID_IDs))
                    IDs = []
                    for iPID in pids: IDs.extend(PrepareIDs[self._OperationMode._PIDs.index(iPID)])
        return self.readData(sorted(IDs), dts = list(dts))
    def equals(self, other):
        if self is other: return True
        if not super().equals(other): return False
        if self._DataContent != other._DataContent: return False
        if not (isinstance(self._Data, type(other._Data)) or isinstance(other._Data, type(self._Data))): return False
        if isinstance(self._Data, (pd.DataFrame, pd.Series)) and (not self._Data.equals(other._Data)): return False
        return (self._Data == other._Data)

# 为因子表达式产生的因子更改其他信息
def Factorize(factor_object, factor_name, args={}, **kwargs):
    factor_object.Name = factor_name
    for iArg, iVal in args.items(): factor_object._QSArgs[iArg] = iVal
    if "logger" in kwargs: factor_object._QS_Logger = kwargs["logger"]
    return factor_object

# 批量运算时环境
class BatchContext(__QS_Object__):
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        DTRuler = List(dt.datetime, arg_type="DateTimeList", label="时点标尺", order=0)
        CalcDTRuler = List(dt.datetime, arg_type="DateTimeList", label="计算时点标尺", order=1)
        SectionIDs = ListStr(arg_type="IDList", label="截面ID", order=1)
        IOConcurrentNum = Int(100, arg_type="Integer", label="IO并发数", order=2)
        CalcConcurrentNum = Int(0, arg_type="Integer", label="计算并发数", order=3)
        IDSplit = Enum("连续切分", "间隔切分", arg_type="SingleOption", label="ID切分", order=4)
        WriteBatchNum = Int(1, arg_type="Integer", label="写入批次", order=5)
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._isStarted = False# 是否开始
        self._Factors = []# 因子列表, 只包含当前生成数据的因子
        self._FactorDict = {}# 因子字典, {因子名:因子}, 包括所有的因子, 即衍生因子所依赖的描述子也在内
        self._FactorID = {}# {因子名: 因子唯一的 ID 号(int)}, 比如防止操作系统文件大小写不敏感导致缓存文件重名
        self._Factor2RawFactor = {}  # 因子对应的基础因子名称列表, {因子名: {基础因子名}}
        self._FactorStartDT = {}# {因子名: 起始时点}
        self._FactorPrepareIDs = {}# {因子名: 需要准备原始数据的 ID 序列}
        self._iPID = "0"# 对象所在的进程 ID
        self._PIDs = []# 所有的计算进程 ID, 单进程下默认为"0", 多进程为"0-i"
        self._PID_IDs = {}# 每个计算进程分配的 ID 列表, {PID:[ID]}
        self._PID_Lock = {}# 每个计算进程分配的缓存数据锁, {PID:Lock}
        self._Cache = None# 因子缓存对象
        self._Event = {}# {因子名: (Sub2MainQueue, Event)}
        self._IOExecutor = None# IO 执行器
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        
    def start(self):
        self._isStarted = True
        if self._QSArgs.IOConcurrentNum>0:
            self._IOExecutor = ThreadPoolExecutor(max_workers=self._QSArgs.IOConcurrentNum)
        __QS_BatchContext__.append(self)
    
    def end(self):
        self._isStarted = False
        self._IOExecutor = None
        __QS_BatchContext__.remove(self)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
        return (exc_type is None)
