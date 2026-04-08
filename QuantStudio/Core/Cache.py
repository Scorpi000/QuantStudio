# -*- coding: utf-8 -*-
import datetime as dt
from typing import Optional, List, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Object__, __QS_Error__


class Cache(__QS_Object__):
    """缓存: 由键值对组成, 缓存的数据主要是 DataFrame"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        StartMode: Literal["new", "continue"] = Field(default="new", title="启动模式", description="启动缓存的方式: new 表示清空已有数据重新构造缓存；continue 表示通过 load 恢复之前的缓存状态")

    def __init__(self, args: dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._isStarted = False # 缓存是否已经启动

    def getUpdateData(self) -> dict:
        """并发运行后返回需要同步的内容"""
        return {}

    def updateCache(self, update_data: dict):
        """并发运行后更新同步内容"""
        pass
    
    def dump(self):
        """暂存缓存状态"""
        raise NotImplementedError

    def load(self):
        """恢复缓存状态"""
        raise NotImplementedError
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end()

    def start(self):
        """初始化缓存"""
        if self._isStarted: return
        if self._QSArgs.StartMode == "new":
            self.clearData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        self._isStarted = True

    def end(self, clear=False):
        """结束缓存"""
        if not self._isStarted: return
        if clear:
            self.clearData()
        else:
            self.dump()
        self._isStarted = False

    def checkDataExistence(self, key: str, create_if_not_exists: bool=False) -> bool:
        """检查某个数据项是否存在
        
        Args:
            key: 数据项的键
            create_if_not_exists: 如果不存在是否要创建该数据项
        
        Returns:
            是否存在该项数据
        """
        raise NotImplementedError

    def writeData(self, key: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="append", data_type: Optional[Literal["double", "string", "object"]]=None, meta:dict={}):
        """写入数据
        
        Args:
            key: 数据项的键
            data: 待写入的数据
            if_exists: 如果该数据已经存在的更新方式, append 表示只添加新增的数据, replace 表示替换已有数据
            data_type: 写入数据的数据类型, None 表示让系统自己判断
            meta: 数据项的元信息
        """
        raise NotImplementedError

    def readData(self, key: str, data_type: Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        """读取数据
        
        Args:
            key: 数据项的键
            data_type: 数据的数据类型, None 表示让系统自己判断
        
        Returns:
            数据项的值, None 表示该项不存在
        """
        raise NotImplementedError
    
    def clearData(self, key: Optional[str] = None):
        """清理数据
        
        Args:
            key: 数据项的键, None 表示清空所有缓存
        """
        raise NotImplementedError


class DTCache(Cache):
    """时序数据缓存: 缓存的数据主要是 DataFrame(index=[datetime])"""

    class __QS_ArgClass__(Cache.__QS_ArgClass__):
        DTRuler: List[dt.datetime] = Field(title="时点标尺", description="当前运行计算时点标尺", frozen=True)
        MinDTUnit: dt.timedelta = Field(default=dt.timedelta(1), title="最小时间单位", frozen=True)

    def __init__(self, args: dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(args=args, config_file=config_file, **kwargs)
        self._CachedDTRange = {}# 已经缓存的数据时点范围, {key: DataFrame(columns=["StartDT", "EndDT"])}

    def getUpdateData(self, key_list:Optional[List[str]]=None) -> dict:
        if key_list is None:
            return {"_CachedDTRange": self._CachedDTRange}
        else:
            return {"_CachedDTRange": {iKey: self._CachedDTRange[iKey] for iKey in key_list if iKey in self._CachedDTRange}}

    def updateCache(self, update_data: dict):
        for iKey, iDTRange in update_data.get("_CachedDTRange", {}).items():
            if iKey not in self._CachedDTRange:
                self._CachedDTRange[iKey] = iDTRange
            else:
                for iDTRange in iDTRange.astype("O").to_records(index=False):
                    self.updateDTRange(iKey, iDTRange)

    # 更新缓存的时点范围
    def _mergeDTRange(self, cached_dt_range, i):
        DTRuler = self._QSArgs.DTRuler
        iStartDT, iEndDT = cached_dt_range["StartDT"].iloc[i], cached_dt_range["EndDT"].iloc[i]
        DropIdx = []
        if i > 0:
            iPreEndDT = cached_dt_range["EndDT"].iloc[i - 1]
            iPreEndIdx, iStartdIdx = np.searchsorted(DTRuler, iPreEndDT, side="left"), np.searchsorted(DTRuler, iStartDT, side="right")
            if iPreEndIdx == iStartdIdx - 2:  # 两区间连续，合并
                cached_dt_range.at[cached_dt_range.index[i], "StartDT"] = cached_dt_range["StartDT"].iloc[i - 1]
                DropIdx.append(cached_dt_range.index[i - 1])
        if i < cached_dt_range.shape[0] - 1:
            iPostStartDT = cached_dt_range["StartDT"].iloc[i + 1]
            iEndIdx, iPostStartIdx = np.searchsorted(DTRuler, iEndDT, side="left"), np.searchsorted(DTRuler, iPostStartDT, side="right")
            if iEndIdx == iPostStartIdx - 2:  # 两区间连续，合并
                cached_dt_range.at[cached_dt_range.index[i], "EndDT"] = cached_dt_range["EndDT"].iloc[i + 1]
                DropIdx.append(cached_dt_range.index[i + 1])
        if DropIdx: cached_dt_range = cached_dt_range.drop(index=DropIdx)
        return cached_dt_range

    def getDTRange(self, key:str, dt_range:Tuple[dt.datetime, dt.datetime]) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        """给定时点范围，获取缓存中缺失的时点范围

        Args:
            key: 数据项的键
            dt_range: (起始时间, 结束时间), 时点范围
        
        Returns:
            缓存中缺失的时点范围, (起始时间, 结束时间), 返回 None 表示当前的缓存已经覆盖了给定的时点范围
        """
        CachedDTRange = self._CachedDTRange.get(key, None)
        if CachedDTRange is None:
            return dt_range
        else:
            StartIdx = CachedDTRange[
                (CachedDTRange["StartDT"] <= dt_range[0]) & (CachedDTRange["EndDT"] >= dt_range[0])]
            EndIdx = CachedDTRange[
                (CachedDTRange["StartDT"] <= dt_range[1]) & (CachedDTRange["EndDT"] >= dt_range[1])]
            if StartIdx.empty and EndIdx.empty:  # 新区间起始结束点均在空档里
                return dt_range
            elif (not StartIdx.empty) and (not EndIdx.empty):  # 新区间起始结束点均在已有区间里
                StartIdx, EndIdx = StartIdx.index[0], EndIdx.index[0]
                if StartIdx == EndIdx:  # 已有区间完全覆盖新区间
                    return None
                else:  # 新区间跨区间
                    StartDT = StartIdx["EndDT"].iloc[0] + self._QSArgs.MinDTUnit
                    EndDT = EndIdx["StartDT"].iloc[0] - self._QSArgs.MinDTUnit
                    return (StartDT, EndDT)
            elif StartIdx.empty and (not EndIdx.empty):  # 新区间起始点在空档里, 结束点在已有区间里
                EndDT = EndIdx["StartDT"].iloc[0] - self._QSArgs.MinDTUnit
                return (dt_range[0], EndDT)
            else:  # 新区间起始点在已有区间里, 结束点在空档里
                StartDT = StartIdx["EndDT"].iloc[0] + self._QSArgs.MinDTUnit
                return (StartDT, dt_range[1])
    
    def updateDTRange(self, key:str, dt_range:Tuple[dt.datetime, dt.datetime]):
        """更新缓存中数据的时点范围

        Args:
            key: 数据项的键
            dt_range: (起始时间, 结束时间), 时点范围
        """
        CachedDTRange = self._CachedDTRange.get(key, None)
        if CachedDTRange is None:
            self._CachedDTRange[key] = pd.DataFrame([dt_range], columns=["StartDT", "EndDT"])
            return
        StartIdx = CachedDTRange[
            (CachedDTRange["StartDT"] <= dt_range[0]) & (CachedDTRange["EndDT"] >= dt_range[0] - self._QSArgs.MinDTUnit)]
        EndIdx = CachedDTRange[
            (CachedDTRange["StartDT"] <= dt_range[1] + self._QSArgs.MinDTUnit) & (CachedDTRange["EndDT"] >= dt_range[1])]
        nRange = CachedDTRange.shape[0]
        if StartIdx.empty and EndIdx.empty:  # 新区间起始结束点均在空档里
            CachedDTRange = CachedDTRange[
                ~((CachedDTRange["StartDT"] >= dt_range[0]) & (CachedDTRange["EndDT"] <= dt_range[1]))]
            CachedDTRange.loc[nRange] = list(dt_range)
            CachedDTRange = CachedDTRange.sort_values(["StartDT"])
            i = CachedDTRange.index.tolist().index(nRange)
            self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)
        elif (not StartIdx.empty) and (not EndIdx.empty):  # 新区间起始结束点均在已有区间里
            StartIdx, EndIdx = StartIdx.index[0], EndIdx.index[0]
            if StartIdx == EndIdx:  # 已有区间完全覆盖新区间
                return
            else:  # 新区间跨区间
                StartDT, EndDT = CachedDTRange.at[StartIdx, "StartDT"], CachedDTRange.at[EndIdx, "EndDT"]
                CachedDTRange = CachedDTRange[(CachedDTRange.index < StartIdx) | (CachedDTRange.index > EndIdx)]
                CachedDTRange.loc[nRange] = [StartDT, EndDT]
                CachedDTRange = CachedDTRange.sort_values(["StartDT"])
                i = CachedDTRange.index.tolist().index(nRange)
                self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)
        elif StartIdx.empty and (not EndIdx.empty):  # 新区间起始点在空档里, 结束点在已有区间里
            EndDT = EndIdx["EndDT"].iloc[0]
            CachedDTRange = CachedDTRange[
                ~((CachedDTRange["StartDT"] >= dt_range[0]) & (CachedDTRange["EndDT"] <= EndDT))]
            CachedDTRange.loc[nRange] = [dt_range[0], EndDT]
            CachedDTRange = CachedDTRange.sort_values(["StartDT"])
            i = CachedDTRange.index.tolist().index(nRange)
            self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)
        else:  # 新区间起始点在已有区间里, 结束点在空档里
            StartDT = StartIdx["StartDT"].iloc[0]
            CachedDTRange = CachedDTRange[
                ~((CachedDTRange["StartDT"] >= StartDT) & (CachedDTRange["EndDT"] <= dt_range[1]))]
            CachedDTRange.loc[nRange] = [StartDT, dt_range[1]]
            CachedDTRange = CachedDTRange.sort_values(["StartDT"])
            i = CachedDTRange.index.tolist().index(nRange)
            self._CachedDTRange[key] = self._mergeDTRange(CachedDTRange, i).reset_index(drop=True)

    def start(self):
        if self._isStarted: return
        if self._QSArgs.StartMode == "new":
            self.clearData()
            self.clearDTData()
        elif self._QSArgs.StartMode == "continue":
            self.load()
        self._isStarted = True

    def end(self, clear=False):
        if not self._isStarted: return
        if clear:
            self.clearData()
            self.clearDTData()
        else:
            self.dump()
        self._isStarted = False    
    
    def checkDTDataExistence(self, key: str, create_if_not_exists: bool=False) -> bool:
        """检查某个时点数据项是否存在
        
        Args:
            key: 数据项的键
            create_if_not_exists: 如果不存在是否要创建该数据项
        
        Returns:
            是否存在该项数据
        """
        raise NotImplementedError

    def writeDTData(self, key: str, data: pd.DataFrame, if_exists: Literal["append", "replace"]="append", data_type: Optional[Literal["double", "string", "object"]]=None, meta:dict={}):
        """写入时点数据
        
        Args:
            key: 数据项的键
            data: 待写入的数据
            if_exists: 如果该数据已经存在的更新方式, append 表示只添加新增的数据, replace 表示替换已有数据
            data_type: 写入数据的数据类型, None 表示让系统自己判断
            meta: 数据项的元信息
        """
        raise NotImplementedError

    def readDTData(self, key: str, data_type: Optional[Literal["double", "string", "object"]]=None) -> None | pd.DataFrame:
        """读取时点数据
        
        Args:
            key: 数据项的键
            data_type: 数据的数据类型, None 表示让系统自己判断
        
        Returns:
            数据项的值, None 表示该项不存在
        """
        raise NotImplementedError
    
    def clearDTData(self, key: Optional[str] = None):
        """清理时点数据
        
        Args:
            key: 数据项的键, None 表示清空所有缓存
        """
        raise NotImplementedError