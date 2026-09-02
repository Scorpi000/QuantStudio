# -*- coding: utf-8 -*-
"""回测结果库: 存储和管理回测结果"""
import os
import time
import glob as _glob
from typing import Any, Optional, List

import h5py
from filelock import FileLock
from pydantic import Field
from pydantic import DirectoryPath

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Core import __QS_Object__, __QS_Error__
from QuantStudio.Tools.DataTypeFun import writeNestedDict2HDF5, readNestedDictFromHDF5


class BTResultDB(__QS_Object__):
    """回测结果库"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name: str = Field(default="BTResultDB", frozen=True, title="名称")

    @property
    def Name(self) -> str:
        """结果库名称"""
        return self._QSArgs.Name

    def writeResult(self, result: dict, group_name: str, metadata: Optional[dict] = None):
        """写入一组回测结果

        Args:
            result: 嵌套 dict, 叶节点为 DataFrame/Series/str/float 等
            group_name: 结果组名称, 支持路径层级 (如 "A股/IC/沪深300")
            metadata: 可选的元信息标签, 用于后续查询筛选 (如 {"资产": "A股", "策略": "IC"})
        """
        raise NotImplementedError

    def readResult(self, group_name: str) -> Optional[dict]:
        """读取一组回测结果

        Args:
            group_name: 结果组名称

        Returns:
            嵌套 dict, None 表示不存在
        """
        raise NotImplementedError

    def listResults(self, metadata: Optional[dict] = None) -> List[str]:
        """列出已存储的结果组名称

        Args:
            metadata: 按元信息标签筛选, None 表示返回所有。传入的 dict 是子集匹配 (AND 逻辑)

        Returns:
            匹配的结果组名称列表
        """
        raise NotImplementedError

    def readMetaData(self, group_name: str, key: Optional[str] = None) -> Any:
        """读取结果组的元信息

        Args:
            group_name: 结果组名称
            key: 元信息键, None 表示返回所有元信息的 dict

        Returns:
            指定键的值, 或所有元信息的 dict, None 表示不存在
        """
        raise NotImplementedError

    def setMetaData(self, group_name: str, key: Optional[str] = None, value: Any = None, metadata: Optional[dict] = None):
        """设置结果组的元信息

        Args:
            group_name: 结果组名称
            key: 元信息键
            value: 元信息值
            metadata: 若干组键值对元信息, 与 key/value 互斥
        """
        raise NotImplementedError

    @property
    def ResultNames(self) -> List[str]:
        """已存储的结果组名称列表"""
        return self.listResults(metadata=None)


class _HDF5BTResultDB(BTResultDB):
    """基于 HDF5 文件的回测结果库（单文件模式，内部实现）"""

    class __QS_ArgClass__(BTResultDB.__QS_ArgClass__):
        Name: str = Field(default="_HDF5BTResultDB", frozen=True, title="名称")
        FilePath: str = Field(title="文件路径", description="HDF5 文件路径")

    def writeResult(self, result: dict, group_name: str, metadata: Optional[dict] = None):
        writeNestedDict2HDF5(result, self._QSArgs.FilePath, group_name, mode="a")
        if metadata:
            with h5py.File(self._QSArgs.FilePath, mode="a") as f:
                if group_name in f:
                    for k, v in metadata.items():
                        f[group_name].attrs[k] = v

    def readResult(self, group_name: str) -> Optional[dict]:
        if not os.path.isfile(self._QSArgs.FilePath):
            return None
        return readNestedDictFromHDF5(self._QSArgs.FilePath, group_name)

    def listResults(self, metadata: Optional[dict] = None) -> List[str]:
        if not os.path.isfile(self._QSArgs.FilePath):
            return []
        result_groups = []
        with h5py.File(self._QSArgs.FilePath, mode="r") as f:
            def _visit(path, obj):
                if isinstance(obj, h5py.Group) and len(obj.attrs) > 0:
                    if metadata is None or all(obj.attrs.get(k) == v for k, v in metadata.items()):
                        result_groups.append(path)
            f.visititems(_visit)
        return sorted(result_groups)

    def readMetaData(self, group_name: str, key: Optional[str] = None) -> Any:
        if not os.path.isfile(self._QSArgs.FilePath):
            return None
        with h5py.File(self._QSArgs.FilePath, mode="r") as f:
            if group_name not in f:
                return None
            attrs = dict(f[group_name].attrs)
            if not attrs:
                return None
            if key is None:
                return attrs
            return attrs.get(key, None)

    def setMetaData(self, group_name: str, key: Optional[str] = None, value: Any = None, metadata: Optional[dict] = None):
        if not os.path.isfile(self._QSArgs.FilePath):
            return
        with h5py.File(self._QSArgs.FilePath, mode="a") as f:
            if group_name not in f:
                return
            if metadata:
                for k, v in metadata.items():
                    f[group_name].attrs[k] = v
            elif key is not None:
                f[group_name].attrs[key] = value


class HDF5BTResultDB(BTResultDB):
    """基于目录的回测结果库, 每个结果组一个 HDF5 文件, 目录层级对应 group_name 路径层级

    锁机制:
        MainDir 下的 _DB.lock 为库锁, listResults 遍历目录时需要获取库锁。
        每个 .h5 文件旁的 .lock 文件为文件锁, 读写单个结果组时需要获取文件锁。
    """

    class __QS_ArgClass__(BTResultDB.__QS_ArgClass__):
        Name: str = Field(default="HDF5BTResultDB", frozen=True, title="名称")
        MainDir: DirectoryPath = Field(title="主目录", description="存放结果文件的根目录")
        FileOpenRetryNum: int = Field(default=100, title="文件打开重试次数", frozen=False, exclude=True, ge=1, description="打开数据文件错误时的重试次数")

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 HDF5BTResultDB

        Args:
            args: 指定的对象参数集
            config_file: 配置文件路径, 默认配置文件为 "~/QuantStudioConfig/HDF5BTResultDBConfig.json"
        """
        if (not config_file) and os.path.isfile(__QS_ConfigPath__ + os.sep + "HDF5BTResultDBConfig.json"):
            config_file = __QS_ConfigPath__ + os.sep + "HDF5BTResultDBConfig.json"
        return super().__init__(args=args, config_file=config_file, **kwargs)

    def _getLock(self, file_path=None):
        """获取文件锁, file_path=None 时返回库锁"""
        if file_path is None:
            return FileLock(os.path.join(self._QSArgs.MainDir, "_DB.lock"))
        return FileLock(file_path + ".lock")

    def _openHDF5File(self, filename, *args, **kwargs):
        """带重试的 HDF5 文件打开"""
        i = 0
        while i < self._QSArgs.FileOpenRetryNum:
            try:
                f = h5py.File(filename, *args, **kwargs)
            except OSError as e:
                i += 1
                SleepTime = 0.05 + (i % 100) / 100.0
                if i % 100 == 0:
                    self._QS_Logger.warning("Can't open hdf5 file: '%s'\n %s \n try again %s seconds later!" % (filename, str(e), SleepTime))
                time.sleep(SleepTime)
            else:
                return f
        raise __QS_Error__("HDF5BTResultDB._openHDF5File: 打开 HDF5 文件 '%s' 失败!" % filename)

    def _group_to_path(self, group_name: str) -> str:
        """group_name → HDF5 文件路径, 如 'A股/IC/沪深300' → '<dir>/A股/IC/沪深300.h5'"""
        return os.path.join(self._QSArgs.MainDir, group_name + ".h5")

    def _path_to_group(self, file_path: str) -> str:
        """HDF5 文件路径 → group_name, 逆向还原 _group_to_path"""
        rel = os.path.relpath(file_path, self._QSArgs.MainDir)
        if rel.endswith(".h5"):
            rel = rel[:-3]
        return rel.replace(os.sep, "/")

    def writeResult(self, result: dict, group_name: str, metadata: Optional[dict] = None):
        file_path = self._group_to_path(group_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with self._getLock(file_path):
            writeNestedDict2HDF5(result, file_path, "/", mode="w")
            if metadata:
                with self._openHDF5File(file_path, mode="a") as f:
                    for k, v in metadata.items():
                        f.attrs[k] = v

    def readResult(self, group_name: str) -> Optional[dict]:
        file_path = self._group_to_path(group_name)
        if not os.path.isfile(file_path):
            return None
        with self._getLock(file_path):
            return readNestedDictFromHDF5(file_path, "/")

    def listResults(self, metadata: Optional[dict] = None) -> List[str]:
        if not os.path.isdir(self._QSArgs.MainDir):
            return []
        results = []
        with self._getLock():
            for fpath in _glob.glob(os.path.join(self._QSArgs.MainDir, "**", "*.h5"), recursive=True):
                with self._openHDF5File(fpath, mode="r") as f:
                    if len(f.attrs) > 0:
                        if metadata is None or all(f.attrs.get(k) == v for k, v in metadata.items()):
                            results.append(self._path_to_group(fpath))
        return sorted(results)

    def readMetaData(self, group_name: str, key: Optional[str] = None) -> Any:
        file_path = self._group_to_path(group_name)
        if not os.path.isfile(file_path):
            return None
        with self._getLock(file_path):
            with self._openHDF5File(file_path, mode="r") as f:
                attrs = dict(f.attrs)
                if not attrs:
                    return None
                if key is None:
                    return attrs
                return attrs.get(key, None)

    def setMetaData(self, group_name: str, key: Optional[str] = None, value: Any = None, metadata: Optional[dict] = None):
        file_path = self._group_to_path(group_name)
        if not os.path.isfile(file_path):
            return
        with self._getLock(file_path):
            with self._openHDF5File(file_path, mode="a") as f:
                if metadata:
                    for k, v in metadata.items():
                        f.attrs[k] = v
                elif key is not None:
                    f.attrs[key] = value
