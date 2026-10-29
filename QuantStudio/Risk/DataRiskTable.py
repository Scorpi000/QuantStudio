# -*- coding: utf-8 -*-
"""直接由内存协方差数据构造风险表，与 DataFactor 的数据来源职责对应。"""
from collections.abc import Mapping
from datetime import datetime as DateTime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from QuantStudio.Core.Identity import ExecutionEffect
from QuantStudio.Core.Input import EmptyDependencies
from QuantStudio.Core.QSObject import Panel
from .RiskTable import RiskTable


class _CovarianceData(BaseModel):
    """带日期与证券标签的协方差快照，可随图保存为 JSON。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    datetime: DateTime | None = None
    ids: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def _validate_matrix(self):
        matrix = np.asarray(self.values)
        if (not self.ids or len(set(self.ids)) != len(self.ids)
                or matrix.shape != (len(self.ids),) * 2
                or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T)):
            raise ValueError("协方差必须具有非空唯一证券标签，且为有限对称方阵")
        return self


class DataRiskTable(RiskTable):
    """数据风险表：直接持有协方差数据的无依赖来源节点。

    DataFrame 表示可用于任意请求日期的固定协方差；Panel 或日期到 DataFrame
    的映射表示逐期协方差。按日期精确匹配，未提供的日期或证券返回缺失值。
    数据保存为带标签的不可变快照，修改原始输入不会改变节点。
    """

    __QS_DepClass__ = EmptyDependencies
    __QS_ExecutionEffect__ = ExecutionEffect.PURE

    class __QS_ArgClass__(RiskTable.__QS_ArgClass__):
        Name: str = Field(default="DataRiskTable", frozen=True, title="名称")
        SourceRDB: Literal[None] = Field(default=None, frozen=True, exclude=True, repr=False)
        Data: tuple[_CovarianceData, ...] = Field(frozen=True, repr=False, min_length=1, title="风险数据")

    def __init__(self, data=None, args=None, config_file=None, **kwargs):
        """初始化数据风险表。

        Args:
            data: 带证券标签的 DataFrame，或日期到协方差矩阵的映射/Panel。
            args: 对象参数；模板和序列化还原时可使用其中的 Data 快照。
            config_file: 对象配置文件。
            **kwargs: 日志等通用对象选项。
        """
        values = dict(args or {})
        if "rdb" in kwargs or values.get("SourceRDB") is not None:
            raise TypeError("DataRiskTable 直接持有数据，不接受风险数据库")
        if data is not None:
            values["Data"] = self._normalize_data(data)
        super().__init__(rdb=None, args=values, config_file=config_file, **kwargs)
        records = self.Args.Data
        dates = [record.datetime for record in records]
        if (None in dates and len(records) != 1) or len(set(dates)) != len(dates):
            raise ValueError("风险数据日期必须唯一，固定矩阵不能与逐期数据混用")

    @staticmethod
    def _normalize_data(data):
        """将有标签的内存数据复制为可序列化快照。"""
        if isinstance(data, pd.DataFrame):
            entries = [(None, data)]
        elif isinstance(data, Panel):
            entries = [(date, data.loc[date]) for date in data.items]
        elif isinstance(data, Mapping):
            entries = list(data.items())
        else:
            raise TypeError("风险数据必须为 DataFrame、Panel 或日期到 DataFrame 的映射")
        records = []
        for date, matrix in entries:
            if date is not None and not isinstance(date, DateTime):
                raise TypeError("风险数据的日期必须为 datetime")
            if not isinstance(matrix, pd.DataFrame):
                raise TypeError("协方差必须使用带证券标签的 DataFrame")
            if (not matrix.index.is_unique or not matrix.columns.is_unique
                    or set(matrix.index) != set(matrix.columns)):
                raise ValueError("协方差的行列必须具有相同且唯一的证券标签")
            records.append(_CovarianceData(datetime=date, ids=tuple(matrix.index),
                values=tuple(map(tuple, matrix.reindex(columns=matrix.index).to_numpy(dtype=float)))))
        return tuple(records)

    @classmethod
    def from_dependencies(cls, deps, args=None, **kwargs):
        """从参数快照还原无依赖数据来源，不接受来源数据库。"""
        values = args.to_dict(repr=False) if hasattr(args, "to_dict") else dict(args or {})
        return cls(args=values, **kwargs)

    def serialize(self):
        """将内存数据快照及日期序列化为可保存的 JSON 值。"""
        result = super().serialize()
        result["__qsargs__"]["Data"] = [record.model_dump(mode="json") for record in self.Args.Data]
        return result

    def getID(self, idt=None):
        """返回给定日期的证券，未指定日期时返回全部证券的稳定并集。"""
        records = self.Args.Data
        if idt is not None and records[0].datetime is not None:
            records = tuple(record for record in records if record.datetime == idt)
        return list(dict.fromkeys(id_ for record in records for id_ in record.ids))

    def getDateTime(self, start_dt=None, end_dt=None):
        """返回已提供的数据日期；固定矩阵没有固有日期标尺。"""
        return sorted(record.datetime for record in self.Args.Data
                      if record.datetime is not None
                      and (start_dt is None or record.datetime >= start_dt)
                      and (end_dt is None or record.datetime <= end_dt))

    def readCov(self, dts, ids=None):
        """按日期和证券标签读取协方差，不对缺失日期前向填充。

        Args:
            dts: 请求日期序列。
            ids: 请求证券顺序，None 使用全部证券。

        Returns:
            日期 × 证券 × 证券的 Panel；未知标签或缺失日期为 NaN。
        """
        dts = list(dts)
        ids = self.getID() if ids is None else list(ids)
        records = {record.datetime: record for record in self.Args.Data}
        matrices = {}
        for date in dts:
            record = records.get(None, records.get(date))
            matrices[date] = (pd.DataFrame(index=ids, columns=ids, dtype=float) if record is None else
                              pd.DataFrame(record.values, index=record.ids, columns=record.ids).reindex(index=ids, columns=ids))
        return Panel(matrices, items=dts, major_axis=ids, minor_axis=ids)

    def init_compute(self, path, init_data, context):
        """内存叶子无需注册 I/O 准备任务。"""
        return []
