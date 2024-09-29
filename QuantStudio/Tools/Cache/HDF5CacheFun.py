import threading
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import tables

from QuantStudio.Tools.Cache.Constant import CacheConstant


class HDF5Cache:
    def __init__(self, path: str, lock=threading.Lock()):
        self.path = path
        self.file_lock = lock
        self.h5file = tables.open_file(self.path, mode='a')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.h5file.close()

    def store_cache(self, key: str, value: pd.DataFrame, mode='append', batch_size=10000):
        with self.file_lock:
            try:
                if mode == 'replace' and key in self.h5file.root:
                    self.h5file.remove_node(self.h5file.root, key)
                if key not in self.h5file.root:
                    dtypes = self.to_hdf5_dtype(value)
                    table = self.h5file.create_table(self.h5file.root, key, description=dict(dtypes))
                else:
                    table = self.h5file.get_node(self.h5file.root, key)
                records = value.to_records(index=False)
                start = 0
                end = batch_size
                while start < len(records):
                    batch = records[start:end]
                    # 确保批次数据的形状与表结构匹配
                    if isinstance(batch, np.ndarray):
                        # 如果是NumPy数组，则确保维度正确
                        if batch.ndim == 1:
                            batch = batch[np.newaxis, :]
                    table.append(batch)
                    table.flush()
                    start = end
                    end += batch_size
                if end >= len(records):
                    self.set_ready(table)
            except Exception as e:
                print(f"Failed to store cache {key} with error: {e}")

    def get_cache(self,
                  key: str,
                  columns: List[str],
                  query: List[Tuple[str, any]] = None,
                  no_data_await=False) -> pd.DataFrame:
        return self.query(key, columns, query, no_data_await)

    def query(self,
              key: str,
              columns: List[str],
              query_condition: List[Tuple[str, any]] = None,
              no_data_await=False) -> pd.DataFrame:
        try:
            table = self.h5file.get_node(self.h5file.root, key)
            if not table and no_data_await:
                time.sleep(1)
                self.query(key, columns, query_condition, no_data_await)
            if query_condition is not None and len(query_condition) > 0:
                query = " and ".join([f"{col} == {value}" for col, value in query_condition])
                data = table.where(query, columns=columns)
            else:
                data = table.read(columns=columns)
            if not data and no_data_await:
                time.sleep(1)
                self.query(key, columns, query_condition, no_data_await)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"query cache {key} occur error: {e}")
            raise e

    def set_cache_attrs(self, table, attrs: {}):
        if hasattr(table, '_v_attrs'):
            n_attrs = table._v_attrs
            for k, v in attrs.items():
                n_attrs[k] = v
        else:
            raise AttributeError("Node does not support attributes.")

    def get_cache_attrs(self, key):
        if key not in self.h5file.root:
            return {}
        table = self.h5file.get_node(self.h5file.root, key)
        if hasattr(table, '_v_attrs'):
            return table._v_attrs

    def set_ready(self, table):
        return self.set_cache_attrs(table, {CacheConstant.IS_READY: True})

    def is_ready(self, key):
        return self.get_cache_attrs(key)[CacheConstant.IS_READY]

    def to_hdf5_dtype(self, df):
        columns = []
        for col_name, col_type in df.dtypes.items():
            if col_type == 'O':
                # 对象类型，假设为字符串
                columns.append((col_name, tables.StringCol(itemsize=255)))
            elif col_type == 'datetime64[ns]':
                # 日期时间类型，存储为时间戳
                columns.append((col_name, tables.Int64Col()))
            elif col_type == 'float64':
                # 浮点数类型
                columns.append((col_name, tables.Float64Col()))
            elif col_type == 'int64':
                # 整数类型
                columns.append((col_name, tables.Int64Col()))
            else:
                raise ValueError(f"Unsupported type: {col_type}")
        return columns


if __name__ == '__main__':
    filename = 'data.h5'

    # 创建 HDF5Cache 实例
    cache = HDF5Cache(filename)

    # 创建示例 DataFrame
    df = pd.DataFrame({
        'A': np.arange(10000),
        'B': np.arange(10000, 20000),
        'C': ['foo'] * 10000
    })
    cache.store_cache("test", df)

