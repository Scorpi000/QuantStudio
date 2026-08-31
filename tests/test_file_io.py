# -*- coding: utf-8 -*-
"""文件格式读写功能与性能对比测试.

功能测试覆盖:
    - feather / parquet / hdf5 三种格式的写入-读取往返一致性
    - 中文文件名兼容性

性能测试覆盖 (TestFileIOBenchmark):
    对以下 9 种格式进行大 DataFrame (10000×10000, 约 763MB 纯浮点) 的读写耗时和文件大小对比:

    格式             说明
    ──────────────── ─────────────────────────────────────
    feather          Arrow IPC 列式格式, 默认 zstd 压缩
    feather_uncomp   Arrow IPC 无压缩
    parquet          列式存储, 默认 snappy 压缩
    parquet_uncomp   parquet 无压缩
    hdf5             HDF5 fixed 格式, 无压缩, 适合同构纯数值数据
    pickle           Python 原生 pickle 序列化, 最高协议
    arrow_ipc        PyArrow IPC 文件格式, 无压缩
    zarr             分块压缩数组存储, 目录格式
    lance            新一代列式格式, 面向 ML/AI 向量检索场景

    测试结果 (SSD, 2026-08):

    格式                 写入       读取       文件大小
    ──────────────────── ──────── ──────── ──────────
    pickle                0.683s    0.291s    762.9 MB  🥇 写读最快
    hdf5                  0.693s    0.665s    763.2 MB  🥈 写入次快
    feather_uncomp        1.272s    0.480s    766.4 MB
    arrow_ipc             1.360s    0.428s    766.4 MB
    zarr                  1.568s    0.816s    731.2 MB  🥇 体积最小
    feather               1.662s    0.539s    766.7 MB
    parquet_uncomp        6.900s    0.909s    935.2 MB
    parquet               7.351s    0.882s    935.4 MB
    lance                10.279s   12.439s    770.2 MB

    结论:
    - pickle 读写最快, 但不支持列裁剪和跨语言
    - hdf5 写入次快、读取不差, 支持按 key 管理多个 DataFrame
    - feather/arrow_ipc 读取较快, 适合临时缓存和进程间通信
    - zarr 体积最小, 但无明显速度优势, 适合云存储和多维数组场景
    - parquet 压缩收益大但速度最慢, 适合长期存储和跨平台交换
    - lance 面向 ML 向量检索, 对纯数值 DataFrame 场景过重

使用方法:
    # 运行功能测试
    python -m unittest tests.test_file_io.TestFileIO -v

    # 运行性能对比 (仅输出, 不断言)
    python -m unittest tests.test_file_io.TestFileIOBenchmark -v
"""

import os
import pickle
import shutil
import time
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import zarr
import lance


class TestFileIO(unittest.TestCase):
    """文件格式读写功能测试."""

    @classmethod
    def setUpClass(cls):
        """创建临时目录和测试数据."""
        cls._tmp_dir = tempfile.mkdtemp()
        cls._df = pd.DataFrame(np.random.randn(100, 100))

    @classmethod
    def tearDownClass(cls):
        """清理临时文件."""
        for f in os.listdir(cls._tmp_dir):
            os.remove(os.path.join(cls._tmp_dir, f))
        os.rmdir(cls._tmp_dir)

    def _path(self, name: str) -> str:
        """生成临时目录下的文件路径."""
        return os.path.join(self._tmp_dir, name)

    # ==================== 往返一致性 ====================

    def test_feather_roundtrip(self):
        """feather 格式写入后读取, 数据应一致."""
        path = self._path("test.feather")
        self._df.to_feather(path)
        result = pd.read_feather(path)
        pd.testing.assert_frame_equal(result, self._df)

    def test_parquet_roundtrip(self):
        """parquet 格式写入后读取, 数据应一致."""
        path = self._path("test.parquet")
        self._df.to_parquet(path)
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result, self._df)

    def test_hdf5_roundtrip(self):
        """hdf5 格式写入后读取, 数据应一致."""
        path = self._path("test.h5")
        self._df.to_hdf(path, key="data")
        result = pd.read_hdf(path, key="data")
        pd.testing.assert_frame_equal(result, self._df)

    # ==================== 中文路径 ====================

    def test_chinese_filename_feather(self):
        """中文文件名: feather 格式."""
        path = self._path("中文简称.feather")
        self._df.to_feather(path)
        result = pd.read_feather(path)
        pd.testing.assert_frame_equal(result, self._df)

    def test_chinese_filename_parquet(self):
        """中文文件名: parquet 格式."""
        path = self._path("中文简称.parquet")
        self._df.to_parquet(path)
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result, self._df)

    def test_chinese_filename_hdf5(self):
        """中文文件名: hdf5 格式."""
        path = self._path("中文简称.h5")
        self._df.to_hdf(path, key="data")
        result = pd.read_hdf(path, key="data")
        pd.testing.assert_frame_equal(result, self._df)


class TestFileIOBenchmark(unittest.TestCase):
    """文件格式读写性能对比.

    非功能性测试: 仅输出耗时和文件大小, 不做断言, 用于人工对比.
    运行方式: python -m unittest tests.test_file_io.TestFileIOBenchmark -v

    临时目录默认使用系统临时目录 (通常在 C 盘 SSD).
    如需对比不同磁盘的性能差异, 可通过 dir 参数指定路径, 例如:
        cls._tmp_dir = tempfile.mkdtemp(dir=r"D:\\Temp")
    """

    @classmethod
    def setUpClass(cls):
        """创建临时目录和大数据集 (10000×10000 纯浮点, 约 763MB)."""
        cls._tmp_dir = tempfile.mkdtemp()
        cls._df = pd.DataFrame(np.random.randn(10000, 10000))

    @classmethod
    def tearDownClass(cls):
        """清理临时文件和目录."""
        for f in os.listdir(cls._tmp_dir):
            full_path = os.path.join(cls._tmp_dir, f)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
        os.rmdir(cls._tmp_dir)

    def _path(self, name: str) -> str:
        """生成临时目录下的文件路径."""
        return os.path.join(self._tmp_dir, name)

    def _fmt_size(self, size_bytes: int) -> str:
        """格式化文件大小."""
        if size_bytes >= 2**30:
            return f"{size_bytes / 2**30:.2f} GB"
        return f"{size_bytes / 2**20:.1f} MB"

    def _dir_size(self, path: str) -> int:
        """计算目录总大小 (用于 zarr/lance 等目录格式)."""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total

    # ==================== 写入函数 ====================

    def _write_feather(self, path):
        self._df.to_feather(path)

    def _write_feather_uncomp(self, path):
        self._df.to_feather(path, compression="uncompressed")

    def _write_parquet(self, path):
        self._df.to_parquet(path)

    def _write_parquet_uncomp(self, path):
        self._df.to_parquet(path, compression=None)

    def _write_hdf5(self, path):
        self._df.to_hdf(path, key="data", format="fixed")

    def _write_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self._df, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _write_arrow_ipc(self, path):
        table = pa.Table.from_pandas(self._df)
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)

    def _write_zarr(self, path):
        arr = zarr.create_array(path, shape=self._df.shape, dtype="float64", overwrite=True)
        arr[:] = self._df.values

    def _write_lance(self, path):
        table = pa.Table.from_pandas(self._df)
        lance.write_dataset(table, path)

    # ==================== 读取函数 ====================

    def _read_feather(self, path):
        return pd.read_feather(path)

    def _read_parquet(self, path):
        return pd.read_parquet(path)

    def _read_hdf5(self, path):
        return pd.read_hdf(path, key="data")

    def _read_pickle(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _read_arrow_ipc(self, path):
        with ipc.open_file(path) as reader:
            return reader.read_all().to_pandas()

    def _read_zarr(self, path):
        arr = zarr.open_array(path, mode="r")
        return pd.DataFrame(arr[:])

    def _read_lance(self, path):
        return lance.dataset(path).to_table().to_pandas()

    # ==================== 基准测试 ====================

    def test_benchmark_all(self):
        """全格式读写性能与文件大小对比."""
        # (名称, 写入函数, 读取函数, 是否目录格式)
        formats = [
            ("feather",        self._write_feather,        self._read_feather,        False),
            ("feather_uncomp", self._write_feather_uncomp, self._read_feather,        False),
            ("parquet",        self._write_parquet,        self._read_parquet,        False),
            ("parquet_uncomp", self._write_parquet_uncomp, self._read_parquet,        False),
            ("hdf5",           self._write_hdf5,           self._read_hdf5,           False),
            ("pickle",         self._write_pickle,         self._read_pickle,         False),
            ("arrow_ipc",      self._write_arrow_ipc,      self._read_arrow_ipc,      False),
            ("zarr",           self._write_zarr,           self._read_zarr,           True),
            ("lance",          self._write_lance,          self._read_lance,          True),
        ]

        write_times = {}
        read_times = {}
        file_sizes = {}

        for name, writer, reader, is_dir in formats:
            path = self._path(f"bench_{name}")

            # 写入
            start = time.perf_counter()
            writer(path)
            write_times[name] = time.perf_counter() - start

            # 文件/目录大小
            file_sizes[name] = self._dir_size(path) if is_dir else os.path.getsize(path)

            # 读取
            start = time.perf_counter()
            reader(path)
            read_times[name] = time.perf_counter() - start

        # 输出结果表
        print("\n")
        print(f"  {'格式':<20} {'写入':>8} {'读取':>8} {'文件大小':>10}")
        print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 10}")
        for name, *_ in formats:
            print(
                f"  {name:<20} {write_times[name]:>7.3f}s {read_times[name]:>7.3f}s"
                f" {self._fmt_size(file_sizes[name]):>10}"
            )

        # 排名
        print()
        print(f"  写入排名: {', '.join(sorted(write_times, key=write_times.get))}")
        print(f"  读取排名: {', '.join(sorted(read_times, key=read_times.get))}")
        print(f"  大小排名: {', '.join(sorted(file_sizes, key=file_sizes.get))}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
