# -*- coding: utf-8 -*-
"""QSObject 核心对象功能测试.

测试覆盖:
    - FileLock: 多线程/多进程/进程池/ProcessPoolExecutor 并发锁获取与互斥
    - QSQueue: 多进程共享内存队列的 put/get、大数据分片、序列化、上下文管理器
"""

import os
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from filelock import FileLock
from multiprocess import Process, Pool

from QuantStudio.Core.QSObject import QSQueue


# ========== 辅助函数 ==========

def _file_lock_worker(file_path: str, worker_id: int, result_list: list):
    """FileLock 工作进程/线程: 获取锁后记录 worker_id."""
    lock = FileLock(file_path)
    with lock:
        result_list.append(worker_id)


def _file_lock_worker_process(file_path: str, worker_id: int):
    """FileLock 子进程: 获取锁后返回 worker_id (用于 ProcessPoolExecutor/Pool)."""
    lock = FileLock(file_path)
    with lock:
        return worker_id


def _qs_queue_put_worker(worker_id: int, queue: QSQueue, data: str):
    """QSQueue 工作进程: 向队列写入数据."""
    queue.put((worker_id, data))


def _qs_queue_put_large_worker(worker_id: int, queue: QSQueue):
    """QSQueue 工作进程: 写入大数据量 (模拟原脚本的大数据传输)."""
    queue.put((worker_id, "a" * 10000000))


# ========== 测试类 ==========

class TestFileLock(unittest.TestCase):
    """FileLock 文件锁基本功能测试."""

    def setUp(self):
        self.lock_path = os.path.join(os.path.dirname(__file__), "_test_lock_file.lock")

    def tearDown(self):
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)

    def test_lock_acquire_and_release(self):
        """基本的加锁与释放."""
        lock = FileLock(self.lock_path)
        with lock:
            self.assertTrue(lock.is_locked)
        self.assertFalse(lock.is_locked)

    def test_lock_is_reentrant_by_same_thread(self):
        """同一线程可重复获取同一锁 (FileLock 默认可重入)."""
        lock = FileLock(self.lock_path)
        with lock:
            with lock:
                self.assertTrue(lock.is_locked)

    def test_concurrent_threads_mutual_exclusion(self):
        """多线程并发获取锁时应互斥执行."""
        n_workers = 4
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_file_lock_worker, self.lock_path, i, [])
                for i in range(n_workers)
            ]
            for f in futures:
                f.result()

    def test_lock_timeout(self):
        """超时未能获取锁时应抛出 Timeout 错误."""
        from filelock import Timeout
        lock1 = FileLock(self.lock_path)
        lock2 = FileLock(self.lock_path)
        with lock1:
            with self.assertRaises(Timeout):
                lock2.acquire(timeout=0.01)

    def test_concurrent_processes_mutual_exclusion(self):
        """多进程 (multiprocess.Process) 并发获取锁时应互斥执行."""
        n_workers = 4
        procs = []
        for i in range(n_workers):
            p = Process(target=_file_lock_worker_process, args=(self.lock_path, i))
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, f"子进程退出码异常: {p.exitcode}")

    def test_concurrent_process_pool_mutual_exclusion(self):
        """进程池 (multiprocess.Pool) 并发获取锁时应互斥执行."""
        n_workers = 4
        with Pool(processes=n_workers) as pool:
            results = [
                pool.apply_async(_file_lock_worker_process, (self.lock_path, i))
                for i in range(n_workers)
            ]
            worker_ids = sorted([r.get(timeout=30) for r in results])

        self.assertEqual(worker_ids, list(range(n_workers)))

    def test_concurrent_process_pool_executor_mutual_exclusion(self):
        """ProcessPoolExecutor 并发获取锁时应互斥执行."""
        n_workers = 4
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_file_lock_worker_process, self.lock_path, i)
                for i in range(n_workers)
            ]
            worker_ids = sorted([f.result(timeout=30) for f in futures])

        self.assertEqual(worker_ids, list(range(n_workers)))


class TestQSQueue(unittest.TestCase):
    """QSQueue 共享内存队列功能测试."""

    def setUp(self):
        self.queue = QSQueue(cache_size=1, batch_size=64)

    def tearDown(self):
        self.queue.close()

    def test_put_and_get_single_object(self):
        """单个对象的 put/get 应正确往返."""
        obj = {"key": "value", "num": 42}
        self.queue.put(obj)
        result = self.queue.get()
        self.assertEqual(result, obj)

    def test_put_and_get_multiple_objects(self):
        """多个对象按 FIFO 顺序正确取出."""
        objects = [f"item_{i}" for i in range(5)]
        for obj in objects:
            self.queue.put(obj)
        results = [self.queue.get() for _ in range(5)]
        self.assertEqual(results, objects)

    def test_put_and_get_tuple(self):
        """元组类型数据的序列化与反序列化."""
        data = (1, "hello", [1, 2, 3])
        self.queue.put(data)
        result = self.queue.get()
        self.assertEqual(result, data)

    def test_put_and_get_large_data(self):
        """大数据量 (超过单个 batch) 的分片传输."""
        large_str = "x" * (self.queue._BatchDataSize * 3 + 100)
        self.queue.put(large_str)
        result = self.queue.get()
        self.assertEqual(result, large_str)

    def test_empty_queue_returns_true(self):
        """新建队列应为空."""
        self.assertTrue(self.queue.empty())

    def test_non_empty_queue_returns_false(self):
        """put 后队列不为空."""
        self.queue.put("data")
        self.assertFalse(self.queue.empty())

    def test_size_property(self):
        """size 属性应返回 cache_size (MB)."""
        self.assertEqual(self.queue.size, 1)

    def test_multiprocess_put_get(self):
        """多进程并发 put, 主进程逐个 get."""
        n_procs = 4
        data = "test_data"
        procs = []
        for i in range(n_procs):
            p = Process(target=_qs_queue_put_worker, args=(i, self.queue, data))
            p.start()
            procs.append(p)

        results = []
        for _ in range(n_procs):
            results.append(self.queue.get())

        for p in procs:
            p.join(timeout=10)
            self.assertEqual(p.exitcode, 0, f"子进程退出码异常: {p.exitcode}")

        worker_ids = sorted([r[0] for r in results])
        self.assertEqual(worker_ids, list(range(n_procs)))

    def test_multiprocess_put_large_data(self):
        """多进程并发写入大数据量, 验证数据完整性 (模拟原脚本的大数据传输场景)."""
        n_procs = 4
        procs = []
        for i in range(n_procs):
            p = Process(target=_qs_queue_put_large_worker, args=(i, self.queue))
            p.start()
            procs.append(p)

        results = []
        for _ in range(n_procs):
            results.append(self.queue.get())

        for p in procs:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, f"子进程退出码异常: {p.exitcode}")

        # 验证每个进程的数据完整
        results.sort(key=lambda x: x[0])
        for i, (worker_id, data) in enumerate(results):
            self.assertEqual(worker_id, i)
            self.assertEqual(len(data), 10000000)
            self.assertEqual(data, "a" * 10000000)

    def test_context_manager(self):
        """上下文管理器模式应正常关闭队列."""
        with QSQueue(cache_size=1, batch_size=64) as q:
            q.put("hello")
            self.assertEqual(q.get(), "hello")

    def test_pickle_roundtrip_state(self):
        """队列的 pickle 序列化/反序列化应保持一致性."""
        import pickle
        data = pickle.dumps(self.queue)
        restored = pickle.loads(data)
        self.assertEqual(restored.size, self.queue.size)
        self.assertEqual(restored._MaxBatchNum, self.queue._MaxBatchNum)


if __name__ == "__main__":
    unittest.main(verbosity=2)
